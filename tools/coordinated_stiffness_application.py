"""Same Evidence/inputs as the analytic application, isolated stiffness optimizer."""
from __future__ import annotations
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from tools.coordinated_real_case import Evidence,load_registration_evidence,load_image_matches
from tools.digital_q1_dhr_distill import identity_vertices
from qcopt.neural_bijection.dense.coordinated_stiffness import optimize_stiffness_fiber
from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def optimize_stiffness(args):
    start_all=time.perf_counter()
    if args.output.exists() or args.output.with_suffix('.json').exists(): raise FileExistsError(args.output)
    if (args.method!='analytic' or args.cycles!=1 or args.precision!='float64'
            or args.image_precision!='float32' or args.interpolation!='p1_ac'
            or args.strain_model!='p1_arap' or args.strain_weight!=3.
            or args.loss!='mind' or args.mind_order!='transport' or args.mind_frame!='shared_affine'
            or args.control_hierarchy!='fixed' or args.coordinate_mode!='alternating'
            or args.output_selection!='best_full' or args.preprocessing!='raw_inverted'
            or getattr(args,'image_objective','continuation')!='continuation'
            or getattr(args,'fixed_mask',None) is not None
            or getattr(args,'proposal_filter_steps',0) or getattr(args,'fine_patch_cells',0)
            or getattr(args,'capture_prefix','none')!='none'
            or len(args.levels)!=len(args.image_levels) or args.image_levels[-1]!=args.image_side
            or args.levels[-1]!=args.grid_side
            or any((args.grid_side-1)%(level-1) for level in args.levels)):
        raise ValueError('declared fixed shared-affine ARAP3 analytic recipe required')
    inner=getattr(args,'inner_steps_by_level',None) or [args.inner_steps]*len(args.levels)
    if len(inner)!=len(args.levels) or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in inner):
        raise ValueError('one positive integer gradient cap per coefficient level required')
    device=torch.device(args.device); torch.set_num_threads(args.threads)
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    with np.load(args.affine) as archive:
        matrix=np.asarray(archive['post_affine_matrix'],dtype=np.float32)
        offset=np.asarray(archive['post_affine_offset'],dtype=np.float32)
    if (matrix.shape!=(2,2) or offset.shape!=(2,) or not np.isfinite(matrix).all()
            or not np.isfinite(offset).all() or np.linalg.det(matrix.astype(np.float64))<=0):
        raise ValueError('finite positive original affine required')
    fixed,moving,mask,preprocessing=load_registration_evidence(args.fixed,args.moving,args.image_side,
        preprocessing=args.preprocessing,device=device,dtype=torch.float32)
    reference=identity_vertices(args.grid_side,device=device).double()
    current=reference.clone(); sync(); loading_seconds=time.perf_counter()-start_all
    start_features=time.perf_counter(); matches,match_metadata=None,None
    if args.match_weight:
        matches,match_metadata=load_image_matches(args.matches,matrix,offset,fixed_path=args.fixed,
            moving_path=args.moving,image_side=args.image_side,device=device,dtype=torch.float64,
            robust_scale=args.match_robust_scale)
        if getattr(args,'match_p1_sampling','existing')=='frozen':
            match_metadata['fixed_p1_sampling']=matches.prepare_fixed_p1_sampling(args.grid_side,args.grid_side,'ac')
    kwargs=dict(loss='mind',strain_weight=3.,oob_weight=args.oob_weight,shape_weight=args.shape_weight,
        interpolation='p1_ac',matches=matches,match_weight=args.match_weight,strain_model='p1_arap',
        mind_order='transport',image_weight=getattr(args,'image_weight',1.),mind_frame='shared_affine')
    full=Evidence(fixed,moving,torch.as_tensor(matrix,device=device,dtype=torch.float64),
        torch.as_tensor(offset,device=device,dtype=torch.float64),fixed_mask=mask,
        joint_prior_backend=getattr(args,'joint_prior_backend','eager'),**kwargs)
    evidences={args.image_side:full}
    for side in sorted(set(args.image_levels)):
        if side==args.image_side: continue
        reduce=lambda image:F.interpolate(image,size=(side,side),mode='area')
        item=Evidence(reduce(fixed),reduce(moving),full.matrix,full.offset,fixed_mask=reduce(mask),**kwargs)
        item.joint_prior_backend=full.joint_prior_backend; item.joint_p1_priors=full.joint_p1_priors
        evidences[side]=item
    if getattr(args,'p1_sampling','existing')=='frozen':
        for item in evidences.values():
            item.prepare_fixed_p1_sampling(args.grid_side,args.grid_side,dtype=torch.float64,device=device)
    sync(); feature_seconds=time.perf_counter()-start_features
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    with torch.no_grad():
        initial,parts=full(current)
        initial_record=dict(total=float(initial),**{key:float(value) for key,value in parts.items()})
    best_map,best_loss,best_stage=current.clone(),float(initial),None
    counts=dict(gradient_steps=0,accepted_steps=0,objective_evaluations=1,
                trial_evaluations=0,backtracks=0,failed_trials=0)
    stages=[]; trace=[]
    for level,side,budget in zip(args.levels,args.image_levels,inner,strict=True):
        for direction in ((1.,0.),(0.,1.)):
            anchor=current.detach()
            result=optimize_stiffness_fiber(anchor,evidences[side],coefficient_side=level,direction=direction,
                strain_weight=3.,minimum_jacobian=args.minimum_jacobian,maximum_gradients=budget)
            current=result.vertices
            if not validate_q1_map(current,reference)['valid']:
                raise RuntimeError('accepted stage map failed actual independent geometry check')
            with torch.no_grad(): full_loss=float(full(current)[0])
            counts['objective_evaluations']+=1
            for key,value in result.counts.items(): counts[key]+=value
            stage=dict(level=level,image_side=side,direction=list(direction),
                anchor_total=result.initial_objective,accepted_total=result.final_objective,
                accepted_full_total=full_loss,inner_steps=budget,actual_counts=result.counts,
                stop_reason=result.stop_reason,minimum_contracted_slack=result.minimum_contracted_slack)
            stages.append(stage)
            trace.extend(dict(level=level,image_side=side,direction=list(direction),**row) for row in result.trace)
            if full_loss<best_loss:
                best_map,best_loss,best_stage=current.clone(),full_loss,len(stages)-1
    current=best_map; sync(); optimize_seconds=time.perf_counter()-start
    with torch.no_grad(): final,parts=full(current)
    counts['objective_evaluations']+=1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter()
    np.savez(args.output,vertices=current.cpu().numpy(),boundary_reference=reference.cpu().numpy(),
        post_affine_matrix=matrix,post_affine_offset=offset,interpolation=np.asarray('p1_ac'))
    serialization_seconds=time.perf_counter()-tick; tick=time.perf_counter()
    certificate=certify_q1_binary_map(args.output); certification_seconds=time.perf_counter()-tick
    preprocessing.update(original_moving_features_no_affine_prewarp=False,
        original_moving_raster_no_affine_prewarp=True,moving_descriptor_frame='shared_affine',
        moving_descriptor_prewarp_scope='once per raster scale')
    report=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        optimizer='exact_Galerkin_ARAP_stiffness_physical_fiber',
        representation='p1_ac residual then frozen positive affine',landmarks_used=False,
        initial=initial_record,final=dict(total=float(final),**{k:float(v) for k,v in parts.items()}),
        **counts,evaluations=counts['trial_evaluations'],stages=stages,trace=trace,
        stage_stop_reasons=[s['stop_reason'] for s in stages],selected_stage=best_stage,
        output_selection='best_full',image_objective='continuation',image_levels=args.image_levels,
        terminal_full_total=stages[-1]['accepted_full_total'],best_accepted_full_total=best_loss,
        maximum_gradient_steps=2*sum(inner),saved_binary_certificate=certificate,
        loading_seconds=loading_seconds,feature_seconds=feature_seconds,optimize_seconds=optimize_seconds,
        serialization_seconds=serialization_seconds,certification_seconds=certification_seconds,
        end_to_end_seconds=time.perf_counter()-start_all,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        mind_frame='shared_affine',mind_frame_by_resolution={str(s):e.mind_frame_metadata for s,e in evidences.items()},
        image_preprocessing=preprocessing,image_match_evidence=match_metadata,
        global_operator='one exact separable stiffness spectral inverse per gradient; a global structured solve, no sparse factorization/Krylov',
        contracted_fiber='fixed original stage floor eta+.05*(Q0-eta); strict interior; original anchor never refreshed within stage',
        step_rule='a0=min(1,.99*amax); at most12 halvings; full objective Armijo1e-4 and actual rounded contracted checks',
        failed_trial_scope='nonfinite/geometry-invalid trials; ordinary Armijo rejections are backtracks, not failures',
        stop_caution='line_search_exhausted and zero_computed_gradient are numerical statuses, not constrained-stationarity certificates')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
