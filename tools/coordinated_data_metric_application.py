"""Common ordinary prefix and two isolated timed final-stage optimizer packages.

No annotations, matching or image preprocessing changes. Prefix extraction is
charged once per pair; each suffix's complete call separately charges loading,
fixed Evidence construction, two synchronized axis clocks and export.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch

from tools import coordinated_real_case as original
from tools.digital_q1_dhr_distill import identity_vertices
from qcopt.neural_bijection.dense.coordinated_data_metric import optimize_data_metric_fiber,optimize_timed_adam_fiber
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def _plain(args):return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


def _validate_configuration(args):
    from tools.coordinated_matchanything_batch import configuration
    values=_plain(args)
    if values.get('match_weight')!=.2:raise ValueError('frozen independently normalized fused points required')
    values['match_weight']=.1
    configuration(values,values['matches'],values['output'])
    required=dict(learning_rate=.004,lr_calibration='edge',p1_sampling='frozen',geometry_backend='stage_cache')
    if any(values.get(k)!=v for k,v in required.items()):
        raise ValueError('unchanged archived learning rate and sampling/geometry backends required')
    if values.get('pose_mode') not in (None,'frozen','frozen_affine'):
        raise ValueError('no joint pose in timed final comparison')


def extract_prefix(args):
    """Save best-prefix at args.output and raw last accepted prefix at *_suffix_start.npz."""
    _validate_configuration(args)
    started=time.perf_counter();args=copy.deepcopy(args);args.output=Path(args.output)
    prefix_start=args.output.with_name(args.output.stem+'_suffix_start.npz')
    if prefix_start.exists():raise FileExistsError(prefix_start)
    stage_count=2*(len(args.levels)-1);last=[]
    def capture(vertices,stage,elapsed):
        last[:]=[vertices.detach().clone(),copy.deepcopy(stage),elapsed]
    report=original.optimize(args,accepted_stage_callback=capture,maximum_stages=stage_count)
    inner=getattr(args,'inner_steps_by_level',None) or [args.inner_steps]*len(args.levels)
    expected=2*sum(inner[:-1])
    if (len(report['stages'])!=stage_count or report['gradient_steps']!=expected
            or report['failed_trials']!=0 or not last):
        raise RuntimeError('common prefix did not complete its exact original gradient budget')
    tick=time.perf_counter()
    with np.load(args.output,allow_pickle=False) as saved:
        archive={key:saved[key].copy() for key in saved.files}
    archive['vertices']=last[0].cpu().numpy()
    np.savez(prefix_start,**archive)
    certificate=certify_q1_binary_map(prefix_start)
    if not certificate['valid']:raise RuntimeError('raw common prefix export failed actual certificate')
    extra_export=time.perf_counter()-tick
    result=dict(prefix_best=str(args.output.resolve()),prefix_start=str(prefix_start.resolve()),
        prefix_report=str(args.output.with_suffix('.json').resolve()),gradient_steps=report['gradient_steps'],
        prefix_gradient_steps=report['gradient_steps'],failed_trials=report['failed_trials'],
        prefix_best_total=report['final']['total'],suffix_start_total=last[1]['accepted_full_total'],
        accepted_stages=stage_count,raw_start_saved_binary_certificate=certificate,
        raw_start_export_and_certification_seconds=extra_export,prefix_seconds=time.perf_counter()-started,
        distinction='best prefix includes identity and all accepted stages; suffix starts at raw last accepted stage')
    report['prefix_extraction']=result
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def _load_prefix_map(path,args,matrix,offset,device):
    path=Path(path)
    certificate=certify_q1_binary_map(path)
    if not certificate['valid']:raise ValueError('invalid prefix binary certificate')
    with np.load(path,allow_pickle=False) as saved:
        vertices=saved['vertices'].copy();reference=saved['boundary_reference'].copy()
        if (vertices.dtype!=np.float64 or vertices.shape!=(1,args.grid_side,args.grid_side,2)
                or str(saved['interpolation'].item())!='p1_ac'
                or not np.array_equal(saved['post_affine_matrix'],matrix)
                or not np.array_equal(saved['post_affine_offset'],offset)):
            raise ValueError('prefix grid/dtype/interpolation/affine differs')
    actual_reference=identity_vertices(args.grid_side,device='cpu').double().numpy()
    if not np.array_equal(reference,actual_reference):raise ValueError('prefix unit reference differs')
    result=torch.as_tensor(vertices,device=device,dtype=torch.float64)
    if not bool((q1_corner_determinants(result)*(args.grid_side-1)**2>args.minimum_jacobian).all()):
        raise ValueError('prefix strict configured corner floor failed')
    return result


def _build_evidence(args,matrix,offset,device):
    fixed,moving,mask,preprocessing=original.load_registration_evidence(args.fixed,args.moving,args.image_side,
        preprocessing=getattr(args,'preprocessing','raw_inverted'),device=device,dtype=torch.float32,
        fixed_mask_path=getattr(args,'fixed_mask',None))
    matches,match_metadata=None,None
    match_weight=getattr(args,'match_weight',0.)
    if match_weight:
        matches,match_metadata=original.load_image_matches(args.matches,matrix,offset,fixed_path=args.fixed,
            moving_path=args.moving,image_side=args.image_side,device=device,dtype=torch.float64,
            robust_scale=getattr(args,'match_robust_scale',8.))
        if getattr(args,'match_p1_sampling','existing')=='frozen':
            match_metadata['fixed_p1_sampling']=matches.prepare_fixed_p1_sampling(args.grid_side,args.grid_side,'ac')
    evidence=original.Evidence(fixed,moving,torch.as_tensor(matrix,device=device,dtype=torch.float64),
        torch.as_tensor(offset,device=device,dtype=torch.float64),loss=args.loss,strain_weight=args.strain_weight,
        oob_weight=args.oob_weight,shape_weight=getattr(args,'shape_weight',0.),fixed_mask=mask,
        interpolation=getattr(args,'interpolation','q1'),matches=matches,match_weight=match_weight,
        strain_model=getattr(args,'strain_model','displacement_gradient'),mind_order=getattr(args,'mind_order','transport'),
        joint_prior_backend=getattr(args,'joint_prior_backend','eager'),image_weight=getattr(args,'image_weight',1.),
        mind_frame=getattr(args,'mind_frame','original'))
    if getattr(args,'p1_sampling','existing')=='frozen':
        evidence.prepare_fixed_p1_sampling(args.grid_side,args.grid_side,dtype=torch.float64,device=device)
    preprocessing.update(original_moving_features_no_affine_prewarp=False,
        original_moving_raster_no_affine_prewarp=True,moving_descriptor_frame='shared_affine',
        moving_descriptor_prewarp_scope='once at original512 scale')
    return evidence,preprocessing,match_metadata


def optimize_timed_final(args,*,prefix_best,prefix_start,arm,seconds_per_axis=2.):
    """Write ordinary certified NPZ/report; gradient_steps includes shared prefix work."""
    _validate_configuration(args)
    if arm not in ('adam','data_metric') or seconds_per_axis!=2.:
        raise ValueError('one declared timed arm and exactly2seconds per axis required')
    args=copy.deepcopy(args);args.output=Path(args.output)
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError(args.output)
    started=time.perf_counter();device=torch.device(args.device);torch.set_num_threads(args.threads)
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    prefix_best,prefix_start=Path(prefix_best),Path(prefix_start)
    prefix_report_path=prefix_best.with_suffix('.json')
    prefix_report=json.loads(prefix_report_path.read_text(encoding='utf-8'))
    prefix_info=prefix_report.get('prefix_extraction',{})
    inner=getattr(args,'inner_steps_by_level',None) or [args.inner_steps]*len(args.levels)
    expected=2*sum(inner[:-1])
    if (prefix_report.get('gradient_steps')!=expected or prefix_report.get('failed_trials')!=0
            or prefix_report.get('landmarks_used') is not False
            or len(prefix_report.get('stages',[]))!=2*(len(args.levels)-1)
            or Path(prefix_info.get('prefix_start','')).name!=prefix_start.name):
        raise ValueError('same complete image-only prefix and its raw suffix-start map required')
    original_cfg=prefix_report['configuration']
    for key,value in _plain(args).items():
        if key!='output' and key in original_cfg and value!=original_cfg[key]:
            raise ValueError(f'suffix changed prefix configuration: {key}')
    with np.load(args.affine,allow_pickle=False) as archive:
        matrix=np.asarray(archive['post_affine_matrix'],dtype=np.float32)
        offset=np.asarray(archive['post_affine_offset'],dtype=np.float32)
    current=_load_prefix_map(prefix_start,args,matrix,offset,device)
    best_map=_load_prefix_map(prefix_best,args,matrix,offset,device)
    sync();loading_seconds=time.perf_counter()-started
    tick=time.perf_counter();evidence,preprocessing,match_metadata=_build_evidence(args,matrix,offset,device)
    sync();feature_seconds=time.perf_counter()-tick
    with torch.no_grad():
        initial,parts=evidence(current)
        initial_record=dict(total=float(initial),**{k:float(v) for k,v in parts.items()})
        best_loss=float(evidence(best_map)[0])
    prefix_best_total=best_loss
    suffix_objective_evaluations=2;stages=[];trace=[];selected=None
    final_lr=args.learning_rate*(args.levels[0]-1)/(args.levels[-1]-1) if args.lr_calibration=='edge' else args.learning_rate
    start_optimization=time.perf_counter()
    for direction in ((1.,0.),(0.,1.)):
        if arm=='data_metric':
            result=optimize_data_metric_fiber(current,evidence,direction=direction,seconds=seconds_per_axis,
                minimum_jacobian=args.minimum_jacobian)
        else:
            result=optimize_timed_adam_fiber(current,evidence,direction=direction,seconds=seconds_per_axis,
                minimum_jacobian=args.minimum_jacobian,learning_rate=final_lr)
        current=result.vertices
        suffix_objective_evaluations+=result.counts['objective_evaluations']
        stage=dict(level=args.grid_side,image_side=args.image_side,direction=list(direction),
            anchor_total=result.initial_objective,accepted_total=result.final_objective,
            accepted_full_total=result.final_objective,counts=result.counts,stop_reason=result.stop_reason,
            elapsed_seconds=result.elapsed_seconds,time_overrun_seconds=result.time_overrun_seconds,
            budget_elapsed=result.elapsed_seconds>=seconds_per_axis,
            minimum_contracted_slack=result.minimum_contracted_slack,
            geometry_acceptance=('strict original Q>.001; contracted slack diagnostic only' if arm=='adam'
                else 'strict fixed contracted Q>Qfloor and actual original Q>.001'),
            physical_lr=final_lr if arm=='adam' else None)
        stages.append(stage)
        trace.extend(dict(stage=len(stages)-1,direction=list(direction),**row) for row in result.trace)
        if result.final_objective<best_loss:
            best_loss=result.final_objective;best_map=current.detach().clone();selected=len(stages)-1
    sync();optimization_seconds=time.perf_counter()-start_optimization
    with torch.no_grad():
        final,parts=evidence(best_map)
        suffix_objective_evaluations+=1
    args.output.parent.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    reference=identity_vertices(args.grid_side,device=device).double()
    np.savez(args.output,vertices=best_map.cpu().numpy(),boundary_reference=reference.cpu().numpy(),
        post_affine_matrix=matrix,post_affine_offset=offset,interpolation=np.asarray('p1_ac'))
    serialization_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    certificate=certify_q1_binary_map(args.output);certification_seconds=time.perf_counter()-tick
    if not certificate['valid']:raise RuntimeError('selected final saved binary certificate failed')
    sums={key:sum(s['counts'][key] for s in stages) for key in stages[0]['counts']}
    suffix_gradients=sums['gradient_steps'];prefix_gradients=prefix_report['gradient_steps']
    report=dict(configuration=_plain(args),optimizer='timed_final_'+arm,
        representation='p1_ac residual then frozen original positive affine',landmarks_used=False,
        initial=initial_record,final=dict(total=float(final),**{k:float(v) for k,v in parts.items()}),
        prefix_best_total=prefix_best_total,suffix_start_total=initial_record['total'],
        prefix_best=str(prefix_best.resolve()),prefix_start=str(prefix_start.resolve()),prefix_report=str(prefix_report_path.resolve()),
        prefix_gradient_steps=prefix_gradients,suffix_gradient_steps=suffix_gradients,
        gradient_steps=prefix_gradients+suffix_gradients,
        gradient_count_scope='shared original prefix plus this timed suffix; no fixed300 assertion',
        failed_trials=prefix_report['failed_trials']+sums['failed_trials'],suffix_counts=sums,
        objective_evaluations=prefix_report['objective_evaluations']+suffix_objective_evaluations,
        prefix_objective_evaluations=prefix_report['objective_evaluations'],suffix_objective_evaluations=suffix_objective_evaluations,
        evaluations=prefix_report['evaluations']+sums['trial_evaluations'],
        query_count=args.image_side**2,control_vertices=args.grid_side**2,stages=stages,trace=trace,
        terminal_budget=dict(protocol='timed_final',arm=arm,seconds_per_axis=seconds_per_axis,stage_records=stages,
            scope='all per-axis setup/gradient/linearization/solve/trials/synchronization; finish started bounded iteration'),
        output_selection='best_full',selected_stage=selected,
        selected_stage_scope='None=common best prefix including identity;0/1=accepted timed x/y suffix',
        terminal_full_total=stages[-1]['accepted_full_total'],best_accepted_full_total=float(final),
        loading_seconds=loading_seconds,feature_seconds=feature_seconds,optimize_seconds=optimization_seconds,
        serialization_seconds=serialization_seconds,certification_seconds=certification_seconds,
        end_to_end_seconds=time.perf_counter()-started,common_prefix_seconds=prefix_info.get('prefix_seconds'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        saved_binary_certificate=certificate,mind_frame='shared_affine',image_levels=list(args.image_levels),
        mind_frame_by_resolution={str(args.image_side):evidence.mind_frame_metadata},
        image_preprocessing=preprocessing,image_match_evidence=match_metadata,
        timing_scope='common prefix charged once per pair; per-suffix loading/features/export charged separately from both2s axis clocks',
        stop_caution='legal retained maps and elapsed time are not stationarity or completed gradient-budget claims')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
