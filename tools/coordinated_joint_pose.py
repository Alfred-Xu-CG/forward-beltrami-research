"""Bounded joint positive-affine pose / hard-feasible P1 residual experiment.

This isolated application optimizer does not modify the frozen-pose optimizer.
The physical pose is frozen during residual blocks; caches never contain a
trainable pose graph. No labels, rematching, adaptive support or fold repair.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from qcopt.neural_bijection.dense.coordinated_fixed_sampling import FrozenP1Evaluator
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_pixel_centers
from qcopt.neural_bijection.dense.coordinated_stage_cache import FrozenAnchorCoordinatedUpdate
from qcopt.neural_bijection.dense.coordinated_update import interpolate_proposal
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants, validate_q1_map
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers
from tools.coordinated_real_case import corner_symmetric_dirichlet, load_image_matches, load_registration_evidence
from tools.digital_mind_objective_probe import self_similarity
from tools.digital_q1_dhr_distill import identity_vertices


def pose_affine(physical):
    """Physical [tx,ty,theta,s,d,k] -> B,c, with G(z)=z@B.T+c."""
    if physical.shape != (6,) or physical.dtype != torch.float64:
        raise ValueError('six float64 physical pose parameters required')
    tx,ty,theta,s,d,k = physical.unbind()
    co,si = torch.cos(theta),torch.sin(theta)
    rotation = torch.stack((co,-si,si,co)).reshape(2,2)
    symmetric = torch.stack((s+d,k,k,s-d)).reshape(2,2)
    B = rotation@torch.matrix_exp(symmetric)
    center = physical.new_tensor([.5,.5])
    c = center+torch.stack((tx,ty))-B@center
    return B,c


def pose_from_chart(chart, lower):
    """Block-local [tx,ty,theta,r,d,k]; lower is a frozen physical bound."""
    return torch.stack((chart[0],chart[1],chart[2],chart[3].new_tensor(lower)+F.softplus(chart[3]),chart[4],chart[5]))


def _reference(vertices):
    rows,columns = vertices.shape[1:3]
    y,x = torch.meshgrid(torch.linspace(0,1,rows,dtype=torch.float64,device=vertices.device),
                         torch.linspace(0,1,columns,dtype=torch.float64,device=vertices.device),indexing='ij')
    return torch.stack((x,y),-1)[None]


def residual_minimum_ratio(vertices):
    return (q1_corner_determinants(vertices.double())/q1_corner_determinants(_reference(vertices))).amin()


def rebase_pose(physical, vertices, eta):
    """Return detached chart/bound without moving the current physical pose."""
    if physical.requires_grad or vertices.requires_grad:
        raise ValueError('rebase requires frozen physical pose and residual')
    m = float(residual_minimum_ratio(vertices))
    if not math.isfinite(m) or not m>eta:
        raise ValueError('residual must strictly exceed its original floor')
    lower = .5*math.log(eta/m)
    distance = physical[3]-lower
    if not bool(torch.isfinite(physical).all()) or not float(distance)>0:
        raise ValueError('current physical scale must strictly exceed pose bound')
    # Stable inverse softplus for positive distance; no clamping or pose jump.
    r = distance+torch.log(-torch.expm1(-distance))
    chart = physical.detach().clone();chart[3] = r
    if not bool(torch.isfinite(chart).all()):
        raise ValueError('nonfinite rebased pose chart')
    return chart,lower


def combined_affine(matrix, offset, physical):
    B,c = pose_affine(physical)
    return matrix@B,matrix@c+offset


class JointPoseEvidence:
    """Rebuild Phi(original-moving prewarped by A G) for a changing pose."""
    def __init__(self,fixed,moving,matrix,offset,mask,matches,*,match_weight=.2,
                 strain_weight=3.,shape_weight=1e-4,oob_weight=1.,grid_side=None):
        if (fixed.shape!=moving.shape or fixed.shape!=mask.shape or fixed.ndim!=4
                or fixed.shape[:2]!=(1,1) or matrix.shape!=(2,2) or offset.shape!=(2,)
                or matrix.dtype!=torch.float64 or offset.dtype!=torch.float64
                or any(x.requires_grad for x in (fixed,moving,matrix,offset,mask))):
            raise ValueError('frozen B1HW images/mask and float64 original affine required')
        if not bool(torch.isfinite(mask).all() and (mask>=0).all() and (mask<=1).all()) or float(mask.sum())<=0:
            raise ValueError('nonempty fixed mask required')
        if not all(bool(torch.isfinite(x).all()) for x in (fixed,moving,matrix,offset)) or not float(torch.linalg.det(matrix))>0:
            raise ValueError('finite images and positive original affine required')
        self.fixed,self.moving,self.mask = fixed,moving,mask
        self.matrix,self.offset = matrix.detach().clone(),offset.detach().clone()
        self.denominator = mask.sum()
        self.matches,self.match_weight = matches,float(match_weight)
        self.strain_weight,self.shape_weight,self.oob_weight = strain_weight,shape_weight,oob_weight
        self.fixed_feature = self_similarity(fixed)[0]
        self.raster_queries = fixed_pixel_centers(*fixed.shape[-2:],dtype=torch.float64,device=fixed.device)
        self.evaluator = None if grid_side is None else FrozenP1Evaluator(grid_side,grid_side,self.raster_queries,'ac')
        self.feature_rebuilds = 0

    def query(self,vertices):
        return (self.evaluator(vertices) if self.evaluator is not None
                else p1_map_at_pixel_centers(vertices,*self.fixed.shape[-2:],diagonal='ac'))

    def moving_feature(self,physical):
        Aout,bout = combined_affine(self.matrix,self.offset,physical)
        original = self.raster_queries@Aout.T+bout
        aligned = F.grid_sample(self.moving,(2*original-1).to(self.moving.dtype),
                                mode='bilinear',padding_mode='zeros',align_corners=False)
        self.feature_rebuilds += 1
        return self_similarity(aligned)[0]

    def point_term(self,vertices,physical):
        if self.matches is None:
            return vertices.new_zeros(())
        B,c = pose_affine(physical)
        # P1 interpolation commutes with an affine postcomposition. Targets and
        # normalized static weights remain in the ORIGINAL A-aligned frame.
        return self.matches(vertices@B.T+c,self.matrix,'p1_ac')

    def __call__(self,vertices,physical,cached_feature=None):
        if cached_feature is not None and (physical.requires_grad or cached_feature.requires_grad):
            raise ValueError('cached descriptor only for frozen physical pose')
        feature = self.moving_feature(physical) if cached_feature is None else cached_feature
        query = self.query(vertices).double()
        warped = F.grid_sample(feature,(2*query-1).to(feature.dtype),mode='bilinear',padding_mode='zeros',align_corners=False)
        image = ((self.fixed_feature-warped).abs().mean(1,keepdim=True)*self.mask).sum()/self.denominator
        Aout,bout = combined_affine(self.matrix,self.offset,physical)
        original = query@Aout.T+bout
        outside = (F.relu(-original)+F.relu(original-1)).square().sum(-1)[:,None]
        oob = (outside*self.mask).sum()/self.denominator
        outside_fraction = (((original<0)|(original>1)).any(-1)[:,None]*self.mask).sum()/self.denominator
        strain = p1_arap_energy(vertices,diagonal='ac',validate=False)
        shape = corner_symmetric_dirichlet(vertices)
        total = image+self.strain_weight*strain+self.oob_weight*oob+self.shape_weight*shape
        parts = dict(image=image,strain=strain,oob=oob,outside_fraction=outside_fraction,shape=shape)
        if self.match_weight:
            match = self.point_term(vertices,physical)
            total = total+self.match_weight*match;parts['match'] = match
        return total,parts


def pose_rms_scales(chart,lower,vertices,evidence):
    """Six block-frozen physical pixel RMS scales, including ds/dr."""
    chart = chart.detach()
    def coefficients(w):
        B,c = pose_affine(pose_from_chart(w,lower))
        return torch.cat((B.reshape(-1),c))
    derivative = torch.autograd.functional.jacobian(coefficients,chart,create_graph=False)
    query = evidence.query(vertices).detach()
    scales = []
    for j in range(6):
        velocity = (query@derivative[:4,j].reshape(2,2).T+derivative[4:,j])@evidence.matrix.T*512
        scales.append(((velocity.square().sum(-1)[:,None]*evidence.mask).sum()/evidence.denominator).sqrt())
    scales = torch.stack(scales).detach()
    if not bool(torch.isfinite(scales).all() and (scales>1e-12).all()):
        raise ValueError('degenerate physical RMS pose chart; no clipping/fallback')
    return scales


def _pair_geometry(vertices,physical,matrix,offset,eta):
    if not bool(torch.isfinite(vertices).all() and torch.isfinite(physical).all()):
        return dict(valid=False,reason='nonfinite residual or pose')
    Aout,bout = combined_affine(matrix,offset,physical)
    m = float(residual_minimum_ratio(vertices))
    determinant = float(torch.linalg.det(Aout));initial_det = float(torch.linalg.det(matrix))
    ratio = determinant/initial_det
    valid = (bool(torch.isfinite(Aout).all() and torch.isfinite(bout).all())
             and math.isfinite(m) and math.isfinite(ratio) and determinant>0
             and m>eta and ratio*m>eta)
    return dict(valid=valid,residual_minimum_corner_ratio=m,
                original_affine_normalized_minimum_corner_ratio=ratio*m,
                pose_determinant_ratio=ratio,combined_affine_determinant=determinant)


def _exact_det(matrix):
    x = [Fraction.from_float(float(v)) for v in np.asarray(matrix).reshape(-1)]
    return x[0]*x[3]-x[1]*x[2]


def validate_joint_export(path,minimum_jacobian=.001):
    """Actual saved residual certificate and stored combined-affine checks."""
    with np.load(path,allow_pickle=False) as saved:
        Y = np.asarray(saved['vertices']);reference = np.asarray(saved['boundary_reference'])
        Aout,bout = np.asarray(saved['post_affine_matrix']),np.asarray(saved['post_affine_offset'])
        A,b = np.asarray(saved['original_post_affine_matrix']),np.asarray(saved['original_post_affine_offset'])
        physical = np.asarray(saved['pose_parameters'])
        interpretation = str(saved['interpolation'])
    certificate = certify_q1_binary_map(path)
    finite = all(np.isfinite(x).all() for x in (Y,reference,Aout,bout,A,b,physical))
    shape_ok = Aout.shape==A.shape==(2,2) and bout.shape==b.shape==(2,) and physical.shape==(6,)
    if not finite or not shape_ok:
        return dict(valid=False,reason='nonfinite or malformed saved pose',saved_binary_certificate=certificate)
    det_out,det_original = _exact_det(Aout),_exact_det(A)
    q = q1_corner_determinants(torch.from_numpy(Y).double())
    unit_reference = _reference(torch.from_numpy(Y)).numpy()
    declared_reference = bool(Y.dtype==np.float64 and reference.dtype==np.float64
                              and np.array_equal(reference,unit_reference))
    qref = q1_corner_determinants(torch.from_numpy(unit_reference))
    m = float((q/qref).amin())
    ratio = float(det_out/det_original) if det_original>0 else float('nan')
    expected_A,expected_b = combined_affine(torch.tensor(A,dtype=torch.float64),torch.tensor(b,dtype=torch.float64),torch.tensor(physical,dtype=torch.float64))
    # Exact stored affine is authoritative. This tolerance checks metadata
    # consistency across CPU/GPU matrix_exp rounding, not topology.
    composition_consistent = bool(np.allclose(Aout,expected_A.numpy(),rtol=2e-12,atol=2e-12)
                                  and np.allclose(bout,expected_b.numpy(),rtol=2e-12,atol=2e-12))
    valid = bool(certificate['valid'] and declared_reference and interpretation=='p1_ac' and det_out>0 and det_original>0
                 and np.isfinite(m) and m>minimum_jacobian and ratio*m>minimum_jacobian and composition_consistent)
    return dict(valid=valid,residual_minimum_corner_ratio=m,
        original_affine_normalized_minimum_corner_ratio=ratio*m,
        combined_affine_positive_exact=bool(det_out>0),original_affine_positive_exact=bool(det_original>0),
        pose_determinant_ratio=ratio,composition_consistent=composition_consistent,
        declared_float64_unit_grid_reference=declared_reference,
        saved_binary_certificate=certificate,
        scope='exact binary residual signs plus exact stored affine determinant sign; numerical strict quantitative floors relative to original affine; affine postcomposition preserves P1 connectivity')


def _record(total,parts):
    return dict(total=float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()})


def optimize_joint_pose(args):
    """Same ordinary saved-map/report contract, explicit pose/residual counts."""
    started = time.perf_counter()
    args.output = Path(args.output)
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError(args.output)
    required = dict(method='analytic',loss='mind',interpolation='p1_ac',strain_model='p1_arap',
                    precision='float64',image_precision='float32',mind_frame='shared_affine',
                    cycles=1,lr_calibration='edge',output_selection='best_full')
    if any(getattr(args,k,v)!=v for k,v in required.items()):raise ValueError('joint pose requires declared analytic shared-affine P1ac continuation')
    for key,default in [('control_hierarchy','fixed'),('image_objective','continuation'),('coordinate_mode','alternating'),
                        ('seed_initializer','identity'),('capture_prefix','none'),('proposal_filter_steps',0),('fine_patch_cells',0),
                        ('mind_order','transport'),('image_weight',1.),('joint_prior_backend','eager'),
                        ('pose_mode','joint_positive_affine'),('match_p1_sampling','existing')]:
        if getattr(args,key,default)!=default:raise ValueError('unsupported joint-pose variant: '+key)
    if getattr(args,'fixed_mask',None) is not None or getattr(args,'preprocessing','raw_inverted')!='raw_inverted':
        raise ValueError('joint pose preserves original raw grayscale support')
    levels = list(args.levels);resolutions = list(args.image_levels)
    counts = list(getattr(args,'inner_steps_by_level',None) or [args.inner_steps]*len(levels))
    pose_steps = getattr(args,'pose_steps_per_level',10)
    if (not levels or len(levels)!=len(resolutions) or len(counts)!=len(levels) or min(levels)<3
            or max(levels)>args.grid_side or resolutions[-1]!=args.image_side or min(resolutions)<8
            or max(resolutions)>args.image_side or any(not isinstance(n,int) or n<1 for n in counts)
            or not isinstance(pose_steps,int) or pose_steps<1 or args.learning_rate<=0 or not 0<args.minimum_jacobian<1):
        raise ValueError('valid positive pose/residual budgets, grids and floor required')
    device = torch.device(args.device);torch.set_num_threads(args.threads)
    sync = lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    with np.load(args.affine,allow_pickle=False) as stored:
        a,b = np.asarray(stored['post_affine_matrix'],dtype=np.float32),np.asarray(stored['post_affine_offset'],dtype=np.float32)
    if a.shape!=(2,2) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.det(a.astype(np.float64))<=0:
        raise ValueError('finite positive original affine required')
    A,offset = torch.tensor(a,dtype=torch.float64,device=device),torch.tensor(b,dtype=torch.float64,device=device)
    fixed,moving,mask,preprocessing = load_registration_evidence(args.fixed,args.moving,args.image_side,
        preprocessing='raw_inverted',device=device,dtype=torch.float32)
    preprocessing={**preprocessing,
        'original_moving_features_no_affine_prewarp':False,
        'original_moving_raster_no_affine_prewarp':True,
        'moving_descriptor_frame':'shared_joint_positive_affine',
        'moving_descriptor_prewarp_scope':'original moving raster prewarped by A G before descriptors at each pose evaluation; physical-pose-frozen cache during residual blocks'}
    matches,match_metadata = None,None
    if args.match_weight:
        matches,match_metadata = load_image_matches(Path(args.matches),a,b,fixed_path=Path(args.fixed),moving_path=Path(args.moving),
            image_side=args.image_side,device=device,dtype=torch.float64,robust_scale=args.match_robust_scale)
    sync();loading_seconds = time.perf_counter()-started;feature_start = time.perf_counter()
    evidence = {}
    for side in sorted(set(resolutions+[args.image_side])):
        f,m,w = ((fixed,moving,mask) if side==args.image_side else
                 tuple(F.interpolate(x,size=(side,side),mode='area') for x in (fixed,moving,mask)))
        evidence[side] = JointPoseEvidence(f,m,A,offset,w,matches,match_weight=args.match_weight,
            strain_weight=args.strain_weight,shape_weight=args.shape_weight,oob_weight=args.oob_weight,grid_side=args.grid_side)
    full = evidence[args.image_side]
    reference = identity_vertices(args.grid_side,device=device).double()
    current,physical = reference.clone(),torch.zeros(6,dtype=torch.float64,device=device)
    sync();feature_seconds = time.perf_counter()-feature_start
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    stages,trace,failures,forward_times,backward_times = [],[],[],[],[]
    evaluations=0;pose_gradients=0;residual_gradients=0
    def evaluate(ev,Y,p,cached=None):
        nonlocal evaluations
        evaluations+=1
        return ev(Y,p,cached_feature=cached)
    initial = _record(*evaluate(full,current,physical))
    best_Y,best_pose,best_full,best_stage = current.clone(),physical.clone(),initial['total'],None
    optimize_start = time.perf_counter()

    def accept_pair(stage):
        nonlocal best_Y,best_pose,best_full,best_stage
        if not validate_q1_map(current,reference)['valid']:
            raise RuntimeError('accepted residual failed actual map check')
        geometry = _pair_geometry(current,physical,A,offset,args.minimum_jacobian)
        if not geometry['valid']:raise RuntimeError('accepted pair failed both actual floors')
        with torch.no_grad():value=float(evaluate(full,current,physical)[0])
        if not math.isfinite(value):raise RuntimeError('nonfinite accepted full objective')
        stage.update(accepted_full_total=value,pose_parameters=physical.cpu().tolist(),**geometry)
        stages.append(stage)
        if value<best_full:
            best_Y,best_pose,best_full,best_stage=current.clone(),physical.clone(),value,len(stages)-1

    for level_index,(level,side,nsteps) in enumerate(zip(levels,resolutions,counts)):
        ev = evidence[side]
        # Pose chart is rebased ONCE while the current residual is frozen.
        chart,lower = rebase_pose(physical,current,args.minimum_jacobian)
        scales = pose_rms_scales(chart,lower,current,ev)
        z = torch.nn.Parameter(torch.zeros(6,dtype=torch.float64,device=device))
        pose_lr = 512*args.learning_rate*16/(level-1)
        optimizer = torch.optim.Adam([z],lr=pose_lr)
        with torch.no_grad():anchor_total=float(evaluate(ev,current,physical)[0])
        best_value,block_pose=anchor_total,physical.clone()
        for step in range(pose_steps+1):
            optimizer.zero_grad(set_to_none=True);sync();tick=time.perf_counter()
            trial_pose = pose_from_chart(chart+z/scales,lower)
            geometry = _pair_geometry(current,trial_pose.detach(),A,offset,args.minimum_jacobian)
            if not geometry['valid']:
                failures.append(dict(level=level,direction='pose',step=step,reason='nonfinite pose or rounded floor',**geometry));break
            total,parts = evaluate(ev,current,trial_pose)
            sync();forward_times.append(time.perf_counter()-tick);value=float(total.detach())
            if not math.isfinite(value):
                failures.append(dict(level=level,direction='pose',step=step,reason='nonfinite objective'));break
            if value<best_value:best_value,block_pose=value,trial_pose.detach().clone()
            trace.append(dict(level=level,image_side=side,direction='pose',step=step,**_record(total,parts),**geometry))
            if step<pose_steps:
                tick=time.perf_counter();total.backward();sync();backward_times.append(time.perf_counter()-tick)
                if z.grad is None or not bool(torch.isfinite(z.grad).all()):
                    failures.append(dict(level=level,direction='pose',step=step,reason='missing/nonfinite gradient'));break
                pose_gradients+=1;optimizer.step()
        physical = block_pose.detach()
        accept_pair(dict(level=level,image_side=side,direction='pose',inner_steps=pose_steps,
                         physical_lr=pose_lr,rms_scales=scales.cpu().tolist(),scale_lower_bound=lower,
                         anchor_total=anchor_total,accepted_total=best_value))
        # Freeze actual physical B, not a chart whose lower bound could vary.
        fixed_pose = physical.detach().clone()
        detB = float(torch.linalg.det(pose_affine(fixed_pose)[0]))
        floor = max(args.minimum_jacobian,args.minimum_jacobian/detB)
        cached = ev.moving_feature(fixed_pose).detach()
        for direction in ((1.,0.),(0.,1.)):
            anchor = current.detach()
            coefficients = torch.nn.Parameter(torch.zeros(1,1,level-2,level-2,dtype=torch.float64,device=device))
            residual_lr = args.learning_rate*(levels[0]-1)/(level-1)
            optimizer = torch.optim.Adam([coefficients],lr=residual_lr)
            layer = FrozenAnchorCoordinatedUpdate(anchor,reference=reference,direction=direction,
                                                  mode='analytic',minimum_jacobian=floor,theta=.95)
            with torch.no_grad():anchor_total=float(evaluate(ev,anchor,fixed_pose,cached)[0])
            best_value,block_map=anchor_total,anchor
            for step in range(nsteps+1):
                optimizer.zero_grad(set_to_none=True);sync();tick=time.perf_counter()
                proposal = interpolate_proposal(F.pad(coefficients,(1,1,1,1))[:,0],(args.grid_side,args.grid_side))
                decoded = layer(proposal,validate=False);candidate = decoded.vertices
                geometry = _pair_geometry(candidate.detach(),fixed_pose,A,offset,args.minimum_jacobian)
                geometry.update(scale=float(decoded.scale.detach().min()),gauge=float(decoded.gauge.detach().max()),
                                margin=float(decoded.normalized_margin_min.detach().min()))
                if not geometry['valid'] or not geometry['margin']>0:
                    failures.append(dict(level=level,direction=direction,step=step,reason='rounded residual/full floor',**geometry));break
                total,parts = evaluate(ev,candidate,fixed_pose,cached)
                sync();forward_times.append(time.perf_counter()-tick);value=float(total.detach())
                if not math.isfinite(value):
                    failures.append(dict(level=level,direction=direction,step=step,reason='nonfinite objective'));break
                if value<best_value:best_value,block_map=value,candidate.detach().clone()
                trace.append(dict(level=level,image_side=side,direction=direction,step=step,**_record(total,parts),**geometry))
                if step<nsteps:
                    tick=time.perf_counter();total.backward();sync();backward_times.append(time.perf_counter()-tick)
                    if coefficients.grad is None or not bool(torch.isfinite(coefficients.grad).all()):
                        failures.append(dict(level=level,direction=direction,step=step,reason='missing/nonfinite gradient'));break
                    residual_gradients+=1;optimizer.step()
            current,physical=block_map.detach(),fixed_pose
            accept_pair(dict(level=level,image_side=side,direction=direction,inner_steps=nsteps,
                physical_lr=residual_lr,residual_floor=floor,anchor_total=anchor_total,accepted_total=best_value))

    terminal_total = stages[-1]['accepted_full_total']
    current,physical = best_Y,best_pose
    final = _record(*evaluate(full,current,physical))
    sync();optimize_seconds=time.perf_counter()-optimize_start
    Aout,bout = combined_affine(A,offset,physical)
    args.output.parent.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    np.savez(args.output,vertices=current.cpu().numpy(),boundary_reference=reference.cpu().numpy(),
        post_affine_matrix=Aout.detach().cpu().numpy(),post_affine_offset=bout.detach().cpu().numpy(),
        original_post_affine_matrix=a,original_post_affine_offset=b,
        pose_parameters=physical.cpu().numpy(),interpolation=np.asarray('p1_ac'))
    serialization_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    export = validate_joint_export(args.output,args.minimum_jacobian)
    if not export['valid']:raise RuntimeError('saved joint-pose pair failed export validation')
    certification_seconds=time.perf_counter()-tick
    from tools.coordinated_shared_affine_features import prepare_shared_affine_intensity
    tick=time.perf_counter()
    aligned,support=prepare_shared_affine_intensity(full.moving,Aout.detach(),bout.detach(),full.mask)
    with torch.no_grad():
        aligned_mean=(aligned*full.mask).sum()/full.denominator
        aligned_variance=((aligned-aligned_mean).square()*full.mask).sum()/full.denominator
        final_feature=full.moving_feature(physical)
        descriptor_mean=(final_feature*full.mask).sum((0,2,3))/(full.denominator)
        descriptor_variance=((final_feature-descriptor_mean[None,:,None,None]).square()*full.mask).sum()/(8*full.denominator)
    sync();final_diagnostics_seconds=time.perf_counter()-tick
    report = dict(configuration={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                  'pose_mode':'joint_positive_affine','pose_steps_per_level':pose_steps},
        pose_mode='joint_positive_affine',mind_frame='shared_affine',mind_order='transport',
        representation='exact source-P1 residual followed by stored positive combined affine A G',
        initial=initial,final=final,gradient_steps=pose_gradients+residual_gradients,
        pose_gradient_steps=pose_gradients,residual_gradient_steps=residual_gradients,
        planned_gradient_steps=len(levels)*pose_steps+2*sum(counts),failed_trials=len(failures),failures=failures,
        stages=stages,trace=trace,evaluations=evaluations,objective_evaluations=evaluations,
        optimize_seconds=optimize_seconds,loading_seconds=loading_seconds,feature_seconds=feature_seconds,
        serialization_seconds=serialization_seconds,certification_seconds=certification_seconds,
        end_to_end_seconds=time.perf_counter()-started,
        median_forward_objective_seconds=float(np.median(forward_times)),median_vjp_seconds=float(np.median(backward_times)) if backward_times else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        saved_binary_certificate=export['saved_binary_certificate'],joint_export_validation=export,
        landmarks_used=False,supplied_tissue_annotation=False,mask='original fixed inverted grayscale >.04, constant denominator',
        pose_parameters=physical.cpu().tolist(),pose_parameter_order=['tx','ty','theta','s','d','k'],
        pose_matrix=pose_affine(physical)[0].cpu().tolist(),pose_determinant=export['pose_determinant_ratio'],
        original_post_affine_matrix=a.tolist(),original_post_affine_offset=b.tolist(),
        post_affine_matrix=Aout.detach().cpu().tolist(),post_affine_offset=bout.detach().cpu().tolist(),
        feature_rebuilds_by_resolution={str(k):v.feature_rebuilds for k,v in evidence.items()},
        final_diagnostics_seconds=final_diagnostics_seconds,diagnostic_descriptor_rebuilds=1,
        final_nominal_descriptor_support=support,final_texture=dict(fixed_intensity_variance=float(full.fixed.var()),
            original_moving_intensity_variance=float(full.moving.var()),
            selected_pose_aligned_intensity_masked_variance=float(aligned_variance),
            selected_pose_aligned_descriptor_masked_channel_variance=float(descriptor_variance)),
        objective_description='I(Phi(I_fixed),Phi(original_moving prewarped by A G) sampled at fY)+3ARAP(Y)+1e-4Shape(Y)+.2 merged normalized points(A G fY versus A p)+OOB(A G fY); global affine unregularized',
        image_match_evidence=match_metadata,image_preprocessing=preprocessing,image_levels=resolutions,
        inner_steps_by_level=counts,output_selection='best_full',selected_stage=best_stage,
        selected_stage_scope='None=identity pair; otherwise accepted pose/residual block; selection stores BOTH physical pose and residual',
        terminal_full_total=terminal_total,best_accepted_full_total=best_full,
        acceptance_objective='complete current-resolution stage objective; selected paired state by full-resolution complete objective',
        geometry_dtype='torch.float64',evidence_dtype='torch.float32',strain_model='p1_arap',
        control_vertices=args.grid_side**2,cells=(args.grid_side-1)**2,corner_constraints=4*(args.grid_side-1)**2,
        query_count=args.image_side**2,geometry_backend='stage_cache',p1_sampling='frozen',
        pose_schedule='fresh Adam defaults on frozen RMS-normalized rebased chart;512*.004*16/(level-1) pixels; no label selection',
        budget_scope='pose feature recomputation charged;300 gradients not equal compute to frozen300')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
