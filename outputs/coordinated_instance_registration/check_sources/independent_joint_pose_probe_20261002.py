"""Independent literal NumPy pose/image/point oracles; no training or labels."""
import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from independent_stain_proxy_probe_20261002 import literal_p1
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from tools.coordinated_real_case import Evidence
from PIL import Image

torch.set_num_threads(1)


def numpy_pose(physical):
    tx,ty,theta,s,d,k=np.asarray(physical,dtype=float)
    values,vectors=np.linalg.eigh(np.array([[s+d,k],[k,s-d]]))
    stretch=(vectors*np.exp(values))@vectors.T
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    matrix=rotation@stretch
    offset=np.array([.5+tx,.5+ty])-matrix@np.array([.5,.5])
    return matrix,offset


def numpy_chart(chart,lower):
    result=np.array(chart,dtype=float).copy()
    result[3]=lower+np.logaddexp(0.,result[3])
    return result


def pixels(side):
    yy,xx=np.meshgrid((np.arange(side)+.5)/side,(np.arange(side)+.5)/side,indexing='ij')
    return np.stack((xx,yy),-1)


def corners(vertices):
    a,b,c,d=vertices[:-1,:-1],vertices[:-1,1:],vertices[1:,1:],vertices[1:,:-1]
    cross=lambda u,v:u[...,0]*v[...,1]-u[...,1]*v[...,0]
    return np.stack((cross(b-a,d-a),cross(b-a,c-a),cross(b-d,c-d),cross(c-a,d-a)))*(vertices.shape[0]-1)**2


def bilinear(image,queries):
    """Literal zero-padding, align_corners=False in unit coordinates."""
    values=np.asarray(image,dtype=float)
    if values.ndim==2:values=values[...,None]
    height,width,channels=values.shape
    xy=queries*np.array([width,height])-.5
    ij=np.floor(xy).astype(int);uv=xy-ij
    result=np.zeros((*queries.shape[:-1],channels))
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        x=ij[...,0]+dx;y=ij[...,1]+dy
        valid=(x>=0)&(x<width)&(y>=0)&(y<height)
        weight=(uv[...,0] if dx else 1-uv[...,0])*(uv[...,1] if dy else 1-uv[...,1])
        result+=values[np.clip(y,0,height-1),np.clip(x,0,width-1)]*(weight*valid)[...,None]
    return result


def descriptor(image):
    side=image.shape[0];pad=np.pad(image,2,mode='edge');channels=[]
    for dx,dy in ((2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2)):
        square=(image-pad[2+dy:2+dy+side,2+dx:2+dx+side])**2
        z=np.pad(square,1);ones=np.pad(np.ones_like(square),1)
        channels.append(sum(z[y:y+side,x:x+side] for y in range(3) for x in range(3))/sum(ones[y:y+side,x:x+side] for y in range(3) for x in range(3)))
    distance=np.stack(channels,-1)
    return np.exp(-(distance-distance.min(-1,keepdims=True))/(distance.mean(-1,keepdims=True)+1e-4))


def numpy_image(vertices,physical,a,b,fixed,moving,mask):
    matrix,offset=numpy_pose(physical);full_matrix=a@matrix;full_offset=a@offset+b
    grid=pixels(len(fixed));prepared=bilinear(moving,grid@full_matrix.T+full_offset)[...,0]
    mapped=literal_p1(vertices,grid.reshape(-1,2)).reshape(grid.shape)
    features=bilinear(descriptor(prepared),mapped)
    image=float((np.abs(descriptor(fixed)-features).mean(-1)*mask).sum()/mask.sum())
    world=mapped@full_matrix.T+full_offset
    excess=np.maximum(-world,0)+np.maximum(world-1,0)
    oob=float(((excess**2).sum(-1)*mask).sum()/mask.sum())
    return image,oob,prepared


def numpy_points(vertices,physical,a,b,q,p,confidence):
    matrix,offset=numpy_pose(physical);mapped=literal_p1(vertices,q)
    original_world=p@a.T+b;eligible=np.array([all(0<=x<=1 for x in xy) for xy in original_world])
    weight=confidence*eligible;weight/=weight.sum()
    error=((mapped@matrix.T+offset-p)@a.T)*64
    return float(((np.sqrt(1+(error**2).sum(-1))-1)*weight).sum())


def fixture(side=16):
    rng=np.random.default_rng(99173)
    axis=np.arange(9)/8;yy,xx=np.meshgrid(axis,axis,indexing='ij');vertices=np.stack((xx,yy),-1)
    vertices[1:-1,1:-1]+=rng.normal(0,.005,(7,7,2))
    fixed=rng.uniform(.05,.95,(side,side));moving=rng.uniform(.05,.95,(side,side))
    mask=np.zeros((side,side));mask[4,5]=.3;mask[7,9]=.8;mask[10,4]=1.;mask[12,12]=.6
    a=np.array([[.74,-.35],[.31,.91]]);b=np.array([.28,-.12])
    q=rng.uniform(.03,.97,(18,2));p=rng.uniform(.03,.97,(18,2));confidence=rng.uniform(.2,.9,18)
    q[:4]=[[0,0],[1,0],[0,1],[1,1]];p[:4]=q[:4];confidence[4]=0
    eligible=((p@a.T+b>=0)&(p@a.T+b<=1)).all(-1)
    matches=ImageCorrespondences(torch.tensor(q),torch.tensor(p),torch.tensor(confidence*eligible),pixel_scale=512.,robust_scale=8.)
    return dict(vertices=vertices,a=a,b=b,fixed=fixed,moving=moving,mask=mask,q=q,p=p,confidence=confidence,matches=matches)


def oracle_checks():
    item=fixture();physical=np.array([.037,-.026,.083,-.044,.072,-.039]);a=item['a'];b=item['b'];vertices=item['vertices']
    matrix,offset=numpy_pose(physical);full_matrix=a@matrix;full_offset=a@offset+b
    assert abs(np.linalg.det(matrix)-np.exp(2*physical[3]))<5e-16
    left=literal_p1(vertices@full_matrix.T+full_offset,item['q'])
    right=(literal_p1(vertices,item['q'])@matrix.T+offset)@a.T+b
    error=float(np.abs(left-right).max());assert error<4e-16
    scaled=corners(vertices@full_matrix.T+full_offset)/np.linalg.det(a)
    assert np.abs(scaled-np.linalg.det(matrix)*corners(vertices)).max()<3e-14
    # A point outside the old aligned square is still inside the original image.
    z=np.array([.8,.4]);pose=np.array([.4,0,0,0,0,0]);m,c=numpy_pose(pose)
    aligned=z@m.T+c;original=aligned@np.diag([.5,.7])+np.array([0,.1])
    assert aligned[0]>1 and ((original>0)&(original<1)).all()
    # Reconstruct the previously identified zero-prewarp support limitation.
    side=32;z=pixels(side);m=np.eye(2)*.032;c=np.array([1.016,.484]);source=np.ones((side,side))
    invalid=bilinear(source,z@m.T+c)[...,0]
    assert np.linalg.det(m)>.001 and (invalid==0).all() and (descriptor(invalid)==1).all()
    return dict(affine_P1_composition_maximum_error=error,corner_scaling=True,intermediate_outside_original_inside=True,positive_floor_does_not_certify_informative_support=True)


def implementation_checks():
    from tools.coordinated_joint_pose import JointPoseEvidence,pose_affine,pose_from_chart,rebase_pose,pose_rms_scales
    item=fixture();vertices=item['vertices'];a=item['a'];b=item['b'];mask=item['mask']
    tensor_image=lambda key:torch.tensor(item[key][None,None])
    y=torch.tensor(vertices[None],requires_grad=True)
    evidence=JointPoseEvidence(tensor_image('fixed'),tensor_image('moving'),torch.tensor(a),torch.tensor(b),
        tensor_image('mask'),item['matches'],match_weight=.2,strain_weight=3.,shape_weight=1e-4,oob_weight=1.,grid_side=9)
    baseline=Evidence(tensor_image('fixed'),tensor_image('moving'),torch.tensor(a),torch.tensor(b),'mind',3.,1.,1e-4,
        fixed_mask=tensor_image('mask'),interpolation='p1_ac',matches=item['matches'],match_weight=.2,strain_model='p1_arap',mind_frame='shared_affine')
    baseline.prepare_fixed_p1_sampling(9,9,dtype=torch.float64,device='cpu')
    original_buffers={key:value.clone() for key,value in item['matches'].named_buffers()}
    zero=torch.zeros(6,dtype=torch.float64)
    old_value,old_parts=baseline(y);new_value,new_parts=evidence(y,zero)
    identity_error=abs(float((new_value-old_value).detach()));assert identity_error<3e-13
    for key in ('image','strain','shape','match','oob'):assert abs(float((new_parts[key]-old_parts[key]).detach()))<3e-13
    old_gradient=torch.autograd.grad(old_value,y)[0];new_gradient=torch.autograd.grad(new_value,y)[0]
    identity_vjp=float((old_gradient-new_gradient).abs().max());assert identity_vjp<3e-12
    # Also check the production float32-evidence / float64-geometry convention.
    mixed_inputs=[tensor_image(key).float() for key in ('fixed','moving','mask')]
    mixed=JointPoseEvidence(mixed_inputs[0],mixed_inputs[1],torch.tensor(a),torch.tensor(b),mixed_inputs[2],item['matches'],grid_side=9)
    mixed_old=Evidence(mixed_inputs[0],mixed_inputs[1],torch.tensor(a),torch.tensor(b),'mind',3.,1.,1e-4,
        fixed_mask=mixed_inputs[2],interpolation='p1_ac',matches=item['matches'],match_weight=.2,strain_model='p1_arap',mind_frame='shared_affine')
    mixed_old.prepare_fixed_p1_sampling(9,9,dtype=torch.float64,device='cpu')
    mixed_new_value,_=mixed(y,zero);mixed_old_value,_=mixed_old(y)
    assert torch.equal(mixed_new_value,mixed_old_value)
    assert torch.equal(torch.autograd.grad(mixed_new_value,y)[0],torch.autograd.grad(mixed_old_value,y)[0])
    physical=np.array([.037,-.026,.083,-.044,.072,-.039]);pose=torch.tensor(physical,requires_grad=True)
    expected_matrix,expected_offset=numpy_pose(physical);matrix,offset=pose_affine(pose)
    assert np.abs(matrix.detach().numpy()-expected_matrix).max()<1e-14 and np.abs(offset.detach().numpy()-expected_offset).max()<1e-14
    value,parts=evidence(y,pose);expected_image,expected_oob,prepared=numpy_image(vertices,physical,a,b,item['fixed'],item['moving'],mask)
    expected_point=numpy_points(vertices,physical,a,b,item['q'],item['p'],item['confidence'])
    assert abs(float(parts['image'].detach())-expected_image)<3e-13 and abs(float(parts['oob'].detach())-expected_oob)<1e-14
    assert abs(float(parts['match'].detach())-expected_point)<3e-13
    feature=evidence.moving_feature(pose).detach().numpy()[0].transpose(1,2,0)
    assert np.abs(feature-descriptor(prepared)).max()<3e-13
    def oracle_dynamic(parameters,verts=vertices):
        image,oob,_=numpy_image(verts,parameters,a,b,item['fixed'],item['moving'],mask)
        return image+oob+.2*numpy_points(verts,parameters,a,b,item['q'],item['p'],item['confidence'])
    gradient=torch.autograd.grad(value,pose)[0].detach().numpy();finite=[];stability=[]
    for index in range(6):
        estimates=[]
        for epsilon in (1e-7,5e-8):
            step=np.eye(6)[index]*epsilon;estimates.append((oracle_dynamic(physical+step)-oracle_dynamic(physical-step))/(2*epsilon))
        finite.append(estimates[-1]);stability.append(abs(estimates[0]-estimates[1]))
    derivative_error=float(np.abs(gradient-finite).max())
    assert max(stability)<1e-5 and derivative_error<2e-5
    # Rebase a changed floor without changing the represented physical map.
    chart,lower=rebase_pose(pose.detach(),y.detach(),.001)
    recovered=pose_from_chart(chart,lower)
    assert float((recovered-pose.detach()).abs().max())<2e-14
    changed=vertices.copy();changed[4,4,0]+=.017
    chart2,lower2=rebase_pose(pose.detach(),torch.tensor(changed[None]),.001)
    assert abs(float(lower2)-float(lower))>1e-4
    assert float((pose_from_chart(chart2,lower2)-pose.detach()).abs().max())<2e-14
    assert abs(float(lower)-.5*np.log(.001/corners(vertices).min()))<1e-13
    scales=pose_rms_scales(chart,lower,y.detach(),evidence).detach().numpy()
    mapped=literal_p1(vertices,pixels(len(mask)).reshape(-1,2)).reshape((*mask.shape,2));wanted=[]
    for index in range(6):
        epsilon=1e-6;step=np.eye(6)[index]*epsilon
        plus=numpy_pose(numpy_chart(chart.detach().numpy()+step,float(lower)))
        minus=numpy_pose(numpy_chart(chart.detach().numpy()-step,float(lower)))
        velocity=((mapped@plus[0].T+plus[1])-(mapped@minus[0].T+minus[1]))@a.T*512/(2*epsilon)
        wanted.append(np.sqrt(((velocity**2).sum(-1)*mask).sum()/mask.sum()))
    scale_error=float(np.abs(scales-wanted).max());assert scale_error<2e-7 and np.all(scales>1e-12)
    # Caching a PHYSICAL frozen pose must leave the entire residual derivative intact.
    frozen=pose.detach().clone();cached=evidence.moving_feature(frozen).detach()
    y1=torch.tensor(vertices[None],requires_grad=True);v1,_=evidence(y1,frozen)
    y2=torch.tensor(vertices[None],requires_grad=True);v2,_=evidence(y2,frozen,cached_feature=cached)
    g1=torch.autograd.grad(v1,y1)[0];g2=torch.autograd.grad(v2,y2)[0]
    assert abs(float(v1.detach()-v2.detach()))<1e-14 and float((g1-g2).abs().max())<1e-13
    for key,before in original_buffers.items():assert torch.equal(before,dict(item['matches'].named_buffers())[key])
    # A large pose displacement must not recalculate original-target eligibility.
    shifted=physical.copy();shifted[:2]=[.61,-.22]
    _,shifted_parts=evidence(y,torch.tensor(shifted))
    assert abs(float(shifted_parts['match'].detach())-numpy_points(vertices,shifted,a,b,item['q'],item['p'],item['confidence']))<1e-12
    try:evidence(y,pose,cached_feature=cached)
    except ValueError:pass
    else:raise AssertionError('trainable pose accepted a stale feature cache')
    return dict(identity_total_error=identity_error,identity_vertex_VJP_error=identity_vjp,
        production_mixed_precision_identity_value_and_vertex_VJP_bitwise_equal=True,
        nonidentity_image_point_OOB_and_prepared_descriptor_match_literal_oracle=True,
        physical_pose_gradient_maximum_error=derivative_error,finite_difference_halving_stability=max(stability),
        rebased_chart_has_no_physical_jump=True,RMS_chart_gradient_maximum_error=scale_error,
        physical_pose_cached_and_uncached_residual_VJP_equal=True,original_static_correspondence_buffers_unchanged=True)


def export_and_outside_checks():
    from tools.coordinated_joint_pose import JointPoseEvidence,pose_affine,validate_joint_export
    item=fixture();vertices=item['vertices'];a=item['a'];b=item['b'];physical=np.array([.037,-.026,.083,-.044,.072,-.039])
    axis=np.arange(9)/8;yy,xx=np.meshgrid(axis,axis,indexing='ij');reference=np.stack((xx,yy),-1)[None]
    def record(parameters=physical):
        matrix,offset=numpy_pose(parameters)
        return dict(vertices=vertices[None].copy(),boundary_reference=reference.copy(),
            post_affine_matrix=a@matrix,post_affine_offset=a@offset+b,original_post_affine_matrix=a.copy(),
            original_post_affine_offset=b.copy(),pose_parameters=parameters.copy(),interpolation=np.asarray('p1_ac'))
    with tempfile.TemporaryDirectory(prefix='independent_pose_export_',dir=ROOT/'outputs/coordinated_instance_registration') as folder:
        path=Path(folder)/'map.npz';good=record();np.savez(path,**good);report=validate_joint_export(path)
        assert report['valid'] and report['combined_affine_positive_exact']
        m=float(corners(vertices).min());ratio=np.linalg.det(good['post_affine_matrix'])/np.linalg.det(a)
        assert abs(report['residual_minimum_corner_ratio']-m)<1e-14 and abs(report['original_affine_normalized_minimum_corner_ratio']-ratio*m)<1e-14
        low=physical.copy();low[3]=.5*np.log(.0008/m);under=record(low)
        assert np.linalg.det(under['post_affine_matrix'])>0
        shifted=record();shifted['vertices']+=1;shifted['boundary_reference']+=1
        rescaled=record();rescaled['vertices']*=2;rescaled['boundary_reference']*=2
        low_precision=record();low_precision['vertices']=low_precision['vertices'].astype(np.float32);low_precision['boundary_reference']=low_precision['boundary_reference'].astype(np.float32)
        changed_boundary=record();changed_boundary['vertices'][0,0,0,0]+=.01
        wrong_composition=record();wrong_composition['post_affine_offset'][0]+=.02
        wrong_sign=record();wrong_sign['post_affine_matrix'][:,0]*=-1
        variants={'positive_affine_below_full_floor':under,'shifted_reference':shifted,'rescaled_reference':rescaled,
            'float32_residual':low_precision,'changed_boundary':changed_boundary,'wrong_pose_composition':wrong_composition,'negative_combined_affine':wrong_sign}
        for name,data in variants.items():
            np.savez(path,**data);value=validate_joint_export(path)
            assert not value['valid'],name+' accepted by export validator'
    # The feature rebuild must use the original raster, not the old aligned canvas.
    a=np.diag([.5,.7]);b=np.array([0.,.1]);parameters=np.array([.4,0,0,0,0,0]);mask=np.ones_like(item['fixed'])
    images=lambda image:torch.tensor(image[None,None])
    evidence=JointPoseEvidence(images(item['fixed']),images(item['moving']),torch.tensor(a),torch.tensor(b),images(mask),None,match_weight=0.,grid_side=9)
    expected,_,prepared=numpy_image(vertices,parameters,a,b,item['fixed'],item['moving'],mask)
    _,parts=evidence(torch.tensor(vertices[None]),torch.tensor(parameters))
    assert abs(float(parts['image'])-expected)<3e-13
    z=pixels(len(mask));g=z+np.array([.4,0]);original=g@a.T+b
    relevant=(g[...,0]>1)&((original>0)&(original<1)).all(-1)
    assert relevant.any() and (prepared[relevant]>0).all()
    return dict(actual_saved_affine_and_residual_floors_match_literal=True,rejected_export_variants=list(variants),
        moving_feature_rebuilt_from_original_raster_when_intermediate_outside=True,
        intermediate_outside_original_inside_pixels=int(relevant.sum()))


def vector_p1(vertices,query):
    n=len(vertices)-1;cell=np.minimum(np.floor(query*n).astype(int),n-1);uv=query*n-cell
    x,y=cell[...,0],cell[...,1];u,v=uv[...,0,None],uv[...,1,None]
    return np.where(v<=u,(1-u)*vertices[y,x]+(u-v)*vertices[y,x+1]+v*vertices[y+1,x+1],
        (1-v)*vertices[y,x]+u*vertices[y+1,x+1]+(v-u)*vertices[y+1,x])


def numpy_priors(vertices):
    a,b,c,d=vertices[:-1,:-1],vertices[:-1,1:],vertices[1:,1:],vertices[1:,:-1];n=len(vertices)-1
    faces=np.stack((np.stack((b-a,c-b),-1),np.stack((c-d,d-a),-1)))*n
    singular=np.linalg.svd(faces,compute_uv=False)
    strain=float(.5*((singular-1)**2).sum(-1).mean())
    dx=np.stack((b-a,b-a,c-d,c-d))*n;dy=np.stack((d-a,c-b,c-b,d-a))*n
    determinants=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    frobenius=(dx**2).sum(-1)+(dy**2).sum(-1)
    shape=float((frobenius*(1+1/determinants**2)-4).mean())
    return strain,shape


def actual_postrun():
    from independent_stain_proxy_probe_20261002 import original_landmark_case
    from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
    base=ROOT/'outputs/coordinated_instance_registration';directory=base/'joint_pose_all50_t24'
    archived=base/'match_fusion_all50_t23/fusion';data=Path('D:/QC_optimization_data/digital_topology_wsi')
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
    outer=read(directory/'predictions.json');manifests={arm:read(directory/arm/'predictions.json') for arm in ('frozen250','joint300')}
    assert outer['prediction_complete'] and outer['attempt_denominator']==50 and not outer['annotations_read']
    for manifest in manifests.values():
        assert manifest['prediction_complete'] and manifest['all50_terminal'] and not manifest['annotations_read']
        assert len(manifest['rows'])==25 and all(r['status']=='ok' and r['budget_complete'] for r in manifest['rows'])
    # No evaluation coordinates are read before the preceding all50 checks.
    scores={arm:{r['name']:r for r in read(directory/arm/'miit_scores.json')['rows']+read(directory/arm/'existing_scores.json')['rows']} for arm in manifests}
    controls={r['name']:r for r in read(archived/'predictions.json')['rows']}
    axis=np.arange(257)/256;yy,xx=np.meshgrid(axis,axis,indexing='ij');identity=np.stack((xx,yy),-1)
    exactdet=lambda matrix:Fraction(float(matrix[0,0]))*Fraction(float(matrix[1,1]))-Fraction(float(matrix[0,1]))*Fraction(float(matrix[1,0]))
    image_cache={};rows=[];labels=0;score_error=0.;native_error=0.;outside_error=0.;composition_error=0.;texture_error=0.;exponential_det_error=0.;objective_error=0.;point_error=0.
    reports={arm:{} for arm in ('frozen300','frozen250','joint300')}
    def image_data(name,role,cfg):
        filename=Path(cfg[role]).name
        folder=base/'miit_three_rotations_t153' if name.startswith('miit_') else data/('birl_anhir_dev/canvas' if name in ('histo','rat_kidney') else 'lung_lesion3_eval/canvas')
        path=folder/filename
        if path not in image_cache:image_cache[path]=1-np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255
        return image_cache[path]
    for arm,manifest in manifests.items():
        for row in manifest['rows']:
            name=row['name'];report=read(directory/arm/row['report']);cfg=report['configuration'];old=read(archived/controls[name]['report']);oldcfg=old['configuration']
            reports[arm][name]=report;reports['frozen300'][name]=old
            extras={'pose_mode','pose_steps_per_level'} if arm=='joint300' else {'pose_mode'}
            assert set(cfg)-set(oldcfg)==extras and not set(oldcfg)-set(cfg)
            assert {k for k in oldcfg if cfg[k]!=oldcfg[k]}=={'inner_steps','inner_steps_by_level','output'}
            assert cfg['matches']==oldcfg['matches'] and cfg['match_weight']==.2
            assert report['image_match_evidence']==old['image_match_evidence']
            if arm=='frozen250':assert report['image_preprocessing']==old['image_preprocessing']
            else:
                # Historical raw-loader flag is stale; preserve the files and
                # report this erratum, not an invented objective equivalence.
                assert report['image_preprocessing']['name']=='raw_inverted'
                assert report['image_preprocessing']['mask_source']==old['image_preprocessing']['mask_source']
                assert report['image_preprocessing']['original_moving_features_no_affine_prewarp'] is True
            assert abs(report['initial']['total']-old['initial']['total'])<2e-13 and not report['landmarks_used']
            budget=300 if arm=='joint300' else 250;calls=347 if arm=='joint300' else 282
            assert report['gradient_steps']==budget and report['objective_evaluations']==calls and report['failed_trials']==0
            assert report['final']['total']==min([report['initial']['total']]+[s['accepted_full_total'] for s in report['stages']])
            saved_path=directory/arm/row['output']
            with np.load(saved_path,allow_pickle=False) as saved:
                vertices=saved['vertices'][0];reference=saved['boundary_reference'][0];aout=saved['post_affine_matrix'];bout=saved['post_affine_offset']
                assert str(saved['interpolation'])=='p1_ac'
                if arm=='joint300':a=saved['original_post_affine_matrix'];b=saved['original_post_affine_offset'];physical=saved['pose_parameters']
                else:a=aout;b=bout;physical=np.zeros(6)
            with np.load(archived/controls[name]['output'],allow_pickle=False) as original:
                assert np.array_equal(a,original['post_affine_matrix']) and np.array_equal(b,original['post_affine_offset'])
            assert vertices.dtype==reference.dtype==np.float64 and vertices.shape==(257,257,2) and np.array_equal(reference,identity)
            assert all(np.array_equal(vertices[index],identity[index]) for index in (0,-1))
            assert np.array_equal(vertices[:,0],identity[:,0]) and np.array_equal(vertices[:,-1],identity[:,-1])
            assert certify_q1_binary_map(saved_path)['valid'] and report['saved_binary_certificate']['valid']
            assert exactdet(aout)>0 and exactdet(a)>0
            minimum=float(corners(vertices).min());ratio=float(exactdet(aout)/exactdet(a));full_minimum=minimum*ratio
            assert minimum>.001 and full_minimum>.001
            assert abs(row['actual_minimum_corner_ratio']-full_minimum)<3e-13
            B,c=numpy_pose(physical);error=max(np.abs(a@B-aout).max(),np.abs(a@c+b-bout).max())
            # Stored affine is authoritative; torch's small-norm matrix_exp
            # approximant is not an exact symmetric exponential in float64.
            composition_error=max(composition_error,float(error));assert error<1e-9
            if arm=='joint300':
                assert report['residual_gradient_steps']==250 and report['pose_gradient_steps']==50 and len(report['stages'])==15
                assert len(report['trace'])==315 and report['feature_rebuilds_by_resolution']=={'32':13,'64':13,'128':13,'256':13,'512':31}
                assert report['diagnostic_descriptor_rebuilds']==1 and np.array_equal(physical,report['pose_parameters'])
                assert abs(report['pose_determinant']-ratio)<1e-14
                exponential_det_error=max(exponential_det_error,abs(np.exp(2*physical[3])-ratio));assert abs(np.exp(2*physical[3])-ratio)<1e-9
                selected=report['selected_stage'];assert selected is not None
                assert np.array_equal(physical,report['stages'][selected]['pose_parameters'])
                assert report['stages'][selected]['accepted_full_total']==report['final']['total']
                assert abs(report['stages'][selected]['residual_minimum_corner_ratio']-minimum)<3e-13
                previous_min=1.
                for index in range(5):
                    pose_stage,horizontal,vertical=report['stages'][3*index:3*index+3]
                    assert pose_stage['direction']=='pose' and pose_stage['physical_lr']==[2.048,1.024,.512,.256,.128][index]
                    assert abs(pose_stage['scale_lower_bound']-.5*np.log(.001/previous_min))<1e-13
                    assert all(np.isfinite(x) and x>1e-12 for x in pose_stage['rms_scales'])
                    for residual_stage in (horizontal,vertical):
                        assert residual_stage['pose_parameters']==pose_stage['pose_parameters'] and residual_stage['inner_steps']==25
                        assert abs(residual_stage['residual_floor']-max(.001,.001/pose_stage['pose_determinant_ratio']))<1e-15
                    previous_min=vertical['residual_minimum_corner_ratio']
            fixed=image_data(name,'fixed',cfg);mask=fixed>.04;grid=pixels(512);mapped=vector_p1(vertices,grid);world=mapped@aout.T+bout
            actual_outside=float((((world<0)|(world>1)).any(-1)*mask).sum()/mask.sum())
            outside_error=max(outside_error,abs(actual_outside-report['final']['outside_fraction']));assert abs(actual_outside-report['final']['outside_fraction'])<4e-8
            table=read(base/'match_fusion_all50_t23/fused_tables'/(name+'.json'))
            q=np.array(table['source_points_unit']);p=np.array(table['target_points_unit']);confidence=np.array(table['confidence'])
            target_world=p@a.T+b;eligible=((target_world>=0)&(target_world<=1)).all(-1);weight=confidence*eligible;weight/=weight.sum()
            prediction=vector_p1(vertices,q)@aout.T+bout;errors64=(prediction-target_world)*64
            point=float(((np.sqrt(1+(errors64**2).sum(-1))-1)*weight).sum());point_error=max(point_error,abs(point-report['final']['match']))
            assert abs(point-report['final']['match'])<2e-10
            moving=image_data(name,'moving',cfg)
            normalized=(2*(grid@aout.T+bout)-1).astype(np.float32);aligned=bilinear(moving,(normalized.astype(float)+1)/2)[...,0]
            feature=descriptor(aligned);normalized_query=(2*mapped-1).astype(np.float32)
            sampled=bilinear(feature,(normalized_query.astype(float)+1)/2)
            image=float((np.abs(descriptor(fixed)-sampled).mean(-1)*mask).sum()/mask.sum())
            excess=np.maximum(-world,0)+np.maximum(world-1,0);oob=float(((excess**2).sum(-1)*mask).sum()/mask.sum())
            strain,shape=numpy_priors(vertices);objective=image+3*strain+1e-4*shape+.2*point+oob
            objective_error=max(objective_error,abs(objective-report['final']['total']));assert abs(objective-report['final']['total'])<3e-7
            assert abs(strain-report['final']['strain'])<1e-12 and abs(shape-report['final']['shape'])<1e-10
            if arm=='joint300':
                support=report['final_nominal_descriptor_support'];flat=grid.reshape(-1,2);lo=np.floor(flat*512-.5)-3;hi=np.ceil(flat*512-.5)+3
                fixed_valid=(lo>=0).all(-1)&(hi<=511).all(-1)
                extremes=np.stack((lo,np.c_[hi[:,0],lo[:,1]],hi,np.c_[lo[:,0],hi[:,1]]),1)
                mapped_extremes=(((extremes+.5)/512)@aout.T+bout)*512-.5
                supported=fixed_valid&(np.floor(mapped_extremes)>=0).all((1,2))&(np.ceil(mapped_extremes)<=511).all((1,2))
                assert support['fixed_mask_denominator']==int(mask.sum())
                assert support['shared_affine_descriptor_support']['full_domain_fraction']==float(supported.mean())
                assert support['shared_affine_descriptor_support']['fixed_mask_weighted_fraction']==float((supported*mask.reshape(-1)).sum()/mask.sum())
                mean=float((aligned*mask).sum()/mask.sum());variance=float((((aligned-mean)**2)*mask).sum()/mask.sum())
                feature_mean=(feature*mask[...,None]).sum((0,1))/mask.sum();feature_variance=float((((feature-feature_mean)**2)*mask[...,None]).sum((0,1)).mean()/mask.sum())
                expected=report['final_texture'];texerr=max(abs(expected['selected_pose_aligned_intensity_masked_variance']-variance),abs(expected['selected_pose_aligned_descriptor_masked_channel_variance']-feature_variance))
                texture_error=max(texture_error,texerr);assert texerr<3e-6
            layout,points,ids=original_landmark_case(name);score=scores[arm][name];assert ids==score['available_pair_labels']
            unit=lambda xy,l:((xy+.5)*l['effective_original_to_canvas_scale_xy']+l['padding_xy'])/512
            queries=unit(points['fixed'],layout['fixed']);predicted=literal_p1(vertices@aout.T+bout,queries)
            errors={'canvas_pixels':np.linalg.norm((predicted-unit(points['moving'],layout['moving']))*512,axis=1),
                'native_moving_pixels':np.linalg.norm((predicted*512-layout['moving']['padding_xy'])/layout['moving']['effective_original_to_canvas_scale_xy']-.5-points['moving'],axis=1)}
            metrics=(score['methods']['analytic']['metrics'] if name.startswith('miit_') else score['metrics'])
            for key,values in errors.items():
                discrepancy=max(abs(value-metrics[key]['per_label'][label]) for label,value in zip(ids,values))
                assert discrepancy<1e-9 and abs(values.mean()-metrics[key]['mean'])<1e-10 and abs(np.percentile(values,90)-metrics[key]['p90'])<1e-10
                if key=='canvas_pixels':score_error=max(score_error,discrepancy)
                else:native_error=max(native_error,discrepancy)
            labels+=len(ids)
            rows.append(dict(name=name,arm=arm,mean=metrics['canvas_pixels']['mean'],p90=metrics['canvas_pixels']['p90'],maximum=metrics['canvas_pixels']['maximum'],
                residual_minimum=minimum,full_minimum=full_minimum,pose_determinant=ratio,final_outside_fraction=actual_outside,selected_stage=report['selected_stage'],
                complete_call_seconds=row['complete_call_seconds']))
    assert labels==4148
    comparison=read(directory/'comparison.json');assert len(comparison['rows'])==25
    metrics={arm:{r['name']:{k:r[k] for k in ('mean','p90','maximum')} for r in rows if r['arm']==arm} for arm in manifests}
    metrics['frozen300']={r['name']:(r['methods']['analytic']['metrics'] if r['name'].startswith('miit_') else r['metrics'])['canvas_pixels'] for r in read(archived/'miit_scores.json')['rows']+read(archived/'existing_scores.json')['rows']}
    pairs={'frozen250_minus_frozen300':('frozen250','frozen300'),'joint300_minus_frozen300':('joint300','frozen300'),'joint300_minus_frozen250':('joint300','frozen250')}
    for row in comparison['rows']:
        name=row['name']
        for metric in ('mean','p90','maximum'):
            for arm in metrics:assert abs(row['metrics'][metric]['values'][arm]-metrics[arm][name][metric])<1e-12
            for key,(first,second) in pairs.items():assert abs(row['metrics'][metric]['deltas'][key]-(metrics[first][name][metric]-metrics[second][name][metric]))<1e-12
        for arm in metrics:
            report=reports[arm][name];diagnostic=row['diagnostics'][arm]
            assert diagnostic['final_outside_fraction']==report['final']['outside_fraction'] and diagnostic['selected_stage']==report['selected_stage']
            support=report['final_nominal_descriptor_support'] if arm=='joint300' else report['mind_frame_by_resolution']['512']
            assert diagnostic['nominal_descriptor_support']==support
            if arm=='joint300':
                assert diagnostic['pose_parameters']==report['pose_parameters'] and diagnostic['pose_determinant']==report['pose_determinant']
                assert diagnostic['final_texture']==report['final_texture']
        for key,(first,second) in pairs.items():
            for metric in ('full_domain_fraction','fixed_mask_weighted_fraction'):
                expected=row['diagnostics'][first]['nominal_descriptor_support']['shared_affine_descriptor_support'][metric]-row['diagnostics'][second]['nominal_descriptor_support']['shared_affine_descriptor_support'][metric]
                assert abs(row['nominal_support_deltas'][key][metric]-expected)<1e-14
    groups={group:[r['name'] for r in comparison['rows'] if r['cohort']==group] for group in comparison['cohorts']}
    for group,names in groups.items():
        for arm in metrics:
            group_record=comparison['cohorts'][group]['methods'][arm]
            assert group_record['pair_denominator']==group_record['scored_pairs']==len(names) and group_record['failure_count']==0
            for key,metric in [('mean_pair_mean','mean'),('mean_pair_p90','p90'),('worst_pair_maximum','maximum')]:
                values=[metrics[arm][name][metric] for name in names];wanted=max(values) if metric=='maximum' else sum(values)/len(values)
                assert abs(group_record['all_directions_canvas_pixels'][key]-wanted)<1e-12
    for key,(first,second) in pairs.items():
        for label,metric in [('mean_pair_mean','mean'),('mean_pair_p90','p90')]:
            wanted=sum(sum(metrics[first][name][metric]-metrics[second][name][metric] for name in names)/len(names) for names in groups.values())/4
            assert abs(comparison['equal_specimen_deltas'][key][label]-wanted)<1e-12
    costs={}
    for arm in metrics:
        manifest=read(archived/'predictions.json') if arm=='frozen300' else manifests[arm]
        cost=comparison['costs'][arm]['measures']
        for measure in ('complete_call_seconds','optimize_seconds','peak_allocated_bytes'):
            values=[r[measure] for r in manifest['rows']];actual=cost[measure]
            assert actual['recorded_count']==actual['denominator']==25 and actual['missing_count']==0
            assert abs(actual['mean']-sum(values)/25)<1e-10 and actual['minimum']==min(values) and actual['maximum']==max(values)
            if measure!='peak_allocated_bytes':assert abs(actual['total']-sum(values))<1e-10
            else:assert actual['total'] is None
        costs[arm]=cost['complete_call_seconds']['total']
    joint=[r for r in rows if r['arm']=='joint300'];frozen=[r for r in rows if r['arm']=='frozen250']
    joint_reports=list(reports['joint300'].values())
    summary=dict(minimum_frozen250_ratio=min(r['full_minimum'] for r in frozen),minimum_joint_residual_ratio=min(r['residual_minimum'] for r in joint),
        minimum_joint_full_ratio=min(r['full_minimum'] for r in joint),joint_pose_determinant_range=[min(r['pose_determinant'] for r in joint),max(r['pose_determinant'] for r in joint)],
        joint_outside_fraction_range=[min(r['final_outside_fraction'] for r in joint),max(r['final_outside_fraction'] for r in joint)],
        joint_nominal_masked_support_range=[min(r['final_nominal_descriptor_support']['shared_affine_descriptor_support']['fixed_mask_weighted_fraction'] for r in joint_reports),max(r['final_nominal_descriptor_support']['shared_affine_descriptor_support']['fixed_mask_weighted_fraction'] for r in joint_reports)],
        joint_aligned_descriptor_variance_range=[min(r['final_texture']['selected_pose_aligned_descriptor_masked_channel_variance'] for r in joint_reports),max(r['final_texture']['selected_pose_aligned_descriptor_masked_channel_variance'] for r in joint_reports)],
        all_selected_final_accepted_pair=all(r['selected_stage']==14 for r in joint),complete_call_seconds=costs,comparisons_and_diagnostics_reproduced=True)
    return dict(all50_complete_correct_budgets=True,total_landmarks=labels,maximum_canvas_score_error=score_error,maximum_native_score_error=native_error,
        maximum_saved_pose_composition_error=composition_error,maximum_physical_exponential_determinant_error=exponential_det_error,maximum_outside_fraction_error=outside_error,maximum_texture_variance_error=texture_error,
        maximum_full_objective_recompute_error=objective_error,maximum_static_point_recompute_error=point_error,summary=summary,
        historical_report_erratum='joint image_preprocessing.original_moving_features_no_affine_prewarp=True is stale raw-loader metadata; actual joint descriptors are rebuilt after original-raster A G prewarp',
        rows=rows)


if __name__=='__main__':
    result=actual_postrun() if '--postrun' in sys.argv else (dict(oracle=oracle_checks(),implementation=implementation_checks(),exports=export_and_outside_checks()) if '--implementation' in sys.argv else oracle_checks())
    print(json.dumps(result,indent=2))
