"""Independent literal NumPy pose/image/point oracles; no training or labels."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from independent_stain_proxy_probe_20261002 import literal_p1
from qcopt.neural_bijection.dense.coordinated_correspondence import ImageCorrespondences
from tools.coordinated_real_case import Evidence

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


if __name__=='__main__':
    print(json.dumps(dict(oracle=oracle_checks(),implementation=implementation_checks(),exports=export_and_outside_checks()) if '--implementation' in sys.argv else oracle_checks(),indent=2))
