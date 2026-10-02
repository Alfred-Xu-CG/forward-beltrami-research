"""Independent scalar-coordinate oracle for native shared-affine conversion."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import deeperhistreg as dhr
from tools.coordinated_dhr_native_shared import shared_affine_to_theta,shared_pipeline_class,initial_field_audit,canvas_query_audit
from tools.coordinated_dhr_released_baseline import configure
pipeline_module=sys.modules[dhr.direct_registration.DeeperHistReg_FullResolution.__module__]
warp=pipeline_module.w

def canvas_to_z(q,layout,pad,loader,ratio,wh):
    native=(512*q-np.array(layout['padding_xy']))/np.array(layout['effective_original_to_canvas_scale_xy'])-.5
    return 2*((native+.5)*loader+pad)/(ratio*wh)-1

def z_to_canvas(z,layout,pad,loader,ratio,wh):
    native=((z+1)*ratio*wh/2-pad)/loader-.5
    return ((native+.5)*np.array(layout['effective_original_to_canvas_scale_xy'])+np.array(layout['padding_xy']))/512

def border(field,z):
    h,w=field.shape[:2];x=np.clip((z[...,0]+1)*w/2-.5,0,w-1);y=np.clip((z[...,1]+1)*h/2-.5,0,h-1)
    i=np.floor(x).astype(int);j=np.floor(y).astype(int);a=(x-i)[...,None];b=(y-j)[...,None]
    return (1-a)*(1-b)*field[j,i]+a*(1-b)*field[j,np.minimum(i+1,w-1)]+a*b*field[np.minimum(j+1,h-1),np.minimum(i+1,w-1)]+(1-a)*b*field[np.minimum(j+1,h-1),i]

def main():
    torch.set_num_threads(1);records=[]
    layouts={'fixed':dict(effective_original_to_canvas_scale_xy=[.213,.195],padding_xy=[3,49]),
             'moving':dict(effective_original_to_canvas_scale_xy=[.192,.217],padding_xy=[53,7])}
    a=np.array([[.93,.12],[-.06,1.04]]);b=np.array([-.017,.024]);hw=(75,113);wh=np.array(hw[::-1])
    # Each result is evaluated through scalar native-coordinate conversions, not
    # through matrices returned by the production _frames helper.
    for ratio,rhof,rhom in [(1.,1.,1.),(1.82373046875,1.,1.),(1.82373046875,.83,1.07)]:
        p=dict(initial_resampling=ratio!=1,initial_resample_ratio=ratio,target_resample_ratio=rhof,source_resample_ratio=rhom,pad_2=[[11,12],[3,4]],pad_1=[[5,6],[17,18]])
        theta=shared_affine_to_theta(a,b,layouts,p,hw)
        q=np.array([[.5,.5],[0,0],[1,1],[-.2,1.1],[.123,.897]])
        z=canvas_to_z(q,layouts['fixed'],[3,11],rhof,ratio,wh)
        expected=q@a.T+b
        actual=z_to_canvas(z@theta[:,:2].T+theta[:,2],layouts['moving'],[17,5],rhom,ratio,wh)
        algebra_error=float(np.max(np.abs(actual-expected)));assert algebra_error<1e-14
        field=warp.tc_transform_to_tc_df(torch.tensor(theta,dtype=torch.float32)[None],(1,1,*hw))
        y,x=np.meshgrid(np.arange(hw[0]),np.arange(hw[1]),indexing='ij');nodes=np.stack([2*(x+.5)/wh[0]-1,2*(y+.5)/wh[1]-1],-1)
        fixedq=z_to_canvas(nodes,layouts['fixed'],[3,11],rhof,ratio,wh)
        expectedz=canvas_to_z(fixedq@a.T+b,layouts['moving'],[17,5],rhom,ratio,wh)
        displacement=field[0].numpy().astype(float);node_error=float(abs(displacement-(expectedz-nodes)).max())
        assert node_error<8e-7
        audited=initial_field_audit(field,theta,layouts,p,chunk_rows=9)
        assert abs(audited['maximum_normalized_error']-node_error)<1e-14
        y,x=np.meshgrid(np.arange(512),np.arange(512),indexing='ij');q=np.stack([(x+.5)/512,(y+.5)/512],-1)
        z=canvas_to_z(q,layouts['fixed'],[3,11],rhof,ratio,wh)
        sampled=border(displacement,z.astype(np.float32).astype(float))
        sampledq=z_to_canvas(z+sampled,layouts['moving'],[17,5],rhom,ratio,wh)
        errors=np.linalg.norm((sampledq-(q@a.T+b))*512,axis=-1)
        inside=(abs(z[...,0])<=1-1/wh[0])&(abs(z[...,1])<=1-1/wh[1])
        checked=canvas_query_audit(field,theta,a,b,layouts,p)
        assert checked['native_lattice_inside_count']==int(inside.sum()) and checked['native_lattice_outside_count']==int((~inside).sum())
        assert abs(checked['literal_border_sample_max_canvas_pixels']-errors.max())<1e-4
        if inside.any():assert abs(checked['literal_border_sample_inside_max_canvas_pixels']-errors[inside].max())<1e-4
        records.append(dict(ratio=ratio,loader_fixed=rhof,loader_moving=rhom,algebra_error=algebra_error,actual_field_node_error=node_error,inside_count=int(inside.sum()),outside_count=int((~inside).sum())))
    # Verify the actual release's nonintegral scale-factor center rule directly.
    ramp=torch.arange(113,dtype=torch.float64)[None,None,None,:].expand(1,1,75,113)
    r=1.82373046875;scaled=pipeline_module.u.resample(ramp,r)
    wanted=(np.arange(scaled.shape[-1])+.5)*r-.5
    np.testing.assert_allclose(scaled[0,0,5].numpy(),wanted,rtol=0,atol=3e-14)
    assert abs((113/scaled.shape[-1])-r)>1e-3
    base=dhr.direct_registration.DeeperHistReg_FullResolution;sub=shared_pipeline_class(base,warp.tc_transform_to_tc_df)
    assert {k for k,v in sub.__dict__.items() if callable(v)}=={'run_initial_registration'}
    preset=dhr.configs.default_initial_nonrigid();configured=configure(preset,device='cpu',output=ROOT)
    for section,allowed in [('nonrigid_registration_params',{'device','save_results'}),('preprocessing_params',{'save_results'})]:
        assert {k for k in preset[section] if preset[section][k]!=configured[section][k]}<=allowed
    print(json.dumps(dict(coordinate_cases=records,installed_release_rasterizer_used=True,declared_scale_factor_center_rule_checked=True,only_initial_method_overridden=True,nonrigid_and_preprocessing_parameters_preserved=True),indent=2))

if __name__=='__main__':main()
