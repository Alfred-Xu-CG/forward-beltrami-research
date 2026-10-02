"""Read-only numerical provenance of the actual Histo common canvas affine."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import SimpleITK as sitk
from PIL import Image

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
DATA=Path('D:/QC_optimization_data/digital_topology_wsi')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.digital_compare_appearance import dhr_map_at_unit_queries
torch.set_num_threads(2)
read=lambda p:json.loads(Path(p).read_text())


def literal_field_target(field,params,query,fw,mw):
    """Independent float64 center-frame formulas and literal border bilinear."""
    pf=np.array([params['pad_2'][1][0],params['pad_2'][0][0]],float)
    pm=np.array([params['pad_1'][1][0],params['pad_1'][0][0]],float)
    r=params['initial_resample_ratio'];sf=params['target_resample_ratio'];sm=params['source_resample_ratio']
    x=(query*np.array(fw)*sf+pf)/r-.5
    cx=np.clip(x[...,0],0,field.shape[2]-1);cy=np.clip(x[...,1],0,field.shape[1]-1)
    ix=np.floor(cx).astype(int);iy=np.floor(cy).astype(int)
    jx=np.minimum(ix+1,field.shape[2]-1);jy=np.minimum(iy+1,field.shape[1]-1)
    u=cx-ix;v=cy-iy
    f=field.transpose(1,2,0).astype(float)
    sampled=(1-u)[...,None]*(1-v)[...,None]*f[iy,ix]+u[...,None]*(1-v)[...,None]*f[iy,jx]+(1-u)[...,None]*v[...,None]*f[jy,ix]+u[...,None]*v[...,None]*f[jy,jx]
    return ((x+sampled+.5)*r-pm)/(np.array(mw)*sm)


def main():
    directory=DATA/'DHR_CD68_CD4_initial_only';cfg=read(directory/'config.json');runtime=read(directory/'runtime.json')
    assert cfg['device']=='cpu' and cfg['run_initial_registration'] and cfg['run_nonrigid_registration'] is False
    assert runtime['initial_only'] and runtime['nonrigid_seconds']==0
    native=directory/'HistoReg_CD68_to_CD4_initial_only/Results_Final'
    params=read(native/'postprocessing_params.json');field=sitk.GetArrayFromImage(sitk.ReadImage(str(native/'displacement_field.mha')))
    assert field.shape==(2,991,747) and np.isfinite(field).all()
    with Image.open(runtime['fixed']) as im:fw=im.size
    with Image.open(runtime['moving']) as im:mw=im.size
    assert fw==(7171,9916) and mw==(7470,9890)
    q=np.stack(np.meshgrid(np.arange(257,dtype=np.float32)/256,np.arange(257,dtype=np.float32)/256,indexing='xy'),-1)[None]
    # Reproduce the historic float32 sampling exactly, independently refit with
    # NumPy least squares, and separately check a hand-coded float64 sampler.
    actual=dhr_map_at_unit_queries(field,params,fixed_size=fw,moving_size=mw,query=torch.from_numpy(q)).numpy()
    literal=literal_field_target(field,params,q.astype(float),fw,mw)
    sampler_error=float(abs(actual-literal).max());assert sampler_error<4e-7
    with np.load(ROOT/'docs/digital_topology_wsi/dhr_initial_histo_affine_257_float32.npz') as z:
        assert np.array_equal(z['boundary_reference'],q)
        assert np.array_equal(z['raw_teacher_vertices'],actual)
        archive_A=z['post_affine_matrix'];archive_b=z['post_affine_offset']
    design=np.column_stack((q.reshape(-1,2).astype(float),np.ones(257**2)))
    coefficient=np.linalg.lstsq(design,actual.reshape(-1,2).astype(float),rcond=None)[0]
    A=coefficient[:2].T.astype(np.float32);b=coefficient[2].astype(np.float32)
    assert np.array_equal(A,archive_A) and np.array_equal(b,archive_b)
    for path in (ROOT/'docs/digital_topology_wsi/q1_dhr_initial_affine_histo_257_float32.npz',DATA/'q1_dhr_initial_affine_histo_257_float32.npz'):
        with np.load(path) as z:assert np.array_equal(A,z['post_affine_matrix']) and np.array_equal(b,z['post_affine_offset'])
    layout=read(DATA/'birl_anhir_dev/canvas/histo_layout.json')
    sf=np.array(layout['fixed']['resized_wh'],float)/512;sm=np.array(layout['moving']['resized_wh'],float)/512
    pf=np.array(layout['fixed']['padding_xy'],float)/512;pm=np.array(layout['moving']['padding_xy'],float)/512
    # Row/column scale conjugation, without importing the production converter.
    cm=sm[:,None]*A.astype(float)/sf[None,:];cb=sm*b.astype(float)+pm-cm@pf
    with np.load(DATA/'birl_anhir_dev/canvas/histo_initial_affine.npz') as z:
        assert np.array_equal(cm.astype(np.float32),z['post_affine_matrix'])
        assert np.array_equal(cb.astype(np.float32),z['post_affine_offset'])
        actualA=z['post_affine_matrix'];actualb=z['post_affine_offset']
    with np.load(ROOT/'outputs/coordinated_instance_registration/match_fusion_all50_t23/fusion/histo_analytic.npz') as z:
        assert np.array_equal(actualA,z['post_affine_matrix']) and np.array_equal(actualb,z['post_affine_offset'])
    provenance=read(DATA/'birl_anhir_dev/canvas/histo_initial_affine.json')
    assert np.array_equal(provenance['native_matrix'],A) and np.array_equal(provenance['native_offset'],b)
    assert abs(cm-np.array(provenance['canvas_matrix_float64'])).max()<3e-16
    report=dict(scope='read-only actual Histo initializer provenance; no labels, registration, fitting optimization or map edits',
        saved_initial_only_field_reproduces_all66049_raw_targets_bitwise=True,
        independent_literal_float64_sampler_max_error=sampler_error,
        ordinary_all_vertex_LS_float32_native_A_b_bitwise=True,
        independent_canvas_conjugation_float32_A_b_bitwise=True,
        current_fusion_map_affine_bitwise_same=True,native_matrix=A.tolist(),native_offset=b.tolist(),
        canvas_matrix=actualA.tolist(),canvas_offset=actualb.tolist(),
        config=dict(device=cfg['device'],loader_ratios=[cfg['loading_params']['source_resample_ratio'],cfg['loading_params']['target_resample_ratio']],
            initial_resolution=cfg['preprocessing_params']['initial_resolution'],
            initial_registration=cfg['initial_registration_params'],run_nonrigid_registration=False),
        conclusion='actual common positive affine is LS-factored from the saved DHR INITIAL-ONLY field and conjugated to canvas, not a generic direct SG similarity and not extracted from a nonrigid optimized field')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
