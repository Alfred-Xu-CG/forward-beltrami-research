import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.coordinated_dhr_native_shared import shared_affine_to_theta, initial_field_audit, canvas_query_audit


def frames():
    layouts = {'fixed': dict(effective_original_to_canvas_scale_xy=[.17,.19], padding_xy=[7,21]),
               'moving': dict(effective_original_to_canvas_scale_xy=[.23,.13], padding_xy=[19,9])}
    padding = dict(target_resample_ratio=1., source_resample_ratio=1.,
                   initial_resampling=True, initial_resample_ratio=1.82373046875,
                   pad_2=[[11,12],[3,4]], pad_1=[[5,6],[17,18]])
    return layouts, padding


def test_exact_native_conjugacy_including_outside_canvas():
    layouts, p = frames()
    a=np.array([[1.1,.07],[-.03,.92]]); b=np.array([.02,-.04]); hw=(61,93)
    theta=shared_affine_to_theta(a,b,layouts,p,hw)
    q=np.array([[.5,.5],[-.4,1.2],[0.,0.],[1.,1.]])
    sf=np.array(layouts['fixed']['effective_original_to_canvas_scale_xy'])
    sm=np.array(layouts['moving']['effective_original_to_canvas_scale_xy'])
    pf=np.array(layouts['fixed']['padding_xy']); pm=np.array(layouts['moving']['padding_xy'])
    native=(512*q-pf)/sf-.5
    z=2*((native+.5)+[3,11])/p['initial_resample_ratio']/np.array(hw[::-1])-1
    mapped=z@theta[:,:2].T+theta[:,2]
    native_m=((mapped+1)*.5*np.array(hw[::-1])*p['initial_resample_ratio']-[17,5])-.5
    actual=((native_m+.5)*sm+pm)/512
    np.testing.assert_allclose(actual,q@a.T+b,atol=3e-15,rtol=0)
    # Replacing the declared interpolation ratio by an extent ratio is not equivalent.
    wrong={**p,'initial_resample_ratio':1.8}
    assert np.max(np.abs(shared_affine_to_theta(a,b,layouts,wrong,hw)-theta))>1e-5


def test_literal_float32_initial_field_and_chunked_audit():
    layouts,p=frames(); hw=(61,93)
    a=np.array([[1.1,.07],[-.03,.92]]); b=np.array([.02,-.04])
    theta=shared_affine_to_theta(a,b,layouts,p,hw)
    t=torch.tensor(theta,dtype=torch.float32)[None]
    field=F.affine_grid(t,(1,1,*hw),align_corners=False)-F.affine_grid(torch.eye(2,3)[None],(1,1,*hw),align_corners=False)
    report=initial_field_audit(field,theta,layouts,p,chunk_rows=7)
    assert report['preprocessed_nodes']==61*93
    assert report['maximum_normalized_error']<2e-5
    assert report['maximum_canvas_pixel_error']<1e-3
    assert report['dense_extrapolation_outside_lattice_guaranteed'] is False
    canvas=canvas_query_audit(field,theta,a,b,layouts,p)
    assert canvas['query_count']==512*512 and canvas['no_queries_dropped']
    assert canvas['analytic_theta64_max_canvas_pixels']<1e-9
    assert canvas['cast_theta32_max_canvas_pixels']<1e-3
    assert canvas['native_lattice_outside_count']>0
    assert canvas['literal_border_sample_outside_max_canvas_pixels']>1


def test_all25_terminal_before_subset_manifests_and_inherited_stages(tmp_path,monkeypatch):
    import argparse
    import json
    from types import SimpleNamespace
    from PIL import Image
    import tools.coordinated_dhr_native_shared as module
    from tools.coordinated_dhr_existing_inputs import existing_rows
    names=['miit_2_to_3','miit_7_to_8','miit_10_to_11']+[r['name'] for r in existing_rows(tmp_path)]
    images=[]
    for role in ('fixed','moving'):
        path=tmp_path/(role+'.png');Image.new('RGB',(31,27)).save(path);images.append(str(path))
    layouts={role:dict(original_wh=[31,27],effective_original_to_canvas_scale_xy=[1.,1.],padding_xy=[0,0])
        for role in ('fixed','moving')}
    rows=[dict(name=name,cohort='miit' if i<3 else 'existing',fixed=images[0],moving=images[1],status='pending',
        shared_affine_matrix=[[1.,0.],[0.,1.]],shared_affine_offset=[.01,.02],accepted512_layouts=layouts)
        for i,name in enumerate(names)]
    monkeypatch.setattr(module,'prepare_cases',lambda args:rows)
    base=dict(device='cpu',run_initial_registration=True,run_nonrigid_registration=True,
        loading_params=dict(loader='tiff',source_resample_ratio=.2,target_resample_ratio=.2),
        saving_params=dict(final_saver='tiff'),preprocessing_params=dict(initial_resolution=4096,save_results=True),
        initial_registration_params=dict(device='cuda',cuda=True,save_results=True),
        nonrigid_registration_params=dict(device='cuda',save_results=True,registration_size=4096,
            num_levels=8,used_levels=8,iterations=[100]*7+[200],learning_rates=[.005]+[.0025]*6+[.0015],
            alphas=[1.5]*7+[1.8]))
    calls=[];output=tmp_path/'output'
    class Pipeline:
        def __init__(self,params):self.registration_parameters=params
        def run_initial_registration(self):raise AssertionError('native initializer must be replaced')
        def run_registration(self,moving,fixed,folder):
            assert not (output/'miit_predictions.json').exists()
            assert not (output/'existing_predictions.json').exists()
            for k in ('registration_size','num_levels','used_levels','iterations','learning_rates','alphas'):
                assert self.registration_parameters['nonrigid_registration_params'][k]==base['nonrigid_registration_params'][k]
            calls.append((moving,fixed))
            if len(calls)==2:raise RuntimeError('intentional failure retained')
            self.pre_source=self.pre_target=self.source=torch.zeros(1,1,27,31)
            self.padding_params=dict(target_resample_ratio=1,source_resample_ratio=1,
                initial_resampling=False,pad_1=[[0,0],[0,0]],pad_2=[[0,0],[0,0]])
            self.run_initial_registration()
            assert self.current_displacement_field is self.initial_displacement_field
            self.preprocessing_time=self.nonrigid_registration_time=.01;self.total_registration_time=.1
            final=__import__('pathlib').Path(folder)/'released_dhr/Results_Final';final.mkdir(parents=True)
            (final/'displacement_field.mha').touch();(final/'postprocessing_params.json').write_text('{}')
    def rasterize(t,shape):return F.affine_grid(t,shape,align_corners=False)-F.affine_grid(torch.eye(2,3)[None],shape,align_corners=False)
    dhr=SimpleNamespace(configs=SimpleNamespace(default_initial_nonrigid=lambda:base),
        direct_registration=SimpleNamespace(DeeperHistReg_FullResolution=Pipeline))
    args=argparse.Namespace(output=output,device='cpu',threads=1,fusion_predictions=tmp_path/'fusion.json')
    result=module.run(args,dhr=dhr,transform_to_field=rasterize)
    assert len(calls)==25 and result['successful_pairs']==24 and result['all25_terminal']
    assert result['rows'][1]['status']=='failed'
    assert len(json.loads((output/'miit_predictions.json').read_text())['rows'])==3
    assert len(json.loads((output/'existing_predictions.json').read_text())['rows'])==22
    assert result['rows'][0]['initial_field_audit']['accepted512_query_audit']['no_queries_dropped']


@pytest.mark.parametrize('a,b',[(np.eye(2),[np.nan,0]),(np.diag([-1,1]),[0,0]),(np.zeros((3,3)),[0,0])])
def test_invalid_affine_rejected(a,b):
    layouts,p=frames()
    with pytest.raises(ValueError):shared_affine_to_theta(a,b,layouts,p,(61,93))
