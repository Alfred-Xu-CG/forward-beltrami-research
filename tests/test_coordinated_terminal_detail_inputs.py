import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from tools import coordinated_terminal_detail_inputs as inputs
from tools import coordinated_terminal_detail_batch as batch
from tests.test_coordinated_match_fusion import fixtures,original

def test_nonsquare_layout_preserves_centers_and_discloses_upsampling(tmp_path):
    pixels=np.random.default_rng(5).integers(0,256,(7,11,3),dtype=np.uint8)
    source=tmp_path/'original.png';accepted=tmp_path/'accepted.png'
    Image.fromarray(pixels).save(source)
    layout=dict(original_wh=[11,7],resized_wh=[7,5],padding_xy=[0,1],
        effective_original_to_canvas_scale_xy=[7/11,5/7])
    canvas=Image.new('RGB',(8,8),'white')
    canvas.paste(Image.fromarray(pixels).resize((7,5),Image.Resampling.BILINEAR),(0,1));canvas.save(accepted)
    values,meta=inputs.render_original(source,layout,accepted,side=8)
    assert values.shape==(1,1,16,16) and values.dtype==np.float32
    assert meta['upscaled_axes']==[True,True] and meta['accepted512_reconstruction_exact']
    q=np.array([[0.,0.],[3.3,4.1],[10.,6.]])
    old=((q+.5)*[7/11,5/7]+[0,1])/8
    new=((q+.5)*[14/11,10/7]+[0,2])/16
    assert np.array_equal(old,new)
    Image.new('RGB',(8,8),'black').save(accepted)
    with pytest.raises(ValueError,match='reproduce'):inputs.render_original(source,layout,accepted,side=8)

def test_prepared_float_lift_and_mask_have_no_uint8_roundtrip(tmp_path):
    yy,xx=np.meshgrid(np.arange(512),np.arange(512),indexing='ij')
    image=((xx+yy)%256).astype(np.uint8)
    sources={};roles={};cfg={}
    for role in ('fixed','moving'):
        path=tmp_path/(role+'.png');Image.fromarray(image).convert('RGB').save(path)
        cfg[role]=str(path);sources[role]=path
        roles[role]=dict(original_wh=[512,512],resized_wh=[512,512],padding_xy=[0,0],
            effective_original_to_canvas_scale_xy=[1.,1.])
    cfg['affine']='unused.npz'
    record=inputs.prepare_pair(dict(name='tiny',original_configuration=cfg),tmp_path/'prepared',sources,roles)
    with np.load(record['bundles']['lift512'],allow_pickle=False) as saved:
        gray=1.-torch.from_numpy(image.astype(np.float32))[None,None]/255
        expected=torch.nn.functional.interpolate(gray,scale_factor=2,mode='bilinear',align_corners=False).numpy()
        assert np.array_equal(saved['fixed'],expected)
        assert np.array_equal(saved['fixed_mask'],(gray>.04).float().repeat_interleave(2,-2).repeat_interleave(2,-1).numpy())
        assert np.any(np.abs(saved['fixed']*255-np.round(saved['fixed']*255))>.01)
        assert json.loads(str(saved['metadata']))['point_prediction_side']==512
    assert record['mask_normalized_mass_512']==record['mask_normalized_mass_1024']


def test_lossless_decoder_bridge_preserves_layout_and_checks_identity(tmp_path):
    pixels=np.random.default_rng(9).integers(0,256,(7,11,3),dtype=np.uint8)
    source=tmp_path/'original.jpg';Image.fromarray(pixels).save(source)
    # Deliberately emulate a distinct historical decoder result.
    bridge=tmp_path/'historical.png';Image.fromarray(pixels).save(bridge)
    layout=dict(original_wh=[11,7],resized_wh=[7,5],padding_xy=[0,1],
        effective_original_to_canvas_scale_xy=[7/11,5/7])
    accepted=tmp_path/'accepted.png';canvas=Image.new('RGB',(8,8),'white')
    canvas.paste(Image.fromarray(pixels).resize((7,5),Image.Resampling.BILINEAR),(0,1));canvas.save(accepted)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(decoder=dict(pillow='historical'),rows=[dict(source=str(source),
        source_name=source.name,decoded_rgb=bridge.name,original_wh=[11,7])])))
    decoded=inputs.decoder_cache(manifest)[source.name]
    record=json.loads(manifest.read_text());record['rows'][0]['decoder']={'pillow':'source_specific'}
    manifest.write_text(json.dumps(record))
    assert inputs.decoder_cache(manifest)[source.name]['decoder']['pillow']=='source_specific'
    with pytest.raises(ValueError,match='reproduce'):inputs.render_original(source,layout,accepted,side=8)
    value,meta=inputs.render_original(source,layout,accepted,side=8,decoded=decoded)
    assert meta['accepted512_reconstruction_exact'] and meta['decoding']['decoder']['pillow']=='historical'
    assert value.shape==(1,1,16,16)
    with pytest.raises(ValueError,match='identity'):
        inputs.render_original(source,layout,accepted,side=8,decoded={**decoded,'source_name':'wrong.jpg'})

def test_configuration_preserves_point_units_and_coarse_schedule(tmp_path):
    source=original(tmp_path);source['match_weight']=.2
    cfg=batch.configuration(source,tmp_path/'output.npz')
    assert cfg.image_side==1024 and cfg.image_levels==[32,64,128,256,1024]
    assert cfg.match_robust_scale==8 and cfg.match_weight==.2 and cfg.inner_steps==30
    assert cfg.terminal_source_image_side==512 and cfg.matches==Path(source['matches'])

def test_every_arm_finishes_before_scoring_and_preparation_failures_remain(tmp_path,monkeypatch):
    cases,_=fixtures(tmp_path)
    for case in cases:case['original_configuration']['match_weight']=.2
    monkeypatch.setattr(batch,'native_cases',lambda _: {})
    monkeypatch.setattr(batch,'input_pair',lambda *_: ({},{}))
    output=tmp_path/'run';prepared=[];calls=[]
    def prepare(case,out,*_):
        prepared.append(case['name'])
        if case['name']==cases[0]['name']:raise ValueError('explicit rendering failure')
        return dict(name=case['name'],bundles={arm:str(out/(case['name']+'_'+arm+'.npz')) for arm in inputs.ARMS})
    def optimize(cfg,bundle):
        assert len(prepared)==25
        for arm in inputs.ARMS:
            assert not (output/arm/'miit_predictions.json').exists()
            assert not json.loads((output/arm/'predictions.json').read_text())['prediction_complete']
        calls.append((cfg.output.parent.name,cfg.output.name))
        if cfg.output.parent.name=='lift512' and cfg.output.name.startswith(cases[-1]['name']):
            raise ValueError('last optimizer failure')
        return dict(gradient_steps=300,failed_trials=0,query_count=1024**2,control_vertices=257**2)
    result=batch.run('unused','unused','unused','unused',output,source_loader=lambda _:copy.deepcopy(cases),
        preparer=prepare,optimizer=optimize,validator=lambda *_:.3)
    assert result['prediction_complete'] and len(calls)==48
    assert result['arms']['direct_original']['failed']==1 and result['arms']['lift512']['failed']==2
    for arm in inputs.ARMS:
        manifest=json.loads((output/arm/'predictions.json').read_text())
        assert len(manifest['rows'])==25 and manifest['all50_terminal']
        assert json.loads((output/arm/'miit_predictions.json').read_text())['all50_terminal']
