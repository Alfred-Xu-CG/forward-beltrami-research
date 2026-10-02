"""Small deterministic tests; independent ideal-absorption oracle, no labels."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from tools.coordinated_stain_proxy import (
    hematoxylin_proxy, load_proxy_image, configuration, source_cases,
    unwarped_mind_diagnostics, write_scoring_manifests,
)
from tools.coordinated_real_case import load_registration_evidence, Evidence
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def absorption(h,e,d):
    # Independent channelwise forward model, not the implementation's B.
    return np.exp(-np.stack((.65*h+.07*e+.27*d,
                            .70*h+.99*e+.57*d,
                            .29*h+.11*e+.78*d),axis=-1))


def test_pure_mixed_orientation_and_fixed_support_invariance():
    h=np.linspace(0,2,60).reshape(6,10)
    zeros=np.zeros_like(h); support=np.ones_like(h,dtype=bool)
    expected=-np.expm1(-h)
    expected=np.clip(expected/np.quantile(expected,.99,method="linear"),0,1).astype(np.float32)
    for e,d in ((zeros,zeros),(h*.3,h*.2),(h*.8+.1,h*.2+.7)):
        feature,diag=hematoxylin_proxy(absorption(h,e,d),support)
        np.testing.assert_allclose(feature,expected,atol=2e-7,rtol=0)
        assert not diag["numerical_floor_active"]
    for e,d in ((h,zeros),(zeros,h)):
        feature,_=hematoxylin_proxy(absorption(zeros,e,d),support)
        assert feature.max()<2e-9


def test_scalar_cofactor_oracle_and_linear_quantile():
    rgb=np.random.default_rng(9).uniform(.02,1,(11,13,3))
    od=-np.log(rgb)
    recovered=(.7095*od[...,0]-.0249*od[...,1]-.2274*od[...,2])/.377799
    h=1-np.exp(-np.maximum(recovered,0))
    support=np.indices(h.shape).sum(axis=0)%3!=0
    feature,diag=hematoxylin_proxy(rgb,support)
    scale=np.quantile(h[support],.99,method="linear")
    np.testing.assert_allclose(feature,np.clip(h/scale,0,1).astype(np.float32),atol=1e-7)
    assert diag["s_raw"]==pytest.approx(scale,abs=2e-15)
    assert diag["negative_h_fraction"]==np.mean(recovered[support]<0)


def test_white_zero_rgb_negative_h_and_empty_support():
    rgb=np.array([[[1.,1.,1.],[0.,0.,0.],[1.,.1,1.]]])
    feature,diag=hematoxylin_proxy(rgb,np.ones((1,3),bool))
    assert feature.dtype==np.float32 and np.isfinite(feature).all()
    assert feature.min()>=0 and feature.max()<=1
    assert feature[0,0]==feature[0,2]==0
    assert diag["negative_h_fraction"]==1/3
    assert diag["rgb_floor_channel_fraction"]==1/3
    empty,diag=hematoxylin_proxy(rgb,np.zeros((1,3),bool))
    assert diag["s_raw"]==0 and diag["s"]==1e-6 and diag["numerical_floor_active"]
    assert diag["h_zero_fraction"] is None and np.isfinite(empty).all()


def test_allzero_and_sparse_q99zero_floor_without_fallback():
    rgb=np.ones((20,20,3))
    feature,diag=hematoxylin_proxy(rgb,np.ones((20,20),bool))
    assert np.count_nonzero(feature)==0 and diag["s_raw"]==0
    rgb[0,0]=absorption(np.array(.4),np.array(0.),np.array(0.))
    feature,diag=hematoxylin_proxy(rgb,np.ones((20,20),bool))
    assert diag["s_raw"]==0 and diag["numerical_floor_active"]
    assert np.count_nonzero(feature)==1 and feature[0,0]==1


def test_default_mask_affine_evidence_and_frozen_pyramid(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(17)
    paths=[]
    for role in ("fixed","moving"):
        rgb=rng.integers(30,250,(512,512,3),dtype=np.uint8)
        rgb[:60]=255
        path=tmp_path/(role+".png"); Image.fromarray(rgb).save(path); paths.append(path)
    raw_f,raw_m,raw_mask,raw_meta=load_registration_evidence(*paths,512)
    direct_f,_=_read_gray_thumbnail(paths[0],512)
    direct_m,_=_read_gray_thumbnail(paths[1],512)
    assert torch.equal(raw_f,direct_f) and torch.equal(raw_m,direct_m)
    assert torch.equal(raw_mask,(direct_f>.04).float())
    explicit=load_registration_evidence(*paths,512,preprocessing="raw_inverted")
    for a,b in zip((raw_f,raw_m,raw_mask),explicit[:3]): assert torch.equal(a,b)
    assert raw_meta==explicit[3]
    h_f,h_m,h_mask,metadata=load_registration_evidence(*paths,512,preprocessing="hematoxylin_proxy")
    assert torch.equal(h_mask,raw_mask)
    a=torch.tensor([[.8,.1],[-.1,1.1]],dtype=torch.float64); b=torch.tensor([.03,.01],dtype=torch.float64)
    original_a=a.clone();original_b=b.clone()
    for side in (32,64,128,256,512):
        reduced=F.interpolate(h_m,size=(side,side),mode="area")
        mask=F.interpolate(raw_mask,size=(side,side),mode="area")
        assert torch.equal(mask,F.interpolate(h_mask,size=(side,side),mode="area"))
        evidence=Evidence(F.interpolate(h_f,size=(side,side),mode="area"),reduced,a,b,"mind",3.,1.,
                          fixed_mask=mask,interpolation="p1_ac",mind_frame="shared_affine")
        from tools.coordinated_shared_affine_features import prepare_shared_affine_intensity
        from tools.digital_mind_objective_probe import self_similarity
        prepared,_=prepare_shared_affine_intensity(reduced,a,b,mask)
        assert torch.equal(evidence.moving_feature,self_similarity(prepared)[0])
    assert torch.equal(a,original_a) and torch.equal(b,original_b)
    assert len(metadata["moving"]["unwarped_normalized_mind_by_resolution"])==5
    # No second normalization is performed after averaging.
    reduced=F.interpolate(h_f,size=(32,32),mode="area")
    assert float(reduced.max())<.9


def control_configuration():
    return dict(method="analytic",loss="mind",output_selection="best_full",grid_side=257,image_side=512,
        image_levels=[32,64,128,256,512],levels=[17,33,65,129,257],inner_steps=30,cycles=1,
        strain_weight=3.,strain_model="p1_arap",shape_weight=1e-4,match_weight=.1,oob_weight=1.,
        minimum_jacobian=.001,precision="float64",image_precision="float32",interpolation="p1_ac",
        mind_frame="shared_affine",preprocessing="raw_inverted",fixed="fixed.png",moving="moving.png",
        affine="affine.npz",matches="matches.json",output="old.npz",learning_rate=.004)


def test_configuration_changes_exactly_one_preprocessing_choice():
    source=control_configuration(); old=copy.deepcopy(source)
    cfg=vars(configuration(source,"new.npz"))
    comparable={k:str(v) if isinstance(v,Path) else v for k,v in cfg.items()}
    assert {k for k in source if comparable[k]!=source[k]}=={"preprocessing","output"}
    assert source==old
    for key,value in (("seed_initializer","coupled_mind"),("mind_frame","original"),
                      ("fixed_mask","mask.png"),("inner_steps_by_level",[30,30,30,30,90]),
                      ("preprocessing","native_dhr")):
        bad={**source,key:value}
        with pytest.raises(ValueError):configuration(bad,"new.npz")


def test_static_variance_diagnostics_and_no_early_scoring(tmp_path):
    image=torch.zeros(1,1,8,8); support=torch.zeros_like(image);support[:,:,1:7,1:7]=1
    result=unwarped_mind_diagnostics(image,support,levels=(4,8))
    for row in result.values():
        assert row["positive_support_linear_variance_median"]==0
        assert row["original_mask_weighted_fraction_at_or_below_epsilon"]==1
    empty=unwarped_mind_diagnostics(image,torch.zeros_like(image),levels=(4,))
    assert empty["4"]["positive_support_linear_variance_median"] is None
    with pytest.raises(ValueError):write_scoring_manifests(dict(prediction_complete=False,rows=[]),tmp_path)


def test_terminal_adapters_keep_failed_denominators_and_relocate(tmp_path):
    from tools.coordinated_miit_score import load_manifest, PAIRS
    from tools.coordinated_dhr_existing_inputs import existing_rows
    source=tmp_path/"source";source.mkdir()
    output=tmp_path/"output";output.mkdir()
    miit=[]
    for moving,fixed in PAIRS:
        miit.append(dict(name=f"miit_{moving}_to_{fixed}",moving_section=moving,fixed_section=fixed,
            map_direction="fixed_canvas_to_moving_canvas",status="ok",fixed="fixed.png",moving="moving.png",
            affine="affine.npz",layout="layout.json",raw_matches=dict(path="matches.json"),
            methods={k:dict(status="ok",output=k+".npz",report=k+".json") for k in ("analytic","f2","dhr")}))
    existing=[dict(name=r["name"],status="ok") for r in existing_rows(Path("unused"))]
    rows=[]
    for cohort,original in (("miit",miit),("existing",existing)):
        path=source/(cohort+".json")
        path.write_text(json.dumps(dict(rows=original,cohort_size=len(original),
            prediction_complete=True,annotations_read=False)),encoding="utf-8")
        rows.extend(dict(name=r["name"],status="failed",error="deliberate",cohort=cohort,
            output=r["name"]+".npz",source_manifest=str(path)) for r in original)
    write_scoring_manifests(dict(rows=rows,prediction_complete=True,protocol="test",
        changed_variables=["preprocessing","output"]),output)
    value,_=load_manifest(output/"miit_predictions.json")
    assert len(value["rows"])==3 and value["all25_terminal"]
    assert all(r["methods"]["analytic"]["status"]=="failed" for r in value["rows"])
    assert (output/value["rows"][0]["layout"]).resolve()==source/"layout.json"
    assert (output/value["rows"][0]["methods"]["f2"]["output"]).resolve()==source/"f2.npz"
    value=json.loads((output/"existing_predictions.json").read_text(encoding="utf-8"))
    assert len(value["rows"])==22 and all(r["status"]=="failed" for r in value["rows"])
