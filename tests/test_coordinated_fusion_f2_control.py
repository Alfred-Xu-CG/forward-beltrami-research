import copy
import json
from pathlib import Path
import pytest
from tools import coordinated_fusion_f2_control as control


def source():
    return dict(method='analytic',cycles=1,patch_cells=8,f2_accepted_gain=1.,f2_floor_safety_fraction=1.,
        geometry_backend='stage_cache',joint_prior_backend='eager',match_p1_sampling='existing',
        match_weight=.2,mind_frame='shared_affine',fixed='f',moving='m',affine='a',matches='p',
        device='cpu',levels=[17,33,65,129,257],inner_steps=30)


def test_exact_f2_configuration_diff_and_equal_budget():
    original=source();saved=copy.deepcopy(original)
    a=control.configuration(original,'analytic','a.npz');b=control.configuration(original,'f2','b.npz')
    differences={k for k in vars(a) if getattr(a,k)!=getattr(b,k)}
    assert differences=={'output','method','cycles','f2_floor_safety_fraction','geometry_backend'}
    assert b.patch_cells==8 and b.f2_accepted_gain==1 and b.f2_floor_safety_fraction==.95
    assert control.budget(a)==control.budget(b)==dict(gradient_steps=300,evaluations=310,objective_evaluations=332,failed_trials=0)
    assert original==saved


def test_all50_declared_alternation_failure_retention_and_adapter_after_terminal(tmp_path):
    events=[]
    def loader(_):return [dict(name=f'case{i}',cohort='existing',original_configuration=source()) for i in range(25)]
    def optimize(cfg):
        record=json.loads((tmp_path/'out'/'comparison.json').read_text())
        assert len(record['planned_calls'])==50
        events.append(cfg.method)
        if len(events)==2:raise RuntimeError('deliberate failed F2; no retry')
        return dict(**control.budget(cfg),failures=[],landmarks_used=False,saved_binary_certificate={'valid':True})
    def adapter(arms,_):
        assert len(events)==50
        assert all(m['all50_terminal'] and m['prediction_complete'] for m in arms.values())
        assert arms['f2']['rows'][0]['status']=='failed'
    result=control.run('unused',tmp_path/'out',loader=loader,optimizer=optimize,export_validator=lambda *_:1.,adapter=adapter)
    assert result['successful_calls']==49 and result['all50_terminal']
    assert events==[m for i in range(25) for m in (('analytic','f2') if i%2==0 else ('f2','analytic'))]


def test_actual_tiny_analytic_and_f2(tmp_path):
    from test_coordinated_lung_all20 import assets,fake_match
    from tools.coordinated_lung_all20 import pairs,make_configuration
    from tools.coordinated_real_case import optimize
    args=assets(tmp_path);args.output.mkdir()
    cfg=make_configuration(pairs(args)[0],'analytic',args,production=False)
    fake_match(type('Args',(),dict(fixed=cfg.fixed,moving=cfg.moving,affine=cfg.affine,
        output=cfg.matches,image_side=cfg.image_side))())
    cfg.mind_frame='shared_affine';cfg.match_weight=.2;cfg.patch_cells=8
    cfg.joint_prior_backend='eager';cfg.match_p1_sampling='existing';cfg.f2_floor_safety_fraction=1.
    values={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}
    # A tiny two-level fixture must use valid two-cell F2 patches; production remains8.
    for method in ('analytic','f2'):
        actual=control.configuration(values,method,tmp_path/(method+'.npz'))
        actual.patch_cells=2
        result=optimize(actual)
        assert all(result[k]==v for k,v in control.budget(actual).items())
        assert result['saved_binary_certificate']['valid'] and result['landmarks_used'] is False
