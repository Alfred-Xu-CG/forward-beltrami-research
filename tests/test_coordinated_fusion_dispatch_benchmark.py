import copy
import json
from pathlib import Path
import pytest
from tools import coordinated_fusion_dispatch_benchmark as bench


def test_schedule_and_only_two_dispatch_flags():
    rows=bench.schedule()
    assert len(rows)==14
    assert [r['backend'] for r in rows]==['eager','inductor']+['eager','inductor','inductor','eager']*3
    source=dict(fixed='f',moving='m',affine='a',matches='points',match_weight=.2,inner_steps=30,extra=['unchanged'])
    saved=copy.deepcopy(source)
    for p in rows:
        cfg=vars(bench.configuration(source,p,'out.npz'))
        for key in source:
            assert str(cfg[key])==source[key] if key in ('fixed','moving','affine','matches') else cfg[key]==source[key]
        assert cfg['match_p1_sampling']==('existing' if p['backend']=='eager' else 'frozen')
    assert source==saved


def test_failed_probe_preserves_first_calls_and_all56_denominator(tmp_path,monkeypatch):
    events=[]
    def loader(_):
        return [dict(name=n,configuration=dict(fixed='f',moving='m',affine='a',matches='p'),anchor='anchor.npz') for n in bench.CASES]
    def execute(cfg,plan):
        data=json.loads((tmp_path/'out'/'benchmark.json').read_text())
        assert sum(len(c['runs']) for c in data['cases'])==56
        events.append(plan['backend'])
        return dict(**plan,status='complete',output=str(cfg.output))
    def probe(*args,**kwargs):
        events.append('probe');return dict(passed=False,status='failed')
    monkeypatch.setattr(bench.comparison,'compare_runs',lambda rows:[])
    for key in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):monkeypatch.setenv(key,'old_shared_cache')
    result=bench.run('unused',tmp_path/'out',executor=execute,probe=probe,case_loader=loader)
    assert events==['eager','inductor','probe']
    assert result['attempt_denominator']==56 and result['attempted_calls']==2
    assert result['all_attempts_terminal'] and not result['speed_evidence_eligible']
    assert sum(r['status']=='not_run_after_failure' for c in result['cases'] for r in c['runs'])==54
    assert all(Path(p).is_dir() and not list(Path(p).iterdir()) for p in result['compiler_caches'].values())
    with pytest.raises(FileExistsError):bench.run('unused',tmp_path/'out',case_loader=loader)


def test_actual_archived_fusion_loader_not_sg1():
    path=Path('outputs/coordinated_instance_registration/match_fusion_all50_t23/fusion/predictions.json')
    if not path.exists():pytest.skip('archived local fixture absent')
    rows=bench.load_cases(path)
    assert tuple(r['name'] for r in rows)==bench.CASES
    assert all(r['configuration']['match_weight']==.2 for r in rows)


def test_executor_exception_keeps_all56_terminal(tmp_path,monkeypatch):
    def loader(_):
        return [dict(name=n,configuration=dict(fixed='f',moving='m',affine='a',matches='p'),anchor='anchor.npz') for n in bench.CASES]
    def execute(*_):raise RuntimeError('outer collector failure fixture')
    for key in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):monkeypatch.setenv(key,'old_shared_cache')
    result=bench.run('unused',tmp_path/'out',executor=execute,case_loader=loader)
    assert result['all_attempts_terminal'] and result['attempted_calls']==1
    assert result['cases'][0]['runs'][0]['status']=='failed_no_fallback'
    assert sum(r['status']=='not_run_after_failure' for c in result['cases'] for r in c['runs'])==55
