import csv
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/phase5/route1_engineering_sweep.py'


def module():
    assert SCRIPT.exists(), 'sweep driver is not implemented'
    spec = importlib.util.spec_from_file_location('phase5_sweep_test', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def config(m, **kwargs):
    return replace(m.BenchmarkConfig(control=3, image_resolution=8, backend='direct',
                                    dtype='float64', warmup=0, repeats=1), **kwargs)


def receipt(c):
    return dict(config=asdict(c), status='ok', environment={'host':'example'}, commit='abc',
                timing={'mean_end_to_end_seconds':2}, memory={'gpu':None}, per_layer=[])


def test_append_before_next_job_and_resume_complete_config(tmp_path):
    m = module(); p = tmp_path/'all.jsonl'; summary = tmp_path/'all.csv'
    c = config(m); changed = replace(c, warmup=1)
    def interrupted(job):
        if job == changed:
            assert json.loads(p.read_text()) == receipt(c)
            raise KeyboardInterrupt()
        return receipt(job)
    with pytest.raises(KeyboardInterrupt):
        m.run_sweep([c,changed], p, summary, runner=interrupted)
    assert len(p.read_text().splitlines()) == 1
    seen = []
    m.run_sweep([c,changed], p, summary, resume=True,
                runner=lambda c:(seen.append(c) or receipt(c)))
    assert seen == [changed]
    rows = list(csv.DictReader(summary.open()))
    assert len(rows)==2 and rows[0]['host']=='example' and rows[0]['commit']=='abc'
    with pytest.raises(FileExistsError):
        m.run_sweep([c],p,summary,runner=receipt)


@pytest.mark.parametrize('damage', ['truncated','duplicate','missing_config_field','nan','overflow_number','duplicate_json_key','missing_newline','blank'])
def test_resume_rejects_corruption_without_running_or_appending(tmp_path, damage):
    m=module(); c=config(m); p=tmp_path/'bad.jsonl'; data=receipt(c)
    if damage=='missing_config_field': del data['config']['seed']
    text=json.dumps(data)+'\n'
    if damage=='truncated': text += '{'
    elif damage=='duplicate': text += text
    elif damage=='nan': text=text.replace('"mean_end_to_end_seconds": 2','"mean_end_to_end_seconds": NaN')
    elif damage=='overflow_number': text=text.replace('"mean_end_to_end_seconds": 2','"mean_end_to_end_seconds": 1e999')
    elif damage=='duplicate_json_key': text=text.replace('"status": "ok"','"status": "ok", "status": "failure"')
    elif damage=='missing_newline': text=text.rstrip('\n')
    elif damage=='blank': text+='\n'
    p.write_text(text)
    with pytest.raises(ValueError):
        m.run_sweep([c],p,tmp_path/'out.csv',resume=True,runner=lambda c:pytest.fail('must not run'))
    assert p.read_text()==text


def test_keys_cover_every_config_field_and_reject_duplicate_request(tmp_path):
    m=module(); c=config(m)
    for key,value in asdict(c).items():
        replacement = value+'x' if isinstance(value,str) else value+1
        assert m.config_key(c)!=m.config_key(replace(c,**{key:replacement}))
    with pytest.raises(ValueError,match='duplicate'):
        m.run_sweep([c,c],tmp_path/'out.jsonl',tmp_path/'out.csv',runner=receipt)
    assert not (tmp_path/'out.jsonl').exists()


def test_summary_failure_unknowns_and_fallback_are_not_zero(tmp_path):
    m=module(); c=config(m); p=tmp_path/'out.jsonl'; out=tmp_path/'out.csv'
    good=receipt(c)
    good['per_layer']=[dict(true_primal_relative_residual_max=1.1e-5,
        topology=[dict(flip_count=0,minimum_signed_area=.1,global_injectivity_certificate=True)],
        solver_forward=dict(method='stationary_fallback',iterations=[[12,14]],relative_residual=[[1e-5,9e-6]],primary_failure='breakdown'),
        solver_adjoint=dict(method='bicgstab',iterations=[[4,5]],relative_residual=[[3e-6,4e-6]]))]
    bad=replace(c,seed=22)
    def run(c):
        return good if c.seed!=22 else dict(config=asdict(c),status='failure',stage='forward',error={'message':'oops'})
    m.run_sweep([c,bad],p,out,runner=run)
    rows=list(csv.DictReader(out.open()))
    assert rows[0]['fallback_forward_layers']=='1'
    assert rows[0]['max_forward_iterations']=='14'
    assert rows[0]['true_primal_relative_residual_max']=='1.1e-05'
    assert rows[1]['gpu_peak_allocated_bytes']==rows[1]['flip_count_sum']==''
    assert rows[1]['error']=='oops'
    m.run_sweep([bad],p,out,resume=True,runner=lambda c:pytest.fail('failed receipt is still recorded'))


def test_invalid_zero_layer_failure_never_becomes_vacuous_topology_success():
    m = module()
    invalid = config(m, layers=0)
    row = m.summary_row(dict(
        config=asdict(invalid),
        status='failure',
        stage='validation',
        error={'message': 'layers must be positive'},
        per_layer=[],
    ))
    assert row['all_layer_sample_topology_certified'] is None


def test_returned_wrong_config_rejected_and_paths_cannot_alias(tmp_path):
    m=module(); c=config(m); p=tmp_path/'out.jsonl'
    with pytest.raises(ValueError,match='config'):
        m.run_sweep([c],p,tmp_path/'out.csv',runner=lambda c:receipt(replace(c,seed=99)))
    assert p.read_text()==''
    with pytest.raises(ValueError):
        m.run_sweep([c],p,p,resume=True,runner=receipt)


def test_cli_cartesian_submatrix_and_fresh_process_real_receipts(tmp_path):
    module(); p=tmp_path/'tiny.jsonl'; out=tmp_path/'tiny.csv'
    command=[sys.executable,'-B',str(SCRIPT),'--backends','direct','--controls','3',
             '--batches','1','--layers','1','--resolutions','8','--seeds','11','12',
             '--dtype','float64','--device','cpu','--warmup','0','--repeats','1',
             '--jsonl',str(p),'--csv',str(out)]
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows)==2 and all(row['status']=='ok' for row in rows)
    assert [row['config']['seed'] for row in rows]==[11,12]
    assert all(row['solve_counts']['measured_primal_systems']==1 for row in rows)
    assert len(list(csv.DictReader(out.open())))==2
    result=subprocess.run(command+['--resume'],text=True,capture_output=True)
    assert result.returncode==0 and len(p.read_text().splitlines())==2
