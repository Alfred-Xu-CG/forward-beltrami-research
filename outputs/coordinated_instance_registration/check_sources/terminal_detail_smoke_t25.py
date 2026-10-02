"""Actual-source label-free rendering and both terminal1024 smoke calls."""
import argparse
import json
from pathlib import Path
import time
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_terminal_detail_batch import configuration,validate_output
from tools.coordinated_terminal_detail_inputs import input_pair,native_cases,prepare_pair,ARMS,decoder_cache
from tools.coordinated_terminal_detail import optimize_terminal_detail

root=Path.cwd()
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=root/'results/terminal_detail_smoke_t25')
parser.add_argument('--decoded-source-manifest',type=Path)
args=parser.parse_args();out=args.output
decoded_sources=decoder_cache(args.decoded_source_manifest)
out.mkdir(exist_ok=False)
cases=source_cases(root/'results/match_fusion_all50_t23/fusion')
native=native_cases(root/'data_native22_t20/remote_inputs.json')
report=dict(annotations_read=False,actual_rendering=[],smoke=[])
for case in cases:
    sources,layouts=input_pair(case,root/'data_terminal_t25/layouts',native,root/'data_miit_t15/source_data')
    prepared=prepare_pair(case,out/'rendered_inputs',sources,layouts,decoded_sources=decoded_sources)
    report['actual_rendering'].append(prepared)
    (out/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
case=cases[0]
for arm in ARMS:
    cfg=configuration(case['original_configuration'],out/(arm+'.npz'))
    start=time.perf_counter()
    result=optimize_terminal_detail(cfg,Path(report['actual_rendering'][0]['bundles'][arm]))
    ratio=validate_output(cfg,result)
    report['smoke'].append(dict(arm=arm,seconds=time.perf_counter()-start,minimum_corner_ratio=ratio,
        gradient_steps=result['gradient_steps'],query_count=result['query_count'],control_vertices=result['control_vertices'],
        peak_allocated_bytes=result['peak_allocated_bytes'],terminal_evidence=result['terminal_evidence']))
    (out/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(prepared_pairs=len(report['actual_rendering']),
    smoke=[{k:v for k,v in row.items() if k!='terminal_evidence'} for row in report['smoke']]),indent=2),flush=True)
