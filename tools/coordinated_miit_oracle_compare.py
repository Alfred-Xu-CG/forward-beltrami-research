"""One fixed warmup+ABBA geometry comparison using manual labels, NOT registration.

No rate/patch/budget search. Every call starts from the SAME archived analytic
map and consumes the SAME107 points. All outputs remain explicit label oracles.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from tools import coordinated_miit_capacity_witness as witness

SCHEDULE=(('warmup','analytic'),('warmup','f2'),
          ('measured','analytic'),('measured','f2'),('measured','f2'),('measured','analytic'))


def run(args):
    output=Path(args.output)
    if output.exists():raise FileExistsError('fresh oracle comparison directory required')
    output.mkdir(parents=True)
    rows=[dict(index=i,scope=scope,method=method,status='pending',
               output=f'{i}_{scope}_{method}_label_oracle.npz')
          for i,(scope,method) in enumerate(SCHEDULE)]
    report=dict(protocol='ONE warmup each then fixed ABBA label-oracle motion comparison',
        label_oracle=True,landmarks_used=True,not_registration_initializer_or_training_teacher=True,
        comparison_complete=False,pair_name='miit_7_to_8',rows=rows,
        predictions=str(args.predictions),source_data=str(args.source_data),device=args.device,
        fixed_scope='same original analytic incoming257map, affine,107points,J,eta.001,boundary; no priors during fit',
        threshold='first certified accepted stage with ALL107 errors <=1canvaspixel; never stop full300budget early',
        budget='300gradients/310trials each; analytic scalar xy stages, F2 two-cycle vector stages; geometry passes/parameters NOT equal',
        interpretation='one known sparse-oracle task; NOT production image registration, population timing or generalization',
        no_sweep='one fixed existing recipe per method, no parameter/budget/patch retuning')
    tick=time.perf_counter()
    def persist():
        report['elapsed_seconds']=time.perf_counter()-tick
        (output/'oracle_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    persist()
    for row in rows:
        call=argparse.Namespace(predictions=Path(args.predictions),source_data=Path(args.source_data),
            output=output/row['output'],pair_name='miit_7_to_8',device=args.device,threads=args.threads,method=row['method'])
        start=time.perf_counter()
        try:
            result=witness.run(call)
            if not result.get('label_oracle') or not result.get('landmarks_used'):
                raise ValueError('explicit oracle declaration required')
            if result.get('configuration',{}).get('method')!=row['method']:
                raise ValueError('requested oracle decoder must be reported; no fallback')
            if (not result.get('budget_complete') or result['counts']['gradient_steps']!=300
                    or result['counts']['decoder_trials']!=310):
                raise ValueError('fixed complete300/310budget required; retain incomplete result')
            if not result.get('saved_binary_certificate',{}).get('valid'):
                raise ValueError('actual final certificate required')
            row.update(status='ok',report=Path(row['output']).with_suffix('.json').name,
                capacity_witness_success=result['capacity_witness_success'],
                counts=result['counts'],fit_seconds=result['fit_seconds'],
                end_to_end_seconds=result['end_to_end_seconds'],peak_allocated_bytes=result['peak_allocated_bytes'],
                first_threshold=result.get('first_certified_max_error_le_one'),
                first_threshold_output=result.get('first_threshold_output'),
                total_stage_scalar_parameter_count=result.get('total_stage_scalar_parameter_count'),
                final_geometry=result['final_geometry'],
                final_canvas={k:v for k,v in result['end_errors']['canvas_pixels'].items() if k!='per_label'},
                end_original_e1=result['end_original_e1'])
        except Exception as error:
            row.update(status='failed',error=f'{type(error).__name__}: {error}',
                artifact_scope='any produced map/report retained; no replacement recipe or discarded attempt')
        row['wrapper_call_seconds']=time.perf_counter()-start
        persist();print(json.dumps({k:row[k] for k in ('index','method','status')}),flush=True)
    report['comparison_complete']=True;persist();return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('predictions','source_data','output'):p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    result=run(p.parse_args())
    if any(row['status']!='ok' for row in result['rows']):raise SystemExit(1)


if __name__=='__main__':main()
