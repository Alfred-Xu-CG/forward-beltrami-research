"""Label-free actual257-control GPU smoke for the paired timed optimizer."""
import argparse
import json
from pathlib import Path
import time
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_data_metric_batch import configuration,validate_prefix,validate_output
from tools.coordinated_data_metric_application import extract_prefix,optimize_timed_final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion-predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);case=source_cases(args.fusion_predictions)[0]
    cfg=configuration(case['original_configuration'],out/'common_prefix_best.npz')
    start=time.perf_counter();prefix=extract_prefix(cfg);validate_prefix(cfg,prefix)
    records={}
    for arm in ('adam','data_metric'):
        cfg=configuration(case['original_configuration'],out/(arm+'.npz'))
        report=optimize_timed_final(cfg,prefix_best=prefix['prefix_best'],prefix_start=prefix['prefix_start'],arm=arm)
        ratio=validate_output(cfg,report,arm)
        records[arm]={k:report[k] for k in ('initial','final','suffix_gradient_steps','suffix_counts','stages',
            'prefix_best_total','suffix_start_total','peak_allocated_bytes','end_to_end_seconds','selected_stage')}
        records[arm]['minimum_corner_ratio']=ratio
    result=dict(scope='label-free actual257control/512query first predetermined case; not an accuracy result',
        name=case['name'],prefix=prefix,arms=records,elapsed_seconds=time.perf_counter()-start)
    (out/'smoke.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
