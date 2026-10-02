"""Single exact-recipe 257-control GPU smoke, without annotation access."""
import argparse
import json
from pathlib import Path
import time
from tools.coordinated_joint_pose_batch import source_cases,configuration
from tools.coordinated_joint_pose import optimize_joint_pose,validate_joint_export

parser=argparse.ArgumentParser()
parser.add_argument('--fusion-predictions',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
case=source_cases(args.fusion_predictions)[0]
cfg=configuration(case['original_configuration'],'joint300',args.output)
start=time.perf_counter()
report=optimize_joint_pose(cfg)
validation=validate_joint_export(cfg.output,minimum_jacobian=cfg.minimum_jacobian)
assert validation['valid'] and report['failed_trials']==0
assert report['gradient_steps']==300 and report['pose_gradient_steps']==50 and report['residual_gradient_steps']==250
print(json.dumps(dict(name=case['name'],complete_call_seconds=time.perf_counter()-start,
    initial=report['initial'],final=report['final'],pose=report['pose_parameters'],
    validation=validation,peak_allocated_bytes=report['peak_allocated_bytes'],
    gradient_steps=report['gradient_steps'],objective_evaluations=report['objective_evaluations'],
    selected_stage=report['selected_stage'],annotations_read=False)),flush=True)
