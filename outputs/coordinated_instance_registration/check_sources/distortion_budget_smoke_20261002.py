"""Actual257/512 incoming-map distortion-budget smoke; no annotation access."""
import json
from pathlib import Path
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_data_metric_batch import configuration
from tools.coordinated_distortion_budget_application import optimize_distortion_budget
from tools.coordinated_distortion_budget_batch import validate_output

output=Path('results/distortion_budget_smoke_t28')
output.mkdir(exist_ok=False)
case=source_cases(Path('results/match_fusion_all50_t23/fusion/predictions.json'))[0]
cfg=configuration(case['original_configuration'],output/(case['name']+'.npz'))
result=optimize_distortion_budget(cfg,initial_map=Path(case['source_report']).with_suffix('.npz'))
ratio=validate_output(cfg,result)
print(json.dumps(dict(name=case['name'],initial=result['initial'],final=result['final'],
    gradients=result['gradient_steps'],counts=result['counts'],ratio=ratio,
    stop_reasons=[stage['stop_reason'] for stage in result['stages']],
    seconds=result['end_to_end_seconds'],peak=result['peak_allocated_bytes'],
    annotations_read=False)),flush=True)
print('DISTORTION_BUDGET_SMOKE_PASS',flush=True)
