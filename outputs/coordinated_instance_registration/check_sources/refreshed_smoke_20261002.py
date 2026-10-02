"""One actual MIIT case: current-map refresh and matched incoming-map suffix.

This checks implementation, timing and topology without reading annotations.
"""
import json
from pathlib import Path
import torch
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_data_metric_batch import configuration
from tools.coordinated_matchanything_batch import validate_table, validate_export
from tools.coordinated_match_fusion import fuse_tables
from tools.coordinated_refreshed_matches import FrozenDeformationMatchAnything
from tools.coordinated_real_case import optimize

torch.set_num_threads(2)
output = Path('results/refreshed_match_smoke_t27r')
output.mkdir(exist_ok=False)
case = source_cases(Path('results/match_fusion_all50_t23/fusion/predictions.json'))[0]
cfg = configuration(case['original_configuration'], output/'unused.npz')
incoming = Path(case['source_report']).with_suffix('.npz')
old_table = json.loads(cfg.matches.read_text())
sg_path = Path(old_table['original_sg_table'])
sg = json.loads(sg_path.read_text())
model = FrozenDeformationMatchAnything(Path('matchanything_cache/source'),
    Path('matchanything_cache/weights/matchanything_eloftr.ckpt'), device='cuda')
try:
    table = model.extract(cfg.fixed, cfg.moving, cfg.affine, incumbent=incoming,
                          output=output/'refreshed.json')
    assert table['status'] == 'ok', table['status']
    fused = fuse_tables(sg, table)
    fused.update(original_sg_table=str(sg_path), refreshed_ma_table=str(output/'refreshed.json'))
    (output/'fused.json').write_text(json.dumps(fused, indent=2, allow_nan=False)+'\n')
finally:
    model.close()
results = dict(name=case['name'], annotations_read=False,
               matches=table['eligible_matches'], extraction_seconds=table['extraction_seconds'])
for arm in ('frozen_suffix', 'refreshed_suffix'):
    cfg = configuration(case['original_configuration'], output/(arm+'.npz'))
    if arm == 'refreshed_suffix':
        cfg.matches = output/'fused.json'
    validate_table(cfg)
    result = optimize(cfg, initial_map=incoming)
    ratio = validate_export(cfg, result)
    assert result['gradient_steps'] == 300
    results[arm] = {key: result[key] for key in ('initial', 'final', 'initial_map',
        'gradient_steps', 'peak_allocated_bytes', 'end_to_end_seconds')}
    results[arm]['actual_minimum_corner_ratio'] = ratio
    print(json.dumps({arm: results[arm]}, allow_nan=False), flush=True)
(output/'smoke.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
print('REFRESHED_SMOKE_PASS', flush=True)
