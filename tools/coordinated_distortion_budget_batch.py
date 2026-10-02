"""One all25 capped-distortion suffix collection, with an archived same-start control."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_data_metric_batch import configuration
from tools.coordinated_stain_proxy import write_scoring_manifests


def validate_budget(report):
    """Validate maximum-budget accounting without mislabelling early stops as300."""
    if (report.get('budget_mode') != 'distortion_cap'
            or report.get('optimizer') != 'incumbent_ARAP_budget_spectral_QCQP'
            or report.get('numerical_failure') is not False
            or report.get('schedule_complete') is not True
            or report.get('maximum_gradient_steps') != 300):
        raise ValueError('completed capped-distortion schedule without numerical failure required')
    stages = report.get('stages', [])
    expected = [(level, side, direction) for level, side in zip((17,33,65,129,257),(32,64,128,256,512))
                for direction in ([1.,0.],[0.,1.])]
    if len(stages) != 10:
        raise ValueError('all ten scheduled axis stages required')
    budget = report.get('distortion_budget')
    if isinstance(budget, bool) or not isinstance(budget, (float,int)) or not np.isfinite(budget) or budget < 0:
        raise ValueError('finite nonnegative fixed distortion budget required')
    total = {}
    for stage, (level, side, direction) in zip(stages, expected, strict=True):
        counts = stage.get('counts', {})
        gradients = counts.get('gradient_steps')
        if (stage.get('level') != level or stage.get('image_side') != side or stage.get('direction') != direction
                or stage.get('maximum_gradients') != 30 or stage.get('numerical_failure') is not False
                or not isinstance(stage.get('stop_reason'), str)
                or isinstance(gradients, bool) or not isinstance(gradients, int) or not 0 <= gradients <= 30
                or counts.get('data_vjps') != gradients or counts.get('distortion_vjps') != gradients
                or counts.get('numerical_failures') != 0 or stage.get('distortion_budget') != budget):
            raise ValueError('explicit actual stage work and finite retained-map stop required')
        for key in ('anchor_distortion','accepted_distortion','actual_distortion'):
            value = stage.get(key)
            if not isinstance(value,(float,int)) or not np.isfinite(value) or not 0 <= value <= budget:
                raise ValueError('every reported stage must retain original distortion cap')
        for key, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('nonnegative integer work counts required')
            total[key] = total.get(key, 0) + value
    if total != report.get('counts') or total['gradient_steps'] != report.get('gradient_steps'):
        raise ValueError('actual stage counts must sum to suffix counts')
    for state in ('initial','final'):
        for key in ('total','strain','original_total'):
            value = report.get(state,{}).get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
                raise ValueError('finite declared selected/initial data, ARAP and original objectives required')
    if (report.get('initial',{}).get('strain') != budget
            or report.get('final',{}).get('strain',float('inf')) > budget
            or report.get('final',{}).get('total',float('inf')) > report.get('initial',{}).get('total',-float('inf'))):
        raise ValueError('cap comes from incoming runtime R and final selection uses D')
    return budget


def validate_output(cfg, report):
    from tools.coordinated_lung_all20 import _check_safe_export
    from tools.coordinated_f2_floor_compare import _actual_ratio
    validate_budget(report)
    with np.load(cfg.affine, allow_pickle=False) as saved:
        _check_safe_export(cfg.output, report, saved['post_affine_matrix'], saved['post_affine_offset'], 257)
    ratio = _actual_ratio(cfg.output)
    if ratio <= .001 or report.get('query_count') != 512**2 or report.get('control_vertices') != 257**2:
        raise ValueError('actual257-control/512-query export and strict floor required')
    return ratio


def check_control(cases, control):
    if (control.get('prediction_complete') is not True or control.get('all50_terminal') is not True
            or control.get('annotations_read') is not False or control.get('arm') != 'frozen_suffix'
            or [r.get('name') for r in control.get('rows',[])] != [c['name'] for c in cases]):
        raise ValueError('complete original same-incumbent frozen-table control required')
    for case, row in zip(cases, control['rows'], strict=True):
        incoming = str(Path(case['source_report']).with_suffix('.npz'))
        cfg = configuration(case['original_configuration'], Path('unused.npz'))
        plain = {k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}
        if (row.get('status') != 'ok' or row.get('gradient_steps') != 300
                or Path(row.get('incumbent','')) != Path(incoming)
                or any(row.get('configuration',{}).get(k) != v for k,v in plain.items() if k != 'output')):
            raise ValueError('frozen control changed incumbent, point evidence, or original schedule')


def run(fusion_predictions, frozen_suffix_predictions, output, *, source_loader=None,
        optimizer=None, output_validator=None):
    from tools.coordinated_distortion_budget_application import optimize_distortion_budget
    optimizer = optimizer or optimize_distortion_budget
    output_validator = output_validator or validate_output
    cases = (source_loader or source_cases)(fusion_predictions)
    control = json.loads(Path(frozen_suffix_predictions).read_text())
    check_control(cases, control)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = []
    for case in cases:
        cfg = configuration(case['original_configuration'], output/(case['name']+'_analytic.npz'))
        rows.append(dict(**copy.deepcopy(case), status='pending', output=cfg.output.name,
            report=cfg.output.with_suffix('.json').name,
            incumbent=str(Path(case['source_report']).with_suffix('.npz')),
            configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}))
    manifest = dict(protocol='fixed incumbent ARAP budget; unchanged frozen SG+MA and image data; maximum300 NEW outer steps',
        rows=rows, cohort_size=25, specimen_count=4, annotations_read=False, prediction_complete=False,
        budget_mode='distortion_cap', maximum_gradient_steps=300,
        source_fusion_predictions=str(Path(fusion_predictions).resolve()),
        frozen_suffix_predictions=str(Path(frozen_suffix_predictions).resolve()),
        changed_variables=['optimizer','ARAP penalty replaced by incoming ARAP cap','own full-D selector','output'],
        comparison_scope='same incoming map and nominal maximum schedule; different objective/optimizer/selection packages, not equal completed work',
        costs='historical fusion300 and its initializer/SG/MA plus this suffix; none are free')
    started = time.perf_counter()
    def persist():
        manifest['elapsed_seconds'] = time.perf_counter()-started
        (output/'predictions.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    persist()
    for case, row in zip(cases, rows, strict=True):
        tick = time.perf_counter()
        try:
            cfg = configuration(case['original_configuration'], output/row['output'])
            result = optimizer(cfg, initial_map=Path(row['incumbent']))
            if str(cfg.device).startswith('cuda'):
                torch.cuda.synchronize()
            for key in ('gradient_steps','counts','maximum_gradient_steps','numerical_failure','schedule_complete',
                'objective_evaluations','evaluations','initial','final','distortion_budget','budget_mode','stages',
                'initial_map','selected_stage','output_selection','saved_binary_certificate','peak_allocated_bytes',
                'loading_seconds','feature_seconds','optimize_seconds','serialization_seconds','certification_seconds',
                'end_to_end_seconds','control_vertices','query_count','optimizer'):
                row[key] = result.get(key)
            if result.get('numerical_failure'):
                raise RuntimeError('numerical failure retained in report; output is diagnostic, not a successful registration')
            row.update(actual_minimum_corner_ratio=output_validator(cfg,result), status='ok')
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
        row['complete_call_seconds'] = time.perf_counter()-tick
        persist()
        print(json.dumps(dict(name=row['name'],status=row['status'],gradients=row.get('gradient_steps'))),flush=True)
    manifest.update(prediction_complete=True, all25_terminal=True)
    persist()
    write_scoring_manifests(manifest, output)
    for cohort in ('miit','existing'):
        path = output/(cohort+'_predictions.json')
        value = json.loads(path.read_text())
        value.update(budget_mode='distortion_cap', maximum_gradient_steps=300, all25_terminal=True)
        if cohort == 'miit':
            for row in value['rows']:
                row['archived_raw_matches'] = row.get('raw_matches')
                row['raw_matches'] = dict(path=row['methods']['analytic']['configuration']['matches'])
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fusion-predictions','frozen-suffix-predictions','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    result=run(**vars(parser.parse_args()))
    if any(r['status']!='ok' for r in result['rows']):
        raise SystemExit(1)


if __name__=='__main__':
    main()
