"""One explicitly label-informed frozen-affine-feature pilot; no annotation input.

Reuses the three original pairs, affines, inputs and machine matches. Only
mind_frame and output paths change. Native DHR is an unchanged archived reference.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time
import torch
from tools.coordinated_miit_score import load_manifest
from tools.coordinated_lung_all20 import _check_safe_export
from tools.coordinated_lung_all20_score import _affine
from tools.coordinated_miit_transfer import _finite_json,_actual_ratio
from tools import coordinated_real_case as application


def configuration(report,output):
    values=copy.deepcopy(report['configuration'])
    if values.get('mind_frame','original')!='original':
        raise ValueError('original-frame archived recipe required')
    if values.get('method') not in ('analytic','f2') or values.get('output_selection')!='best_full':
        raise ValueError('original analytic/F2 E1-prefix recipe required')
    for name in ('fixed','moving','affine','matches'):
        if values.get(name) is not None:values[name]=Path(values[name])
    values.update(output=Path(output),mind_frame='shared_affine')
    return argparse.Namespace(**values)


def run(source,output):
    manifest,directory=load_manifest(source)
    if output.exists():raise FileExistsError('fresh pilot directory required')
    prefix=Path(os.path.relpath(directory.resolve(),output.resolve()))
    def relocate(value):
        p=Path(value);return str(p if p.is_absolute() else prefix/p)
    rows=copy.deepcopy(manifest['rows'])
    for row in rows:
        for key in ('fixed','moving','affine','layout'):row[key]=relocate(row[key])
        for key in ('path',):
            if key in row.get('raw_matches',{}):row['raw_matches'][key]=relocate(row['raw_matches'][key])
        for key in ('report',):
            if key in row.get('affine_estimation',{}):row['affine_estimation'][key]=relocate(row['affine_estimation'][key])
        native=row['methods']['dhr']
        for key in ('output_directory','report','field','configuration','postprocessing_params'):
            if isinstance(native.get(key),str):native[key]=relocate(native[key])
        native['pilot_reference']='unchanged archived original DHR; NOT rerun'
        for method in ('analytic','f2'):
            row['methods'][method]=dict(status='pending',output=row['name']+'_'+method+'.npz',
                report=row['name']+'_'+method+'.json')
        row['status']='pending'
    report=dict(protocol='ONE exploratory frozen-known-affine descriptor pilot',cohort_size=3,
        prediction_complete=False,annotations_read=False,label_informed_development=True,
        labels_informed_variant_choice='same328 development IDs used in prior ranking diagnostic; NOT untouched confirmation',
        source_predictions=str(source),rows=rows,
        changed_variable='mind_frame original to shared_affine ONLY; output paths are fresh',
        frozen_oob='zero prewarped intensity may produce descriptor ones; no validity mask or descriptor padding repair',
        checkpoint_selection='minimum complete NEW E_A full-resolution objective, never TRE',
        objective_caution='E_A and original E_R are different functionals; do not compare numeric totals as a matched energy',
        timing='single serial complete calls include image/feature loading, per-scale prewarp/features/support metadata, optimization/export/certificate; NOT warm ABBA')
    output.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    def persist():
        report['elapsed_seconds']=time.perf_counter()-tick
        (output/'predictions.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    persist()  # ALL pending attempts declared before the first optimize call.
    for original,row in zip(manifest['rows'],rows,strict=True):
        for method in ('analytic','f2'):
            record=row['methods'][method];start=time.perf_counter()
            try:
                old=original['methods'][method]
                if old['status']!='ok':raise ValueError('successful original prediction required')
                old_path=Path(old['report']);old_path=old_path if old_path.is_absolute() else directory/old_path
                config=configuration(json.loads(old_path.read_text(encoding='utf-8')),output/record['output'])
                result=application.optimize(config)
                if config.device.startswith('cuda'):torch.cuda.synchronize()
                if result.get('mind_frame')!='shared_affine':
                    raise ValueError('requested shared-affine frame was not reported; no fallback')
                a,b=_affine(config.affine);_check_safe_export(config.output,result,a,b,config.grid_side)
                clean,bad=_finite_json({key:result.get(key) for key in (
                        'initial','final','gradient_steps','evaluations','objective_evaluations','failed_trials',
                        'saved_binary_certificate','peak_allocated_bytes','feature_seconds','loading_seconds',
                        'serialization_seconds','certification_seconds','optimize_seconds','end_to_end_seconds',
                        'mind_frame','mind_frame_by_resolution')})
                record.update(clean)
                if bad:raise ValueError('nonfinite optimizer diagnostics')
                ratio=_actual_ratio(config.output)
                clean,bad=_finite_json(dict(actual_minimum_corner_ratio=ratio))
                record.update(clean)
                record['actual_strict_floor_valid']=not bad and ratio>config.minimum_jacobian
                if not record['actual_strict_floor_valid']:raise ValueError('strict actual eta floor required')
                if method=='f2' and result.get('configuration',{}).get('f2_floor_safety_fraction')!=.95:
                    raise ValueError('explicit F2 floor reserve .95 must be recorded')
                record.update(status='ok',budget_complete=result['gradient_steps']==300 and result['failed_trials']==0,
                    expected_gradient_steps=300)
            except Exception as error:
                record.update(status='failed',error=f'{type(error).__name__}: {error}')
            record['complete_call_seconds']=time.perf_counter()-start;persist()
        row['status']='ok' if all(r['status']=='ok' for r in row['methods'].values()) else 'partial_failure'
        persist();print(json.dumps(dict(name=row['name'],status=row['status'])),flush=True)
    report['prediction_complete']=True;persist();return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.predictions,a.output)
    if any(row['status']!='ok' for row in r['rows']):raise SystemExit(1)


if __name__=='__main__':main()
