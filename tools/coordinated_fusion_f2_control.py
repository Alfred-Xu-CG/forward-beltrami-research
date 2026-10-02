"""One fixed all25 current-fusion analytic/F2 comparison; no labels or retries."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from tools.coordinated_conditional_pipeline import ResetAwarePeaks
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_matchanything_batch import validate_export
from tools.coordinated_real_case import optimize
from tools.coordinated_stain_proxy import write_scoring_manifests


def configuration(source, method, output):
    if method not in ('analytic','f2'): raise ValueError('analytic or f2 required')
    values=copy.deepcopy(source)
    required=dict(method='analytic',cycles=1,patch_cells=8,f2_accepted_gain=1.,
        geometry_backend='stage_cache',joint_prior_backend='eager',match_p1_sampling='existing',
        match_weight=.2,mind_frame='shared_affine')
    if any(values.get(k)!=v for k,v in required.items()):
        raise ValueError('unchanged current fusion analytic source required')
    values['output']=Path(output)
    if method=='f2':
        values.update(method='f2',cycles=2,f2_floor_safety_fraction=.95,geometry_backend='existing')
    for key in ('fixed','moving','affine','matches'):values[key]=Path(values[key])
    return argparse.Namespace(**values)


def budget(cfg):
    stages=cfg.cycles*len(cfg.levels)*(2 if cfg.method=='analytic' else 1)
    gradients=stages*cfg.inner_steps
    return dict(gradient_steps=gradients,evaluations=gradients+stages,
                objective_evaluations=gradients+3*stages+2,failed_trials=0)


def scoring_manifests(arms,output):
    if any(m.get('all50_terminal') is not True for m in arms.values()):
        raise ValueError('all50 must terminate before any scoring adapters')
    # Existing adapter places actual analytic rows correctly and retains dataset frames.
    write_scoring_manifests(arms['analytic'],output/'analytic')
    path=output/'analytic'/'miit_predictions.json'
    miit=json.loads(path.read_text(encoding='utf-8'))
    replacements=[r for r in arms['f2']['rows'] if r['cohort']=='miit']
    for row,replacement in zip(miit['rows'],replacements,strict=True):
        if row['name']!=replacement['name']:raise ValueError('ordered MIIT rows required')
        record=copy.deepcopy(replacement)
        for key in ('output','report'):record[key]='../f2/'+record[key]
        row['methods']['f2']=record
        row['status']='ok' if all(v['status']=='ok' for v in row['methods'].values()) else 'partial_failure'
    miit.update(all50_terminal=True,actual_methods=['analytic','f2'],
        protocol=arms['analytic']['protocol'],archived_dhr_reference_only=True)
    path.write_text(json.dumps(miit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    path=output/'analytic'/'existing_predictions.json'
    existing=json.loads(path.read_text(encoding='utf-8'))
    existing.update(rows=[copy.deepcopy(r) for r in arms['f2']['rows'] if r['cohort']=='existing'],
        method='f2',all50_terminal=True)
    (output/'f2'/'existing_predictions.json').write_text(json.dumps(existing,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(predictions,output,*,optimizer=optimize,export_validator=validate_export,
        loader=source_cases,adapter=scoring_manifests):
    cases=loader(predictions)
    if len(cases)!=25 or len({r['name'] for r in cases})!=25:raise ValueError('all25 original cases required')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    arms={}
    for method in ('analytic','f2'):
        folder=output/method;folder.mkdir()
        rows=[]
        for case in cases:
            cfg=configuration(case['original_configuration'],method,folder/(case['name']+'_'+method+'.npz'))
            rows.append(dict(**copy.deepcopy(case),method=method,status='pending',output=cfg.output.name,
                report=cfg.output.with_suffix('.json').name,
                configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}))
        arms[method]=dict(protocol='current fusion evidence: fresh analytic300 versus established F2reserve300',
            method=method,cohort_size=25,specimen_count=4,rows=rows,prediction_complete=False,
            all50_terminal=False,annotations_read=False,
            changed_variables=['output'] if method=='analytic' else ['method','cycles','f2_floor_safety_fraction','geometry_backend','output'])
    schedule=[dict(case_index=i,name=case['name'],method=method)
              for i,case in enumerate(cases) for method in (('analytic','f2') if i%2==0 else ('f2','analytic'))]
    report=dict(protocol=arms['analytic']['protocol'],attempt_denominator=50,planned_calls=schedule,
        prediction_complete=False,all50_terminal=False,annotations_read=False,runs=[],
        scope='Equal300 gradients and common objective, NOT equal walltime, decoder-only, or identical continuation chronology/parameter dimension/internal passes.',
        timing='No extra warmup; first application costs included. One alternating-order paired observation per case, not repeated speed estimate. Synchronized whole calls include load/features/optimizer/export/validation and reset-aware allocation peaks. Process imports, manifest setup and initial device synchronization/context initialization are excluded.',
        f2_geometry='existing eight-cell staggered patch implementation, gain1, original .75 geometry safety, .95 strict floor reserve')
    started=time.perf_counter()
    def persist():
        report['elapsed_seconds']=time.perf_counter()-started
        (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        for method,manifest in arms.items():
            (output/method/'predictions.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    persist() # all50 declared before any call
    for plan in schedule:
        method=plan['method'];row=arms[method]['rows'][plan['case_index']]
        cfg=configuration(cases[plan['case_index']]['original_configuration'],method,output/method/row['output'])
        device=torch.device(cfg.device)
        sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
        tick=time.perf_counter();peaks=None
        try:
            sync();tick=time.perf_counter()
            with ResetAwarePeaks(device) as peaks:
                result=optimizer(cfg)
                for key in ('gradient_steps','evaluations','objective_evaluations','failed_trials','failures',
                    'initial','final','selected_stage','optimize_seconds','loading_seconds','feature_seconds',
                    'serialization_seconds','certification_seconds','saved_binary_certificate','landmarks_used'):
                    row[key]=result.get(key)
                row['budget_complete']=all(row[k]==v for k,v in budget(cfg).items())
                if not row['budget_complete']:
                    raise ValueError('incomplete declared budget; retained, no retry')
                row['actual_minimum_corner_ratio']=export_validator(cfg,result)
                row['status']='ok'
            sync()
        except Exception as error:
            row.update(status='failed',error=f'{type(error).__name__}: {error}')
        row['complete_call_seconds']=time.perf_counter()-tick
        if peaks is not None:row['whole_call_memory']=peaks.report()
        report['runs'].append(dict(**plan,status=row['status'],complete_call_seconds=row['complete_call_seconds']))
        persist()
    for manifest in arms.values():manifest.update(prediction_complete=True,all50_terminal=True)
    report.update(prediction_complete=True,all50_terminal=True,
        successful_calls=sum(r['status']=='ok' for m in arms.values() for r in m['rows']))
    persist()
    adapter(arms,output)
    report['scoring_manifests_ready']=True;persist()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.predictions,args.output)
    print(json.dumps({k:result[k] for k in ('successful_calls','all50_terminal','elapsed_seconds')}))
    if result['successful_calls']!=50:raise SystemExit(1)

if __name__=='__main__':main()
