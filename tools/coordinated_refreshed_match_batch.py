"""One observation refresh versus frozen-table continuation from identical incumbents."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from tools.coordinated_joint_pose_batch import source_cases
from tools.coordinated_data_metric_batch import configuration
from tools.coordinated_matchanything_batch import validate_table,validate_export
from tools.coordinated_match_fusion import fuse_tables
from tools.coordinated_stain_proxy import write_scoring_manifests

ARMS=('frozen_suffix','refreshed_suffix')


def run(fusion_predictions,source_root,checkpoint,output,*,source_loader=None,model_factory=None,
        optimizer=None,export_validator=None):
    from tools.coordinated_refreshed_matches import FrozenDeformationMatchAnything
    from tools.coordinated_real_case import optimize
    model_factory=model_factory or FrozenDeformationMatchAnything;optimizer=optimizer or optimize
    export_validator=export_validator or validate_export
    cases=(source_loader or source_cases)(fusion_predictions);output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);(output/'refreshed_tables').mkdir();(output/'fused_tables').mkdir()
    manifests={};start=time.perf_counter()
    for arm in ARMS:
        (output/arm).mkdir();rows=[]
        for case in cases:
            cfg=configuration(case['original_configuration'],output/arm/(case['name']+'_analytic.npz'))
            if arm=='refreshed_suffix':cfg.matches=output/'fused_tables'/(case['name']+'.json')
            rows.append(dict(**copy.deepcopy(case),arm=arm,status='pending',output=cfg.output.name,
                report=cfg.output.with_suffix('.json').name,incumbent=str(Path(case['source_report']).with_suffix('.npz')),
                configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}))
        manifests[arm]=dict(protocol='same saved fusion300 incumbent;300NEW safe gradients; only MAtable changes',arm=arm,
            cohort_size=25,specimen_count=4,prediction_complete=False,annotations_read=False,rows=rows,
            expected_gradient_steps=300,historical_incumbent_gradients=300,
            changed_variables=['incoming_absolute_map','output']+(['matches'] if arm=='refreshed_suffix' else []),
            unchanged_dense_evidence='original affine-only shared-gray MIND',
            selection='incoming map and accepted suffix states by this arms ownfullE only; no crossarm choice')
    outer=dict(protocol='ONE deformation-conditioned existingMA refresh; all25extractions then all50suffixes',
        prediction_complete=False,extraction_complete=False,annotations_read=False,attempt_denominator=50,
        source_fusion_predictions=str(Path(fusion_predictions).resolve()),extractions=[],
        arm_order='fixed parity: frozen first on even caseindex, refreshed first on odd',
        cost_scope='historical incumbent300+its initializer/SG/oldMA preparation, ONEnewMAsetup/extraction, new300suffix; incumbent is not free')
    def persist():
        outer['elapsed_seconds']=time.perf_counter()-start
        (output/'predictions.json').write_text(json.dumps(outer,indent=2,allow_nan=False)+'\n')
        for arm,value in manifests.items():
            (output/arm/'predictions.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    persist();model=None;setup_error=None
    device=cases[0]['original_configuration']['device']
    if any(c['original_configuration']['device']!=device for c in cases):raise ValueError('one serial device required')
    torch.set_num_threads(cases[0]['original_configuration'].get('threads',2))
    tick=time.perf_counter()
    try:
        model=model_factory(Path(source_root),Path(checkpoint),device=device)
        outer.update(model_setup_status='ok',model_setup=model.setup_report)
    except Exception as error:
        setup_error=f'{type(error).__name__}: {error}';outer.update(model_setup_status='failed',model_setup_error=setup_error)
    outer['model_setup_complete_seconds']=time.perf_counter()-tick;persist()
    try:
        for i,case in enumerate(cases):
            tick=time.perf_counter();record=dict(name=case['name'],status='pending')
            try:
                if setup_error is not None:raise RuntimeError('matcher setup failed: '+setup_error)
                cfg=configuration(case['original_configuration'],output/'unused.npz')
                incumbent=Path(manifests['frozen_suffix']['rows'][i]['incumbent'])
                original_fusion=json.loads(cfg.matches.read_text())
                # Use original SG, not the normalized combined table as a second SG source.
                sg_path=Path(original_fusion['original_sg_table'])
                sg=json.loads(sg_path.read_text())
                table=output/'refreshed_tables'/(case['name']+'.json')
                extracted=model.extract(cfg.fixed,cfg.moving,cfg.affine,incumbent=incumbent,output=table)
                if extracted['status']!='ok':raise ValueError('refreshed matcher has insufficient eligible matches; no fallback')
                fused=fuse_tables(sg,extracted)
                fused.update(original_sg_table=str(sg_path),refreshed_ma_table=str(table),incumbent=str(incumbent))
                cfg.matches=output/'fused_tables'/(case['name']+'.json')
                cfg.matches.write_text(json.dumps(fused,indent=2,allow_nan=False)+'\n')
                metadata=validate_table(cfg)
                record.update(status='ok',table=str(table),fused_table=str(cfg.matches),point_loader=metadata,
                    extraction={k:v for k,v in extracted.items() if k not in ('source_points_unit','target_points_unit','warped_target_points_unit','confidence')})
            except Exception as error:
                record.update(status='failed',error=f'{type(error).__name__}: {error}')
            record['complete_call_seconds']=time.perf_counter()-tick;outer['extractions'].append(record);persist()
            print(json.dumps(dict(name=case['name'],extraction=record['status'])),flush=True)
    finally:
        if model is not None:model.close()
    outer['extraction_complete']=True;outer['model_released_before_optimization']=True;persist()
    for i,case in enumerate(cases):
        order=ARMS if i%2==0 else tuple(reversed(ARMS))
        for arm in order:
            row=manifests[arm]['rows'][i];tick=time.perf_counter()
            try:
                if arm=='refreshed_suffix' and outer['extractions'][i]['status']!='ok':
                    raise ValueError('refresh extraction failed; incumbent not substituted as successful result')
                cfg=configuration(case['original_configuration'],output/arm/row['output'])
                if arm=='refreshed_suffix':cfg.matches=output/'fused_tables'/(case['name']+'.json')
                result=optimizer(cfg,initial_map=Path(row['incumbent']))
                if str(device).startswith('cuda'):torch.cuda.synchronize()
                ratio=export_validator(cfg,result)
                if Path(result['initial_map']['path']).resolve()!=Path(row['incumbent']).resolve():
                    raise ValueError('wrong saved incoming map used')
                fields=('gradient_steps','failed_trials','evaluations','objective_evaluations','initial','final',
                    'initial_map','selected_stage','selected_stage_scope','stages','query_count','control_vertices',
                    'saved_binary_certificate','peak_allocated_bytes','loading_seconds','feature_seconds',
                    'serialization_seconds','certification_seconds','optimize_seconds','end_to_end_seconds')
                row.update({k:result.get(k) for k in fields},status='ok',actual_minimum_corner_ratio=ratio,
                    arm_position_in_pair=order.index(arm),new_suffix_gradient_steps=result['gradient_steps'])
            except Exception as error:row.update(status='failed',error=f'{type(error).__name__}: {error}')
            row['complete_call_seconds']=time.perf_counter()-tick;persist()
            print(json.dumps(dict(name=case['name'],arm=arm,status=row['status'])),flush=True)
    outer['prediction_complete']=True
    outer['arms']={a:dict(successful=sum(r['status']=='ok' for r in m['rows']),failed=sum(r['status']=='failed' for r in m['rows']))
        for a,m in manifests.items()}
    for manifest in manifests.values():manifest.update(prediction_complete=True,all50_terminal=True)
    persist()
    for arm,manifest in manifests.items():
        write_scoring_manifests(manifest,output/arm)
        path=output/arm/'miit_predictions.json';value=json.loads(path.read_text())
        for row in value['rows']:
            row['archived_raw_matches']=row.get('raw_matches')
            row['raw_matches']=dict(path=row['methods']['analytic']['configuration']['matches'],arm=arm)
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return outer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fusion-predictions','source-root','checkpoint','output'):parser.add_argument('--'+name,type=Path,required=True)
    result=run(**vars(parser.parse_args()))
    if any(a['failed'] for a in result['arms'].values()):raise SystemExit(1)


if __name__=='__main__':main()
