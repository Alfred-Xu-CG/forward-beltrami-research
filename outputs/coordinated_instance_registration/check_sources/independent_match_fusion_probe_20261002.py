"""Literal unequal-mass SG+MA fusion algebra, loader dtype and vertex VJP."""
import copy
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

ROOT=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from tools.coordinated_match_fusion import fuse_tables,configuration,CONVENTION
from tools.coordinated_real_case import load_image_matches
from independent_matchanything_probe_20261002 import literal_value_gradient
torch.set_num_threads(1)


def fixture(count,seed,a,b,scale):
    rng=np.random.default_rng(seed);q=rng.uniform(.01,.99,(count,2));p=rng.uniform(.01,.99,(count,2));c=rng.uniform(.4,1,count)*scale
    q[:4]=[[0,0],[1,0],[0,1],[1,1]];p[:4]=q[:4];c[4]=0
    return dict(fixed='fixed.png',moving='moving.png',post_affine_matrix=a.tolist(),post_affine_offset=b.tolist(),
        image_side=512,coordinate_convention=CONVENTION,targets_manual_landmarks_or_dense_teacher_loaded=False,
        source_points_unit=q.tolist(),target_points_unit=p.tolist(),confidence=c.tolist())


def algebra_check():
    a=np.array([[.92,-.61],[.47,.84]],dtype=np.float32);b=np.array([.25,-.12],dtype=np.float32)
    sg=fixture(35,718,a,b,.3);ma=fixture(81,274,a,b,.97);saved=copy.deepcopy((sg,ma));fused=fuse_tables(sg,ma)
    assert (sg,ma)==saved
    q=np.array(fused['source_points_unit']);p=np.array(fused['target_points_unit']);confidence=np.array(fused['confidence'])
    for key in ('source_points_unit','target_points_unit'):assert fused[key]==sg[key]+ma[key]
    expected=[];masses=[];eligible_counts=[]
    for source in (sg,ma):
        target=np.array(source['target_points_unit']);c=np.array(source['confidence']);world=target@a.T+b
        eligible=np.array([all(0<=v<=1 for v in xy) for xy in world]);mass=sum(x for x,valid in zip(c,eligible) if valid)
        masses.append(mass);eligible_counts.append(int(np.count_nonzero(eligible*c)));expected.extend(c/mass)
        assert mass>1 and not np.isclose(mass,c.sum()) and eligible_counts[-1]>=8
    assert np.allclose(confidence,expected,atol=3e-17,rtol=0) and len(confidence)==116
    axis=np.linspace(0,1,9);yy,xx=np.meshgrid(axis,axis,indexing='ij');vertices=np.stack((xx,yy),-1)
    vertices[1:-1,1:-1]+=np.random.default_rng(197).normal(0,.009,(7,7,2))
    originals=[literal_value_gradient(vertices,np.array(r['source_points_unit']),np.array(r['target_points_unit']),np.array(r['confidence']),a,b) for r in (sg,ma)]
    wanted=.1*(originals[0][0]+originals[1][0]);gradient=.1*(originals[0][1]+originals[1][1])
    with tempfile.TemporaryDirectory(prefix='independent_fusion_',dir=ROOT/'outputs/coordinated_instance_registration') as directory:
        path=Path(directory)/'fused.json';path.write_text(json.dumps(fused),encoding='utf-8')
        module,metadata=load_image_matches(path,a,b,fixed_path=Path('fixed.png'),moving_path=Path('moving.png'),image_side=512,device='cpu',dtype=torch.float64,robust_scale=8.)
        assert abs(metadata['confidence_denominator']-2)<5e-16
        value_errors=[];vjp_errors=[]
        for frozen in (False,True):
            if frozen:module.prepare_fixed_p1_sampling(9,9,'ac')
            v=torch.tensor(vertices[None],requires_grad=True);value=.2*module(v,torch.from_numpy(a).double(),'p1_ac');actual=torch.autograd.grad(value,v)[0].numpy()[0]
            value_errors.append(abs(float(value.detach())-wanted));vjp_errors.append(float(np.abs(actual-gradient).max()))
            assert value_errors[-1]<2e-14 and vjp_errors[-1]<2e-14
        same=fuse_tables(sg,sg);path.write_text(json.dumps(same),encoding='utf-8')
        module,_=load_image_matches(path,a,b,fixed_path=Path('fixed.png'),moving_path=Path('moving.png'),image_side=512,device='cpu',dtype=torch.float64,robust_scale=8.)
        assert abs(float(.2*module(torch.from_numpy(vertices[None]),torch.from_numpy(a).double(),'p1_ac'))-.2*originals[0][0])<2e-14
    # A low positive eligible mass can be valid; an ineligible row can make the
    # same rescaling incompatible with the old confidence<=1 contract.
    low=fixture(12,344,a,b,.001);low['target_points_unit']=[[.5,.5]]*12;low['confidence']=[.001]*12
    valid=fuse_tables(low,low);assert max(valid['confidence'])<1
    bad=copy.deepcopy(low);bad['target_points_unit'][-1]=[0.,1.];bad['confidence'][-1]=1.
    assert ((np.array(bad['target_points_unit'][-1])@a.T+b)<0).any()
    variants=[bad,{**sg,'confidence':[0.]*35},{**sg,'confidence':[float('nan')]+sg['confidence'][1:]},
              {**sg,'post_affine_offset':[.3,-.12]},{**sg,'fixed':'wrong.png'}]
    for invalid in variants:
        try:fuse_tables(invalid,ma)
        except ValueError:pass
        else:raise AssertionError('invalid/incompatible table accepted or clipped')
    return dict(rows_preserved=116,original_eligible_masses=masses,positive_eligible_counts=eligible_counts,
        combined_loader_mass=metadata['confidence_denominator'],value_maximum_error=max(value_errors),VJP_maximum_error=max(vjp_errors),same_source_equals_sg2=True,valid_subunit_mass_accepted=True,incompatible_ineligible_confidence_rejected=True)


def actual_config_check():
    from tools.coordinated_stain_proxy import source_cases
    base=ROOT/'outputs/coordinated_instance_registration';cases=source_cases(base/'miit_multiscale_control_t19',base/'existing22_shared_affine_a300_t20')
    for case in cases:
        original=case['original_configuration']
        for arm,expected in [('fusion',{'matches','match_weight','output'}),('sg2',{'match_weight','output'})]:
            changed=vars(configuration(original,arm,'new.npz','fused.json'))
            assert set(changed)==set(original)
            delta={k for k in changed if changed[k]!=(Path(original[k]) if isinstance(changed[k],Path) else original[k])}
            assert delta==expected and changed['match_weight']==.2 and changed['preprocessing']=='raw_inverted'
    return dict(cases=25,arms=2,fusion_delta=['matches','match_weight','output'],sg2_delta=['match_weight','output'])


def actual_postrun():
    from independent_stain_proxy_probe_20261002 import postrun_checks
    base=ROOT/'outputs/coordinated_instance_registration';directory=base/'match_fusion_all50_t23'
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    outer=read(directory/'predictions.json')
    assert outer['prediction_complete'] and outer['composition_complete'] and outer['attempt_denominator']==50 and not outer['annotations_read']
    manifests={arm:read(directory/arm/'predictions.json') for arm in ('fusion','sg2')}
    assert all(v['prediction_complete'] and v['all50_terminal'] and not v['annotations_read'] and len(v['rows'])==25 and all(r['status']=='ok' and r['budget_complete'] for r in v['rows']) for v in manifests.values())
    assert len(outer['composition_rows'])==25 and all(r['status']=='ok' for r in outer['composition_rows'])
    def local(path):
        return base/str(path).replace('\\','/').split('results/',1)[1]
    table_rows=[]
    for composition,frow,srow in zip(outer['composition_rows'],manifests['fusion']['rows'],manifests['sg2']['rows'],strict=True):
        name=composition['name'];assert name==frow['name']==srow['name']
        fused=read(directory/'fused_tables'/(name+'.json'));sg=read(local(fused['original_sg_table']));ma=read(local(fused['original_ma_table']))
        assert not fused['targets_manual_landmarks_or_dense_teacher_loaded'] and fused['rows_removed']==fused['rows_clipped']==0
        for key in ('source_points_unit','target_points_unit'):
            assert fused[key]==sg[key]+ma[key]
        a=np.array(sg['post_affine_matrix'],dtype=np.float32);b=np.array(sg['post_affine_offset'],dtype=np.float32)
        expected=[];eligible_counts=[]
        for label,table,metadata in zip(('sg','ma'),(sg,ma),fused['composition'],strict=True):
            assert table['post_affine_matrix']==fused['post_affine_matrix'] and table['post_affine_offset']==fused['post_affine_offset']
            p=np.array(table['target_points_unit']);c=np.array(table['confidence']);world=p@a.T+b
            eligible=np.array([all(0<=v<=1 for v in xy) for xy in world]);mass=sum(x for x,valid in zip(c,eligible) if valid)
            assert abs(mass-metadata['original_eligible_confidence_mass'])<1e-10 and metadata['matcher']==label
            normalized=c/mass;expected.extend(normalized);eligible_counts.append(int(np.count_nonzero(eligible*c)))
            assert abs(sum(x for x,valid in zip(normalized,eligible) if valid)-1)<2e-14
        assert np.allclose(fused['confidence'],expected,rtol=3e-14,atol=1e-17)
        report=read(directory/'fusion'/frow['report']);used=report['image_match_evidence'];loader=composition['combined_point_loader']
        for actual in (used,loader):
            assert actual['raw_matches']==len(expected) and actual['eligible_matches']==sum(eligible_counts) and abs(actual['confidence_denominator']-2)<1e-14
        assert Path(report['configuration']['matches']).name==name+'.json' and report['configuration']['match_weight']==.2
        sreport=read(directory/'sg2'/srow['report']);assert sreport['configuration']['matches']==srow['original_configuration']['matches'] and sreport['configuration']['match_weight']==.2
        for arm in ('fusion','sg2'):
            if name.startswith('miit_'):
                score_input=next(r for r in read(directory/arm/'miit_predictions.json')['rows'] if r['name']==name)
                assert score_input['raw_matches']['path']==score_input['methods']['analytic']['configuration']['matches']
        table_rows.append(dict(name=name,total_rows=len(expected),positive_eligible=sum(eligible_counts)))
    arms={arm:postrun_checks('match_fusion_all50_t23/'+arm,point_substitution=True,
        configuration_delta={'matches','match_weight','output'} if arm=='fusion' else {'match_weight','output'}) for arm in ('fusion','sg2')}
    for value in arms.values():
        value.pop('unique_original_canvas_images');value.pop('weak_miit7moving');value.pop('all25_same_recipe_two_key_delta')
    comparison=read(directory/'comparison.json')
    score_rows={'sg1':read(base/'miit_multiscale_control_t19/landmark_scores.json')['rows']+read(base/'existing22_shared_affine_a300_t20/landmark_scores.json')['rows']}
    score_rows.update({arm:read(directory/arm/'miit_scores.json')['rows']+read(directory/arm/'existing_scores.json')['rows'] for arm in arms})
    metrics={arm:{r['name']:(r['methods']['analytic']['metrics'] if r['name'].startswith('miit_') else r['metrics'])['canvas_pixels'] for r in rows} for arm,rows in score_rows.items()}
    pairs={'fusion_minus_sg1':('fusion','sg1'),'sg2_minus_sg1':('sg2','sg1'),'fusion_minus_sg2':('fusion','sg2')}
    assert len(comparison['rows'])==25 and comparison['composition_failures']==0
    for row in comparison['rows']:
        for metric in ('mean','p90','maximum'):
            for arm in metrics:assert abs(row['metrics'][metric]['values'][arm]-metrics[arm][row['name']][metric])<1e-12
            for label,(first,second) in pairs.items():
                assert abs(row['metrics'][metric]['deltas'][label]-(metrics[first][row['name']][metric]-metrics[second][row['name']][metric]))<1e-12
    groups={group:[r['name'] for r in comparison['rows'] if r['cohort']==group] for group in comparison['cohorts']}
    for group,names in groups.items():
        assert len(names)==({'miit':3,'lung_all20':20,'histo':1,'rat_kidney':1}[group])
        for arm in metrics:
            actual=comparison['cohorts'][group]['methods'][arm]
            assert actual['pair_denominator']==actual['scored_pairs']==len(names) and actual['failure_count']==0
            for key,metric in [('mean_pair_mean','mean'),('mean_pair_p90','p90'),('worst_pair_maximum','maximum')]:
                values=[metrics[arm][name][metric] for name in names]
                expected=max(values) if metric=='maximum' else sum(values)/len(values)
                assert abs(actual['all_directions_canvas_pixels'][key]-expected)<1e-12
    for label,(first,second) in pairs.items():
        for key,metric in [('mean_pair_mean','mean'),('mean_pair_p90','p90')]:
            expected=sum(sum(metrics[first][name][metric]-metrics[second][name][metric] for name in names)/len(names) for names in groups.values())/4
            assert abs(comparison['equal_specimen_deltas'][label][key]-expected)<1e-12
    timings={}
    for arm,manifest in manifests.items():
        actual=comparison['costs'][arm]['optimizer_complete_calls'];expected=sum(r['complete_call_seconds'] for r in manifest['rows'])
        assert actual['recorded_count']==actual['denominator']==25 and actual['missing_count']==0 and abs(actual['total']-expected)<1e-10
        timings[arm]=expected
    assert abs(comparison['composition_seconds']-sum(r['complete_call_seconds'] for r in outer['composition_rows']))<1e-10
    historical=read(base/'matchanything_all25_t22/predictions.json');cost=comparison['historical_ma_costs']
    assert cost['one_time_setup_seconds']==historical['model_setup_seconds'] and cost['extraction_calls']==25
    assert abs(cost['extraction_complete_call_seconds']-sum(r['extraction_complete_call_seconds'] for r in historical['rows']))<1e-10
    return dict(all50_terminal_before_scoring=True,all_original_rows_preserved_and_normalized=True,table_rows=table_rows,arms=arms,
        comparison_rows_cohorts_equal_specimen_and_costs_reproduced=True,optimizer_complete_call_seconds=timings,
        composition_seconds=comparison['composition_seconds'],historical_MA_setup_seconds=cost['one_time_setup_seconds'],historical_MA_extraction_seconds=cost['extraction_complete_call_seconds'])


if __name__=='__main__':print(json.dumps(actual_postrun() if '--postrun' in sys.argv else dict(algebra=algebra_check(),configurations=actual_config_check()),indent=2))
