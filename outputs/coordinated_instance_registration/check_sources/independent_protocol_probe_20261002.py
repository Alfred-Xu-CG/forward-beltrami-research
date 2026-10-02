import sys, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
import torch
import SimpleITK as sitk

root=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(root),str(root/'src')]
from tools.coordinated_score_case import p1_at_queries_numpy
from tools.digital_compare_appearance import dhr_map_at_unit_queries
data=Path('D:/QC_optimization_data')
miit=data/'miit_v4/extracted/test_data/test_data/source_data'
run=root/'outputs/coordinated_instance_registration/miit_three_rotations_t153'

def unit(p,l):return ((p+.5)*np.array(l['effective_original_to_canvas_scale_xy'])+l['padding_xy'])/512
def points(path,x='x',y='y',label='label'):
    return {r[label].strip():np.array([float(r[x]),float(r[y])]) for r in csv.DictReader(path.open(encoding='utf-8-sig'))}
def outside(q):return ((q<0)|(q>1)).any(-1)
def foreground(path):return 1-np.array(Image.open(path).convert('L'),dtype=float)/255>.04
def inspect_pair(folder,name,layout,pa,pb,shift=0.):
    with np.load(folder/(name+'_affine.npz')) as a:A=a['post_affine_matrix'].astype(float);b=a['post_affine_offset'].astype(float)
    ids=[k for k in pa if np.isfinite(pa[k]).all() and np.isfinite(pb[k]).all()]
    q=unit(np.stack([pa[k] for k in ids])+shift,layout['fixed'])
    target=unit(np.stack([pb[k] for k in ids])+shift,layout['moving'])
    residual_target=(target-b)@np.linalg.inv(A).T
    raw=json.loads((folder/(name+'_raw_matches.json')).read_text())
    rq=np.array(raw['source_points_unit']);rp=np.array(raw['target_points_unit']);world=rp@A.T+b
    eligible=~outside(world)
    l=layout['moving'];lo=np.array(l['padding_xy'])/512;hi=(np.array(l['padding_xy'])+l['resized_wh'])/512
    padtarget=((world<lo)|(world>hi)).any(-1)&eligible
    output={'n':len(ids),'target_outside_affine_image':int(outside(residual_target).sum()),
            'residual_target_min_edge_distance_px':float(np.minimum(residual_target,1-residual_target).min()*512),
            'matches':len(rq),'eligible':int(eligible.sum()),'eligible_match_target_in_canvas_padding':int(padtarget.sum())}
    maps={'affine':q@A.T+b}
    for method in ['analytic','f2']:
        p=folder/(name+'_'+method+'.npz')
        if p.exists():
            with np.load(p) as v:verts=v['vertices'][0].astype(float)@A.T+b
            maps[method]=p1_at_queries_numpy(verts,q,'ac')
    dhr=folder/(name+'_dhr/common_affine_dhr/Results_Final')
    if dhr.exists():
        field=sitk.GetArrayFromImage(sitk.ReadImage(str(dhr/'displacement_field.mha')))
        params=json.loads((dhr/'postprocessing_params.json').read_text())
        maps['dhr']=dhr_map_at_unit_queries(field,params,fixed_size=(512,512),moving_size=(512,512),query=torch.tensor(q,dtype=torch.float32)[None,None])[0,0].numpy()
    output['scores']={k:dict(mean=float(np.linalg.norm((v-target)*512,axis=1).mean()),p90=float(np.percentile(np.linalg.norm((v-target)*512,axis=1),90))) for k,v in maps.items()}
    return output

for moving,fixed in [(2,3),(7,8),(10,11)]:
    name=f'miit_{moving}_to_{fixed}';layout=json.loads((run/(name+'_layout.json')).read_text())
    for key,sec in [('moving',moving),('fixed',fixed)]:
        original=miit/str(sec)/'images/image.tif'
        with tifffile.TiffFile(original) as t:
            shape=t.series[0].shape;axes=t.series[0].axes;arr=t.asarray();meta={k:str(t.pages[0].tags[k].value) for k in ['XResolution','YResolution','ResolutionUnit','Orientation'] if k in t.pages[0].tags}
        image=Image.open(original).convert('RGB');l=layout[key]
        expected=Image.new('RGB',(512,512),'white');expected.paste(image.resize(tuple(l['resized_wh']),Image.Resampling.BILINEAR),tuple(l['padding_xy']))
        saved=Image.open(run/f'{name}_{key}512.png')
        mask=tifffile.imread(miit/str(sec)/'masks/tissue_mask.tif')
        print(json.dumps(dict(section=sec,shape=shape,axes=axes,tags=meta,pil_equals_tiff=bool(np.array_equal(np.array(image),arr)),canvas_max_difference=int(np.abs(np.array(expected,dtype=int)-np.array(saved,dtype=int)).max()),tissue_mask_shape=mask.shape,tissue_mask_unique=np.unique(mask).tolist(),intensity_mask_fraction=float(foreground(run/f'{name}_{key}512.png').mean()))))
    pa=points(miit/str(fixed)/f'landmarks/{fixed:02}.csv');pb=points(miit/str(moving)/f'landmarks/{moving:02}.csv')
    print(json.dumps(dict(pair=name,probe=inspect_pair(run,name,layout,pa,pb),origin_minus1=inspect_pair(run,name,layout,pa,pb,-1.)['scores'],origin_plushalf=inspect_pair(run,name,layout,pa,pb,.5)['scores'])))

lung=root/'outputs/coordinated_instance_registration/lung_all20_fixedrecipe_t123'
canvas=data/'digital_topology_wsi/lung_lesion3_eval/canvas';annot=data/'digital_topology_wsi/lung_lesion3_eval/annotations50'
affines=data/'digital_topology_wsi/ACROBAT_train_subset/scaleup102/imageonly_extension/lung_all20_hybrid_dynamic_strain02'
stains=dict(he='He',cc10='Cc10-5',cd31='CD31-3',ki67='Ki67-7',prospc='proSPC-4')
layouts={s:json.loads((canvas/('cc10_layout.json' if s=='he' else s+'_layout.json')).read_text())['fixed' if s=='he' else 'moving'] for s in stains}
landmarks={s:{k:(v+.5)/10-.5 for k,v in points(annot/f'29-041-Izd2-w35-{v}-les3.csv',x='X',y='Y',label=' ').items()} for s,v in stains.items()}
lung_rows=[]
for row in json.loads((lung/'predictions.json').read_text())['rows']:
    name=row['name'];f,m=row['fixed_stain'],row['moving_stain'];pa,pb=landmarks[f],landmarks[m]
    with np.load(affines/(name+'_affine.npz')) as ar:A=ar['post_affine_matrix'].astype(float);b=ar['post_affine_offset'].astype(float)
    q=unit(np.array(list(pa.values())),layouts[f]);t=unit(np.array(list(pb.values())),layouts[m]);r=(t-b)@np.linalg.inv(A).T
    vals={};q2=unit(np.array(list(pa.values()))+.45,layouts[f]);t2=unit(np.array(list(pb.values()))+.45,layouts[m])
    for method in ['affine','analytic','f2','dhr']:
        if method=='affine':mapped=q@A.T+b;mapped2=q2@A.T+b
        elif method=='dhr':
            dr=lung/(name+'_dhr/common_affine_dhr/Results_Final');field=sitk.GetArrayFromImage(sitk.ReadImage(str(dr/'displacement_field.mha')));params=json.loads((dr/'postprocessing_params.json').read_text())
            evaluate=lambda x:dhr_map_at_unit_queries(field,params,fixed_size=(512,512),moving_size=(512,512),query=torch.tensor(x,dtype=torch.float32)[None,None])[0,0].numpy()
            mapped=evaluate(q);mapped2=evaluate(q2)
        else:
            with np.load(lung/(name+'_'+method+'.npz')) as ar:v=ar['vertices'][0].astype(float)@A.T+b
            mapped=p1_at_queries_numpy(v,q,'ac');mapped2=p1_at_queries_numpy(v,q2,'ac')
        vals[method]=[float(np.linalg.norm((mapped-t)*512,axis=1).mean()),float(np.linalg.norm((mapped2-t2)*512,axis=1).mean())]
    lung_rows.append(dict(pair=name,outside_affine_image=int(outside(r).sum()),min_edge_px=float(np.minimum(r,1-r).min()*512),scores_current_and_publisher_scale=vals))
print(json.dumps(dict(lung=lung_rows,aggregate={method:np.mean([r['scores_current_and_publisher_scale'][method] for r in lung_rows],axis=0).tolist() for method in ['affine','analytic','f2','dhr']})))
