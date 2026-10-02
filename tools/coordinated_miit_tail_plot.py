"""Post-hoc worst-center anatomy view; does not modify maps, labels or scoring.

Shows the pre-existing7-to8 original analytic worst center, all original
predictions and the explicitly manual-label oracle. It is NOT a new experiment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tools.coordinated_miit_score import load_manifest,read_points,validate_layout
from tools.coordinated_lung_all20_score import _affine,_safe_map,_native
from tools.coordinated_score_case import p1_at_queries_numpy
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit,canvas_unit_to_original_pixel
from tools.digital_compare_appearance import dhr_map_at_unit_queries


def run(args):
    output=Path(args.output)
    if output.exists() or output.suffix!='.png':raise ValueError('fresh PNG required')
    manifest,directory=load_manifest(args.predictions);row=manifest['rows'][1]
    resolve=lambda value:Path(value) if Path(value).is_absolute() else directory/value
    layout=validate_layout(json.loads(resolve(row['layout']).read_text(encoding='utf-8')))
    score=json.loads(Path(args.scores).read_text(encoding='utf-8'))['rows'][1]
    if score['name']!=row['name']:raise ValueError('matching original7-to8 score required')
    errors=score['methods']['analytic']['metrics']['canvas_pixels']['per_label']
    label=max(sorted(errors),key=lambda key:errors[key])
    native_images={};native_points={}
    for role in ('fixed','moving'):
        section=row[role+'_section'];root=Path(args.source_data)/str(section)
        with Image.open(root/'images/image.tif') as image:native_images[role]=np.asarray(image.convert('RGB'))
        points=read_points(root/'landmarks'/f'{section:02d}.csv',layout[role]['original_wh'],allow_missing_inf=True)
        native_points[role]=points[label]
    query=original_pixel_to_canvas_unit(native_points['fixed'][None],layout['fixed'],512)
    truth=original_pixel_to_canvas_unit(native_points['moving'][None],layout['moving'],512)[0]
    a,b=_affine(resolve(row['affine']));predictions={}
    for method in ('analytic','f2'):
        vertices,_=_safe_map(row['methods'][method],directory,a,b)
        predictions[method]=p1_at_queries_numpy(vertices,query,'ac')[0]
    field,params,_=_native(row['methods']['dhr'],directory,a,b)
    predictions['dhr']=dhr_map_at_unit_queries(field,params,fixed_size=(512,512),moving_size=(512,512),
        query=torch.from_numpy(query.astype(np.float32)).reshape(1,1,1,2))[0,0,0].numpy().astype(float)
    with np.load(args.oracle,allow_pickle=False) as data:
        if not bool(data['label_oracle']) or not bool(data['landmarks_used']):raise ValueError('explicit oracle archive required')
        if not np.array_equal(data['post_affine_matrix'],a) or not np.array_equal(data['post_affine_offset'],b):raise ValueError('same affine required')
        y=data['vertices']
        if y.ndim!=4 or y.shape[0]!=1 or y.shape[-1]!=2:raise ValueError('one explicit oracle vertex table required')
        effective=y[0]@a.astype(float).T+b.astype(float)
    predictions['label oracle']=p1_at_queries_numpy(effective,query,'ac')[0]
    # Coordinate consistency with the existing scores, not a replacement score.
    for method in ('analytic','f2','dhr'):
        if abs(np.linalg.norm(512*(predictions[method]-truth))-score['methods'][method]['metrics']['canvas_pixels']['per_label'][label])>1e-9:
            raise ValueError('plot coordinates disagree with recorded original score')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    for col,role in enumerate(('fixed','moving')):
        with Image.open(resolve(row[role])) as image:canvas=np.asarray(image.convert('RGB'))
        axes[0,col].imshow(canvas);axes[0,col].set_title(role+'512 canvas')
        point=(query[0] if role=='fixed' else truth)*512-.5
        axes[0,col].plot(*point,'rx',ms=10,mew=2,label='manual annotation')
        axes[0,col].set(xlim=(-.5,511.5),ylim=(511.5,-.5))
    colors={'analytic':'#0072B2','f2':'#E69F00','dhr':'#CC79A7','label oracle':'#009E73'}
    for name,p in predictions.items():
        xy=512*p-.5;t=512*truth-.5
        axes[0,1].plot([t[0],xy[0]],[t[1],xy[1]],color=colors[name],lw=1)
        axes[0,1].plot(*xy,'o',mfc='none',mec=colors[name],ms=8,
            label=f'{name}: {np.linalg.norm(512*(p-truth)):.3f}px')
    native_predictions={name:canvas_unit_to_original_pixel(p[None],layout['moving'],512)[0] for name,p in predictions.items()}
    for col,role in enumerate(('fixed','moving')):
        center=native_points[role]
        if role=='fixed':lo=center-200;hi=center+200
        else:
            cloud=np.stack([center,*native_predictions.values()]);lo=cloud.min(0)-140;hi=cloud.max(0)+140
        height,width=native_images[role].shape[:2]
        low=np.maximum(0,np.floor(lo).astype(int));high=np.minimum([width,height],np.ceil(hi).astype(int)+1)
        crop=native_images[role][low[1]:high[1],low[0]:high[0]]
        axes[1,col].imshow(crop,extent=(low[0]-.5,high[0]-.5,high[1]-.5,low[1]-.5))
        axes[1,col].plot(*center,'rx',ms=12,mew=2)
        axes[1,col].set_title(role+' native-resolution context; native pixel axes')
        if role=='moving':
            for name,p in native_predictions.items():
                axes[1,col].plot([center[0],p[0]],[center[1],p[1]],color=colors[name],lw=1)
                axes[1,col].plot(*p,'o',mfc='none',mec=colors[name],ms=9)
    axes[0,1].legend(fontsize=8,loc='upper right')
    for ax in axes.flat:ax.set_xlabel('x pixel index');ax.set_ylabel('y pixel index')
    fig.suptitle(f'Post-hoc7-to8 worst original analytic center: {label}\nManual-label oracle is NOT image registration',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('predictions','scores','source_data','oracle','output'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
