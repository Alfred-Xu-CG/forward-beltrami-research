"""Read-only, label-informed failure illustration; never optimizer input."""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.coordinated_miit_score import read_points
from tools.digital_birl_landmark_score import original_pixel_to_canvas_unit
from tools.coordinated_score_case import p1_at_queries_numpy

BASE=Path('outputs/coordinated_instance_registration')
INPUT=BASE/'miit_three_rotations_t153'
CASE='miit_7_to_8'
OUT=BASE/'match_fusion_all50_t23'
DATA=Path('D:/QC_optimization_data/miit_v4/extracted/test_data/test_data/source_data')

def main():
    layout=json.loads((INPUT/f'{CASE}_layout.json').read_text())
    with np.load(INPUT/f'{CASE}_affine.npz') as f:
        a=f['post_affine_matrix'].astype(float);b=f['post_affine_offset'].astype(float)
    with np.load(OUT/'fusion'/f'{CASE}_analytic.npz') as f:
        y=f['vertices'].astype(float).squeeze()
    fixed=np.asarray(Image.open(INPUT/f'{CASE}_fixed512.png').convert('RGB'))/255.
    moving=np.asarray(Image.open(INPUT/f'{CASE}_moving512.png').convert('RGB'))/255.
    coords=(np.arange(512)+.5)/512
    xx,yy=np.meshgrid(coords,coords)
    query=np.stack((xx,yy),-1)
    sample=torch.from_numpy(2*(query@a.T+b)-1).unsqueeze(0)
    aligned=F.grid_sample(torch.from_numpy(moving).permute(2,0,1).unsqueeze(0),
        sample,mode='bilinear',padding_mode='zeros',align_corners=False)[0].permute(1,2,0).numpy()
    pts={}
    for key,section in [('fixed',8),('moving',7)]:
        pts[key]=read_points(DATA/str(section)/'landmarks'/f'{section:02d}.csv',
            layout[key]['original_wh'],allow_missing_inf=True)
    ids=[k for k in sorted(pts['fixed']) if np.isfinite(pts['fixed'][k]).all() and np.isfinite(pts['moving'][k]).all()]
    q=original_pixel_to_canvas_unit(np.stack([pts['fixed'][k] for k in ids]),layout['fixed'],512)
    target=original_pixel_to_canvas_unit(np.stack([pts['moving'][k] for k in ids]),layout['moving'],512)
    target_aligned=np.linalg.solve(a,(target-b).T).T
    prediction=p1_at_queries_numpy(y,q,'ac')
    errors=512*np.linalg.norm(prediction@a.T+b-target,axis=1)
    scores=json.loads((OUT/'fusion'/'miit_scores.json').read_text())['rows'][1]['methods']['analytic']['metrics']['canvas_pixels']['per_label']
    assert max(abs(errors[i]-scores[k]) for i,k in enumerate(ids))<1e-9
    tables={
        'SG':json.loads((INPUT/f'{CASE}_raw_matches.json').read_text()),
        'MA':json.loads((BASE/'matchanything_all25_t22'/f'{CASE}_matchanything.json').read_text())}
    selected=np.argsort(errors)[-3:][::-1]
    rows=[];fig,axes=plt.subplots(3,3,figsize=(12,11))
    for row,idx in enumerate(selected):
        source=q[idx]*512-.5;truth=target_aligned[idx]*512-.5;pred=prediction[idx]*512-.5
        center=(np.minimum.reduce([source,truth,pred])+np.maximum.reduce([source,truth,pred]))/2
        radius=max(48.,.5*np.max(np.ptp(np.stack([source,truth,pred]),axis=0))+14)
        for ax,img,title in zip(axes[row,:2],[fixed,aligned],['Fixed image','Original moving, affine aligned']):
            ax.imshow(img);ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]+radius,center[1]-radius)
            ax.scatter(*source,marker='+',s=150,c='lime',label='fixed query')
            ax.scatter(*truth,marker='*',s=120,c='cyan',edgecolors='black',linewidths=.5,label='annotated target')
            ax.scatter(*pred,marker='x',s=100,c='yellow',label='fusion prediction')
            ax.set_title(f'{ids[idx]}: {title}');ax.set_aspect('equal')
        ax=axes[row,2];ax.imshow(aligned,alpha=.6)
        record=dict(id=ids[idx],tre_canvas=float(errors[idx]),fixed_query=q[idx].tolist(),
            expected_original=target[idx].tolist(),expected_aligned=target_aligned[idx].tolist(),
            predicted_aligned=prediction[idx].tolist(),local_matches={})
        for tag,color in [('SG','red'),('MA','blue')]:
            table=tables[tag]
            mq=np.asarray(table['source_points_unit']);mp=np.asarray(table['target_points_unit'])
            distance=512*np.linalg.norm(mq-q[idx],axis=1);local=np.flatnonzero(distance<=24)
            mapped=mp@a.T+b;eligible=((mapped>=0)&(mapped<=1)).all(1)
            local=local[eligible[local]]
            pos=mq[local]*512-.5;delta=(mp[local]-mq[local])*512
            ax.quiver(pos[:,0],pos[:,1],delta[:,0],delta[:,1],angles='xy',scale_units='xy',scale=1,
                color=color,alpha=.6,width=.005,label=f'{tag}: {len(local)} within 24 px')
            record['local_matches'][tag]=dict(count=int(len(local)),
                mean_proposed_delta_px=delta.mean(0).tolist() if len(local) else None,
                median_proposed_delta_px=np.median(delta,axis=0).tolist() if len(local) else None)
        ax.scatter(*source,marker='+',s=150,c='lime');ax.scatter(*truth,marker='*',s=120,c='cyan',edgecolors='black',linewidths=.5)
        ax.scatter(*pred,marker='x',s=100,c='yellow');ax.set_xlim(center[0]-radius,center[0]+radius)
        ax.set_ylim(center[1]+radius,center[1]-radius);ax.set_aspect('equal')
        ax.set_title(f'TRE {errors[idx]:.2f} original-canvas px');ax.legend(fontsize=7,loc='lower left')
        record['required_aligned_delta_px']=(truth-source).tolist()
        record['predicted_aligned_delta_px']=(pred-source).tolist();rows.append(record)
    axes[0,0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Post-hoc MIIT 7-to-8 tail context; all labels retained; no fitting or point filtering',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'miit_tail_context.png',dpi=140);plt.close(fig)
    result=dict(scope='label-informed read-only failure illustration; not evidence that annotation is wrong',
        interpolation='P1-ac source triangles; coordinates shown in original affine-aligned frame',
        evaluation_labels=107,selected='three largest completed-fusion errors; illustrative, not a new metric',rows=rows)
    (OUT/'miit_tail_context.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__': main()
