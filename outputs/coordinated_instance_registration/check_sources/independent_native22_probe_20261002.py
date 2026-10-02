"""Bounded native22 wrapper oracle; no registration and no real label access."""
import sys,json
from pathlib import Path
import numpy as np
root=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
sys.path[:0]=[str(root),str(root/'src')]
from tools.coordinated_dhr_existing_score import native_metrics,_aggregate
from tools.coordinated_dhr_existing_inputs import existing_rows

def sample(field,points):
    h,w=field.shape[1:];q=np.clip(points,[0,0],[w-1,h-1]);i=np.floor(q).astype(int);t=q-i
    j=np.minimum(i+1,[w-1,h-1]);out=[]
    for (x,y),(xx,yy),(u,v) in zip(i,j,t):
        out.append((1-u)*(1-v)*field[:,y,x]+u*(1-v)*field[:,y,xx]+(1-u)*v*field[:,yy,x]+u*v*field[:,yy,xx])
    return np.array(out)

results=[]
for sf,sm,r,pf,pm,hw in [(1.,1.,1.37,[0,6],[7,0],(75,92)),(.73,.61,1.19,[0,0],[12,2],(55,77))]:
    fixed_wh=(127,91);moving_wh=(113,103);h,w=hw
    y,x=np.mgrid[:h,:w];field=np.stack((.2*np.sin(x*.13)+.03*y,-.3*np.cos(y*.11)+.017*x)).astype(np.float32)
    params=dict(target_resample_ratio=sf,source_resample_ratio=sm,initial_resample_ratio=r,
        pad_2=[[pf[1],pf[1]],[pf[0],pf[0]]],pad_1=[[pm[1],pm[1]],[pm[0],pm[0]]])
    fixed=np.array([[21.17,22.31],[83.13,61.77],[47.63,39.21]])
    lattice=((fixed+.5)*sf+pf)/r-.5
    expected=((lattice+sample(field,lattice)+.5)*r-pm)/sm-.5
    target=expected+np.array([[.4,-.7],[-1.3,.9],[.8,1.7]])
    scale=np.array([2.3,3.1]);layout=dict(effective_original_to_canvas_scale_xy=scale.tolist(),padding_xy=[19,23])
    scored=native_metrics(field,params,fixed,target,fixed_wh,moving_wh,layout,['a','b','c'])
    native=np.linalg.norm(expected-target,axis=1);canvas=np.linalg.norm((expected-target)*scale,axis=1)
    en=max(abs(native[k]-scored['native_moving_pixels']['per_label'][label]) for k,label in enumerate(['a','b','c']))
    ec=max(abs(canvas[k]-scored['canvas_pixels']['per_label'][label]) for k,label in enumerate(['a','b','c']))
    assert en<3e-5 and ec<8e-5
    results.append(dict(source_ratio=sm,target_ratio=sf,initial_ratio=r,native_error=en,canvas_error=ec))

rows=existing_rows(Path('unopened_root'))
assert len(rows)==22 and len({r['name'] for r in rows})==22
assert len({r[k] for r in rows for k in ['fixed','moving']})==9
assert len(rows[:20])==20 and {r['name'] for r in rows[:20]}=={f'{a}_to_{b}' for a in ['he','cc10','cd31','ki67','prospc'] for b in ['he','cc10','cd31','ki67','prospc'] if a!=b}
assert rows[-2]['fixed'].endswith('Images_CD4.jpg') and rows[-2]['moving'].endswith('Images_CD68.jpg')
assert rows[-1]['fixed'].endswith('Rat-Kidney_HE.jpg') and rows[-1]['moving'].endswith('Rat-Kidney_PanCytokeratin.jpg')
fake=[dict(status='ok',metrics={unit:dict(mean=float(i),p90=float(i+1)) for unit in ['canvas_pixels','native_moving_pixels']}) for i in range(20)]
assert _aggregate(fake)['all_directions']['canvas_pixels']['mean_pair_mean']==9.5
fake[-1]=dict(status='failed')
assert _aggregate(fake)['all_directions'] is None and _aggregate(fake)['failed_directions']==1
print(json.dumps(dict(coordinates=results,exact22directions=True,nine_images=True,failed_denominator_retained=True,real_annotations_read=False)))
