import json,csv
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile

data=Path('D:/QC_optimization_data');root=Path('D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan')
miit=data/'miit_v4/extracted/test_data/test_data/source_data';run=root/'outputs/coordinated_instance_registration/miit_three_rotations_t153'
for m,f in [(2,3),(7,8),(10,11)]:
    name=f'miit_{m}_to_{f}';layout=json.loads((run/(name+'_layout.json')).read_text())
    for key,s in [('fixed',f),('moving',m)]:
        l=layout[key];fg=(1-np.asarray(Image.open(run/(name+'_'+key+'512.png')).convert('L'),dtype=float)/255)>.04
        mask=tifffile.imread(miit/str(s)/'masks/tissue_mask.tif').astype(np.float32)
        rendered=Image.new('F',(512,512),0.);rendered.paste(Image.fromarray(mask).resize(tuple(l['resized_wh']),Image.Resampling.BILINEAR),tuple(l['padding_xy']))
        tissue=np.array(rendered)
        print(json.dumps(dict(section=s,fixed=(key=='fixed'),eligible_pixels=int(fg.sum()),eligible_tissue_mass=float((fg*tissue).sum()),background_mass_fraction_of_eligible=float((fg*(1-tissue)).sum()/fg.sum()),excluded_tissue_fraction=float(((~fg)*tissue).sum()/tissue.sum()))))

canvas=data/'digital_topology_wsi/lung_lesion3_eval/canvas';annot=data/'digital_topology_wsi/lung_lesion3_eval/annotations50';affines=data/'digital_topology_wsi/ACROBAT_train_subset/scaleup102/imageonly_extension/lung_all20_hybrid_dynamic_strain02'
stains=dict(he='He',cc10='Cc10-5',cd31='CD31-3',ki67='Ki67-7',prospc='proSPC-4')
layouts={s:json.loads((canvas/('cc10_layout.json' if s=='he' else s+'_layout.json')).read_text())['fixed' if s=='he' else 'moving'] for s in stains}
def unit(p,l):return ((p+.5)*np.array(l['effective_original_to_canvas_scale_xy'])+l['padding_xy'])/512
landmarks={s:{r[' '].strip():(np.array([float(r['X']),float(r['Y'])])+.5)/10-.5 for r in csv.DictReader((annot/f'29-041-Izd2-w35-{v}-les3.csv').open())} for s,v in stains.items()}
all_lower=[];affected=0
for f in stains:
    for m in stains:
        if f==m:continue
        with np.load(affines/(f+'_'+ 'to_'+m+'_affine.npz')) as ar:A=ar['post_affine_matrix'].astype(float);b=ar['post_affine_offset'].astype(float)
        keys=list(landmarks[m]);t=unit(np.array(list(landmarks[m].values())),layouts[m]);r=(t-b)@np.linalg.inv(A).T
        # Initializer is a similarity: coordinate clipping is the exact Euclidean projection.
        assert np.max(np.abs(A.T@A-np.eye(2)*np.trace(A.T@A)/2))<1e-7
        lower=np.linalg.norm((r-np.clip(r,0,1))@A.T*512,axis=1);all_lower.extend(lower)
        if (lower>0).any():
            affected+=1
            print(json.dumps(dict(lung_direction=f+'_to_'+m,unattainable_labels={k:float(e) for k,e in zip(keys,lower) if e>0},lower_bound_mean_px=float(lower.mean()),maximum_lower_bound_px=float(lower.max()))))
print(json.dumps(dict(affected_lung_directions=affected,unattainable_total=int(np.count_nonzero(all_lower)),mean_error_lower_bound_all20=float(np.mean(all_lower)),maximum_error_lower_bound=float(np.max(all_lower)))))
