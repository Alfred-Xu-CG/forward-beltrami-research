"""Read-only saved-map/evidence diagnostic; no inference, optimization or labels."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from tools.coordinated_real_case import Evidence, load_registration_evidence, load_image_matches, corner_symmetric_dirichlet
from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries

BASE = Path('outputs/coordinated_instance_registration')
RUN = BASE / 'matchanything_all25_t22'
torch.set_num_threads(2)

def local(path):
    text = str(path).replace('\\', '/')
    if 'results/' in text:
        return BASE / text.split('results/', 1)[1]
    root = Path('D:/QC_optimization_data/digital_topology_wsi')
    if '/data_all20_t12/canvas/' in text:
        return root/'lung_lesion3_eval/canvas'/Path(text).name
    if text.startswith('data/'):
        return root/'birl_anhir_dev/canvas'/Path(text).name
    raise ValueError(text)

def cosine(a, b):
    return float((a*b).sum() / (a.norm()*b.norm()+1e-30))

def table(path):
    r = json.loads(path.read_text())
    q,p,c = [np.asarray(r[k]) for k in ('source_points_unit','target_points_unit','confidence')]
    A,b = np.asarray(r['post_affine_matrix']),np.asarray(r['post_affine_offset'])
    valid = ((p@A.T+b>=0)&(p@A.T+b<=1)).all(1)&(c>0)
    q,p,c = q[valid],p[valid],c[valid]
    w = c/c.sum()
    distance,neighbor = cKDTree(q*512).query(q*512,k=2)
    near = distance[:,1]<=16
    motion = (p-q)@A.T*512
    coherence = np.linalg.norm(motion-motion[neighbor[:,1]],axis=1)
    bins = np.minimum((q*8).astype(int),7)
    mass = np.bincount(bins[:,0]+8*bins[:,1],weights=w,minlength=64)
    cell = np.minimum(np.floor(q*64).astype(int),63)
    _,inverse,counts = np.unique(cell,axis=0,return_inverse=True,return_counts=True)
    multiple = counts[inverse]>1
    diameters = []
    for index in np.where(counts>1)[0]:
        targets = p[inverse==index]*512
        diameters.append(float(np.linalg.norm(targets[:,None]-targets[None],axis=-1).max()))
    return q,p,w,A,dict(count=len(q),effective_count=float(1/(w*w).sum()),neighbor16_fraction=float(near.mean()),neighbor16_motion_difference=float(coherence[near].mean()),bin8_mass=mass.tolist(),initial_motion=float((w*np.linalg.norm(motion,axis=1)).sum()),exact_source_duplicates=len(q)-len(np.unique(q,axis=0)),source8px_cells=len(counts),multi_cell_mass=float(w[multiple].sum()),multi_cell_target_diameter_p90=float(np.quantile(diameters,.9)) if diameters else None,multi_cells_target_diameter_over16=int(sum(d>16 for d in diameters)))

def main():
    rows = json.loads((RUN/'predictions.json').read_text())['rows']
    results = []
    for row in rows:
        c = row['original_configuration']
        paths = {'sg':local(c['matches']), 'ma':RUN/row['match_table']}
        tables = {k:table(v) for k,v in paths.items()}
        A = tables['ma'][3]
        b = np.asarray(json.loads(paths['ma'].read_text())['post_affine_offset'])
        fixed,moving,mask,_ = load_registration_evidence(local(c['fixed']),local(c['moving']),512,preprocessing=c['preprocessing'])
        ev = Evidence(fixed,moving,torch.tensor(A),torch.tensor(b),'mind',3,1,shape_weight=.0001,fixed_mask=mask,interpolation='p1_ac',strain_model='p1_arap',mind_frame='shared_affine')
        ev.prepare_fixed_p1_sampling(257,257,dtype=torch.float64,device='cpu')
        modules = {k:load_image_matches(path,A,b,fixed_path=local(c['fixed']),moving_path=local(c['moving']),image_side=512,device='cpu',dtype=torch.float64,robust_scale=8)[0] for k,path in paths.items()}
        result = {'name':row['name'],'tables':{k:v[-1] for k,v in tables.items()}}
        for label,path in [('sgmap',local(c['output'])),('mamap',RUN/row['output'])]:
            Y = torch.tensor(np.load(path)['vertices'])
            delta = torch.zeros(1,2,33,33,dtype=Y.dtype,requires_grad=True)
            free = torch.zeros_like(delta)
            free[:,:,1:-1,1:-1] = 1
            V = Y+torch.nn.functional.interpolate(delta*free,size=(257,257),mode='bilinear',align_corners=True).permute(0,2,3,1)
            I,O,_ = ev.image_terms(V)
            S,M = [.1*modules[k](V,torch.tensor(A),'p1_ac') for k in ('sg','ma')]
            R = 3*p1_arap_energy(V,diagonal='ac',validate=False)
            H = .0001*corner_symmetric_dirichlet(V)
            gi,gs,gm,gr = [torch.autograd.grad(z,delta,retain_graph=True)[0] for z in (I,S,M,R)]
            result[label] = dict(image=float(I.detach()),sg_point=float(S.detach()),ma_point=float(M.detach()),arap=float(R.detach()),oob=float(O.detach()),normI=float(gi.norm()),normSG=float(gs.norm()),normMA=float(gm.norm()),normR=float(gr.norm()),cosSG_MA=cosine(gs,gm),cosI_SG=cosine(gi,gs),cosI_MA=cosine(gi,gm),cosR_SG=cosine(gr,gs),cosR_MA=cosine(gr,gm))
            result[label].update(shape=float(H.detach()),energySG=float((I+O+R+H+S).detach()),energyMA=float((I+O+R+H+M).detach()))
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='tables'}),flush=True)
    aggregates = {}
    for name,selection in [('miit',results[:3]),('lung',results[3:23]),('histo',results[23:24]),('kidney',results[24:])]:
        aggregates[name] = {stage:{k:float(np.mean([r[stage][k] for r in selection])) for k in selection[0][stage]} for stage in ['sgmap','mamap']}
    output = dict(question='Does MA alter correspondence preference or merely point mass/optimization?',labels_read=False,gradient_coordinates='Additive residual physical Y perturbations, fixed-zero boundary 33x33 bilinear prolongation to257, unit coordinates; no active inequality projection and not the production Adam/control gradient',point_terms_include_weight=.1,rows=results,aggregates=aggregates)
    destination = RUN/'cross_functional_diagnostic.json'
    destination.write_text(json.dumps(output,indent=2)+'\n')
    print('AGGREGATES',json.dumps(aggregates))

if __name__ == '__main__':
    main()
