"""One frozen-MA observation refresh through an absolute incumbent P1 map."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from tools.coordinated_matchanything import FrozenMatchAnything,read_matcher_gray,point_record,IMAGE_SIDE,SOURCE_REVISION
from tools.coordinated_registration_start import load_incumbent
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_pixel_centers,p1_map_at_queries


def render_incumbent(moving,vertices,matrix,offset):
    if moving.shape!=(1,1,512,512) or moving.dtype!=torch.float32 or vertices.dtype!=torch.float64:
        raise ValueError('original float32 grayscale512 and float64 fine P1 geometry required')
    residual=p1_map_at_pixel_centers(vertices,512,512,'ac').to(moving)
    # Preserve the ORIGINAL MA float32 affine operation order; one image sample.
    world=residual@matrix.to(moving).T+offset.to(moving)
    return F.grid_sample(moving,2*world-1,mode='bilinear',padding_mode='border',align_corners=False)


def refreshed_point_record(points0,points1,confidence,matrix,offset,vertices,*,fixed_support=None):
    # The old adapter implements ALL-raw validation followed by q/s unit filtering.
    record=point_record(points0,points1,confidence,matrix,offset,fixed_support=fixed_support)
    q=np.asarray(record['source_points_unit'],dtype=np.float64).reshape(-1,2)
    s=np.asarray(record['target_points_unit'],dtype=np.float64).reshape(-1,2)
    c=np.asarray(record['confidence'],dtype=np.float64)
    query=torch.as_tensor(s,device=vertices.device,dtype=torch.float64)[None]
    p=p1_map_at_queries(vertices,query,'ac')[0].detach().cpu().numpy()
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('transformed P1 target outside unit domain or nonfinite; no clipping/filter repair')
    a,b=np.asarray(matrix,dtype=np.float64),np.asarray(offset,dtype=np.float64)
    world=p@a.T+b;eligible=((world>=0)&(world<=1)).all(-1);positive=eligible&(c>0)
    bins=np.minimum(np.floor(q[positive]*4).astype(np.int64),3)
    counts=np.unique(p,axis=0,return_counts=True)[1]
    mass=float(c[eligible].sum())
    record.update(status='ok' if int(positive.sum())>=8 and np.isfinite(mass) and mass>0 else 'insufficient_matches',
        target_points_unit=p.tolist(),warped_target_points_unit=s.tolist(),
        eligible_matches=int(positive.sum()),static_outside_original_moving_matches=int((~eligible).sum()),
        eligible_confidence_mass=mass,occupied_quadrants_4x4=len(set(map(tuple,bins))),
        unique_target_coordinates=len(counts),maximum_target_multiplicity=int(counts.max()) if len(counts) else 0,
        target_conversion='float64 p=f_incumbent(s) in exact declared P1ac representation; no inverse/second affine',
        original_world_eligibility='A*p+b in unit square; never A*s+b',
        conditioning_rounds=1,incumbent_disagreement_filter_used=False)
    return record


class FrozenDeformationMatchAnything(FrozenMatchAnything):
    def extract(self,fixed,moving,affine,*,incumbent,output=None):
        if self.model is None:raise RuntimeError('matcher closed')
        fixed,moving,affine,incumbent=map(Path,(fixed,moving,affine,incumbent))
        output=None if output is None else Path(output)
        if output is not None and output.exists():raise FileExistsError(output)
        sync=lambda:torch.cuda.synchronize(self.device) if self.device.type=='cuda' else None
        sync()
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
        start=time.perf_counter()
        with np.load(affine,allow_pickle=False) as saved:
            a,b=saved['post_affine_matrix'].copy(),saved['post_affine_offset'].copy()
        vertices,incumbent_record=load_incumbent(incumbent,a,b,257,device=self.device)
        sync();incumbent_seconds=time.perf_counter()-start;tick=time.perf_counter()
        fixed_gray,moving_gray=read_matcher_gray(fixed),read_matcher_gray(moving)
        support=((1.-fixed_gray[0,0]).numpy()>.04)
        with torch.inference_mode():
            fixed_gray,moving_gray=fixed_gray.to(self.device),moving_gray.to(self.device)
            aligned=render_incumbent(moving_gray,vertices,torch.as_tensor(a,device=self.device),torch.as_tensor(b,device=self.device))
            sync();render_seconds=time.perf_counter()-tick;tick=time.perf_counter()
            batch={'image0':fixed_gray,'image1':aligned};self.model(batch)
            p0,p1,c=[batch[k].detach().cpu().numpy() for k in ('mkpts0_f','mkpts1_f','mconf')]
            sync();forward_seconds=time.perf_counter()-tick;tick=time.perf_counter()
            record=refreshed_point_record(p0,p1,c,a,b,vertices,fixed_support=support)
            sync();adaptation_seconds=time.perf_counter()-tick
        record.update(fixed=str(fixed),moving=str(moving),affine=str(affine),incumbent=incumbent_record,
            post_affine_matrix=a.tolist(),post_affine_offset=b.tolist(),image_side=IMAGE_SIDE,
            method='ONE deformation-conditioned extraction with the same frozen released MatchAnything ELoFTR',
            matcher_input='PIL L/255; float64 directP1 then float32 affine; original moving image sampled ONCE bilinear BORDER alignFalse',
            source_revision=SOURCE_REVISION,incumbent_loading_seconds=incumbent_seconds,
            raster_loading_render_seconds=render_seconds,model_forward_seconds=forward_seconds,point_adaptation_seconds=adaptation_seconds,
            extraction_seconds=time.perf_counter()-start,peak_allocated_bytes=self._peak(),
            extraction_time_scope='incumbent validation,rasterload/render,model andadaptation; setup/JSONwrite separate',
            targets_manual_landmarks_or_dense_teacher_loaded=False,affine_estimated_or_changed=False,
            self_confirmation_caution='incumbent-conditioned observations are not independent or anatomical truth')
        if output is not None:
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        return record
