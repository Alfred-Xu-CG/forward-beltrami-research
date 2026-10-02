"""One frozen affine intensity prewarp and nominal support metadata at setup.

No mask repair, query dropping, optimizer, or trainable-affine derivative API.
"""
import torch
import torch.nn.functional as F
import numpy as np

from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers


def prepare_shared_affine_intensity(moving,matrix,offset,mask):
    if (moving.ndim!=4 or moving.shape[1]!=1 or moving.dtype not in (torch.float32,torch.float64)
            or mask.shape!=moving.shape or matrix.shape!=(2,2) or offset.shape!=(2,)
            or matrix.device!=moving.device or offset.device!=moving.device
            or any(t.requires_grad for t in (moving,matrix,offset,mask))):
        raise ValueError("frozen B1HW intensity/mask and frozen same-device2x2 affine required")
    if (not all(bool(torch.isfinite(t).all()) for t in (moving,matrix,offset,mask))
            or not bool(torch.linalg.det(matrix.double())>0)):
        raise ValueError("finite frozen inputs and positive affine required")
    height,width=moving.shape[-2:]
    query=fixed_pixel_centers(height,width,dtype=torch.float64,device=moving.device)
    original=query@matrix.double().T+offset.double()
    warped=F.grid_sample(moving,(2*original-1).to(moving.dtype).expand(moving.shape[0],-1,-1,-1),
        mode="bilinear",padding_mode="zeros",align_corners=False)
    # Only setup-time CPU geometry. Never recomputed at a trial or used as a mask.
    q=query.reshape(-1,2).cpu().numpy()
    a=matrix.double().cpu().numpy();b=offset.double().cpu().numpy()
    wh=np.asarray([width,height]);lo=np.floor(q*wh-.5)-3;hi=np.ceil(q*wh-.5)+3
    fixed=(lo>=0).all(-1)&(hi<=wh-1).all(-1)
    corners=np.stack((lo,np.stack((hi[:,0],lo[:,1]),-1),hi,np.stack((lo[:,0],hi[:,1]),-1)),1)
    indices=(((corners+.5)/wh)@a.T+b)*wh-.5
    aligned=fixed&(np.floor(indices)>=0).all((1,2))&(np.ceil(indices)<=wh-1).all((1,2))
    raw=q@a.T+b
    raw_lo=np.floor(raw*wh-.5)-3;raw_hi=np.ceil(raw*wh-.5)+3
    raw_support=fixed&(raw_lo>=0).all(-1)&(raw_hi<=wh-1).all(-1)
    weights=mask.cpu().numpy().reshape(moving.shape[0],-1).astype(np.float64)
    denominator=weights.sum()
    if denominator<=0:raise ValueError("nonempty fixed mask required")
    def summarize(flags):return dict(full_domain_fraction=float(flags.mean()),
        fixed_mask_weighted_fraction=float((weights*flags[None]).sum()/denominator))
    metadata=dict(raster_hw=[height,width],intensity_prewarp_count=1,descriptor_construction_count=1,
        coordinate_dtype="torch.float64",normalized_sampling_grid_dtype=str(moving.dtype),
        sampling="bilinear zeros align_corners=False; original moving intensity prewarped ONCE at this scale",
        fixed_descriptor_support=summarize(fixed),shared_affine_descriptor_support=summarize(aligned),
        original_frame_descriptor_support=summarize(raw_support),fixed_mask_denominator=float(denominator),
        support_scope="nominal float64 offset2+pool1+query-bilinear footprint; four affine-mapped extreme pixel centers with original moving bilinear floor/ceil inside raster; NOT an exact float32 access certificate",
        support_query_scope="STATIC setup fractions at undeformed residual/identity raster pixel centers, NOT current f(q) support or a guarantee for deformed foreground samples",
        boundary_difference="outside-original prewarp intensity is zero; its self-similarity descriptors may be ones, unlike zero padding of original descriptor fields",
        mask_policy="ALL original fixed-mask weights retained; support fractions are reporting ONLY")
    return warped,metadata
