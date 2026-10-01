"""Frozen image-generated point evidence, independent of feasibility decoding."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .coordinated_sampling import p1_map_at_queries


class ImageCorrespondences(torch.nn.Module):
    """q->p in fixed/affine-aligned coordinates; confidences are fixed weights.

    Error is measured AFTER the common affine, in original moving-canvas pixels.
    Robust scale is also in that pixel unit, not a coefficient or grid spacing.
    """
    def __init__(self,source,target,confidence,*,pixel_scale=512.,robust_scale=8.):
        super().__init__()
        if source.ndim!=2 or source.shape[-1]!=2 or target.shape!=source.shape or confidence.shape!=(len(source),):
            raise ValueError("finite Nx2 source/target and N confidences required")
        if source.dtype!=target.dtype or source.dtype!=confidence.dtype or source.device!=target.device or source.device!=confidence.device:
            raise ValueError("matching point/confidence dtype and device required")
        if not bool(torch.isfinite(source).all() and torch.isfinite(target).all() and torch.isfinite(confidence).all()):
            raise ValueError("nonfinite point evidence")
        if not bool((source>=0).all() and (source<=1).all() and (target>=0).all() and (target<=1).all()):
            raise ValueError("point coordinates must lie in the declared unit rectangle")
        if not bool((confidence>=0).all() and (confidence<=1).all()) or float(confidence.sum())<=0:
            raise ValueError("fixed nonnegative confidence with positive total required")
        if not math.isfinite(pixel_scale) or not math.isfinite(robust_scale) or min(pixel_scale,robust_scale)<=0:
            raise ValueError("positive finite pixel and robust scales required")
        self.register_buffer("source",source.detach().clone()[None])
        self.register_buffer("target",target.detach().clone()[None])
        self.register_buffer("weights",(confidence/confidence.sum()).detach().clone())
        self.pixel_scale,self.robust_scale=float(pixel_scale),float(robust_scale)

    def forward(self,vertices,matrix,interpolation="q1"):
        if vertices.ndim!=4 or vertices.shape[-1]!=2 or min(vertices.shape[1:3])<2 or vertices.shape[0]!=1:
            raise ValueError("point evidence currently declares a single image pair")
        if matrix.shape!=(2,2) or vertices.dtype!=self.source.dtype or vertices.device!=self.source.device or matrix.dtype!=vertices.dtype or matrix.device!=vertices.device:
            raise ValueError("matching geometry/affine/point dtype and device required")
        if interpolation=="q1":
            mapped=F.grid_sample(vertices.permute(0,3,1,2),2*self.source[:,:,None]-1,
                mode="bilinear",padding_mode="border",align_corners=True)[:,:,:,0].permute(0,2,1)
        elif interpolation in ("p1_ac","p1_bd"):
            mapped=p1_map_at_queries(vertices,self.source,interpolation[-2:],validate_queries=False)
        else:
            raise ValueError("declare q1, p1_ac or p1_bd point interpolation")
        error=(mapped-self.target)@matrix.T*(self.pixel_scale/self.robust_scale)
        # Equivalent to sqrt(1+r²)-1 but avoids cancellation at small r.
        squared=error.square().sum(-1)[0]
        penalty=squared/(torch.sqrt(1+squared)+1)
        return (penalty*self.weights).sum()
