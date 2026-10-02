"""One sum of frozen raster image terms, with fine-raster priors/OOB exactly once."""
from __future__ import annotations

import math

import torch


class SimultaneousImageEvidence:
    """E = sum_s w_s I_s + the finest Evidence's non-image terms.

    Each Evidence owns its immutable image/mask/source-query geometry. Only the
    finest object's full call computes priors and matches. No map is resampled,
    and gradients pass through every scalar image contribution. Optional weights
    support algebraic control tests; the application always uses equal weights.
    """

    def __init__(self, by_resolution, weights=None):
        if not by_resolution:
            raise ValueError("nonempty raster evidence pyramid required")
        self.by_resolution=dict(sorted(by_resolution.items()))
        self.finest=self.by_resolution[max(self.by_resolution)]
        keys=set(self.by_resolution)
        if weights is None:
            weights={side:1./len(keys) for side in keys}
        if set(weights)!=keys or any(isinstance(w,bool) or not isinstance(w,(float,int))
                or not math.isfinite(w) or w<0 for w in weights.values()):
            raise ValueError("one finite nonnegative weight per scale required")
        if not math.isclose(sum(weights.values()),1.,rel_tol=0.,abs_tol=1e-14):
            raise ValueError("image pyramid weights must sum to one")
        self.weights={side:float(weights[side]) for side in self.by_resolution}
        for side,item in self.by_resolution.items():
            if item.fixed.shape[-2:]!=(side,side):
                raise ValueError("raster key must equal square image resolution")
            if (item.loss!="mind" or item.mind_order!="transport"
                    or item.mind_frame!="shared_affine"
                    or item.interpolation!=self.finest.interpolation
                    or item.image_weight!=self.finest.image_weight
                    or not torch.equal(item.matrix,self.finest.matrix)
                    or not torch.equal(item.offset,self.finest.offset)):
                raise ValueError("common map/frame/affine and transport MIND image weights required")

    def __getattr__(self,name):
        return getattr(self.finest,name)

    def __call__(self,vertices):
        total,parts=self.finest(vertices)
        fine_side=max(self.by_resolution)
        # Exact fine-only control: no subtract/add rounding or extra gradients.
        if self.weights[fine_side]==1.:
            return total,parts
        images={fine_side:parts["image"]}
        for side,item in self.by_resolution.items():
            if side!=fine_side and self.weights[side]:
                images[side]=item.image_terms(vertices)[0]
        image=sum(self.weights[side]*images[side] for side in self.by_resolution
                  if self.weights[side])
        # Reconstruct from already-computed parts, rather than subtracting two
        # float32 image scalars from the float64 complete objective. The latter
        # adds avoidable cancellation rounding in mixed-precision runs.
        weighted=image if self.finest.image_weight==1. else self.finest.image_weight*image
        total=weighted+self.finest.strain_weight*parts["strain"]+self.finest.oob_weight*parts["oob"]
        if "shape" in parts:total=total+self.finest.shape_weight*parts["shape"]
        if "match" in parts:total=total+self.finest.match_weight*parts["match"]
        parts={**parts,"image":image,
               **{f"image_{side}":value for side,value in images.items()}}
        return total,parts
