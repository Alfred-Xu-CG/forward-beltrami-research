"""Four-quarter MIND-transport image quadrature on ORIGINAL frozen fields.

Only IMAGE changes. Center-based OOB accounting and the existing complete
objective's priors/points remain intact. No feature recomputation or extra
raster information; this is an image quadrature ablation, not a new descriptor.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from qcopt.neural_bijection.dense.coordinated_fixed_sampling import FrozenP1Evaluator
from qcopt.neural_bijection.dense.coordinated_arap import p1_arap_energy
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_pixel_centers
from tools.coordinated_real_case import Evidence,corner_symmetric_dirichlet
from tools.digital_mind_safe_optimize import strain_penalty


def _normalize_evidence_device(requested,actual):
    """Resolve aliases to a concrete field device without moving any tensor.

    Unindexed CUDA means the CURRENT CUDA device, not any arbitrary GPU whose
    fields happen to exist. Explicit different indices remain mismatches. CPU
    indices are aliases of the same actual CPU storage device.
    """
    actual=torch.device(actual)
    if requested is None:return actual
    declared=torch.device(requested)
    if declared.type!=actual.type:
        raise ValueError("declared device must match actual evidence device")
    if declared.type=="cuda" and declared.index is None:
        declared=torch.device("cuda",torch.cuda.current_device())
    elif declared.type=="cpu":declared=torch.device("cpu")
    if declared!=actual:
        raise ValueError("declared device index must match actual evidence device")
    return actual


def pixel_quarter_queries(height:int,width:int,*,dtype=torch.float64,device="cpu"):
    """Four shared (1,H,W,2) source arrays, ordered row then column quarter."""
    if any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in (height,width)):
        raise ValueError("positive integer raster sizes required")
    if dtype not in (torch.float32,torch.float64):raise ValueError("float32/64 query precision required")
    output=[]
    for dy,dx in ((.25,.25),(.25,.75),(.75,.25),(.75,.75)):
        yy,xx=torch.meshgrid((torch.arange(height,dtype=dtype,device=device)+dy)/height,
                            (torch.arange(width,dtype=dtype,device=device)+dx)/width,indexing="ij")
        output.append(torch.stack((xx,yy),-1)[None])
    return tuple(output)


class FourQuarterMindEvidence:
    """Wrap existing Evidence with cached fixed-quarter values and source P1.

    Frozen base fields are SHARED, not re-extracted/affine-prewarped. Do not
    mutate the base fields/configuration after construction. Each quarter query
    inherits its ORIGINAL pixel mask; image denominator is 4*sum(mask), with no
    mask interpolation, query dropping or overlap-dependent denominator.
    Fixed AND moving descriptors use zero-padding bilinear align_corners=False.
    Fixed descriptor quarter samples are cached once; moving samples are fresh
    at all four mapped query arrays. Center map queries remain the original OOB
    term/fraction. __call__ never calls full base Evidence then subtracts image.

    Geometry dtype/device are explicit (default matrix dtype/device); image field
    precision stays exactly base precision. Full vertex gradients are ordinary
    AD through P1 and raster interpolation. No trainable fields/query-grad claim.
    Existing nested_priors, if present, is reused without any formula changes.
    Source query cache setup and resident bytes are reported separately; setup
    feature_extractions=0 since ORIGINAL descriptor computation belongs to base.
    """
    def __init__(self,base_evidence:Evidence,rows:int,columns:int|None=None,*,dtype=None,device=None):
        if not isinstance(base_evidence,Evidence):raise ValueError("existing Evidence instance required")
        base=base_evidence
        if base.loss!="mind" or base.mind_order!="transport" or base.interpolation not in ("p1_ac","p1_bd"):
            raise ValueError("quarter evidence requires MIND/transport/declared P1")
        if base.strain_model not in ("displacement_gradient","p1_arap"):
            raise ValueError("declared existing strain model required")
        columns=rows if columns is None else columns
        if any(isinstance(n,bool) or not isinstance(n,int) or n<2 for n in (rows,columns)):
            raise ValueError("integer source grid rows/columns >=2 required")
        if rows!=columns and base.strain_model=="displacement_gradient" and base.nested_priors is None:
            raise ValueError("existing displacement-gradient prior requires a square grid")
        dtype=base.matrix.dtype if dtype is None else dtype
        device=_normalize_evidence_device(device,base.matrix.device)
        if dtype not in (torch.float32,torch.float64):raise ValueError("float32/64 geometry precision required")
        for name in ("fixed","moving","fixed_feature","moving_feature","matrix","offset","mask","denominator"):
            value=getattr(base,name)
            if (not isinstance(value,torch.Tensor) or value.requires_grad
                    or value.dtype not in (torch.float32,torch.float64) or value.device!=device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("finite immutable float32/64 evidence fields on declared device required")
        if (base.matrix.shape!=(2,2) or base.offset.shape!=(2,) or base.matrix.dtype!=dtype
                or base.offset.dtype!=dtype or not bool(torch.linalg.det(base.matrix.double())>0)):
            raise ValueError("positive affine matching geometry precision required")
        if (base.fixed.ndim!=4 or base.fixed.shape[1]!=1 or base.moving.ndim!=4 or base.moving.shape[1]!=1
                or base.fixed.shape[0]<1 or base.fixed.shape[0]!=base.moving.shape[0]
                or base.fixed_feature.ndim!=4 or base.moving_feature.ndim!=4
                or base.fixed_feature.shape[0]!=base.fixed.shape[0]
                or base.moving_feature.shape[0]!=base.moving.shape[0]
                or base.fixed_feature.shape[-2:]!=base.fixed.shape[-2:]
                or base.moving_feature.shape[-2:]!=base.moving.shape[-2:]
                or base.fixed_feature.shape[1]<1 or base.fixed_feature.shape[1]!=base.moving_feature.shape[1]
                or base.fixed_feature.dtype!=base.moving_feature.dtype
                or base.mask.shape!=base.fixed.shape or not bool((base.mask>=0).all() and (base.mask<=1).all())
                or base.denominator.ndim!=0 or not bool(base.denominator>0)
                or not torch.equal(base.denominator,base.mask.sum())):
            raise ValueError("matching fixed image/descriptor/mask and fixed denominator required")
        if any(not math.isfinite(float(weight)) or weight<0 for weight in
               (base.strain_weight,base.shape_weight,base.oob_weight,base.match_weight)):
            raise ValueError("finite nonnegative objective weights required")
        if base.match_weight and base.matches is None:
            raise ValueError("positive point weight requires existing frozen point evidence")
        if base.matches is not None and any(buffer.requires_grad for buffer in base.matches.buffers()):
            raise ValueError("point evidence must remain constant")
        self.base,self.rows,self.columns,self.dtype,self.device=base,rows,columns,dtype,device
        sync=lambda:torch.cuda.synchronize(device) if device.type=="cuda" else None
        sync();start=time.perf_counter()
        height,width=base.fixed.shape[-2:]
        self.quarter_evaluators=[];self.fixed_quarter_features=[]
        for query in pixel_quarter_queries(height,width,dtype=dtype,device=device):
            self.quarter_evaluators.append(FrozenP1Evaluator(rows,columns,query,base.interpolation[-2:]))
            sampled=F.grid_sample(base.fixed_feature,(2*query.expand(base.fixed.shape[0],-1,-1,-1)-1).to(base.fixed_feature.dtype),
                                  mode="bilinear",padding_mode="zeros",align_corners=False)
            self.fixed_quarter_features.append(sampled)
        existing=base.fixed_p1_evaluator
        if existing is not None and not (existing.rows==rows and existing.columns==columns
                and existing.diagonal==base.interpolation[-2:]
                and existing.weights.dtype==dtype and existing.weights.device==device):
            raise ValueError("existing center sampler must match declared geometry")
        self.center_shared=existing is not None
        self.center_evaluator=existing  # Preserve the EXACT original center path, including None.
        sync();self.setup_seconds=time.perf_counter()-start
        self.metadata=dict(image_quadrature="four_quarters_original_pixel_mask",quarter_map_queries=4,
            center_map_queries=1,moving_descriptor_interpolations_per_call=4,
            fixed_descriptor_interpolations_setup=4,feature_extractions_setup=0,
            full_base_evidence_calls_per_call=0,center_cache_shared=self.center_shared,
            denominator_factor=4,mask_interpolated=False,queries_dropped=False,
            setup_seconds=self.setup_seconds,resident_constant_bytes=self.resident_constant_bytes)

    @property
    def resident_constant_bytes(self):
        """Additional quarter caches, unique storage.

        Excludes SHARED original Evidence fields/shared existing center sampler;
        this is not the whole-process resident or peak memory measurement.
        """
        values=list(self.fixed_quarter_features)
        evaluators=self.quarter_evaluators
        values.extend(buffer for evaluator in evaluators for buffer in evaluator.buffers())
        storages={value.untyped_storage().data_ptr():value.untyped_storage().nbytes() for value in values}
        return sum(storages.values())

    def __call__(self,vertices):
        base=self.base
        if (not isinstance(vertices,torch.Tensor) or vertices.ndim!=4
                or vertices.shape!=(base.fixed.shape[0],self.rows,self.columns,2)
                or vertices.dtype!=self.dtype or vertices.device!=self.device):
            raise ValueError("vertices must match declared batch/grid/geometry dtype/device")
        image_numerator=base.fixed_feature.new_zeros(())
        for evaluator,fixed in zip(self.quarter_evaluators,self.fixed_quarter_features):
            query=evaluator(vertices)@base.matrix.T+base.offset
            warped=F.grid_sample(base.moving_feature,(2*query-1).to(base.moving_feature.dtype),
                                 mode="bilinear",padding_mode="zeros",align_corners=False)
            errors=(fixed-warped).abs().mean(1,keepdim=True)
            image_numerator=image_numerator+(errors*base.mask).sum()
        image=image_numerator/(4*base.denominator)
        center=(self.center_evaluator(vertices) if self.center_evaluator is not None else
                p1_map_at_pixel_centers(vertices,*base.fixed.shape[-2:],diagonal=base.interpolation[-2:]))
        query=center@base.matrix.T+base.offset
        outside=(F.relu(-query)+F.relu(query-1)).square().sum(-1)[:,None]
        oob=(outside*base.mask).sum()/base.denominator
        if base.nested_priors is None:
            strain=(strain_penalty(vertices) if base.strain_model=="displacement_gradient" else
                    p1_arap_energy(vertices,diagonal=base.interpolation[-2:],validate=False))
            shape=corner_symmetric_dirichlet(vertices) if base.shape_weight else None
        else:
            strain,nested_shape=base.nested_priors(vertices)
            shape=nested_shape if base.shape_weight else None
        total=image+base.strain_weight*strain+base.oob_weight*oob
        if shape is not None:total=total+base.shape_weight*shape
        match=base.matches(vertices,base.matrix,base.interpolation) if base.match_weight else None
        if match is not None:total=total+base.match_weight*match
        outside_fraction=(((query<0)|(query>1)).any(-1)[:,None]*base.mask).sum()/base.denominator
        parts=dict(image=image,strain=strain,oob=oob,outside_fraction=outside_fraction)
        if shape is not None:parts["shape"]=shape
        if match is not None:parts["match"]=match
        return total,parts
