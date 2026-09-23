"""Exact nested P1 refinement with image-conditioned safe fine-grid feedback."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...mesh import structured_rectangle
from ..tutte.dense_warp import StructuredDenseQueryTable
from .photometric_hint import local_photometric_logits
from .qc_radial_relaxation import SafeColoredQCRadialRelaxation
from .sine_spectral import spectralize_bounded_logits
from .face_qc import structured_p1_face_beltrami_modulus


class NestedP1PhotometricFeedbackLayer(nn.Module):
    """Map a coarse P1 homeomorphism to a certified finer original-grid P1 map.

    Inputs: float32 fixed/moving images (B,1,H,H) and coarse vertex map
    (B,C,C,2). The coarse boundary must be the identity square; every coarse
    face must have positive orientation. With fine_passes > 0, the coarse
    map must additionally have max facewise |mu| below qc_cap.

    Call prepare(device=...) after moving the module to that device. The
    output (B,F,F,2) is the exact P1 subdivision of the coarse map followed
    by zero or more topology/QC-preserving local image-feedback updates.
    Actual floating-point output is checked on all original fine faces.
    """

    def __init__(
        self, coarse_side: int, fine_side: int, *,
        fine_passes: int = 2, gain: float = 1.0,
        qc_cap: float = 0.795, window: int = 3, ridge: float = 1.0,
        spectral_modes: int = 16, floor_fraction: float = 0.8,
        checkpoint_refiner: bool = True,
        checkpoint_full_pass: bool = False,
    ) -> None:
        super().__init__()
        if (coarse_side < 3 or fine_side < coarse_side
                or (fine_side-1)%(coarse_side-1) != 0
                or fine_passes < 0 or gain <= 0 or not 0 < qc_cap < 1
                or window < 1 or window%2 != 1 or ridge <= 0
                or spectral_modes < 1):
            raise ValueError("invalid nested refinement configuration")
        self.coarse_side=coarse_side
        self.fine_side=fine_side
        self.fine_passes=fine_passes
        self.gain=float(gain)
        self.qc_cap=float(qc_cap)
        self.window=window
        self.ridge=float(ridge)
        self.spectral_modes=spectral_modes
        self.checkpoint_refiner=bool(checkpoint_refiner)
        self.checkpoint_full_pass=bool(checkpoint_full_pass)
        self.refiner=SafeColoredQCRadialRelaxation(
            fine_side,qc_cap=qc_cap,floor_fraction=floor_fraction)
        self._table=StructuredDenseQueryTable.from_mesh(
            structured_rectangle(coarse_side-1,coarse_side-1),
            height=fine_side,width=fine_side)
        self._prepared_device: torch.device | None = None
        self.last_stats: dict[str,float] = {}

    def prepare(self, *, device: torch.device | str) -> None:
        device=torch.device(device)
        if device.type=="cuda" and device.index is None:
            device=torch.device("cuda",torch.cuda.current_device())
        self._table.prepare(device=device,dtype=torch.float32)
        self._prepared_device=device

    @staticmethod
    def _minimum_area_ratio(mapped: torch.Tensor) -> float:
        a,b=mapped[:,:-1,:-1],mapped[:,:-1,1:]
        c,d=mapped[:,1:,1:],mapped[:,1:,:-1]
        def cross(first,second):
            return first[...,0]*second[...,1]-first[...,1]*second[...,0]
        minimum=torch.minimum(
            cross(b-a,c-a).amin(),cross(c-a,d-a).amin())
        return float(minimum*(mapped.shape[1]-1)**2)

    @staticmethod
    def _boundary_error(mapped: torch.Tensor) -> float:
        side=mapped.shape[1]
        line=torch.linspace(0,1,side,device=mapped.device,dtype=mapped.dtype)
        errors=(
            (mapped[:,0,:,0]-line).abs().amax(),
            mapped[:,0,:,1].abs().amax(),
            (mapped[:,-1,:,0]-line).abs().amax(),
            (mapped[:,-1,:,1]-1).abs().amax(),
            mapped[:,:,0,0].abs().amax(),
            (mapped[:,:,0,1]-line).abs().amax(),
            (mapped[:,:,-1,0]-1).abs().amax(),
            (mapped[:,:,-1,1]-line).abs().amax(),
        )
        return float(torch.stack(errors).amax())

    @torch.no_grad()
    def _certificate(self,mapped: torch.Tensor,*,requires_cap: bool) -> dict[str,float]:
        if not torch.isfinite(mapped).all():
            raise RuntimeError("nonfinite P1 map")
        boundary=self._boundary_error(mapped)
        area=self._minimum_area_ratio(mapped)
        if boundary>1e-6 or area<=0:
            raise RuntimeError(
                f"P1 topology precondition/certificate failed: boundary={boundary}, area={area}")
        maximum_mu=float(structured_p1_face_beltrami_modulus(mapped).amax())
        if requires_cap and not maximum_mu<self.qc_cap:
            raise RuntimeError(
                f"QC cap precondition/certificate failed: max_mu={maximum_mu}, cap={self.qc_cap}")
        return {"minimum_signed_area_ratio":area,
                "maximum_beltrami_modulus":maximum_mu,
                "maximum_boundary_error":boundary}

    def forward(
        self, fixed: torch.Tensor, moving: torch.Tensor,
        coarse_map: torch.Tensor,
    ) -> torch.Tensor:
        if self._prepared_device is None:
            raise RuntimeError("call prepare(device=...) before forward")
        if (fixed.shape!=moving.shape or fixed.ndim!=4 or fixed.shape[1]!=1
                or fixed.shape[-2]!=fixed.shape[-1]
                or coarse_map.shape!=(fixed.shape[0],self.coarse_side,self.coarse_side,2)
                or fixed.dtype!=torch.float32 or moving.dtype!=torch.float32
                or coarse_map.dtype!=torch.float32
                or fixed.device!=moving.device or fixed.device!=coarse_map.device
                or fixed.device!=self._prepared_device):
            raise ValueError("images/coarse_map must be same-device float32 with matching batch")
        coarse_stats=self._certificate(coarse_map,requires_cap=self.fine_passes>0)
        batch=coarse_map.shape[0]
        mapped=self._table.interpolate(coarse_map.reshape(batch,-1,2)).reshape(
            batch,self.fine_side,self.fine_side,2)
        area_floor=(self.refiner.compute_area_floor(mapped)
                    if self.fine_passes else None)
        def refine(current:torch.Tensor,proposal:torch.Tensor,
                   floor:torch.Tensor|None)->torch.Tensor:
            return self.refiner(current,proposal,area_floor=floor)
        def full_refine(current:torch.Tensor,fixed_image:torch.Tensor,
                        moving_image:torch.Tensor,
                        floor:torch.Tensor|None)->torch.Tensor:
            hint=local_photometric_logits(
                fixed_image,moving_image,current,window=self.window,
                ridge=self.ridge,raw_span=self.refiner.raw_span)
            hint=spectralize_bounded_logits(
                hint,side=self.fine_side,raw_span=self.refiner.raw_span,
                count=self.spectral_modes)
            return refine(current,self.gain*hint,floor)
        for _ in range(self.fine_passes):
            if self.checkpoint_full_pass and torch.is_grad_enabled():
                mapped=checkpoint(full_refine,mapped,fixed,moving,area_floor,
                                  use_reentrant=False)
            elif self.checkpoint_refiner and torch.is_grad_enabled():
                hint=local_photometric_logits(
                    fixed,moving,mapped,window=self.window,ridge=self.ridge,
                    raw_span=self.refiner.raw_span)
                hint=spectralize_bounded_logits(
                    hint,side=self.fine_side,raw_span=self.refiner.raw_span,
                    count=self.spectral_modes)
                mapped=checkpoint(refine,mapped,self.gain*hint,area_floor,
                                  use_reentrant=False)
            else:
                mapped=full_refine(mapped,fixed,moving,area_floor)
        fine_stats=self._certificate(mapped,requires_cap=self.fine_passes>0)
        self.last_stats={
            "coarse_minimum_signed_area_ratio":
                coarse_stats["minimum_signed_area_ratio"],
            "coarse_maximum_beltrami_modulus":
                coarse_stats["maximum_beltrami_modulus"],
            **fine_stats,
        }
        return mapped
