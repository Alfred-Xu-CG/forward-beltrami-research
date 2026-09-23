"""Image-conditioned spectral conductance layer with a small photometric solve.

This is a restricted known-frequency construction, not a generic Beltrami
solver. Positive edge weights and a fixed square boundary give an exact P1
homeomorphism; the finite-precision PCG implementation also face-checks.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ...mesh import structured_rectangle
from ..tutte.dense_warp import StructuredDenseQueryTable
from .sine_pcg_tutte import SinePreconditionedTutteLayer, _laplacian, _pcg


class PhotometricSpectralTutteLayer(torch.nn.Module):
    """Map an image pair to a dense P1 map through 18 (by default) edge modes.

    Call ``prepare(device=...)`` after ``to(device)`` and before forward.
    The boundary is the identity square. ``raw_mode_gains`` are the only
    trainable parameters; the latent itself is estimated from image data by a
    differentiable ridge-normal equation. The large graph solve has an
    implicit first-order VJP; no PCG trajectory is retained by autograd.
    """

    def __init__(
        self, side: int, *, fit_side: int = 128,
        frequencies: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        ridge: float = 0.005, coefficient_bound: float = 4.0,
        tolerance: float = 1e-10, response_chunk_size: int = 1,
    ) -> None:
        super().__init__()
        if (side < 3 or fit_side < 2 or ridge <= 0 or coefficient_bound <= 0
                or not frequencies or response_chunk_size < 1
                or any(k < 1 or k >= (side-1)//2 for k in frequencies)):
            raise ValueError("invalid side, fit resolution, modes, or ridge")
        self.side = side
        self.fit_side = fit_side
        self.frequencies = tuple(frequencies)
        self.ridge = float(ridge)
        self.coefficient_bound = float(coefficient_bound)
        self.response_chunk_size = int(response_chunk_size)
        self.solver = SinePreconditionedTutteLayer(
            side, minimum_conductance=1, maximum_conductance=16,
            tolerance=tolerance, max_iterations=120,
        )
        self.raw_mode_gains = torch.nn.Parameter(
            torch.zeros(3*len(frequencies), dtype=torch.float64))
        self.register_buffer("_horizontal_basis", torch.empty(0,dtype=torch.float64),
                             persistent=False)
        self.register_buffer("_vertical_basis", torch.empty(0,dtype=torch.float64),
                             persistent=False)
        self.register_buffer("_diagonal_basis", torch.empty(0,dtype=torch.float64),
                             persistent=False)
        self.register_buffer("_response_query", torch.empty(0),persistent=False)

    @property
    def mode_gains(self) -> torch.Tensor:
        return torch.exp(3*torch.tanh(self.raw_mode_gains))

    @torch.no_grad()
    def prepare(self, *, device: torch.device | str) -> dict[str, float | int]:
        device = torch.device(device)
        if self.solver._source.device != device or self.raw_mode_gains.device != device:
            raise ValueError("move the layer to the target device before prepare")
        side, modes = self.side, self.frequencies
        count = 3*len(modes)
        line = torch.linspace(0,1,side,device=device,dtype=torch.float64)
        half = 0.5*(line[:-1]+line[1:])
        yh,xh = torch.meshgrid(line,half,indexing="ij")
        yv,xv = torch.meshgrid(half,line,indexing="ij")
        yd,xd = torch.meshgrid(half,half,indexing="ij")
        horizontal = torch.zeros(count,side,side-1,device=device,dtype=torch.float64)
        vertical = torch.zeros(count,side-1,side,device=device,dtype=torch.float64)
        diagonal = torch.zeros(count,side-1,side-1,device=device,dtype=torch.float64)
        for index,frequency in enumerate(modes):
            omega = 2*math.pi*frequency
            horizontal[index] = torch.cos(omega*xh)*torch.sin(omega*yh)
            vertical[len(modes)+index] = torch.sin(omega*xv)*torch.cos(omega*yv)
            diagonal[2*len(modes)+index] = (
                torch.cos(omega*xd)*torch.sin(omega*yd)
                +torch.sin(omega*xd)*torch.cos(omega*yd))/math.sqrt(2)
        bases = (horizontal,vertical,diagonal)
        sigmoid = torch.sigmoid(torch.tensor(-0.8,device=device,dtype=torch.float64))
        constant = 1+15*sigmoid
        table = StructuredDenseQueryTable.from_mesh(
            structured_rectangle(side-1,side-1),
            height=self.fit_side,width=self.fit_side)
        table.prepare(device=device,dtype=torch.float32)
        query_parts = []
        iterations,residual = 0,0.0
        for start in range(0,count,self.response_chunk_size):
            stop = min(start+self.response_chunk_size,count)
            current = tuple(part[start:stop] for part in bases)
            derivative = tuple(15*sigmoid*(1-sigmoid)*part for part in current)
            conductances = tuple(torch.full_like(part,constant) for part in current)
            source = self.solver._source[None].expand(stop-start,-1,-1,-1)
            rhs = -_laplacian(source,*derivative)[:,1:-1,1:-1]
            response,steps,error = _pcg(
                rhs,conductances,self.solver._eigenvalues,
                tolerance=1e-10,max_iterations=160)
            full = F.pad(response.permute(0,3,1,2),(1,1,1,1)).permute(0,2,3,1)
            query_parts.append(table.interpolate(full.float().reshape(stop-start,-1,2)))
            iterations = max(iterations,steps)
            residual = max(residual,error)
            del derivative,conductances,source,rhs,response,full
        self._horizontal_basis = horizontal
        self._vertical_basis = vertical
        self._diagonal_basis = diagonal
        self._response_query = torch.cat(query_parts,dim=0)
        return {"response_iterations":iterations,
                "response_true_relative_residual":residual}

    def _latent(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        side = self.fit_side
        fixed_fit = F.interpolate(fixed,size=(side,side),mode="bilinear",
                                  align_corners=True)
        moving_fit = F.interpolate(moving,size=(side,side),mode="bilinear",
                                   align_corners=True)
        padded_x = F.pad(moving_fit,(1,1,0,0),mode="replicate")
        padded_y = F.pad(moving_fit,(0,0,1,1),mode="replicate")
        grad_x = 0.5*(side-1)*(padded_x[...,2:]-padded_x[...,:-2])
        grad_y = 0.5*(side-1)*(padded_y[...,2:,:]-padded_y[...,:-2,:])
        response = self._response_query
        design = (response[None,:,:,:,0]*grad_x[:,None,0]
                  +response[None,:,:,:,1]*grad_y[:,None,0]).flatten(2).double()
        residual = (fixed_fit-moving_fit).flatten(2).double()
        normal = design@design.transpose(1,2)/design.shape[-1]
        rhs = (design*residual).mean(dim=-1)
        scale = normal.diagonal(dim1=1,dim2=2).mean(dim=1).clamp_min(1e-12)
        identity = torch.eye(normal.shape[1],device=normal.device,dtype=normal.dtype)
        estimated = torch.linalg.solve(
            normal+(self.ridge*scale)[:,None,None]*identity,rhs[...,None])[...,0]
        bounded = self.coefficient_bound*torch.tanh(estimated/self.coefficient_bound)
        return bounded*self.mode_gains[None]

    def forward_with_latent(
        self, fixed: torch.Tensor, moving: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._response_query.numel()==0:
            raise RuntimeError("call prepare(device=...) before forward")
        if (fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1
                or fixed.shape[-2] != fixed.shape[-1] or fixed.dtype != torch.float32
                or moving.dtype != torch.float32 or fixed.device != self._response_query.device):
            raise ValueError("fixed/moving must be same-device float32 (B,1,H,H) images")
        latent = self._latent(fixed,moving)
        bases = (self._horizontal_basis,self._vertical_basis,self._diagonal_basis)
        logits = tuple(-0.8+torch.einsum("bk,k...->b...",latent,part)
                       for part in bases)
        mapped = self.solver(*logits).float()
        return mapped,latent

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        return self.forward_with_latent(fixed,moving)[0]
