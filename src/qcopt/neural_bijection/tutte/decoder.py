"""Composable rectangle-boundary Tutte decoder and fixed dense queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...mesh import TriMesh
from .boundary import StructuredRectangleBoundary
from .dense_warp import StructuredDenseQueryTable


@dataclass(frozen=True)
class TutteDecodeResult:
    boundary: torch.Tensor
    control: torch.Tensor
    dense: torch.Tensor


class TutteRectangleDecoder(torch.nn.Module):
    """Combine an ordered rectangle boundary, Tutte solve, and dense P1 warp.

    ``solver`` may be any fixed-mesh module with the Route I contract
    ``solver(latent, boundary) -> control_vertices`` and a ``system`` member.
    Dense query data are immutable but intentionally not module buffers;
    non-default device/dtype combinations must be prepared explicitly outside
    the timed forward path with :meth:`prepare`.
    """

    def __init__(
        self,
        mesh: TriMesh,
        solver: torch.nn.Module,
        *,
        image_height: int,
        image_width: int,
        minimum_height: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if not hasattr(solver, "system"):
            raise TypeError("solver must expose its fixed mesh through a system member")
        system = solver.system
        if (
            system.n_vertices != mesh.n_vertices
            or not np.array_equal(system.faces, mesh.faces)
            or not np.array_equal(system.source_vertices, mesh.vertices)
        ):
            raise ValueError("solver fixed mesh does not match decoder mesh")
        self.solver = solver
        self.boundary = StructuredRectangleBoundary(mesh, minimum_height=minimum_height)
        self.query_table = StructuredDenseQueryTable.from_mesh(
            mesh,
            height=image_height,
            width=image_width,
        )
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def prepare(self, *, device: torch.device | str, dtype: torch.dtype) -> None:
        self.query_table.prepare(device=device, dtype=dtype)

    def forward(
        self,
        latent: torch.Tensor,
        boundary_logits: torch.Tensor,
        raw_modulus: torch.Tensor,
    ) -> TutteDecodeResult:
        boundary = self.boundary(boundary_logits, raw_modulus)
        control = self.solver(latent, boundary)
        dense = self.query_table.interpolate(control)
        return TutteDecodeResult(boundary=boundary, control=control, dense=dense)
