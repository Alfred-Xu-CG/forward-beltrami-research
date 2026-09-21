"""By-construction ordered rectangle boundary parameterization."""

from __future__ import annotations

from math import expm1, log

import numpy as np
import torch
import torch.nn.functional as torch_functional

from ...mesh import TriMesh


class StructuredRectangleBoundary(torch.nn.Module):
    """Map positive side-segment logits to one oriented rectangle boundary.

    The output follows ``mesh.boundary_loops[0]``.  Flat logits are attached
    to the correspondingly indexed oriented boundary edges.  Each side is
    normalized independently, so its segment lengths sum to the full side.
    Width is fixed to one; height is ``minimum_height + softplus(raw)``.
    """

    def __init__(self, mesh: TriMesh, minimum_height: float = 1.0e-3):
        super().__init__()
        if not np.isfinite(minimum_height) or minimum_height <= 0.0:
            raise ValueError("minimum_height must be finite and strictly positive")
        _validate_structured_rectangle(mesh)
        loop = mesh.boundary_loops[0]
        source = mesh.vertices[loop]
        corners_xy = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        corner_positions: list[int] = []
        for corner in corners_xy:
            matches = np.flatnonzero(np.all(np.isclose(source, corner, atol=1e-14), axis=1))
            if len(matches) != 1:
                raise ValueError("structured rectangle must contain each unit-square corner once")
            corner_positions.append(int(matches[0]))

        boundary_count = len(loop)
        for side in range(4):
            start = corner_positions[side]
            stop = corner_positions[(side + 1) % 4]
            positions = [start]
            while positions[-1] != stop:
                positions.append((positions[-1] + 1) % boundary_count)
                if len(positions) > boundary_count + 1:
                    raise ValueError("boundary loop does not traverse rectangle corners in order")
            values = source[np.asarray(positions)]
            coordinate = values[:, 0] if side in (0, 2) else values[:, 1]
            constant = values[:, 1] if side in (0, 2) else values[:, 0]
            expected_constant = (0.0, 1.0, 1.0, 0.0)[side]
            differences = np.diff(coordinate)
            direction_ok = np.all(differences > 0.0) if side in (0, 1) else np.all(differences < 0.0)
            if not np.allclose(constant, expected_constant, atol=1e-14, rtol=0.0) or not direction_ok:
                raise ValueError("boundary loop does not follow the four oriented rectangle sides")
            tensor = torch.tensor(positions, dtype=torch.int64)
            self.register_buffer(f"_side_positions_{side}", tensor, persistent=False)

        self.minimum_height = float(minimum_height)
        self.n_segments = boundary_count
        self.corner_loop_positions = tuple(corner_positions)

    @property
    def side_vertex_loop_positions(self) -> tuple[torch.Tensor, ...]:
        """Current-device registered side paths, including both end corners."""
        return tuple(getattr(self, f"_side_positions_{side}") for side in range(4))

    def raw_modulus_for_height(self, height: float) -> float:
        if not np.isfinite(height) or height <= self.minimum_height:
            raise ValueError("height must be finite and greater than minimum_height")
        difference = height - self.minimum_height
        # Stable inverse of softplus: x + log(1-exp(-x)).
        return difference + log(-expm1(-difference))

    def height(self, raw_modulus: torch.Tensor) -> torch.Tensor:
        return self.minimum_height + torch_functional.softplus(raw_modulus)

    def _realize_height(self, edge_logits: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        """Realize a rectangle at an already represented positive height."""
        if not isinstance(edge_logits, torch.Tensor) or not isinstance(height, torch.Tensor):
            raise TypeError("edge_logits and height must be torch tensors")
        if edge_logits.ndim < 1 or edge_logits.shape[-1] != self.n_segments:
            raise ValueError(f"edge_logits must have trailing shape ({self.n_segments},)")
        if edge_logits.dtype not in (torch.float32, torch.float64):
            raise ValueError("edge_logits dtype must be float32 or float64")
        if height.dtype != edge_logits.dtype:
            raise ValueError("edge_logits and height dtype must match")
        if height.device != edge_logits.device:
            raise ValueError("edge_logits and height device must match")
        if not bool(torch.isfinite(edge_logits).all()) or not bool(torch.isfinite(height).all()):
            raise ValueError("edge_logits and height must be finite")
        if bool(torch.any(height <= 0.0)):
            raise ValueError("realized rectangle height must be strictly positive")
        if self._side_positions_0.device != edge_logits.device:
            raise ValueError("move StructuredRectangleBoundary to the input device before use")

        try:
            batch_shape = torch.broadcast_shapes(edge_logits.shape[:-1], height.shape)
        except RuntimeError as error:
            raise ValueError("edge_logits and height batch shapes are not broadcastable") from error
        logits = edge_logits.expand(batch_shape + (self.n_segments,))
        realized_height = height.expand(batch_shape)
        output = torch.zeros(
            batch_shape + (self.n_segments, 2),
            dtype=edge_logits.dtype,
            device=edge_logits.device,
        )

        for side, positions in enumerate(self.side_vertex_loop_positions):
            edge_positions = positions[:-1]
            probabilities = torch.softmax(logits.index_select(-1, edge_positions), dim=-1)
            if not bool(torch.isfinite(probabilities).all()) or bool(torch.any(probabilities <= 0.0)):
                raise ValueError("rectangle side segment lengths must remain strictly positive")
            interior_cumulative = torch.cumsum(probabilities, dim=-1)[..., :-1]
            cumulative = torch.cat(
                (
                    torch.zeros(batch_shape + (1,), dtype=logits.dtype, device=logits.device),
                    interior_cumulative,
                    torch.ones(batch_shape + (1,), dtype=logits.dtype, device=logits.device),
                ),
                dim=-1,
            )
            if side == 0:
                coordinates = torch.stack((cumulative, torch.zeros_like(cumulative)), dim=-1)
            elif side == 1:
                coordinates = torch.stack(
                    (torch.ones_like(cumulative), realized_height[..., None] * cumulative), dim=-1
                )
            elif side == 2:
                coordinates = torch.stack(
                    (1.0 - cumulative, realized_height[..., None].expand_as(cumulative)), dim=-1
                )
            else:
                coordinates = torch.stack(
                    (torch.zeros_like(cumulative), realized_height[..., None] * (1.0 - cumulative)), dim=-1
                )
            ordered_coordinate = coordinates[..., 0] if side in (0, 2) else coordinates[..., 1]
            increments = torch.diff(ordered_coordinate, dim=-1)
            strictly_ordered = torch.all(increments > 0.0) if side in (0, 1) else torch.all(increments < 0.0)
            if not bool(strictly_ordered):
                raise ValueError("realized rectangle boundary vertices must remain strictly ordered")
            output = output.index_copy(-2, positions, coordinates)
        return output

    def at_height(self, edge_logits: torch.Tensor, height: float | torch.Tensor) -> torch.Tensor:
        """Realize a fixed represented height without an inverse-softplus round trip.

        This is used by unit-square compositions whose modulus is deliberately
        fixed rather than learned.  In particular, binary-exact ``1.0`` stays
        exactly one in both float32 and float64 across Torch implementations.
        """
        if not isinstance(edge_logits, torch.Tensor):
            raise TypeError("edge_logits must be a torch tensor")
        represented = height if isinstance(height, torch.Tensor) else edge_logits.new_tensor(height)
        return self._realize_height(edge_logits, represented)

    def forward(self, edge_logits: torch.Tensor, raw_modulus: torch.Tensor) -> torch.Tensor:
        if not isinstance(edge_logits, torch.Tensor) or not isinstance(raw_modulus, torch.Tensor):
            raise TypeError("edge_logits and raw_modulus must be torch tensors")
        if raw_modulus.dtype != edge_logits.dtype:
            raise ValueError("edge_logits and raw_modulus dtype must match")
        if raw_modulus.device != edge_logits.device:
            raise ValueError("edge_logits and raw_modulus device must match")
        if not bool(torch.isfinite(raw_modulus).all()):
            raise ValueError("raw_modulus must be finite")
        height = self.height(raw_modulus)
        if not bool(torch.isfinite(height).all()):
            raise ValueError("realized rectangle height must be finite")
        return self._realize_height(edge_logits, height)


def _validate_structured_rectangle(mesh: TriMesh) -> None:
    x_values = np.unique(mesh.vertices[:, 0])
    y_values = np.unique(mesh.vertices[:, 1])
    nx, ny = len(x_values) - 1, len(y_values) - 1
    if nx < 1 or ny < 1 or mesh.n_vertices != (nx + 1) * (ny + 1):
        raise ValueError("mesh is not a complete structured rectangle")
    expected_x = np.linspace(0.0, 1.0, nx + 1)
    expected_y = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(expected_x, expected_y, indexing="xy")
    expected_vertices = np.column_stack((xx.ravel(), yy.ravel()))
    if not np.allclose(mesh.vertices, expected_vertices, atol=1e-14, rtol=0.0):
        raise ValueError("mesh vertices do not match structured_rectangle ordering")
    expected_faces: list[tuple[int, int, int]] = []
    stride = nx + 1
    for row in range(ny):
        for col in range(nx):
            v00 = row * stride + col
            v10, v01 = v00 + 1, v00 + stride
            v11 = v01 + 1
            expected_faces.extend(((v00, v10, v11), (v00, v11, v01)))
    if not np.array_equal(mesh.faces, np.asarray(expected_faces, dtype=np.int64)):
        raise ValueError("mesh faces do not match structured_rectangle diagonals")
