"""Deterministic maps and images shared by all Phase V decoder routes.

The registration convention is explicit: a decoder returns a *backward*
coordinate map from fixed-image coordinates to moving-image coordinates.
Consequently ``grid_sample(moving, map)`` is the warped moving image.  No
inverse solve is hidden in the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi

import numpy as np
import torch
import torch.nn.functional as torch_functional

from ..injectivity import InjectivityReport, audit_injectivity
from ..mesh import TriMesh


ANALYTIC_DEFORMATION_NAMES = (
    "affine_anisotropic",
    "shear",
    "rotation_stretch",
    "twist",
    "sinusoidal_shear",
    "local_compression",
    "local_expansion",
    "boundary_sliding",
)

IMAGE_NAMES = (
    "checkerboard",
    "smooth_blobs",
    "medical_phantom",
    "textured",
)


@dataclass(frozen=True)
class AnalyticDeformationSample:
    name: str
    vertices: np.ndarray
    continuous_determinant_lower_bound: float
    injectivity_report: InjectivityReport


@dataclass(frozen=True)
class RegistrationPair:
    moving: torch.Tensor
    fixed: torch.Tensor
    backward_map: torch.Tensor
    convention: str = "backward_map_fixed_to_moving"


def _continuous_determinant_lower_bound(name: str) -> float:
    bounds = {
        "affine_anisotropic": 1.15 * 0.82,
        "shear": 1.0,
        "rotation_stretch": 1.15 * 0.85,
        "twist": 1.0,
        "sinusoidal_shear": 1.0,
        "local_compression": 0.75**2,
        # For q=1+a exp(-t), the radial singular value is
        # 1+a exp(-t)(1-2t), whose minimum for a>0 is at t=3/2.
        "local_expansion": 1.0 - 0.5 * exp(-1.5),
        "boundary_sliding": 1.0 - 0.12 * pi,
    }
    try:
        return bounds[name]
    except KeyError as error:
        raise ValueError(f"unknown analytic deformation: {name}") from error


def _deform_points(points: torch.Tensor, name: str) -> torch.Tensor:
    if points.shape[-1] != 2 or not torch.is_floating_point(points):
        raise ValueError("points must be a floating tensor with final dimension two")
    x, y = points.unbind(dim=-1)
    if name == "affine_anisotropic":
        return torch.stack((1.15 * x, 0.82 * y), dim=-1)
    if name == "shear":
        return torch.stack((x + 0.25 * y, y), dim=-1)
    if name == "rotation_stretch":
        centered = points - 0.5
        stretched_x = 1.15 * centered[..., 0]
        stretched_y = 0.85 * centered[..., 1]
        angle = torch.as_tensor(0.25, dtype=points.dtype, device=points.device)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotated = torch.stack(
            (
                cosine * stretched_x - sine * stretched_y,
                sine * stretched_x + cosine * stretched_y,
            ),
            dim=-1,
        )
        return rotated + 0.5
    if name == "twist":
        centered = points - 0.5
        squared_radius = centered.square().sum(dim=-1)
        angle = 0.60 * squared_radius
        cosine, sine = torch.cos(angle), torch.sin(angle)
        return 0.5 + torch.stack(
            (
                cosine * centered[..., 0] - sine * centered[..., 1],
                sine * centered[..., 0] + cosine * centered[..., 1],
            ),
            dim=-1,
        )
    if name == "sinusoidal_shear":
        return torch.stack((x + 0.08 * torch.sin(2.0 * pi * y), y), dim=-1)
    if name in ("local_compression", "local_expansion"):
        centered = points - 0.5
        squared_radius = centered.square().sum(dim=-1, keepdim=True)
        amplitude = -0.25 if name == "local_compression" else 0.25
        scale = 1.0 + amplitude * torch.exp(-squared_radius / 0.12)
        return 0.5 + scale * centered
    if name == "boundary_sliding":
        return torch.stack((x + 0.06 * torch.sin(2.0 * pi * x), y), dim=-1)
    raise ValueError(f"unknown analytic deformation: {name}")


def sample_analytic_deformation(mesh: TriMesh, name: str) -> AnalyticDeformationSample:
    """Sample a continuously orientation-preserving map and audit its P1 interpolant."""
    lower_bound = _continuous_determinant_lower_bound(name)
    source = torch.tensor(mesh.vertices.copy(), dtype=torch.float64)
    vertices = np.ascontiguousarray(_deform_points(source, name).numpy())
    vertices.setflags(write=False)
    report = audit_injectivity(mesh, vertices)
    if not report.certified:
        raise ValueError(
            f"sampled P1 map for {name!r} failed the independent injectivity audit"
        )
    return AnalyticDeformationSample(name, vertices, lower_bound, report)


def identity_query_grid(
    height: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if height < 2 or width < 2:
        raise ValueError("height and width must both be at least two")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("query grid dtype must be torch.float32 or torch.float64")
    y = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device)
    x = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def synthetic_image(
    name: str,
    *,
    height: int,
    width: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return one deterministic normalized image with shape ``(1,1,H,W)``."""
    grid = identity_query_grid(height, width, dtype=dtype, device=device)
    x, y = grid.unbind(dim=-1)
    if name == "checkerboard":
        values = torch.remainder(torch.floor(8.0 * x) + torch.floor(8.0 * y), 2.0)
    elif name == "smooth_blobs":
        values = (
            1.00 * torch.exp(-((x - 0.30) ** 2 + (y - 0.34) ** 2) / 0.018)
            + 0.75 * torch.exp(-((x - 0.70) ** 2 + (y - 0.62) ** 2) / 0.032)
            + 0.45 * torch.exp(-((x - 0.52) ** 2 + (y - 0.18) ** 2) / 0.010)
        )
    elif name == "medical_phantom":
        head_level = 1.0 - ((x - 0.50) / 0.39) ** 2 - ((y - 0.51) / 0.46) ** 2
        organ_a = 1.0 - ((x - 0.39) / 0.16) ** 2 - ((y - 0.53) / 0.25) ** 2
        organ_b = 1.0 - ((x - 0.64) / 0.12) ** 2 - ((y - 0.45) / 0.18) ** 2
        cavity = 1.0 - ((x - 0.51) / 0.08) ** 2 - ((y - 0.67) / 0.10) ** 2
        values = (
            0.22 * torch.sigmoid(55.0 * head_level)
            + 0.58 * torch.sigmoid(70.0 * organ_a)
            + 0.38 * torch.sigmoid(70.0 * organ_b)
            - 0.20 * torch.sigmoid(80.0 * cavity)
        )
    elif name == "textured":
        values = (
            0.40 * torch.sin(6.0 * pi * x + 0.3)
            + 0.27 * torch.cos(10.0 * pi * y - 0.5)
            + 0.18 * torch.sin(8.0 * pi * (x + 0.7 * y))
            + 0.15 * torch.cos(14.0 * pi * (0.4 * x - y))
            + 0.35 * torch.exp(-((x - 0.67) ** 2 + (y - 0.31) ** 2) / 0.020)
        )
    else:
        raise ValueError(f"unknown synthetic image: {name}")
    minimum, maximum = values.amin(), values.amax()
    if not bool(torch.isfinite(values).all()) or float(maximum - minimum) <= 0.0:
        raise ValueError("synthetic image construction produced invalid values")
    normalized = (values - minimum) / (maximum - minimum)
    return normalized[None, None]


def warp_image_backward(image: torch.Tensor, backward_map: torch.Tensor) -> torch.Tensor:
    """Sample ``image`` at a unit-square fixed-to-moving coordinate map."""
    if image.ndim != 4 or not torch.is_floating_point(image):
        raise ValueError("image must be a floating tensor with shape (B,C,H,W)")
    if backward_map.ndim == 3:
        backward_map = backward_map.unsqueeze(0)
    if backward_map.ndim != 4 or backward_map.shape[-1] != 2:
        raise ValueError("backward_map must have shape (H,W,2) or (B,H,W,2)")
    if image.device != backward_map.device or image.dtype != backward_map.dtype:
        raise ValueError("image and backward_map must share device and dtype")
    if not bool(torch.isfinite(image).all()) or not bool(torch.isfinite(backward_map).all()):
        raise ValueError("image and backward_map must contain only finite values")
    if backward_map.shape[0] == 1 and image.shape[0] != 1:
        backward_map = backward_map.expand(image.shape[0], -1, -1, -1)
    if backward_map.shape[0] != image.shape[0]:
        raise ValueError("image and backward_map batch dimensions are incompatible")
    normalized_grid = 2.0 * backward_map - 1.0
    return torch_functional.grid_sample(
        image,
        normalized_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def make_registration_pair(
    image_name: str,
    *,
    deformation: str,
    height: int,
    width: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> RegistrationPair:
    """Construct fixed image as the exact backward warp of a moving image."""
    moving = synthetic_image(
        image_name,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
    )
    identity = identity_query_grid(height, width, dtype=dtype, device=device)
    backward_map = _deform_points(identity, deformation)
    tolerance = 64.0 * torch.finfo(dtype).eps
    if bool(torch.any(backward_map < -tolerance)) or bool(
        torch.any(backward_map > 1.0 + tolerance)
    ):
        raise ValueError(
            "registration backward map leaves the sampled unit square; "
            "an explicit extended-domain or validity-mask protocol is required"
        )
    fixed = warp_image_backward(moving, backward_map)
    return RegistrationPair(moving, fixed, backward_map)
