"""Joint two-latent x-then-y instance stage on the SAME material vertex table.

Accepted anchor/reference are frozen constants. The y-step geometry depends on
the trainable x-step output and MUST stay connected to both proposal gradients.
This is not function composition followed by resampling, a new map family, or
a guarantee of better image/anatomical optimization than alternating stages.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .coordinated_update import CoordinatedQ1Update
from .coordinated_stage_cache import FrozenAnchorCoordinatedUpdate
from .coordinated_explicit_vjp import explicit_coordinated_candidate


@dataclass
class JointCoordinatedResult:
    vertices: torch.Tensor
    scales: torch.Tensor
    gauges: torch.Tensor
    normalized_margin_min: torch.Tensor
    substep_margin_min: torch.Tensor


class FrozenAnchorJointCoordinatedUpdate:
    """Two physical scalar proposals (B,R,C), jointly trainable, fixed boundary.

    ``ordinary`` uses two full ordinary AD layers. ``cached_manual`` caches ONLY
    the constant first anchor, then uses the FULL current-Y manual adjoint on
    the second step. It supports FIRST derivatives only. Its second NO-GRAD
    ordinary pass supplies exact scale/gauge/output-margin diagnostics because
    the existing explicit API exposes candidates only. That EXTRA computation
    is explicit and must count in timings; no speedup is asserted.

    Diagnostics are detached for BOTH backends; gradients of them are not this
    API's purpose. Candidate gradients include both raw proposals. Construct on
    the intended device/precision; no hidden conversion of caller references or
    trainable anchors, and no mutable global cache. Caller anchor/reference are
    copied; changed accepted geometry requires a NEW joint stage.

    ``substep_margin_min`` records ACTUAL rounded x/y candidate margins (B,2),
    even with ``validate=False``. Trusted callers must check BOTH substeps,
    not just the final ``normalized_margin_min`` (the identical y margin).
    """
    def __init__(self, anchor: torch.Tensor, *, reference: torch.Tensor | None = None,
                 mode: str = "analytic", boundary: str = "fixed",
                 minimum_jacobian: float = .001, theta: float = .95,
                 backend: str = "cached_manual"):
        if backend not in ("ordinary", "cached_manual"):
            raise ValueError("backend must be ordinary or cached_manual")
        if boundary != "fixed":
            raise ValueError("joint stage currently supports fixed boundary only")
        if (not isinstance(anchor, torch.Tensor) or anchor.ndim != 4 or
                anchor.shape[0] < 1 or anchor.shape[-1] != 2 or min(anchor.shape[1:3]) < 2 or
                anchor.dtype not in (torch.float32, torch.float64)):
            raise ValueError("float32/64 anchor (B,R>=2,C>=2,2) required")
        if anchor.requires_grad:
            raise ValueError("accepted anchor must be constant, not require gradients")
        if reference is not None:
            if (not isinstance(reference, torch.Tensor) or reference.requires_grad or
                    reference.ndim != 4 or reference.shape[1:] != anchor.shape[1:] or
                    reference.shape[0] not in (1, anchor.shape[0]) or
                    reference.dtype != anchor.dtype or reference.device != anchor.device):
                raise ValueError("constant reference must match anchor grid/dtype/device and batch1/B")
            reference = reference.double().clone()
        else:
            rows, columns = anchor.shape[1:3]
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, rows, dtype=torch.float64, device=anchor.device),
                torch.linspace(0, 1, columns, dtype=torch.float64, device=anchor.device), indexing="ij")
            reference = torch.stack((xx, yy), -1)[None]
        self._anchor = anchor.clone()
        self._reference = reference
        self.backend, self.mode = backend, mode
        self.minimum_jacobian, self.theta = minimum_jacobian, theta
        self.safe_substeps = 2
        self.extra_no_grad_diagnostic_passes = int(backend == "cached_manual")
        options = dict(mode=mode, boundary="fixed", minimum_jacobian=minimum_jacobian, theta=theta)
        self._x_layer = CoordinatedQ1Update((1., 0.), **options)
        self._y_layer = CoordinatedQ1Update((0., 1.), **options)
        if backend == "cached_manual":
            self._x_cache = FrozenAnchorCoordinatedUpdate(
                self._anchor, reference=self._reference, direction=(1., 0.), mode=mode,
                minimum_jacobian=minimum_jacobian, theta=theta)
        else:
            self._x_cache = None
            # Same upfront finite/reference/current-margin requirement as cache.
            with torch.no_grad():
                self._x_layer(self._anchor, torch.zeros_like(self._anchor[..., 0]),
                              reference=self._reference, validate=True)

    def __call__(self, proposal_x: torch.Tensor, proposal_y: torch.Tensor, *,
                 alpha_trial: float | torch.Tensor = 1., validate: bool = True) -> JointCoordinatedResult:
        for proposal in (proposal_x, proposal_y):
            if (not isinstance(proposal, torch.Tensor) or proposal.shape != self._anchor.shape[:-1] or
                    proposal.dtype != self._anchor.dtype or proposal.device != self._anchor.device):
                raise ValueError("both raw (B,R,C) proposals must match anchor dtype/device/grid")
        if isinstance(alpha_trial, torch.Tensor) and alpha_trial.requires_grad:
            raise ValueError("trial must be constant, not require gradients")
        trial = torch.as_tensor(alpha_trial, dtype=torch.float64, device=self._anchor.device)
        if (trial.ndim > 1 or (trial.ndim == 1 and trial.shape != (self._anchor.shape[0],)) or
                not bool(torch.isfinite(trial).all() and (trial >= 0).all())):
            raise ValueError("trial must be finite nonnegative scalar or shape(B,)")
        if self.backend == "ordinary":
            x_result = self._x_layer(self._anchor, proposal_x, reference=self._reference,
                                     alpha_trial=trial, validate=validate)
            y_result = self._y_layer(x_result.vertices, proposal_y, reference=self._reference,
                                     alpha_trial=trial, validate=validate)
            vertices = y_result.vertices
        else:
            x_result = self._x_cache(proposal_x, alpha_trial=trial, validate=validate)
            # NO DETACH: second geometry changes with proposal_x and its full Y
            # adjoint is essential for the joint two-latent derivative.
            vertices = explicit_coordinated_candidate(
                x_result.vertices, proposal_y, reference=self._reference, direction=(0., 1.),
                mode=self.mode, minimum_jacobian=self.minimum_jacobian, theta=self.theta,
                alpha_trial=trial, validate=validate, backward_backend="torch_manual")
            # No private grad_fn/saved-tensor inspection. This extra pass uses
            # the identical candidate arithmetic and ACTUAL rounded corners;
            # the default explicit pass independently checks returned vertices.
            with torch.no_grad():
                y_result = self._y_layer(x_result.vertices, proposal_y, reference=self._reference,
                                         alpha_trial=trial, validate=validate)
        return JointCoordinatedResult(
            vertices, torch.stack((x_result.scale.detach(), y_result.scale.detach()), 1),
            torch.stack((x_result.gauge.detach(), y_result.gauge.detach()), 1),
            y_result.normalized_margin_min.detach(),
            torch.stack((x_result.normalized_margin_min.detach(),
                         y_result.normalized_margin_min.detach()), 1))

    @property
    def resident_constant_bytes(self):
        tensors = (self._anchor, self._reference)
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
        return sum(storages.values()) + (self._x_cache.resident_constant_bytes if self._x_cache is not None else 0)
