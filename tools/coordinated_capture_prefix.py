"""Two fixed safe attempts for a discrete IMAGE proposal, not a new decoder.

Raw proposals are in pre-affine normalized units. No raw field is exported as
a map. Both attempts use the existing analytic operator and full objective.
This per-instance initializer does not differentiate through discrete argmin.
"""
from __future__ import annotations

import math
import torch

from qcopt.neural_bijection.dense.coordinated_update import (
    CoordinatedQ1Update, interpolate_proposal,
)
from qcopt.neural_bijection.dense.digital_q1 import (
    q1_corner_determinants, validate_q1_map,
)


@torch.no_grad()
def apply_capture_proposal(current, reference, proposal, evidence, *,
                           anchor_total, minimum_jacobian=.001):
    """Try x, then y, once each; reject rather than repair a failed candidate."""
    if (current.shape != reference.shape or current.ndim != 4
            or current.shape[0] != 1 or current.shape[-1] != 2
            or proposal.ndim != 4 or proposal.shape[0] != 1
            or proposal.shape[-1] != 2 or min(proposal.shape[1:3]) < 3
            or proposal.dtype != current.dtype or proposal.device != current.device
            or not bool(torch.isfinite(proposal).all())
            or not math.isfinite(anchor_total)):
        raise ValueError("one finite geometry-matched raw vector proposal required")
    if (bool(proposal[:, 0].any()) or bool(proposal[:, -1].any())
            or bool(proposal[:, :, 0].any()) or bool(proposal[:, :, -1].any())):
        raise ValueError("coarse proposal boundary must be exactly zero")
    qref = q1_corner_determinants(reference.double())
    ratio = q1_corner_determinants(current.double()) / qref
    if not bool((ratio > minimum_jacobian).all()) or not validate_q1_map(current, reference)["valid"]:
        raise ValueError("capture anchor must already satisfy strict actual topology")
    accepted = current.detach()
    total = float(anchor_total)
    record = dict(geometry_attempts=0, objective_evaluations=0,
                  accepted_axes=0, failed_attempts=0, attempts=[],
                  anchor_total=total, theta=.95, alpha_trial=1.,
                  order="x then y; same frozen raw proposal; refresh geometry after acceptance")
    for component, direction in enumerate(((1., 0.), (0., 1.))):
        event = dict(component=component, anchor_total=total, accepted=False)
        record["geometry_attempts"] += 1
        raw = interpolate_proposal(proposal[..., component], current.shape[1:3])
        layer = CoordinatedQ1Update(direction, mode="analytic", boundary="fixed",
                                    minimum_jacobian=minimum_jacobian, theta=.95)
        try:
            result = layer(accepted, raw, reference=reference, alpha_trial=1., validate=True)
            actual_ratio = float((q1_corner_determinants(result.vertices.double()) / qref).min())
            if actual_ratio <= minimum_jacobian or not validate_q1_map(result.vertices, reference)["valid"]:
                raise ValueError("rounded capture candidate violates actual topology")
            record["objective_evaluations"] += 1
            candidate_total, parts = evidence(result.vertices)
            candidate_total = float(candidate_total)
            if not math.isfinite(candidate_total) or any(not math.isfinite(float(value)) for value in parts.values()):
                raise ValueError("nonfinite capture objective or diagnostic")
            event.update(status="valid", candidate_total=candidate_total,
                         candidate_parts={key: float(value) for key, value in parts.items()},
                         scale=float(result.scale), gauge=float(result.gauge),
                         actual_minimum_corner_ratio=actual_ratio)
            if candidate_total < total:
                accepted, total = result.vertices.detach(), candidate_total
                event["accepted"] = True
                record["accepted_axes"] += 1
        except (ValueError, RuntimeError) as error:
            event.update(status="failed", error=f"{type(error).__name__}: {error}")
            record["failed_attempts"] += 1
        record["attempts"].append(event)
    record["accepted_total"] = total
    return accepted, record
