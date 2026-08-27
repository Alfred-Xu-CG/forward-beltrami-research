import numpy as np
import pytest
import torch

from qcopt.autograd import lbs_from_raw
from qcopt.constraints import rectangle_sliding_constraints
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle
from qcopt.optimize import (
    certified_interpolate_raw,
    run_direct_optimization,
    run_mu_optimization,
)


def test_certified_raw_interpolation_backtracks_to_a_bijective_map():
    mesh = structured_rectangle(2, 2)
    old = np.array([0.0])
    proposed = np.array([1.0])

    def map_from_raw(value):
        uv = mesh.vertices.copy()
        uv[4, 1] = 0.5 - 2.0 * float(value[0])
        return uv

    accepted, uv, backtracks = certified_interpolate_raw(
        old, proposed, map_from_raw, mesh, rectangle=True, max_backtracks=12
    )

    assert 0.0 < accepted[0] < 1.0
    assert backtracks > 0
    assert audit_injectivity(mesh, uv, rectangle=True).certified


def test_mu_optimizer_keeps_every_accepted_iterate_certified():
    mesh = structured_rectangle(2, 2)
    constraints = rectangle_sliding_constraints(mesh)
    raw = torch.zeros(mesh.n_faces, 2, dtype=torch.double)

    result = run_mu_optimization(
        mesh,
        raw,
        lambda value: lbs_from_raw(value, mesh, constraints, k_max=0.9),
        lambda uv: -(uv[4, 0] + uv[4, 1]),
        iterations=4,
        learning_rate=0.8,
        rectangle=True,
    )

    assert result.history
    assert all(item["certified"] for item in result.history)
    assert audit_injectivity(mesh, result.uv, rectangle=True).certified


def test_unprotected_direct_map_can_fold_but_safeguarded_variants_do_not():
    mesh = structured_rectangle(2, 2)
    objective = lambda uv: -(uv[4, 0] + uv[4, 1])
    unprotected = run_direct_optimization(
        mesh,
        mesh.vertices,
        objective,
        method="none",
        iterations=3,
        learning_rate=1.0,
        rectangle=True,
    )
    assert not audit_injectivity(mesh, unprotected.uv, rectangle=True).certified

    for method in ("lim_style", "slim_style", "amips_style"):
        protected = run_direct_optimization(
            mesh,
            mesh.vertices,
            objective,
            method=method,
            iterations=3,
            learning_rate=1.0,
            barrier_weight=0.05,
            rectangle=True,
        )
        assert audit_injectivity(mesh, protected.uv, rectangle=True).certified
        assert all(item["certified"] for item in protected.history)


def test_direct_optimizer_rejects_unknown_baseline_label():
    mesh = structured_rectangle(1, 1)
    with pytest.raises(ValueError, match="unknown method"):
        run_direct_optimization(
            mesh, mesh.vertices, lambda uv: uv.square().mean(), method="SLIM"
        )
