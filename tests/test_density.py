import numpy as np
import torch

from qcopt.density import (
    area_loss,
    manufactured_area_target,
    normalize_area_factors,
    stress_area_target,
    torch_area_ratios,
)
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def test_normalization_preserves_total_rectangle_area():
    mesh = structured_rectangle(4, 3)
    raw = np.linspace(0.1, 3.0, mesh.n_faces)
    factors = normalize_area_factors(mesh, raw)
    assert np.all(factors > 0.0)
    assert np.isclose(np.sum(mesh.areas * factors), np.sum(mesh.areas))


def test_manufactured_target_matches_a_known_certified_bijection():
    mesh = structured_rectangle(6, 6)
    target = manufactured_area_target(mesh, amplitude=0.10)
    ratios = np.linalg.det(
        np.einsum(
            "fki,fkj->fij", target.known_map[mesh.faces], mesh.gradients
        )
    )
    assert target.feasibility_status == "manufactured_feasible"
    assert np.allclose(target.factors, ratios)
    assert audit_injectivity(mesh, target.known_map, rectangle=True).certified


def test_all_stress_targets_are_positive_normalized_and_unproven():
    mesh = structured_rectangle(5, 5)
    for kind in ("checkerboard", "spike", "random"):
        target = stress_area_target(mesh, kind, seed=17)
        assert target.feasibility_status == "sum_normalized_feasibility_unproven"
        assert np.all(target.factors > 0.0)
        assert np.isclose(np.sum(mesh.areas * target.factors), 1.0)


def test_torch_area_loss_is_zero_on_its_own_target_and_differentiable():
    mesh = structured_rectangle(3, 3)
    target = manufactured_area_target(mesh, amplitude=0.08)
    uv = torch.tensor(target.known_map, dtype=torch.double, requires_grad=True)
    factors = torch.tensor(target.factors, dtype=torch.double)
    ratios = torch_area_ratios(mesh, uv)
    loss = area_loss(ratios, factors)
    loss.backward()
    assert loss < 1e-20
    assert uv.grad is not None
    assert torch.all(torch.isfinite(uv.grad))
