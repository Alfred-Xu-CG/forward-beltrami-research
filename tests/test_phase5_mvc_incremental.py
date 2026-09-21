"""Independent algebra and solver tests for Route-II incremental Tutte solves."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as sparse_linalg
from qcopt.neural_bijection.tutte import incremental as incremental_module

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.direct import DirectTutteLayer
from qcopt.neural_bijection.tutte.incremental import (
    assemble_directed_system,
    bicgstab_two_rhs,
    correction_right_hand_side,
    woodbury_row_update,
)


def _case(side: int = 6):
    mesh = structured_rectangle(side - 1, side - 1)
    layer = DirectTutteLayer(mesh)
    system = layer.system
    rng = np.random.default_rng(250922)
    old_logits = 0.12 * rng.standard_normal((system.n_rows, system.max_degree))
    boundary = mesh.vertices[system.loop].copy()
    old_a, old_b, old_p = assemble_directed_system(system, old_logits, boundary)
    old_y = np.linalg.solve(old_a.toarray(), old_b)
    return system, rng, old_logits, boundary, old_a, old_b, old_p, old_y


def test_correction_equation_is_an_algebraic_identity_for_global_update() -> None:
    system, rng, logits, boundary, old_a, old_b, _, old_y = _case()
    new_logits = logits + 0.03 * rng.standard_normal(logits.shape)
    new_boundary = boundary.copy()
    new_boundary[:, 0] += 0.01 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(boundary), endpoint=False))
    new_a, new_b, _ = assemble_directed_system(system, new_logits, new_boundary)
    rhs = correction_right_hand_side(old_a, old_b, old_y, new_a, new_b)
    expected = new_b - new_a @ old_y
    np.testing.assert_allclose(rhs, expected, atol=2e-15, rtol=2e-15)

    new_y = np.linalg.solve(new_a.toarray(), new_b)
    np.testing.assert_allclose(new_a @ (new_y - old_y), rhs, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("global_update", [False, True])
def test_cold_warm_and_correction_solve_the_identical_new_system(global_update: bool) -> None:
    system, rng, logits, boundary, old_a, old_b, _, old_y = _case(8)
    new_logits = logits.copy()
    if global_update:
        new_logits += 2.0e-3 * rng.standard_normal(new_logits.shape)
    else:
        new_logits[5] += 2.0e-3 * rng.standard_normal(new_logits.shape[1])
    new_a, new_b, _ = assemble_directed_system(system, new_logits, boundary)
    direct = np.linalg.solve(new_a.toarray(), new_b)
    atol = 1.0e-11 * max(1.0, float(np.linalg.norm(new_b)))

    cold = bicgstab_two_rhs(new_a, new_b, atol=atol)
    warm = bicgstab_two_rhs(new_a, new_b, x0=old_y, atol=atol)
    correction_rhs = correction_right_hand_side(old_a, old_b, old_y, new_a, new_b)
    correction = bicgstab_two_rhs(new_a, correction_rhs, atol=atol)
    corrected = old_y + correction.solution

    for candidate in (cold.solution, warm.solution, corrected):
        np.testing.assert_allclose(candidate, direct, atol=2e-10, rtol=2e-10)
        # The public contract is applied independently to each coordinate RHS;
        # do not accidentally compare the two-column Frobenius norm to one
        # column's tolerance.
        residual_columns = np.linalg.norm(new_b - new_a @ candidate, axis=0)
        assert np.all(residual_columns <= 1.05 * atol)
    # With the same zero initial correction, warm start and the exact
    # correction equation are the same Krylov problem in shifted coordinates.
    assert warm.iterations == correction.iterations
    np.testing.assert_allclose(warm.solution, corrected, atol=2e-12, rtol=2e-12)


def test_local_row_woodbury_update_matches_refactorization() -> None:
    system, rng, logits, boundary, old_a, _, _, _ = _case(7)
    changed = np.array([2, 9], dtype=np.int64)
    new_logits = logits.copy()
    new_logits[changed] += 0.08 * rng.standard_normal(new_logits[changed].shape)
    new_boundary = boundary.copy()
    new_boundary[:, 1] *= 0.97
    new_a, new_b, _ = assemble_directed_system(system, new_logits, new_boundary)

    updated = woodbury_row_update(old_a, new_a, new_b, changed)
    expected = np.linalg.solve(new_a.toarray(), new_b)
    np.testing.assert_allclose(updated.solution, expected, atol=2e-13, rtol=2e-13)
    assert updated.updated_rows == len(changed)
    assert updated.schur_condition >= 1.0
    assert np.linalg.norm(new_b - new_a @ updated.solution) < 1e-12


def test_woodbury_can_reuse_an_existing_old_factor(monkeypatch) -> None:
    system, rng, logits, boundary, old_a, _, _, _ = _case(7)
    new_logits = logits.copy()
    new_logits[4] += 0.04 * rng.standard_normal(new_logits.shape[1])
    new_a, new_b, _ = assemble_directed_system(system, new_logits, boundary)
    existing = sparse_linalg.splu(old_a.tocsc())

    def unexpected(*_args, **_kwargs):
        pytest.fail("a supplied old factor must be reused")

    monkeypatch.setattr(sparse_linalg, "splu", unexpected)
    updated = woodbury_row_update(old_a, new_a, new_b, np.array([4]), old_factor=existing)
    np.testing.assert_allclose(updated.solution, np.linalg.solve(new_a.toarray(), new_b), atol=2e-13)


def test_woodbury_rejects_an_incomplete_changed_row_set() -> None:
    system, rng, logits, boundary, old_a, _, _, _ = _case()
    new_logits = logits.copy()
    new_logits[[1, 3]] += 0.05 * rng.standard_normal(new_logits[[1, 3]].shape)
    new_a, new_b, _ = assemble_directed_system(system, new_logits, boundary)
    with pytest.raises(ValueError, match="outside declared rows"):
        woodbury_row_update(old_a, new_a, new_b, np.array([1]))


def test_invalid_masked_or_nonfinite_logits_fail_closed() -> None:
    system, _, logits, boundary, *_ = _case()
    logits = logits.copy()
    logits[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        assemble_directed_system(system, logits, boundary)


def test_bicgstab_old_scipy_tol_signature_keeps_absolute_contract(monkeypatch) -> None:
    """SciPy before the rtol rename must receive tol=0, not a looser default."""

    matrix = np.array([[2.0, -0.2], [-0.1, 1.5]])
    rhs = np.array([[1.0, 0.4], [0.2, 1.3]])
    observed = []

    def old_signature(a, b, x0=None, tol=1e-5, maxiter=None, callback=None, atol=None):
        observed.append((tol, atol, maxiter, x0 is None))
        solution = np.linalg.solve(a.toarray(), b)
        if callback is not None:
            callback(solution)
        return solution, 0

    monkeypatch.setattr(incremental_module.sparse_linalg, "bicgstab", old_signature)
    result = bicgstab_two_rhs(
        incremental_module.sparse.csr_matrix(matrix), rhs, atol=3.0e-12
    )
    np.testing.assert_allclose(result.solution, np.linalg.solve(matrix, rhs), atol=1e-15)
    assert observed == [(0.0, 3.0e-12, 20, True), (0.0, 3.0e-12, 20, True)]
