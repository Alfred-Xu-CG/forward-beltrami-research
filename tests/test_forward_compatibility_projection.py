import numpy as np
from scipy.optimize._numdiff import approx_derivative

from qcopt.beltrami import face_beltrami
from qcopt.forward.compatibility_projection import project_facewise_mu, projection_mu_jacobian
import qcopt.forward.compatibility_projection as projection_module
from qcopt.forward.benchmarks import smooth_twist_map
from qcopt.mesh import structured_rectangle


def test_projection_recovers_a_manufactured_realizable_face_field():
    mesh = structured_rectangle(4, 4)
    truth = smooth_twist_map(mesh.vertices, amplitude=0.03)
    target = face_beltrami(mesh, truth)
    result = project_facewise_mu(mesh, target, boundary_map=truth, max_nfev=100)
    assert result.success
    assert result.residual_norm < 1e-7
    np.testing.assert_allclose(result.map[mesh.boundary_loops[0]], truth[mesh.boundary_loops[0]])


def test_projection_exposes_incompatible_random_field_residual():
    mesh = structured_rectangle(3, 3)
    rng = np.random.default_rng(19)
    target = 0.4 * (rng.normal(size=mesh.n_faces) + 1j * rng.normal(size=mesh.n_faces))
    target *= np.minimum(1.0, 0.7 / np.maximum(np.abs(target), 1e-12))
    result = project_facewise_mu(mesh, target, max_nfev=80)
    assert np.isfinite(result.residual_norm)
    assert result.residual_norm > 1e-6


def test_projection_sparse_jacobian_matches_finite_difference(monkeypatch):
    mesh = structured_rectangle(3, 3)
    truth = smooth_twist_map(mesh.vertices, amplitude=0.03)
    target = face_beltrami(mesh, truth)
    original = projection_module.least_squares
    errors = []

    def wrapped(fun, x0, **kwargs):
        analytic = kwargs["jac"](x0).toarray()
        numeric = approx_derivative(fun, x0, method="3-point")
        errors.append(float(np.max(np.abs(analytic - numeric))))
        return original(fun, x0, **kwargs)

    monkeypatch.setattr(projection_module, "least_squares", wrapped)
    result = project_facewise_mu(
        mesh,
        target,
        boundary_map=truth,
        initial_map=mesh.vertices,
        max_nfev=20,
    )
    assert result.success
    assert errors and errors[0] < 1e-6


def test_exported_projection_jacobian_matches_face_mu_derivative():
    mesh = structured_rectangle(3, 3)
    truth = smooth_twist_map(mesh.vertices, amplitude=0.02)
    interior = np.asarray(sorted(set(range(mesh.n_vertices)) - set(mesh.boundary_loops[0].tolist())), dtype=np.int64)
    values = truth.copy()
    base = values[interior].reshape(-1)

    def residual(vector):
        trial = values.copy()
        trial[interior] = vector.reshape(-1, 2)
        mu = face_beltrami(mesh, trial)
        return np.concatenate((mu.real, mu.imag))

    analytic = projection_mu_jacobian(mesh, values, regularization=0.0, interior=interior).toarray()
    numeric = approx_derivative(residual, base, method="3-point")
    assert np.max(np.abs(analytic - numeric)) < 1e-6
