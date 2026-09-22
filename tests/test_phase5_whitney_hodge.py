"""Whitney identities checked independently of the implementation's assembly."""
import importlib.util

import numpy as np
import pytest
import torch


def system(vertices, faces, **kwargs):
    assert importlib.util.find_spec("qcopt.forward.whitney_hodge") is not None
    from qcopt.forward.whitney_hodge import WhitneyHodge
    return WhitneyHodge(np.asarray(vertices, float), np.asarray(faces, int), **kwargs)


def grid(n=4):
    p = np.array([(x, y) for y in np.linspace(0, 1, n) for x in np.linspace(0, 1, n)])
    f = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j*n+i
            f.extend(((a, a+1, a+n+1), (a, a+n+1, a+n)))
    b = np.flatnonzero(np.any((p == 0) | (p == 1), axis=1))
    return p, np.asarray(f), b


@pytest.mark.parametrize("faces", [[[0, 1, 2]], [[0, 1, 2], [0, 2, 3]]])
def test_energy_and_spd_against_independent_affine_polynomials(faces):
    p = np.array([[0., 0.], [1.2, .1], [1., 1.], [-.2, .8]])[:max(map(max, faces))+1]
    s = system(p, faces)
    a = torch.tensor([[[2., .4], [.4, 1.]]]*len(faces), dtype=torch.double)
    h = s.hodge(a).to_dense()
    b0 = s.b0.to_dense()
    assert torch.linalg.eigvalsh(h).min() > 0
    assert torch.max(torch.abs(s.b1.to_dense() @ b0)) == 0
    k = b0.T @ h @ b0
    u = torch.arange(len(p), dtype=torch.double).sin()
    independent = 0.
    for t, face in enumerate(faces):
        xy = p[face]
        coefficients = np.linalg.solve(np.column_stack((np.ones(3), xy)), u.numpy()[face])
        area = abs(np.linalg.det(np.column_stack((xy[1]-xy[0], xy[2]-xy[0]))))/2
        independent += area * coefficients[1:] @ a[t].numpy() @ coefficients[1:]
    assert float(u @ k @ u) == pytest.approx(independent, abs=1e-12)
    assert torch.linalg.eigvalsh(k)[1] > 0
    assert torch.max(torch.abs(k @ torch.ones(len(p), dtype=torch.double))) < 1e-12
    torch.testing.assert_close(s.apply(a, u), k @ u)


def test_face_permutation_and_global_edge_reversal():
    p, f, _ = grid()
    s = system(p, f)
    rev = system(p, f[:, [2, 1, 0]], edge_signs=-np.ones(len(s.edges)))
    a = torch.tensor([[[2., .4], [.4, 1.]]]*len(f), dtype=torch.double)
    torch.testing.assert_close(s.stiffness(a), rev.stiffness(a))
    torch.testing.assert_close(s.hodge(a).to_dense(), rev.hodge(a).to_dense())


def test_individual_edge_reversal_is_hodge_congruence():
    p, f, _ = grid(3)
    s = system(p, f)
    signs = np.where(np.arange(len(s.edges)) % 2, -1., 1.)
    r = system(p, f, edge_signs=signs)
    a = torch.eye(2, dtype=torch.double)[None].repeat(len(f), 1, 1)
    d = torch.tensor(signs)
    torch.testing.assert_close(r.hodge(a).to_dense(), d[:, None]*s.hodge(a).to_dense()*d[None, :])
    torch.testing.assert_close(r.stiffness(a), s.stiffness(a))


def test_dual_stream_affine_centroid_values_with_correct_sign():
    p, f, b = grid(4)
    s = system(p, f)
    a = torch.eye(2, dtype=torch.double)[None].repeat(len(f), 1, 1)
    u = torch.tensor(p[:, 0])
    stream, residual = s.dual_stream(s.flux_cochain(a, u))
    exact = p[f].mean(axis=1)[:, 1]
    np.testing.assert_allclose(stream, exact-exact[0], atol=1e-11)
    assert residual < 1e-11


def test_solve_dirichlet_spd_dual_stream_and_affine_reproduction():
    p, f, b = grid(5)
    s = system(p, f)
    a = torch.tensor([[[2., .4], [.4, 1.]]]*len(f), dtype=torch.double)
    boundary = torch.tensor((2*p[b, 0] - .7*p[b, 1]), dtype=torch.double)
    u = s.solve(a, b, boundary)
    torch.testing.assert_close(u, torch.tensor(2*p[:, 0]-.7*p[:, 1]))
    interior = np.setdiff1d(np.arange(len(p)), b)
    k = s.stiffness(a)
    assert torch.linalg.eigvalsh(k[interior][:, interior]).min() > 0
    q = s.flux_cochain(a, u)
    v, residual = s.dual_stream(q)
    assert residual < 1e-12
    assert v.shape == (len(f),)
    assert torch.max(torch.abs(s.apply(a, u)[interior])) < 1e-12


def test_implicit_tensor_and_boundary_vjp_matches_fd_and_dense_autograd():
    p, f, b = grid(4)
    s = system(p, f)
    torch.manual_seed(5)
    a = (torch.eye(2, dtype=torch.double)[None].repeat(len(f), 1, 1)*2).requires_grad_()
    boundary = torch.randn(len(b), 2, dtype=torch.double, requires_grad=True)
    weight = torch.randn(len(p), 2, dtype=torch.double)
    u = s.solve(a, b, boundary)
    ga, gb = torch.autograd.grad((u*weight).sum(), (a, boundary))
    interior = np.setdiff1d(np.arange(len(p)), b)
    k = s.stiffness(a)
    dense = torch.zeros_like(u).index_copy(0, torch.tensor(b), boundary)
    ui = torch.linalg.solve(k[interior][:, interior], -k[interior][:, b] @ boundary)
    dense = dense.index_copy(0, torch.tensor(interior, dtype=torch.long), ui)
    da, db = torch.autograd.grad((dense*weight).sum(), (a, boundary))
    torch.testing.assert_close(ga, da, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(gb, db, atol=1e-11, rtol=1e-10)
    direction = torch.randn_like(a)
    direction = (direction + direction.transpose(-1, -2))/2
    direction_b = torch.randn_like(boundary)
    eps = 1e-6
    fd = (((s.solve(a+eps*direction, b, boundary+eps*direction_b)-s.solve(a-eps*direction, b, boundary-eps*direction_b))*weight).sum()/(2*eps))
    expected = (ga*direction).sum()+(gb*direction_b).sum()
    torch.testing.assert_close(fd, expected, atol=2e-8, rtol=2e-8)


def test_dual_stream_detects_nonconserved_cochain():
    p, f, _ = grid()
    s = system(p, f)
    q = torch.arange(len(s.edges), dtype=torch.double).sin()
    assert s.dual_stream(q)[1] > 1e-3
