import torch

from qcopt.experiments.barrier_kkt_matrixfree_audit import _cg_solve_torch


def test_torch_matrix_free_cg_solves_quadratic_hessian() -> None:
    target = torch.zeros(5, dtype=torch.float64)
    q = torch.zeros(5, dtype=torch.float64)

    def energy(value: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(value, value)

    rhs = torch.arange(1.0, 6.0, dtype=torch.float64)
    solution, info, calls = _cg_solve_torch(
        q, target, energy, rhs, damping=0.25, rtol=1e-12, maxiter=20
    )
    assert info == 0
    assert calls <= 3
    assert torch.allclose(solution, rhs / 1.25, atol=1e-11, rtol=1e-11)
