import numpy as np
import torch

from qcopt.forward.beurling_direct import direct_beurling_apply
from qcopt.forward.beurling_torch import direct_beurling_apply_torch


def test_torch_scattered_beurling_matches_numpy_reference() -> None:
    rng = np.random.default_rng(20260919)
    source = 0.05 + 0.42 * rng.random(37) + 1j * (0.05 + 0.9 * rng.random(37))
    target = 0.55 + 0.4 * rng.random(29) + 1j * (0.05 + 0.9 * rng.random(29))
    values = rng.normal(size=source.size) + 1j * rng.normal(size=source.size)
    weights = 0.4 + 0.8 * rng.random(source.size)
    reference = direct_beurling_apply(source, values, weights, target_points=target, block_size=11)
    output = direct_beurling_apply_torch(
        torch.from_numpy(source),
        torch.from_numpy(values),
        torch.from_numpy(weights),
        target_points=torch.from_numpy(target),
        block_size=13,
    )
    assert torch.allclose(output, torch.from_numpy(reference), atol=2e-12, rtol=2e-12)


def test_torch_scattered_beurling_has_value_vjp() -> None:
    source = torch.tensor([0.1 + 0.2j, 0.4 + 0.3j, 0.7 + 0.8j], dtype=torch.complex128)
    target = torch.tensor([0.2 + 0.6j, 0.9 + 0.4j], dtype=torch.complex128)
    values = torch.tensor([1.0 + 0.2j, -0.3 + 0.5j, 0.4 - 0.1j], dtype=torch.complex128, requires_grad=True)
    weights = torch.ones(3, dtype=torch.float64)
    output = direct_beurling_apply_torch(source, values, weights, target_points=target, block_size=2)
    loss = torch.real(torch.vdot(output, output))
    loss.backward()
    assert values.grad is not None
    assert torch.all(torch.isfinite(values.grad))
