import torch

from qcopt.forward.monotone_torch import positive_increment_knots


def test_positive_increment_torch_decoder_is_differentiable_and_monotone():
    logits = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64, requires_grad=True)
    knots = positive_increment_knots(logits)
    loss = torch.sum((knots - torch.linspace(0.0, 1.0, 33, dtype=torch.float64)) ** 2)
    loss.backward()
    assert torch.all(torch.diff(knots) > 0.0)
    assert torch.isfinite(logits.grad).all()
