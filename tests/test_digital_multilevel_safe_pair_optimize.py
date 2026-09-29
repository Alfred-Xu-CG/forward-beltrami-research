"""Small 257-square differentiability and topology fixture for multilevel steering."""

import torch
from torch import nn

from qcopt.neural_bijection.dense.digital_q1 import validate_q1_map
from tools.digital_kernel_match_safe257 import steer
from tools.digital_multilevel_safe_pair_optimize import multilevel_goal
from tools.digital_q1_dhr_distill import identity_vertices


def test_multilevel_control_has_vjp_and_valid_fixed_grid_map():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline = identity_vertices(257, device=device)
    controls = nn.ParameterList([
        nn.Parameter(torch.zeros((1, 2, s - 2, s - 2), device=device))
        for s in (17, 33, 65)
    ])
    with torch.no_grad():
        controls[0][:, 0, 4:8, 4:8] = .002
        controls[1][:, 1, 10:16, 10:16] = -.001
    goal = multilevel_goal(baseline, controls)
    mapped = steer(baseline, goal, passes=4)
    assert validate_q1_map(mapped.detach(), baseline)["valid"]
    loss = (mapped - baseline).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in controls)
    assert sum(float(p.grad.abs().sum()) for p in controls) > 0


if __name__ == "__main__":
    test_multilevel_control_has_vjp_and_valid_fixed_grid_map()
    print("passed")
