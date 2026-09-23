"""Nested SW-NE triangulation preserves the complete coarse P1 map."""

import math

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import evaluate_structured_p1_with_jacobian
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def test_nested_p1_refinement_is_exact_and_differentiable():
    torch.set_num_threads(4)
    coarse_side, fine_side = 9, 33
    coarse_mesh = structured_rectangle(coarse_side-1,coarse_side-1)
    source = torch.tensor(coarse_mesh.vertices.copy(),dtype=torch.float32)
    smooth = torch.sin(math.pi*source[:,0])*torch.sin(math.pi*source[:,1])
    mapped = torch.stack((
        source[:,0]+0.02*smooth,
        source[:,1]-0.015*smooth,
    ),dim=-1).reshape(1,coarse_side,coarse_side,2).requires_grad_()
    fine_table = StructuredDenseQueryTable.from_mesh(
        coarse_mesh,height=fine_side,width=fine_side)
    fine_table.prepare(device="cpu",dtype=torch.float32)
    refined = fine_table.interpolate(mapped.reshape(1,-1,2)).reshape(
        1,fine_side,fine_side,2)
    generator = torch.Generator().manual_seed(1427)
    queries = torch.rand(1,10000,2,generator=generator)
    coarse_value,coarse_jacobian = evaluate_structured_p1_with_jacobian(
        mapped,queries)
    fine_value,fine_jacobian = evaluate_structured_p1_with_jacobian(
        refined,queries)
    assert (coarse_value-fine_value).abs().max() < 2e-6
    assert (coarse_jacobian-fine_jacobian).abs().max() < 5e-5
    assert torch.linalg.det(fine_jacobian).min() > 0
    objective = fine_value.square().mean()
    gradient = torch.autograd.grad(objective,mapped)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
