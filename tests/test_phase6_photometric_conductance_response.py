"""Response construction and end-to-end first-order gain VJP on a dense P1 grid."""

import torch
import torch.nn.functional as F

from phase6_eval_crossroute_photographic import _edges_and_midpoints
from phase6_test_c_photometric_response import _analytic_responses, _edge_bases, _estimate
from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import SinePreconditionedTutteLayer
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def test_photometric_conductance_response_and_gain_vjp():
    torch.set_num_threads(4)
    side, image_side = 67, 64
    mesh = structured_rectangle(side-1, side-1)
    bases = _edge_bases(side, _edges_and_midpoints(side,mesh.vertices),
                        torch.device("cpu"))
    solver = SinePreconditionedTutteLayer(
        side, minimum_conductance=1, maximum_conductance=16,
        tolerance=1e-10, max_iterations=120,
    )
    responses, validation = _analytic_responses(solver,bases,-0.8)
    assert validation["true_relative_residual"] < 1e-10
    assert max(validation["finite_difference_max_abs"].values()) < 1e-7
    table = StructuredDenseQueryTable.from_mesh(
        mesh,height=image_side,width=image_side)
    table.prepare(device="cpu",dtype=torch.float32)
    response_query = table.interpolate(responses.float().reshape(18,-1,2))
    fixed,moving,*_ = make_dataset(1,image_side,4412,target_family="base",
                                   return_coefficients=True)
    coefficients,_ = _estimate(fixed,moving,response_query,
                               ridge=0.005,ridge_mode="mean",
                               coefficient_bound=4.0)

    def loss(gain):
        logits = tuple(-0.8+torch.einsum("bk,k...->b...",
                                           gain*coefficients,part) for part in bases)
        mapped = solver(*logits).float()
        dense = table.interpolate(mapped.reshape(1,-1,2))
        warped = F.grid_sample(moving,2*dense-1,mode="bilinear",
                               padding_mode="border",align_corners=True)
        return (warped-fixed).square().mean(),mapped

    gain = torch.tensor(1.0,dtype=torch.float64,requires_grad=True)
    objective,mapped = loss(gain)
    analytical = torch.autograd.grad(objective,gain)[0]
    step = 1e-2
    numerical = (loss(gain.detach()+step)[0]-loss(gain.detach()-step)[0])/(2*step)
    assert torch.isfinite(analytical)
    assert torch.allclose(analytical.float(),numerical,rtol=0.03,atol=1e-5)
    assert solver.last_forward_stats["minimum_signed_area_ratio"] > 0
    assert torch.isfinite(mapped).all()
