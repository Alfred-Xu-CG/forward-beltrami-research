from types import SimpleNamespace

import torch

from tools.coordinated_multilevel_benchmark import benchmark_case


def test_tiny_full_chain_benchmark_storage_lifetime_and_complete_counts():
    yy,xx=torch.meshgrid(torch.linspace(0,1,5,dtype=torch.float64),
                         torch.linspace(0,1,5,dtype=torch.float64),indexing="ij")
    reference=torch.stack((xx,yy),-1)[None]
    anchor=reference.clone();anchor[:,1:-1,1:-1,0]+=.015*torch.sin(torch.pi*yy[1:-1,1:-1])
    args=SimpleNamespace(seed=20261001,minimum_jacobian=.001,theta=.95,
                         amplitude=.005,diagonal="ac",warmup=3,repeats=10)
    result=benchmark_case(anchor,reference,[3,5],args)
    assert result["status"]=="success" and result["stage_count"]==4
    assert result["latent_parameter_count"]==20
    assert len(result["stage_diagnostics"])==4
    for method in ("existing","manual","checkpointed_manual"):
        row=result[method]
        assert row["candidate_maximum_absolute_error"]==0
        assert row["all_hook_handles_collected"] and row["fixed_boundary_equal"]
        assert row["full_control_latent"] and row["actual_normalized_qmin"]>.001
        assert len(row["forward_seconds"])==len(row["vjp_seconds"])==10
        assert len(row["gradient_errors"])==3
        assert all(e["relative_l2_error"]<1e-9 for e in row["gradient_errors"])
