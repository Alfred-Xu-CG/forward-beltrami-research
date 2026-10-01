import importlib.util
import gc
import weakref
from types import SimpleNamespace
import torch


def test_benchmark_counts_retained_storage_and_both_gradients():
    assert importlib.util.find_spec("tools.coordinated_explicit_vjp_benchmark") is not None,"explicit benchmark not implemented"
    from tools.coordinated_explicit_vjp_benchmark import benchmark_pair
    y,x=torch.meshgrid(torch.linspace(0,1,5,dtype=torch.float64),torch.linspace(0,1,7,dtype=torch.float64),indexing="ij")
    reference=torch.stack((x,y),-1)[None]
    anchor=reference.clone();anchor[0,2,3,0]+=.02
    args=SimpleNamespace(warmup=3,repeats=10,seed=42,minimum_jacobian=.001,theta=.95)
    result=benchmark_pair(anchor,reference,"analytic",.02,args)
    assert result["candidate_max_difference"]==0
    assert result["current_y_gradient_max_difference"]<1e-12
    assert result["proposal_gradient_max_difference"]<1e-12
    assert result["current_y_gradient_relative_l2_error"]<1e-10
    assert result["proposal_gradient_relative_l2_error"]<1e-10
    assert result["explicit"]["retained_saved_tensor_count"]==11
    for method in ("existing","explicit"):
        assert len(result[method]["forward_seconds"])==10
        assert result[method]["actual_normalized_qmin"]>.001
        assert result[method]["retained_saved_unique_storage_bytes"]>0


def test_storage_hook_does_not_keep_reciprocal_graph_cycle_after_counting():
    from tools.coordinated_explicit_vjp_benchmark import retained_storage
    refs=[]
    def evaluator(y,z):
        reciprocal=(y+2).reciprocal()
        refs.append(weakref.ref(reciprocal))
        return reciprocal+z
    y=torch.ones(2,dtype=torch.float64,requires_grad=True)
    z=torch.zeros_like(y,requires_grad=True)
    gc.collect();was_enabled=gc.isenabled();gc.disable()
    try:
        retained_storage(evaluator,y,z)
        assert refs[0]() is None,"storage counter retained a differentiable reciprocal graph cycle"
    finally:
        if was_enabled:gc.enable()
        gc.collect()
