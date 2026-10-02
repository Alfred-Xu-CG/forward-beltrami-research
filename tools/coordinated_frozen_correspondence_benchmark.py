"""Same saved machine matches/map: existing versus frozen P1 query dispatch.

Setup is separate; ten alternating warm pairs measure forward and full
vertex/affine VJP. No Inductor, image loading, matcher, labels, or new energy.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from tools.coordinated_real_case import load_image_matches


def sync(device):
    if device.type=="cuda":
        torch.cuda.synchronize(device)


def initialize_device(device):
    if device.type not in ("cpu","cuda"):
        raise ValueError("cpu or cuda required")
    if device.type=="cuda":
        index=0 if device.index is None else device.index
        torch.cuda.set_device(index)
        torch.cuda.init()
        return torch.device("cuda",index)
    return device


def measure(layer,vertices,matrix,interpolation):
    vertices=vertices.detach().clone().requires_grad_()
    matrix=matrix.detach().clone().requires_grad_()
    device=vertices.device
    sync(device)
    baseline=torch.cuda.memory_allocated(device) if device.type=="cuda" else None
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    value=layer(vertices,matrix,interpolation)
    sync(device);forward=time.perf_counter()-started
    started=time.perf_counter()
    gradients=torch.autograd.grad(value,(vertices,matrix))
    sync(device);backward=time.perf_counter()-started
    row=dict(forward_seconds=forward,vjp_seconds=backward,forward_plus_vjp_seconds=forward+backward,
        cuda_baseline_allocated_bytes=baseline,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        device=str(device),geometry_shape=list(vertices.shape),point_count=int(layer.source.shape[1]))
    return row,(value.detach().cpu(),*(gradient.detach().cpu() for gradient in gradients))


def compare(left,right):
    def number(value):
        value=float(value)
        return value if math.isfinite(value) else None
    report={}
    for key,a,b in zip(("value","full_vertex_gradient","affine_matrix_gradient"),left,right):
        delta=b-a
        report[key]=dict(passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and
            torch.allclose(a,b,rtol=1e-9,atol=1e-10)),max_abs=number(delta.abs().max()),
            rms=number(delta.square().mean().sqrt()),shape=list(a.shape))
    report["passed"]=all(row["passed"] for row in report.values())
    report["tolerances"]=dict(rtol=1e-9,atol=1e-10)
    report["existing_value"]=number(left[0])
    report["frozen_value"]=number(right[0])
    return report


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.threads<1:raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device=initialize_device(torch.device(args.device))
    raw=json.loads(args.matches.read_text(encoding="utf-8"))
    if raw.get("status")!="ok" or raw.get("global_geometric_ransac_used") is not False:
        raise ValueError("original successful raw-confidence matches required")
    with np.load(args.anchor,allow_pickle=False) as saved:
        vertices=torch.from_numpy(saved["vertices"].copy()).to(device)
        matrix_array=saved["post_affine_matrix"].copy()
        offset=saved["post_affine_offset"].copy()
        interpolation=str(saved["interpolation"].item())
    if vertices.dtype!=torch.float64 or interpolation not in ("p1_ac","p1_bd"):
        raise ValueError("saved float64 P1 geometry required")
    if vertices.ndim!=4 or vertices.shape[0]!=1 or vertices.shape[-1]!=2 or not bool(torch.isfinite(vertices).all()):
        raise ValueError("finite single-map geometry required")
    basename=lambda path:Path(str(path).replace("\\","/")).name
    sync(device);started=time.perf_counter()
    baseline,metadata=load_image_matches(args.matches,matrix_array,offset,
        fixed_path=Path(basename(raw["fixed"])),moving_path=Path(basename(raw["moving"])),
        image_side=raw["image_side"],device=device,dtype=torch.float64,robust_scale=8.)
    sync(device);loading_seconds=time.perf_counter()-started
    cached=copy.deepcopy(baseline)
    sync(device);started=time.perf_counter()
    prepared=cached.prepare_fixed_p1_sampling(*vertices.shape[1:3],interpolation[-2:])
    sync(device);preparation_seconds=time.perf_counter()-started
    matrix=torch.as_tensor(matrix_array,device=device,dtype=torch.float64)
    report=dict(status="running",question="Does freezing only existing immutable point-query P1 geometry reduce dispatch cost without changing energy or map/affine VJPs?",
        source_anchor=str(args.anchor),source_matches=str(args.matches),match_metadata=metadata,
        preparation=prepared,preparation_seconds=preparation_seconds,loading_seconds=loading_seconds,
        torch_version=torch.__version__,device=str(device),threads=torch.get_num_threads(),
        cuda_device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        geometry_shape=list(vertices.shape),interpolation=interpolation,
        images_annotations_matcher_read=False,inductor_used=False,
        timing_scope="Input clones/CPU copies/checks excluded; synchronized grad-enabled F and full vertex+affine VJP separately. One first pair plus three warmup pairs excluded from ten alternating timed-pair medians. Preparation includes only evaluator construction, not deepcopy or match loading.",
        numerical_scope="Same mathematical P1 evaluation, differently grouped rounded arithmetic; not a bitwise equality contract. All outputs and full vertex/affine gradients checked on every pair.",
        samples=[],comparisons=[])
    layers=dict(existing=baseline,frozen=cached)
    for phase,count in (("first_observed",1),("warmup",3),("timed",10)):
        for repeat in range(count):
            order=("existing","frozen") if repeat%2==0 else ("frozen","existing")
            results={}
            for position,name in enumerate(order):
                row,results[name]=measure(layers[name],vertices,matrix,interpolation)
                row.update(phase=phase,repeat=repeat,order=list(order),position=position,backend=name)
                report["samples"].append(row)
            agreement=compare(results["existing"],results["frozen"])
            agreement.update(phase=phase,repeat=repeat)
            report["comparisons"].append(agreement)
            if not agreement["passed"]:
                report["status"]="mismatch_no_speed_claim"
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
                return report
    summary={}
    for name in layers:
        rows=[row for row in report["samples"] if row["phase"]=="timed" and row["backend"]==name]
        summary[name]={key:statistics.median(row[key] for row in rows)
                       for key in ("forward_seconds","vjp_seconds","forward_plus_vjp_seconds")}
    report.update(status="complete",warm_medians=summary,
        existing_over_frozen={key:summary["existing"][key]/summary["frozen"][key] for key in summary["existing"]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor",type=Path,required=True,help="actual saved HE->CC10 map NPZ")
    parser.add_argument("--matches",type=Path,required=True,help="original HE->CC10 raw-match JSON")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    report=run(parser.parse_args())
    print(json.dumps({key:report.get(key) for key in ("status","preparation_seconds","warm_medians","existing_over_frozen")}))
    if report["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
