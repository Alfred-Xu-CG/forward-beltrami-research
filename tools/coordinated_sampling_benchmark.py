"""Fixed-query P1 sampling only: existing versus precomputed geometry.

No image/geometry optimizer or certificate is measured. Fixed cached queries
have no gradients; both paths differentiate the same nonregular vertex table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import numpy as np

from qcopt.neural_bijection.dense.coordinated_fixed_sampling import FrozenP1Evaluator
from qcopt.neural_bijection.dense.coordinated_sampling import p1_map_at_queries
from qcopt.neural_bijection.dense.q1_image_sampling import fixed_pixel_centers


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.repeats<10 or args.warmup<3:raise ValueError("at least10 repeats/3 warmups required")
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    with np.load(args.anchor) as archive:
        base=torch.from_numpy(archive["vertices"].copy()).double()
        diagonal=str(archive["interpolation"].item()).removeprefix("p1_")
    if base.ndim!=4 or base.shape[0]!=1 or diagonal not in ("ac","bd"):
        raise ValueError("batch1 declared P1 anchor required")
    base=base.to(device);rows,columns=base.shape[1:3]
    sync=lambda:torch.cuda.synchronize(device) if device.type=="cuda" else None
    results=[]
    for count in args.points:
        total=args.image_side**2
        if count<1 or count>total or total%count:raise ValueError("point count must divide pixel count")
        for batch in args.batches:
            if batch<1:raise ValueError("positive mapbatch required")
            values=base.expand(batch,-1,-1,-1).clone()
            generator=torch.Generator(device=device);generator.manual_seed(args.seed)
            upstream=torch.randn(batch,count,2,generator=generator,device=device,dtype=torch.float64)/(batch*count*2)
            # Untimed equality check on actual anchor with the entire query array.
            query=fixed_pixel_centers(args.image_side,args.image_side,dtype=torch.float64,device=device).reshape(1,-1,2)[:,::total//count]
            check=FrozenP1Evaluator(rows,columns,query,diagonal)
            vertex_check=values.detach().clone().requires_grad_()
            expected=p1_map_at_queries(vertex_check,query,diagonal,validate_queries=False)
            actual=check(vertex_check)
            value_difference=float((actual-expected).abs().max())
            eg,=torch.autograd.grad(expected,vertex_check,upstream)
            ag,=torch.autograd.grad(actual,vertex_check,upstream)
            gradient_difference=float((ag-eg).abs().max())
            if value_difference>2e-15 or gradient_difference>2e-14:raise RuntimeError("sampler equality check failed")
            del query,check,vertex_check,expected,actual,eg,ag
            for method in ("existing","fixed"):
                sync();tick=time.perf_counter()
                query=fixed_pixel_centers(args.image_side,args.image_side,dtype=torch.float64,device=device).reshape(1,-1,2)[:,::total//count].contiguous()
                if method=="fixed":
                    evaluator=FrozenP1Evaluator(rows,columns,query,diagonal)
                    cache_bytes=sum(b.numel()*b.element_size() for b in evaluator.buffers())
                    function=evaluator
                    del query  # Fixed forward does not retain its construction query.
                else:
                    cache_bytes=query.numel()*query.element_size()
                    function=lambda v:p1_map_at_queries(v,query,diagonal,validate_queries=False)
                sync();setup=time.perf_counter()-tick
                forward_times=[];vjp_times=[];resident=[];peak=[]
                for repeat in range(args.warmup+args.repeats):
                    vertices=values.detach().clone().requires_grad_()
                    sync()
                    if device.type=="cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        baseline=torch.cuda.memory_allocated(device)
                    else:baseline=None
                    tick=time.perf_counter();output=function(vertices);sync()
                    forward=time.perf_counter()-tick
                    tick=time.perf_counter();gradient,=torch.autograd.grad(output,vertices,upstream);sync()
                    vjp=time.perf_counter()-tick
                    if not bool(torch.isfinite(output).all() and torch.isfinite(gradient).all()):raise RuntimeError("nonfinite sampler/VJP")
                    if repeat>=args.warmup:
                        forward_times.append(forward);vjp_times.append(vjp);resident.append(baseline)
                        peak.append(torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
                    del vertices,output,gradient
                row=dict(method=method,batch=batch,points=count,source_rows=rows,source_columns=columns,
                    cache_bytes=cache_bytes,setup_seconds=setup,value_max_difference=value_difference,
                    vertex_vjp_max_difference=gradient_difference,median_forward_seconds=statistics.median(forward_times),
                    median_vjp_seconds=statistics.median(vjp_times),forward_seconds=forward_times,vjp_seconds=vjp_times,
                    resident_allocated_bytes=resident,peak_total_allocated_bytes=peak,
                    peak_increment_over_resident_bytes=[p-b for p,b in zip(peak,resident)] if peak[0] is not None else None)
                results.append(row);print(json.dumps(row),flush=True)
                del function
                if method=="fixed":del evaluator
                else:del query
            del values,upstream
    payload=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        results=results,device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
        scope="geometry float64, vertex-only VJP, same shared original-reference pixel queries and upstream; smaller point sets are strided pixel-center subsets, not uniform dense coverage; existing validates trusted queries=False; cached geometry setup excluded but counted in resident memory",
        caution="no query gradients supported, no neural-training/image-pipeline speedup claim; live allocated memory not allocator reserved memory")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--image-side",type=int,default=512);p.add_argument("--points",type=int,nargs="+",default=[4096,262144])
    p.add_argument("--batches",type=int,nargs="+",default=[1,4]);p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2);p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=10);p.add_argument("--seed",type=int,default=20261001)
    run(p.parse_args())


if __name__=="__main__":main()
