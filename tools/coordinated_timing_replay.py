"""Counterbalanced repeat of one real registration configuration, without labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tools.coordinated_real_case import optimize


def replay_configuration(report, output, sampling,comparison="sampling",grid_side=None):
    config=dict(report["configuration"])
    for name in ("fixed","moving","affine","matches"):
        if config.get(name) is not None:
            config[name]=Path(config[name])
    config["output"]=Path(output)
    if grid_side is not None:
        if not isinstance(grid_side,int) or grid_side<config["levels"][-1]:
            raise ValueError("grid override must not shrink the template's final control size")
        config["grid_side"]=grid_side
        config["levels"]=list(config["levels"][:-1])+[grid_side]
    if comparison=="sampling":config["p1_sampling"]=sampling
    elif comparison=="geometry":config["geometry_backend"]=sampling
    elif comparison=="hierarchy":config["control_hierarchy"]=sampling
    elif comparison=="nested_evidence":
        config["control_hierarchy"]="nested_p1"
        config["nested_evaluation"]=sampling
    else:raise ValueError("declare sampling, geometry, hierarchy or nested_evidence comparison")
    if config.get("interpolation") not in ("p1_ac","p1_bd"):
        raise ValueError("timing comparison requires a declared actual P1 output")
    return argparse.Namespace(**config)


def sampling_order(repeat,variants=("existing","frozen")):
    return variants if repeat%2==0 else tuple(reversed(variants))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats<4 or args.repeats%2:
        raise ValueError("an even number of at least four measured pairs is required")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=json.loads(args.report.read_text())
    rows=[]
    comparison=getattr(args,"comparison","sampling")
    variants={"sampling":("existing","frozen"),"geometry":("existing","stage_cache"),
              "hierarchy":("fixed","nested_p1"),"nested_evidence":("full_fine","coarse_exact")}[comparison]
    # Both full configurations warm up in the SAME process before any timed pair.
    for repeat in range(-1,args.repeats):
        for sampling in sampling_order(max(repeat,0),variants):
            suffix="warmup" if repeat<0 else f"repeat{repeat}"
            output=args.output.parent/(args.output.stem+"_"+suffix+"_"+sampling+".npz")
            config=replay_configuration(report,output,sampling,comparison,getattr(args,"grid_side",None))
            tick=time.perf_counter()
            result=optimize(config)
            call_seconds=time.perf_counter()-tick
            row=dict(repeat=repeat,sampling=sampling,warmup=repeat<0,
                comparison=comparison,variant=sampling,
                complete_call_seconds=call_seconds,
                optimize_seconds=result["optimize_seconds"],
                end_to_end_seconds=result["end_to_end_seconds"],
                feature_seconds=result["feature_seconds"],
                peak_allocated_bytes=result["peak_allocated_bytes"],
                final=result["final"],gradient_steps=result["gradient_steps"],
                failed_trials=result["failed_trials"],
                certificate_valid=result["saved_binary_certificate"]["valid"],
                output=str(output))
            rows.append(row)
            print(json.dumps(row),flush=True)
    medians={}
    for sampling in variants:
        selected=[row for row in rows if not row["warmup"] and row["sampling"]==sampling]
        medians[sampling]={name:statistics.median(row[name] for row in selected)
            for name in ("optimize_seconds","end_to_end_seconds","complete_call_seconds","feature_seconds","peak_allocated_bytes")
            if selected[0][name] is not None}
    payload=dict(template=str(args.report),comparison=comparison,configuration=report["configuration"],
        actual_configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(config).items()},
        grid_side_override=getattr(args,"grid_side",None),rows=rows,medians=medians,
        scope="warmed steady-state same-process AB/BA registrations; no anatomical labels loaded; complete_call_seconds includes final diagnostic/report serialization; historical end_to_end_seconds stops before these; peak_allocated_bytes is optimizer-phase peak after feature setup, includes resident cache but not setup temporaries",
        precision_caution="CUDA reduction/optimizer branches may differ slightly; caches do not change the deformation family; hierarchy changes the feasible subspace trajectory while preserving the fine-grid objective")
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=4)
    parser.add_argument("--comparison",choices=("sampling","geometry","hierarchy","nested_evidence"),default="sampling")
    parser.add_argument("--grid-side",type=int,help="Optional larger actual final CONTROL size, replacing final coefficient level; image/query resolution unchanged")
    run(parser.parse_args())


if __name__=="__main__":
    main()
