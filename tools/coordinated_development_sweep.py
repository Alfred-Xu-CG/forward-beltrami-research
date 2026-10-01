"""Small declared development matrix; no anatomy labels or model selection."""
import argparse
import json
from pathlib import Path
from tools.coordinated_real_case import optimize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cases",nargs="+",default=["histo","lesions","rat_kidney"])
    p.add_argument("--methods",nargs="+",default=["radial","analytic","f1"])
    p.add_argument("--loss",choices=["mind","local_ncc"],required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--f2-accepted-gain",type=float,default=1.)
    p.add_argument("--strain-weight",type=float,default=.05)
    p.add_argument("--regional-min-level",type=int,default=3)
    p.add_argument("--precision",choices=("float32","float64"),default="float32")
    p.add_argument("--image-precision",choices=("same","float32","float64"),default="same")
    p.add_argument("--shape-weight",type=float,default=0.)
    p.add_argument("--image-levels",type=int,nargs="+")
    p.add_argument("--inner-steps",type=int,default=5)
    p.add_argument("--budget-multiplier",type=int,default=1)
    p.add_argument("--base-cycles",type=int,default=2,help="coordinate cycles; joint controls use twice this count")
    p.add_argument("--interpolation",choices=("q1","p1_ac","p1_bd"),default="q1")
    p.add_argument("--output-selection",choices=("last","best_full"),default="last")
    p.add_argument("--matches-dir",type=Path)
    p.add_argument("--match-weight",type=float,default=0.)
    p.add_argument("--match-robust-scale",type=float,default=8.)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for case in args.cases:
        for method in args.methods:
            config=argparse.Namespace(fixed=args.data/(case+"_fixed512.png"),
                moving=args.data/(case+"_moving512.png"),affine=args.data/(case+"_initial_affine.npz"),
                output=args.output/(case+"_"+method+"_"+args.loss+"_edge257.npz"),method=method,
                loss=args.loss,grid_side=257,image_side=512,image_levels=args.image_levels,levels=[17,33,65,129,257],inner_steps=args.inner_steps,
                cycles=args.base_cycles*(2 if method in ("f1","f2") else 1)*args.budget_multiplier,learning_rate=.004,lr_calibration="edge",patch_cells=8,
                f2_accepted_gain=args.f2_accepted_gain,
                regional_cells=32,regional_min_level=args.regional_min_level,
                strain_weight=args.strain_weight,oob_weight=1.,minimum_jacobian=.001,precision=args.precision,
                image_precision=args.image_precision,shape_weight=args.shape_weight,
                interpolation=args.interpolation,
                output_selection=args.output_selection,
                matches=args.matches_dir/(case+"_common_sg_raw_matches.json") if args.matches_dir else None,
                match_weight=args.match_weight,match_robust_scale=args.match_robust_scale,
                device=args.device,threads=2)
            try:
                report=optimize(config)
            except Exception as error:
                row=dict(case=case,method=method,loss=args.loss,status="failed",
                         error=type(error).__name__+": "+str(error))
                results.append(row)
                print(json.dumps(row),flush=True)
                continue
            row=dict(case=case,method=method,loss=args.loss,status="complete",final=report["final"],
                     failed_trials=report["failed_trials"],gradient_steps=report["gradient_steps"],
                     seconds=report["optimize_seconds"],peak_bytes=report["peak_allocated_bytes"])
            results.append(row)
            print(json.dumps(row),flush=True)
    (args.output/(args.loss+"_summary.json")).write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
