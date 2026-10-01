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
    p.add_argument("--preprocessing",choices=("raw_inverted","native_dhr"),default="raw_inverted")
    p.add_argument("--device",default="cuda")
    p.add_argument("--f2-accepted-gain",type=float,default=1.)
    p.add_argument("--strain-weight",type=float,default=.05)
    p.add_argument("--strain-model",choices=("displacement_gradient","p1_arap"),default="displacement_gradient")
    p.add_argument("--regional-min-level",type=int,default=3)
    p.add_argument("--precision",choices=("float32","float64"),default="float32")
    p.add_argument("--image-precision",choices=("same","float32","float64"),default="same")
    p.add_argument("--shape-weight",type=float,default=0.)
    p.add_argument("--image-levels",type=int,nargs="+")
    p.add_argument("--continuation-scope",choices=("all_cycles","first_cycle"),default="all_cycles")
    p.add_argument("--inner-steps",type=int,default=5)
    p.add_argument("--budget-multiplier",type=int,default=1)
    p.add_argument("--base-cycles",type=int,default=2,help="coordinate cycles; joint controls use twice this count")
    p.add_argument("--lr-calibration",choices=("edge","physical"),default="edge",
                   help="Edge-scaled physical Adam step or unchanged physical step across coefficient levels")
    p.add_argument("--interpolation",choices=("q1","p1_ac","p1_bd"),default="q1")
    p.add_argument("--p1-sampling",choices=("existing","frozen"),default="existing")
    p.add_argument("--geometry-backend",choices=("existing","stage_cache"),default="existing",
                   help="Cache only supported global radial/analytic methods; controls retain existing decoder")
    p.add_argument("--coordinate-mode",choices=("alternating","joint"),default="alternating")
    p.add_argument("--joint-backend",choices=("ordinary","cached_manual"),default="cached_manual")
    p.add_argument("--proposal-filter-steps",type=int,default=0)
    p.add_argument("--proposal-filter-min-level",type=int,default=129)
    p.add_argument("--levels",type=int,nargs="+",default=[17,33,65,129,257])
    p.add_argument("--output-selection",choices=("last","best_full"),default="last")
    p.add_argument("--matches-dir",type=Path)
    p.add_argument("--match-weight",type=float,default=0.)
    p.add_argument("--match-robust-scale",type=float,default=8.)
    args=p.parse_args()
    if args.proposal_filter_steps and any(method not in ("radial","analytic") for method in args.methods):
        raise ValueError("proposal filter sweep requires only global radial/analytic methods")
    if args.coordinate_mode=="joint" and (args.geometry_backend!="existing" or
            any(method not in ("radial","analytic") for method in args.methods)):
        raise ValueError("joint sweep requires only global radial/analytic methods and geometry_backend existing")
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for case in args.cases:
        for method in args.methods:
            config=argparse.Namespace(fixed=args.data/(case+"_fixed512.png"),
                moving=args.data/(case+"_moving512.png"),affine=args.data/(case+"_initial_affine.npz"),
                output=args.output/(case+"_"+method+"_"+args.loss+"_"+args.lr_calibration+"257.npz"),method=method,
                loss=args.loss,grid_side=257,image_side=512,image_levels=args.image_levels,levels=args.levels,inner_steps=args.inner_steps,
                preprocessing=args.preprocessing,
                continuation_scope=args.continuation_scope,
                cycles=args.base_cycles*(2 if method in ("f1","f2") else 1)*args.budget_multiplier,learning_rate=.004,lr_calibration=args.lr_calibration,patch_cells=8,
                f2_accepted_gain=args.f2_accepted_gain,
                regional_cells=32,regional_min_level=args.regional_min_level,
                strain_weight=args.strain_weight,strain_model=args.strain_model,oob_weight=1.,minimum_jacobian=.001,precision=args.precision,
                image_precision=args.image_precision,shape_weight=args.shape_weight,
                interpolation=args.interpolation,
                p1_sampling=args.p1_sampling,
                geometry_backend=args.geometry_backend if method in ("radial","analytic") else "existing",
                coordinate_mode=args.coordinate_mode,joint_backend=args.joint_backend,
                proposal_filter_steps=args.proposal_filter_steps,proposal_filter_min_level=args.proposal_filter_min_level,
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
