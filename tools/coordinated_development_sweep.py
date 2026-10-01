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
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for case in args.cases:
        for method in args.methods:
            config=argparse.Namespace(fixed=args.data/(case+"_fixed512.png"),
                moving=args.data/(case+"_moving512.png"),affine=args.data/(case+"_initial_affine.npz"),
                output=args.output/(case+"_"+method+"_"+args.loss+"_edge257.npz"),method=method,
                loss=args.loss,grid_side=257,image_side=512,levels=[17,33,65,129,257],inner_steps=5,
                cycles=4 if method in ("f1","f2") else 2,learning_rate=.004,lr_calibration="edge",patch_cells=8,
                f2_accepted_gain=args.f2_accepted_gain,
                strain_weight=args.strain_weight,oob_weight=1.,minimum_jacobian=.001,precision="float32",
                device=args.device,threads=2)
            report=optimize(config)
            row=dict(case=case,method=method,loss=args.loss,final=report["final"],
                     failed_trials=report["failed_trials"],gradient_steps=report["gradient_steps"],
                     seconds=report["optimize_seconds"],peak_bytes=report["peak_allocated_bytes"])
            results.append(row)
            print(json.dumps(row),flush=True)
    (args.output/(args.loss+"_summary.json")).write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
