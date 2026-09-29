"""Run frozen recurrent safe-257 image maps over declared input pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_mind_recurrent_predict import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--pair", action="append", nargs=4,
                        metavar=("NAME", "FIXED", "MOVING", "AFFINE"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--match-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rows = []
    for name, fixed, moving, affine in args.pair:
        output = args.output_dir / f"{name}_{args.prefix}_safe257.npz"
        row = predict(args.base, args.recurrent, Path(fixed), Path(moving),
                      Path(affine), output, device_name=args.device,
                      repeats=args.repeats,
                      matches_path=(None if args.match_dir is None else
                                    args.match_dir / f"{name}_alignedSG_matches.npz"))
        rows.append({"name": name, "output": str(output),
                     "image": row["recurrent_256_P1_descriptor_loss"],
                     "finite_vjp": row["finite_full_parameter_vjp"]})
    print(json.dumps({"count": len(rows), "rows": rows}))


if __name__ == "__main__":
    main()
