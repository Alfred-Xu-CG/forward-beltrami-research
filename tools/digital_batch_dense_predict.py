"""Run frozen safe-257 checkpoints over a declared image-only pair manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_mind_dense257_predict import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--pair", action="append", nargs=4,
                        metavar=("NAME", "FIXED", "MOVING", "AFFINE"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not args.prefix.replace("_", "").isalnum():
        raise ValueError("simple alphanumeric prefix required")
    rows = []
    for name, fixed, moving, affine in args.pair:
        output = args.output_dir / f"{name}_{args.prefix}_safe257.npz"
        row = predict(args.base, args.head, Path(fixed), Path(moving),
                      Path(affine), output, device_name=args.device,
                      repeats=args.repeats)
        rows.append({"name": name, "output": str(output),
                     "image": row["dense_256_P1_descriptor_loss"],
                     "finite_vjp": row["finite_full_parameter_vjp"]})
    print(json.dumps({"count": len(rows), "rows": rows}))


if __name__ == "__main__":
    main()
