"""Evaluate one fixed image-only baseline plus a frozen safe residual head on a pair list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_frozen_residual_predict import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "one_head", "residual", "output_dir", "tag"):
        parser.add_argument("--" + key.replace("_", "-"),
                            type=Path if key != "tag" else str, required=True)
    parser.add_argument("--pair", action="append", nargs=5, required=True,
                        metavar=("NAME", "FIXED", "MOVING", "AFFINE", "MATCHES"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    rows = []
    for name, fixed, moving, affine, matches in args.pair:
        output = args.output_dir / f"{name}_{args.tag}_safe257.npz"
        row = predict(args.base, args.one_head, args.residual,
                      Path(fixed), Path(moving), Path(affine),
                      None if matches == "none" else Path(matches), output,
                      device_name=args.device, repeats=args.repeats)
        rows.append({"name": name, "map": str(output),
                     "image": row["residual_image"],
                     "forward_seconds": row["full_forward_seconds_median"],
                     "forward_vjp_seconds": row[
                         "full_forward_and_parameter_match_vjp_seconds_median"],
                     "finite_full_vjp": row["finite_full_vjp"]})
    report = {"tag": args.tag, "count": len(rows), "rows": rows}
    output_file = args.output_dir / f"{args.tag}_batch_report.json"
    if output_file.exists():
        raise FileExistsError(output_file)
    output_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tag": args.tag, "count": len(rows),
                      "all_finite_vjp": all(r["finite_full_vjp"] for r in rows)}))


if __name__ == "__main__":
    main()
