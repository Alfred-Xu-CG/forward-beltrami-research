"""Read-only teacher fit on a recurrent model's own 102 training pairs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from qcopt.neural_bijection.dense.q1_image_network import Q1ImageRegistrationNetwork
from tools.digital_acrobat_recurrent_probe import _load_model
from tools.digital_acrobat_teacher_probe import load_case


def evaluate(root: Path, model_output: Path, result: Path, *, device: str) -> dict:
    if result.exists():
        raise FileExistsError(result)
    manifest = json.loads((model_output / "train_manifest.json").read_text())
    ids = manifest["train_case_ids"]
    model = _load_model(model_output, device)
    rows = []
    target_device = torch.device(device)
    with torch.no_grad():
        for case in ids:
            data = load_case(root, case, target_device)
            mapped, _, _ = model(data["fixed"], data["prewarped"])
            predicted = Q1ImageRegistrationNetwork.apply_affine(
                mapped, data["matrix"], data["offset"],
            )
            teacher = data["raw_teacher"]
            error = torch.sqrt((predicted - teacher).square().sum(-1).mean())
            rows.append({"case": case, "full_DHR_vertex_rmse": float(error)})
    report = {
        "question": "how well does this frozen-base recurrent network fit its own training teachers",
        "model_output": str(model_output),
        "train_case_ids_only": ids,
        "case_count": len(rows),
        "mean_case_rmse": statistics.mean(row["full_DHR_vertex_rmse"] for row in rows),
        "median_case_rmse": statistics.median(row["full_DHR_vertex_rmse"] for row in rows),
        "rows": rows,
    }
    result.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate(args.root, args.model_output, args.result, device=args.device)
    print(json.dumps({key: report[key] for key in
                      ("case_count", "mean_case_rmse", "median_case_rmse")}, indent=2))


if __name__ == "__main__":
    main()
