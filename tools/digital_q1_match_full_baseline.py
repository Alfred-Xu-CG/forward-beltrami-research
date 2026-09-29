"""Apply the predeclared fixed Gaussian Q1 rule to every accepted match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_q1_dhr_distill import identity_vertices
from tools.digital_q1_forward_kernel import forward_kernel_map
from tools.digital_q1_match_neural_decoder import match_rmse


def run(matches: Path, affine: Path, output: Path, *, device_name: str) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    data = json.loads(matches.read_text(encoding="utf-8"))
    if data["status"] != "ok" or data["ransac_inliers"] < 8:
        raise ValueError("at least eight accepted matches required")
    declared_affine = data.get("affine_map")
    if declared_affine is not None and Path(declared_affine).name != affine.name:
        raise ValueError("match coordinates and affine archive use different declared frames")
    device = torch.device(device_name)
    source = torch.tensor(data["source_points_unit"], device=device, dtype=torch.float32)
    target = torch.tensor(data["target_points_unit"], device=device, dtype=torch.float32)
    with np.load(affine) as factor:
        matrix = np.asarray(factor["post_affine_matrix"], dtype=np.float32)
        offset = np.asarray(factor["post_affine_offset"], dtype=np.float32)
    reference = identity_vertices(257, device=device)
    with torch.no_grad():
        mapped = forward_kernel_map(source, target, final_side=257, sigma=.12)
        fit = float(match_rmse(mapped, source, target))
        np.savez_compressed(output, vertices=mapped.cpu().numpy(),
                            boundary_reference=reference.cpu().numpy(),
                            post_affine_matrix=matrix, post_affine_offset=offset)
    certificate = certify_q1_binary_map(output)
    if not certificate["valid"]:
        raise RuntimeError("baseline saved map invalid")
    result = {"method": "fixed sigma .12 Gaussian proposals at 17/33/65, all matches",
              "matches": str(matches), "affine": str(affine), "map": str(output),
              "input_matches": len(source), "P1_training_match_rmse": fit,
              "certificate": certificate, "landmarks_loaded": False}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.matches, args.affine, args.output, device_name=args.device)
    print(json.dumps({"input_matches": result["input_matches"],
                      "P1_training_match_rmse": result["P1_training_match_rmse"],
                      "certificate_valid": result["certificate"]["valid"]}))


if __name__ == "__main__":
    main()
