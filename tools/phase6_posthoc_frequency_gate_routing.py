"""Route existing complete C21 evaluations by an out-of-cohort scalar gate.

This is exact per-sample selection between two already-evaluated full solvers,
not timing or backpropagation of a newly implemented conditional layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase6_frequency_gate_threshold_transfer import _fit_threshold, _scores


def _evaluations(neutral_path: Path, trained_path: Path, count: int) -> tuple[list[dict], list[dict]]:
    neutral = json.loads(neutral_path.read_text(encoding="utf-8"))
    trained = json.loads(trained_path.read_text(encoding="utf-8"))
    if len(neutral["samples"]) != count or len(trained["samples"]) != count:
        raise ValueError("sample count mismatch")
    for key in ("control_vertices", "fit_side", "image_side", "target_family"):
        if neutral[key] != trained[key]:
            raise ValueError(f"mismatched evaluation: {key}")
    if neutral["control_vertices"] != 1025**2 or neutral["fit_side"] != 256:
        raise ValueError("expected 1025² control and 256² fit")
    for a, b in zip(neutral["samples"], trained["samples"]):
        if a["true_fine_amplitude"] != b["true_fine_amplitude"]:
            raise ValueError("per-sample ordering mismatch")
    return neutral["samples"], trained["samples"]


def _summary(samples: list[dict]) -> dict[str, float]:
    result = {}
    for name in ("image_mse", "query_map_mse", "face_beltrami_mse"):
        value = float(np.mean([sample[name] for sample in samples]))
        result[name if name == "image_mse" else name.replace("mse", "rmse")] = (
            value if name == "image_mse" else float(np.sqrt(value)))
    result["minimum_face_determinant"] = float(min(
        sample["minimum_face_determinant"] for sample in samples))
    result["maximum_predicted_beltrami_modulus"] = float(max(
        sample["maximum_predicted_beltrami_modulus"] for sample in samples))
    return result


def _route(scores: np.ndarray, threshold: float,
           neutral: list[dict], trained: list[dict]) -> dict:
    use_trained = scores > threshold
    selected = [trained[i] if choice else neutral[i]
                for i, choice in enumerate(use_trained)]
    return {
        "count": len(scores),
        "trained64_selected": int(np.count_nonzero(use_trained)),
        "neutral64": _summary(neutral),
        "trained64": _summary(trained),
        "routed": _summary(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.results
    old_low, old_high, _ = _scores(directory / "c21_neutral_latent_frequency_gate_oldphoto96_pair_fit256_cpu.json")
    new_low, new_high, _ = _scores(directory / "c21_neutral_latent_frequency_gate_photo_high64_pair_fit256_corrected_cpu.json")
    threshold_old, _ = _fit_threshold(old_low, old_high)
    threshold_new, _ = _fit_threshold(new_low, new_high)
    synthetic = json.loads((directory / "c21_neutral_latent_frequency_gate_fresh128_photo32_cpu.json").read_text(encoding="utf-8"))
    synthetic_high = np.asarray(synthetic["synthetic"]["high64"]["fraction64"])
    cohorts = {
        "old6_high32_test_new4_threshold": (
            old_low, threshold_new,
            "c21_trainedold_neutral64_photo96_fit256_cpu.json",
            "c21_high64_trained_photo96_fit256_cpu.json"),
        "new4_high32_test_old6_threshold": (
            new_low, threshold_old,
            "c21_neutral64_newphotos32_fit256_cpu.json",
            "c21_fulltrained_newphotos32_fit256_cpu.json"),
        "synthetic_high64_test_old6_threshold": (
            synthetic_high, threshold_old,
            "c21_trainedold_neutral64_high64_fresh128_fit256_cpu.json",
            "c21_fulltrained_high64_fresh128_fit256_cpu.json"),
        "synthetic_high64_test_new4_threshold": (
            synthetic_high, threshold_new,
            "c21_trainedold_neutral64_high64_fresh128_fit256_cpu.json",
            "c21_fulltrained_high64_fresh128_fit256_cpu.json"),
    }
    output = {"threshold_old6": threshold_old, "threshold_new4": threshold_new,
              "basis": "post-hoc exact selection between pre-evaluated complete C21 maps; no gate VJP or timing",
              "cohorts": {}}
    for label, (scores, threshold, neutral_name, trained_name) in cohorts.items():
        neutral, trained = _evaluations(directory / neutral_name,
                                        directory / trained_name, len(scores))
        output["cohorts"][label] = _route(scores, threshold, neutral, trained)
    result = json.dumps(output, sort_keys=True, separators=(",", ":"))
    if args.output is not None:
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
