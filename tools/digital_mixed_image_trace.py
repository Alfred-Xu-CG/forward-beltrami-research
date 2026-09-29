"""Archive actual-image mixed F2/F1 round endpoints without opening teachers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map
from tools.digital_acrobat_recurrent_probe import _load_model
from tools.digital_acrobat_teacher_probe import load_case_inputs


def trace(root: Path, model_output: Path, cases: list[int], output: Path,
          device: str) -> dict:
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("distinct nonempty case IDs required")
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("fresh trace outputs required")
    model = _load_model(model_output, device)
    if model.update_families_by_side.get(257) != "f2_f1":
        raise ValueError("model must have a mixed F2/F1 257-level head")
    patch = model.updates["257"]["f2"]
    f1 = model.updates["257"]["f1"]
    stages: list[torch.Tensor] = []

    def entry(_module, args):
        stages.append(args[0].detach().cpu().clone())

    def endpoint(_module, _args, result):
        stages.append(result.detach().cpu().clone())

    handles = (
        patch.passes[0].register_forward_pre_hook(entry),
        patch.passes[-1].register_forward_hook(endpoint),
        f1.register_forward_hook(endpoint),
    )
    rows = []
    all_states = []
    try:
        with torch.no_grad():
            for case in cases:
                data = load_case_inputs(root, case, torch.device(device))
                stages.clear()
                final = model(data["fixed"], data["prewarped"])[0]
                count = model.rounds_by_side[257]
                if len(stages) != 3 * count or not torch.equal(
                    stages[-1], final.cpu(),
                ):
                    raise AssertionError("mixed stage capture mismatch")
                case_states = torch.cat(stages, dim=0).numpy().astype(np.float32)
                all_states.append(case_states)
                rounds = []
                for k in range(count):
                    start, after_f2, after_f1 = stages[3 * k:3 * k + 3]
                    rounds.append({
                        "round": k + 1,
                        "f2_componentwise_rms_update": float(
                            (after_f2 - start).square().mean().sqrt()),
                        "f1_componentwise_rms_update": float(
                            (after_f1 - after_f2).square().mean().sqrt()),
                        "f2_changed_coordinate_components": int(
                            (after_f2 != start).sum()),
                        "f1_changed_coordinate_components": int(
                            (after_f1 != after_f2).sum()),
                        "f2_minimum_normalized_q1_corner": float(
                            q1_corner_determinants(after_f2).amin() * 256 ** 2),
                        "f1_minimum_normalized_q1_corner": float(
                            q1_corner_determinants(after_f1).amin() * 256 ** 2),
                    })
                rows.append({"case": case, "rounds": rounds})
    finally:
        for handle in handles:
            handle.remove()
    values = np.concatenate(all_states, axis=0)
    axis = np.arange(257, dtype=np.float32) / 256
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    reference = np.stack((xx, yy), axis=-1)[None]
    archive = output.with_suffix(".npz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, vertices=values,
                        boundary_reference=np.broadcast_to(reference, values.shape).copy(),
                        case_ids=np.asarray(cases, dtype=np.int64))
    certificate = certify_q1_binary_map(archive)
    if not certificate["valid"]:
        raise ArithmeticError("a saved mixed-model stage is not homeomorphic")
    report = {
        "question": "Do both trained mixed families make nonzero current-state updates?",
        "scope": "reused development case IDs; actual images; no teacher opened",
        "model_output": str(model_output), "case_ids": cases,
        "control_side": 257, "rounds": model.rounds_by_side[257],
        "stage_order_per_case": ["round_entry", "after_f2", "after_f1"],
        "saved_archive": archive.name, "certificate": certificate,
        "rows": rows,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = trace(args.root, args.model_output, args.cases, args.output, args.device)
    print(json.dumps({"case_count": len(result["rows"]),
                      "certificate": result["certificate"]}, indent=2))


if __name__ == "__main__":
    main()
