"""Read-only per-round pseudo-teacher diagnostic for a saved 257² model.

The teacher is loaded only after image-only inference has completed. This is
development analysis, not a network input or a held-out accuracy claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from tools.digital_acrobat_recurrent_probe import _load_model
from tools.digital_acrobat_teacher_probe import load_case_inputs


def vector_rmse(vertices: np.ndarray, matrix: np.ndarray,
                offset: np.ndarray, teacher: np.ndarray) -> float:
    predicted = np.einsum("hwi,ji->hwj", vertices, matrix) + offset
    return float(np.sqrt(np.mean(np.sum((predicted - teacher) ** 2, axis=-1))))


def corner_min_and_nonpositive(vertices: np.ndarray) -> tuple[float, int]:
    """Independent four-corner float64 diagnostic of represented vertices."""
    a = vertices[:-1, :-1]
    b = vertices[:-1, 1:]
    c = vertices[1:, 1:]
    d = vertices[1:, :-1]

    def det(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    corners = (det(b - a, d - a), det(b - a, c - b),
               det(c - d, c - b), det(c - d, d - a))
    return min(float(item.min()) for item in corners), sum(
        int(np.count_nonzero(item <= 0)) for item in corners)


def trace(root: Path, model_output: Path, cases: list[int], output: Path,
          device: str, states_archive: Path | None = None,
          budget_audit: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if states_archive is not None and states_archive.exists():
        raise FileExistsError(states_archive)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("distinct nonempty case IDs required")
    model = _load_model(model_output, device)
    if 257 not in model.rounds_by_side:
        raise ValueError("this diagnostic requires a 257² recurrent head")
    captured: list[np.ndarray] = []

    def hook(_module, inputs, returned):
        if not captured:
            captured.append(inputs[0][0].detach().cpu().numpy().astype(np.float64))
        captured.append(returned[0].detach().cpu().numpy().astype(np.float64))

    handle = model.updates["257"].register_forward_hook(hook)
    update = model.updates["257"]
    original_update_color = update._update_color
    budget_entries: list[dict[str, int]] = []

    def audited_update_color(self, current, latent, area_floor, color):
        """Count the exact production freeze branches without changing their result."""
        vertices = getattr(self, f"_vertices_{color}")
        opposite = vertices[:, None, None] + self._opposite_offsets[None]
        local = ((vertices // self.side - 1) * (self.side - 2)
                 + vertices % self.side - 1)
        point64 = current[:, vertices].to(torch.float64)
        start = current[:, opposite[..., 0]].to(torch.float64)
        end = current[:, opposite[..., 1]].to(torch.float64)
        edge = end - start
        relative = point64[:, :, None] - start
        areas = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
        budget = self.safety_fraction * areas
        if area_floor is not None:
            budget = torch.minimum(
                budget, (areas - area_floor[:, None, None].to(torch.float64)).clamp_min(0),
            )
        valid_margin = (budget > 0).all(dim=-1)
        safe_budget = torch.where(budget > 0, budget, torch.ones_like(budget))
        raw = self._raw_proposal(current, vertices, latent[:, local])
        adverse = -(edge[..., 0] * raw[:, :, None, 1]
                    - edge[..., 1] * raw[:, :, None, 0]) / safe_budget
        valid_maximum = torch.isfinite(adverse.clamp_min(0).amax(dim=-1))
        budget_entries.append({
            "color": color,
            "active_vertices": int(valid_margin.numel()),
            "no_positive_margin": int((~valid_margin).sum().item()),
            "nonfinite_maximum": int((~valid_maximum).sum().item()),
            "frozen_total": int((~(valid_margin & valid_maximum)).sum().item()),
        })
        return original_update_color(current, latent, area_floor, color)

    if budget_audit:
        update._update_color = MethodType(audited_update_color, update)
    rows = []
    all_states = []
    all_references = []
    try:
        for case in cases:
            captured.clear()
            budget_entries.clear()
            example = load_case_inputs(root, case, torch.device(device))
            with torch.no_grad():
                final, _, _ = model(example["fixed"], example["prewarped"])
            if len(captured) != model.rounds_by_side[257] + 1:
                raise RuntimeError("unexpected number of captured 257² states")
            if not np.array_equal(captured[-1].astype(np.float32),
                                  final[0].detach().cpu().numpy()):
                raise RuntimeError("last captured map is not model output")
            if budget_audit and len(budget_entries) != 4 * model.rounds_by_side[257]:
                raise RuntimeError("unexpected number of color-budget records")
            with np.load(model_output / f"{case}_actual_safe_q1.npz",
                         allow_pickle=False) as archive:
                saved_final = archive["vertices"][0].astype(np.float64)
                reference = archive["boundary_reference"]
            if states_archive is not None:
                all_states.append(np.stack(captured).astype(np.float32))
                all_references.append(np.repeat(reference,
                                                len(captured), axis=0))
            with np.load(root / f"{case}_DHR_physical_full_teacher_affine.npz",
                         allow_pickle=False) as archive:
                teacher = archive["raw_teacher_vertices"][0].astype(np.float64)
            matrix = example["matrix"][0].cpu().numpy().astype(np.float64)
            offset = example["offset"][0].cpu().numpy().astype(np.float64)
            errors = [vector_rmse(state, matrix, offset, teacher)
                      for state in captured]
            signs = [corner_min_and_nonpositive(state) for state in captured]
            rows.append({"case": case, "round_vector_rmse": errors,
                         "final_minus_parent": errors[-1] - errors[0],
                         "minimum_intermediate_corner_float64": min(
                             value[0] for value in signs),
                         "nonpositive_intermediate_corners_float64": sum(
                             value[1] for value in signs),
                         "max_abs_replay_minus_saved_vertices": float(
                             np.max(np.abs(captured[-1] - saved_final))),
                         "steps_improving_pseudo_teacher": sum(
                             after < before for before, after in zip(
                                 errors[:-1], errors[1:]))})
            if budget_audit:
                rows[-1]["budget_audit"] = [
                    budget_entries[4 * t:4 * (t + 1)]
                    for t in range(model.rounds_by_side[257])
                ]
    finally:
        handle.remove()
        if budget_audit:
            update._update_color = original_update_color
    report = {
        "question": "Do successive learned 257² F1 rounds improve the saved DHR pseudo-teacher fit?",
        "scope": "previously opened development cases; same frozen checkpoint; no retraining or anatomy claim",
        "metric": "Euclidean-vector RMSE over 257² vertices after saved external affine",
        "corner_metric": "float64 arithmetic on captured binary32 vertices; not an exact-integer sign certificate",
        "rounds": model.rounds_by_side[257],
        "rows": rows,
        "mean_round_vector_rmse": np.mean(
            [row["round_vector_rmse"] for row in rows], axis=0).tolist(),
    }
    if budget_audit:
        report["budget_audit"] = {
            "scope": "all four actual 257² F1 colors at every round entry; diagnostic replay",
            "no_positive_margin": sum(
                entry["no_positive_margin"] for row in rows
                for round_entries in row["budget_audit"] for entry in round_entries),
            "nonfinite_maximum": sum(
                entry["nonfinite_maximum"] for row in rows
                for round_entries in row["budget_audit"] for entry in round_entries),
            "frozen_total": sum(
                entry["frozen_total"] for row in rows
                for round_entries in row["budget_audit"] for entry in round_entries),
        }
    if states_archive is not None:
        states_archive.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(states_archive,
                            vertices=np.concatenate(all_states),
                            boundary_reference=np.concatenate(all_references))
        report["states_archive"] = str(states_archive)
        report["state_count"] = len(cases) * (model.rounds_by_side[257] + 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--states-archive", type=Path,
                        help="optional saved binary32 stack for full exact-sign audit")
    parser.add_argument("--budget-audit", action="store_true",
                        help="count production no-margin and nonfinite freeze branches")
    args = parser.parse_args()
    result = trace(args.root, args.model_output, args.cases, args.output,
                   args.device, args.states_archive, args.budget_audit)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}))


if __name__ == "__main__":
    main()
