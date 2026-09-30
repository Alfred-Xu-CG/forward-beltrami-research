"""Compare image-proxy and selected-match gradients at saved safe maps.

The gradients are with respect to aligned interior vertex coordinates, before
the topology-safe update parameterization. No anatomical labels are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from tools.digital_acrobat_teacher_probe import _case_images
from tools.digital_affine_prewarp import warp_moving_to_fixed
from tools.digital_mind_amortized_network import structural_loss
from tools.digital_mind_objective_probe import self_similarity
from tools.digital_mind_safe_optimize import strain_penalty
from tools.digital_mind_sparse_match_finetune import p1_at_points, robust_match_loss
from tools.digital_match_feedback257 import AnalyticMatchFeedback257
from tools.digital_q1_real_optimize import _read_gray_thumbnail


def audit(root: Path, matches: Path, maps: Path, report: Path,
          device: torch.device, feedback_maps: Path | None = None,
          feedback_gains: tuple[float, ...] = (),
          teachers: Path | None = None) -> dict:
    prior = json.loads(report.read_text(encoding="utf-8"))
    rows = []
    for previous in prior["rows"]:
        case = int(previous["case"])
        map_path = maps / f"{case}_hybrid_dynamic_strain02_safe257.npz"
        with np.load(map_path) as data:
            vertices = data["vertices"].astype(np.float32)
            matrix = data["post_affine_matrix"].astype(np.float32)
            offset = data["post_affine_offset"].astype(np.float32)
        with np.load(root / f"{case}_directSG_affine.npz") as data:
            if not (np.array_equal(matrix, data["post_affine_matrix"])
                    and np.array_equal(offset, data["post_affine_offset"])):
                raise ValueError(f"saved-map affine mismatch: {case}")
        with np.load(matches / f"{case}_alignedSG_matches.npz") as data:
            if "post_affine_matrix" in data and not (
                    np.array_equal(matrix, data["post_affine_matrix"])
                    and np.array_equal(offset, data["post_affine_offset"])):
                raise ValueError(f"match affine mismatch: {case}")
            source = torch.from_numpy(data["source_fixed_unit"].copy()).to(device)
            target = torch.from_numpy(data["target_aligned_unit"].copy()).to(device)
        fixed_path, moving_path = _case_images(root, case)
        fixed, _ = _read_gray_thumbnail(fixed_path, 512)
        moving, _ = _read_gray_thumbnail(moving_path, 512)
        fixed, moving = fixed.to(device), moving.to(device)
        with torch.no_grad():
            aligned = warp_moving_to_fixed(
                moving, torch.tensor(matrix, device=device),
                torch.tensor(offset, device=device), height=512, width=512)
            fsmall = F.interpolate(fixed, size=(256, 256), mode="area")
            msmall = F.interpolate(aligned, size=(256, 256), mode="area")
            fdesc, fscale = self_similarity(fsmall)
            mdesc, _ = self_similarity(msmall)
            mask = ((fsmall > .04) & (fscale > 1e-4)).to(fixed.dtype)
        mapped = torch.from_numpy(vertices.copy()).to(device).requires_grad_(True)
        image_loss = structural_loss(fdesc, mdesc, mask, mapped)
        match_loss = robust_match_loss(p1_at_points(mapped, source), target)
        image_grad = torch.autograd.grad(image_loss, mapped)[0][:, 1:-1, 1:-1]
        match_grad = torch.autograd.grad(match_loss, mapped)[0][:, 1:-1, 1:-1]
        dot = (image_grad * match_grad).sum()
        image_norm = torch.linalg.vector_norm(image_grad)
        match_norm = torch.linalg.vector_norm(match_grad)
        cosine = dot / (image_norm * match_norm)
        match_support = match_grad.abs().sum(dim=-1) > 0
        image_on_match_support_norm = torch.linalg.vector_norm(
            image_grad[match_support])
        support_cosine = dot / (image_on_match_support_norm * match_norm)
        if not bool(torch.isfinite(cosine)) or image_norm <= 0 or match_norm <= 0:
            raise FloatingPointError(f"invalid objective gradients: {case}")
        if not bool(torch.isfinite(support_cosine)):
            raise FloatingPointError(f"invalid support-restricted cosine: {case}")
        expected = float(previous["student_image"])
        if abs(float(image_loss) - expected) > 2e-5:
            raise ValueError(f"image objective/frame mismatch: {case}")
        row = {
            "case": case, "selected_match_count": len(source),
            "image_loss": float(image_loss), "robust_match_loss_px": float(match_loss),
            "image_grad_interior_norm": float(image_norm),
            "match_grad_interior_norm": float(match_norm),
            "interior_gradient_dot": float(dot),
            "interior_gradient_cosine": float(cosine),
            "match_gradient_support_vertex_count": int(match_support.sum()),
            "image_gradient_energy_on_match_support_fraction": float(
                (image_on_match_support_norm / image_norm).square()),
            "support_restricted_gradient_cosine": float(support_cosine),
            "same_direction_first_order": bool(dot > 0),
            "map_strain": float(strain_penalty(mapped).detach()),
        }
        if teachers is not None:
            with np.load(teachers / f"{case}_multilevel25_safe257.npz") as data:
                if not (np.array_equal(matrix, data["post_affine_matrix"])
                        and np.array_equal(offset, data["post_affine_offset"])):
                    raise ValueError(f"teacher affine mismatch: {case}")
                teacher = torch.from_numpy(
                    data["vertices"].astype(np.float32)).to(device)
            if teacher.shape != mapped.shape:
                raise ValueError(f"teacher shape mismatch: {case}")
            map_loss = (mapped - teacher).square().sum(-1).mean()
            strain_loss = strain_penalty(mapped)
            map_grad = torch.autograd.grad(map_loss, mapped)[0][:, 1:-1, 1:-1]
            strain_grad = torch.autograd.grad(strain_loss, mapped)[0][:, 1:-1, 1:-1]
            weighted = {
                "map": 1000. * map_grad,
                "image": .1 * image_grad,
                "match": .01 * match_grad,
                "strain": .2 * strain_grad,
            }
            norms = {key: float(torch.linalg.vector_norm(value))
                     for key, value in weighted.items()}
            if not all(np.isfinite(value) for value in norms.values()):
                raise FloatingPointError(f"invalid weighted gradient: {case}")
            total_grad = sum(weighted.values())
            row.update({
                "teacher_map_mse": float(map_loss),
                "weighted_interior_gradient_norms": norms,
                "dominant_weighted_interior_gradient": max(norms, key=norms.get),
                "weighted_total_interior_gradient_norm": float(
                    torch.linalg.vector_norm(total_grad)),
                "weighted_map_image_cosine": float(torch.sum(
                    weighted["map"] * weighted["image"]) / (
                        norms["map"] * norms["image"])),
                "weighted_map_match_cosine": float(torch.sum(
                    weighted["map"] * weighted["match"]) / (
                        norms["map"] * norms["match"])),
                "weighted_strain_match_cosine": float(torch.sum(
                    weighted["strain"] * weighted["match"]) / (
                        norms["strain"] * norms["match"])),
            })
        if feedback_maps is not None:
            feedback_path = feedback_maps / (
                f"{case}_hybrid_matchfeedback1_safe257.npz")
            with np.load(feedback_path) as data:
                if not (np.array_equal(matrix, data["post_affine_matrix"])
                        and np.array_equal(offset, data["post_affine_offset"])):
                    raise ValueError(f"feedback affine mismatch: {case}")
                feedback_vertices = data["vertices"].astype(np.float32)
            feedback = torch.from_numpy(feedback_vertices.copy()).to(device)
            delta = (feedback - mapped.detach())[:, 1:-1, 1:-1]
            first_image = (image_grad * delta).sum()
            first_match = (match_grad * delta).sum()
            with torch.no_grad():
                image_after = structural_loss(fdesc, mdesc, mask, feedback)
                match_after = robust_match_loss(p1_at_points(feedback, source), target)
            row.update({
                "feedback_vertex_delta_interior_norm": float(
                    torch.linalg.vector_norm(delta)),
                "image_first_order_along_feedback": float(first_image),
                "match_first_order_along_feedback": float(first_match),
                "image_actual_feedback_change": float(image_after - image_loss),
                "match_actual_feedback_change_px": float(match_after - match_loss),
            })
        if feedback_gains:
            gains = []
            with torch.no_grad():
                for gain in feedback_gains:
                    variant = AnalyticMatchFeedback257(gain=gain).to(device)(
                        mapped.detach(), source, target)
                    area = q1_corner_determinants(variant) * 256**2
                    if not bool((area > 0).all()):
                        raise ValueError(f"nonpositive numeric corner: {case}, {gain}")
                    image_variant = structural_loss(fdesc, mdesc, mask, variant)
                    match_variant = robust_match_loss(
                        p1_at_points(variant, source), target)
                    gains.append({
                        "gain": gain,
                        "image_change": float(image_variant - image_loss),
                        "robust_match_change_px": float(match_variant - match_loss),
                        "minimum_normalized_q1_corner_numeric": float(area.amin()),
                        "max_difference_from_saved_feedback": (
                            None if feedback_maps is None or gain != 1. else
                            float((variant - feedback).abs().max())),
                    })
            row["gain_sweep"] = gains
        rows.append(row)
    cosines = [row["interior_gradient_cosine"] for row in rows]
    support_cosines = [row["support_restricted_gradient_cosine"] for row in rows]
    result = {"question": "Do image and match losses oppose locally in aligned "
            "interior-vertex space at the retained safe map?",
            "scope": "unconstrained first-order gradient only; not a safe finite "
            "update, anatomical metric or end-to-end matcher gradient",
            "case_count": len(rows),
            "opposing_gradient_count": sum(value < 0 for value in cosines),
            "mean_cosine": float(np.mean(cosines)),
            "median_cosine": float(np.median(cosines)),
            "mean_support_restricted_cosine": float(np.mean(support_cosines)),
            "median_support_restricted_cosine": float(np.median(support_cosines)),
            "rows": rows, "manual_anatomy_labels_read": False}
    if feedback_maps is not None:
        result.update({
            "feedback_first_order_image_uphill_count": sum(
                row["image_first_order_along_feedback"] > 0 for row in rows),
            "feedback_first_order_match_downhill_count": sum(
                row["match_first_order_along_feedback"] < 0 for row in rows),
            "feedback_actual_image_worse_count": sum(
                row["image_actual_feedback_change"] > 0 for row in rows),
            "feedback_actual_match_better_count": sum(
                row["match_actual_feedback_change_px"] < 0 for row in rows),
        })
    if teachers is not None:
        result.update({
            "teacher_map_source": str(teachers),
            "loss_weights_map_image_match_strain": [1000., .1, .01, .2],
            "weighted_output_gradient_scope": "aligned 255x255 interior vertex "
                "coordinates, not neural-parameter VJP",
            "mean_weighted_gradient_norms": {
                key: float(np.mean([
                    row["weighted_interior_gradient_norms"][key]
                    for row in rows]))
                for key in ("map", "image", "match", "strain")},
            "dominant_gradient_counts": {
                key: sum(row["dominant_weighted_interior_gradient"] == key
                         for row in rows)
                for key in ("map", "image", "match", "strain")},
            "mean_weighted_total_gradient_norm": float(np.mean([
                row["weighted_total_interior_gradient_norm"] for row in rows])),
        })
    if feedback_gains:
        result["gain_sweep_summary"] = [{
            "gain": gain,
            "image_worse_count": sum(row["gain_sweep"][i]["image_change"] > 0
                                     for row in rows),
            "match_better_count": sum(
                row["gain_sweep"][i]["robust_match_change_px"] < 0
                for row in rows),
            "mean_image_change": float(np.mean([
                row["gain_sweep"][i]["image_change"] for row in rows])),
            "mean_match_change_px": float(np.mean([
                row["gain_sweep"][i]["robust_match_change_px"] for row in rows])),
            "minimum_numeric_q1_corner": min(
                row["gain_sweep"][i]["minimum_normalized_q1_corner_numeric"]
                for row in rows),
        } for i, gain in enumerate(feedback_gains)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "matches", "maps", "report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feedback-maps", type=Path)
    parser.add_argument("--feedback-gains", type=float, nargs="*", default=[])
    parser.add_argument("--teachers", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0 < gain <= 1 for gain in args.feedback_gains):
        raise ValueError("diagnostic feedback gains must lie in (0,1]")
    result = audit(args.root, args.matches, args.maps, args.report,
                   torch.device(args.device), args.feedback_maps,
                   tuple(args.feedback_gains), args.teachers)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "case_count", "opposing_gradient_count", "mean_cosine", "median_cosine")}))


if __name__ == "__main__":
    main()
