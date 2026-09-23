"""Measure Route C's returned float32 residual separately from its float64 PCG residual."""

from __future__ import annotations

import argparse
import json

import torch

from phase6_train_multisample_image import make_dataset
from qcopt.neural_bijection.dense import PhotometricSpectralTutteLayer
from qcopt.neural_bijection.dense.sine_pcg_tutte import (
    _conductances, _laplacian, _minimum_signed_area_ratio,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--side", type=int, default=1025)
    parser.add_argument("--fit-side", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1170031)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    frequencies = tuple(int(item) for item in state["args"]["frequencies"].split(","))
    layer = PhotometricSpectralTutteLayer(
        args.side, fit_side=args.fit_side, frequencies=frequencies).to(device)
    with torch.no_grad():
        layer.raw_mode_gains.copy_(state["raw_gains"].to(device))
        layer.prepare(device=device)
        fixed, moving, *_ = (
            value.to(device) for value in make_dataset(
                1, 512, args.seed, target_family="high64",
                return_coefficients=True))
        latent = layer._latent(fixed, moving)
        bases = (layer._horizontal_basis, layer._vertical_basis,
                 layer._diagonal_basis)
        logits = tuple(-0.8 + torch.einsum("bk,k...->b...", latent, basis)
                       for basis in bases)
        internal = layer.solver(*logits)
        returned = internal.float()
        conductances = _conductances(
            logits, layer.solver.minimum_conductance,
            layer.solver.maximum_conductance)
        source = layer.solver._source[None]
        rhs = -_laplacian(source, *conductances)[:, 1:-1, 1:-1]
        denominator = torch.linalg.vector_norm(rhs)

        def residual(mapped: torch.Tensor) -> float:
            interior = _laplacian(mapped, *conductances)[:, 1:-1, 1:-1]
            return float(torch.linalg.vector_norm(interior) / denominator)

        edge = torch.zeros_like(returned, dtype=torch.bool)
        edge[:, 0] = edge[:, -1] = True
        edge[:, :, 0] = edge[:, :, -1] = True
        output = {
            "side": args.side, "control_vertices": args.side**2,
            "control_faces": 2*(args.side-1)**2,
            "seed": args.seed, "fit_side": args.fit_side,
            "frequencies": frequencies, "checkpoint": args.checkpoint,
            "device": str(device),
            "solver_logged_float64_residual":
                layer.solver.last_forward_stats["true_relative_residual"],
            "independently_recomputed_float64_residual": residual(internal),
            "returned_float32_castback_residual": residual(returned.double()),
            "maximum_cast_coordinate_difference":
                float((internal-returned.double()).abs().amax()),
            "returned_float32_boundary_difference":
                float((returned-source.float())[edge].abs().amax()),
            "minimum_float64_face_area_ratio":
                _minimum_signed_area_ratio(internal),
            "minimum_returned_float32_face_area_ratio":
                _minimum_signed_area_ratio(returned),
        }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
