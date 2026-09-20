"""Phase diagram for positive mesh-native conductance cones.

The continuous Beltrami tensor is SPD for every ``|mu|<1``.  A fixed positive
integer stencil is stricter: its cone only contains tensors whose principal
axis directions lie between available mesh directions.  This audit scans
coefficient magnitude and spatial angle variation on a 256² field, while
sampling the local cone at 4,096 sites to keep the active-set enumeration
tractable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qcopt.forward.mmatrix import batch_positive_directional_conductances, integer_wide_stencil_directions


def _tensor_field(n: int, rho: float, angle_amplitude: float) -> np.ndarray:
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    angle = 0.15 + angle_amplitude * (
        np.sin(2.0 * np.pi * xx) + 0.7 * np.cos(2.0 * np.pi * yy)
    )
    mu = rho * np.exp(1j * angle)
    denominator = 1.0 - rho * rho
    tensor = np.empty((n, n, 2, 2), dtype=np.float64)
    tensor[..., 0, 0] = (1.0 - 2.0 * mu.real + rho * rho) / denominator
    tensor[..., 0, 1] = tensor[..., 1, 0] = -2.0 * mu.imag / denominator
    tensor[..., 1, 1] = (1.0 + 2.0 * mu.real + rho * rho) / denominator
    return tensor


def run(
    output_dir: Path,
    n: int = 256,
    stride: int = 4,
    rhos: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 0.9),
    angle_amplitudes: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5),
    max_steps: tuple[int, ...] = (1, 2, 3, 4),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for rho in rhos:
        for angle_amplitude in angle_amplitudes:
            tensor = _tensor_field(n, rho, angle_amplitude)[::stride, ::stride]
            for max_step in max_steps:
                directions = integer_wide_stencil_directions(max_step)
                fit = batch_positive_directional_conductances(tensor, directions)
                residual = fit.residual
                records.append(
                    {
                        "rho": rho,
                        "angle_amplitude": angle_amplitude,
                        "max_step": max_step,
                        "direction_count": int(len(directions)),
                        "sample_count": int(residual.size),
                        "exact_fraction": float(np.mean(residual < 1e-10)),
                        "residual_mean": float(np.mean(residual)),
                        "residual_p95": float(np.quantile(residual, 0.95)),
                        "residual_max": float(np.max(residual)),
                        "positive_weights": bool(np.all(fit.values >= -1e-12)),
                    }
                )
    result = {
        "grid": f"{n}x{n}",
        "stride": stride,
        "rhos": list(rhos),
        "angle_amplitudes": list(angle_amplitudes),
        "max_steps": list(max_steps),
        "records": records,
        "scope": "Beltrami SPD tensor positive-cone phase diagram on a realistic structured field",
        "interpretation": "SPD/ellipticity is separated from exact representation by a fixed positive integer stencil; exact_fraction and residual quantify the mesh-native cone barrier",
        "limitation": "local periodic cone-fit diagnostic; it does not by itself assemble a boundary solver or prove vector-map injectivity",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "mmatrix_cone_phase_diagram_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.stride), indent=2))


if __name__ == "__main__":
    main()
