"""GPU/CPU throughput and projector-consistency audit for batched P1 symbols."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcopt.forward.p1_symbols import p1_block_symbol_grid
from qcopt.forward.p1_symbols_torch import p1_block_symbol_grid_torch


def run(output_dir: Path, n: int = 2048, repeats: int = 20, device: str | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(selected)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    small_n = 32
    numpy_s, numpy_p = p1_block_symbol_grid(small_n, small_n, face_weights=np.array([1.0, 2.0]))
    torch_s, torch_p = p1_block_symbol_grid_torch(small_n, small_n, face_weights=(1.0, 2.0), device=torch_device)
    parity_s = float(np.max(np.abs(torch_s.detach().cpu().numpy() - numpy_s)))
    parity_p = float(np.max(np.abs(torch_p.detach().cpu().numpy() - numpy_p)))
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    start = time.perf_counter()
    s = p = None
    for _ in range(repeats):
        s, p = p1_block_symbol_grid_torch(n, n, face_weights=(1.0, 2.0), device=torch_device, dtype=torch.complex64)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - start
    assert s is not None and p is not None
    projector_error = torch.max(torch.abs(p @ p - p)).item()
    result = {
        "device": str(torch_device),
        "torch_version": torch.__version__,
        "grid": f"{n}x{n} Fourier modes",
        "modes": int(n * n),
        "repeats": repeats,
        "dtype": "complex64",
        "seconds_total": elapsed,
        "milliseconds_per_batch": 1000.0 * elapsed / repeats,
        "modes_per_second": float(repeats * n * n / elapsed),
        "small_grid_numpy_parity_symbol_max": parity_s,
        "small_grid_numpy_parity_projector_max": parity_p,
        "projector_max_idempotence_error": float(projector_error),
        "finite": bool(torch.all(torch.isfinite(s)) and torch.all(torch.isfinite(p))),
        "scope": "batched P1 two-face block-symbol GPU/CPU implementation",
        "limitation": "periodic regular triangulation only; throughput does not remove physical-boundary or general-mesh accuracy limits",
    }
    (output_dir / "p1_symbol_gpu_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.n, args.repeats, args.device), indent=2))


if __name__ == "__main__":
    main()
