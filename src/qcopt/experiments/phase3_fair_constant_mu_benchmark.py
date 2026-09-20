"""Same-target CPU control for the MBM and directed-Tutte prototypes.

The target is deliberately simple: a constant real Beltrami coefficient
``a=0.3`` on the unit square has the exact map
``(x,y) -> (x, ((1-a)/(1+a))*y)``.  Each candidate is run in a separate
process (invoke this module once per candidate) so the reported RSS delta is
not contaminated by the other solver's imports or allocations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import psutil
import torch

from qcopt.forward.tutte_directed_implicit import (
    DirectedTutteSystem,
    directed_tutte_embedding_torch_implicit,
)
from qcopt.forward.mbm_lbs_implicit import mbm_lbs_torch_implicit
from qcopt.mesh import structured_rectangle


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.uname().processor


def _target_grid(n: int, a: float) -> np.ndarray:
    s = (1.0 - a) / (1.0 + a)
    yy, xx = np.mgrid[0:n, 0:n]
    return np.stack((xx / (n - 1), s * yy / (n - 1)), axis=-1)


def _minimum_face_ratio(vertices: np.ndarray, faces: np.ndarray, target_det: float) -> float:
    p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
    signed = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    return float(np.min(signed) / target_det)


def _run_mbm(n: int, a: float) -> dict[str, object]:
    real = torch.full((n, n), a, dtype=torch.float64, requires_grad=True)
    imag = torch.zeros((n, n), dtype=torch.float64, requires_grad=True)
    target = torch.as_tensor(_target_grid(n, a), dtype=torch.float64)
    process = psutil.Process()
    gc.collect()
    before = process.memory_info().rss
    started = time.perf_counter()
    mapped = mbm_lbs_torch_implicit(real, imag)
    forward = time.perf_counter() - started
    loss = torch.mean((mapped - target) ** 2)
    started = time.perf_counter()
    loss.backward()
    backward = time.perf_counter() - started
    after = process.memory_info().rss
    mesh = structured_rectangle(n - 1, n - 1)
    output = mapped.detach().cpu().numpy().reshape(-1, 2)
    # _minimum_face_ratio uses the signed double area, so the reference is
    # 2*(triangle area) = hx*hy*s for either right-triangle split.
    target_det = (1.0 / (n - 1)) * ((1.0 - a) / (1.0 + a) / (n - 1))
    return {
        "candidate": "MBM_implicit",
        "vertices_per_axis": n,
        "cells_per_axis": n - 1,
        "a": a,
        "forward_seconds": forward,
        "backward_seconds": backward,
        "rss_delta_MB": (after - before) / (1024.0 * 1024.0),
        "map_rmse": float(torch.sqrt(loss).detach().cpu()),
        "min_face_area_ratio": _minimum_face_ratio(output, mesh.faces, target_det),
        "finite_output": bool(torch.isfinite(mapped).all()),
        "finite_gradient": bool(torch.isfinite(real.grad).all() and torch.isfinite(imag.grad).all()),
    }


def _run_tutte(cells: int, a: float) -> dict[str, object]:
    mesh = structured_rectangle(cells, cells)
    system = DirectedTutteSystem.from_mesh(mesh)
    target_np = mesh.vertices.copy()
    target_np[:, 1] *= (1.0 - a) / (1.0 + a)
    boundary = torch.as_tensor(target_np[mesh.boundary_loops[0]], dtype=torch.float64)
    logits = torch.zeros((system.n_rows, system.max_degree), dtype=torch.float64, requires_grad=True)
    target = torch.as_tensor(target_np, dtype=torch.float64)
    process = psutil.Process()
    gc.collect()
    before = process.memory_info().rss
    started = time.perf_counter()
    mapped = directed_tutte_embedding_torch_implicit(mesh, boundary, logits, system)
    forward = time.perf_counter() - started
    loss = torch.mean((mapped - target) ** 2)
    started = time.perf_counter()
    loss.backward()
    backward = time.perf_counter() - started
    after = process.memory_info().rss
    target_det = (1.0 / cells) * ((1.0 - a) / (1.0 + a) / cells)
    return {
        "candidate": "directed_Tutte_zero_logits",
        "vertices_per_axis": cells + 1,
        "cells_per_axis": cells,
        "a": a,
        "forward_seconds": forward,
        "backward_seconds": backward,
        "rss_delta_MB": (after - before) / (1024.0 * 1024.0),
        "map_rmse": float(torch.sqrt(loss).detach().cpu()),
        "min_face_area_ratio": _minimum_face_ratio(mapped.detach().cpu().numpy(), mesh.faces, target_det),
        "finite_output": bool(torch.isfinite(mapped).all()),
        "finite_gradient": bool(torch.isfinite(logits.grad).all()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=("mbm", "tutte"), required=True)
    ap.add_argument("--vertices", type=int, default=129)
    ap.add_argument("--a", type=float, default=0.3)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if args.vertices < 3:
        raise ValueError("vertices must be at least 3")
    result = _run_mbm(args.vertices, args.a) if args.candidate == "mbm" else _run_tutte(args.vertices - 1, args.a)
    result.update({"cpu": _cpu_name(), "omp_num_threads": os.environ["OMP_NUM_THREADS"], "torch": torch.__version__, "numpy": np.__version__, "seed": 0, "target": "constant real mu exact affine map"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
