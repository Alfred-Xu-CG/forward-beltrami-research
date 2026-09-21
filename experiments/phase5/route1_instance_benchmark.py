"""Run one compact, reproducible Route-I single-instance benchmark matrix.

The script deliberately builds one CPU float64 target and passes that same
object to every requested backend.  Target generation is outside training
timings and is reported once.  Each backend run includes decoder setup,
forward/backward, dense warp, optimizer work, and the independent per-evaluation
P1 audit.  GPU memory is a synchronized PyTorch allocator peak; Linux RSS/HWM
is a process measurement and is most interpretable when one backend is run in
the fresh process.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
from time import perf_counter
from typing import Any

import numpy as np
import scipy
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from qcopt.mesh import structured_rectangle  # noqa: E402
from qcopt.neural_bijection.tutte.instance_optimization import (  # noqa: E402
    build_directed_target,
    run_image_instance,
    run_supervised_instance,
)


def _parse_csv(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("comma-separated list must be nonempty")
    return result


def _linux_memory_bytes() -> dict[str, int | None]:
    status = Path("/proc/self/status")
    result: dict[str, int | None] = {"rss_bytes": None, "hwm_bytes": None}
    if not status.exists():
        return result
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            result["rss_bytes"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            result["hwm_bytes"] = int(line.split()[1]) * 1024
    return result


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return result


def _run_one(
    args: argparse.Namespace,
    *,
    backend: str,
    target,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if backend == "direct" and device.type != "cpu":
        return {"backend_requested": backend, "status": "not_applicable", "reason": "CPU-only"}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        gpu_baseline_allocated = torch.cuda.memory_allocated(device)
        gpu_baseline_reserved = torch.cuda.memory_reserved(device)
    else:
        gpu_baseline_allocated = gpu_baseline_reserved = None
    memory_before = _linux_memory_bytes()
    wrapper_start = perf_counter()
    try:
        common = dict(
            control_vertices=args.control_side,
            image_resolution=args.image_resolution,
            steps=args.steps,
            backend=backend,
            optimizer_name=args.optimizer,
            dtype=dtype,
            seed=args.seed,
            target_strength=args.target_strength,
            learning_rate=args.learning_rate,
            device=device,
            target=target,
            objective_threshold=args.objective_threshold,
        )
        if args.task == "supervised_map":
            result = run_supervised_instance(**common)
        else:
            result = run_image_instance(**common, image_name=args.image_name)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wrapper_seconds = perf_counter() - wrapper_start
        memory_after = _linux_memory_bytes()
        row = asdict(result)
        row.update(
            {
                "status": "success",
                "backend_requested": backend,
                "wrapper_wall_seconds": wrapper_seconds,
                "rss_before_bytes": memory_before["rss_bytes"],
                "rss_after_bytes": memory_after["rss_bytes"],
                "process_hwm_bytes": memory_after["hwm_bytes"],
                "peak_gpu_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
                "peak_gpu_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
                ),
                "gpu_baseline_allocated_bytes": gpu_baseline_allocated,
                "gpu_baseline_reserved_bytes": gpu_baseline_reserved,
                "threshold_semantics": "first independently audited evaluation; LBFGS trials included",
                "target_object_identity": id(target),
            }
        )
        if not args.include_trace:
            row.pop("trace", None)
        return row
    except Exception as exc:  # failures are benchmark observations, not deleted rows
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "backend_requested": backend,
            "status": "failure",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "wrapper_wall_seconds": perf_counter() - wrapper_start,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("supervised_map", "image_registration"), required=True)
    parser.add_argument("--backends", type=_parse_csv, required=True)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), required=True)
    parser.add_argument("--control-side", type=int, required=True)
    parser.add_argument("--image-resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--objective-threshold", type=float)
    parser.add_argument("--target-strength", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--image-name", default="medical_phantom")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.control_side < 3 or args.image_resolution < 2 or args.steps < 1:
        parser.error("control-side>=3, image-resolution>=2, and steps>=1 are required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mesh = structured_rectangle(args.control_side - 1, args.control_side - 1)
    target_start = perf_counter()
    target = build_directed_target(
        mesh,
        image_height=args.image_resolution,
        image_width=args.image_resolution,
        seed=args.seed,
        strength=args.target_strength,
        height=0.9 if args.task == "supervised_map" else 1.0,
    )
    target_setup_seconds = perf_counter() - target_start
    receipt = {
        "schema": "phase5_route1_instance_v1",
        "research_question": "I1 supervised fitting" if args.task == "supervised_map" else "I2 image registration",
        "configuration": {
            "task": args.task,
            "control_vertices_per_side": args.control_side,
            "control_vertex_count": mesh.n_vertices,
            "face_count": len(mesh.faces),
            "image_resolution": args.image_resolution,
            "optimizer": args.optimizer,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "objective_threshold": args.objective_threshold,
            "target_strength": args.target_strength,
            "seed": args.seed,
            "image_name": args.image_name if args.task == "image_registration" else None,
            "dtype": args.dtype,
            "device": str(device),
            "backends": args.backends,
            "warp_convention": "fixed-to-moving backward coordinates; no inverse is computed",
        },
        "environment": _environment(device),
        "shared_target": {
            "generated_once": True,
            "setup_primal_solves": 1,
            "setup_seconds": target_setup_seconds,
            "metrics": asdict(target.metrics),
            "object_identity": id(target),
        },
        "runs": [
            _run_one(args, backend=backend, target=target, device=device, dtype=dtype)
            for backend in args.backends
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_clean_json(receipt), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
