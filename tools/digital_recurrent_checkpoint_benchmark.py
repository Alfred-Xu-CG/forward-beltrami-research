"""Matched 257² recurrent image-network memory/VJP checkpoint comparison."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from tools.digital_acrobat_recurrent_probe import _load_model
from tools.digital_acrobat_teacher_probe import load_case_inputs


def benchmark(root: Path, checkpoint_dir: Path, output: Path, *, case: int,
              batch: int, device: str, repeats: int) -> dict:
    if output.exists() or batch < 1 or repeats < 1:
        raise ValueError("fresh output and positive batch/repeats required")
    target = torch.device(device)
    model = _load_model(checkpoint_dir, device).train()
    manifest = json.loads((checkpoint_dir / "train_manifest.json").read_text())
    for side in manifest["initialized_head_sides"]:
        for parameter in model.heads[str(side)].parameters():
            parameter.requires_grad_(False)
    example = load_case_inputs(root, case, target)
    fixed = example["fixed"].repeat(batch, 1, 1, 1)
    moving = example["prewarped"].repeat(batch, 1, 1, 1)

    def sync() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    modes = {}
    for label, checkpointed in (("normal", False), ("round_checkpoint", True)):
        model.checkpoint_rounds = checkpointed
        timings = []
        forward_times = []
        saved = None
        gradients = None
        peak = None
        for trial in range(repeats + 1):
            model.zero_grad(set_to_none=True)
            sync()
            if target.type == "cuda":
                torch.cuda.reset_peak_memory_stats(target)
            started = time.perf_counter()
            mapped = model(fixed, moving)[0]
            sync()
            forward_time = time.perf_counter() - started
            mapped.square().mean().backward()
            sync()
            total_time = time.perf_counter() - started
            if trial == 0:
                continue
            timings.append(total_time)
            forward_times.append(forward_time)
            allocation = (torch.cuda.max_memory_allocated(target)
                          if target.type == "cuda" else None)
            peak = allocation if peak is None else max(peak, allocation)
            if saved is None:
                saved = mapped.detach().cpu().numpy().copy()
                gradients = {
                    key: value.grad.detach().cpu().numpy().copy()
                    for key, value in model.heads["257"].named_parameters()
                }
        modes[label] = {
            "median_forward_seconds": statistics.median(forward_times),
            "median_forward_plus_vjp_seconds": statistics.median(timings),
            "peak_torch_cuda_allocated_bytes": peak,
            "map": saved, "head_gradients": gradients,
        }
    first, second = modes["normal"], modes["round_checkpoint"]
    map_equal = bool(np.array_equal(first["map"], second["map"]))
    gradient_max_abs_difference = max(
        float(np.max(np.abs(first["head_gradients"][key]
                            - second["head_gradients"][key])))
        for key in first["head_gradients"]
    )
    report = {
        "question": "does round checkpointing reduce 257-grid image-network VJP memory without changing its result",
        "case": case, "batch": batch, "device": device, "repeats_after_warmup": repeats,
        "control_vertices_per_case": 257**2, "image_side": 512,
        "rounds_by_side": manifest["rounds_by_side"],
        "loss": "mean squared output coordinate; first-order head-parameter VJP",
        "map_bitwise_equal": map_equal,
        "maximum_head_gradient_absolute_difference": gradient_max_abs_difference,
        "arms": {label: {key: value for key, value in item.items()
                         if key not in ("map", "head_gradients")}
                 for label, item in modes.items()},
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.root, args.checkpoint_dir, args.output,
                               case=args.case, batch=args.batch,
                               device=args.device, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
