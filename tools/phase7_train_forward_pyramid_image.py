"""Real image-only training through the fixed-grid forward P1 pyramid.

The known analytic target map is used only for post-training evaluation. The
training loss sees fixed/moving image intensities and the decoded warp.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase6_train_multisample_image import make_dataset
from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.dense import ForwardP1ImageEncoder, ForwardP1Pyramid
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def minimum_jacobian(mapped: torch.Tensor) -> float:
    a, b = mapped[:, :-1, :-1], mapped[:, :-1, 1:]
    c, d = mapped[:, 1:, 1:], mapped[:, 1:, :-1]
    def cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return float(torch.minimum(cross(b - a, c - a).amin(), cross(c - a, d - a).amin())
                 * (mapped.shape[1] - 1) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=257)
    parser.add_argument("--image-side", type=int, default=512)
    parser.add_argument("--target-family", choices=("base", "high32", "high64"), default="high32")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--seed-passes", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--training-objective", choices=("image", "map"), default="image")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--load-state", default=None)
    args = parser.parse_args()
    if min(args.side, args.image_side, args.train_count, args.test_count,
           args.width, args.batch, args.steps) < 1:
        raise ValueError("dimensions, counts, batch and steps must be positive")
    torch.manual_seed(20260924)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    train = tuple(t.to(device) for t in make_dataset(
        args.train_count, args.image_side, 55101, target_family=args.target_family,
    ))
    test = tuple(t.to(device) for t in make_dataset(
        args.test_count, args.image_side, 99317, target_family=args.target_family,
    ))
    decoder = ForwardP1Pyramid(5, args.side, seed_passes=args.seed_passes,
                               minimum_jacobian=0.05).to(device)
    encoder = ForwardP1ImageEncoder(
        5, decoder.level_sides, seed_passes=args.seed_passes,
        feature_side=min(args.side, 257), width=args.width,
    ).to(device)
    if args.load_state is not None:
        saved = torch.load(args.load_state, map_location=device, weights_only=True)
        encoder.load_state_dict(saved["encoder"])
    table = StructuredDenseQueryTable.from_mesh(
        structured_rectangle(args.side - 1, args.side - 1),
        height=args.image_side, width=args.image_side,
    )
    table.prepare(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(6019)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward(fixed: torch.Tensor, moving: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seed, levels = encoder(fixed, moving)
        control = decoder(seed, levels)
        query = table.interpolate(control.reshape(control.shape[0], -1, 2))
        warped = F.grid_sample(moving, 2 * query - 1,
                               mode="bilinear", padding_mode="border", align_corners=True)
        image_loss = (warped - fixed).square().mean()
        return image_loss, control, query

    @torch.no_grad()
    def evaluate(dataset: tuple[torch.Tensor, ...]) -> dict[str, float]:
        image_errors, map_errors, margins = [], [], []
        for start in range(0, dataset[0].shape[0], args.batch):
            fixed, moving, true_map = (t[start:start + args.batch] for t in dataset)
            loss, control, query = forward(fixed, moving)
            image_errors.append(float(loss) * fixed.shape[0])
            map_errors.append(float((query.reshape_as(true_map) - true_map).square().sum(dim=-1).mean()) * fixed.shape[0])
            margins.append(minimum_jacobian(control))
        count = dataset[0].shape[0]
        return {
            "image_mse": sum(image_errors) / count,
            "query_map_vector_rmse": math.sqrt(sum(map_errors) / count),
            "minimum_jacobian": min(margins),
        }

    encoder.eval()
    initial = evaluate(test)
    encoder.train()
    history = []
    times = []
    sync()
    began = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        idx = torch.randint(args.train_count, (args.batch,), generator=generator).to(device)
        fixed, moving, true_map = (tensor[idx] for tensor in train)
        optimizer.zero_grad(set_to_none=True)
        sync()
        tick = time.perf_counter()
        image_loss, control, query = forward(fixed, moving)
        loss = (
            image_loss if args.training_objective == "image"
            else (query.reshape_as(true_map) - true_map).square().sum(dim=-1).mean()
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"nonfinite training loss at step {step}")
        loss.backward()
        optimizer.step()
        sync()
        times.append(time.perf_counter() - tick)
        with torch.no_grad():
            margin = minimum_jacobian(control)
        if margin <= 0 or not math.isfinite(margin):
            raise RuntimeError(f"P1 face failure at training step {step}: Jmin={margin}")
        if step in {1, args.steps} or step % max(1, args.steps // 10) == 0:
            history.append({"step": step, "train_batch_objective": float(loss.detach()),
                            "train_batch_image_mse": float(image_loss.detach()),
                            "train_batch_minimum_jacobian": margin})
    sync()
    train_seconds = time.perf_counter() - began
    allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    encoder.eval()
    final_train = evaluate(train)
    final_test = evaluate(test)
    if args.save_state:
        torch.save({"encoder": encoder.state_dict(), "config": vars(args)}, args.save_state)
    print(json.dumps({
        "method": "phase7_forward_p1_image_encoder",
        "training_objective": "image_only_pixel_MSE" if args.training_objective == "image" else "target_query_map_vector_MSE",
        "target_map_use": "evaluation_only" if args.training_objective == "image" else "training_supervision_and_evaluation",
        "target_family": args.target_family,
        "side": args.side,
        "control_vertices": args.side ** 2,
        "control_faces": 2 * (args.side - 1) ** 2,
        "image_side": args.image_side,
        "image_queries": args.image_side ** 2,
        "batch": args.batch,
        "train_count": args.train_count,
        "test_count": args.test_count,
        "train_seed": 55101,
        "test_seed": 99317,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "seed_passes": args.seed_passes,
        "width": args.width,
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "device": str(device),
        "torch_version": torch.__version__,
        "initial_test": initial,
        "final_train": final_train,
        "final_test": final_test,
        "history": history,
        "train_seconds": train_seconds,
        "median_training_step_seconds": statistics.median(times),
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "checkpoint_path": args.save_state,
        "loaded_checkpoint_path": args.load_state,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
