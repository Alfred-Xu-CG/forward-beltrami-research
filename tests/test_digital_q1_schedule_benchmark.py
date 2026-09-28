"""One complete small-scale schedule VJP with a saved binary certificate."""

from __future__ import annotations

from tools.digital_q1_schedule_benchmark import benchmark_schedule


def test_schedule_benchmark_smoke(tmp_path):
    result = benchmark_schedule(
        schedule="21", final_side=33, device="cpu", repeats=1,
        output_map=tmp_path / "schedule.npz",
    )
    assert result["sides"] == [17, 33]
    assert result["passes_per_stage"] == [4, 1]
    assert result["latent_scalar_count"] > 0
    assert result["all_latent_gradients_finite"]
    assert result["tensor_nonpositive_corners"] == 0
    assert result["saved_binary_valid"]


def test_schedule_benchmark_counts_repeated_seed_passes(tmp_path):
    result = benchmark_schedule(
        schedule="21", final_side=33, seed_repeats=2,
        device="cpu", repeats=1, output_map=tmp_path / "repeated.npz",
    )
    assert result["seed_repeats"] == 2
    assert result["passes_per_stage"] == [8, 1]
    assert result["all_latent_gradients_finite"]
    assert result["saved_binary_valid"]
