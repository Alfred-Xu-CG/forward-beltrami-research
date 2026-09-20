import json

from qcopt.experiments.benchmark_lsqc import run_lsqc_benchmark


def test_lsqc_backend_benchmark_writes_reproducible_metrics(tmp_path):
    report = run_lsqc_benchmark(tmp_path, grid_sizes=(3,), repeats=1, seed=211)

    assert report["seed"] == 211
    assert report["repeats"] == 1
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    required = {
        "n_vertices",
        "n_faces",
        "augmented_system_size",
        "fast_system_size",
        "augmented_system_nnz",
        "fast_system_nnz",
        "augmented_factor_nnz",
        "fast_factor_nnz",
        "augmented_forward_seconds",
        "fast_forward_seconds",
        "augmented_backward_seconds",
        "fast_backward_seconds",
        "augmented_adjoint_solve_seconds",
        "fast_adjoint_solve_seconds",
        "augmented_contraction_seconds",
        "fast_contraction_seconds",
        "augmented_training_step_seconds",
        "fast_training_step_seconds",
        "forward_speedup",
        "backward_speedup",
        "training_step_speedup",
        "max_map_difference",
        "max_gradient_difference",
    }
    assert required <= case.keys()
    assert case["fast_system_size"] < case["augmented_system_size"]
    assert case["fast_system_nnz"] < case["augmented_system_nnz"]
    assert case["fast_factor_nnz"] < case["augmented_factor_nnz"]
    assert case["augmented_adjoint_solve_seconds"] > 0.0
    assert case["fast_adjoint_solve_seconds"] > 0.0
    assert case["augmented_contraction_seconds"] > 0.0
    assert case["fast_contraction_seconds"] > 0.0
    assert case["augmented_training_step_seconds"] > 0.0
    assert case["fast_training_step_seconds"] > 0.0
    assert case["max_map_difference"] < 1e-9
    assert case["max_gradient_difference"] < 1e-8

    saved = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert saved == report
    assert (tmp_path / "benchmark.csv").is_file()
