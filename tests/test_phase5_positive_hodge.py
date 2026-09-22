from __future__ import annotations

import importlib.util
from itertools import permutations
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

# Match fresh-process production import order on the supported Windows host.
np.linalg.cond(np.eye(1, dtype=np.float64))

import torch

import qcopt.neural_bijection.tutte.positive_hodge as positive_hodge_module
from qcopt.neural_bijection.metrics import compute_p1_map_metrics
from qcopt.neural_bijection.tutte.positive_hodge import (
    LearnedPositiveDirectionMap,
    PositiveHodgeTutteLayer,
    aggregate_face_direction_conductances,
    audit_positive_planar_graph,
    beltrami_tensor,
    build_center_split_square_graph,
    build_standard_square_graph,
    build_stellar_square_graph,
    direction_design_matrix,
    edge_conductance_gradient,
    fit_direction_tensor_nnls,
    inverse_softplus_conductances,
)


REPOSITORY = Path(__file__).resolve().parents[1]
COVERAGE_SCRIPT = REPOSITORY / "experiments/phase5/route3_positive_hodge_coverage.py"
INSTANCE_SCRIPT = REPOSITORY / "experiments/phase5/route3_positive_hodge_instance_benchmark.py"


def _coverage_module():
    assert COVERAGE_SCRIPT.is_file(), "positive-Hodge coverage experiment is missing"
    spec = importlib.util.spec_from_file_location("phase5_positive_hodge_coverage", COVERAGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _instance_module():
    assert INSTANCE_SCRIPT.is_file(), "positive-Hodge instance benchmark is missing"
    spec = importlib.util.spec_from_file_location("phase5_positive_hodge_instance", INSTANCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _degrees(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(values)


@pytest.mark.parametrize(
    ("builder", "expected_degrees", "vertices", "faces", "dividing_edges"),
    [
        (build_standard_square_graph, [0.0, 45.0, 90.0], 9, 8, 2),
        (build_center_split_square_graph, [0.0, 45.0, 90.0, 135.0], 13, 16, 0),
        (
            build_stellar_square_graph,
            [0.0, math.degrees(math.atan(0.5)), 45.0, math.degrees(math.atan(2.0)), 90.0, 135.0],
            17,
            24,
            2,
        ),
    ],
)
def test_three_explicit_planar_graphs_have_declared_directions_and_disk_certificate(
    builder, expected_degrees, vertices, faces, dividing_edges
) -> None:
    graph = builder(2)
    assert graph.mesh.n_vertices == vertices
    assert graph.mesh.n_faces == faces
    np.testing.assert_allclose(_degrees(graph.direction_angles), expected_degrees, atol=1.0e-12)

    audit = audit_positive_planar_graph(graph)
    assert audit.proper_crossing_count == 0
    assert audit.collinear_overlap_count == 0
    assert audit.boundary_loop_count == 1
    assert audit.euler_characteristic == 1
    assert audit.disk_topology
    # The fixed SW--NE diagonal creates two Floater dividing edges.  They are
    # allowed because their target chords lie inside, rather than along, the
    # rectangular boundary; the hard solver checks that condition per decode.
    assert audit.dividing_edge_count == dividing_edges


def test_beltrami_tensor_is_spd_unit_determinant_and_has_known_axis_case() -> None:
    mu = np.asarray([0.3 + 0.0j, 0.2j, 0.8 * np.exp(1j * 0.7)])
    tensors = beltrami_tensor(mu)
    assert tensors.shape == (3, 2, 2)
    np.testing.assert_allclose(tensors, np.swapaxes(tensors, -1, -2))
    np.testing.assert_allclose(np.linalg.det(tensors), 1.0, rtol=2.0e-14, atol=2.0e-14)
    assert np.all(np.linalg.eigvalsh(tensors) > 0.0)
    expected = np.diag([(1.0 - 0.3) / (1.0 + 0.3), (1.0 + 0.3) / (1.0 - 0.3)])
    np.testing.assert_allclose(tensors[0], expected, rtol=2.0e-14, atol=2.0e-14)

    with pytest.raises(ValueError, match=r"strictly inside"):
        beltrami_tensor(1.0 + 0.0j)


def test_frobenius_direction_design_and_strict_nnls_floor() -> None:
    standard = build_standard_square_graph(2)
    center = build_center_split_square_graph(2)
    identity = np.eye(2)
    floor = 1.0e-4
    excess = 1.0e-8

    standard_fit = fit_direction_tensor_nnls(
        identity,
        standard.direction_angles,
        minimum_conductance=floor,
        minimum_excess=excess,
    )
    center_fit = fit_direction_tensor_nnls(
        identity,
        center.direction_angles,
        minimum_conductance=floor,
        minimum_excess=excess,
    )
    assert np.all(standard_fit.conductances >= floor + excess)
    assert np.all(np.isfinite(standard_fit.logits))
    # Isotropy is on the boundary of the {0,45,90} cone.  A strict positive
    # floor therefore has a real, nonzero approximation cost.
    assert standard_fit.frobenius_relative_error > 1.0e-9
    # Adding the opposite diagonal makes isotropy an interior point.
    assert center_fit.frobenius_relative_error < 1.0e-12

    design = direction_design_matrix(center.direction_angles)
    vector = design @ center_fit.conductances
    expected_vector = np.asarray(
        [center_fit.fitted_tensor[0, 0], math.sqrt(2.0) * center_fit.fitted_tensor[0, 1], center_fit.fitted_tensor[1, 1]]
    )
    np.testing.assert_allclose(vector, expected_vector, atol=1.0e-14)

    recovered = floor + torch.nn.functional.softplus(
        inverse_softplus_conductances(
            torch.tensor(center_fit.conductances.copy(), dtype=torch.float64), floor
        )
    )
    torch.testing.assert_close(
        recovered,
        torch.tensor(center_fit.conductances.copy(), dtype=torch.float64),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_redundant_direction_nnls_uses_unique_minimum_norm_tie_break() -> None:
    """An exact tensor fit must not depend on the platform NNLS active set."""

    directions = build_center_split_square_graph(2).direction_angles
    expected = np.full(4, 0.5, dtype=np.float64)
    canonical = fit_direction_tensor_nnls(np.eye(2), directions)
    np.testing.assert_allclose(canonical.conductances, expected, rtol=0.0, atol=2.0e-12)

    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    permuted = fit_direction_tensor_nnls(np.eye(2), directions[permutation])
    restored = np.empty_like(permuted.conductances)
    restored[permutation] = permuted.conductances
    np.testing.assert_allclose(restored, expected, rtol=0.0, atol=2.0e-12)


def test_canonical_nnls_resolves_near_tied_supports_permutation_equivariantly() -> None:
    directions = build_center_split_square_graph(2).direction_angles
    lower = 1.0e-6 + 1.0e-10
    perturbation = 1.0e-7
    first_diagonal = math.sqrt(3.0 + 4.0 * (lower + perturbation) ** 2) - 2.0 * (
        lower + perturbation
    )
    second_diagonal = 1.0 / first_diagonal
    tensor = np.diag([first_diagonal, second_diagonal])
    assert np.linalg.det(tensor) == pytest.approx(1.0, rel=0.0, abs=2.0e-15)
    shared_diagonal = (first_diagonal + second_diagonal) / 4.0
    expected = np.asarray(
        [
            first_diagonal - shared_diagonal,
            shared_diagonal,
            second_diagonal - shared_diagonal,
            shared_diagonal,
        ],
        dtype=np.float64,
    )
    canonical = fit_direction_tensor_nnls(tensor, directions)
    np.testing.assert_allclose(canonical.conductances, expected, rtol=0.0, atol=2.0e-12)

    for permutation_tuple in permutations(range(4)):
        permutation = np.asarray(permutation_tuple, dtype=np.int64)
        permuted = fit_direction_tensor_nnls(tensor, directions[permutation])
        restored = np.empty_like(permuted.conductances)
        restored[permutation] = permuted.conductances
        np.testing.assert_allclose(restored, expected, rtol=0.0, atol=2.0e-12)
        np.testing.assert_allclose(
            restored,
            canonical.conductances,
            rtol=0.0,
            atol=2.0e-13,
        )


def test_canonical_nnls_rejects_a_non_unit_trace_dictionary() -> None:
    design = np.asarray(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="unit-direction trace identity"):
        positive_hodge_module._verified_nnls(
            design,
            np.asarray([1.0, 0.0, 1.0], dtype=np.float64),
            conductance_offset=np.full(2, 1.0e-6, dtype=np.float64),
        )


def test_stellar_nnls_cycling_phase_has_verified_active_set_fallback() -> None:
    graph = build_stellar_square_graph(2)
    phase = 5.342014692511909
    tensor = beltrami_tensor(0.9 * np.exp(1j * phase))
    fit = fit_direction_tensor_nnls(
        tensor,
        graph.direction_angles,
        minimum_conductance=1.0e-8,
        minimum_excess=1.0e-12,
    )
    assert np.all(np.isfinite(fit.conductances))
    assert np.all(fit.conductances >= 1.0e-8 + 1.0e-12)
    design = direction_design_matrix(graph.direction_angles)
    target_vector = np.asarray(
        [tensor[0, 0], math.sqrt(2.0) * tensor[0, 1], tensor[1, 1]]
    )
    lower = 1.0e-8 + 1.0e-12
    free = fit.conductances - lower
    gradient = design.T @ (design @ free - (target_vector - design @ np.full(len(free), lower)))
    tolerance = 2.0e-10
    assert np.min(gradient) >= -tolerance
    assert np.max(np.abs(gradient[free > tolerance])) < tolerance


def test_learned_local_map_is_deterministic_strict_positive_and_differentiable() -> None:
    directions = build_center_split_square_graph(2).direction_angles
    first = LearnedPositiveDirectionMap(directions, hidden_features=12, seed=901)
    second = LearnedPositiveDirectionMap(directions, hidden_features=12, seed=901)
    target = torch.as_tensor(
        beltrami_tensor(np.asarray([0.2 + 0.1j, -0.4j])), dtype=torch.float64
    ).requires_grad_(True)
    first = first.double()
    second = second.double()
    conductances = first(target)
    torch.testing.assert_close(conductances, second(target.detach()))
    assert bool(torch.all(conductances > first.minimum_conductance))
    fitted = first.fitted_tensors(conductances)
    loss = (fitted - target).square().mean()
    loss.backward()
    assert target.grad is not None and bool(torch.isfinite(target.grad).all())
    assert all(parameter.grad is not None for parameter in first.parameters())


def test_face_coefficients_aggregate_to_active_edges_without_losing_positivity() -> None:
    graph = build_center_split_square_graph(2)
    values = np.arange(1, graph.mesh.n_faces * len(graph.direction_angles) + 1, dtype=np.float64)
    values = values.reshape(graph.mesh.n_faces, len(graph.direction_angles))
    edges = aggregate_face_direction_conductances(graph, values)
    assert edges.shape == (len(graph.active_edges),)
    assert np.all(edges > 0.0)
    for edge_position, (face0, face1) in enumerate(graph.active_edge_faces):
        direction = graph.active_edge_direction_indices[edge_position]
        samples = [values[face0, direction]]
        if face1 >= 0:
            samples.append(values[face1, direction])
        assert edges[edge_position] == pytest.approx(np.mean(samples))


def test_positive_hodge_wrapper_returns_only_certified_decoder_output_and_backpropagates() -> None:
    graph = build_standard_square_graph(3)
    projector = LearnedPositiveDirectionMap(
        graph.direction_angles,
        hidden_features=10,
        minimum_conductance=1.0e-4,
        seed=77,
    ).double()
    layer = PositiveHodgeTutteLayer(
        graph,
        projector,
        minimum_conductance=1.0e-4,
        relative_tolerance=1.0e-12,
    ).double()
    tensors = torch.as_tensor(
        np.repeat(np.eye(2)[None, :, :], graph.mesh.n_faces, axis=0),
        dtype=torch.float64,
    ).requires_grad_(True)
    loop = layer.system.loop
    boundary = torch.as_tensor(graph.mesh.vertices[loop], dtype=torch.float64)
    mapped = layer(tensors, boundary)
    metrics = compute_p1_map_metrics(graph.mesh, mapped.detach().numpy())
    assert metrics.global_injectivity_certificate
    assert metrics.flip_count == 0
    assert layer.last_topology_certified is True
    assert layer.last_edge_conductances is not None
    assert bool(torch.all(layer.last_edge_conductances > 0.0))

    mapped.square().mean().backward()
    assert tensors.grad is not None and bool(torch.isfinite(tensors.grad).all())
    assert layer.solver.last_adjoint_stats is not None


def _dense_dirichlet_solve(layer, conductances: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    interior = layer.interior_vertices
    loop = layer.boundary_vertices
    interior_index = {int(vertex): row for row, vertex in enumerate(interior)}
    boundary_index = {int(vertex): row for row, vertex in enumerate(loop)}
    matrix = np.zeros((len(interior), len(interior)), dtype=np.float64)
    rhs = np.zeros((len(interior), 2), dtype=np.float64)
    for value, (first, second) in zip(conductances, layer.active_edges, strict=True):
        first_i = interior_index.get(int(first))
        second_i = interior_index.get(int(second))
        if first_i is not None and second_i is not None:
            matrix[first_i, first_i] += value
            matrix[second_i, second_i] += value
            matrix[first_i, second_i] -= value
            matrix[second_i, first_i] -= value
        else:
            i, b = (first_i, int(second)) if first_i is not None else (second_i, int(first))
            assert i is not None
            matrix[i, i] += value
            rhs[i] += value * boundary[boundary_index[b]]
    interior_values = np.linalg.solve(matrix, rhs)
    result = np.empty((layer.system.n_vertices, 2), dtype=np.float64)
    result[loop] = boundary
    result[interior] = interior_values
    return result


def test_implicit_edge_gradient_identity_matches_finite_difference() -> None:
    graph = build_standard_square_graph(3)
    projector = LearnedPositiveDirectionMap(graph.direction_angles, seed=8).double()
    wrapper = PositiveHodgeTutteLayer(graph, projector).double()
    solver = wrapper.solver
    rng = np.random.default_rng(92)
    conductances = 0.5 + rng.random(solver.n_conductances)
    boundary = graph.mesh.vertices[solver.boundary_vertices]
    control = _dense_dirichlet_solve(solver, conductances, boundary)
    target = control.copy()
    target[solver.interior_vertices] += 0.05 * rng.standard_normal(
        (solver.n_interior, 2)
    )
    loss_gradient = control[solver.interior_vertices] - target[solver.interior_vertices]

    # L lambda = d loss / d x for the fixed-boundary Dirichlet problem.
    basis = np.zeros((solver.n_interior, solver.n_interior), dtype=np.float64)
    for column in range(solver.n_interior):
        probe = np.zeros((solver.n_interior, 1), dtype=np.float64)
        probe[column, 0] = 1.0
        basis[:, column] = solver.apply_interior(
            torch.as_tensor(conductances), torch.as_tensor(probe)
        ).numpy()[:, 0]
    adjoint = np.linalg.solve(basis, loss_gradient)
    analytic = edge_conductance_gradient(solver, control, adjoint)

    def objective(values: np.ndarray) -> float:
        mapped = _dense_dirichlet_solve(solver, values, boundary)
        residual = mapped[solver.interior_vertices] - target[solver.interior_vertices]
        return 0.5 * float(np.sum(residual * residual))

    finite_difference = np.empty_like(conductances)
    step = 2.0e-6
    for edge in range(len(conductances)):
        plus = conductances.copy()
        minus = conductances.copy()
        plus[edge] += step
        minus[edge] -= step
        finite_difference[edge] = (objective(plus) - objective(minus)) / (2.0 * step)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-6, atol=2.0e-8)


@pytest.mark.parametrize("builder", [build_standard_square_graph, build_center_split_square_graph, build_stellar_square_graph])
def test_graph_builders_reject_too_small_side(builder) -> None:
    with pytest.raises(ValueError, match="at least two"):
        builder(1)


def test_tiny_coverage_separates_tensor_assembled_operator_and_map_layers() -> None:
    receipt = _coverage_module().run_coverage(
        cells_per_side=2,
        angles_per_split=4,
        radii=(0.0, 0.4),
        train_steps=3,
        train_learning_rate=2.0e-3,
        hidden_features=8,
        device=torch.device("cpu"),
    )
    assert receipt["status"] == "ok", receipt.get("failures")
    assert receipt["schema"] == "phase5_route3_positive_hodge_coverage_v1"
    assert receipt["scope"]["local_direction_moment_is_not_global_operator"]
    assert receipt["scope"]["finite_directions_exact_arbitrary_spd"] is False
    assert len(receipt["graphs"]) == 3
    assert len(receipt["summaries"]) == 3 * 2 * 2  # graphs * radii * {NNLS, learned held-out}
    for row in receipt["summaries"]:
        assert row["strict_minimum_conductance"] > 0.0
        assert row["tensor_frobenius_relative_error"]["maximum"] >= 0.0
        assert row["assembled_fixed_boundary_operator_relative_error"]["maximum"] >= 0.0
        assert row["decoded_map_rmse"]["maximum"] >= 0.0
        assert row["decoded_mu_rmse"]["maximum"] >= 0.0
        assert row["student_topology_failure_count"] == 0
        assert row["sample_count"] == 4
    json.dumps(receipt, allow_nan=False)


def test_coverage_cli_writes_strict_json_and_rejects_bad_grid(tmp_path: Path) -> None:
    output = tmp_path / "coverage.json"
    command = [
        sys.executable,
        "-W",
        "error",
        str(COVERAGE_SCRIPT),
        "--cells",
        "2",
        "--angles-per-split",
        "4",
        "--radii",
        "0,0.4",
        "--train-steps",
        "2",
        "--hidden-features",
        "8",
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "ok"
    json.dumps(receipt, allow_nan=False)

    bad = subprocess.run(
        command[: command.index("--angles-per-split") + 1]
        + ["3"]
        + command[command.index("--angles-per-split") + 2 :],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0
    assert "even" in bad.stderr


def test_tiny_instance_optimization_runs_full_chain_and_exact_solve_accounting() -> None:
    receipt = _instance_module().run_benchmark(
        task="map",
        methods=("diagnostic_O2_row_softmax_directed", "P1_positive_uniform", "P1_positive_qc_initialized"),
        control_side=4,
        resolution=9,
        steps=2,
        learning_rate=1.0e-3,
        seed=20260922,
        strength=0.12,
        image_name="medical_phantom",
        projector_steps=3,
        projector_learning_rate=3.0e-3,
        projector_hidden_features=8,
        objective_threshold=1.0e9,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert receipt["status"] == "ok", receipt["failures"]
    assert receipt["schema"] == "phase5_route3_positive_hodge_instance_v1"
    assert receipt["protocol_kind"] == "optimized_instance_with_projection_only_baseline"
    assert receipt["configuration"]["fixed_target_boundary"]
    assert receipt["configuration"]["requested_global_solves"] == 5
    assert len(receipt["methods"]) == 3
    for row in receipt["methods"]:
        if row["name"] == "P1_positive_qc_initialized":
            assert row["protocol_kind"] == "oracle_initialized_optimized_upper_bound"
            assert row["equal_budget_rank_eligible"] is False
        elif row["name"] == "diagnostic_O2_row_softmax_directed":
            assert row["protocol_kind"] == "diagnostic_row_softmax_O2"
            assert row["equal_budget_rank_eligible"] is False
        else:
            assert row["protocol_kind"] == "optimized_instance"
            assert row["equal_budget_rank_eligible"] is True
        assert row["completed_primal_solves"] == 3
        assert row["completed_adjoint_solves"] == 2
        assert row["completed_global_solves"] == 5
        assert row["all_iterates_topology_certified"]
        assert row["independent_redecode"]["maximum_vertex_error"] < 2.0e-10
        assert row["independent_redecode"]["topology_certified"]
        assert row["independent_redecode"]["agreement_passed"]
        assert row["independent_redecode"]["authority"].startswith("cpu_float64")
        assert len(row["final_state"]["control"]) == 16
        assert len(row["trace"]) == 3
        assert row["objective_threshold"] == 1.0e9
        assert row["evaluations_to_threshold"] == 1
        assert row["global_solves_to_threshold"] == 1
        assert row["wall_seconds_to_threshold"] >= 0.0
        assert row["time_to_useful"]["projector_setup_plus_method_wall_seconds"] >= row["wall_seconds"]
        assert row["protocol_slices"]["primary_exact_global_solves"]["missing"]
    p1 = next(row for row in receipt["methods"] if row["name"] == "P1_positive_qc_initialized")
    assert p1["chain"] == [
        "latent_w",
        "mu_in_unit_disk",
        "A(mu)_spd",
        "learned_softplus_positive_local_map",
        "shared_edge_average",
        "symmetric_spd_tutte_solve",
        "dense_p1_warp",
        "loss",
    ]
    assert receipt["target_authority_audit"]["required"] is False
    json.dumps(receipt, allow_nan=False)


def test_projection_only_is_disclosed_and_not_given_equal_budget() -> None:
    receipt = _instance_module().run_benchmark(
        task="image",
        methods=("P1_positive_qc_projection_only",),
        control_side=4,
        resolution=9,
        steps=2,
        learning_rate=1.0e-3,
        seed=101,
        strength=0.1,
        image_name="medical_phantom",
        projector_steps=2,
        projector_learning_rate=3.0e-3,
        projector_hidden_features=8,
        objective_threshold=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert receipt["status"] == "ok"
    row = receipt["methods"][0]
    assert row["protocol_kind"] == "projection_only"
    assert row["completed_primal_solves"] == 1
    assert row["completed_adjoint_solves"] == 0
    assert row["equal_budget_rank_eligible"] is False
    assert row["final_objective"] >= 0.0
    assert row["final_metrics"]["global_injectivity_certificate"]


def test_instance_cli_fresh_process_writes_strict_json(tmp_path: Path) -> None:
    output = tmp_path / "instance.json"
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            str(INSTANCE_SCRIPT),
            "--task",
            "map",
            "--methods",
            "P1_positive_uniform",
            "--N",
            "4",
            "--R",
            "9",
            "--steps",
            "1",
            "--lr",
            "0.001",
            "--seed",
            "20260922",
            "--strength",
            "0.1",
            "--projector-steps",
            "2",
            "--projector-hidden-features",
            "8",
            "--dtype",
            "float64",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "ok"
    assert receipt["methods"][0]["completed_global_solves"] == 3
    json.dumps(receipt, allow_nan=False)


def test_formal_target_authority_audit_is_fail_closed() -> None:
    module = _instance_module()
    authority_path = (
        REPOSITORY
        / "docs/research_phase5/raw_results/route2_mvc_instance_formal_map_gpu_O1_ai_74b452c_clean.json"
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    numeric = authority["shared_target"]["numeric_identity"]
    current = {
        "control_shape": numeric["control"]["shape"],
        "control_sum": numeric["control"]["sum"],
        "control_l2": numeric["control"]["l2"],
        "control_min": numeric["control"]["minimum"],
        "control_max": numeric["control"]["maximum"],
        "dense_shape": numeric["dense"]["shape"],
        "dense_sum": numeric["dense"]["sum"],
        "dense_l2": numeric["dense"]["l2"],
        "dense_min": numeric["dense"]["minimum"],
        "dense_max": numeric["dense"]["maximum"],
    }
    audit = module._audit_clean_target_authority("map", current, required=True)
    assert audit["matched"]
    assert audit["authority_commit"].startswith("74b452c")
    corrupted = dict(current)
    corrupted["dense_sum"] += 1.0e-4
    with pytest.raises(RuntimeError, match="target authority mismatch"):
        module._audit_clean_target_authority("map", corrupted, required=True)


def test_exact_83_slice_uses_observation_counts_not_post_backward_counts() -> None:
    module = _instance_module()
    trace = [
        {
            "iteration": iteration,
            "objective": 1.0 / (iteration + 1),
            "map_rmse": 0.1,
            "mu_rmse": 0.2,
            "maximum_mu_error": 0.3,
            "minimum_area_ratio": 0.4,
            "topology_certified": True,
            "observation_completed_global_solves": 2 * iteration + 1,
            "observation_wall_seconds": 0.01 * iteration,
        }
        for iteration in range(82)
    ]
    slices = module._extract_protocol_slices(trace)
    primary = slices["primary_exact_global_solves"]
    assert not primary["missing"]
    assert primary["row"]["iteration"] == 41
    assert primary["row"]["completed_global_solves"] == 83
    assert slices["final_observation"]["completed_global_solves"] == 163


def test_actual_tiny_runner_emits_unique_exact_83_slice() -> None:
    receipt = _instance_module().run_benchmark(
        task="map",
        methods=("P1_positive_uniform",),
        control_side=3,
        resolution=5,
        steps=41,
        learning_rate=1.0e-3,
        seed=43,
        strength=0.08,
        image_name="medical_phantom",
        projector_steps=1,
        projector_learning_rate=3.0e-3,
        projector_hidden_features=6,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    row = receipt["methods"][0]
    assert row["completed_global_solves"] == 83
    primary = row["protocol_slices"]["primary_exact_global_solves"]
    assert not primary["missing"]
    assert primary["row"]["iteration"] == 41
    assert primary["row"]["completed_global_solves"] == 83
    cumulative = primary["row"]["cumulative_timing_at_observation"]
    assert cumulative["solver_seconds"] > 0.0
    assert cumulative["dense_interpolation_seconds"] > 0.0
    assert cumulative["task_objective_seconds"] > 0.0
    assert cumulative["backward_seconds"] > 0.0
