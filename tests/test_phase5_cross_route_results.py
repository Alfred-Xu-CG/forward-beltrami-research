"""Cross-route table tests use receipts as evidence, never hand-copied numbers."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/phase5/cross_route_results.py"
RAW = ROOT / "docs/research_phase5/raw_results"


def module():
    spec = importlib.util.spec_from_file_location("cross_route_results_test", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def authority_paths() -> list[Path]:
    paths = []
    for task in ("map", "image"):
        for method in ("O1", "O3", "O4"):
            paths.append(
                RAW
                / f"route2_mvc_instance_formal_{task}_gpu_{method}_ai_74b452c_clean.json"
            )
    return paths


def p1_paths() -> list[Path]:
    return [
        RAW / f"route3_p1_formal_{task}_gpu3_35a1878.json"
        for task in ("map", "image")
    ]


def pref_paths() -> list[Path]:
    return [
        RAW / f"route3_pref_common_{task}_cpu_0da8ffa.json"
        for task in ("map", "image")
    ]


def save(tmp_path: Path, receipt: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_real_authority_uses_exact83_not_step40_or_final() -> None:
    rows = module().extract_route2_authority(authority_paths())
    assert len(rows) == 6
    by_key = {(row["task"], row["method"]): row for row in rows}

    o1_map = by_key["map", "O1_sigmoid_positive"]
    assert o1_map["primary_slice"] == "exact83"
    assert o1_map["primary_global_solves"] == 83
    assert o1_map["primary_objective"] == pytest.approx(7.00551377538489e-05)
    assert o1_map["final_objective"] == pytest.approx(4.547660719699229e-05)
    assert o1_map["primary_objective"] != o1_map["final_objective"]
    assert o1_map["final_global_solves"] == 163
    assert o1_map["threshold_global_solves"] == 43
    assert o1_map["algorithm_wall_seconds"] < o1_map["wrapper_wall_seconds"]
    assert "not parameterization-fair tuning" in o1_map["notes"]
    assert "no ranking before P1" in o1_map["notes"]

    o3_map = by_key["map", "O3_mvc_adam"]
    assert o3_map["route_label"] == "T2"
    assert o3_map["selected_for_common_input"] is True
    assert o3_map["primary_global_solves"] == 83

    # Timing columns are cumulative through the exact-83 observation, not the
    # single trace row's per-observation timing.
    assert o1_map["primary_forward_seconds"] == pytest.approx(14.806887775659561)
    assert o1_map["primary_backward_seconds"] == pytest.approx(10.005911007523537)
    assert o1_map["primary_dense_seconds"] == pytest.approx(0.03535051271319389)


def test_o4_image_failure_is_preserved_without_imputed_exact83() -> None:
    rows = module().extract_route2_authority(authority_paths())
    row = next(r for r in rows if r["task"] == "image" and r["method"] == "O4_covariance_retraction")
    assert row["status"] == "optimizer_failure"
    assert row["primary_slice"] == "exact83_missing"
    assert row["primary_global_solves"] is None
    assert row["primary_objective"] is None
    assert row["map_rmse"] is None
    assert "step 18 retraction_decode" in row["failure_reason"]
    assert "last_certified_accepted_state_before_failure" in row["notes"]
    assert row["final_global_solves"] == 19
    assert row["final_objective"] == pytest.approx(0.004339071676058662)
    assert row["final_topology_certified"] is True
    assert row["selected_for_common_input"] is False


def test_units_and_target_identity_are_explicit() -> None:
    rows = module().extract_route2_authority(authority_paths())
    row = rows[0]
    assert row["target_control_sum"] == pytest.approx(611.298547235479)
    assert row["target_dense_l2"] == pytest.approx(205.46519723260525)
    assert row["cuda_peak_memory_GB_decimal"] == pytest.approx(0.01839104)
    assert row["cuda_peak_memory_GiB_binary"] == pytest.approx(0.01712799072265625)
    assert row["task"] == "image"
    assert row["image_mse"] == row["primary_objective"]
    assert next(item for item in rows if item["task"] == "map")["image_mse"] is None
    assert row["latent_dimension"] is None
    assert row["process_hwm_GB_decimal"] == pytest.approx(1.309868032)
    assert row["hard_topology_scope"] == "positive_Tutte_decoder_with_independent_iterate_audits"
    assert len(module().FIELDS) == 57


def test_common_table_adds_p1_and_pref_without_false_ranking() -> None:
    rows = module().extract_common_results(authority_paths(), p1_paths(), pref_paths())
    assert len(rows) == 10
    by_key = {(row["task"], row["method"]): row for row in rows}

    p1_map = by_key["map", "P1_positive_uniform"]
    assert p1_map["route_label"] == "P1"
    assert p1_map["primary_slice"] == "exact83"
    assert p1_map["primary_global_solves"] == 83
    assert p1_map["primary_objective"] == pytest.approx(7.65868296606389e-6)
    assert p1_map["primary_forward_seconds"] == pytest.approx(11.108095470815897)
    assert p1_map["primary_backward_seconds"] == pytest.approx(5.467638202011585)
    assert p1_map["primary_dense_seconds"] == pytest.approx(0.005829419940710068)
    assert p1_map["map_rmse"] == pytest.approx(0.004605230120138025)
    assert p1_map["mu_rmse"] == pytest.approx(0.09918527430055511)
    assert p1_map["topology_certified"] is True
    assert p1_map["latent_dimension"] == 2304
    assert p1_map["gradient_check_relative_error"] is None
    assert p1_map["hard_topology_scope"] == "positive_symmetric_Tutte_fixed_graph_and_convex_boundary"

    pref_map = by_key["map", "P-ref_full_Whitney_teacher"]
    assert pref_map["route_label"] == "P-ref"
    assert pref_map["protocol_kind"] == "target_derived_one_shot_reference"
    assert pref_map["primary_slice"] == "one_forward_one_adjoint"
    assert pref_map["primary_global_solves"] == 2
    assert pref_map["primary_objective"] == pytest.approx(2.0720980749075907e-31)
    assert pref_map["comparable"] is False
    assert pref_map["selected_for_common_input"] is True
    assert pref_map["explicit_state_MB_decimal"] == pytest.approx(6.864144)
    assert pref_map["hard_topology_scope"] == "observed_this_target_only_no_general_guarantee"
    assert pref_map["gradient_check_relative_error"] == pytest.approx(1.0784960953491835e-7)

    for task in ("map", "image"):
        route2 = by_key[task, "O1_sigmoid_positive"]
        p1 = by_key[task, "P1_positive_uniform"]
        pref = by_key[task, "P-ref_full_Whitney_teacher"]
        for key in (
            "target_control_sum",
            "target_control_l2",
            "target_dense_sum",
            "target_dense_l2",
        ):
            assert p1[key] == pytest.approx(route2[key], rel=1e-12, abs=1e-12)
            assert pref[key] == pytest.approx(route2[key], rel=1e-12, abs=1e-12)


def test_authority_rejects_target_mismatch_and_nonexact_slice(tmp_path: Path) -> None:
    copied = []
    for path in authority_paths():
        receipt = json.loads(path.read_text(encoding="utf-8"))
        copied.append(save(tmp_path, receipt, path.name))

    data = json.loads(copied[1].read_text(encoding="utf-8"))
    data["shared_target"]["numeric_identity"]["control"]["sum"] += 1e-5
    copied[1] = save(tmp_path, data, copied[1].name)
    with pytest.raises(ValueError, match="target numeric identity"):
        module().extract_route2_authority(copied)

    copied = []
    for path in authority_paths():
        receipt = json.loads(path.read_text(encoding="utf-8"))
        copied.append(save(tmp_path, receipt, "again_" + path.name))
    data = json.loads(copied[0].read_text(encoding="utf-8"))
    row = data["protocol_slices"]["primary_exact_global_solves"]["rows"]["O1_sigmoid_positive"]
    row["global_solves"] = 81
    copied[0] = save(tmp_path, data, copied[0].name)
    with pytest.raises(ValueError, match="exactly 83"):
        module().extract_route2_authority(copied)


def test_authority_rejects_dirty_wrong_commit_and_duplicate_json_key(tmp_path: Path) -> None:
    data = json.loads(authority_paths()[0].read_text(encoding="utf-8"))
    data["environment"]["dirty"] = True
    with pytest.raises(ValueError, match="clean"):
        module().extract_route2_authority([save(tmp_path, data, "dirty.json")])

    data["environment"]["dirty"] = False
    data["environment"]["commit"] = "0" * 40
    with pytest.raises(ValueError, match="74b452c"):
        module().extract_route2_authority([save(tmp_path, data, "wrong.json")])

    text = authority_paths()[0].read_text(encoding="utf-8")
    text = text.replace('"schema":', '"schema": "duplicate", "schema":', 1)
    path = tmp_path / "duplicate.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        module().extract_route2_authority([path])


def test_threshold_fields_are_rederived_from_trace(tmp_path: Path) -> None:
    paths = authority_paths()
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    method = next(iter(data["methods"].values()))
    method["global_solves_to_threshold"] = 999
    changed = save(tmp_path, data, paths[0].name)
    with pytest.raises(ValueError, match="threshold"):
        module().extract_route2_authority([changed, *paths[1:]])
