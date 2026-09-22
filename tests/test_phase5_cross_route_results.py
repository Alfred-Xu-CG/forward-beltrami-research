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
    assert len(module().FIELDS) == 50


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
