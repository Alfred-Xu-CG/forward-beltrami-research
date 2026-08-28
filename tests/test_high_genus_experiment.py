import json

from qcopt.experiments.high_genus_registration import (
    TrialConfig,
    build_synthetic_genus2_case,
    primary_trial_config,
    run_trial,
)
from qcopt.experiments.audit_high_genus_registration import (
    audit_saved_trial,
    combine_audit_files,
)
from qcopt.surface_registration import RegistrationConfig


def test_primary_trial_preset_is_the_reported_fixed_budget():
    config = primary_trial_config()
    assert config.perturbation_amplitude == 0.5
    assert config.landmarks == 20
    assert config.audit_faces == 64
    assert config.registration.iterations == 10
    assert config.registration.smoothing == 1.5
    assert config.registration.descriptor_smoothing == 0.5
    assert config.registration.max_update_fraction == 0.05
    assert config.registration.integration_substeps == 8
    assert config.registration.cfl_limit == 0.15


def test_synthetic_experiment_case_is_a_true_two_surface_genus2_pair():
    case = build_synthetic_genus2_case(resolution=18, maximum_samples=60)
    assert case.source.topology.genus == 2
    assert case.target.topology.genus == 2
    assert case.source.topology.closed and case.target.topology.closed
    assert case.source.has_same_connectivity(case.target)
    assert len(case.source_samples) == 60
    assert len(case.ground_truth_target_points) == 60
    assert not (case.source.vertices == case.target.vertices).all()


def test_quick_trial_writes_replayable_metrics_states_history_and_figures(tmp_path):
    case = build_synthetic_genus2_case(resolution=18, maximum_samples=48)
    config = TrialConfig(
        perturbation_amplitude=0.2,
        landmarks=8,
        audit_faces=12,
        registration=RegistrationConfig(
            iterations=1,
            landmark_weight=1.0,
            curvature_weight=1.0,
            smoothing=0.5,
            descriptor_smoothing=0.25,
            max_update_fraction=0.04,
            integration_substeps=4,
            cfl_limit=0.18,
        ),
    )
    metrics = run_trial(case, seed=3, mode="combined", output_dir=tmp_path, config=config)

    assert metrics["case"] == "synthetic_genus2"
    assert metrics["genus"] == 2
    assert metrics["sample_count"] == 48
    assert metrics["landmark_count"] == 8
    assert metrics["objective_monotone"]
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "flow_history.npz").exists()
    assert (tmp_path / "surface_points.npz").exists()
    assert (tmp_path / "correspondence.png").exists()
    assert (tmp_path / "objective.png").exists()
    reloaded = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert reloaded["configuration"]["seed"] == 3
    assert reloaded["certificate"]["audited_faces"] > 0
    independent = audit_saved_trial(tmp_path, case=case)
    assert independent["passed"]
    assert independent["certificate_recomputed"]
    assert independent["dense_metrics_recomputed"]
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"trials": [independent]}), encoding="utf-8")
    combined = combine_audit_files(
        [partial],
        output_directory=tmp_path / "combined",
        require_complete_matrix=False,
    )
    assert combined["overall"]["trial_count"] == 1
    assert (tmp_path / "combined" / "research_report.md").exists()

    reloaded["valid_run"] = not reloaded["valid_run"]
    (tmp_path / "metrics.json").write_text(
        json.dumps(reloaded), encoding="utf-8"
    )
    corrupted = audit_saved_trial(tmp_path, case=case)
    assert not corrupted["saved_status_matches_recomputed"]
    assert not corrupted["passed"]
