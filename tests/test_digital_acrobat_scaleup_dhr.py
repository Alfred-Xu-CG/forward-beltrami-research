"""Confirmation initial-only preparation cannot silently run a full teacher."""

import json

from tools import digital_acrobat_scaleup_dhr as dhr


def test_pending_stage_filter_keeps_initial_and_full_separate(monkeypatch, tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "new_confirmation": {"ER": [{"case": 1, "stain": "ER"}]},
    }))
    (tmp_path / "1_HE_physical512.png").touch()
    (tmp_path / "1_ER_physical512.png").touch()
    (tmp_path / "1_DHR_physical_initial_teacher_affine.npz").touch()
    calls = []

    def fake_run(item, root, gpu, stages):
        calls.append((item["case"], gpu, stages))
        return {"case": item["case"], "gpu": gpu, "stages": list(stages)}

    monkeypatch.setattr(dhr, "_run_one", fake_run)
    initial = dhr.run_selected(selection, tmp_path, tmp_path / "initial.json",
                               section="new_confirmation", devices=[0],
                               available_pending_only=True, stages=("initial",))
    assert initial["selected_cases"] == 0 and calls == []
    full = dhr.run_selected(selection, tmp_path, tmp_path / "full.json",
                            section="new_confirmation", devices=[0],
                            available_pending_only=True, stages=("full",))
    assert full["selected_cases"] == full["successful_cases"] == 1
    assert calls == [(1, 0, ("full",))]
