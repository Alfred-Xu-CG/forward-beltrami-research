"""The scale-up split is metadata-only and records its ordering protocol."""

import json
from dataclasses import dataclass

from tools import digital_acrobat_select_scaleup as selector


@dataclass
class Member:
    filename: str
    compress_size: int = 100


def test_scaleup_records_order_and_excludes_prepared_confirmation(monkeypatch, tmp_path):
    members = []
    for index, stain in enumerate(("ER", "PGR", "KI67", "HER2")):
        for local in range(1, 7):
            case = index * 100 + local
            members.extend((Member(f"{case}_HE.tiff"),
                            Member(f"{case}_{stain}.tiff")))

    class FakeRemoteZip:
        def __init__(self, url, timeout):
            assert url == "metadata://archive" and timeout == 120

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def infolist(self):
            return members

    monkeypatch.setattr(selector, "RemoteZip", FakeRemoteZip)
    old_train = tmp_path / "train.json"
    old_dev = tmp_path / "dev.json"
    old_confirm = tmp_path / "confirm.json"
    old_train.write_text(json.dumps({"train_case_ids": [999]}))
    old_dev.write_text(json.dumps({"test_case_ids": [998]}))
    old_confirm.write_text(json.dumps({"test_case_ids": [997]}))
    output = tmp_path / "selection.json"
    report = selector.choose(
        "metadata://archive", old_train, old_dev, old_confirm, output,
        train_per_stain=2, confirm_per_stain=2,
        stain_order=("HER2", "ER", "PGR", "KI67"),
        additional_confirm_exclude_case_ids=(301,),
        search_stain_orders=True,
    )
    assert json.loads(output.read_text()) == report
    assert report["requested_initial_stain_order"] == ["HER2", "ER", "PGR", "KI67"]
    assert report["stain_order"] == ["HER2", "ER", "PGR", "KI67"]
    assert report["confirmation_stain_order"] == ["HER2", "ER", "PGR", "KI67"]
    assert report["additional_confirm_excluded_ids"] == [301]
    assert len(set(report["combined_train_ids"] + report["new_confirmation_ids"])) == 17
    assert 301 not in report["new_confirmation_ids"]
