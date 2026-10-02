import argparse
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.coordinated_dhr_existing_inputs import existing_rows, prepare
from tools.coordinated_dhr_released_baseline import input_rows


def test_existing22_direction_and_image_only_transfer(tmp_path):
    root = tmp_path / "data"
    rows = existing_rows(root)
    assert len(rows) == 22 and len({r["name"] for r in rows}) == 22
    assert rows[0]["name"] == "he_to_cc10"
    assert rows[-2]["fixed"].endswith("Images_CD4.jpg")
    assert rows[-2]["moving"].endswith("Images_CD68.jpg")
    assert rows[-1]["fixed"].endswith("Rat-Kidney_HE.jpg")
    assert all(set(r) == {"name", "fixed", "moving"} for r in rows)
    for path in {r[k] for r in rows for k in ("fixed", "moving")}:
        image = Path(path)
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (17, 23)).save(image)
    result = prepare(root, tmp_path / "prepared")
    assert len(result["images"]) == 9
    assert all(r["local"].endswith(".jpg") for r in result["images"])
    remote = json.loads((tmp_path / "prepared/remote_inputs.json").read_text())
    assert len(remote["rows"]) == 22
    assert all(r[k].startswith("images/") for r in remote["rows"] for k in ("fixed", "moving"))
    args = argparse.Namespace(source_data=None, input_rows=tmp_path / "prepared/local_inputs.json")
    loaded, metadata = input_rows(args)
    assert [r["name"] for r in loaded] == [r["name"] for r in rows]
    assert metadata["input_manifest"].endswith("local_inputs.json")


@pytest.mark.parametrize("change", ["duplicate", "traversal", "annotations", "affine", "self_pair"])
def test_manifest_rejects_unsafe_or_non_image_payload_before_output(tmp_path, change):
    row = dict(name="case", fixed="a.jpg", moving="b.jpg")
    rows = [row]
    if change == "duplicate": rows.append(dict(row))
    if change == "traversal": row["name"] = "../outside"
    if change == "annotations": row["landmarks"] = "labels.csv"
    if change == "affine": row["affine"] = "affine.npz"
    if change == "self_pair": row["moving"] = row["fixed"]
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(dict(scope="test", rows=rows)))
    with pytest.raises(ValueError):
        input_rows(argparse.Namespace(source_data=None, input_rows=path))


def test_relative_image_paths_resolve_at_manifest_not_cwd(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(dict(scope="test", rows=[dict(name="case", fixed="images/f.jpg", moving="images/m.jpg")])))
    rows, _ = input_rows(argparse.Namespace(source_data=None, input_rows=path))
    assert rows[0]["fixed"] == str((tmp_path / "images/f.jpg").resolve())
