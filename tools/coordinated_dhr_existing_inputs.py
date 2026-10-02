"""Prepare image-only native DHR inputs for the existing lung20/Histo/kidney cases.

No images are copied, no labels are opened and no registration is launched.
remote_inputs.json resolves nine unique JPEGs under images/ next to that file.
The transfer list is an ordinary copy list, not a new benchmark/data framework.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from PIL import Image


STAINS = {"he": "He", "cc10": "Cc10-5", "cd31": "CD31-3", "ki67": "Ki67-7", "prospc": "proSPC-4"}
PREFIX = "29-041-Izd2-w35-"
SCOPE = "20 correlated lung-lesion3 directions plus Histo CD4-to-CD68 and kidney HE-to-PanCytokeratin; three previously viewed specimens, no new data"


def existing_rows(data_root):
    root = Path(data_root)
    lung = root / "external_lung_lesion3_borda/dataset/lung-lesion_3/scale-5pc"
    image = lambda stain: str(lung / f"{PREFIX}{STAINS[stain]}-les3.jpg")
    rows = [dict(name=f"{fixed}_to_{moving}", fixed=image(fixed), moving=image(moving))
            for fixed, moving in itertools.permutations(STAINS, 2)]
    histo = root / "HistoReg_CD68_CD4"
    kidney = root / "birl_anhir_dev/images/rat-kidney_/scale-5pc"
    rows.extend([dict(name="histo", fixed=str(histo / "Images_CD4.jpg"), moving=str(histo / "Images_CD68.jpg")),
                 dict(name="rat_kidney", fixed=str(kidney / "Rat-Kidney_HE.jpg"), moving=str(kidney / "Rat-Kidney_PanCytokeratin.jpg"))])
    return rows


def prepare(data_root, output):
    root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = existing_rows(root)
    paths = sorted({r[role] for r in rows for role in ("fixed", "moving")})
    if len({Path(p).name for p in paths}) != len(paths):
        raise ValueError("image basenames must be unique for the flat remote copy list")
    images = []
    for source in paths:
        with Image.open(source) as image:
            if image.mode not in ("RGB", "L") or image.getexif().get(274, 1) != 1:
                raise ValueError("identity-oriented native RGB/L image required")
            images.append(dict(local=source, remote_relative="images/" + Path(source).name,
                               original_wh=list(image.size), mode=image.mode, bytes=Path(source).stat().st_size))
    remote = [dict(name=r["name"], **{role: "images/"+Path(r[role]).name for role in ("fixed", "moving")}) for r in rows]
    output.mkdir(parents=True)
    for name, content in (("local_inputs.json", dict(scope=SCOPE, rows=rows)),
                          ("remote_inputs.json", dict(scope=SCOPE, rows=remote))):
        (output / name).write_text(json.dumps(content, indent=2)+"\n", encoding="utf-8")
    report = dict(scope=SCOPE, pair_count=22, image_count=len(images), annotations_read=False,
                  images=images, image_bytes=sum(item["bytes"] for item in images),
                  launch="not launched; select STANDARD, serial GPU jobs, check host RAM and free VRAM first",
                  histo_memory_note="native ~74M-pixel RGB images copied to GPU before resizing; two float32 padded inputs alone ~1.8GB, additional workspace required",
                  excluded="old BIRL lesions_ crop is not the lung-lesion3 frame and is not a 23rd job")
    (output / "transfer_list.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.data_root, args.output)
    print(json.dumps({key: report[key] for key in ("pair_count", "image_count", "image_bytes", "annotations_read")}))


if __name__ == "__main__":
    main()
