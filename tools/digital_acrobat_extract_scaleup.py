"""Extract only the frozen scale-up ACROBAT subset by HTTP ZIP byte ranges.

The official archive is not downloaded wholesale. Each selected TIFF is
written to a temporary file in the D: output directory, length-checked, and
renamed only when complete. Existing complete files are reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from remotezip import RemoteZip


def extract_selection(selection_path: Path, output_dir: Path, *,
                      section: str = "new_train",
                      stains: list[str] | None = None) -> dict:
    if section not in ("new_train", "new_confirmation"):
        raise ValueError("unknown selection section")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    available = selection[section]
    if stains is not None and (len(set(stains)) != len(stains)
                               or not set(stains) <= set(available)):
        raise ValueError("stains must be unique names in selected section")
    items = [item for stain, rows in available.items()
             if stains is None or stain in stains for item in rows]
    members = [name for item in items for name in item["members"]]
    if len(set(members)) != len(members) or any(
        Path(name).name != name or not name.endswith(".tiff") for name in members
    ):
        raise ValueError("unique TIFF basenames required")
    output_dir.mkdir(parents=True, exist_ok=True)
    reused, extracted, bytes_extracted = 0, 0, 0
    with RemoteZip(selection["archive_url"], timeout=120) as archive:
        infos = [archive.getinfo(name) for name in members]
        needed = sum(info.file_size for info in infos if not (output_dir / info.filename).exists())
        if shutil.disk_usage(output_dir).free < needed + 10_000_000_000:
            raise OSError("insufficient free space with 10 GB headroom")
        for index, info in enumerate(infos, 1):
            if info.file_size > 1_000_000_000:
                raise ValueError(f"selected file exceeds 1 GB cap: {info.filename}")
            target = output_dir / info.filename
            if target.exists():
                if target.stat().st_size != info.file_size:
                    raise ValueError(f"existing file has wrong size: {target}")
                reused += 1
                print(f"{index}/{len(infos)} reused {info.filename}", flush=True)
                continue
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=output_dir, prefix=info.filename + ".part.",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    with archive.open(info) as source:
                        shutil.copyfileobj(source, stream, length=1 << 20)
                if temporary.stat().st_size != info.file_size:
                    raise IOError(f"incomplete extraction: {info.filename}")
                if target.exists():
                    raise FileExistsError(target)
                os.replace(temporary, target)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            extracted += 1
            bytes_extracted += info.file_size
            print(f"{index}/{len(infos)} extracted {info.filename} {info.file_size}",
                  flush=True)
    return {"section": section, "stains": stains,
            "selected_cases": len(items), "files": len(infos),
            "reused": reused, "extracted": extracted,
            "uncompressed_bytes_extracted": bytes_extracted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--section", choices=("new_train", "new_confirmation"),
                        default="new_train")
    parser.add_argument("--stains", nargs="+")
    args = parser.parse_args()
    print(json.dumps(extract_selection(args.selection, args.output_dir,
                                       section=args.section,
                                       stains=args.stains)), flush=True)


if __name__ == "__main__":
    main()
