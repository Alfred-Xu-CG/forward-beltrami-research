"""Extract only named public ACROBAT training TIFFs from its huge ZIP.

The official train archive supports HTTP byte ranges. No archive-wide download,
validation/test images, landmarks, or login bypass is involved. The selected
case IDs are development samples only; they are not a benchmark cohort.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from remotezip import RemoteZip


DEFAULT_MEMBERS = (
    "638_HE.tiff", "638_PGR.tiff",
    "315_HE.tiff", "315_HER2.tiff",
    "156_HE.tiff", "156_KI67.tiff",
    "585_HE.tiff", "585_ER.tiff",
    "733_HE.tiff", "733_HER2.tiff",
    "586_HE.tiff", "586_PGR.tiff",
    "495_HE.tiff", "495_HER2.tiff",
    "399_HE.tiff", "399_KI67.tiff",
    "100_HE.tiff", "100_PGR.tiff",
    "330_HE.tiff", "330_HER2.tiff",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--member", action="append", default=None,
                        help="Optional exact archive member; repeat as needed")
    args = parser.parse_args()
    members = tuple(args.member) if args.member else DEFAULT_MEMBERS
    if len(set(members)) != len(members) or any(
        Path(name).name != name or not name.endswith(".tiff")
        for name in members
    ):
        raise ValueError("members must be unique TIFF basenames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with RemoteZip(args.archive_url, timeout=120) as archive:
        info = [archive.getinfo(name) for name in members]
        total = sum(item.file_size for item in info)
        if total > 4_000_000_000 or any(item.file_size > 1_000_000_000
                                      for item in info):
            raise ValueError("selected files exceed the deliberate subset budget")
        for item in info:
            target = args.output_dir / item.filename
            if target.exists():
                if target.stat().st_size != item.file_size:
                    raise ValueError(f"existing file has wrong size: {target}")
                print(f"already-present {item.filename} {item.file_size}", flush=True)
                continue
            archive.extract(item, path=args.output_dir)
            if target.stat().st_size != item.file_size:
                raise IOError(f"incomplete extraction: {target}")
            print(f"extracted {item.filename} {item.file_size}", flush=True)


if __name__ == "__main__":
    main()
