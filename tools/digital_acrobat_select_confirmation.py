"""Select fresh ACROBAT training-archive cases from ZIP metadata alone."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from remotezip import RemoteZip


STAINS = ("ER", "PGR", "KI67", "HER2")
MEMBER = re.compile(r"^(\d+)_(HE|ER|PGR|KI67|HER2)\.tiff$")
MAX_COMBINED_STORED_BYTES = 500_000_000


def select_cases(members: Iterable, *, seed: int, exclude: set[int],
                 per_stain: int = 2) -> dict[str, list[dict]]:
    """Return disjoint stain-stratified cases under a fixed stored-size cap."""
    if per_stain < 1:
        raise ValueError("per_stain must be positive")
    index: dict[int, dict[str, list]] = {}
    for member in members:
        match = MEMBER.fullmatch(member.filename)
        if match:
            case, stain = int(match[1]), match[2]
            index.setdefault(case, {}).setdefault(stain, []).append(member)
    rng = np.random.default_rng(seed)
    used = set(exclude)
    selection: dict[str, list[dict]] = {}
    for stain in STAINS:
        candidates = []
        for case, files in sorted(index.items()):
            if case in used or len(files.get("HE", [])) != 1 or len(files.get(stain, [])) != 1:
                continue
            he, ihc = files["HE"][0], files[stain][0]
            total = he.compress_size + ihc.compress_size
            if total < MAX_COMBINED_STORED_BYTES:
                candidates.append({
                    "case": case, "stain": stain,
                    "members": [he.filename, ihc.filename],
                    "combined_stored_bytes": total,
                })
        chosen = []
        for position in rng.permutation(len(candidates)):
            candidate = candidates[int(position)]
            if candidate["case"] not in used:
                chosen.append(candidate)
                used.add(candidate["case"])
                if len(chosen) == per_stain:
                    break
        if len(chosen) != per_stain:
            raise ValueError(f"not enough eligible {stain} cases")
        selection[stain] = chosen
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with RemoteZip(args.archive_url, timeout=120) as archive:
        selected = select_cases(archive.infolist(), seed=args.seed,
                                exclude=set(args.exclude))
    report = {
        "source": "official_ACROBAT_training_archive_part1_zip_metadata_only",
        "archive_url": args.archive_url,
        "seed": args.seed,
        "excluded_case_ids": sorted(set(args.exclude)),
        "max_combined_stored_bytes_exclusive": MAX_COMBINED_STORED_BYTES,
        "selection": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
