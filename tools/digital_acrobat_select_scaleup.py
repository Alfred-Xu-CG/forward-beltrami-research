"""Freeze a larger ACROBAT training subset and a separate confirmation set.

Reads only ZIP central-directory metadata and prior case-ID manifests, not
images, fields, labels, or model outputs. Does not download TIFF content.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from remotezip import RemoteZip

from tools.digital_acrobat_select_confirmation import select_cases


def _flatten(selection: dict) -> list[dict]:
    return [item for rows in selection.values() for item in rows]


def choose(archive_url: str, old_train_manifest: Path, old_dev_report: Path,
           prior_confirmation_report: Path, output: Path, *,
           train_per_stain: int = 20, confirm_per_stain: int = 2,
           stain_order: tuple[str, ...] = ("ER", "PGR", "KI67", "HER2"),
           additional_confirm_exclude_selection: Path | None = None,
           additional_confirm_exclude_case_ids: tuple[int, ...] = (),
           search_stain_orders: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(output)
    with old_train_manifest.open(encoding="utf-8") as stream:
        old_train = json.load(stream)["train_case_ids"]
    with old_dev_report.open(encoding="utf-8") as stream:
        old_dev = json.load(stream)["test_case_ids"]
    with prior_confirmation_report.open(encoding="utf-8") as stream:
        old_confirm = json.load(stream)["test_case_ids"]
    excluded = set(old_train) | set(old_dev) | set(old_confirm)
    if len(excluded) != len(old_train) + len(old_dev) + len(old_confirm):
        raise ValueError("prior train/development/confirmation IDs overlap")
    additional_confirm_exclude: set[int] = set(additional_confirm_exclude_case_ids)
    if additional_confirm_exclude_selection is not None:
        with additional_confirm_exclude_selection.open(encoding="utf-8") as stream:
            prior_selection = json.load(stream)
        additional_confirm_exclude |= (
            set(prior_selection["new_train_ids"])
            | set(prior_selection["new_confirmation_ids"])
        )
    with RemoteZip(archive_url, timeout=120) as archive:
        members = archive.infolist()
        orders = [stain_order]
        if search_stain_orders:
            orders += [order for order in itertools.permutations(stain_order)
                       if order != stain_order]
        selected = None
        for train_order in orders:
            try:
                trial_train = select_cases(members, seed=20261001,
                                           exclude=excluded,
                                           per_stain=train_per_stain,
                                           stain_order=train_order)
            except ValueError:
                continue
            trial_ids = [item["case"] for item in _flatten(trial_train)]
            for confirm_order in orders:
                try:
                    trial_confirm = select_cases(
                        members, seed=20261002,
                        exclude=excluded | set(trial_ids)
                        | additional_confirm_exclude,
                        per_stain=confirm_per_stain,
                        stain_order=confirm_order)
                except ValueError:
                    continue
                selected = (trial_train, trial_ids, trial_confirm,
                            train_order, confirm_order)
                break
            if selected is not None:
                break
        if selected is None:
            raise ValueError("no eligible train/confirmation stain-order pair")
        added_train, added_ids, confirm, train_order, confirm_order = selected
    confirm_ids = [item["case"] for item in _flatten(confirm)]
    if len(set(added_ids + confirm_ids)) != len(added_ids) + len(confirm_ids):
        raise AssertionError("selection overlap")
    report = {
        "source": "official_ACROBAT_training_archive_part1_ZIP_metadata_only",
        "archive_url": archive_url,
        "old_train_ids": old_train,
        "excluded_old_dev_ids": old_dev,
        "excluded_prior_confirmation_ids": old_confirm,
        "train_seed": 20261001, "confirm_seed": 20261002,
        "requested_initial_stain_order": list(stain_order),
        "stain_order": list(train_order),
        "confirmation_stain_order": list(confirm_order),
        "searched_stain_orders": search_stain_orders,
        "additional_confirm_excluded_ids": sorted(additional_confirm_exclude),
        "new_train_per_stain": train_per_stain,
        "confirm_per_stain": confirm_per_stain,
        "new_train": added_train, "new_train_ids": added_ids,
        "combined_train_ids": old_train + added_ids,
        "new_confirmation": confirm, "new_confirmation_ids": confirm_ids,
        "new_train_stored_bytes": sum(item["combined_stored_bytes"]
                                      for item in _flatten(added_train)),
        "new_confirmation_stored_bytes": sum(item["combined_stored_bytes"]
                                             for item in _flatten(confirm)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", required=True)
    for name in ("old_train_manifest", "old_dev_report", "prior_confirmation_report",
                 "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--train-per-stain", type=int, default=20)
    parser.add_argument("--confirm-per-stain", type=int, default=2)
    parser.add_argument("--stain-order", nargs=4, default=["ER", "PGR", "KI67", "HER2"])
    parser.add_argument("--additional-confirm-exclude-selection", type=Path)
    parser.add_argument("--additional-confirm-exclude-case-ids", nargs="*", type=int,
                        default=[])
    parser.add_argument("--search-stain-orders", action="store_true")
    args = parser.parse_args()
    report = choose(args.archive_url, args.old_train_manifest, args.old_dev_report,
                    args.prior_confirmation_report, args.output,
                    train_per_stain=args.train_per_stain,
                    confirm_per_stain=args.confirm_per_stain,
                    stain_order=tuple(args.stain_order),
                    additional_confirm_exclude_selection=args.additional_confirm_exclude_selection,
                    additional_confirm_exclude_case_ids=tuple(args.additional_confirm_exclude_case_ids),
                    search_stain_orders=args.search_stain_orders)
    print(json.dumps({key: report[key] for key in (
        "new_train_ids", "new_confirmation_ids", "new_train_stored_bytes",
        "new_confirmation_stored_bytes")}, indent=2))


if __name__ == "__main__":
    main()
