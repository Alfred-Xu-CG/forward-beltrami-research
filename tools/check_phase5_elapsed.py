"""Enforce the Phase V real wall-clock review and completion gates."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FINAL_REVIEW_SECONDS = 17 * 3600 + 15 * 60
COMPLETION_SECONDS = 18 * 3600
DEFAULT_START_FILE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "research_phase5"
    / "START_TIME.json"
)


@dataclass(frozen=True)
class Phase5Elapsed:
    start_unix_seconds: int
    now_unix_seconds: int
    elapsed_seconds: int
    final_review_allowed: bool
    completion_allowed: bool


def read_start_receipt(path: Path = DEFAULT_START_FILE) -> dict[str, Any]:
    """Read and validate the immutable start receipt without modifying it."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"start_iso8601_utc", "start_unix_seconds"}
    if set(payload) != required:
        raise ValueError(f"start receipt must contain exactly {sorted(required)}")
    if not isinstance(payload["start_iso8601_utc"], str):
        raise ValueError("start_iso8601_utc must be a string")
    if not isinstance(payload["start_unix_seconds"], int):
        raise ValueError("start_unix_seconds must be an integer")
    return payload


def evaluate_elapsed(start_unix_seconds: int, now_unix_seconds: int) -> Phase5Elapsed:
    """Evaluate both gates at explicit integer UTC timestamps."""
    elapsed = now_unix_seconds - start_unix_seconds
    if elapsed < 0:
        raise ValueError("system clock precedes Phase V start")
    return Phase5Elapsed(
        start_unix_seconds=start_unix_seconds,
        now_unix_seconds=now_unix_seconds,
        elapsed_seconds=elapsed,
        final_review_allowed=elapsed >= FINAL_REVIEW_SECONDS,
        completion_allowed=elapsed >= COMPLETION_SECONDS,
    )


def current_status(path: Path = DEFAULT_START_FILE) -> Phase5Elapsed:
    """Read the receipt and evaluate it against the real system UTC clock."""
    payload = read_start_receipt(path)
    return evaluate_elapsed(payload["start_unix_seconds"], int(time.time()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        choices=("status", "review", "complete"),
        default="status",
    )
    parser.add_argument("--start-file", type=Path, default=DEFAULT_START_FILE)
    args = parser.parse_args()
    status = current_status(args.start_file)
    print(json.dumps(asdict(status), indent=2, sort_keys=True))

    allowed = {
        "status": True,
        "review": status.final_review_allowed,
        "complete": status.completion_allowed,
    }[args.gate]
    if not allowed:
        print("EARLY_FINALIZATION_DENIED")
        return 2
    if args.gate == "review":
        print("FINAL_REVIEW_ALLOWED")
    elif args.gate == "complete":
        print("PHASE5_COMPLETION_ALLOWED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
