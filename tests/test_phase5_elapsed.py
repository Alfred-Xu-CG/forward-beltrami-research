from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.check_phase5_elapsed import evaluate_elapsed, read_start_receipt


def test_elapsed_gate_uses_exact_review_and_completion_thresholds() -> None:
    start = 1_000_000

    before_review = evaluate_elapsed(start, start + 17 * 3600 + 14 * 60 + 59)
    at_review = evaluate_elapsed(start, start + 17 * 3600 + 15 * 60)
    before_complete = evaluate_elapsed(start, start + 17 * 3600 + 59 * 60 + 59)
    at_complete = evaluate_elapsed(start, start + 18 * 3600)

    assert not before_review.final_review_allowed
    assert at_review.final_review_allowed
    assert not before_complete.completion_allowed
    assert at_complete.completion_allowed


def test_read_start_receipt_is_read_only(tmp_path: Path) -> None:
    receipt = tmp_path / "START_TIME.json"
    payload = {
        "start_iso8601_utc": "2026-09-21T16:10:26Z",
        "start_unix_seconds": 1_790_007_026,
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    original = receipt.read_bytes()

    parsed = read_start_receipt(receipt)

    assert parsed == payload
    assert receipt.read_bytes() == original


def test_elapsed_rejects_clock_before_start() -> None:
    with pytest.raises(ValueError, match="precedes Phase V start"):
        evaluate_elapsed(100, 99)
