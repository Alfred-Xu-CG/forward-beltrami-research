"""Train/evaluate phase separation for unseen ACROBAT cases."""

from pathlib import Path

import pytest

from tools.digital_acrobat_fresh_probe import validate_train_ids


def test_train_ids_reject_duplicate_and_nonpositive():
    assert validate_train_ids([100, 156, 315]) == [100, 156, 315]
    with pytest.raises(ValueError):
        validate_train_ids([100, 100])
    with pytest.raises(ValueError):
        validate_train_ids([0, 100])
    with pytest.raises(ValueError):
        validate_train_ids([])


def test_eval_ids_must_be_disjoint_from_frozen_training_manifest(tmp_path: Path):
    from tools.digital_acrobat_fresh_probe import validate_eval_ids

    assert validate_eval_ids([73, 193], [100, 156]) == [73, 193]
    with pytest.raises(ValueError):
        validate_eval_ids([73, 100], [100, 156])
    with pytest.raises(ValueError):
        validate_eval_ids([73, 73], [100, 156])
