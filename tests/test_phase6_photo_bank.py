"""Keep the original grayscale photo protocol while allowing held-out RGB content."""

import torch

from phase6_eval_photographic_content import _photo_bank, _dataset


def test_default_photo_bank_matches_explicit_original_six():
    names, default = _photo_bank(32)
    explicit_names, explicit = _photo_bank(32, names)
    assert names == explicit_names
    assert default.shape == (6, 1, 32, 32)
    assert torch.equal(default, explicit)


def test_unseen_rgb_photo_bank_and_variant_count():
    names = ["astronaut", "coffee", "chelsea", "rocket"]
    selected, bank = _photo_bank(32, names)
    assert selected == names
    assert bank.shape == (4, 1, 32, 32)
    assert torch.isfinite(bank).all()
    actual_names, photo_ids, dataset = _dataset(
        2, 32, 703231, torch.device("cpu"), names)
    assert actual_names == names
    assert photo_ids == [0, 0, 1, 1, 2, 2, 3, 3]
    assert dataset[0].shape == (8, 1, 32, 32)
    _, _, high64 = _dataset(2, 32, 703231, torch.device("cpu"), names,
                            target_family="high64")
    torch.testing.assert_close(high64[3][:, :2], dataset[3][:, :2])
    torch.testing.assert_close(high64[3][:, 2], 0.5 * dataset[3][:, 2])
