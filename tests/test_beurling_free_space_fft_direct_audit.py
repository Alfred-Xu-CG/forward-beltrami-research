from pathlib import Path

from qcopt.experiments.beurling_free_space_fft_direct_audit import compute_fft
from qcopt.experiments.beurling_free_space_fft_direct_audit import generate


def test_free_space_fft_audit_generates_disjoint_target_outputs(tmp_path: Path):
    data = tmp_path / "data.npz"
    out = tmp_path / "out"
    result = generate(data, n=32, target_side=4)
    assert result["source_count"] == 1024
    records = compute_fft(data, out, (2, 4))
    assert set(records) == {"2", "4"}
    assert (out / "fft_padding_2.npy").exists()
    assert (out / "fft_padding_4.npy").exists()
