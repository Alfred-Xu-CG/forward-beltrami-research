"""Deterministic fresh-case selection operates on archive metadata only."""

from dataclasses import dataclass

from tools.digital_acrobat_select_confirmation import select_cases


@dataclass
class Member:
    filename: str
    compress_size: int


def test_selection_excludes_seen_missing_and_oversize_cases():
    members = []
    for stain_index, stain in enumerate(("ER", "PGR", "KI67", "HER2")):
        for local in range(1, 8):
            case = 100 * stain_index + local
            members.extend((Member(f"{case}_HE.tiff", 100),
                            Member(f"{case}_{stain}.tiff", 100)))
    members += [Member("99_HE.tiff", 100), Member("99_ER.tiff", 100)]
    members += [Member("98_HE.tiff", 300_000_000),
                Member("98_ER.tiff", 300_000_000)]
    first = select_cases(members, seed=20260930, exclude={99}, per_stain=2)
    second = select_cases(list(reversed(members)), seed=20260930,
                          exclude={99}, per_stain=2)
    assert first == second
    ids = [item["case"] for group in first.values() for item in group]
    assert len(ids) == 8 and len(set(ids)) == 8
    assert 98 not in ids and 99 not in ids
