"""Portable NPZ serialization for independently replayable surface flows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .surface_flow import FlowHistory, FlowStep, IntrinsicFaceField


def save_flow_history(path: str | Path, history: FlowHistory) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray([1], dtype=np.int64),
        "step_count": np.asarray([len(history.steps)], dtype=np.int64),
        "durations": np.asarray([step.duration for step in history.steps]),
        "substeps": np.asarray([step.substeps for step in history.steps], dtype=np.int64),
        "types": np.asarray(
            [
                "intrinsic_p2"
                if isinstance(step.vertex_field, IntrinsicFaceField)
                else "vertex"
                for step in history.steps
            ],
            dtype="<U16",
        ),
    }
    for index, step in enumerate(history.steps):
        prefix = f"step_{index:04d}"
        if isinstance(step.vertex_field, IntrinsicFaceField):
            payload[f"{prefix}_edge"] = step.vertex_field.edge_values
            payload[f"{prefix}_bubble"] = step.vertex_field.bubble_values
        else:
            payload[f"{prefix}_vertex"] = step.vertex_field
    np.savez_compressed(path, **payload)


def load_flow_history(path: str | Path) -> FlowHistory:
    with np.load(Path(path), allow_pickle=False) as archive:
        version = int(archive["format_version"][0])
        if version != 1:
            raise ValueError(f"unsupported flow history format version: {version}")
        count = int(archive["step_count"][0])
        types = archive["types"]
        durations = archive["durations"]
        substeps = archive["substeps"]
        if not (len(types) == len(durations) == len(substeps) == count):
            raise ValueError("inconsistent flow history metadata")
        steps: list[FlowStep] = []
        for index in range(count):
            prefix = f"step_{index:04d}"
            kind = str(types[index])
            if kind == "intrinsic_p2":
                field: np.ndarray | IntrinsicFaceField = IntrinsicFaceField(
                    archive[f"{prefix}_edge"],
                    archive[f"{prefix}_bubble"],
                )
            elif kind == "vertex":
                field = archive[f"{prefix}_vertex"]
            else:
                raise ValueError(f"unknown flow field type: {kind}")
            steps.append(
                FlowStep(field, float(durations[index]), int(substeps[index]))
            )
    return FlowHistory(tuple(steps))
