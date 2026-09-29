"""Exploratory BIRL prediction with the frozen-base recurrent Q1 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.digital_acrobat_recurrent_probe import _load_model
from tools.digital_birl_frozen_predict import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "moving", "initial_affine", "recurrent_output", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = _load_model(args.recurrent_output, args.device)
    result = predict(
        args.fixed, args.moving, args.initial_affine,
        args.recurrent_output / "recurrent_heads.npz", args.output,
        device=args.device, supplied_model=model,
    )
    result["mode"] = "exploratory_recurrent_BIRL_development_prediction"
    result["confirmation_teacher_previously_opened_before_architecture_test"] = True
    (args.output / "prediction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
