"""Load frozen official DINOv2 patch features for safe-registration probing."""

from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14")
    model = model.to(args.device).eval()
    image = torch.zeros((1, 3, 448, 448), device=args.device)
    with torch.no_grad():
        features = model.forward_features(image)["x_norm_patchtokens"]
    print(json.dumps({"patch_tokens": list(features.shape),
                      "finite": bool(torch.isfinite(features).all()),
                      "load_and_first_forward_seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
