"""Create one compact D-drive thumbnail archive for slide holdout experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.digital_q1_real_optimize import _read_gray_thumbnail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--side", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.side < 16:
        raise ValueError("side must be >=16")
    images = []
    for path in args.image:
        image, _ = _read_gray_thumbnail(path, args.side)
        images.append(image.numpy()[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
                        images=np.stack(images).astype(np.float32),
                        sources=np.asarray([str(p) for p in args.image]))


if __name__ == "__main__":
    main()
