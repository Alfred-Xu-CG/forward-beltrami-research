"""Export the exact resized input tensor used by a translation-training runtime."""

import argparse
from pathlib import Path

import numpy as np

from tools.digital_q1_real_optimize import _read_gray_thumbnail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tensor, _ = _read_gray_thumbnail(args.image, args.side)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, tensor.numpy())
    print(f"{args.output}: {tuple(tensor.shape)} {tensor.dtype}")


if __name__ == "__main__":
    main()
