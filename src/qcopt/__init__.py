"""Differentiable computational quasiconformal geometry tools.

The supported Anaconda runtime ships different Intel OpenMP copies with MKL
and PyTorch.  A sequential MKL backend avoids loading the second runtime and
also makes timing/gradient reference runs deterministic.  Users can override
this before importing :mod:`qcopt` when using a repaired environment.
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from .mesh import TriMesh, structured_rectangle

__all__ = ["TriMesh", "structured_rectangle"]
