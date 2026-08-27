# qcopt: differentiable computational QC prototype

This repository is a research reference implementation for facewise variable-μ reconstruction, exact implicit gradients, numerical injectivity auditing, large-distortion registration, prescribed-area deformation, and transition-aware multi-chart registration.

It is intentionally positioned as an **exact sparse baseline**, not as the first differentiable LSQC method. The principal use is to measure gradient fidelity and solver cost, test where μ-space optimization helps, and compare it fairly against safeguarded direct-map optimization.

## Implemented components

- Sparse LBS with arbitrary linear constraints, including a rectangle boundary whose points slide tangentially on their assigned sides.
- Weighted and unweighted LSQC using an augmented saddle system, not normal equations.
- Exact implicit VJP returning gradients for every `Re(mu_T), Im(mu_T)` pair.
- PyTorch CPU custom autograd layers with radial squashing into `|mu| < k_max`.
- Floating-point disk-map audit: positive oriented faces, simple oriented boundary, interior one-ring branch index, and rectangle side membership/order.
- LIM-, SLIM/symmetric-Dirichlet-, and AMIPS-style direct-map baselines with certified line search.
- Analytic I-to-S registration and prescribed-area benchmarks.
- Matched multi-chart LSQC with explicit source/target affine transitions, nonlinear transition Jacobians, and conformal Beltrami covariance diagnostics.
- Independent result recomputation, artifact hashes, and negative corruption tests.

## Runtime

The validated interpreter is:

```text
C:\Users\xuzhehao\anaconda3\python.exe
Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.5.1+cpu
```

This Anaconda installation contains incompatible duplicate Intel OpenMP DLLs in MKL and PyTorch. Set `MKL_THREADING_LAYER=SEQUENTIAL` **before Python starts**; do not use `KMP_DUPLICATE_LIB_OK`.

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:PYTHONPATH='src'
& 'C:\Users\xuzhehao\anaconda3\python.exe' -m pytest -q -W error
```

## Reproduce everything

Quick CI-sized run:

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:PYTHONPATH='src'
& 'C:\Users\xuzhehao\anaconda3\python.exe' -m qcopt.experiments.run_all --output artifacts_quick --quick
```

Full synthetic run used in the report:

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:PYTHONPATH='src'
& 'C:\Users\xuzhehao\anaconda3\python.exe' -m qcopt.experiments.run_all --output artifacts
```

The runner writes experiment configurations, JSON metrics, CSV trajectories/scaling, maps, figures, an independent `audit.json`, environment information, and SHA-256 hashes in `manifest.json`.

## Minimal differentiable-layer example

```python
import torch

from qcopt.autograd import lbs_from_raw
from qcopt.constraints import rectangle_sliding_constraints
from qcopt.mesh import structured_rectangle

mesh = structured_rectangle(16, 16)
constraints = rectangle_sliding_constraints(mesh)
raw_mu = torch.zeros(mesh.n_faces, 2, dtype=torch.double, requires_grad=True)
uv = lbs_from_raw(raw_mu, mesh, constraints, k_max=0.92)
loss = (uv - torch.as_tensor(mesh.vertices.copy())).square().mean()
loss.backward()

assert raw_mu.grad.shape == (mesh.n_faces, 2)
```

`raw_mu.grad[T, 0]` and `raw_mu.grad[T, 1]` are the gradients for the two unconstrained parameters feeding `Re(mu_T)` and `Im(mu_T)`. Call `lbs_layer` or `lsqc_layer` directly when already-bounded μ components are the optimization variables.

## Evidence and interpretation

- [Verified numerical results](docs/results.md)
- [Limitations and unresolved research questions](docs/limitations.md)
- [Design specification](docs/superpowers/specs/2026-08-28-variable-mu-qc-design.md)
- [Implementation plans](docs/superpowers/plans/2026-08-28-core-variable-mu.md)

The word "certified" in metrics means the documented floating-point discrete audit passed. It is not an exact-predicate or formal topological proof, and it does not imply a good lower bound on area or distortion.
