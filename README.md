# qcopt: differentiable computational QC prototype

This repository is a research reference implementation for facewise variable-μ reconstruction, exact implicit gradients, numerical injectivity auditing, large-distortion registration, prescribed-area deformation, and transition-aware multi-chart registration.

It is intentionally positioned as an **exact sparse baseline**, not as the first differentiable LSQC method. The principal use is to measure gradient fidelity and solver cost, test where μ-space optimization helps, and compare it fairly against safeguarded direct-map optimization.

## Implemented components

- Sparse LBS with arbitrary linear constraints, including a rectangle boundary whose points slide tangentially on their assigned sides.
- Weighted and unweighted LSQC using an augmented saddle system as the general
  reference backend.
- Fast weighted LSQC for hard coordinate-selector constraints: direct paper
  Hessian assembly, exact pin elimination, fill-reducing symmetric sparse LU,
  and factorization reuse in the adjoint.
- Exact implicit VJP returning gradients for every `Re(mu_T), Im(mu_T)` pair.
- PyTorch CPU custom autograd layers with radial squashing into `|mu| < k_max`.
- Floating-point disk-map audit: positive oriented faces, simple oriented boundary, interior one-ring branch index, and rectangle side membership/order.
- LIM-, SLIM/symmetric-Dirichlet-, and AMIPS-style direct-map baselines with certified line search.
- Analytic I-to-S registration and prescribed-area benchmarks.
- Matched multi-chart LSQC with explicit source/target affine transitions, nonlinear transition Jacobians, and conformal Beltrami covariance diagnostics.
- Native PL-atlas registration for closed genus-2, genus-3, and genus-5
  surfaces: global `(target face id, barycentric coordinates)` map state,
  dynamic edge crossing, landmark/curvature refinement, reversible flow
  history, and an independent numerical homeomorphism audit.
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

High-genus native multi-chart suite (requires the separately downloaded
official archive described in
[the dataset inventory](docs/high_genus_dataset_inventory.md)):

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:PYTHONPATH='src'
& 'C:\Users\xuzhehao\anaconda3\python.exe' -m qcopt.experiments.high_genus_registration `
  --preset primary-2026-08-28 `
  --data-root external_data/s2020-intersurfacemaps-data `
  --output artifacts/high_genus_registration
```

Every trial stores its surface-point states and reversible flow history.  Use
`qcopt.experiments.audit_high_genus_registration` with explicit trial
directories to reload and recompute the states, objectives, certificate, and
artifact hashes.  The accepted 36-trial matrix and exact commands are recorded
in [the high-genus validation report](docs/high_genus_registration_results.md).

The runner writes experiment configurations, JSON metrics, CSV trajectories/scaling, maps, figures, an independent `audit.json`, environment information, and SHA-256 hashes in `manifest.json`.

## Deliver to the requested D-drive directory

After granting the current Windows account Modify permission on `D:\QC_optimization`, run:

```powershell
& .\scripts\sync_delivery.ps1
```

The delivery script refuses a nonempty unrelated target, copies hidden Git state as well as source/tests/artifacts, verifies every artifact byte count and SHA-256 entry in `artifacts/manifest.json`, and verifies the destination Git HEAD when both repositories are valid.

## Minimal differentiable-layer example

```python
import torch

from qcopt.autograd import lsqc_fast_from_raw
from qcopt.constraints import two_pin_constraints
from qcopt.mesh import structured_rectangle

mesh = structured_rectangle(16, 16)
constraints = two_pin_constraints(
    mesh.n_vertices,
    [0, mesh.n_vertices - 1],
    mesh.vertices[[0, mesh.n_vertices - 1]],
)
raw_mu = torch.zeros(mesh.n_faces, 2, dtype=torch.double, requires_grad=True)
uv = lsqc_fast_from_raw(raw_mu, mesh, constraints, k_max=0.92)
loss = (uv - torch.as_tensor(mesh.vertices.copy())).square().mean()
loss.backward()

assert raw_mu.grad.shape == (mesh.n_faces, 2)
```

`raw_mu.grad[T, 0]` and `raw_mu.grad[T, 1]` are the gradients for the two
unconstrained parameters feeding `Re(mu_T)` and `Im(mu_T)`. Call
`lsqc_fast_layer` directly when already-bounded μ components are the
optimization variables. The fast backend is weighted LSQC and accepts exact
coordinate selectors: two complex pins, fixed vertices, or axis-aligned
rectangle sliding constraints. Use `lsqc_layer` for unweighted LSQC or mixed
linear constraints.

## Reproducible LSQC backend benchmark

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:PYTHONPATH='src'
& 'C:\Users\xuzhehao\anaconda3\python.exe' `
  -m qcopt.experiments.benchmark_lsqc `
  --output artifacts/fast_lsqc_benchmark/2026-08-31 `
  --grid-sizes 12 24 36 48 `
  --repeats 7
```

The JSON and CSV outputs include matrix dimensions/nonzeros, LU fill, assembly,
forward, isolated adjoint-solve and contraction costs, total backward and
independently timed complete forward-plus-backward steps, residuals, and
map/gradient discrepancies against the augmented weighted-LSQC reference.
Timings are reported rather than asserted in tests because they are
machine-dependent.

## Evidence and interpretation

- [Verified numerical results](docs/results.md)
- [Fast weighted-LSQC implementation and benchmark](docs/fast_weighted_lsqc_results.md)
- [High-genus registration validation](docs/high_genus_registration_results.md)
- [Official high-genus dataset inventory](docs/high_genus_dataset_inventory.md)
- [Limitations and unresolved research questions](docs/limitations.md)
- [Design specification](docs/superpowers/specs/2026-08-28-variable-mu-qc-design.md)
- [Implementation plans](docs/superpowers/plans/2026-08-28-core-variable-mu.md)

The word "certified" in metrics means the documented floating-point discrete audit passed. It is not an exact-predicate or formal topological proof, and it does not imply a good lower bound on area or distortion.

## Forward Beltrami research archive

The ongoing forward-Beltrami exploration is synchronized to the private
[forward-beltrami-research GitHub repository](https://github.com/Alfred-Xu-CG/forward-beltrami-research).
The self-contained route summary starts at
[`docs/forward_beltrami/summary/00_executive_answer.md`](docs/forward_beltrami/summary/00_executive_answer.md),
and the formal equation-by-equation route formulation is in
[`docs/forward_beltrami/summary/formal_route_formulations.md`](docs/forward_beltrami/summary/formal_route_formulations.md).
The latter maps the equations directly to the implemented operators and
explicitly separates local numerical evidence from global theorems.
and the repository workflow is documented in
[`RESEARCH_REPOSITORY.md`](RESEARCH_REPOSITORY.md). Use
`powershell -ExecutionPolicy Bypass -File scripts/sync_research.ps1` after a
reviewed change to synchronize compact receipts and documentation.
