# Forward Beltrami Phase 0 Implementation Plan

> **For agentic workers:** Execute this plan inline task-by-task with TDD checkpoints. Do not modify the existing LSQC/LBS files while Phase 0 is being built.

**Goal:** Build a shared, independently tested P1 benchmark core for all forward Beltrami routes.

**Architecture:** Reuse the repository's immutable `TriMesh` as the source mesh, and add a new `qcopt.forward` namespace containing manufactured-map generation, exact facewise Beltrami/Jacobian utilities, periodic-grid metadata, and topology metrics. The benchmark runner writes only JSON/CSV/NPZ artifacts under an explicitly supplied D-drive output directory. Existing LSQC/LBS implementations remain reference baselines and are not changed in this phase.

**Tech Stack:** Python 3.12, NumPy, SciPy, pytest, existing `qcopt.mesh`, `qcopt.beltrami`, and `qcopt.injectivity`.

---

### Task 1: Establish the forward benchmark API with failing tests

**Files:**
- Create: `tests/test_forward_phase0.py`
- Create: `src/qcopt/forward/__init__.py`
- Create: `src/qcopt/forward/benchmarks.py`
- Create: `src/qcopt/forward/operators.py`

- [ ] **Step 1: Write the failing tests**

  Tests must cover four behaviors: a manufactured affine map has constant exact Beltrami coefficient; a smooth manufactured map preserves positive face Jacobians on a moderate grid; periodic indexing has the expected wrap-around neighbors; and the benchmark metric reports zero residual when evaluated on its own manufactured map.

- [ ] **Step 2: Run the focused tests and verify the expected failure**

  Run: `$env:PYTHONPATH='src'; & 'C:\Users\xuzhehao\anaconda3\python.exe' -m pytest -q tests/test_forward_phase0.py`

  Expected: collection or import failures because `qcopt.forward` and its functions do not yet exist.

- [ ] **Step 3: Implement the minimal API**

  Implement `affine_map`, `smooth_twist_map`, `manufactured_mu`, `periodic_grid_edges`, `face_jacobian_determinants`, and `beltrami_residual` using existing `TriMesh`/`face_beltrami` primitives. Reject shape mismatches and non-finite inputs.

- [ ] **Step 4: Run the focused tests and verify they pass**

  Run the same pytest command. Expected: all Phase 0 unit tests pass with no warnings.

---

### Task 2: Add independent topology and scalability metrics

**Files:**
- Modify: `src/qcopt/forward/operators.py`
- Modify: `tests/test_forward_phase0.py`

- [ ] **Step 1: Add failing metric tests**

  Test that the identity and a positive affine map return positive minimum signed area, zero flips, and zero Beltrami equation residual; test that a deliberately flipped triangle is reported as non-certifiable without using the solver residual as evidence.

- [ ] **Step 2: Run the tests and verify the new assertions fail**

  Run: `$env:PYTHONPATH='src'; & 'C:\Users\xuzhehao\anaconda3\python.exe' -m pytest -q tests/test_forward_phase0.py -k topology`

- [ ] **Step 3: Implement metrics**

  Add a frozen `ForwardMetrics` dataclass and `evaluate_forward_metrics(mesh, uv, target_mu=None)` returning `min_det`, `flipped_faces`, `mu_l2`, `mu_linf`, and `equation_linf`. Keep topology evidence independent by calling `audit_injectivity` separately and storing its result fields rather than inferring topology from residuals.

- [ ] **Step 4: Run the focused and full existing tests**

  Run the focused test, then `$env:MKL_THREADING_LAYER='SEQUENTIAL'; $env:PYTHONPATH='src'; & 'C:\Users\xuzhehao\anaconda3\python.exe' -m pytest -q -W error`. Existing dirty-tree tests must remain untouched.

---

### Task 3: Add a D-drive benchmark runner

**Files:**
- Create: `src/qcopt/experiments/forward_phase0.py`
- Create: `tests/test_forward_phase0_runner.py`

- [ ] **Step 1: Write the failing runner test**

  Invoke the runner on `structured_rectangle(32, 24)` into a pytest temporary directory and assert that it writes `config.json`, `metrics.json`, and `maps.npz`, with explicit mesh size and finite metrics.

- [ ] **Step 2: Run the runner test and verify it fails**

  Run: `$env:PYTHONPATH='src'; & 'C:\Users\xuzhehao\anaconda3\python.exe' -m pytest -q tests/test_forward_phase0_runner.py`

- [ ] **Step 3: Implement the runner**

  Use an explicit `--output` path, refuse missing/ambiguous output roots, and write only JSON/NPZ artifacts. Include deterministic seed, grid resolution, method name, and SHA-256 hashes in `manifest.json`.

- [ ] **Step 4: Run the runner test and inspect artifacts**

  Run the same test, then run a 256x256 grid locally and inspect the JSON metrics before dispatching the same runner to remote CPU/GPU hosts.

---

### Task 4: Realistic-resolution remote benchmark

**Files:**
- Create: `docs/forward_beltrami/phase0_results.md`
- Create: `artifacts/forward_phase0/` (generated only under D:)

- [ ] **Step 1: Run 256x256 and 512x512 CPU benchmarks on turing and element**

  Use the new SSH sessions and explicit D-drive output directories. Record wall time, peak resident memory, face count, minimum determinant, and residuals.

- [ ] **Step 2: Run 512x512 and 1024x1024 GPU-compatible manufactured-map metrics on ai**

  Use only an idle GPU selected from a fresh `nvidia-smi` check; do not kill or interfere with existing jobs. If any SSH session drops, pause the goal immediately and report the exact host/error.

- [ ] **Step 3: Independently recompute metrics**

  Reload `maps.npz` and recompute all reported metrics in a separate process. Require exact agreement within recorded floating-point tolerances before describing the phase as validated.

