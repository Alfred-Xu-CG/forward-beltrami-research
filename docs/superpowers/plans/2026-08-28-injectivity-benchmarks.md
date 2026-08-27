# Injectivity and Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent disk-map auditing, certified μ line search, fair direct-map safeguards, and both synthetic benchmarks.

**Architecture:** Training energies and acceptance certificates remain separate. Shared deterministic loops compare μ-space and direct-f methods under the same mesh, loss, initialization, seed, and budget.

**Tech Stack:** NumPy, SciPy, PyTorch CPU, Matplotlib, scikit-image, pytest.

---

### Task 1: Disk-map certificate

**Files:** `src/qcopt/injectivity.py`, `tests/test_injectivity.py`

- [ ] Write failing identity and negative-control tests for a flip, boundary crossing, branch-index violation, and rectangle side-order violation.
- [ ] Verify RED.
- [ ] Implement tolerance-aware segment intersections, boundary simplicity/orientation, positive faces, one-ring branch index, and side membership/order in `InjectivityReport`.
- [ ] Verify each negative control fails for its intended reason.
- [ ] Commit `feat: add discrete injectivity audit`.

### Task 2: Map energies

**Files:** `src/qcopt/energies.py`, `tests/test_energies.py`

- [ ] Write failing value/gradient tests for log-det, symmetric Dirichlet, LIM-style reciprocal determinant, and AMIPS-style energies.
- [ ] Verify RED.
- [ ] Implement stable torch energies with explicit infeasible sentinels.
- [ ] Verify autograd against finite differences and invariance tests.
- [ ] Commit `feat: add direct-map injectivity energies`.

### Task 3: Optimizers

**Files:** `src/qcopt/optimize_mu.py`, `src/qcopt/optimize_map.py`, `tests/test_optimizers.py`

- [ ] Write failing tests where a μ proposal folds but bisection finds a certified intermediate, and where each direct-f method reports infeasible trials.
- [ ] Verify RED.
- [ ] Implement Adam proposals plus resolve-and-audit μ backtracking, and common-budget `none/lim_style/slim_style/amips_style` direct-map loops.
- [ ] Verify certificate persistence, identical stopping rules, and complete history.
- [ ] Commit `feat: add certified and baseline optimizers`.

### Task 4: I/S fields and benchmark

**Files:** `src/qcopt/registration.py`, `src/qcopt/experiments/i_to_s.py`, `tests/test_registration.py`, `tests/test_i_to_s.py`

- [ ] Write failing tests for bounded differentiable thick-I/S fields and a reduced benchmark requiring all six methods and common metrics.
- [ ] Verify RED.
- [ ] Implement smooth signed-distance compositions, mismatch/soft Dice, deterministic trials, trajectory CSV, metrics JSON, and deformation/image figures.
- [ ] Run smoke and full experiments; verify artifact manifests.
- [ ] Commit `experiment: compare large-distortion registration methods`.

### Task 5: Area targets and benchmark

**Files:** `src/qcopt/density.py`, `src/qcopt/experiments/density_equalizing.py`, `tests/test_density.py`, `tests/test_density_experiment.py`

- [ ] Write failing tests for normalization, known-map manufactured factors, checkerboard/spike/random stress fields, and output feasibility metadata.
- [ ] Verify RED.
- [ ] Implement differentiable log-area/relative-area losses and identical μ/direct-f comparisons with complete metrics/artifacts.
- [ ] Independently recompute manufactured areas, target sums, residuals, and fold counts.
- [ ] Commit `experiment: benchmark prescribed-area deformation`.
