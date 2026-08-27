# Multi-chart Registration and Final Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement transition-aware matched-atlas registration and produce reproducible evidence and a defect audit.

**Architecture:** Charts store local coordinates, overlap point identities, and explicit source/target transitions. Affine transition compatibility is imposed in sparse KKT rows; nonlinear compatibility uses Gauss-Newton. Final auditing recomputes critical metrics from saved maps.

**Tech Stack:** NumPy, SciPy sparse, PyTorch CPU, Matplotlib, pytest, Markdown.

---

### Task 1: Atlas model

**Files:** `src/qcopt/multichart.py`, `tests/test_multichart.py`

- [ ] Write failing tests for overlapping cylinder charts, inverse affine transitions, round trips, and invalid correspondences.
- [ ] Verify RED.
- [ ] Implement immutable `Chart`, `Transition`, `Atlas`, and overlap validation.
- [ ] Verify transition round-trip residuals.
- [ ] Commit `feat: add matched atlas transition model`.

### Task 2: Compatibility constraints

**Files:** `src/qcopt/multichart.py`, `tests/test_chart_compatibility.py`

- [ ] Write a failing test for `g_d(tau_source(p))=tau_target(g_c(p))` plus a negative control proving raw coordinate equality wrong.
- [ ] Verify RED.
- [ ] Assemble affine compatibility rows in global chart unknowns and transition-coordinate seam diagnostics.
- [ ] Verify a manufactured residual `<1e-10`.
- [ ] Commit `feat: enforce transition-aware chart seams`.

### Task 3: Coupled solve

**Files:** `src/qcopt/multichart_solver.py`, `tests/test_multichart_solver.py`

- [ ] Write a failing two-chart manufactured solve with local μ, gauge pins, landmarks, and hard seams.
- [ ] Verify RED.
- [ ] Assemble block-local LSQC residuals plus global compatibility/landmark constraints into one augmented sparse solve.
- [ ] Verify reconstruction, landmarks, and seams together.
- [ ] Commit `feat: solve coupled multi-chart QC maps`.

### Task 4: Differential covariance/nonlinear transitions

**Files:** `src/qcopt/multichart.py`, `src/qcopt/multichart_solver.py`, `tests/test_chart_covariance.py`

- [ ] Write failing conformal/Möbius covariance and nonlinear-transition linearization tests.
- [ ] Verify RED.
- [ ] Implement transition Jacobians, Beltrami covariance diagnostics, and Gauss-Newton seam residual/Jacobian updates.
- [ ] Verify covariance `<1e-8` and residual reduction.
- [ ] Commit `feat: add nonlinear chart transition support`.

### Task 5: Cylinder experiment

**Files:** `src/qcopt/experiments/multichart_registration.py`, `tests/test_multichart_experiment.py`

- [ ] Write a failing smoke test requiring transition-aware and raw-equality controls, landmarks/features, angular distortion, seams, and chart certificates.
- [ ] Verify RED.
- [ ] Implement source/deformed-target cylinder atlases, coupled optimization, deterministic outputs, and plots.
- [ ] Run smoke/full experiments and inspect seam maps and metrics.
- [ ] Commit `experiment: validate multi-chart surface registration`.

### Task 6: Runner and independent audit

**Files:** `src/qcopt/experiments/{run_all,audit_results}.py`, `tests/test_run_all.py`, `tests/test_audit_results.py`, `README.md`, `docs/{results,limitations}.md`

- [ ] Write failing quick end-to-end, manifest, and corruption-detection tests for folds, areas, seams, gradients, and missing artifacts.
- [ ] Verify RED.
- [ ] Implement quick/full runners, environment capture, hashes, independent recomputation, and evidence-linked reports.
- [ ] Run the complete suite, all full experiments, and requirement-by-requirement audit.
- [ ] Commit `docs: report verified QC prototype results and limitations`.
