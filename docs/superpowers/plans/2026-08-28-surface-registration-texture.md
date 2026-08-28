# Surface Registration Texture Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render dense source textures and inverse-map error fields on fixed high-genus target meshes for saved registration trials.

**Architecture:** A focused experiment module reconstructs the perturbation prefix from the saved full and correction histories, applies inverse flows at target face centroids, transfers locations through the paired common refinement, and evaluates one source-defined texture on fixed target geometry. A small CLI loads the same official/synthetic cases as the experiment runner and writes static Matplotlib evidence, a rotatable Plotly HTML comparison, and machine-readable metrics beside each trial.

**Tech Stack:** Python, NumPy, SciPy surface locator, Matplotlib Agg, Plotly, pytest.

---

### Task 1: Pullback and texture primitives

**Files:**
- Create: `tests/test_surface_registration_texture.py`
- Create: `src/qcopt/experiments/surface_registration_texture.py`

- [ ] **Step 1: Write failing tests**

Test that identity history returns the common-map source positions, that a common-refinement barycentric point transfers exactly, that procedural RGB values are finite and lie in `[0, 1]`, and that the perturbation prefix is recovered from full/correction histories.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_surface_registration_texture.py -q`

Expected: collection fails because `qcopt.experiments.surface_registration_texture` does not exist.

- [ ] **Step 3: Implement minimal primitives**

Add `split_perturbation_history`, `pullback_source_positions`, `source_texture_rgb`, and dataclasses carrying locator/flow diagnostics. Use `points_at_face_centroids`, `FlowHistory.inverse`, and `SurfaceLocator.locate_many`; transfer each located common-target face/barycentric coordinate to `common.source`. A regression test must fail if the implementation samples P2-fixed mesh vertices instead.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_surface_registration_texture.py -q`

Expected: all focused tests pass.

### Task 2: Deterministic surface renderers

**Files:**
- Modify: `tests/test_surface_registration_texture.py`
- Modify: `src/qcopt/experiments/surface_registration_texture.py`

- [ ] **Step 1: Add a failing image-output test**

Construct a small closed case with identity histories, render all artifacts into a temporary directory, and assert nonempty PNGs, a parseable Plotly HTML file containing four 3D scenes, plus a JSON record with zero pullback error.

- [ ] **Step 2: Verify RED**

Run the focused test and confirm it fails because the rendering entry point is absent.

- [ ] **Step 3: Implement rendering and CLI**

Add PCA display alignment, per-face texture conversion, fixed-camera `Poly3DCollection` panels, shared-scale error heatmaps, Plotly `Mesh3d` panels with linked target cameras, JSON serialization, case loading, and CLI arguments `--case`, `--run-directory`, `--data-root`, and optional output paths.

- [ ] **Step 4: Verify GREEN**

Run the complete focused test file and confirm all assertions pass.

### Task 3: Real-data artifacts

**Files:**
- Create generated PNG/JSON files below representative `artifacts/high_genus_registration_*` trial directories.
- Modify: `docs/high_genus_registration_results.md`

- [ ] **Step 1: Render representative trials**

Run the CLI for synthetic genus-2 combined seed 3, official genus-3 landmark seed 3, official genus-5 landmark seed 29, and official pretzel genus-3 landmark seed 3.

- [ ] **Step 2: Inspect every generated image**

Open each PNG at original detail, verify readable texture isolines, identical target views, non-clipped titles/colorbars, and visibly distinct initial/refined fields.

- [ ] **Step 3: Document how to read and reproduce the figures**

Add the exact inverse-pullback definition, representative artifact paths, CLI commands, and the caveat about published-map reference truth.

### Task 4: Independent verification

**Files:**
- Modify only files needed to address verified failures.

- [ ] **Step 1: Recompute JSON metrics independently from saved histories**

Reload the four cases and compare reported vertexwise error statistics against a fresh call to the pullback function.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_surface_registration_texture.py -q`

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`

- [ ] **Step 4: Review the final diff and commit**

Check `git diff --check`, inspect `git status --short`, and commit source, tests, documentation, and reproducible visualization artifacts.
