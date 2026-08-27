# Core Variable-μ Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build sparse LBS/LSQC forward solvers and exact facewise implicit gradients with manufactured validation.

**Architecture:** Mesh geometry supplies constant face-gradient operators. LBS and augmented LSQC assemble sparse KKT systems. Custom PyTorch operations invoke SciPy factorization and analytic local VJPs.

**Tech Stack:** Python, NumPy, SciPy sparse, PyTorch CPU, pytest.

---

### Task 1: Package and mesh geometry

**Files:** `pyproject.toml`, `src/qcopt/{__init__,mesh}.py`, `tests/test_mesh.py`

- [ ] Write a failing unit-triangle test requiring area `0.5`, affine-coordinate gradients `[1,0]`, `[0,1]`, valid structured rectangle connectivity, and one ordered boundary loop.
- [ ] Run `python -m pytest tests/test_mesh.py -v`; require failure because `qcopt.mesh` is absent.
- [ ] Implement immutable `TriMesh`, orientation validation, areas, barycentric gradients, boundary extraction, and `structured_rectangle(nx,ny)`.
- [ ] Re-run focused and full tests; require PASS.
- [ ] Commit `feat: add validated triangular mesh geometry`.

### Task 2: Beltrami primitives

**Files:** `src/qcopt/beltrami.py`, `tests/test_beltrami.py`

- [ ] Write failing tests for identity μ, analytic diagonal affine μ `(a-b)/(a+b)`, signed Jacobians, dilation, and radial-squash bound/zero limit.
- [ ] Verify RED with focused pytest.
- [ ] Implement `face_jacobians`, `face_beltrami`, `qc_dilation`, and stable torch `radial_squash`.
- [ ] Verify values and torch gradcheck.
- [ ] Commit `feat: add discrete Beltrami primitives`.

### Task 3: Constraints

**Files:** `src/qcopt/constraints.py`, `tests/test_constraints.py`

- [ ] Write failing tests for two complex pins, fixed vertices, and rectangle side-sliding rows.
- [ ] Verify RED.
- [ ] Implement `LinearConstraints(C,d)`, `two_pin_constraints`, `fixed_vertex_constraints`, and `rectangle_sliding_constraints`; corners receive both required rows.
- [ ] Verify rank, target values, and tangential freedom.
- [ ] Commit `feat: add QC map linear constraints`.

### Task 4: Sparse LBS

**Files:** `src/qcopt/lbs.py`, `tests/test_lbs.py`

- [ ] Write failing identity/affine manufactured tests and a rejection test for `|μ|>=1`.
- [ ] Verify RED.
- [ ] Implement tensor `A(μ)`, local stiffness assembly, block KKT construction, SuperLU solve, and residual diagnostics in `SolveResult(uv, primal_residual, constraint_residual, state)`.
- [ ] Verify reconstruction and `<1e-10` residuals.
- [ ] Commit `feat: implement sparse variable-mu LBS`.

### Task 5: Augmented LSQC

**Files:** `src/qcopt/lsqc.py`, `tests/test_lsqc.py`

- [ ] Write failing tests for the two residual rows, identity/affine reconstruction, similarity gauge, and weighted/unweighted distinction.
- [ ] Verify RED.
- [ ] Implement `assemble_lsqc_operator` and augmented `solve_lsqc` without `B.T@B`.
- [ ] Verify exact manufactured cases and compare augmented versus normal-equation conditioning.
- [ ] Commit `feat: implement augmented LSQC solver`.

### Task 6: Exact adjoints

**Files:** `src/qcopt/adjoint.py`, `tests/test_adjoint.py`

- [ ] Write failing all-face directional finite-difference tests for LBS and weighted/unweighted LSQC.
- [ ] Verify RED.
- [ ] Implement local matrix derivatives and `-z.T @ dM @ y` using one transpose solve.
- [ ] Run step sizes `1e-2...1e-7`; require stable-window median relative error `<1e-5`.
- [ ] Commit `feat: add exact sparse QC adjoints`.

### Task 7: PyTorch layers

**Files:** `src/qcopt/autograd.py`, `tests/test_autograd.py`

- [ ] Write failing double-precision gradcheck tests for LBS and LSQC map losses.
- [ ] Verify RED.
- [ ] Implement custom `torch.autograd.Function` wrappers saving solver state and invoking exact VJPs.
- [ ] Verify gradcheck and a multi-step optimization smoke test.
- [ ] Commit `feat: expose differentiable QC layers`.

### Task 8: Core validation

**Files:** `src/qcopt/experiments/validate_solver.py`, `tests/test_validate_solver.py`

- [ ] Write a failing CLI smoke test requiring reconstruction, forward/adjoint residual, gradient sweep, time, memory, and acceptance fields.
- [ ] Verify RED.
- [ ] Implement fixed-seed affine/smooth cases, refinement sweep, JSON/CSV/PNG output, and explicit acceptance flags.
- [ ] Run full validation and inspect each acceptance field.
- [ ] Commit `experiment: validate variable-mu forward and adjoint`.
