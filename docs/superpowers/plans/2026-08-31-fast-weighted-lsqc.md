# Fast Weighted LSQC Implementation Plan

**Goal:** Add an exact, hard-pin-eliminated weighted LSQC forward/adjoint path
whose backward is fully vectorized and reuses the forward factorization.

**Architecture:** Assemble the `2|V|` paper Hessian directly from the LBS tensor
block and the mu-independent area coupling, eliminate selector constraints,
factor only the free block, and use the same factor for the transpose adjoint.
Keep augmented LSQC unchanged as the reference backend.

**Tech stack:** NumPy, SciPy sparse/SuperLU, PyTorch custom autograd, pytest.

## Task 1: RED equivalence and state tests

- Add tests for direct-Hessian versus weighted `B.T @ B`.
- Add two-pin, fixed-boundary, and rectangle-sliding map equivalence tests.
- Add reduced-dimension and unsupported mixed-constraint tests.
- Add a factorization-reuse test.

## Task 2: Fast forward

- Add a reduced hard-pin solve state.
- Add selector extraction and direct weighted-Hessian assembly.
- Add `solve_lsqc_fast` with residual diagnostics.
- Run focused tests.

## Task 3: Fast vectorized adjoint

- Add all-face tensor contractions using gathered face gradients and `einsum`.
- Add `lsqc_fast_mu_vjp` and vectorize the existing LBS VJP using the same core.
- Validate against augmented gradients and finite differences.

## Task 4: Differentiable layer

- Add `lsqc_fast_layer` and `lsqc_fast_from_raw`.
- Pass double-precision gradcheck and all-face gradient tests.

## Task 5: Benchmark and regression

- Add a reproducible backend benchmark CLI.
- Run focused, full-suite, and benchmark validation.
- Record exact matrix sizes, discrepancies, and measured speedups without
  turning machine-specific timings into unit-test requirements.

