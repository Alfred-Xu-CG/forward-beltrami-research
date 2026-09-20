# Route B3: nonperiodic rectangle finite-difference BVP

`rectangle_beltrami_fd` assembles the central-difference equation

\[
 \bar\partial f-\mu\,\partial f=0
\]

on interior grid nodes and takes full-grid Dirichlet values as the
normalization/boundary condition. The sparse system is square and can be solved
directly or by LSQR. Unlike the FFT route, it has no periodic extension or
zero-mode deletion; unlike a boundary-free forward decoder, it makes the
boundary data explicit.

## Affine resolution audit

For the constant coefficient `mu=0.18+0.11i`, the exact affine map was supplied
on the boundary. Sparse direct solves produced:

| grid | unknowns | time | equation residual | max map error |
|---:|---:|---:|---:|---:|
| 128x128 | 15,876 | 0.419 s | `2.91e-12` | `3.26e-14` |
| 256x256 | 64,516 | 2.068 s | `1.09e-10` | `7.14e-13` |
| 512x512 | 260,100 | 9.863 s | `1.04e-9` | `3.03e-12` |

The 512 grid is a realistic sparse-BVP check, not a coarse demonstration. An
LSQR-only pilot at 512 with a 1000-iteration cap stalled with residual `8.44e-2`,
whereas the direct sparse solve remained accurate. This cleanly separates
“sparse solve exists” from “iterative solver/preconditioner is adequate.”

## Scope and limitation

This is a rectangle boundary-value reference, not yet a complete arbitrary-μ
forward layer: boundary values are an input, central differences are only
second-order, and direct factorization will eventually need a reusable sparse
factor/preconditioner plus an implicit adjoint. It nevertheless provides the
nonperiodic control needed to interpret the FFT boundary audit. Receipt:
`artifacts/rectangle_fd_resolution_audit/rectangle_fd_resolution_audit.json`.

The same solver also reproduced a non-affine manufactured QC map whose
discrete coefficient was computed from the same central-difference stencil.
For `max|mu|=0.2694` / `0.2695`, the `128²` and `256²` RMS map errors were
`2.46e-15` and `4.84e-15` (maximum errors `5.75e-14` and `1.28e-13`). This is
a discrete consistency check rather than an independent continuum convergence
claim, but it verifies that the rectangle system can carry spatially varying
coefficients when the boundary and operator discretization are matched.

The sparse rectangle system also has a transpose-solve boundary VJP
(`rectangle_fd_boundary_vjp`). A complex finite-difference check on a `12x12`
grid agreed within `2e-6`, with adjoint residual below `1e-9`. This removes the
need to unroll the direct solve for boundary gradients. A coefficient VJP is
now also implemented using the same transpose solve and the discrete `dz f`
factor; a second finite-difference check passed below `2e-6`. A reusable
high-resolution factorization is still needed before this becomes a full neural
rectangle layer.

That reuse path is now present as `RectangleFDFactorization`: at `256²`, sparse
factorization took `2.12 s`, and two subsequent boundary solves completed in
about `0.09 s` total, with residuals `2.72e-11` and `4.87e-11`. This directly
addresses the practical observation that repeated linear solves can be cheap
after matrix assembly/factorization; the remaining memory issue is the sparse
factor itself, not unrolled iteration state.

## ILU-preconditioned GMRES and transpose control

`solve_rectangle_fd_gmres` adds a reusable ILU-preconditioned GMRES path and a
conjugate-transpose solve for implicit differentiation. On the realistic
constant- and variable-coefficient cases, the forward solves converged with
`info=0` and agreed with the sparse-direct maps to relative errors below
`1e-11`:

| case | grid | GMRES iterations | forward equation residual | direct time | iterative total* |
|---|---:|---:|---:|---:|---:|
| constant | 256² | 10 | `7.68e-9` | `2.09 s` | `3.69 s` |
| constant | 512² | 116 | `2.30e-7` | `9.81 s` | `59.59 s` |
| variable | 256² | 10 | `1.50e-8` | `2.01 s` | `3.67 s` |
| variable | 512² | 32 | `1.31e-7` | `9.68 s` | `35.85 s` |

`*` includes assembly, ILU setup, forward GMRES, and the reported transpose
solve. Transpose systems also converged (`9/141` iterations for the 256²/512²
constant cases and `9/96` for the variable cases), with relative residuals
below `7e-11`. The centered complex stencil can produce an exact ILU pivot
breakdown at tight drop tolerances; the implementation records a looser,
higher-fill fallback without changing the operator.

This is positive evidence for an implicit, non-unrolled rectangle layer, but
not a blanket speed claim: at 512² the direct sparse solve remains faster than
this CPU ILU-GMRES configuration, especially for the constant field. The
remaining engineering target is a stronger multilevel/preconditioned operator,
while the boundary-free arbitrary-μ and higher-order discretization gates are
still open. Receipt:
`artifacts/rectangle_fd_iterative_audit_512/rectangle_fd_iterative_audit.json`.
