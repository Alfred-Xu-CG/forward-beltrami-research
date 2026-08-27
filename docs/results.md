# Verified results

## Evidence status

The full runner completed with seed `20260828`. `artifacts/manifest.json` hashes 18 evidence files and records the exact environment. `artifacts/audit.json` reports `VERIFIED` with no mismatches after independently recomputing:

- a fresh manufactured LBS reconstruction and adjoint directional derivative;
- all six I-to-S maps and their losses, Dice, folds, minimum Jacobian, and certificates;
- all 20 density maps and their area errors, folds, minimum Jacobian, and certificates;
- correct and deliberately wrong multi-chart physical seam residuals.

The audit corruption test changes a saved fold count and is required to produce `FAILED`.

## Forward and adjoint solver

| Quantity | Maximum observed |
|---|---:|
| Manufactured reconstruction error | `2.40e-14` |
| Forward algebraic residual | `2.75e-14` |
| Adjoint algebraic residual | `2.39e-14` |
| Stable-window gradient relative error | `5.45e-9` |

The independent auditor used different affine data and obtained reconstruction error `3.33e-16`, forward residual `2.44e-16`, adjoint residual `1.51e-16`, and directional-gradient relative error `7.56e-10`.

Sparse CPU scaling at the largest tested structured mesh:

| Solver | Vertices | Faces | System dimension | Nonzeros | Forward time |
|---|---:|---:|---:|---:|---:|
| LBS | 4,225 | 8,192 | 8,962 | 59,138 | 0.054 s |
| Augmented LSQC | 4,225 | 8,192 | 24,838 | 213,000 | 0.509 s |

These timings are single-run reference measurements, not statistically stable performance claims. A changed μ requires a new numeric factorization; geometry and sparsity can be cached, but the factor itself cannot generally be reused unchanged.

## I-to-S large-distortion registration

Common setting: `12 x 12` rectangle grid, 40 iterations, identical analytic source/target fields and seed.

| Method | Data loss | Soft Dice | Flips | min det | max K | Numerical certificate |
|---|---:|---:|---:|---:|---:|---:|
| μ-LBS, fixed sliding boundary | 0.0623 | 0.9325 | 0 | 0.1928 | 8.34 | yes |
| μ-LSQC, free boundary | 0.2283 | 0.6354 | 0 | 0.0461 | 14.41 | yes |
| direct-f, no safeguard | **0.00358** | **0.9910** | 116 | -15.06 | infinity | no |
| direct-f + LIM-style | 0.0299 | 0.9617 | 0 | 4.78e-5 | 18,282.5 | yes |
| direct-f + SLIM-style | 0.1320 | 0.7985 | 0 | 0.0178 | 110.63 | yes |
| direct-f + AMIPS-style | 0.1580 | 0.7915 | 0 | 0.00365 | 322.30 | yes |

Interpretation:

- The unconstrained direct map achieves the best image metric by folding heavily, so the image loss alone is invalid evidence of registration quality.
- Adding safeguards to direct-f does avoid folding in this experiment. Therefore μ optimization is **not** uniquely capable of producing a bijection.
- The LIM-style run has better Dice than μ-LBS but approaches degeneracy. A positive Jacobian certificate alone does not imply usable element quality.
- μ-LBS provides the best observed compromise among Dice, minimum area margin, and maximum distortion in this configuration.
- Free-boundary LSQC is slower and less accurate here; free boundary is a capability, not automatically an advantage.

## Prescribed face-area deformation

Common setting: `10 x 10` grid and 40 iterations. Selected log-area RMSE values:

| Target | Feasibility status | μ-LBS | direct none | LIM-style | SLIM-style | AMIPS-style |
|---|---|---:|---:|---:|---:|---:|
| Manufactured | known feasible | **0.0282** | 0.1357 | 0.1320 | 0.6028 | 0.1383 |
| Checkerboard | unproven | **0.4118** | infinity / 50 flips | 1.1815 | 1.2379 | 1.4534 |
| Spike | unproven | **0.0582** | 0.4508 | 0.2624 | 0.2536 | 0.2719 |
| Random | unproven | **1.0473** | infinity / 57 flips | 1.1438 | 1.3812 | 1.1586 |

All safeguarded runs passed the numerical certificate. μ-LBS was best on all four target distributions under this fixed budget, but the random target still had large residual (`relative RMSE 4.66`) and minimum determinant `0.00589`. Since the three stress targets are only sum-normalized, their exact PL feasibility is not asserted.

## Multi-chart registration

The full two-chart cylinder atlas used 91 matched overlap points. Results:

- transition-aware seam residual: `2.66e-15`;
- deliberately wrong raw-coordinate-equality physical seam residual: `1.0`;
- algebraic residual: `1.37e-15`;
- landmark RMSE: `9.08e-16`;
- maximum measured angular dilation: `2.187`;
- both chart maps passed the numerical disk-map audit.

This demonstrates the central compatibility equation

\[
g_d\circ\tau^S_{dc}=\tau^T_{dc}\circ g_c
\]

for matched affine transitions. Unit tests also cover nonlinear transition Jacobians/Gauss-Newton residual reduction and conformal Beltrami covariance.

## Research conclusion from this prototype

The defensible next paper direction is not “optimize μ because direct-f folds.” A stronger claim is:

> Exact implicit QC reconstruction provides a residual-controlled geometry layer and a useful distortion parameterization; its value appears when the task needs explicit control or regularization in Beltrami space, repeated map reconstruction under structured boundary/atlas constraints, and an auditable relationship between local distortion and global topology. Direct-map barrier methods remain essential baselines and can be superior for some objectives.

The most promising follow-up is a three-way study of exact adjoints, learned SBN surrogates, and safeguarded direct-f solvers at much larger meshes, with identical stopping criteria and downstream losses.
