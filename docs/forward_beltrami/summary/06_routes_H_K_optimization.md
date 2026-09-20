# Routes H and K: barriers, KKT systems, and implicit differentiation

## Method

Barrier methods add a penalty that diverges as a triangle determinant approaches
zero. KKT means Karush-Kuhn-Tucker conditions: equations describing a stationary
point of a constrained optimization problem. An implicit adjoint differentiates
these equations without unrolling every optimization iteration, usually by
solving a linearized transpose system.

## Common-target evidence

On a 256² common target and four Adam steps, the unconstrained direct map had
loss `3.7963e-4`, minimum determinant `-1.5745e-4`, and 2,754 flips. LIM,
SLIM, and AMIPS-style barriers had losses `8.1137e-4`, `8.6251e-4`, and
`8.4326e-4`, zero flips, minimum determinants approximately
`4.89e-6`, `5.53e-6`, and `4.23e-6`, and wall times around 159--167 seconds.
This is a finite-budget topology/speed comparison, not a claim that the
barriers reached equal optimization convergence.

The matrix-free KKT layer at 256² reached stationarity `2.65e-12`, minimum
determinant `1.0965e-5`, zero flips, and an implicit directional error
`4.55e-6` after tightening the CG tolerance; it used 637 adjoint HVPs.

## Interpretation

These routes provide a principled safety layer and a useful differentiable
implicit-solve template. They do not produce a purely forward map for an
arbitrary Beltrami coefficient: they solve a nonlinear optimization problem,
can be ill-conditioned near active barriers, and need a policy for nonsmooth
active constraints. They also do not automatically prove global injectivity
beyond the audited mesh/domain assumptions.

## Decision

Use barrier/KKT machinery as a certification or correction layer, and as a
baseline for gradient quality. It should not be counted as the completed fast
forward layer.
