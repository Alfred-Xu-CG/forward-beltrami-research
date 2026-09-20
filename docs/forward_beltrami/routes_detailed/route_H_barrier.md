# Route H — determinant barriers and direct map optimization

## 1. Objective

Let `U` be all image vertex coordinates and let `E(U;theta)` be a fitting or
registration energy. A hard no-fold constraint is

`det J_T(U)>m` for every triangle `T`.

The smooth logarithmic barrier replaces it by

`E_beta(U)=E(U)-beta sum_T log(det J_T(U)-m)`,

defined only inside the feasible set.

## 2. Derivatives

Since `J_T(U)` is linear in `U`, `det J_T(U)` is quadratic in the three face
vertices. For a perturbation `delta U`,

`D det J_T[delta U]=cof(J_T):D(delta U)`.

Thus

`grad_U[-beta log(det J_T-m)]
 =-beta/(det J_T-m) * grad_U det J_T`.

The gradient becomes large near the barrier, which prevents crossing but also
causes conditioning problems.

## 3. Direct optimization algorithm

The experiments use finite-budget Adam-style updates for the map variables,
with each candidate map independently audited. LIM, SLIM, and AMIPS-style
variants differ in distortion/barrier energy, but all are nonlinear direct-map
optimizers rather than forward Beltrami integrators.

On the same 256² target, the unconstrained four-step run had 2,754 flipped
faces. The three barrier variants had zero flips and minimum determinants near
`4e-6` to `5.5e-6`, but required about 159--167 seconds.

## 4. What the barrier does and does not prove

The barrier is infinite at the chosen margin, so a mathematically exact
feasible iterate cannot cross it through a finite objective step. In floating
point, line search, optimizer overshoot, incomplete convergence, and an
ill-conditioned gradient can still violate the margin. A zero-flip audit is
therefore retained.

The barrier guarantees only the enforced discrete determinant constraint. It
does not imply arbitrary `mu` expressivity or a global theorem outside the
mesh/boundary assumptions.
