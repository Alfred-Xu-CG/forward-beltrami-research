# Fast Weighted LSQC Forward-Adjoint Design

## Status and scope

This is the approved engineering revision to the variable-mu prototype.  It
implements only the current fixed linear-constraint case.  Moving boundaries,
learned transitions, and joint correspondence are explicitly out of scope.

The existing augmented weighted/unweighted LSQC solver remains the correctness
reference.  The new production fast path is restricted to weighted LSQC with
hard coordinate pins, including two complex pins, fully fixed vertices, and
axis-aligned rectangle sliding constraints.

## Alternatives considered

1. Keep only the augmented residual-coordinate KKT system.  This avoids normal
   equations, but its dimension is `2|F| + 2|V| + |C|` and its current
   facewise Python-loop VJP dominates runtime.
2. Form `B.T @ B` explicitly and eliminate pinned coordinates.  This is simple
   but unnecessarily assembles `B` and performs a sparse matrix product on
   every changed mu.
3. Assemble the paper Hessian directly and eliminate hard pins.  This has the
   smallest system, avoids the residual variables and sparse product, and is
   exactly equivalent to weighted `B.T @ B` in exact arithmetic.  This is the
   selected fast path.

## Fast forward system

Let `x = [u; v]`.  For weighted LSQC,

$$
Q(\mu)=B_w(\mu)^T B_w(\mu)
=\begin{bmatrix}K(\mu)&S\\-S&K(\mu)\end{bmatrix},
$$

where

$$
K_T=|T|G_TA(\mu_T)G_T^T,
\qquad
S_T=|T|G_TJG_T^T,
\qquad
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
$$

`S` is independent of mu.  Direct assembly is tested against `B_w.T @ B_w`
on nonzero random face coefficients.

For selector constraints, collect pinned coordinates `p` and free coordinates
`f`.  Hard elimination gives

$$
Q_{ff}x_f=-Q_{fp}x_p.
$$

The sparse LU factorization of `Q_ff` is stored in the solve state.  No KKT
multiplier or residual variable is introduced.  General mixed linear rows are
rejected rather than silently interpreted as hard pins.

## Fast adjoint and facewise VJP

For an output cotangent `bar(x)`, solve with the same numeric factorization:

$$
Q_{ff}^T p_f=\bar{x}_f,
\qquad p_p=0.
$$

Since `S` is mu-independent,

$$
\frac{\partial L}{\partial\theta_T}
=-p_{u,T}^TK_{T,\theta_T}u_T
 -p_{v,T}^TK_{T,\theta_T}v_T,
\qquad \theta_T\in\{\rho_T,\tau_T\}.
$$

The weighted-LSQC implementation evaluates residuals and both derivative
actions while each primal or adjoint field's face gradients are already in
cache, and then applies the exact product rule for $d(B_w^T B_w)$. All faces
and both parameters are contracted with vectorized NumPy kernels. It does not
materialize the full facewise tensor derivative, and there is no Python loop
over faces or parameters in the fast VJP.

## Verification and performance contract

- Direct Hessian equals weighted `B.T @ B` to floating-point tolerance.
- Fast maps agree with the augmented reference for two pins, fixed boundary,
  and rectangle sliding constraints.
- Fast analytic gradients agree with augmented gradients and central finite
  differences for every face component.
- PyTorch double-precision gradcheck passes.
- One factorization is performed in forward and the saved factor is reused by
  backward.
- The reduced system dimension is exactly `2|V| - number_of_pinned_coordinates`.
- Benchmarks report assembly, factorization/forward, adjoint solve, contraction,
  total backward, matrix shape, and numerical discrepancies.  Timing assertions
  are not placed in unit tests because wall-clock thresholds are machine
  dependent.
