# Route K — equilibrium, DEQ, and implicit KKT differentiation

## 1. General equilibrium form

Many routes can be written as an equilibrium

`G(x,theta)=0`,

where `x` may be the fixed-point field `h`, map vertices `U`, multipliers, or
all of them. If `G_x` is nonsingular, the implicit-function theorem gives

`dx/dtheta=-G_x^{-1}G_theta`.

For reverse mode, solve

`G_x^T p=dL/dx`,

then return

`dL/dtheta=-p^T G_theta`.

## 2. Beurling equilibrium

For Route B,

`G(h,mu)=(I-M_muB)h-mu`.

The adjoint is the periodic/padded system described in Route B.

## 3. Barrier/KKT equilibrium

For Route H,

`G(U,lambda,theta)= [grad_U L(U,lambda,theta); c(U,theta)]`.

The implementation applies matrix-free Hessian-vector products and Newton-CG or
CG solves. This avoids storing dense Jacobians and avoids unrolling all solver
iterations.

## 4. Numerical conditions

The implicit derivative is trustworthy only when:

1. the forward residual `||G||` is small;
2. `G_x` is sufficiently nonsingular;
3. the transpose solve converges to a controlled residual;
4. the active set or branch is fixed or smoothly regularized.

The 256² KKT control reached stationarity about `2.65e-12`; tightening CG gave
implicit directional error about `4.55e-6` with 637 adjoint HVPs. This is a
strong numerical result for the tested system, not a theorem for every
nonlinear map or active-set transition.

## 5. DEQ interpretation

A deep equilibrium layer would iterate a map `x_{r+1}=Phi_theta(x_r)` until
convergence and differentiate only the fixed point through the adjoint above.
The attraction is memory independent of iteration count. The danger is that
convergence of the iteration, invertibility of `I-D_xPhi`, and topology of the
decoded map are separate requirements. Route K supplies the derivative
machinery but not the missing global Beltrami/homeomorphism theorem.
