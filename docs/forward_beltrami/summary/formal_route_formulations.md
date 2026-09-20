# Formal mathematical formulation of the forward-Beltrami routes

This document expands the short route summaries. It distinguishes the
continuous equation, the finite-element map actually represented in code, the
optimization or iteration performed by each route, and the exact point at
which a topology or gradient claim stops being a theorem.

## 1. Common notation and the target problem

Let `Omega` be a planar domain with a positively oriented triangular mesh
`M=(V,F)`. A vertex map is

`U = (U_i)_{i in V}`, `U_i=(u_i,v_i) in R^2`.

Let `phi_i` be the P1 hat basis function. The finite-element map is

`F_U(x) = sum_{i in V} U_i phi_i(x)`.

On a triangle `T=(i,j,k)`, the gradient is constant:

`J_T(U) = DF_U|_T = sum_{a in {i,j,k}} U_a (grad phi_a)^T`.

Writing `J_T=[[u_x,u_y],[v_x,v_y]]`, define

`F_z = 1/2 (u_x+v_y + i(v_x-u_y))`,

`F_barz = 1/2 (u_x-v_y + i(v_x+u_y))`.

The discrete Beltrami coefficient is

`mu_T(U) = F_barz / F_z`, provided `F_z != 0`.

The continuous target equation is

`F_barz = mu F_z`, with `|mu(x)| < 1`.

The local Jacobian determinant satisfies

`det J_T = |F_z|^2 - |F_barz|^2 = |F_z|^2(1-|mu_T|^2)`.

Thus `|mu_T|<1` implies positive local determinant if `F_z` is nonzero.
However, a numerical algorithm may not enforce the exact equation on every
triangle, and positive determinants alone do not prove global injectivity.

For a real-valued objective `L(U,theta)`, a neural-network layer requires a
well-defined vector-Jacobian product (VJP) with respect to its input `theta`.
The phrase “finite gradient” below means that the implemented VJP was finite on
the tested input; it does not mean that the map is differentiable across every
active-set switch or failed solve.

## 2. Route A: shared-edge compatibility and nonlinear projection

### 2.1 Linear compatibility in face variables

On each face introduce the complex scalar `a_T=F_z|_T`. If the requested face
coefficient is `mu_T`, then `b_T=F_barz|_T=mu_T a_T`. For a shared source edge
with complex tangent `dz`, the directional derivative from the two adjacent
faces must agree:

`a_T dz + b_T conjugate(dz) = a_S dz + b_S conjugate(dz)`.

Substituting `b=mu a` gives the complex linear equation

`[(dz + mu_T conjugate(dz))] a_T`
`-[(dz + mu_S conjugate(dz))] a_S = 0`.      (A1)

Assemble one row of a sparse complex matrix `C(mu)` for every interior edge.
The exact facewise coefficient is realizable by a continuous P1 map only if

`C(mu) a = 0`                                                       (A2)

has a nonzero solution compatible with the boundary and the geometric
integrability conditions. On a generic simply connected disk, a realizable
field has a small nullspace associated with global complex scale/normalization;
an arbitrary facewise field usually has only the zero solution.

This is the first obstruction: assigning `mu_T` independently does not assign
a continuous map independently.

### 2.2 Nonlinear map projection

The implemented projection instead fixes boundary vertices and optimizes the
interior vector `q`:

`min_q 1/2 || R(q) ||_2^2 + (lambda/2)||q-q_0||_2^2`,                (A3)

where

`R(q) = [ Re(mu_T(F_q)-mu_T^*) ; Im(mu_T(F_q)-mu_T^*) ]_{T in F}`.

The Jacobian uses the quotient rule. For a vertex perturbation `delta U`,

`D mu_T[delta U] = (F_z D F_barz[delta U] - F_barz D F_z[delta U]) / F_z^2`. (A4)

Because P1 gradients are local, one face touches at most three vertices and
the real Jacobian is sparse. The code supplies this analytic Jacobian to a
trust-region least-squares solver.

### 2.3 Differentiation and limitation

If the optimizer converges to a regular solution, an implicit derivative would
solve

`(R_q^T R_q + lambda I) dq/dtheta = -R_q^T R_theta`.             (A5)

The current experiments mainly verify the forward projection and local sparse
derivative assembly; they do not establish uniqueness of the projection, a
global smooth chart for all target fields, or an exact second-order derivative
through trust-region iterations. For incompatible data, the nonzero residual in
(A3) is mathematically meaningful, not merely a failure of the optimizer.

## 3. Route B: Beurling transform and Fourier discretization

### 3.1 Whole-plane fixed point

Let `h=F_barz`. Since `F_z=1+B h` after applying the Cauchy inverse, the
Beltrami equation becomes

`h = mu (1 + B h)`,

or

`(I - M_mu B)h = mu`,                                            (B1)

where `M_mu` is pointwise multiplication. The implemented Neumann iteration is

`h^{n+1} = mu (1+B h^n)`.                                       (B2)

The map is recovered from a Cauchy inverse `C` by `F=z+C h` up to an affine
normalization.

### 3.2 Periodic Fourier symbol

On a rectangular torus, for Fourier wave number `k=(k_x,k_y) != 0`, the code
uses

`B_hat(k) = (k_x - i k_y)/(k_x + i k_y)`,

and sets the zero mode to zero. Therefore

`B h = FFT^{-1}( B_hat * FFT(h) )`.                            (B3)

This is an exact discretization of the periodic model, not of the free-space
bounded-domain problem.

### 3.3 Zero padding and the periodicity question

If a bounded field is embedded in a larger box of size `L` and padded by zeros,
the FFT computes a periodic repetition of that larger box. Increasing `L`
reduces the interaction with periodic copies but does not mathematically remove
it. It also leaves quadrature error, target interpolation error, finite
frequency truncation, and the treatment of the singular principal value.

### 3.4 Implicit VJP

Let `A=I-M_mu B`, so `Ah=mu`. For a real loss with cotangent `g_h`, solve

`A^* lambda = g_h`.                                               (B4)

Since `d h = A^{-1}(d mu + dM_mu B h)`, the coefficient VJP is

`dL/dmu = lambda * conjugate(1+B h)`                         (B5)

under the real Euclidean inner product convention used in the code. This avoids
unrolling the fixed-point iteration, but it remains the adjoint of the chosen
periodic/zero-padded discretization.

### 3.5 General meshes

For nonuniform points, FFT diagonalization is lost. Direct evaluation costs
`O(N^2)` interactions; treecode/FMM or particle-mesh methods can reduce this,
but a singular near-field quadrature and bounded-domain boundary treatment are
still required. Hence “FFT is fast” does not imply “general mesh is fast.”

## 4. Route C: finite-element Beltrami holomorphic flow

### 4.1 Discrete state and velocity

The state is the vertex map `U^n`. Compute facewise `F_z^n` from the P1 map and
the prescribed face variation `nu_T`. The BHF velocity is a singular integral
of the form

`V_T(w) = -(nu_T (F_z^n(T))^2/pi) * integral_T K(F^n(zeta),F^n(w)) dA_zeta`, (C1)

with the regularized kernel written in the implementation as a principal
`1/(zeta-w)` term plus image/boundary correction terms. The vertex velocity is
assembled from near incident-face Duffy quadrature and blocked far-field
quadrature.

The practical Euler layer is

`U^{n+1} = U^n + tau_n V(U^n,nu)`.                             (C2)

### 4.2 Determinant-safe step

For each face write

`d_T(t)=det(e_1+t delta e_1, e_2+t delta e_2)`
`= c_T+b_T t+a_T t^2`.

The exact controller computes the first positive root at which `d_T(t)` reaches
a margin `m>0`, then chooses

`tau_n = min(tau_requested, safety * min_T root_T)`.              (C3)

Thus the forward step retains `det J_T >= m` in exact arithmetic when the
current state already satisfies the margin. The smooth controller replaces the
hard minimum and absolute values by differentiable upper/lower surrogates; it
is conservative but not identical to the exact active-set rule.

### 4.3 Backward and multilevel variant

The implemented backward replays (C2) instead of storing every dense pairwise
interaction graph. A coarse-to-fine variant computes `U_c`, applies a
differentiable bilinear prolongation `P U_c`, and continues on the fine grid:

`U_f = Phi_f^{(s_f)}( P Phi_c^{(s_c)}(U_0) )`.                     (C4)

The chain rule is

`dL/dU_0 = (D Phi_c)^T P^T (D Phi_f)^T dL/dU_f`.                 (C5)

This is the mathematical reason the PyTorch interpolation experiment retained
backpropagation. It does not prove that bilinear prolongation preserves a
homeomorphism for arbitrary maps; that property must be separately certified
or enforced.

## 5. Route D: Tutte/Floater positive barycentric decoder

Let `B` be boundary vertices with prescribed convex counter-clockwise positions
`Y_b`, and `I` be interior vertices. For every interior vertex `i`, choose
strictly positive weights `w_ij>0` over its graph neighbours and impose

`Y_i = sum_{j in N(i)} p_ij Y_j`,
`p_ij = w_ij / sum_{k in N(i)} w_ik`.                         (D1)

Rearranging gives a sparse linear system

`A(w) Y_I = C(w)Y_B`.                                           (D2)

Under the usual 3-connected planar graph and convex-boundary hypotheses, the
positive barycentric embedding is an injective planar embedding. The code uses
uniform weights, learned positive edge weights, or directed positive rows.

For a scalar loss, the adjoint solves

`A(w)^T lambda = dL/dY_I`.                                      (D3)

Differentiating each row of (D1) gives local weight gradients involving
`(Y_j-Y_i)/sum_k w_ik`, and the code combines endpoint contributions for an
undirected edge.

The guarantee is topological but conditional: it requires positive weights,
valid planar connectivity, and a convex boundary. It does not imply arbitrary
Beltrami expressivity.

## 6. Route E: M-matrix and positive directional conductances

The Beltrami coefficient corresponds to the symmetric tensor

`A(mu)=1/(1-|mu|^2) [[1-2 Re(mu)+|mu|^2, -2 Im(mu)],`
`                         [-2 Im(mu), 1+2 Re(mu)+|mu|^2]]`.      (E1)

For a unit direction `d_k`, a monotone stencil represents

`A approx sum_k c_k d_k d_k^T`, `c_k >= 0`.                  (E2)

If `D` is the 3-by-m dictionary with columns
`(d_{kx}^2,d_{kx}d_{ky},d_{ky}^2)^T`, the local fit is

`min_{c>=0} ||D c - (A_11,A_12,A_22)^T||_2`.                    (E3)

For an exact fit, a scalar discrete operator has the positive-stencil form

`(L_h u)_i = sum_k c_{i,k}[u_{i+r_k}-u_i]`

with the corresponding symmetric/flux-consistent neighbour terms. Positive
off-diagonal structure gives an M-matrix and a discrete maximum principle.

The key obstruction is geometric: an SPD tensor always has positive eigenvector
decomposition in arbitrary directions, but those directions may not be edges of
the mesh. Restricting to mesh-realizable directions turns (E3) into a finite
positive cone, so the residual can be strictly positive even when `|mu|<1`.

## 7. Routes H/K: barrier optimization and implicit KKT adjoints

Let `E(U)` be a map-fitting energy. A determinant barrier uses

`E_beta(U) = E(U) - beta sum_T log(det J_T(U)-m)`,                  (H1)

with `det J_T(U)>m`. A stationary point satisfies `grad E_beta(U*)=0` or,
with other constraints `c(U)=0`, the KKT equations

`G(U,lambda,theta) = [grad_U L(U,lambda,theta); c(U,theta)] = 0`. (H2)

The implicit derivative is obtained from

`G_x dx/dtheta = -G_theta`,                                      (H3)

and the reverse adjoint solves

`G_x^T p = dL/dx`, then `dL/dtheta = -p^T G_theta`.              (H4)

The matrix-free implementation evaluates Hessian-vector products and uses CG.
The barrier strongly discourages inversion and gave zero-flip common-target
controls. It remains a nonlinear solve, and the quality of (H4) depends on
stationarity, conditioning, CG tolerance, and how active topology constraints
are treated.

## 8. Routes F/G/I/J/N: structured forward parameterizations

### Positive triangular flows

A triangular map has the form

`T(x,y)=(T_1(x), T_2(x,y))`,

with positive diagonal derivatives. Compositions preserve local orientation if
each layer has positive diagonal determinant. Positive-increment parameters use
`delta_i=exp(s_i)` (or an equivalent positive transform), ensuring monotone 1D
coordinates. This gives fast differentiability and topology control for the
parameterized family, not arbitrary-QC expressivity.

### Certified continuation

For a direction `Delta U`, the determinant on each face is the quadratic
`c_T+b_T t+a_T t^2`. The safe continuation layer chooses a step before the
smallest positive root. This is a local certificate and is nonsmooth when the
active face changes.

### Prolongation and learned decoders

Given coarse vertex values `U_c`, bilinear prolongation computes
`U_f=P U_c`; its derivative is the sparse interpolation matrix `P`. Positive
increment or triangular parameterizations constrain `P U_c` so selected
derivatives stay positive. Flow matching and latent decoders learn a restricted
distribution of such parameters; they are not direct inversion of an arbitrary
input `mu`.

### Neumann preconditioners

For a linear system `(I-K)x=b`, a truncated Neumann initializer is

`x_0 = sum_{j=0}^p K^j b`,

used as a warm start for GMRES/CG. It can lower iteration counts when the
operator spectrum is favourable, but it does not change the solution set or
provide topology/expression guarantees.

## 9. Side routes L/M

On a flat torus, the coefficient's mean/zero mode changes the affine periods.
For `h=F_barz`, the implemented lift is

`F(z)=z + mean(h) conjugate(z) + C(h-mean(h))`,

with periods `1+mean(h)` and `i(1-mean(h))`. This is a useful periodic control,
not a bounded-domain solver.

Sphere/orbifold experiments use multiple coordinate charts, explicit seam maps,
and orientation audits. They test atlas bookkeeping and cone coordinates, not
the planar general-mesh neural layer.

## 10. Formal completion conditions

To claim the final solver, one must prove or explicitly restrict a theorem
covering: (i) existence and uniqueness of the discrete map for the stated
input class; (ii) global injectivity under the actual mesh/boundary hypotheses;
(iii) differentiability of every accepted forward branch; (iv) a cost/memory
bound for the chosen quadrature or sparse solve; and (v) convergence/error
control as the mesh is refined. The current routes satisfy different subsets
of these conditions, which is why the project conclusion remains “partial,”
not “completed.”
