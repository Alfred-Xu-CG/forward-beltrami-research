# Route A — realizable facewise Beltrami data and compatibility projection

## 1. Problem statement

The input is one coefficient `mu_T` per source triangle. The desired output is
a continuous P1 map `F_U` whose exact face coefficient equals the input:

`mu_T(U)=mu_T^target` for every face `T`.

This is not merely a local algebraic problem. Neighbouring triangles share
vertices and edges, so their constant Jacobians must be compatible with one
another. Route A asks first whether the data are realizable, then computes a
nearest realizable map when they are not.

## 2. Shared-edge derivation

On face `T`, write `a_T=F_z|_T` and `b_T=F_barz|_T`. Along a complex source edge
with tangent `dz`, the directional derivative is

`dF_T=a_T dz+b_T conjugate(dz)`.

If `T` and `S` share the edge, continuity gives

`a_T dz+b_T conjugate(dz)=a_S dz+b_S conjugate(dz)`.

Under prescribed `b_T=mu_T a_T`, this becomes

`(dz+mu_T conjugate(dz))a_T
 -(dz+mu_S conjugate(dz))a_S=0`.                     (A.1)

Putting one row of (A.1) per interior edge produces a sparse complex matrix
`C(mu)`. The face derivative vector must satisfy

`C(mu)a=0`.                                             (A.2)

The matrix depends on the source edge geometry and on `mu`, but not on the
unknown target vertex coordinates. A nonzero null vector is necessary for a
continuous facewise derivative field. It is not by itself sufficient: one must
also integrate the derivatives around cycles and satisfy boundary data.

## 3. Why arbitrary facewise data fail

Each face coefficient has two real degrees of freedom, whereas continuity
introduces one complex constraint per interior edge. Consequently the set of
realizable face fields is a lower-dimensional nonlinear subset of all
facewise assignments. A random field inside the unit disk may satisfy
`|mu_T|<1` on every face and still have no nonzero solution of (A.2).

This explains the observed random-field residual: the obstruction is
integrability, not an implementation bug or a failure of a sparse linear
solver.

## 4. Nonlinear projection used in the experiments

Because exact realization may be impossible, fix boundary coordinates and let
`q` contain all interior image coordinates. Define

`r(q)= [Re(mu_T(F_q)-mu_T^*) ; Im(mu_T(F_q)-mu_T^*)]_{T in F}`.

The implemented projection solves

`min_q 0.5*||r(q)||_2^2 + 0.5*lambda*||q-q0||_2^2`.       (A.3)

`q0` is normally the identity or a supplied initial map. The regularizer makes
the problem locally better conditioned and selects a nearby map; it does not
make an incompatible target exactly realizable.

## 5. Analytic sparse Jacobian

For a perturbation `delta q`,

`D mu_T[delta q]
 = (F_z D F_barz[delta q]-F_barz D F_z[delta q])/F_z^2`. (A.4)

`D F_z` and `D F_barz` are linear combinations of the perturbations of the
three vertices of `T`. Thus each face contributes at most six real columns to
the real Jacobian. The code assembles this Jacobian analytically and passes it
to a sparse trust-region least-squares method.

## 6. Forward algorithm

1. Validate `|mu_T|<1` and a single boundary loop.
2. Fix boundary vertices and initialize interior coordinates.
3. Evaluate all face Jacobians and face coefficients.
4. Assemble residual (A.3) and its sparse analytic Jacobian.
5. Run bounded-budget nonlinear least squares.
6. Recompute the final face coefficients and independently audit orientation.

## 7. Reverse derivative

If `q*` is a regular stationary point, differentiating the first-order
condition gives

`(r_q^T r_q+lambda I)dq*/dtheta
 =-r_q^T r_theta`.                                      (A.5)

An adjoint solve with the transpose matrix yields the VJP without differentiating
through every trust-region iteration. The current route has analytic residual
Jacobians and local VJP controls, but does not claim a globally smooth implicit
projection map: rank loss, branch changes, and nonunique minimizers remain.

## 8. Experimental evidence

Manufactured compatible maps reconstruct at approximately machine precision.
The fixed-R2 256² projection reached relative coefficient error about
`3.01e-9`, zero flips, and minimum area ratio `0.65460`. In contrast, the
random incompatible 256² field remained at relative residual `0.7081017153`
after 11 evaluations under a 20-evaluation budget, although it had zero flips
and minimum area ratio `0.63326`.

## 9. Complexity and decision

The compatibility matrix has `O(|E|)` nonzeros. Sparse assembly is linear in
mesh size, but nonlinear projection repeatedly evaluates all faces and solves a
large least-squares problem. Route A is therefore a principled realizability
projection/preprocessor. It is not a universal exact forward solver for
arbitrary facewise coefficients.
