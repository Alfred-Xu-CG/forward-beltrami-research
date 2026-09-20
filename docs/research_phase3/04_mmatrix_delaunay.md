# Anisotropic \(P_1\) stiffness, Delaunay conditions, and \(M\)-matrices

Status: builder derivation complete; the local/global distinction is
independently checked in `tmp/phase3_correction_checker.md`. A larger mesh
benchmark and an implementation-level checker remain open.

## 1. Discrete diffusion operator

Let \(\mathcal T_h\) be a conforming triangulation of a planar polygonal
domain. On a triangle \(T\), let \(A_T=A_T^T\succ0\) be constant and let
\(\{\phi_i\}\) be the affine nodal basis. The \(P_1\) stiffness matrix is

\[
 K_{ij}=\sum_{T\ni i,j}|T| (\nabla\phi_i|_T)^T A_T(\nabla\phi_j|_T).
\]

For an interior edge \(e=(i,j)\) shared by \(T_1,T_2\), only these two
triangles contribute to \(K_{ij}\). A discrete maximum principle for the
homogeneous operator requires, in addition to positive diagonal and the usual
row/irreducibility conditions, \(K_{ij}\le0\) for every \(i\ne j\).
Calling the whole matrix an \(M\)-matrix is therefore stronger than checking
a single local angle: it includes sign, nonsingularity, and connectivity.

## 2. Local anisotropic nonobtuse condition

On one triangle, apply the affine metric map \(y=A_T^{-1/2}x\). The
transformed triangle has Euclidean area \(|\widetilde T|=|T|/\sqrt{\det A_T}\),
and the local bilinear form is a positive scalar multiple of the Euclidean
form on \(\widetilde T\). Consequently the local off-diagonal entry associated
with the edge opposite transformed angle \(\widetilde\theta_k\) has the
cotangent sign

\[
 K_{ij}^{T}= -\frac{\sqrt{\det A_T}}{2}\cot\widetilde\theta_k.
\]

Thus \(0<\widetilde\theta_k\le\pi/2\) implies \(K_{ij}^{T}\le0\). This
transformed nonobtuse condition is a strong *local sufficient* condition. It
is not necessary for the assembled matrix.

## 3. Edge-wise anisotropic Delaunay condition

For a shared edge, the assembled coefficient is

\[
 K_{ij}= -\frac{\sqrt{\det A_{T_1}}}{2}\cot\widetilde\theta_{1}
          -\frac{\sqrt{\det A_{T_2}}}{2}\cot\widetilde\theta_{2}.
\]

When the same tensor \(A\) is used in both triangles, this is nonpositive
exactly when the two transformed opposite angles satisfy
\(\widetilde\theta_1+\widetilde\theta_2\le\pi\), the anisotropic analogue
of the Delaunay condition. With face-dependent tensors there is generally no
single metric shared across the edge; the weighted cotangent sum itself is
the condition to test.

The local nonobtuse test implies the edge test, but not conversely. An
independent isotropic example uses opposite angles \(120^\circ\) and
\(30^\circ\): the two local entries are \(+0.288675\) and \(-0.866025\),
while the assembled entry is \(-0.577350\). One face is obtuse even though
the edge sign is correct.

## 4. What is and is not proved

The safe implication is

\[
 \text{all transformed local angles nonobtuse}
 \Longrightarrow K_{ij}\le0\text{ for every edge}
 \Longrightarrow \text{the standard FEM maximum-principle proof can apply,}
\]

after irreducibility and boundary assumptions are checked. The reverse
implication is false. Positive definiteness of every \(A_T\) by itself does
not imply an \(M\)-matrix, a discrete maximum principle, or monotone
free-side traces. The Phase III mixed-boundary counterexample in the correction
audit uses SPD face tensors but produces a negative boundary increment.

## 5. Consequence for a differentiable solver

An anisotropic Hodge or electrical layer may safely expose the assembled edge
sign certificate as a diagnostic. It must not silently convert a local
nonobtuse test into a claimed necessary condition, and it must reject or mark
meshes for which \(K_{ij}>0\). A soft penalty on the positive part

\[
 \mathcal P(K)=\sum_{i\ne j}\max(K_{ij},0)^2
\]

is differentiable almost everywhere, but is not a proof of a homeomorphism.
For a guaranteed topology-preserving layer, mesh/boundary hypotheses and the
sign certificate remain explicit inputs or certified preconditions.

## 6. Why one fixed stencil does not settle the anisotropic route

For any single SPD tensor, an eigen-decomposition gives the positive algebraic
representation
\[
A=\lambda_1 e_1e_1^T+\lambda_2 e_2e_2^T,
\qquad \lambda_1,\lambda_2>0,
\quad e_1\perp e_2.
\]
Thus the obstruction found for the four directions
\(e_x,e_y,(1,1),(1,-1)\) is not an obstruction to *all* positive directional
representations. It is an obstruction to using that fixed global cone on the
given graph.

There are three possible escapes, each with a different cost:

1. **Adaptive rotated directions.** Use the principal directions per face. The
   resulting directions generally do not coincide with primal edges or with a
   globally conforming dual complex. Introducing virtual edges changes the
   graph and requires a new incidence/Hodge construction.
2. **(A)-metric orthogonalization.** Transform each face by
   (A_T^{-1/2}), use a circumcentric/cotangent discretization in the metric,
   then pull it back. This yields the weighted cotangent formula above, but
   positivity requires an anisotropic Delaunay condition and neighboring faces
   need not share one metric.
3. **Full SPD local Hodge star.** Retain non-diagonal local Hodge blocks in a
   specified primal/dual basis. This can represent every SPD tensor, but a
   generic (2\times2) edge-pair block is not by itself a conforming global
   Hodge system: the basis, degrees of freedom, orientation, and assembly must
   be defined, and standard local edge constructions may use larger blocks.
   The resulting block mixed system loses the interpretation as independent
   positive scalar conductances and does not automatically imply an
   (M)-matrix or topology guarantee.

Consequently, the Phase II fixed-stencil coverage percentages only ruled out
one diagonal positive-network design. They did not rule out adaptive metric
meshes or compatible mixed finite elements. The new (P_1) conjugacy module is
an exact facewise compatibility prototype, not an implementation of the full
block-Hodge option; it still lacks a scalable global dual complex.

## 7. Independent-check status

The two-face obtuse-angle calculation and the C7 anisotropic counterexample
were recomputed without calling the production helper. A nonuniform Tutte
stress benchmark is now recorded in `05_tutte_neural_layer.md`; the remaining
implementation gap is a scalable assembled-edge certificate for arbitrary
anisotropic meshes.

## 8. Mixed-boundary monotonicity: theorem search and exhaustive tiny search

The literature supports a maximum--minimum principle for positive scalar
conductance graphs and studies response matrices of circular planar networks,
but those facts do not imply ordered traces for an anisotropic facewise FEM
stream solve. Relevant references are [mixed Dirichlet--Neumann boundary value
problems on planar graphs](https://doi.org/10.1016/j.difgeo.2011.03.003),
[circular planar resistor-network response matrices](https://doi.org/10.1016/S0024-3795(98)10087-3),
and the [anisotropic diffusion monotonicity/discrete maximum-principle
literature](https://doi.org/10.1029/2008RG000277). None of these sources gives
the missing universal anisotropic complementary-trace theorem under only
(A_T\succ0).

To replace random testing by a finite exhaustive check, we enumerated all
(3^8=6561) assignments of face coefficients

\[
\mu_T\in\{-0.8i,0,+0.8i\}
\]

on the eight triangles of a (2\times2)-cell square. For each assignment we
assembled the facewise (P_1) matrix independently, fixed the bottom row to
zero and the top row to one, solved the three middle-row unknowns, and tested
both left and right boundary increments. Eighty-seven assignments violated
left-side monotonicity; the smallest observed left increment was
(-0.2749922961). The first counterexample in face order

\[
(-0.8i,-0.8i,-0.8i,-0.8i,+0.8i,0,0,0)
\]

produced middle row

\[
(-0.10057537,\ 0.31426928,\ 0.12142693).
\]

Every face tensor is SPD with determinant one. This is a decisive
fixed-(P_1) counterexample, not evidence that the continuum canonical QC map
loses its boundary order. Under the corrected CQ1 closure-homeomorphism
theorem, the canonical continuum map has ordered marked-side traces; what
remains open is the weaker statement that uniform ellipticity or an arbitrary
independent fixed-P1 complementary solve alone enforces that order. A valid
discrete theorem would need an assembled (M)-matrix/ordered-trace condition,
or a compatible primal--dual construction that prevents the independent solve
from leaving the stream space.

As a restricted positive check, the same (2\times2)-cell triangulation was
exhaustively tested with scalar face conductances in
\(\{1,2,10\}^8\). All 6561 systems were nonsingular and had nonnegative left
and right boundary increments; the smallest left increment was
\(0.0909090909\). This is consistent with the (M)-matrix structure of the
right-triangle isotropic stencil, but it is not a proof for arbitrary planar
graphs or anisotropic tensors.
