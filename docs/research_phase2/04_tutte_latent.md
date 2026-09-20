# Directed Positive Tutte Latent Coordinates

## 1. Decoder definition

Let \(G=(V,E)\) be a planar disk graph with one boundary cycle
\(\partial V\) and interior vertices \(V^\circ\). Let \(Y_{\partial}\in
\mathbb R^{|\partial V|\times2}\) be boundary coordinates in strict convex
cyclic order. For every interior vertex \(i\), let \(N(i)\) be its graph
neighbors and let

\[
p_{ij}>0,\qquad \sum_{j\in N(i)}p_{ij}=1.
\]

The directed Tutte decoder is the two-coordinate linear system

\[
Y_i=\sum_{j\in N(i)}p_{ij}Y_j,\qquad i\in V^\circ.
\tag{T-1}
\]

The weights need not be symmetric. Symmetry would give an undirected energy and
an SPD matrix, but the topological theorem is based on positive
convex-combination rows and graph hypotheses, not on symmetry.

For neural coordinates, logits \(\ell_{ij}\in\mathbb R\) are mapped to

\[
p_{ij}=\frac{\exp(\ell_{ij})}
{\sum_{k\in N(i)}\exp(\ell_{ik})}.
\tag{T-2}
\]

Finite logits therefore produce strictly positive rows automatically.

## 2. A boundary latent with a hard convexity guarantee

Passing an unconstrained array of boundary coordinates to (T-3) is unsafe:
even if every interior row is positive, a self-intersecting or non-convex
boundary invalidates the Tutte theorem. A minimal differentiable boundary
latent is an axis-aligned rectangle with learnable sampling density on each
side. Let there be \(q\geq1\) boundary edges per side and logits
\(r^{(s)}_k\), \(s\in\{0,1,2,3\}\), \(k=0,\ldots,q-1\). Define

\[
\alpha^{(s)}_k=
\frac{\exp r^{(s)}_k}{\sum_{t=0}^{q-1}\exp r^{(s)}_t}>0,
\qquad
(L_0,L_1,L_2,L_3)=(W,H,W,H),
\]

and side lengths \(d^{(s)}_k=L_s\alpha^{(s)}_k\). Starting from
\(X_0=(0,0)\), emit \(q\) vertices on each side and update

\[
X_{m+1}=X_m+d^{(s)}_k e_s,qquad
(e_0,e_1,e_2,e_3)=((1,0),(0,1),(-1,0),(0,-1)).
\tag{T-6}
\]

The final segment closes the cycle, because each row of \(\alpha\) sums to one.
Every emitted side increment is strictly positive for finite logits, so the
boundary is a counter-clockwise, non-degenerate rectangle with strictly
ordered samples along every side. The map from logits to vertices is smooth;
for one side its local derivative is

\[
\frac{\partial \alpha_k}{\partial r_t}
=\alpha_k(\mathbf 1_{k=t}-\alpha_t),
\tag{T-7}
\]

so it can be differentiated without a projection or a topology repair. The
implementation is `rectangle_boundary_from_logits` in
`src/qcopt/forward/tutte_directed_implicit.py`.

This is deliberately a limited latent. It learns side spacing and aspect
ratio, but not an arbitrary convex polygon. A future polygon latent must also
enforce cyclic edge directions, positive edge lengths, and the vector closure
\(\sum_k \ell_k e^{i\theta_k}=0\); simply applying softmax to vertex
coordinates does not provide that closure. The current rectangle latent is
therefore a theorem-safe boundary control, not evidence of arbitrary-μ
expressivity.

On an 8-by-8 cell structured triangulation (81 vertices, 32 boundary
vertices), a 4-by-8 boundary-logit tensor was composed with the directed
implicit solve. All 128 face determinants were positive and both boundary
logit and interior-row logit gradients were finite. This is a composition
test, not a universal theorem beyond the stated Tutte hypotheses.

## 3. Linear algebra and implicit differentiation

Order the interior vertices first. Let \(P_{II}\) contain the coefficients
from interior neighbors to interior vertices, and \(P_{IB}\) contain the
coefficients to boundary vertices. Then (T-1) is

\[
B(P)Y_I=P_{IB}Y_B,\qquad B(P)=I-P_{II}.
\tag{T-3}
\]

Under the standard graph connectivity assumptions, \(B\) is nonsingular. One
sparse factorization gives the forward interior coordinates. For an objective
\(L(Y)\), the adjoint solves

\[
B(P)^\top\lambda=\overline{Y_I}.
\tag{T-4}
\]

For a directed edge \(i\to j\), with \(y_j=Y_j\) if \(j\) is boundary or
interior, the logit derivative is

\[
\frac{\partial L}{\partial \ell_{ij}}
=p_{ij}\,\lambda_i^\top(y_j-Y_i).
\tag{T-5}
\]

The boundary derivative is the direct boundary loss plus
\(P_{IB}^\top\lambda\). This is a constant-number-of-solves implicit layer:
one forward sparse solve and one transpose sparse solve, rather than unrolling
iterations.

## 4. Hard-topology theorem and its exact scope

A directed Tutte theorem gives the following conditional statement. If the
embedded plane graph is connected to the boundary in the required
3-connected sense, the boundary cycle has no repeated vertices and is placed
in convex order, and all internal directed weights are positive, then the
unique equilibrium positions of the interior vertices form non-overlapping
convex cells. For a triangulation, this gives a straight-line planar embedding
with positive face areas, provided the theorem's nondegeneracy hypotheses are
met.

This is a genuine hard-topology decoder guarantee, but it is conditional on:

1. graph validity and boundary-cycle assumptions;
2. strict convexity or the appropriate generalized boundary condition;
3. strictly positive weights;
4. no degenerate boundary or face geometry.

The theorem does not imply:

- that the decoded map has a prescribed Beltrami coefficient;
- that the decoded map is close to an arbitrary target QC map;
- that the smallest determinant margin is numerically well conditioned;
- that a boundary latent parameterization remains convex;
- that a zero-weight limit preserves the theorem.

Therefore this route is a map-primary hard decoder, not by itself a
Beltrami-equation solver.

## 5. Expressivity question

For a fixed valid embedding \(Y\), an interior vertex must lie in the convex
hull of its neighbors for a positive-row representation to exist. If it lies
strictly inside that hull, one can often construct positive directed weights,
but this needs to be checked for the chosen graph and cannot be replaced by
the statement that every arbitrary embedding is representable.

The useful research question is therefore:

\[
\text{valid target PL homeomorphism}
\longrightarrow
\text{strictly positive local barycentric rows?}
\]

The answer is graph- and embedding-dependent. A positive Tutte latent is
expressive enough for a broad family of convex-boundary disk embeddings, but
there is no evidence yet that it can represent every arbitrary Beltrami field
without increasing graph connectivity or changing the boundary.

As a direct local expressivity test, a 16x16 structured triangulation was given
a boundary-fixed smooth deformation with amplitude 0.12. For every one of its
225 interior vertices, a linear program found strict positive neighbor weights
that reproduced the target vertex exactly. The smallest maximin row weight was
0.139, the 5th percentile was 0.141, and the target had
\(\max_T|\mu_T|=0.6595\) with all face determinants positive. A larger amplitude
0.2 produced flipped faces and \(\max|\mu|>1\), so its eight failed rows are
not an expressivity counterexample; the target itself was no longer a valid QC
homeomorphism.

## 6. Conditioning and neural usability

The hard-topology theorem is qualitative. If logits have a large spread, one
weight approaches one and the others approach zero. The equilibrium remains
topologically valid in exact arithmetic, but:

- the smallest face area can approach zero;
- \(B(P)\) becomes poorly conditioned;
- the adjoint gradient in (T-5) can become numerically unstable;
- finite precision may produce apparent flips.

A practical layer must therefore report both a topological certificate and a
conditioning certificate:

\[
\min_T \operatorname{area}(Y_T),\qquad
\kappa(B),\qquad
\|\nabla_{\ell}L\|,\qquad
\text{forward/backward residuals}.
\]

## 7. Existing project evidence

Legacy audits in docs/forward_beltrami already contain realistic-resolution
directed-Tutte experiments on structured 256-squared grids and a nonuniform
12,384-vertex Delaunay mesh. They reported zero flipped faces and finite
implicit VJPs, while large logit spreads produced very small determinant
margins. Those results support the conditional theorem and expose conditioning;
they do not establish arbitrary-mu expressivity.

The current Phase II work will reuse those artifacts without re-running them
just to fill a ledger. The next new experiment should vary graph connectivity
and boundary latent parameterization, not repeat the same fixed-boundary audit.

## 8. Prior art

- [Haas et al., directed Tutte theorem](https://www.cs.tufts.edu/research/geometry/pdf/haas04planar.pdf) — positive directed equilibrium weights and non-overlapping convex cells under explicit plane-graph and boundary hypotheses.
- [An elementary proof of Tutte's planar embedding theorem](https://www.cs.harvard.edu/~sjg/papers/tutte.pdf) — positive convex-combination embeddings and the role of a convex boundary.

## 9. Preliminary decision

Layer B is currently the strongest hard-bijection candidate because its
topology certificate is structural and its backward pass is an adjoint sparse
solve. It is not yet the preferred Beltrami solver: the remaining work is to
measure QC/Beltrami expressivity and to design a boundary latent that preserves
the theorem hypotheses.
