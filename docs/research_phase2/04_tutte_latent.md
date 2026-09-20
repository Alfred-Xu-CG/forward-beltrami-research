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

## 2. Linear algebra and implicit differentiation

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

## 3. Hard-topology theorem and its exact scope

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

## 4. Expressivity question

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

## 5. Conditioning and neural usability

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

## 6. Existing project evidence

Legacy audits in docs/forward_beltrami already contain realistic-resolution
directed-Tutte experiments on structured 256-squared grids and a nonuniform
12,384-vertex Delaunay mesh. They reported zero flipped faces and finite
implicit VJPs, while large logit spreads produced very small determinant
margins. Those results support the conditional theorem and expose conditioning;
they do not establish arbitrary-mu expressivity.

The current Phase II work will reuse those artifacts without re-running them
just to fill a ledger. The next new experiment should vary graph connectivity
and boundary latent parameterization, not repeat the same fixed-boundary audit.

## 7. Prior art

- [Haas et al., directed Tutte theorem](https://www.cs.tufts.edu/research/geometry/pdf/haas04planar.pdf) — positive directed equilibrium weights and non-overlapping convex cells under explicit plane-graph and boundary hypotheses.
- [An elementary proof of Tutte's planar embedding theorem](https://www.cs.harvard.edu/~sjg/papers/tutte.pdf) — positive convex-combination embeddings and the role of a convex boundary.

## 8. Preliminary decision

Layer B is currently the strongest hard-bijection candidate because its
topology certificate is structural and its backward pass is an adjoint sparse
solve. It is not yet the preferred Beltrami solver: the remaining work is to
measure QC/Beltrami expressivity and to design a boundary latent that preserves
the theorem hypotheses.
