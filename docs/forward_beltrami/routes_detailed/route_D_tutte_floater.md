# Route D — Tutte/Floater positive barycentric embeddings

## 1. Problem class

Route D does not prescribe a facewise Beltrami coefficient directly. It maps a
triangulated disk with fixed boundary coordinates to an interior embedding that
is guaranteed to remain planar under classical graph hypotheses.

Let `B` be the boundary vertex set and `I` the interior set. The target boundary
`Y_B` must be a convex counter-clockwise polygon. For each `i in I`, choose
strictly positive weights `w_ij` for graph neighbours `j in N(i)` and define

`p_ij=w_ij/s_i`, `s_i=sum_{k in N(i)}w_ik`.

The barycentric equation is

`Y_i-sum_{j in N(i)}p_ijY_j=0`.                      (D.1)

Separating interior and boundary vertices gives

`A(w)Y_I=C(w)Y_B`.                                  (D.2)

The matrix is sparse and an M-matrix-like row structure appears because the
diagonal is positive and off-diagonal coefficients are nonpositive.

## 2. Why positivity prevents folding under hypotheses

The Tutte theorem states, roughly, that for a suitable 3-connected planar graph,
positive barycentric coordinates and a convex boundary produce a straight-line
planar embedding. The theorem depends on all hypotheses. Positivity alone on an
arbitrary nonplanar or incorrectly ordered graph is not enough.

The implementation independently checks the boundary polygon and recomputes
mapped triangle orientations after solving (D.2).

## 3. Learned weights

Neural logits `s_ij` can be converted to positive weights by

`w_ij=exp(s_ij)` or `w_ij=softplus(s_ij)+epsilon`.

This makes the forward solve differentiable with respect to the logits, but
extreme logits create nearly zero weights and poor conditioning. The route's
topological guarantee requires strictly positive values, not merely numerical
nonnegativity.

## 4. Implicit derivative

Let `Y_I=A(w)^{-1}C(w)Y_B`. For a boundary or weight perturbation,

`A dY_I=dC Y_B+dY_B-A_w[dw]Y_I`.                 (D.3)

For a loss covector `g_I`, solve

`A^T lambda=g_I`.                                  (D.4)

Then the weight VJP is obtained from

`dL=lambda^T[dC Y_B-A_w[dw]Y_I]`.

For a local row, differentiating normalized weights gives

`dp_ij/dw_iell=(delta_jell s_i-w_ij)/s_i^2`.

The code computes this local expression and combines the contributions of both
endpoints of an undirected edge.

## 5. Experiments and limitation

512² positive-weight stresses reached zero flips and minimum determinant about
`2.704e-6`. Nonuniform Delaunay directed rows also had finite VJPs and zero
flips, but minimum determinants can approach `8e-12` under large logit spreads.

The crucial limitation is representational. A positive graph embedding is not
an arbitrary solution of `F_barz=mu F_z`. Route D is therefore a hard-topology
decoder or initializer whose graph/weight family must be enlarged or coupled
to another route for Beltrami expressivity.
