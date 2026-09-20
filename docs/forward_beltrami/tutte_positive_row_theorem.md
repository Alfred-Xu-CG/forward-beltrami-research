# Route D: what positive directed Tutte weights do and do not guarantee

The earlier ledger treated a nonsymmetric positive-row Tutte theorem as
unresolved. A literature check narrows that statement. Floater's disk
parameterization formulation allows arbitrary nonnegative weights that sum to
one, and the planar Tutte theorem is commonly stated for positive barycentric
coefficients, not necessarily a symmetric edge energy. See [Floater 1997,
*Parametrization and smooth approximation of surface
triangulations*](https://web.engr.oregonstate.edu/~grimmc/content/research/Floater97.html),
the directed-weight discussion in [Tutte's barycenter method](https://monge.univ-mlv.fr/~colinde/pub/00tutte.pdf),
and the barycentric-system formulation in [Barycentric systems and
stretchability](https://doi.org/10.1016/j.dam.2005.12.009).

## Conditional theorem used by the decoder

Let `G` be a planar 3-connected graph with one designated outer cycle. Place
the outer-cycle vertices in strictly convex cyclic order. For every interior
vertex `i`, choose coefficients `w_ij>0` on all graph neighbors and normalize
each row so that `sum_j w_ij=1`. The directed linear system

\[
 x_i=\sum_j w_{ij}x_j,
 \qquad
 y_i=\sum_j w_{ij}y_j
\]

has a unique solution for the interior coordinates, and the resulting straight
line drawing is a planar embedding with no flipped faces. Symmetry
`w_ij=w_ji` is not part of this conditional statement. Symmetry is useful for
an SPD energy and solver conditioning, but it is not the topological invariant.

The proof obligation is not “positive matrix implies injectivity” in isolation.
It is the combination of: (i) strict convex-combination equations, (ii) a
convex boundary, and (iii) the planar 3-connected graph hypotheses. Removing
any of these changes the claim. In particular, positivity alone does not
promise a QC coefficient fit, a useful determinant margin, or robustness when
weights approach zero.

## Consequences for this project

1. The directed-logit layer is a legitimate hard-topology decoder when its
   mesh graph and boundary satisfy the theorem hypotheses and the softmax rows
   are strictly positive.
2. The very small determinants observed at logit spread 3 are conditioning
   warnings, not counterexamples to the theorem. They mean that a topology
   certificate can remain true while gradients and finite precision become
   unusable.
3. This theorem is a decoder guarantee, not an arbitrary-`mu` solver. The map
   produced by positive directed weights may have a Beltrami field far from the
   requested target. Expressivity and QC-consistency must still be measured
   separately.
4. If a learned parameterization permits exact zero weights, disconnected
   support, non-convex boundary vertices, or a graph with a cut vertex, the
   theorem no longer applies. A production layer should use finite logits,
   verify graph/boundary assumptions once, and report the determinant margin as
   a numerical conditioning diagnostic.

## Existing realistic-resolution evidence

- `artifacts/tutte_symmetric_nonsymmetric_256/tutte_symmetric_nonsymmetric_audit.json`
  uses a `256²` structured triangulation. Symmetric positive weights and two
  directed positive-row stresses retained zero flips and independent rectangle
  certificates; the spread-3 directed case reached minimum signed area
  `4.94e-10`.
- `artifacts/tutte_directed_implicit_256/tutte_directed_implicit_audit.json`
  uses the sparse implicit directed layer at `256²` with `65,025` interior
  rows. Logit spreads 1 and 3 had finite outputs and VJPs, zero flips, and
  minimum signed areas `3.66e-3` and `1.07e-10`.
- `artifacts/tutte_directed_unstructured_12000/tutte_directed_unstructured_audit.json`
  uses a nonuniform Delaunay mesh with `12,384` vertices and `24,382` faces;
  both directed positive-row cases had finite implicit gradients and zero
  flipped faces, with spread-3 minimum determinant `8.01e-12`.

These results are consistent with the conditional positive-row theorem. The
remaining Route-D work is therefore not to search blindly for a nonsymmetric
counterexample under the theorem hypotheses, but to verify those hypotheses
for every mesh, quantify conditioning, and test whether learned positive rows
can represent the desired Beltrami fields.
