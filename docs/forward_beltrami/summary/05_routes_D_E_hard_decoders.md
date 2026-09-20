# Routes D and E: positive hard decoders and M-matrix conductivities

## Route D: Tutte/Floater

The Tutte/Floater principle solves a sparse barycentric system whose interior
rows are positive combinations of neighbours and whose boundary is convex and
fixed. Under the graph and boundary hypotheses, the planar embedding has strong
non-folding guarantees. Sparse implicit differentiation gives a practical
gradient through the linear solve.

The 512² positive-weight stress had 787,456 edges, minimum determinant
`2.704e-6`, and zero flips. Directed positive rows and a 12,384-vertex
nonuniform Delaunay control also had finite VJPs and zero flips, although their
minimum determinants became as small as `8.01e-12` under extreme logit spread.

The limitation is expressivity: positivity and the chosen graph restrict which
anisotropic fields can be represented. Conditioning also worsens as weights
approach zero. The theorem does not say that every arbitrary Beltrami
coefficient is represented by one fixed graph.

## Route E: M-matrix conductivity decoder

An M-matrix discretization uses nonnegative directional conductances to build a
monotone scalar elliptic operator. For a Beltrami coefficient `mu`, the
associated symmetric positive-definite tensor can be fitted by a cone of
directional rank-one matrices. If the tensor lies outside that finite positive
cone, the fit has a nonzero residual even though `|mu|<1`.

The cone phase diagram showed exact radius-1 fits for small magnitudes, recovery
at radius 2 for magnitude `0.6`, and substantial residuals for magnitudes
`0.8`--`0.9` even at radius 4. A wider two-ring unstructured graph reduced the
local fit residual but produced 1,641 flipped original faces because the graph
was no longer planar. Raster continuation regularization reduced operator
errors to `6.71e-2`/`1.68e-2` at 256²/512² in one control, but it changes the
problem and is not a general theorem.

The radius-4/6/8 active-set sweep was resource limited: a 512² run lasted
about 63 minutes and a 256² run about 23 minutes without receipts. The
vectorized three-direction enumeration is itself expensive.

## Decision

Route D is a strong topology-safe decoder for a restricted positive graph.
Route E explains a fundamental local expressivity barrier. Neither is an
arbitrary-mu universal forward solver without a new planar enrichment and
global compatibility construction.
