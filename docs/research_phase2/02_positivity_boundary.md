# Positivity, M-Matrix, and Boundary Monotonicity

## 1. Discrete operator

For a P1 mesh with facewise symmetric positive-definite tensors A_T,

    K_ij = sum_T |T| (grad phi_i|_T)^T A_T (grad phi_j|_T).

The row sum vanishes because sum_j phi_j = 1, hence K 1 = 0.

For an interior vertex i, if K_ij <= 0 for every j != i, then the equation
K x = 0 can be written as

    x_i = sum_{j != i} p_ij x_j,
    p_ij = -K_ij / K_ii >= 0,
    sum_{j != i} p_ij = 1.

Thus the exact equivalence is:

    row-wise M-matrix sign pattern <=> convex-combination equation.

Strict positivity and irreducibility are separate conditions. They are not
automatically implied by A_T positive definite.

## 2. Relation to injectivity

Positive interior convex combinations plus an ordered convex boundary are a
sufficient Tutte/Floater-type hypothesis for a non-overlapping straight-line
embedding. The M-matrix sign pattern alone is not a necessary-and-sufficient
criterion for a global homeomorphism. In particular, local positive signed area
does not by itself control boundary order or global degree.

## 3. Current boundary-monotonicity status

The unresolved question is:

> For the complementary mixed solve with bottom/top Dirichlet values and
> left/right natural flux, does an irreducible positive planar graph Laplacian
> force the free side traces to be strictly monotone?

A first tiny search on a 3x3-vertex, two-triangle-per-cell grid sampled 100,000
facewise coefficients with |mu_T| < 0.97. It found no negative increment on
either free vertical side. This is only numerical evidence: the search did not
prove monotonicity, did not cover all M-matrix meshes, and did not justify a
continuum theorem. A targeted graph counterexample search and a proof attempt
remain required.

## 4. Geometry of the sign condition

For isotropic A_T = alpha_T I, the off-diagonal entries reduce to weighted
cotangent expressions, so non-obtuse/Delaunay geometry is a familiar sufficient
condition. For anisotropic A_T, the corresponding condition is naturally
expressed in the metric induced by A_T; ordinary Euclidean non-obtuseness is not
sufficient in general. The next experiment will vary |mu|, its argument, and
the diagonal orientation on a regular grid, recording the sign of every
off-diagonal entry.

## 5. References

- [Directed Tutte theorem](https://www.cs.tufts.edu/research/geometry/pdf/haas04planar.pdf) — positive directed equilibrium weights and non-overlapping convex cells under explicit graph/boundary hypotheses.
- [Tutte's planar embedding theorem and Floater generalizations](https://www.cs.harvard.edu/~sjg/papers/tutte.pdf) — positive convex-combination embeddings and their boundary assumptions.
