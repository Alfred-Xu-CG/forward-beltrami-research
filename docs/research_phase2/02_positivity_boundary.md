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
sufficient in general.

The first phase diagram used a 4x4-vertex regular grid, constant \(\mu\), radii
\(0,0.2,0.4,0.6,0.8,0.9\), thirteen angles in \([0,\pi)\), and both square
diagonal orientations. The largest positive off-diagonal entry was:

\[
\begin{array}{c|rrrrrr}
|\mu|&0&0.2&0.4&0.6&0.8&0.9\\ \hline
\text{slash}&0&0.414&0.945&1.861&4.412&9.405\\
\text{backslash}&0&0&0&0.522&1.718&3.847
\end{array}
\]

The result is a decisive negative control against the shortcut
\(A\succ0\Rightarrow K_{ij}\le0\). Even a small anisotropy can violate the
ordinary Euclidean sign pattern for one diagonal orientation. The correct local
condition remains

\[
(\nabla\phi_i|_T)^\top A_T(\nabla\phi_j|_T)\le0
\quad\text{for every local pair }i\ne j,
\]

which is an anisotropic metric-geometry condition, not a scalar bound on
\(\|\mu\|\) alone.

As a separate graph-level check, random positive conductances spanning six
orders of magnitude were assigned to every edge of triangulated 4x4, 5x5, and
6x6 grids. The complementary bottom/top Dirichlet solve produced no negative
increment on either free side in 10,000 samples per grid size. This increases
confidence that a planar-network monotonicity principle may exist, but it is
not a proof and does not cover arbitrary planar graphs or anisotropic FEM
stiffness matrices.

## 5. References

- [Directed Tutte theorem](https://www.cs.tufts.edu/research/geometry/pdf/haas04planar.pdf) — positive directed equilibrium weights and non-overlapping convex cells under explicit graph/boundary hypotheses.
- [Tutte's planar embedding theorem and Floater generalizations](https://www.cs.harvard.edu/~sjg/papers/tutte.pdf) — positive convex-combination embeddings and their boundary assumptions.
