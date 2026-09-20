# Primal-Dual QC Electrical Mapping

## 1. Motivation

The mixed FEM route computes two scalar fields on the same primal mesh and then
has to prove that a projected conjugate is sufficiently exact. A primal-dual
construction changes the representation: the primal potential \(u\) lives on
primal vertices, while its harmonic conjugate \(v^\ast\) lives on dual vertices.
The target cells are rectangles or orthogonal quadrilaterals by construction.

This route is therefore a candidate for a stronger topology certificate, but it
may require a diamond/medial refinement instead of the original image
triangulation.

## 2. Isotropic electrical construction

Let \(G^\bullet=(V^\bullet,E^\bullet)\) be a planar graph with a dual graph
\(G^\circ\). Assign a positive conductance \(c_e>0\) to each primal edge
\(e=(i,j)\). A primal potential solves

\[
\sum_{j:(i,j)\in E^\bullet}c_{ij}(u_i-u_j)=0
\qquad (i\in V^\bullet_{\mathrm{free}}).
\tag{PD-1}
\]

The boundary has two Dirichlet arcs, for example \(u=0\) on the left and
\(u=1\) on the right, and zero net current on the other two arcs.

Define the oriented edge current

\[
I_{ij}=c_{ij}(u_i-u_j),\qquad I_{ji}=-I_{ij}.
\tag{PD-2}
\]

Kirchhoff conservation is precisely the statement that the current 1-form is
closed on the dual graph. Therefore one may integrate it to a dual potential
\(v^\ast:V^\circ\to\mathbb R\), up to an additive constant. If \(e\) separates
dual faces \(f_L,f_R\), the discrete conjugacy relation is

\[
v^\ast(f_L)-v^\ast(f_R)=I_e,
\tag{PD-3}
\]

with signs fixed by the orientation convention.

The pair \((u,v^\ast)\) defines an electrical rectangle tiling. A primal edge
controls one rectangle width \(|u_i-u_j|\); the crossed dual edge controls its
height \(|v^\ast(f_L)-v^\ast(f_R)|\). Positivity of \(c_e\) makes these
width-height products nonnegative, and planar duality gives the tiling
adjacency.

## 3. Modulus and effective conductance

The discrete energy is

\[
E_G(u)=\frac12\sum_{e=(i,j)}c_e(u_i-u_j)^2.
\tag{PD-4}
\]

For the normalized left/right potential, the total current crossing any
transverse cut is the effective conductance \(C_{\mathrm{eff}}\). The electrical
rectangle has width one and height

\[
M=C_{\mathrm{eff}}
\]

under the convention in which the left/right potential difference is one. If
the opposite pair is normalized instead, the reciprocal conductance gives the
reciprocal modulus. This is the discrete analogue of the continuum
energy/modulus identity.

## 4. Minimal \(\mu=0\) unit test

Take a rectangular square grid with unit conductances and left/right Dirichlet
values. The solution is

\[
u_{i,j}=\frac{i}{n_x-1}.
\]

Every horizontal current is constant and every vertical current is zero.
Integrating (PD-3) gives a dual potential linear in the vertical index. Each
cell maps to a congruent axis-aligned rectangle, the interiors are disjoint,
and the boundary order is explicit. This is the smallest decisive test for the
electrical representation before introducing anisotropy.

The test is intentionally not a proof for arbitrary planar graphs. It checks
orientation conventions, current conservation, dual integration, modulus, and
non-overlap on a case whose answer is known exactly.

In a 9-by-7 vertex grid, the sparse graph solve recovered
\(u_{i,j}=i/8\) with maximum error \(4.44\times10^{-16}\). Every horizontal
current was \(0.125\) up to \(3\times10^{-16}\), the integrated dual potential
had top value \(M=0.75\), and the summed rectangle-cell area was \(0.75\).
All cell widths and heights were positive. This validates the isotropic
orientation and modulus bookkeeping, but it is intentionally only the
\(\mu=0\) unit test.

This unit test is now reproducible through
src/qcopt/forward/electrical_rectangle.py and
tests/test_forward_electrical_rectangle.py. The implementation deliberately
rejects a nonuniform current rather than silently treating a non-closed dual
1-form as a conjugate potential.

## 5. The mixed-boundary rectangle theorem that this route can inherit

The isotropic part of this route is not merely a square-grid observation. A
precise discrete theorem is proved by Hersonsky in *Boundary Value Problems on
Planar Graphs and Flat Surfaces with Integer Cone Singularities II: The Mixed
Dirichlet--Neumann Problem*. In the notation needed here, let \(\mathcal R\)
be a topological quadrilateral with a cellular decomposition whose two-cells
are triangles or quadrilaterals. Put a positive symmetric conductance
\(c(e)>0\) on every edge. Let \(g\) solve the graph mixed problem

\[
g=0\ \text{on the left arc},\qquad
g=k\ \text{on the right arc},\qquad
\partial_n g=0\ \text{on the top and bottom arcs},
\tag{PD-6}
\]

and \(\Delta_c g=0\) at free vertices. If \(H\) is the total outward flux
through the right arc, then the theorem constructs a Euclidean rectangle of
width \(k\) and height \(H\), and assigns one embedded Euclidean rectangle to
each edge of the cellular complex so that these edge rectangles form a tiling.
The assignment preserves the boundary order and satisfies

\[
\operatorname{Area}(S_{\mathcal R})=E_c(g).
\tag{PD-7}
\]

The convention in that theorem takes the edge-potential difference as one
rectangle side and \(c(e)\) times that difference (the edge flux) as the other
side. This is the same content as (PD-2)--(PD-4), up to swapping the names of
the horizontal and vertical axes. In particular, for a quadrilateral the
result is an *embedded* rectangle tiling, not only a possibly immersed flat
surface. The paper's more general \(m\)-connected theorem is weaker for our
purpose: it permits a singular flat surface and immersion, so it must not be
quoted as a global planar homeomorphism theorem.

This result closes one specific gap in the isotropic primal-dual route:
positive conductances plus the quadrilateral mixed boundary problem provide a
rigorous boundary-order/non-overlap mechanism on the graph-to-rectangle
complex. It does **not** yet close the neural-layer problem for three reasons.

1. The theorem maps graph edges to rectangles. To expose a conventional
   piecewise-affine map, we still need a deterministic vertex realization and
   a consistent diagonal split of every rectangle.
2. Its conductances are scalar and isotropic. Replacing them by a tensor
   \(A(\mu)\) requires a compatible discrete Hodge star, which is not supplied
   by the theorem.
3. The theorem preserves graph energy and tiling area, not the facewise
   Beltrami coefficient of an arbitrary input field. Beltrami fidelity remains
   a separate approximation measurement.

Therefore the theorem upgrades the route's status from “only a \(\mu=0\)
unit test” to “a rigorously topology-safe isotropic decoder on an admissible
planar graph,” while leaving anisotropic QC fidelity and differentiable
implementation as open gates.

## 6. Anisotropic extension

For a Beltrami field, the continuum energy is

\[
\int_\Omega \nabla u^\top A(\mu)\nabla u\,dx.
\]

The discrete analogue requires a positive discrete Hodge star mapping primal
1-cochains to dual 1-cochains. On an orthogonal primal-dual cell one can use a
diagonal star

\[
(*_A\alpha)(e^\ast)=c_e(A)\,\alpha(e),
\qquad c_e(A)>0.
\tag{PD-5}
\]

The difficult condition is not merely \(A\succ0\). The mesh geometry and the
chosen primal-dual pairing must make all \(c_e(A)\) positive and preserve the
closedness needed for dual integration. Ordinary P1 FEM on an arbitrary
triangulation generally produces a full local tensor matrix rather than a
diagonal positive edge star.

Candidate realizations are:

1. metric pullback to locally orthogonal cells;
2. intrinsic or anisotropic Delaunay retriangulation;
3. diamond/medial refinement with compatible primal and dual scalar spaces;
4. compatible finite elements or discrete exterior calculus.

At present only the isotropic square-grid construction is a complete decoder.
The fixed diamond stencil below is an exact algebraic tensor representation,
but its positive-conductance region collapses at moderate distortion, so it is
not yet an anisotropic decoder.

There is an immediate algebraic obstruction to the simplest anisotropic
shortcut. On an axis-aligned square grid, a diagonal edge-conductance star can
only represent an energy of the form

\[
c_x(\partial_x u)^2+c_y(\partial_y u)^2.
\]

Testing the affine fields \(u=x\) and \(u=y\) forces
\(c_x=A_{11}\) and \(c_y=A_{22}\). Testing \(u=x+y\) then misses the continuum
cross term \(2A_{12}\partial_xu\,\partial_yu\) whenever \(A_{12}\ne0\). Thus
ordinary positive horizontal/vertical conductances cannot exactly encode a
general complex Beltrami coefficient. A rotated/metric orthogonal complex, a
diamond refinement, or a full non-diagonal discrete Hodge star is necessary.

### A concrete positive-stencil gate

To make the obstruction quantitative, consider the smallest enriched fixed
stencil with directions
\[
e_x=(1,0),\quad e_y=(0,1),\quad e_+=(1,1),\quad e_-=(1,-1).
\]
For \(A=\begin{bmatrix}a&b\\b&c\end{bmatrix}\), the exact decomposition
\[
A=c_x e_xe_x^\top+c_y e_ye_y^\top
  +c_+e_+e_+^\top+c_-e_-e_-^\top
\tag{PD-8}
\]
has the unique coefficients
\[
c_x=a-|b|,\qquad c_y=c-|b|,\qquad
c_+=\max(b,0),\qquad c_-=\max(-b,0).
\tag{PD-9}
\]
Thus this stencil represents the tensor exactly, but it is a positive
conductance network if and only if
\[
a\ge |b|\quad\text{and}\quad c\ge |b|.
\tag{PD-10}
\]
Positive definiteness alone does not imply (PD-10). The implementation in
src/qcopt/forward/anisotropic_hodge.py returns both the exact reconstruction
and the positivity flag, so an infeasible tensor cannot be silently treated as
a valid Hodge star.

For the Beltrami tensor \(A(\mu)\), an angular sweep of 4096 coefficients at
each radius gave the following positive fractions:

| \(|\mu|\) | fraction with all four conductances nonnegative |
|---:|---:|
| 0.0 | 1.0000 |
| 0.2 | 1.0000 |
| 0.4 | 1.0000 |
| 0.6 | 0.1846 |
| 0.8 | 0.0322 |
| 0.9 | 0.0068 |

The tensor reconstruction error over all samples was at most
\(3.6\times10^{-15}\). Consequently, the failure at larger distortion is not
a linear-algebra approximation error: it is a sign failure that prevents this
fixed positive primal-dual stencil from representing most Beltrami phases.
The full JSON record is
artifacts/anisotropic_hodge_audit/anisotropic_hodge_audit.json. Enlarging or
rotating the direction dictionary may improve the cone, but then the decoder
graph and the planar tiling theorem must be re-established.

## 7. Relation to a PL map

The electrical tiling naturally produces orthogonal quadrilateral cells on a
primal-dual complex. To obtain a conventional PL map, each rectangle can be
split into two consistently oriented triangles. The split is topology-safe if
every rectangle has positive width and height and neighboring cells agree on
shared edges. This output need not use the original triangulation; changing the
mesh representation is allowed in Phase II.

The main unresolved issue is whether the triangulated tiling remains an accurate
Beltrami representation of the original facewise \(\mu\), rather than only a
topologically safe map.

## 8. Current decision

The primal-dual route remains a serious supporting candidate because it gives a
natural conjugacy and, for an admissible isotropic quadrilateral network, a
published embedded rectangle-tiling theorem. It should not yet be promoted to
the main Beltrami neural layer: the graph-tiling theorem still needs a practical
PL realization, and the fixed positive anisotropic stencil fails most phases
once \(|\mu|\) is above 0.6. A rotated or adaptive Hodge construction would
need a new topology and differentiability audit.

## 9. References

- [Orthodiagonal maps, tilings of rectangles, and convergence to conformal maps](https://www.researchgate.net/publication/382692003_Orthodiagonal_Maps_Tilings_of_Rectangles_and_their_Convergence_to_Conformal_Maps) — discrete harmonic conjugates, electrical networks, and non-overlapping rectangle tilings.
- [Discrete complex analysis on planar quad-graphs](https://link.springer.com/chapter/10.1007/978-3-662-50447-5_2) — primal-dual discrete analytic structures and convergence context.
- [Hersonsky, *Boundary Value Problems on Planar Graphs and Flat Surfaces with Integer Cone Singularities II*](https://arxiv.org/abs/1006.0026) — mixed Dirichlet--Neumann graph problem, embedded rectangle tiling theorem for quadrilaterals, and the energy--area identity.
