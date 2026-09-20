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

## 5. Anisotropic extension

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

At present only the isotropic square-grid construction is exact. The anisotropic
extension is open and must be tested with a tensor sign/positivity audit before
being treated as a decoder.

## 6. Relation to a PL map

The electrical tiling naturally produces orthogonal quadrilateral cells on a
primal-dual complex. To obtain a conventional PL map, each rectangle can be
split into two consistently oriented triangles. The split is topology-safe if
every rectangle has positive width and height and neighboring cells agree on
shared edges. This output need not use the original triangulation; changing the
mesh representation is allowed in Phase II.

The main unresolved issue is whether the triangulated tiling remains an accurate
Beltrami representation of the original facewise \(\mu\), rather than only a
topologically safe map.

## 7. Current decision

The primal-dual route remains a serious supporting candidate because it gives a
natural conjugacy and boundary-order mechanism. It should not yet be promoted to
the main neural layer until the \(\mu=0\) unit test and one anisotropic positive
Hodge-star construction have been implemented and measured.

## 8. References

- [Orthodiagonal maps, tilings of rectangles, and convergence to conformal maps](https://www.researchgate.net/publication/382692003_Orthodiagonal_Maps_Tilings_of_Rectangles_and_their_Convergence_to_Conformal_Maps) — discrete harmonic conjugates, electrical networks, and non-overlapping rectangle tilings.
- [Discrete complex analysis on planar quad-graphs](https://link.springer.com/chapter/10.1007/978-3-662-50447-5_2) — primal-dual discrete analytic structures and convergence context.
