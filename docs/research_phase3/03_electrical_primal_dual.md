# Corrected electrical primal--dual route

Status: the rectangular reference is corrected and independently checked;
general anisotropic compatibility and neural-layer scalability remain open.

## 1. Isotropic graph problem

Take a rectangular vertex grid
\[
 V=\{(i,j):0\le i\le n_x-1,\ 0\le j\le n_y-1\}.
\]
Every horizontal edge \(((i,j),(i+1,j))\) has conductance \(c^h_{j,i}>0\),
and every vertical edge \(((i,j),(i,j+1))\) has conductance
\(c^v_{j,i}>0\). The graph energy is
\[
 E(u)=\sum_{j,i}c^h_{j,i}(u_{i+1,j}-u_{i,j})^2
     +\sum_{j,i}c^v_{j,i}(u_{i,j+1}-u_{i,j})^2.
\]
The left boundary is fixed to zero, the right boundary to one, and the top
and bottom rows are natural. The free vertex equations are the sparse weighted
graph Laplacian equations \(Lu=0\). Strict positivity of all conductances and
the two Dirichlet sides give a unique solution.

For a unit-conductance grid, \(u_{i,j}=i/(n_x-1)\). There are \(n_y\) parallel
rows, so
\[
 M=E(u)=\text{right flux}=\frac{n_y}{n_x-1}.
\tag{EPD-1}
\]
The corrected dual construction has \(n_y+1\) horizontal dual levels and
\(n_y\) strips. Its strip height is the row current and its width is the
horizontal potential drop. Therefore the summed dual area is again \(M\).
The old Phase II implementation used only \(n_y-1\) strips and returned
\((n_y-1)/(n_x-1)\), which is the C1 off-by-one error.

## 2. Independent invariants

Three quantities are computed separately in the reference implementation:

1. graph energy \(u^\mathsf{T}Lu\);
2. total right-boundary flux, the sum of conductance times potential drop on
   the rightmost horizontal edges;
3. reconstructed dual area.

The result is accepted only when these agree to tolerance and when every row
has path-independent horizontal current, which is the extra condition needed
for a literal rectangular strip tiling. The weighted helper therefore does not
claim a theorem for an arbitrary positive planar network.

The tests use the exact values
\[
 (n_x,n_y)=(2,2),(3,2),(3,3),(9,7)
 \quad\Longrightarrow\quad M=2,1,1.5,0.875,
\]
and a nonuniform four-row conductance example whose independently computed
energy, flux, and area all equal \(\sum_j c_j/(n_x-1)\).

## 3. Discrete duality and the missing compatibility condition

In a continuum conductivity problem, \(q=A\nabla u\) is divergence-free and
the rotated one-form \(Jq\) has a global stream potential on a simply connected
domain. A graph analogue needs three separate complexes:

* a primal vertex--edge incidence matrix \(B\);
* a diagonal edge Hodge star \(C\) containing conductances;
* a dual incidence operator that integrates the rotated edge fluxes.

The primal equation is \(BCB^\mathsf{T}u=0\) on free vertices. A dual potential exists
only if the rotated flux vector lies in the image of the dual coboundary. On a
rectangular simply connected grid this is equivalent to the cycle sums being
zero and is satisfied by Kirchhoff conservation. On a general mesh, a naive
four-direction stencil does not automatically supply that dual complex.

## 4. Anisotropic extension

For a facewise tensor \(A_T\succ0\), the \(P_1\) stiffness contribution is
\[
 K^T_{ij}=|T|(\nabla\phi_i)^T A_T\nabla\phi_j.
\]
One may try to factor \(A_T\) into positive directional conductances, but the
four-direction representation has a free signed parameter and may have no
nonnegative member. The precise feasibility interval is recorded in
`04_mmatrix_delaunay.md`; positivity is a certificate, not an automatic
consequence of ellipticity.

## 5. Differentiation and scalability

For fixed sparsity, differentiating a linear solve uses the adjoint equation
\(K^T\lambda=\partial\ell/\partial u\). The solve is differentiable, but
factorization and storage scale with the mesh and with coefficient changes.
The rectangular reference is therefore a validation oracle, not yet a
million-vertex neural layer. A future layer must expose matrix-free stencil
application, checkpointing or recomputation, and an explicit topology
certificate; the present route does not prove these simultaneously.

## 6. Route conclusion so far

The corrected isotropic rectangle is a reliable primal--dual benchmark and
detects the old bookkeeping bug. It does not by itself provide a fast,
general-mesh, anisotropic, fold-free differentiable solver. The missing step is
structure-preserving discrete conjugacy, treated in the next route document.
