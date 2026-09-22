# Route III-A/B: compatible operators and the finite-direction obstruction

Status: builder derivation and targeted source audit, 2026-09-22; independent
Route III checker still required. This document does not close Route III.
Authority: `PLAN.md`, Sections 32–39. The research card is in `WORKLOG.md`.
No decoder implementation or application benchmark is claimed here; Section 7
records only a tiny mathematical spot-check of the derived cone frontier.

## 1. Exact claim and its scope

Write \(\mathbb S_+^2\) for positive **semidefinite** symmetric matrices and
\(\mathbb S_{++}^2\) for strictly positive definite matrices. For a fixed,
finite list of nonzero vectors \(D=(d_1,\ldots,d_m)\subset\mathbb R^2\), set

\[
 C_D=\left\{\sum_k c_kd_kd_k^T:c_k\geq0\right\},\qquad
 C_D^{>0}=\left\{\sum_k c_kd_kd_k^T:c_k>0\right\}.
\]

**Claim FD.** \(C_D\) is a closed finitely generated convex cone contained
strictly in \(\mathbb S_+^2\). More strongly,
\(\mathbb S_{++}^2\not\subset C_D\): there are strictly SPD tensors outside
it, not merely missing rank-one boundary tensors. Consequently neither
\(C_D^{>0}\), nor its closure, represents all SPD tensors. The same is true
when the target is restricted to determinant-one SPD tensors.

This is a statement about an additive, positive, scalar **directional tensor
representation**. It is not a theorem that all finite graphs, all positive
methods, or all compatible discretizations fail. In particular it does not
by itself characterize coefficient-dependent effective tensors obtained by
eliminating internal nodes, homogenization, or nonlinear composition.

### 1.1 Self-contained proof, including closure

Normalize \(s_k=d_k/\|d_k\|\), absorbing the squared lengths into the
coefficients. If \(A_n=\sum_k a_{nk}s_ks_k^T\to A\), with \(a_{nk}\geq0\),
then \(\sum_k a_{nk}=\operatorname{tr}A_n\) is bounded. A subsequence of the
finite coefficient vectors therefore converges to \(a_k\geq0\), giving
\(A=\sum_k a_ks_ks_k^T\). Thus \(C_D\) is closed. This also supplies the
closure step often omitted when moving from a PSD obstruction to an SPD one.

For a nonzero vector \(v\), the ray \(\{t vv^T:t\geq0\}\) is an extreme
ray of \(\mathbb S_+^2\). Indeed, if \(vv^T=P+Q\), with \(P,Q\succeq0\),
then for \(w\perp v\),
\(0=w^TPw+w^TQw\); both terms are nonnegative and hence zero. A PSD
matrix with \(w^TPw=0\) has \(Pw=0\), as follows from its spectral
decomposition. Both ranges lie in \(\operatorname{span}v\), so \(P,Q\)
are nonnegative multiples of \(vv^T\). Conversely a positive definite
matrix has two positive spectral summands with independent ranges, so its
ray is not extreme. There are infinitely many rank-one rays indexed by
unoriented lines. A finitely generated cone can have extreme rays only
along its finitely many generators: otherwise a representation with two
nonparallel active summands splits a proposed extreme ray. Thus the PSD
cone is not finitely generated, and \(C_D\ne\mathbb S_+^2\).

More directly, choose a unit \(v\) parallel to none of the \(s_k\). If
\(vv^T=\sum_k a_ks_ks_k^T\), evaluating on a unit \(w\perp v\) forces
every coefficient to vanish, a contradiction. If every SPD matrix belonged
to \(C_D\), then \(vv^T+\epsilon I\in C_D\) for every \(\epsilon>0\).
Closedness would imply \(vv^T\in C_D\). Hence some strictly SPD tensors
are missing.

For the strict coefficient set, adding \(\epsilon\sum_k d_kd_k^T\) to
any element of \(C_D\) shows \(\overline{C_D^{>0}}=C_D\). If the
directions span \(\mathbb R^2\), every strictly positive combination is
SPD, because \(x^TAx=\sum_k c_k(d_k^Tx)^2>0\) for nonzero \(x\).
Spanning two-dimensional vector space is sufficient for this positivity,
but is not sufficient for coverage of three-dimensional symmetric-matrix
space. If they do not span, the cone contains no SPD tensor at all.

Finally, normalize a missing SPD matrix by
\(A\mapsto A/\sqrt{\det A}\). Membership in a cone is invariant under
positive scaling. The obstruction therefore remains on \(\det A=1\).
This proves FD without treating the open SPD set as a closed polyhedral cone.

### 1.2 Explicit strict-SPD witness

Take unit \(v\) missing from the finite direction list, unit \(w\perp v\),
and let \(\delta=\min_k\angle(\operatorname{span}v,\operatorname{span}d_k)>0\)
be the smaller projective angle in \([0,\pi/2]\). Choose \(v\) also so
that \(\delta<\pi/2\). Every \(A\in C_D\) satisfies

\[
 w^TAw\ \geq\ \sin^2\delta\,\operatorname{tr}A.
\]

But \(A_\epsilon=vv^T+\epsilon ww^T\succ0\) violates this inequality
whenever \(0<\epsilon<\tan^2\delta\). Choosing additionally
\(\epsilon<1\) makes \(v\) the principal eigenvector. This is an explicit
separating inequality and an arbitrarily anisotropic SPD counterexample,
not a floating-point failure of NNLS.

## 2. Executable tensor-cone criterion and angular frontier

For \(A=\begin{pmatrix}a&b\\b&d\end{pmatrix}\), \(t=a+d>0\), define

\[
 z(A)=((a-d)/t,\,2b/t),\qquad
 p_k=(\cos2\theta_k,\sin2\theta_k),\quad
 P_D=\operatorname{conv}\{p_k\},
\]

where \(\theta_k=\arg d_k\pmod\pi\). The trace-one PSD section is the
closed unit disk in \(z\); its SPD section is the open disk. Directly
expanding \(s_ks_k^T\) proves

\[
 A\in C_D\quad\Longleftrightarrow\quad z(A)\in P_D.
 \tag{FD-criterion}
\]

These are necessary **and** sufficient conditions for this tensor model.
For a nondegenerate polygon, all prescribed direction coefficients can be
strictly positive iff \(z(A)\in\operatorname{int}P_D\). More generally use
relative interior. To see the nontrivial direction, subtract a sufficiently
small positive multiple of the average of all vertices from an interior
point, renormalize the remaining convex weights, and add the equal positive
weights back. A positive combination of all polygon vertices cannot lie on
a supporting edge. Duplicate directions do not change this conclusion.

Sort unique \(\theta_k\in[0,\pi)\), and include the wraparound gap to
\(\theta_1+\pi\). If every projective gap is at most \(\pi/2\), the
origin lies in \(P_D\), and the largest origin-centered disk it contains
has radius

\[
 R_D=\cos\Delta_{\max},\qquad
 \Delta_{\max}=\max_k(\theta_{k+1}-\theta_k).
 \tag{FD-radius}
\]

Proof: a chord between adjacent doubled-angle unit vectors lies at signed
distance \(\cos(\theta_{k+1}-\theta_k)\) from the origin, with the polygon
on its inner side. Intersect these half-planes. If a gap exceeds \(\pi/2\),
the origin is outside; do not use the formula as a positive coverage radius.

Use the repository Beltrami convention, for \(\mu=r e^{i\phi}\):

\[
 A(\mu)=\frac1{1-r^2}
 \begin{pmatrix}1-2r\cos\phi+r^2&-2r\sin\phi\\
 -2r\sin\phi&1+2r\cos\phi+r^2\end{pmatrix},
 \quad z(A)=-\frac{2r}{1+r^2}(\cos\phi,\sin\phi).
\]

Thus \(\rho=\|z(A)\|=2r/(1+r^2)\), and all orientations at this
\(r\) are covered iff \(\rho\leq R_D\). This is an exact *closed-cone*
condition; strict positivity requires \(\rho<R_D\) when testing all
orientations and the origin is interior. The tensor eigenvalue ratio is
\(\kappa(A)=((1+r)/(1-r))^2\); do not confuse it with map dilatation
\((1+r)/(1-r)\). Worst angular directions are the midpoints of largest
projective gaps in the eigenvector coordinate; translate them to \(\arg\mu\)
using the displayed minus sign and doubled angle.

An NNLS implementation should vectorize tensors as
\((a,\sqrt2 b,d)\), and dyads identically, so Euclidean residual equals
the Frobenius norm. Report relative error
\(\|A-\widehat A\|_F/\|A\|_F\). With a positive floor \(c_k\geq\eta\),
fit \(A-\eta\sum_k d_kd_k^T\) with nonnegative coefficients and add the
floor back. Unit versus physical-length directions give the same zero-floor
cone but different meanings for a fixed floor; record which was used.

### 2.1 Three-coordinate active-subset oracle

The conic Caratheodory bound here is **three**, the dimension of symmetric
2-by-2 matrices, not two, the spatial dimension. Every member of \(C_D\)
has a representation using at most three dyads. For a self-contained proof,
start with any representation with more than three positive active terms.
Its active dyads are linearly dependent, so choose a nonzero dependence
\(\sum_k\alpha_kd_kd_k^T=0\). Taking traces shows the \(\alpha_k\)
have both signs. Subtract \(t\alpha\) from the coefficient vector, with
\(t=\min_{\alpha_k>0}c_k/\alpha_k\). All coefficients stay nonnegative
and at least one active term disappears. Repeat until at most three remain.

For \(M_k=(d_{kx}^2,\sqrt2d_{kx}d_{ky},d_{ky}^2)^T\), the Euclidean
projection onto the closed cone exists. Apply the same reduction to that
projected point. On a minimal independent active support its positive
coefficients satisfy the unconstrained normal equations for that support.
Thus one can enumerate all subsets of sizes 1, 2, and 3, solve each small
least-squares problem, retain nonnegative solutions, and compare their
residuals with the zero solution. This is an exhaustive *closed-cone tensor*
projection oracle (in exact arithmetic), useful against numerical NNLS
failures on a redundant generator matrix. It is not an efficient general
large-graph fitting algorithm or a theorem about shared global edge weights,
full operator matching, or a Schur-complement response.

The sparse oracle deliberately permits zero coefficients. Its result is an
attained minimum for \(c\geq0\); for \(c>0\) the same value may only be
an unattained infimum at a boundary point. A positive floor is a different
optimization problem, reducible by the shift given above.

## 3. Three fixed noncrossing planar meshes

Use square cells of side \(h\), with corners
\(a=(0,0),b=(h,0),c=(h,h),d=(0,h)\) translated by cell indices.
All listed triangles are counterclockwise. Global vertex IDs identify
shared grid corners; edges are unordered vertex-ID pairs extracted from
triangles and deduplicated. Boundary cycle is the perimeter of the original
square grid, with the four corners marked. There are no additional boundary
vertices in these three variants. Here \(n\) is the number of original
grid vertices per side, so the number of cells is \((n-1)^2\).

| mesh | per-cell triangles / added vertices | actual projective edge directions | counts \((V,F)\) | closed tensor cone |
|---|---|---|---|---|
| Standard triangular | \((a,b,c),(a,c,d)\); no additions | \(0,\pi/4,\pi/2\) | \(n^2,2(n-1)^2\) | dyads of \((1,0),(1,1),(0,1)\) |
| Center-split | \(o=(a+c)/2\); \((a,b,o),(b,c,o),(c,d,o),(d,a,o)\) | \(0,\pi/4,\pi/2,3\pi/4\) | \(n^2+(n-1)^2,4(n-1)^2\) | add dyad of \((1,-1)\) |
| Triangle-centroid stellar refinement | Start with standard triangles. For each oriented \((i,j,k)\), add its centroid \(g\), replace by \((i,j,g),(j,k,g),(k,i,g)\) | \(0,\arctan(1/2),\pi/4,\arctan(2),\pi/2,3\pi/4\) | \(n^2+2(n-1)^2,6(n-1)^2\) | add dyads of \((2,1),(1,2),(1,-1)\) to standard cone |

The third row's cell centroids are \((2h/3,h/3)\) and
\((h/3,2h/3)\). Enumerating their three corner displacements gives exactly
the stated six unoriented directions.

Planarity follows constructively: square interiors are disjoint; the standard
diagonal stays in its square; each center fan partitions its convex square;
each centroid lies strictly inside its parent triangle and its spokes
partition that triangle. Intersections occur only at shared vertices or
shared edges. Center-split **must remove** unsplit diagonals: four half-diagonal
edges meet at the actual center vertex. It is not a graph containing two
crossing diagonals without a vertex. A centroid refinement retains parent
edges and adds only interior spokes. Euler's disk check is \(V-E+F=1\),
and \(3F=2E-E_B\), with \(E_B=4(n-1)\); these checks supplement rather
than replace the constructive noncrossing argument.

For the standard mesh's aggregate direction cone, the condition is
\(0\leq b\leq\min(a,d)\). For center-split it is
\(|b|\leq\min(a,d)\). Indeed for unnormalized diagonal vectors,
\(b=c_+-c_-\), and \(s=c_++c_-\) can be chosen in
\([|b|,\min(a,d)]\). This reproduces the existing four-direction algebra
without interpreting it as a new DEC construction.

The standard cone has \(\Delta_{\max}=\pi/2\), hence no nonzero
orientation-uniform Beltrami radius. Its \(A=I\) lies on the cone boundary,
requiring the \((1,1)\) coefficient to be zero. A strictly positive diagonal
floor necessarily introduces tensor bias already at \(\mu=0\).
Center-split has \(\Delta_{\max}=\pi/4\), giving
\(R_D=1/\sqrt2\) and \(r_{\max}=\sqrt2-1\). The stellar mesh still has
45-degree gaps from 90 to 135 and from 135 to 180 degrees. Its worst
orientation-uniform threshold is therefore **the same**, although its cone
strictly contains the center-split cone and improves selected orientations.
This is a useful negative prediction, not an implementation defect.

### 3.1 Necessary separation between aggregate and local coverage

The table describes the union of edge directions available in one periodic
cell. It is a tensor-direction cone, not a promise that all those directions
occur in every vertex star or every small triangle. At a center-split cell
center only the two diagonal directions occur; at a stellar centroid only
its three spokes occur. For a local fit use the actual local support.

For affine \(u(x)=p^Tx\), the graph energy is
\(\sum_e c_e(u_j-u_i)^2=p^T(\sum_e c_e d_ed_e^T)p\).
Consequently an aggregate affine-energy tensor is exactly in the cone of
actual physical edge displacements, up to a specified area normalization.
This explains the cone model, but only tests affine functions. Matching
these three tensor components does not match the full operator on all nodal
functions, enforce equilibrium of an affine function at inserted nodes,
or reproduce spatially varying facewise tensors.

For each triangle the stronger identity for all local nodal values is

\[
 |T|\,A_T=\sum_{e\subset T}c_{T,e}\,d_ed_e^T.
\]

Its three coefficients are determined by its local P1 stiffness:
\(c_{T,ij}=-K^T_{ij}\). Allowing nonnegative local coefficients requires
membership in **that triangle's three-direction cone**. Assembling these
local coefficients gives \(c_{ij}=\sum_{T\supset ij}c_{T,ij}\).
Nonnegative local coefficients are sufficient, not necessary, for
nonnegative assembled edge coefficients: negative and positive local terms
can cancel. Conversely, independently fitting a tensor at every vertex
does not automatically give one shared symmetric coefficient per edge.

Therefore coverage experiments must distinguish (a) aggregate cone NNLS,
(b) actual local/assembled operator mismatch, and (c) decoded map/Beltrami
error. Only (a) is characterized by FD-criterion. The phase's required
operator and resulting-map measurements cannot be replaced by (a).

## 4. What escapes the obstruction, and what remains unproved

An individual \(A\succ0\) always admits a two-direction spectral
decomposition with positive eigenvalues; the eigenvectors depend on \(A\).
Thus adaptive directions escape the fixed-D premise. Building a globally
conforming noncrossing mesh or dual complex from them is a separate problem.
Changing connectivity among a fixed finite vertex set merely changes a
finite direction list; the impossibility still holds if the union of all
allowed directions is finite. Adaptive vertex locations or increasing
direction sets under refinement need separate analysis.

### 4.1 Negative result: a full Whitney Hodge is still P1 on exact gradients

A full edge-form Hodge mass matrix
\((H_A)_{ee'}=\sum_T\int_T w_e^T A_T w_{e'}\), for consistently oriented
Whitney one-forms \(w_e\), is not a positive scalar directional sum. For
SPD \(A_T\), it is SPD on independent global edge degrees of freedom.
The identity \(\sum_e(B_0u)_e w_e=\nabla u_h\) gives
\(B_0^TH_AB_0=K_{P1,A}\) and exact P1 energy consistency. These are
direct algebraic consequences of the basis and bilinear form, not evidence
of stronger conjugacy than P1. In particular, simply factoring the same P1
matrix through a full Hodge does not improve its primal solution or create
a continuous normal-flux trace. A flux space, dual complex, and boundary
stream integration must be specified and measured separately.

With RT0, edge normal-flux DOFs and one cellwise pressure DOF give
\(H(\mathrm{div})\) conformity and cell balance. This supports a different
flux claim from Whitney edge cochain Kirchhoff balance. Neither claim proves
a nonfolding spatial map. A general SPD reduced system need not be an
M-matrix, and a discrete maximum principle is not itself a two-coordinate
global homeomorphism theorem. Positive Tutte weights require the applicable
planar disk/connectivity and convex boundary hypotheses; for collinear square
side vertices and boundary ears, use the already checked square-boundary
version rather than silently invoking a strictly convex-polygon theorem.

## 5. Targeted primary-source table (PLAN Section 33)

Search scope: named author/title queries, arXiv author preprints, author or
publisher pages; checked 2026-09-22. This is an AI-assisted targeted audit,
not a systematic review or a priority search. “Unknowns” means mathematical
solution DOFs; explicit retrieval limitations are stated below. A dash under
learning/backprop means it is not a claim of that work, not impossibility.

| work / primary source | unknowns | exact structure | learned quantity | linear system / solve | backprop | guarantee, with hypotheses | relevance | does NOT guarantee |
|---|---|---|---|---|---|---|---|---|
| Hersonsky, mixed Dirichlet–Neumann graph uniformization (2011), [paper](https://arxiv.org/pdf/1006.0026), Theorem 0.6 and Section 2; [publisher](https://doi.org/10.1016/j.difgeo.2011.03.003) | Vertex harmonic potential; edge currents and rectangle data | Kirchhoff balance; rectangle area/graph energy correspondence | None | Positive conductance mixed graph Laplacian, then tiling construction | Not a neural result | Quadrilateral graph with marked cyclic boundary arcs and the paper's conductance/boundary conventions gives an energy-preserving rectangle tiling | Exact discrete electrical reference; boundary corners and zero-current edges need faithful treatment | A tiling assigning rectangles to graph edges is not automatically a nondegenerate PL map on our original triangles, nor exact arbitrary anisotropic tensor recovery |
| Mercat, [Discrete Riemann Surfaces and the Ising Model](https://math.univ-lyon1.fr/~mercat/articles/CMP.pdf) (2001), Sections 2–3; [author preprint](https://arxiv.org/abs/0909.3600) | Cochains/forms on primal plus dual cellular complex; periods | Discrete Cauchy–Riemann relation, Hodge star, harmonic theory on the doubled complex | None | Weighted discrete Laplacian / period constraints | Not studied as learned layer | Discrete identities; continuum limit under specified critical-embedding hypotheses | Makes dual complex and discrete conformal structure explicit | Arbitrary fixed triangulations with arbitrary facewise \(\mu\) are not covered merely by calling an operator Hodge; no generic injective PL decoder |
| Desbrun–Hirani–Leok–Marsden, [Discrete Exterior Calculus](https://arxiv.org/abs/math/0508341) (2005), [Caltech record](https://authors.library.caltech.edu/records/9zse2-91c10) | Simplicial cochains and dual cochains | Coboundary squared zero; discrete Stokes; circumcentric primal/dual calculus | None | Incidence/Hodge systems; problem-dependent constraints | Not a neural method | Algebraic identities are topological; metric positivity requires appropriate geometric/Hodge conditions | Incidence orientation and dual accounting for prototype | Chain identity alone is not exactness on nontrivial topology, positive circumcentric Hodge on arbitrary meshes, or bijection |
| Raviart–Thomas, [A mixed finite element method for 2-nd order elliptic problems](https://link.springer.com/chapter/10.1007/BFb0064470) (1977); RT0 case | Edge integrated normal flux, cellwise scalar potential | Normal continuity and cellwise divergence balance in the compatible pair | None | Mixed saddle system with flux mass and divergence blocks; boundary constraints | Not in original; a fixed nonsingular discretization admits an adjoint by general solve calculus | Mixed stability/error results under their discretization and ellipticity assumptions | Concrete conservative flux reference, unlike independently solved nodal conjugate | RT0 flux is not in general \(A\nabla u_{P1}\); conservation gives no positive Jacobian or exact prescribed facewise Beltrami field |
| Arnold–Falk–Winther, [Finite element exterior calculus: from Hodge theory to numerical stability](https://arxiv.org/abs/0906.4325) (2010); [2006 author paper](https://sites.math.rutgers.edu/~falk/papers/acta.pdf) | Compatible finite element differential forms, including mixed variables | Subcomplex, commuting bounded cochain projections, treatment of harmonic subspaces | None | Mixed Hodge Laplacian, appropriate gauges/boundary constraints | Not a learned solver contribution | Stability/convergence requires both subcomplex and bounded projection hypotheses | Explains which structure a learnable discretization must retain | Merely imposing \(B_1B_0=0\) is not the full stability theorem; no spatial embedding guarantee |
| Trask–Huang–Hu, [Enforcing exact physics in scientific machine learning: a data-driven exterior calculus on graphs](https://arxiv.org/pdf/2012.11799) (2022 journal work), Sections 3–4 / Algorithm 1 | Coarse graph cochain PDE state | Weighted compatible operators, conservation and exact-sequence structure | Metric/operator weights and nonlinear elliptic perturbation | Nonlinear forward Newton solve plus linear transpose-Jacobian adjoint | PDE-constrained equality optimization, explicit adjoint | Well-posedness for the analyzed bounded nonlinear perturbation class; algebraic conservation to solve tolerance | Direct prior art for learned metric/Hodge plus implicit solver | No unrestricted learned-operator well-posedness, planar positive conductance representation of every SPD tensor, or image deformation bijection |
| Kinch et al., [Structure-Preserving Digital Twins via Conditional Neural Whitney Forms](https://arxiv.org/html/2508.06981v1) (2025 preprint), Sections 2.1–3, Theorem 2.1 | Reduced Whitney state and nonlinear flux | Partition of unity, induced Whitney complex, antisymmetric flux cancellation | Conditional reduced FE basis and nonlinear conservation law | Nonlinear reduced forward system, Newton/backtracking, KKT adjoint | AD for Jacobians and adjoint; gradient optimization | Uniqueness under diffusion-dominance/Lipschitz hypotheses; Section 3 distinguishes sufficient theoretical bounds from empirical successful solves | Close prior art for conditioned compatible neural layers | The paper's empirical training success does not certify theorem constants at every step; learned basis is not a bijective deformation of a fixed mesh |
| Shaffer et al., [Structure-Preserving Learning Improves Geometry Generalization in Neural PDEs (Geo-NeW)](https://arxiv.org/html/2602.02788v1) (2026 preprint), Sections 3.3–3.4 and Appendix B | Reduced PDE coefficients on input geometry | FEEC-compatible basis and conservative flux architecture | Geometry-conditioned reduced basis, nonlinear flux, SPD reduced Hodge \(\mathcal H\mathcal H^T+\epsilon I\) | Batched nonlinear Newton forward; transpose Jacobian solve | Explicit implicit adjoint in Section 3.3 | Stated local invertibility/uniqueness under contraction bound involving Lipschitz and inverse-operator norms | Direct recent overlap: learning full Hodge and differentiating a compatible solve is already prior art | SPD is not M-matrix; conservation/generalization experiments do not prove PL-homeomorphism or our QC accuracy/frontier |
| Huang, [Discrete maximum principle and a Delaunay-type mesh condition for linear FEM approximations of 2D anisotropic diffusion](https://arxiv.org/pdf/1008.0562), Lemma 4.3, Theorem 4.1 | P1 nodal scalar potential | Assembled stiffness off-diagonal sign under metric edge condition | None | Anisotropic FEM diffusion system with specified boundary data | Not a neural claim | DMP under stated mesh/boundary hypotheses; assembled condition weaker than elementwise nonobtuse condition | Exact test for when FEM coefficients yield a positive-network interpretation | SPD tensor alone, local-angle necessity, arbitrary mixed-side trace ordering, and two-coordinate bijection are not proved |
| Binder–Pechersky, [Orthodiagonal Maps, Tilings of Rectangles, and their Convergence to Conformal Maps](https://arxiv.org/html/2407.20851v1) (2024 preprint), Sections 2.3–2.4 and convergence theorem | Primal/dual potentials and rectangle-tiling coordinates | Electrical tiling and orthodiagonal discrete holomorphy | None | Primal/dual harmonic network solves with geometric conductances | Not a neural contribution | Uniform convergence on compact subsets for finer orthodiagonal approximations of marked simply connected domains under the paper's approximation conditions | Shows a credible continuum limit for a specified geometric discrete class | Arbitrary positive weights or anisotropic full Hodge on our three meshes do not meet those conditions automatically; limit convergence is not fixed-mesh PL injectivity |

Source limitations: the original RT chapter's publisher metadata was verified,
but its complete chapter text was not available in the retrieved page. The
RT0 DOF/system characterization above is the standard special case supported
also by the author FEEC text, not a claimed fresh audit of every 1977 proof.
Neural and convergence preprints are identified as preprints; no benchmark
speedup or unrestricted guarantee is imported from their abstracts.
No exhaustive absence-of-prior-art or “first neural Hodge” claim follows.

## 6. Reconciliation with Phase III evidence and code

Read as reconnaissance, not present-route closure:

- `docs/research_phase3/02_discrete_conjugacy.md` and
  `src/qcopt/forward/discrete_conjugacy.py`: the local compatible-P1 theorem
  is sufficient. The stream helper averages incident face predictions,
  integrates a spanning tree, then separately checks edge inconsistency and
  all remaining cycle residuals. It does not supply a general compatible
  mixed reference. Small averaged edge/cycle residuals cannot replace the
  separately reported original face-flux discrepancy.
- `docs/research_phase3/03_electrical_primal_dual.md` and
  `src/qcopt/forward/electrical_rectangle.py`: the corrected reference uses
  \(n_y+1\) dual levels / \(n_y\) strips. Its weighted solver explicitly
  rejects horizontally nonconstant row current. That is a restriction of
  this strip implementation, not a necessary hypothesis of general planar
  rectangle-tiling theory. Its energy/flux/area identities remain useful
  tests after the dual boundary bookkeeping is changed.
- `docs/research_phase3/04_mmatrix_delaunay.md` and
  `src/qcopt/forward/anisotropic_hodge.py`: the four-direction feasibility
  interval \([|A_{01}|,\min(A_{00},A_{11})]\) is an algebraic cone test.
  It has no primal/dual assembly or general Hodge mass matrix. Prior wide
  two-ring flip results warn about lost planarity, but do not rule out the
  explicitly noncrossing refinements in Section 3.

For anisotropic P1 edge \(ij\), the precise assembled sign test is

\[
 K_{ij}=-\tfrac12\sum_{T\supset ij}\sqrt{\det A_T}\,
 \cot\theta^{A_T^{-1}}_{T,ij}\leq0.
\]

With one constant metric on both incident triangles, this is equivalent to
the opposite metric-angle sum at most \(\pi\). With differing metrics,
use the weighted sum. A local nonobtuse condition is sufficient and is not
necessary for the assembled sign. This is consistent with the original
Huang result, rather than a new monotonicity theorem.

## 7. Builder self-check and independent-check handoff

Tiny question: does an independent support-enumeration projection reproduce
the exact worst-orientation threshold? A one-off Python/NumPy calculation
enumerated supports of sizes 1–3 using `numpy.linalg.lstsq`, retaining
coefficients at least `-1e-10`, for 144 evenly spaced arguments on
\([0,2\pi)\) and radii \(0,0.2,0.4,\sqrt2-1,0.6,0.8,0.9\).
The coefficient tolerance is a floating-point feasibility allowance only;
the proof above uses exact nonnegativity. This is a builder spot-check, not
independent review, a local mesh-operator experiment, or a raw-data benchmark.

| direction set | maximum relative Frobenius error at \(r=0.4\) | at \(r=\sqrt2-1\) | at \(r=0.9\) |
|---|---:|---:|---:|
| standard triangular | 0.5677329558 | 0.5773502692 | 0.7051453312 |
| center-split | \(5.77\times10^{-16}\) | \(4.44\times10^{-16}\) | 0.1663711505 |
| stellar six directions | \(3.00\times10^{-16}\) | \(3.20\times10^{-16}\) | 0.1663711505 |

The initial SciPy `nnls` cross-check agreed where it returned, but raised
`Maximum number of iterations reached` for a six-direction case at \(r=0.9\).
The exhaustive three-coordinate oracle completed the entire sweep. This is
why solver nonconvergence must be distinguished from an infeasible tensor.

FD proof covers rank-deficient PSD boundary, open SPD interior, closed-cone
limit, strict coefficient closure, determinant-one normalization, and an
explicit separating inequality. It does not infer an embedding from SPD.
The trace polygon criterion is necessary and sufficient for the specified
directional tensor model only. The mesh table includes exact coordinates,
incidences, counts, and a noncrossing construction. The anisotropy threshold
uses the squared tensor condition number and the repository's minus sign in
the Beltrami tensor. Numerical coverage and learned-map results belong in
the prototype/neural documents, not in this proof.

Independent checker should challenge: (i) the missing-SPD argument and strict
coefficient boundary cases; (ii) directions extracted from each actual mesh;
(iii) distinguishing the aggregate cone from local face/node supports;
(iv) whether a Whitney prototype is merely a refactorization of P1;
(v) flux DOFs, boundary dual cells, and what exactness/conjugacy is actually
measured; (vi) source hypotheses and conditional neural solvability claims.
Findings and adjudication belong in `12_primal_dual_checker.md`.
