# Route I: differentiable Tutte maps with reusable and matrix-free solvers

## 1. Scope and status

This chapter specifies the mathematical map, its derivatives, and the current Route I algorithms. It is a methods description, not a completed performance comparison. Experimental results and route-level conclusions remain to be supplied after the planned accuracy and engineering comparisons. The implementation history, including failed numerical configurations, is recorded in the [Phase V worklog](WORKLOG.md). The [official TutteNet paper/code audit](01_tuttenet_code_audit.md) establishes the prior implementation against which engineering changes must be distinguished.

There are three different statements throughout this chapter:

1. **Exact mathematical guarantee:** under stated disk, boundary, and positive-weight hypotheses, the exact piecewise-affine map is a homeomorphism.
2. **Implemented acceptance rule:** a floating-point solve must pass numerical and geometric checks before it is returned. These checks reject detected failures; they are not an exact-predicate proof of every target geometry.
3. **Measured accuracy or efficiency:** these require comparisons against independent solutions and derivatives. Neither a theorem about the exact map nor a small solver residual is such a comparison.

The public decoder maps latent variables and a boundary parameterization to control-vertex coordinates, then evaluates their P1 interpolant at query points. Here “P1” means continuous and affine on each source triangle. It does not refer to a neural interpolation scheme, a bilinear image grid, or an exact solution of a prescribed Beltrami equation. All custom linear-solve backward implementations described below support first derivatives; differentiating through their implicit backward a second time is not a supported contract.

## 2. Source triangulation, indexing, and topology

### 2.1 Definitions

Let \(\mathcal T=(V,E,F)\) be a finite oriented simplicial complex whose realization \(|\mathcal T|\) is a closed topological disk. Vertices have fixed source positions \(x_i\in\mathbb R^2\). Every triangle \(t=(i,j,k)\in F\) is ordered counterclockwise and has positive source area. Triangles meet only in common simplices. The source realization is thus a planar triangulation of a simple polygonal disk, not just an arbitrary graph accompanied by a face list.

Write \(n=|V|\), \(n_I=|I|\), and \(n_B=|B|\), where \(I\) is the interior vertex set and \(B=V\setminus I\) is the boundary vertex set. The boundary is one cyclically ordered list

\[
\ell=(\ell_0,\ldots,\ell_{n_B-1}),\qquad
\partial\mathcal T=\bigcup_k[\ell_k,\ell_{k+1\bmod n_B}].
\]

Boundary tensors are stored in this loop order. A boundary tensor row is **not** necessarily a global vertex ID. Interior rows also have a separate index map. The solver scatters boundary/interior outputs back into global vertex order; its adjoint gathers using the same permutations.

The neighbor set \(N(i)=\{j:\{i,j\}\in E\}\) includes all mesh-edge neighbors, both interior and boundary. A *dividing edge* is an edge in \(E\setminus\partial\mathcal T\) with both endpoints in \(B\). It is not a boundary edge, even though its endpoints are boundary vertices. This distinction matters when target boundary vertices are collinear.

### 2.2 One-time source validation

[`DirectedTutteSystem.from_mesh`](../../src/qcopt/forward/tutte_directed_implicit.py) provides the shared source setup. It rejects repeated unordered triangles, unused vertices, invalid or incoherently oriented edge incidences, disconnected face complexes, invalid vertex links, multiple boundary cycles, and Euler characteristic different from one. An interior vertex link must be a cycle; a boundary vertex link must be a path. Euler characteristic alone would not exclude a pinched or nonmanifold complex.

The source identity realization is checked independently of any learned solution: each represented source triangle has positive orientation, and the source boundary is a simple counterclockwise polygon. These source predicates integerize the finite binary floating-point coordinates and evaluate the relevant orientation/intersection signs with integers. Their exactness concerns those represented inputs, not unknown real coordinates before rounding. Combined with the disk topology and the PL global-inversion statement in Section 3, this validates the source embedding without presuming that the subsequent Tutte solve succeeds. The system caches the interior/boundary permutations, neighbor tables, faces, and dividing edges.

For the principal implementation family, `structured_rectangle(nx, ny)` uses **cell counts**, \(n_x,n_y\geq1\). Each cell is split from its lower-left to upper-right corner into \((v_{00},v_{10},v_{11})\) and \((v_{00},v_{11},v_{01})\). Consequently,

\[
n=(n_x+1)(n_y+1),\quad |F|=2n_xn_y,\quad
|E|=3n_xn_y+n_x+n_y,
\]
\[
n_B=2(n_x+n_y),\qquad n_I=(n_x-1)(n_y-1).
\]

An \(N\)-by-\(N\) **control-vertex** grid therefore uses \(n_x=n_y=N-1\), has \(N^2\) vertices, and has \((N-2)^2\) interior rows. Thin grids with \(n_x=1\) or \(n_y=1\) are valid disks and can have no interior unknowns. They still require boundary and face checks. Structured rectangles can have dividing edges; their admissibility follows from where those edges' endpoints are mapped, not from an assertion that dividing edges are absent.

This mesh is different from the official TutteNet center-split mesh. In the audited official construction, \(N=11\) gives 221 vertices, 181 interior vertices, 40 boundary vertices, and 400 triangles. Our two-triangle-per-cell \(N=11\) mesh gives 121, 81, 40, and 200 respectively. Equal labels \(N\) are therefore not equal linear-system sizes. See the [official audit](01_tuttenet_code_audit.md).

## 3. What guarantees global injectivity?

### 3.1 The represented P1 map

Given target vertices \(Y=(y_i)_{i\in V}\), define

\[
f_Y(x)=\sum_{a=0}^2\beta_a(x)y_{i_a},
\qquad x\in t=(i_0,i_1,i_2),
\]

where \(\beta_a\) are the source barycentric coordinates. Its constant triangle Jacobian is

\[
J_t=[y_{i_1}-y_{i_0}\;y_{i_2}-y_{i_0}]
       [x_{i_1}-x_{i_0}\;x_{i_2}-x_{i_0}]^{-1}.
\]

Positive target signed areas are equivalent to \(\det J_t>0\) because source orientations are positive. This excludes collapse or reversal in triangle interiors. Alone, it is not a global injectivity statement: the source topology and the boundary behavior are indispensable.

### 3.2 Positive convex-combination theorem

Prescribe an orientation-preserving boundary homeomorphism onto the boundary of a convex polygonal region \(\Omega\). At every interior vertex require a strictly positive convex combination of **all** its mesh neighbors:

\[
y_i=\sum_{j\in N(i)}p_{ij}y_j,\qquad
p_{ij}>0,\qquad\sum_{j\in N(i)}p_{ij}=1. \tag{1}
\]

For a triangulated disk, Floater's weak-convex extension states that this map is injective precisely when no dividing edge is mapped entirely into \(\partial\Omega\). Symmetry of the weights is not required. If all boundary vertices form a strictly convex polygon, a dividing chord cannot lie in its boundary. A rectangle with subdivided sides is only weakly convex, so the dividing-edge condition cannot be omitted. [Floater, *One-to-one piecewise linear mappings over triangulations*, Theorem 6.1 and Corollary 6.2](https://doi.org/10.1090/S0025-5718-02-01466-7); the [indexed primary-paper copy](https://citeseerx.ist.psu.edu/document?doi=bc526c8c7d3821bd03aa0035e0f91aab31b8be94&repid=rep1&type=pdf) supplies the same theorem.

For our structured rectangle, assigning each source side to the corresponding target side, preserving strict side order, and matching the four corners excludes this obstruction: no nonboundary mesh edge has both endpoints on a single source side, and the target assignment preserves this incidence. This also covers the thin-grid cases. Merely checking that a target polygon is convex, without this source-to-target incidence condition, would not cover a generic triangulation.

These are triangulated-disk hypotheses, not a claim that arbitrary positive-weight graphs embed. A separate global three-connectivity test is not substituted for the stated disk/dividing-edge theorem. Boundary reachability suffices for uniqueness of the linear solution (Section 5), but by itself does not imply its injectivity.

### 3.3 A posteriori PL global inversion

Independently of harmonicity, an orientation-preserving, nondegenerate P1 map of a triangulated disk, with an injective orientation-preserving boundary map, is a homeomorphism onto the bounded target domain. In degree terms, the simple boundary has degree one inside and zero outside; every regular interior preimage contributes positively. The disk and nondegeneracy assumptions are essential to extending this conclusion across the mesh skeleton. This is the relevant PL global-inversion result, not an inference from local determinants alone. [Lipman, *Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing*](https://arxiv.org/abs/1310.0955).

This exact theorem motivates checking **every original triangle and the boundary**, rather than testing a few dense pixels. It also explains why an approximate harmonic solution can still define a valid map even when it is not the exact solution of (1).

There is an important implementation qualification. Target boundary and area checks currently use floating-point predicates, not the exact integer predicates used for the source realization. A passed target check is consequently a numerical certificate/screen under its stated arithmetic and tolerances, not a formal exact-real proof for all nearly degenerate inputs. Hereafter “accepted map” means a returned map that passed these checks. No post-hoc fold repair is applied.

## 4. Rectangle boundary and learnable aspect ratio

[`StructuredRectangleBoundary`](../../src/qcopt/neural_bijection/tutte/boundary.py) uses four independent side distributions. For side \(s\) with \(m_s\) source segments and logits \(a_{sr}\), set

\[
q_{sr}=\frac{e^{a_{sr}}}{\sum_{v=1}^{m_s}e^{a_{sv}}},\qquad
\tau_{sk}=\sum_{r=1}^kq_{sr}\quad(1\leq k<m_s),
\quad\tau_{s0}=0,\quad\tau_{s,m_s}=1. \tag{2}
\]

The four counterclockwise side maps, with \(0\leq\tau\leq1\), are

\[
(\tau,0),\quad (1,H\tau),\quad(1-\tau,H),\quad(0,H(1-\tau)). \tag{3}
\]

Shared corners are assigned their exact represented endpoint values. Width is fixed to one and

\[
H=H_{\min}+\operatorname{softplus}(m),\qquad H_{\min}>0. \tag{4}
\]

Thus \(H\) is the target rectangle's height/width aspect parameter. If called a “modulus,” this is the explicitly chosen convention; it is not a recovered extremal length, an electrical-energy identity, or proof of exact Beltrami reproduction. Keeping width fixed removes an otherwise redundant global scale. Side segment counts follow \(n_x,n_y\) separately; nonsquare meshes do not reuse a single side length.

In exact arithmetic, (2) is strictly increasing. In floating point, positive \(q\) can nevertheless have cumulative sums that round to identical coordinates. The code checks the realized side order as well as finite positive probabilities and finite positive height. A latent value being finite does not guarantee a representable valid boundary.

Boundary gradients pass through (2)–(4). For an interior cumulative point,

\[
\frac{\partial\tau_{sk}}{\partial a_{sr}}
=q_{sr}\bigl(\mathbf1_{r\leq k}-\tau_{sk}\bigr),
\qquad \frac{dH}{dm}=\operatorname{sigmoid}(m). \tag{5}
\]

Pinned endpoints have zero side-logit derivative. If \(g_b\) is the boundary-coordinate cotangent from Section 8, then \(g_m=\operatorname{sigmoid}(m)\sum_b g_b\cdot\partial C_b/\partial H\), and side-logit cotangents follow from (5) and (3). This includes the change in the interior extension caused by the boundary, not just the boundary vertices' direct loss.

Square composition calls `at_height(..., 1.0)` instead of converting an inverse-softplus value back through softplus. The latter need not round to exactly one across Torch builds. All current composition layers, including the last, have fixed represented height one; learnable height is a single-layer capability, not a composition capability.

## 5. Directed row-softmax system

For every interior row \(i\), the latent tensor contains one slot per neighbor and padding up to the maximum degree \(D\). Let

\[
p_{ij}=\frac{\exp z_{ij}}{\sum_{k\in N(i)}\exp z_{ik}},\quad j\in N(i). \tag{6}
\]

Padding is masked out before softmax; its probability and derivative are zero. The positive-weight hypothesis applies only to supported slots. Implementations check the actual normalized supported probabilities, because exponentiation or normalization can underflow even for finite logits.

Write \(C\in\mathbb R^{n_B\times2}\) for the ordered boundary coordinates and \(X\in\mathbb R^{n_I\times2}\) for the interior coordinates. Split the neighbor operator into interior and boundary columns, \(Q=P_{II}\) and \(R=P_{IB}\). Then

\[
A=I-Q,\qquad AX=RC,\qquad Y_I=X,\quad Y_B=C. \tag{7}
\]

The two coordinate columns share the same matrix. Generally \(A\ne A^T\); neither undirected source connectivity nor positive probabilities makes this directed operator symmetric. There is no general symmetric Dirichlet-energy minimization interpretation of (7).

### 5.1 Existence and uniqueness in exact arithmetic

\(Q\geq0\) is substochastic. Full positive support on a connected disk gives each interior vertex a finite positive-weight path to the boundary. In the absorbing-chain interpretation, some finite power loses a positive amount of row mass from every interior starting vertex. Since there are finitely many rows, there exist \(k\) and \(\delta>0\) with \(\|Q^k\|_\infty\leq1-\delta\). Therefore \(\rho(Q)<1\) and

\[
A^{-1}=\sum_{r=0}^{\infty}Q^r\geq0. \tag{8}
\]

Thus \(A\) is a nonsingular M-matrix and (7) has a unique solution. \(Q\) itself need not be irreducible; disconnected interior components can each reach the same prescribed boundary. If \(n_I=0\), (7) is empty and the output consists entirely of the correctly permuted boundary coordinates.

This argument supplies no uniform condition-number or iteration-count bound over all finite logits. Very small positive probabilities can make boundary absorption extremely slow. Floating-point row sums are also not exactly one merely because the operation is named softmax.

### 5.2 CPU reference versus improved direct solver

The [`LegacyReferenceTutteLayer`](../../src/qcopt/neural_bijection/tutte/reference.py), exposed as backend `reference`, calls the Phase III [`directed_tutte_embedding_torch_implicit`](../../src/qcopt/forward/tutte_directed_implicit.py) separately for each batch sample. For nonempty interiors, its forward builds \(A\), calls SciPy `factorized(A)` and `factorized(A.T)`, solves the primal with the former, and retains the latter for the adjoint. It therefore incurs **two factorization calls during forward per sample**, not an extra factorization during backward. Its normalization, assembly, and solve use NumPy float64 even for float32 public inputs. The adapter checks normalized support and returned-dtype face orientation, and protects saved boundary state from caller aliases.

The improved [`DirectTutteLayer`](../../src/qcopt/neural_bijection/tutte/direct.py), backend `direct`, instead calls `splu(A)` once per sample, solves the two primal columns together, and retains that factor. Backward calls the *same factor* with `trans='T'`. This reuses the numeric factorization for the transpose solve; it does not claim that solving \(A\lambda=g\) is equivalent to solving \(A^T\lambda=g\). Softmax is evaluated in the input Torch dtype, after which assembly and sparse solves use float64. Consequently float32 `reference` and `direct` need not have bit-identical probabilities.

Both adapters are CPU-only and execute samples sequentially. Factors belong to an individual autograd graph; they are not reused across changed parameters or different forward calls. No cross-forward symbolic-factorization cache is implemented. Boundary-only cases perform no factorization. The improved direct layer retains a separate primal copy, so mutating the returned tensor cannot silently change its backward's saved solution.

These are our baselines, not a reproduction of official KLU behavior. The audited `torch_sparse_solve` path releases its forward factor and performs another factorization for the backward transpose system. The official learning path also ties reverse-edge weights whereas its fitting path permits independent directed weights. Those distinctions, and the official center-split mesh, must be preserved in any comparison with TutteNet. [Official source audit](01_tuttenet_code_audit.md).

## 6. Directed matrix-free primal and adjoint

### 6.1 Operators without sparse assembly

[`MatrixFreeDirectedTutteLayer`](../../src/qcopt/neural_bijection/tutte/iterative.py), backend `directed_iterative`, represents (7) by fixed neighbor/index buffers. For a vector or multiple-column tensor \(v\),

\[
(Av)_i=v_i-\sum_{j\in N(i)\cap I}p_{ij}v_j. \tag{9}
\]

A gather and weighted reduction compute the second term. Its transpose is

\[
(A^T\lambda)_j=\lambda_j-
\sum_{i\in I:\,j\in N(i)}p_{ij}\lambda_i,\qquad j\in I, \tag{10}
\]

implemented by scatter-add. The boundary right-hand side is \((RC)_i=\sum_{b\in N(i)\cap B}p_{ib}C_b\). In particular, the adjoint must use incoming directed contributions in (10), not the outgoing-neighbor reduction in (9). Neither CSR/CSC assembly nor sparse factorization occurs in this backend.

### 6.2 BiCGStab and acceptance

BiCGStab is applied independently to every batch/RHS column, vectorized in tensors of shape \((b,n_I,r)\), with \(r=2\) for coordinates. Each column has its own active mask, recurrence scalars, and convergence count. Inactive columns must not divide by their zero recurrence denominators or drive extra updates in active columns.

For completeness, the ordinary exact-arithmetic recurrence for one active column starts with \(x_0=0\), \(r_0=b-Ax_0\), and fixed shadow residual \(\widetilde r=r_0\). With the usual initialization of search vectors/scalars,

\[
\rho_k=\langle\widetilde r,r_{k-1}\rangle,\quad
\beta_k=(\rho_k/\rho_{k-1})(\alpha_{k-1}/\omega_{k-1}),
\]
\[
p_k=r_{k-1}+\beta_k(p_{k-1}-\omega_{k-1}v_{k-1}),\quad
v_k=Ap_k,\quad\alpha_k=\rho_k/\langle\widetilde r,v_k\rangle,
\]
\[
s_k=r_{k-1}-\alpha_kv_k,\quad t_k=As_k,\quad
\omega_k=\langle t_k,s_k\rangle/\langle t_k,t_k\rangle,
\]
\[
x_k=x_{k-1}+\alpha_kp_k+\omega_ks_k. \tag{11}
\]

The implementation handles half-step convergence, zero/breakdown denominators, and restarts where needed; (11) alone is not its floating-point acceptance rule. It explicitly recomputes the residual \(b-Ax\) each iteration and accepts each column only when

\[
\|b-Ax\|_2\leq\mathrm{atol}+\mathrm{rtol}\|b\|_2. \tag{12}
\]

Norms use maximum-component scaling. Recurrence dot products use scaled inputs and reject unsafe nonfinite or subnormal rescaling/products at every iteration. These guards intentionally sacrifice availability at extreme scales instead of silently accepting a residual norm that underflowed to zero. A half-step whose recursive residual appears converged is still checked against the represented true residual.

Current defaults are `rtol=1e-5` for float32, `rtol=1e-10` for float64, `atol=0`, and 500 primary iterations. They are numerical settings, not prescribed map or VJP error bounds. The worklog records why a stricter CUDA float32 setting was not uniformly available.

### 6.3 Restricted stationary fallback

Only the layer's **CUDA float32** solve wrapper catches BiCGStab breakdown/nonconvergence and attempts a stationary fallback. CPU and float64 do not silently switch algorithms. Starting again from zero, the fallback uses

\[
x^{k+1}=x^k+(b-Ax^k)=Qx^k+b. \tag{13}
\]

For the adjoint it uses \(A^T\), hence iteration matrix \(Q^T\). Equation (8) gives convergence of both exact-arithmetic iterations because \(\rho(Q^T)=\rho(Q)<1\). It does **not** say that \(Q^T\) is an infinity-norm contraction, that transient growth is impossible, or that the convergence rate is uniform.

The fallback recomputes the true residual on every update, masks independently converged columns, and allows `20 * max_iter` iterations. This multiplier is a heuristic budget, not a theorem. Rounded row sums, device reduction order, and finite arithmetic do not inherit (8) exactly. Finite components are insufficient if their norm or the tolerance overflows; these quantities are separately checked. Exhausted budgets, nonfinite states/norms, or a final nonconverged report raise an error.

The final diagnostic records `method="stationary_fallback"` and the primary failure. Its iteration count is the fallback count, not the total failed-primary-plus-fallback work. Elapsed solver time includes both attempts. A logical solve still counts once (Section 11); attempt/matvec accounting is a separate metric. The fallback is not a general Richardson solver justified for arbitrary user operators.

### 6.4 State and device qualifications

The custom implicit function saves the probabilities and an independent primal copy, plus topology/settings, not the sequence of Krylov iterates. Its operator references are captured for that invocation. Detached CPU diagnostic snapshots do not participate in forward/backward computation. `last_diagnostics` exposes the latest invocation; retaining that handle is necessary to inspect an older graph's adjoint after later forwards.

Linear algebra and its buffers can reside on CUDA, but the complete layer is not zero-transfer: boundary and returned-face checks copy geometry to CPU, and iteration stopping uses scalar decisions that can synchronize the device. CUDA scatter-add can be nondeterministic. Device-native arithmetic, constant-in-iteration saved state, CUDA Graph capture, deterministic execution, and high throughput are distinct properties; only the first two are design properties here.

## 7. Symmetric conductances and matrix-free CG

### 7.1 Reduced SPD operator

The [`MatrixFreeSymmetricTutteLayer`](../../src/qcopt/neural_bijection/tutte/symmetric.py), backend `symmetric`, uses one positive scalar per undirected edge incident to an interior vertex:

\[
c_e=c_{\min}+\operatorname{softplus}(z_e),\qquad c_{\min}>0. \tag{14}
\]

Boundary-boundary edges do not affect a Dirichlet extension and are not parameterized. Define the reduced Laplacian \(K\) and boundary coupling \(W\) by

\[
K_{ii}=\sum_{j\in N(i)}c_{ij},\quad
K_{ij}=-c_{ij}\ (i\ne j,\ i,j\in I),\quad W_{ib}=c_{ib}.
\]
\[
KX=WC. \tag{15}
\]

For scalar \(v\) on interior vertices, with boundary values extended by zero,

\[
v^TKv=\sum_{\{i,j\}\in E_{II}}c_{ij}(v_i-v_j)^2
       +\sum_{\{i,b\}\in E_{IB}}c_{ib}v_i^2>0\quad(v\ne0). \tag{16}
\]

Strict positivity follows because every interior component reaches the boundary. Thus \(K\) is SPD. It is applied using edge differences and scatter-adds, with no sparse matrix or factorization. Both primal and adjoint use the same operator.

Scaling all conductances of a sample by one positive number scales both sides of (15) and leaves the solution unchanged. For a nonempty active-edge set, the code uses \(\widehat c=c/s\), \(s=\max_e c_e\), before solving, to moderate arithmetic scale. This normalization remains inside autograd. Finite positive conductances are checked again after division because normalized support can underflow. For a boundary-only disk, \(n_I=0\) and the active-edge set is empty: the code bypasses `amax` and normalization, and scatters the prescribed boundary coordinates directly into global vertex order; boundary and returned-face checks still apply.

### 7.2 Relation to the directed family

Let \(d_i=\sum_jc_{ij}\). Dividing the \(i\)-th equation in (15) by \(d_i\) produces (1) with \(p_{ij}=c_{ij}/d_i\). For an interior-interior edge,

\[
d_ip_{ij}=d_jp_{ji}. \tag{17}
\]

Thus symmetric conductances form a constrained, reversible **operator/parameter subfamily** of independently directed rows. The associated convex quadratic energy is \(\tfrac12\sum_{\{i,j\}\in E_{II}\cup E_{IB}}c_{ij}\|y_i-y_j\|^2\), with boundary fixed; any boundary-boundary terms would be constant and can be omitted. This interpretation is specific to symmetric conductances.

Equation (17) does not by itself establish a strict separation between the sets of attainable vertex maps: different operators can yield the same harmonic coordinates. Conversely, a directed target generated by our benchmark is not automatically known to have a symmetric realization. Symmetric-versus-directed fitting therefore measures approximation and optimization behavior under the shared protocol; it must not be presented as a theorem of universal directed representation or universal symmetric insufficiency.

### 7.3 CG with reliable residual updates

When \(n_I>0\), CG starts each batch/RHS column from zero after scaling its RHS by its maximum absolute component. When \(n_I=0\), the CG helper returns the empty solution and zero-iteration success statistics immediately, before any RHS scaling or CG recurrence. Its adjoint takes the same empty-system bypass: the boundary VJP is the output cotangent gathered in boundary-loop order, and the conductance/logit VJP is empty. For nonempty systems, standard steps use

\[
\alpha_k=\frac{r_k^Tr_k}{p_k^TKp_k},\quad
x_{k+1}=x_k+\alpha_kp_k,\quad
r_{k+1}^{\rm rec}=r_k-\alpha_kKp_k,
\]
\[
\beta_k=\frac{(r_{k+1}^{\rm rec})^Tr_{k+1}^{\rm rec}}{r_k^Tr_k},\quad
p_{k+1}=r_{k+1}^{\rm rec}+\beta_kp_k. \tag{18}
\]

The implemented update modifies (18) at two events: a recursive convergence candidate, or every 32 iterations. For active columns at such an event, it recomputes \(r=b-Kx\). If that true residual is not accepted, the column restarts with \(p=r\), i.e. \(\beta=0\). Otherwise it becomes inactive. Between these events the usual recurrence is retained. The reliable-update mask is intersected with the active mask; a zero or already-converged RHS must not trigger an extra true-residual matvec on every iteration.

All columns undergo a fresh final true-residual test. Let \(\tau_{\rm req}\) denote the requested absolute-plus-relative threshold. On CUDA float32 only, the recurrence activity and reliable-update candidate tests use the stricter internal threshold

\[
\tau_{\rm int}=0.99\,\tau_{\rm req},
\]

whereas final acceptance still requires the freshly recomputed represented residual to be no larger than \(\tau_{\rm req}\). CPU and float64 use \(\tau_{\rm int}=\tau_{\rm req}\). The one-percent headroom addresses measured variation when CUDA scatter accumulation re-evaluates \(Kx\) at the tolerance boundary; it does not relax the public contract, make scatter deterministic, or prove convergence. It can require additional iterations, and failure remains explicit. Nonpositive/nonfinite active curvature, nonfinite residuals, an unmet final threshold, and nonfinite reconstructed solutions raise errors. The current method is **unpreconditioned CG**, not Jacobi-PCG. Defaults are `rtol=1e-5` for float32, `rtol=1e-11` for float64, `atol=0`, and 1000 iterations.

For a nonempty interior and \(s_b=\max_i|b_i|\), the internal tolerance is \(\mathrm{atol}/s_b+\mathrm{rtol}\|b/s_b\|_2\), with a safe unit scale for a zero-valued RHS; this scaling is not evaluated for the boundary-only bypass. Thus the residual test is performed on the scaled represented problem; rounding when rescaling the returned solution is not an exact-arithmetic equivalence. The returned map also passes the geometric checks. Reliable residual replacement corrects false recursive stopping. Relaxing the float32 default to `1e-5` is a separate availability/accuracy tradeoff, not a proof that the original strict tolerance now always converges. See the reliable-GPU entries in the [worklog](WORKLOG.md).

Saved state comprises conductances, boundary values, and an independent interior solution, not a CG history. The module's `last_forward_stats` and `last_adjoint_stats` are latest-call summaries, not persistent per-graph diagnostic handles. Geometry checks still synchronize/copy to CPU on CUDA.

## 8. Implicit derivatives

Use the Frobenius pairing \(\langle U,V\rangle=\operatorname{tr}(U^TV)\). Let a scalar loss supply vertex cotangents \(G_I\) and \(G_B\). These already include any downstream interpolation/image/composition contributions. The boundary is an explicit part of the output, which is why \(G_B\) must not be omitted.

### 8.1 Directed probabilities and boundary

Differentiating (7),

\[
dA=-dQ,\qquad A\,dX=dQ\,X+dR\,C+R\,dC. \tag{19}
\]

Solve exactly one two-column adjoint system

\[
A^T\Lambda=G_I. \tag{20}
\]

Then

\[
d\mathcal L
=\langle\Lambda,dQ\,X+dR\,C+R\,dC\rangle
 +\langle G_B,dC\rangle,
\]

and consequently

\[
\boxed{\ \overline C=G_B+R^T\Lambda,\qquad
\overline p_{ij}=\Lambda_i\cdot y_j\ }. \tag{21}
\]

The positive sign of \(\overline p\) follows from the minus sign in \(dA=-dQ\). Replacing (20) with a primal solve is incorrect for directed weights. Boundary accumulation is a scatter-add over all interior rows adjacent to each boundary vertex.

### 8.2 Directed logits

The row-softmax Jacobian gives

\[
\boxed{\ \overline z_{ij}
=p_{ij}\left(\overline p_{ij}-\sum_{k\in N(i)}p_{ik}\overline p_{ik}\right)
=p_{ij}\Lambda_i\cdot\left(y_j-\sum_kp_{ik}y_k\right).\ } \tag{22}
\]

For an exact harmonic solution this can also be written

\[
\overline z_{ij}=p_{ij}\Lambda_i\cdot(y_j-X_i). \tag{23}
\]

The distinction between (22) and (23) matters numerically: \(X_i=\sum_kp_{ik}y_k\) holds only up to solver residual for approximate solves. The improved direct and matrix-free paths return probability cotangents (21) and let Torch differentiate softmax, retaining the weighted row mean in (22). The legacy reference explicitly uses the row-difference form (23). These formulas describe the derivative of the implicit mathematical solution; an inexact primal/adjoint supplies an approximation to it. They are not the derivative of the discrete branch sequence or stopping iteration of BiCGStab/CG.

Row shifts \(z_i\mapsto z_i+\gamma_i\mathbf1\) leave probabilities unchanged, and \(\sum_j\overline z_{ij}=0\) in exact arithmetic. This is a useful derivative check, not uniqueness of the latent representation.

### 8.3 Symmetric conductances

Differentiating (15) gives

\[
K\,dX=dW\,C+W\,dC-dK\,X,
\qquad K\Lambda=G_I. \tag{24}
\]

For an interior-interior edge \(e=\{i,j\}\), a conductance perturbation adds \((e_i-e_j)(e_i-e_j)^Tdc_e\) to \(K\), so

\[
\boxed{\ \overline c_{ij}=-(\Lambda_i-\Lambda_j)\cdot(X_i-X_j).\ } \tag{25}
\]

For an interior-boundary edge \(e=\{i,b\}\), its perturbation contributes both to the diagonal of \(K\) and to \(W\). Therefore

\[
\boxed{\ \overline c_{ib}=\Lambda_i\cdot(C_b-X_i),\qquad
\overline C_b=(G_B)_b+\sum_{i:\{i,b\}\in E}c_{ib}\Lambda_i.\ } \tag{26}
\]

There is no extra factor of two in (25): each undirected edge is parameterized and counted once in (16). Equations (25)–(26) use the actual conductances passed to the implicit solve, which in the current implementation are \(\widehat c\).

For \(\widehat c_e=c_e/s(c)\), the outer autograd chain is

\[
\overline c_e=\frac{\overline{\widehat c}_e}{s}
 -\frac{\sum_f\overline{\widehat c}_fc_f}{s^2}\frac{\partial s}{\partial c_e},
\qquad \overline z_e=\operatorname{sigmoid}(z_e)\,\overline c_e. \tag{27}
\]

At ties, `amax` uses its implemented subgradient. Exact scale invariance implies \(\sum_e\overline{\widehat c}_e\widehat c_e=0\); finite solve errors need not satisfy this identity exactly. Detaching the normalization would bypass part of (27), and is not the implemented contract.

### 8.4 Batched and broadcast derivatives

Directed latents have trailing shape \((n_I,D)\), symmetric latents \((|E_{II}|+|E_{IB}|)\), and boundaries \((n_B,2)\). Each may be unbatched or have one leading batch dimension. Batch sizes must agree or be singleton; singleton expansion sums cotangents back to the shared input. Output uses global vertex order and is unbatched only when both inputs were unbatched. Dtype/device agreement and prepared device buffers are enforced by the backend contract. Broadcasting does not justify treating two distinct matrices as a single solve.

## 9. Dense evaluation and dynamic composition

### 9.1 Fixed query table

[`StructuredDenseQueryTable`](../../src/qcopt/neural_bijection/tutte/dense_warp.py) stores three vertex indices and three barycentric weights per fixed source query. For \(Q=H_qW_q\) align-corners grid points, the map samples are

\[
F_q=\sum_{a=0}^2\beta_{qa}Y_{v_{qa}}. \tag{28}
\]

The table is constructed once for the verified structured topology, and explicitly prepared for the desired device/dtype outside the timed interpolation path. A prepared `interpolate` call gathers, multiplies, sums, and reshapes; it performs no point location, sparse solve, or factorization. Backward scatters \(\beta_{qa}\overline F_q\) to control vertices. There is no dense \(Q\)-by-\(n\) interpolation matrix.

The analytic locator uses local cell coordinates \((u,v)\in[0,1]^2\). For \(v\leq u\), the vertices are \((v_{00},v_{10},v_{11})\) and weights \((1-u,u-v,v)\); otherwise they are \((v_{00},v_{11},v_{01})\) and weights \((1-v,u,v-u)\). The maximum-domain coordinate is assigned to the last cell. This matches the actual mesh diagonal, including on edges and corners.

Interpolation is not a topology validator for arbitrary user-provided control tensors. Its guarantee is conditional on the solver/control-map contract. Also, samples of a homeomorphism do not certify a new P1 or bilinear interpolant through those samples on an unrelated grid.

### 9.2 Composition of coordinate maps

For unit-square maps \(f_1,\ldots,f_L\), evaluate

\[
q_0=q,\qquad q_\ell=f_\ell(q_{\ell-1}),\qquad
F(q)=f_L\circ\cdots\circ f_1(q). \tag{29}
\]

[`SquareTutteComposition`](../../src/qcopt/neural_bijection/tutte/composition.py) supports one, two, or four layers. It solves each layer once at its own parameter batch size. A singleton layer subsequently shared across another layer's batch does not need duplicate solves. The first layer uses the fixed table; later layers must locate the **dynamic** points \(q_{\ell-1}\) in their structured source triangulations. Claiming that all multi-layer queries are precomputed would be incorrect.

Inside a selected triangle, barycentric coordinates are affine in the query, so

\[
dq_\ell=\sum_a\beta_a(q_{\ell-1})\,dY_{\ell,a}
       +J_{\ell,t}\,dq_{\ell-1}. \tag{30}
\]

The floor/cell/diagonal branch is locally constant away from the mesh skeleton. Autograd therefore gives the usual derivative almost everywhere; on shared edges or vertices it selects a branch derivative, not a globally smooth Jacobian. The function values agree across shared edges in exact arithmetic.

Queries outside the unit square by more than \(64\epsilon_{\rm dtype}\) are rejected. Smaller excursions are clamped to the closed domain; the outward coordinate derivative in that clamped region is zero. This is a disclosed roundoff policy, not image border padding or an extension of the homeomorphism theorem outside the source domain.

If each exact layer is a square homeomorphism, their exact composition is a homeomorphism. Its affine partition is a common refinement involving preimages of subsequent triangle edges. It is **not generally P1 on the original mesh**. Per-layer certificates therefore do not certify original-mesh triangles reconstructed from the composition's samples. The low-level `compose_control_maps` interpolation utility also accepts general in-domain controls; it does not turn arbitrary supplied controls or custom solvers into certified homeomorphisms.

Composition operates on coordinates, not repeatedly resampled images. For registration the geometric convention is fixed-to-moving: \(F(q)\) gives the moving-image coordinate sampled at fixed-image coordinate \(q\). The final image objective uses \(I_m(F(q))\) against \(I_f(q)\), with `align_corners=True`; no differentiable inverse is hidden in this convention. Image interpolation has its own discretization and differentiability properties and is not part of the control-map injectivity theorem.

## 10. Numerical validity, rejection, and guarantee boundaries

The following distinctions are part of the method, not optional benchmark details.

### 10.1 Input, boundary, and topology checks

Source topology and identity geometry are validated once as in Section 2. Supported tensors must have admissible shapes, nonempty/matching-or-singleton batches, supported precision, finite values, matching dtype/device, and the required prepared buffers. The CPU direct/reference adapters reject CUDA input rather than silently moving it. Structured boundary and dense-query modules additionally require their prescribed structured mesh convention.

Target boundary validation rejects nonfinite or repeated/too-close adjacent vertices, detected nonadjacent segment intersections, wrong orientation, and detected nonconvex turns. The inherited weak-convex check currently uses a floating tolerance of `1e-12`; it allows sufficiently small negative turns. It is neither an exact convexity predicate nor uniformly scale invariant. Structured side-order construction provides the stronger intended rectangle incidence.

Legacy/reference and symmetric paths explicitly check dividing edges against the represented target boundary before solving. The improved direct and directed matrix-free paths currently reuse source validation and the weak-convex boundary screen, but do **not** separately call the cached dividing-edge preflight. On the structured rectangle with the prescribed boundary construction the obstruction is excluded by construction; on a generic supplied boundary one must not describe all backends as having executed that additional preflight. Their returned-face screen still rejects detected collapses. This implementation difference does not remove the dividing-edge hypothesis from the exact theorem.

All current Route I adapters check original target faces in the **returned dtype**, after any hidden float64 solve has been cast. Coordinates are copied/cast to CPU float64 for this check; casting does not recover information lost in float32. For the shared face check, the signed double area must exceed

\[
64\epsilon_{64}\,s^2,\qquad
s=\max\{\operatorname{range}(Y_x),\operatorname{range}(Y_y),1\}. \tag{31}
\]

This is a positive-area margin in the implemented arithmetic, not a minimum singular-value bound, a float32 error analysis, or an exact orientation predicate. It can reject valid but very small/skinny maps. Cancellation, extreme coordinate scale, and floating boundary predicates remain reasons not to elevate it to a universal formal certificate. Standalone legacy APIs must not inherit the adapter's post-cast claim merely because their names are similar.

### 10.2 Solver rejection is not topology repair

Normalized weights/conductances must retain finite strict positivity. Sparse failures propagate; iterative paths reject breakdown, unsafe scale, failed finite checks, or unmet residual budgets under the rules above. The matrix-free forward and backward are separately subject to these rules: a successful primal does not promise a successful adjoint for every upstream cotangent. A fallback changes the numerical solution algorithm only; it does not repair a folded map. Failed geometry is rejected, not projected or untangled.

“Fail closed” here means that a detected violation prevents returning an accepted result. It does not mean that every conceivable floating-point degeneracy has an exact decision procedure, or that every finite input is guaranteed to return rather than raise. A production claim would need the corresponding predicate/error-bound and device-synchronization work, not simply more passing examples.

### 10.3 Residual, map error, derivative error, and Beltrami distortion

For an exact operator and an approximate solution, let \(r=b-A\widetilde x\). Then

\[
x-\widetilde x=A^{-1}r,
\quad\|x-\widetilde x\|\leq\|A^{-1}\|\,\|r\|. \tag{32}
\]

An analogous bound holds for the transpose adjoint. Equations (21)–(27) use both primal and adjoint, so both errors affect the VJP. No conditioning-independent gradient accuracy follows from (12). The reported “true residual” is true relative to the represented operator/matvec and its arithmetic, not an exact real residual or necessarily an independently replayed float64 residual. CPU-double replay can differ from native CUDA float32 evaluation.

For \(f=u+iv\), a triangle Jacobian defines

\[
f_z=\tfrac12(u_x+v_y)+\tfrac i2(v_x-u_y),\qquad
f_{\bar z}=\tfrac12(u_x-v_y)+\tfrac i2(v_x+u_y),\qquad
\mu_t=f_{\bar z}/f_z. \tag{33}
\]

For a nondegenerate positive-orientation triangle, \(\det J_t=|f_z|^2-|f_{\bar z}|^2>0\), hence \(|\mu_t|<1\). This is a consequence for this particular P1 map, not a promise of a user-specified margin \(|\mu|\leq k<1\), an exact target coefficient, or a uniformly bounded condition number. Derivatives of the P1 map also involve inverse source edge matrices; small coordinate error need not imply small trianglewise \(\mu\) error. Topology, residual, map error, and Beltrami error must be reported separately.

In particular, neither a continuum diffeomorphism nor a continuum condition \(|\mu|<1\) proves that its samples on an arbitrary fixed mesh form a fold-free P1 interpolant. Route I's theorem is about its specified discrete construction under the explicit hypotheses.

## 11. Computational cost, memory, and counting conventions

Let \(b\) denote batch size, \(r\) RHS columns, \(Q\) query points, and \(k\) realized iterations. Mesh connectivity is fixed. For a planar triangular disk \(|E|=O(n)\), but a padded directed table has \(n_ID\) slots; bounded average degree does not guarantee bounded maximum degree on every generic mesh.

| Component | Work per call/sample as applicable | Principal retained data |
|---|---|---|
| Directed neighbor matvec | \(O(bn_IDr)\) | Neighbor tables; probabilities; constant number of state/work tensors |
| Symmetric edge matvec | \(O(b|E_{\rm active}|r)\) | Edge tables; conductances; constant number of state/work tensors |
| Krylov solve | Matvec/reduction costs times realized \(k\), including explicit residual checks | No iteration-length autograd tape; working gather tensors can be \(O(bn_IDr)\) |
| Stationary fallback | Additional matvec/reduction work after the failed primary attempt | Constant-in-iteration workspace |
| Sparse direct | Factorization cost \(\mathcal F(A)\), then triangular-solve cost determined by factor fill | Numeric factors of size \(\mathcal S(A)\), plus primal/state |
| Fixed dense interpolation | \(O(bQr)\) | \(O(Q)\) immutable three-entry table; output/query activations |
| Dynamic interpolation per later layer | \(O(bQr)\) structured location/interpolation | Query-dependent indices, weights, and activations |
| Current target geometry screen | \(O(bn_B^2+b|F|)\) | Host geometry copies; no asymptotic speed claim |

No linear-time sparse-LU claim is made: \(\mathcal F\) and \(\mathcal S\) depend on ordering and fill. The legacy baseline constructs separate factors for \(A\) and \(A^T\) during forward, but only retains the transpose factor in the autograd context after forward; improved direct retains its single factor. Two factorization calls therefore do not imply two graph-retained factors. Neither is a matrix-free method. For matrix-free implicit differentiation, state storage is independent of iteration limit, not independent of mesh, batch, layers, queries, or the lifetime of multiple autograd graphs. A composition retains per-layer solver state and ordinary coordinate-interpolation activations, approximately \(O(Lbn+LbQ)\) at fixed degree/RHS, in addition to fixed tables. Dense images can dominate memory despite an iteration-independent solver tape.

We distinguish the following counters:

1. A **logical global solve** is one sample's primal or adjoint linear system with its x/y columns handled together. It is not two solves just because \(r=2\).
2. A batched backend invocation may contain \(b\) such sample systems. A composition's sample-system count is the sum of its layers' actual parameter batch sizes; later interpolation broadcasting does not create extra systems.
3. **Factorizations** and **iterations/matvecs** are different counters. Legacy performs two factors per nonempty sample forward, direct one, and matrix-free zero. Reusing a factor in backward still performs an adjoint solve.
4. A fallback is a second numerical attempt at the same logical system. Its work/time must remain visible even if the logical-solve counter does not increase.
5. A boundary-only map has no interior linear system; counters must distinguish a decoder invocation from nonempty numerical work.

For the official architecture's 24 planar sublayers, the audited convention is 24 two-column forward systems, not 48 coordinate-wise factorizations. This count does not make its mesh, solver, parameterization, locator, or backward cost equivalent to ours. A multilayer comparison must account for every layer's primal, adjoint, and dynamic query work. [Official audit](01_tuttenet_code_audit.md).

In the instance runner, \(S\) Adam updates with one initial/final sequence of observations incur \(S+1\) primal evaluations and \(S\) adjoints per sample/layer when all updates succeed. LBFGS has data-dependent closure evaluations; outer steps are not solve counts. Target generation is a separately charged setup solve, not training. Audit time, solver time, dense warp time, optimizer-only overhead, and observation wall time should remain separately identified. A threshold first observed in an LBFGS trial closure is an audited **evaluation**, not automatically an accepted optimizer state. CUDA timings require the benchmark's synchronization policy; raw host launch latency is not completed device work.

## 12. End-to-end algorithm and evaluation handoff

### 12.1 Single layer

The current computation is:

1. Once per source mesh, validate its disk topology and identity realization; build fixed neighbor/edge and boundary indexing; build and prepare the dense query table.
2. Realize the ordered rectangle boundary from side logits and optional height; reject invalid represented support, height, or side order.
3. Realize positive directed probabilities or symmetric conductances, with the backend-specific precision/normalization described above.
4. Form/apply the reduced Dirichlet operator and solve the two coordinate columns; enforce that solver's convergence/finite checks.
5. Scatter the boundary/interior coordinates and check the original faces in the returned precision. Reject detected invalid geometry.
6. Evaluate the fixed P1 query table. During backward, scatter query cotangents, solve the appropriate adjoint, and apply (21)–(27) followed by the boundary chain (5).

For composition, steps 2–5 are performed separately for each square layer, followed by the coordinate recursion (29). The first query table is fixed; later queries are dynamic. No clipping/untangling of failed control maps or repeated image resampling is included in this algorithm.

### 12.2 Evidence schema and completed application handoff

The [worklog](WORKLOG.md) records unit/dense-reference/finite-difference checks, independent reviews, known failure cases, and bounded remote smoke/stress observations. The completed engineering, adversarial, instance-optimization, and extension matrices are aggregated in the independently reviewed [application chapter](03_tutte_application.md), with raw receipts under `raw_results/`. The following fields define the comparison schema; no value is inferred from a method name:

| Backend and environment | Final evidence location | Required numerical-work fields | Accuracy/topology fields kept distinct | Timing/memory scope |
|---|---|---|---|---|
| `reference` / CPU | Application §6.1 | Primal/adjoint systems and two forward factors per nonempty sample | Independent primal replay and returned-map audit; no fitted accuracy is invented | Fresh process, warmup/repeats, synchronized workload, process HWM |
| `direct` / CPU | Application §§6.1 and 7 | One retained factor per nonempty sample forward, multi-RHS and transpose solves | Independent residual plus formal map/Beltrami metrics where a target exists | Setup/audit separated from optimization or engineering time |
| `directed_iterative` / CPU or CUDA | Application §§6.1–6.2, 7, and 8.3–8.4 | Dtype/tolerances, primal/adjoint iterations, primary failure and fallback kept explicit | Residual, map/VJP comparison, topology, and fitted \(\mu\) error reported only when actually measured | Native/float64 and normal/adversarial scopes remain separate |
| `symmetric` / CPU or CUDA | Application §§6.1–6.2, 7, 8.1–8.2, and 8.5 | CG invocation/iteration fields, requested versus internal threshold, no factor count | Native and independent replay residuals, topology, fitting error, and failure blanks remain distinct | Original and clean-postfix matrices remain separate; GPU allocator and process HWM are labelled |

The shared instance target is generated once by a positive directed map and passed unchanged to the compared backends, with target setup outside training timing. Ordinary Git provenance and target shape/sum/L2 summaries help reproduce and compare this setup; summaries are not cryptographic identities, and Python object IDs have meaning only within one process. The fitting target protocol does not prove representability by the symmetric parameter family, global convergence of an optimizer, superiority of a solver, or completion of Route I. No such conclusion is asserted here.
