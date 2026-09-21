# TutteNet: mathematical representation and official implementation audit

## Abstract

This document reconstructs the implemented TutteNet representation from pinned official sources, separating control-mesh construction, sparse embedding solves, and dense-query evaluation. The released learning configuration uses a center-split square mesh with 221 vertices, 40 boundary vertices, and 181 interior unknowns per coordinate. Its 24 planar sublayers each solve a sparse system with two right-hand sides. The inspected torch_sparse_solve backend uses CPU double-precision KLU and shares a factorization between coordinate columns, but not between forward and backward. Learning and fitting differ in their weight parameterizations and forward point-location algorithms. We derive the corresponding systems and adjoints, document source-derived operation counts, and identify the controlled comparison possible with the present factor-reusing SuperLU reference. No official training run, published performance number, or end-to-end speedup is reproduced or claimed here.

## 1. Scope, sources, and evidence categories

This is the Route I-A source audit for Phase V; it does not rank or explore later routes. Four categories of evidence are distinguished:

1. **Static verification:** behavior read from the pinned official sources.
2. **Executed local checks:** official mesh-builder counts and focused tests of this repository's new reference layer.
3. **Mathematical deductions:** algebraic consequences of the specified systems, not measured runtime properties.
4. **Unverified runtime claims:** historical software environments, training/checkpoint reproduction, official backend performance, and end-to-end throughput.

The official repositories were cloned only under the worktree's temporary directory. Exact revisions, verified with git rev-parse HEAD, are:

| Source | Inspected revision | Scope |
|---|---|---|
| [GitBoSun/TutteNet](https://github.com/GitBoSun/TutteNet/tree/cb9f91969ba012f5670c6066298760314d8d31a5) | cb9f91969ba012f5670c6066298760314d8d31a5 | Official learning, fitting, and selected NeRF paths |
| [flaport/torch_sparse_solve](https://github.com/flaport/torch_sparse_solve/tree/89c227c0e7771b5c86dca0f25b78c14b4edc85b1) | 89c227c0e7771b5c86dca0f25b78c14b4edc85b1 | Python wrapper and C++ KLU implementation; module version 0.0.5 |

The paper actually inspected is the authors' [arXiv HTML, 2406.12121v1](https://arxiv.org/html/2406.12121v1). The [CVPR proceedings PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_TutteNet_Injective_3D_Deformations_by_Composition_of_2D_Mesh_Deformations_CVPR_2024_paper.pdf) returned HTTP 403 through the browsing tool; textual identity with the inspected arXiv version was not independently established.

At paper level, §3 defines a piecewise-affine planar map, a rotated prismatic lift, and composition. Equation (2)'s initialization-time inverse concerns local affine coefficients. Algorithm 2 stores per-face affine maps after solving embeddings; Algorithm 1 still locates each propagated query's triangle. Table 3 varies mesh resolution and layer count; Table 4 instead keeps (N=11) and 24 layers fixed while ablating the orientation policy. Section 5 notes limitations in interactive evaluation. None of these statements establishes cached sparse factors or resolution-independent end-to-end cost. [Paper §§3–5](https://arxiv.org/html/2406.12121v1).

The official [installation instructions, lines 13–14](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/README.md#L13-L14) link the backend without pinning its commit. The released requirements also leave torch_geometric unpinned. Consequently, this audit identifies behavior at the stated revisions, not the exact dependency build used for every historical experiment.

## 2. Control mesh, indexing, and unknowns

### 2.1 Center-split topology

Let \(N\) denote the official MESH_RESOLUTION. The square contains \(N\times N\) grid vertices and an additional center vertex in each of its \((N-1)^2\) cells. Each center connects to the four corners, producing four triangles per cell. This is not the two-triangle-per-cell mesh used by this repository's structured_rectangle.

The official constructor first uses \([0,1]^2\): grid vertex \(iN+j\) has coordinates

\[
(j/(N-1),\,1-i/(N-1)),
\]

and center vertex \(N^2+i(N-1)+j\) has the corresponding half-grid offset. The template then maps coordinates to \([-1,1]^2\). Its explicitly assembled boundary loop must not be replaced by sorted vertex IDs. Sources: [build_uniform_square_graph, lines 141–179](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_models/tutte_template.py#L141-L179) and [initialization, lines 30–45](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_models/tutte_template.py#L30-L45).

Writing \(V,B,I,F,E\) for total vertices, boundary vertices, interior vertices, triangles, and undirected edges, respectively,

\[
\begin{aligned}
V&=N^2+(N-1)^2,& B&=4(N-1),& I&=V-B,\\
F&=4(N-1)^2,& E&=2N(N-1)+4(N-1)^2.
\end{aligned}
\]

The code stores both edge orientations and removes duplicates, giving \(2E\) weight slots. This storage choice does not itself establish independently directed weights.

### 2.2 Independently executed counts

The exact official mesh-building method was extracted by Python AST and executed with NumPy without importing or training the full model. Unique oriented face edges and the interior-interior structure were counted. These are topology counts, not timings or LU fill measurements.

| Official \(N\) | \(V\) | \(B\) | \(I\) | \(F\) | Undirected \(E\) | Directed slots | Reduced structural nonzeros |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 85 | 24 | 61 | 144 | 228 | 456 | 341 |
| 11 | 221 | 40 | 181 | 400 | 620 | 1240 | 1117 |
| 17 | 545 | 64 | 481 | 1024 | 1568 | 3136 | 3121 |
| 25 | 1201 | 96 | 1105 | 2304 | 3504 | 7008 | 7361 |

The active [learning configuration, lines 5–13](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/tools/cfgs/smpl_models/coord_model.yaml#L5-L13) sets \(N=11\). Each sample therefore solves a \(181\times181\) reduced system with two coordinate columns. Batched matrix and RHS shapes are \((m,181,181)\) and \((m,181,2)\), where \(m\) is batch size. The full graph has 1240 off-diagonal entries plus 221 diagonal entries before elimination.

A query grid containing \(256^2\) or \(512^2\) points changes neither 181 nor 1117. It changes dense-query evaluation, transfer, and differentiation work.

## 3. Mathematical system and parameterization

### 3.1 Interior equation

Let \(\mathcal I,\mathcal B\) be the interior and ordered boundary sets. For positive supported weights \(w_{ij}\), define \(d_i=\sum_jw_{ij}\), \(D=\operatorname{diag}(d_i)\), and \(L=D-W\). The unknown \(X_{\mathcal I}\in\mathbb R^{I\times2}\) and prescribed boundary \(C\in\mathbb R^{B\times2}\) satisfy

\[
L_{\mathcal I\mathcal I}X_{\mathcal I}
=-L_{\mathcal I\mathcal B}C
=W_{\mathcal I\mathcal B}C. \tag{1}
\]

Both coordinates use the same matrix. Equivalently, dividing each interior equation by \(d_i>0\) gives

\[
(I-P_{\mathcal I\mathcal I})X_{\mathcal I}
=P_{\mathcal I\mathcal B}C,\qquad p_{ij}=w_{ij}/d_i. \tag{2}
\]

This equivalence does not mean that official TutteNet predicts row-softmax logits. The active call is get_laplacian(edges, W_var) without a normalization argument, followed by boundary elimination: [head lines 336–362](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_heads/tutte_head_3d_distortion.py#L336-L362). The [official PyG implementation](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/laplacian.html) confirms the default \(D-W\); the current dependency source is supporting evidence, not a historical pin.

### 3.2 Learning path

The released configuration selects CoordinateEncoderFeature, one outer layer, 24 inner layers, and predicted normals. Encoded DINO/CLIP features and pose parameters condition shared networks for edge weights and boundary increments. A separate network predicts normalized three-component normals. At \(N=11\), the per-outer-layer parameter tensor has shape \((m,1240+40,24)\), accompanied by 24 normal vectors for this configuration. See [encoder lines 163–197](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/backbones/coord_encoder_feat.py#L163-L197).

For active weight and increment logits \(z_{ij},a_k\), the implemented transforms are

\[
w_{ij}=0.2+0.6\,\sigma(z_{ij}/100),\qquad
r_k=0.1+0.8\,\sigma(a_k/100). \tag{3}
\]

These ranges are open in exact arithmetic; finite-precision sigmoid saturation can attain endpoints. The bounded parameterization does not allow arbitrary positive weight ratios. Sources: [configuration lines 29–38](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/tools/cfgs/smpl_models/coord_model.yaml#L29-L38) and [head lines 264–290](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_heads/tutte_head_3d_distortion.py#L264-L290).

### 3.3 Fitting and the symmetry distinction

The standalone fitting script assigns independent nn.Parameter values to all directed edge slots and boundary slots. It uses

\[
w_{ij}=0.2+0.6\,\sigma(z_{ij}),\qquad
r_k=0.2+0.6\,\sigma(a_k), \tag{4}
\]

without division by 100. See [initialization, lines 271–278](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/fitting_experiments/optimize_one_pair.py#L271-L278) and [transforms, lines 332–338](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/fitting_experiments/optimize_one_pair.py#L332-L338).

The learning encoder instead represents an edge by its midpoint. Reverse edges have identical conditioning/midpoint inputs to the same weight network, hence their predicted weights coincide mathematically. This follows from [encoder lines 55–60](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/backbones/coord_encoder_feat.py#L55-L60) and [lines 175–184](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/backbones/coord_encoder_feat.py#L175-L184), not from solver-level symmetrization.

Three notions therefore remain distinct:

- Both implementations store directed edge entries.
- The released learning encoder ties reverse values; the fitting parameters need not.
- Even for \(W=W^\mathsf T\), \(P=D^{-1}W\) is generally nonsymmetric. Symmetry before row scaling must not be conflated with symmetry afterward.

The solver head can consume asymmetric values. Encoder-level tying is not a general restriction of the head, and bounded sigmoid weights do not establish unrestricted expressivity over all positive barycentric systems.

### 3.4 Boundary parameterization

The default non-rotating boundary path defines

\[
\alpha_k=2\pi\frac{r_k}{\sum_jr_j},\qquad
\theta_k=\sum_{\ell=1}^{k}\alpha_\ell,\qquad
c_k=\frac{(\cos\theta_k,\sin\theta_k)}
{\max(|\cos\theta_k|,|\sin\theta_k|)}. \tag{5}
\]

The last expression is the square-ray intersection implemented by tangent/cotangent branches in [head lines 307–334](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_heads/tutte_head_3d_distortion.py#L307-L334). The circle option is not active in this default.

Samples lie in cyclic order on the square perimeter, but four corners are not separately pinned. Joining samples across a corner can cut across it; the image boundary is an inscribed polygon, not necessarily the whole square boundary. Several samples may also be collinear on a side. This audit establishes the parameterization, not a new theorem for every weakly convex, finite-precision, or out-of-domain case. A universal numerical injectivity claim does not follow from identifying the sigmoid alone.

## 4. Sparse KLU forward, adjoint, and factor lifetime

### 4.1 Forward

The backend solves \(AX=R\) for sparse batched \(A\) and dense batched \(R\). Its Python wrapper checks three-dimensional square sparse input and float64 matrix/RHS. Its declared batch contract requires matching sparsity patterns. C++ loops through samples sequentially and converts each COO matrix to CSC.

For each sample, the internal KLU helper initializes defaults, performs klu_analyze and klu_factor, calls klu_solve with the RHS count, then frees both symbolic and numeric objects. This is sparse LU, not Cholesky or an iterative method; the implementation uses int32 indices and raw double arrays. Sources: [Python lines 10–53](https://github.com/flaport/torch_sparse_solve/blob/89c227c0e7771b5c86dca0f25b78c14b4edc85b1/torch_sparse_solve.py#L10-L53), [C++ lines 11–23](https://github.com/flaport/torch_sparse_solve/blob/89c227c0e7771b5c86dca0f25b78c14b4edc85b1/torch_sparse_solve.cpp#L11-L23), and [KLU helper lines 39–54](https://github.com/flaport/torch_sparse_solve/blob/89c227c0e7771b5c86dca0f25b78c14b4edc85b1/torch_sparse_solve.cpp#L39-L54).

The two coordinate columns already share a factorization within one call. It would be incorrect to claim that the official implementation factors separately for x and y.

### 4.2 Backward

For \(G=\partial\mathcal L/\partial X\),

\[
dX=A^{-1}(dR-dA\,X),\qquad A^\mathsf TY=G,
\]

and consequently

\[
\frac{\partial\mathcal L}{\partial R}=Y,\qquad
\frac{\partial\mathcal L}{\partial A}=-YX^\mathsf T. \tag{6}
\]

The backend evaluates the matrix gradient only on stored sparse entries, summing products over RHS columns. This is a nonsymmetric adjoint: replacing \(A^\mathsf T\) with \(A\) would not be justified by the fitting path.

The implemented backward calls the complete forward solver on the coalesced transpose and incoming gradient. This repeats conversion, symbolic analysis, and numerical factorization. The autograd context saves only \(A,R,X\), not factors. Sources: [C++ backward, lines 26–36](https://github.com/flaport/torch_sparse_solve/blob/89c227c0e7771b5c86dca0f25b78c14b4edc85b1/torch_sparse_solve.cpp#L26-L36) and [Python lines 51–60](https://github.com/flaport/torch_sparse_solve/blob/89c227c0e7771b5c86dca0f25b78c14b4edc85b1/torch_sparse_solve.py#L51-L60).

There is no retained factor or symbolic cache across forward/backward, layers, samples, or training iterations in this revision. TutteNet's cached graph masks and elimination indices are not cached KLU symbolic analysis.

### 4.3 Device and dtype

The backend supports CPU tensors only. The learning head prepares graph quantities on CUDA, transfers matrix values and RHS to CPU float64, and returns solved vertices as CUDA float32. It is neither a GPU sparse solve nor an end-to-end float64 forward. Sources: [backend README lines 3–19](https://github.com/flaport/torch_sparse_solve/blob/89c227c0e7771b5c86dca0f25b78c14b4edc85b1/README.md#L3-L19) and [learning head lines 360–366](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_heads/tutte_head_3d_distortion.py#L360-L366). The standalone fitting script instead constructs CPU mesh tensors ([lines 784–797](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/fitting_experiments/optimize_one_pair.py#L784-L797)).

No official native extension was compiled or benchmarked in this audit. Transfer and solver cost proportions remain unmeasured.

## 5. Dense queries and point-location variants

For a query \(q\) inside triangle \(t=(i,j,k)\),

\[
\Psi(q)=\lambda_i(q)X_i+\lambda_j(q)X_j+\lambda_k(q)X_k,\qquad
\lambda_i+\lambda_j+\lambda_k=1. \tag{7}
\]

The code obtains weights from absolute subtriangle areas divided by their sum. These are barycentric coordinates for an actually containing triangle. Absolute-area normalization does not validate an incorrect or out-of-domain triangle selection.

### 5.1 Learning: SciPy per-forward search

The learning layer initializes Delaunay(vertices), then overwrites its simplices with the fixed face array. Each forward detaches the current queries, transfers them to CPU NumPy, and calls Delaunay.find_simplex with bruteforce=True before gathering triangle vertices and computing area interpolation. Sources: [head lines 208–235](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_heads/tutte_head_3d_distortion.py#L208-L235) and [lines 369–451](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_heads/tutte_head_3d_distortion.py#L369-L451).

Face selection is detached; interpolation retains derivatives with respect to coordinates inside the selected face. The inverse constructs a Delaunay object from deformed vertices per sample and searches again ([lines 239–262](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_heads/tutte_head_3d_distortion.py#L239-L262)).

### 5.2 Fitting: regular-grid arithmetic

The recommended standalone fitting script's forward uses its own locator: floor arithmetic identifies the grid cell, and direction/slope masks choose a center-split triangle. It does not use SciPy for forward location; its inverse does. Sources: [forward lines 280–312](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/fitting_experiments/optimize_one_pair.py#L280-L312) and [locator lines 526–566](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/fitting_experiments/optimize_one_pair.py#L526-L566).

The separate NeRF fields/tutte_model.py forward again uses SciPy ([lines 194–202](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/nerf_deformation/nerfstudio/fields/tutte_model.py#L194-L202)). Locator-cost claims must therefore name the implementation path.

### 5.3 Precomputation and unverified edge cases

As a dependency-based deduction, fixed source geometry and fixed projected queries allow precomputing triangle IDs and weights for one layer. Fixed topology permits caching adjacency/elimination structure. Neither permits reuse of a numerical factor after matrix coefficients change.

Later layers receive updated query positions; predicted rotations can alter even the first projection. Therefore fixed all-layer query tables are not generally valid during parameter updates. Frozen parameters would permit caching solved vertices or per-face affine maps, but the inspected forwards invoke the solves anew; a solve-free deployment cache was not verified.

Unresolved static caveats include: missing rejection of negative SciPy face IDs before indexing; version-specific consistency of replacing only Delaunay simplices; and unclamped cell indices plus a perturbed slope denominator in the fitting locator. Boundary, diagonal, and out-of-domain robustness have not been numerically certified here.

## 6. Composition, solve counts, and cost

### 6.1 Twenty-four planar layers

Let \(R_\ell\) denote the local-to-global rotation for a layer. A prismatic map preserves the third local coordinate:

\[
\Phi_\ell(p)=R_\ell
\begin{bmatrix}
\Psi_\ell((R_\ell^\mathsf Tp)_{1:2})\\
(R_\ell^\mathsf Tp)_3
\end{bmatrix},
\qquad f=\Phi_K\circ\cdots\circ\Phi_1. \tag{8}
\]

This describes the rotate–map–rotate-back dependency; a particular stored matrix may represent \(R_\ell\) or its transpose according to the code's convention.

Learning uses one outer block with 24 predicted-normal inner layers. Each iteration updates the query coordinates, and the outer forward propagates its output. Sources: [inner layers, lines 745–776](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_heads/tutte_head_3d_distortion.py#L745-L776) and [outer head, lines 1088–1113](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/pcdet/models/tutte_heads/tutte_head_3d_distortion.py#L1088-L1113).

Fitting sets num_layer=8, but each is a triplane block containing three sequential xy/xz/yz maps. Thus \(8\times3=24\) planar solves, not eight or 72. Sources: [configuration lines 35–36](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/fitting_experiments/optimize_one_pair.py#L35-L36) and [triplane lines 568–593](https://github.com/GitBoSun/TutteNet/blob/cb9f91969ba012f5670c6066298760314d8d31a5/fitting_learning/fitting_experiments/optimize_one_pair.py#L568-L593).

For one sample and one invocation traversing all 24 differentiable sublayers, the inspected backend implies:

- Forward: 24 analyses, 24 numerical LU factorizations, and 24 two-RHS solves.
- Backward through every solve: 24 additional analyses/factorizations of transposed matrices and 24 two-RHS adjoint solves.

These are source-derived counts, not profiler measurements. Batch size multiplies them; additional forward, inverse, or loss evaluations can add work.

### 6.2 Cost interpretation

With query count \(Q\), a useful decomposition is

\[
T_{\rm forward}=T_{\rm encoder}
+\sum_{\ell=1}^{K}
\left[
T_{{\rm assembly},\ell}
+T_{{\rm analysis+factor},\ell}
+T_{{\rm two\ RHS},\ell}
+T_{{\rm location},\ell}(Q)
+T_{{\rm interpolation+rotation},\ell}(Q)
+T_{{\rm transfer},\ell}
\right]. \tag{9}
\]

No term is assigned a measured proportion. Control resolution changes sparse-system size; query resolution changes location/interpolation and saved tensors; layer count affects both. A composition is piecewise affine on its induced partition, not necessarily on the original coarse mesh. Sampling it and constructing a new P1 interpolant on another fixed triangulation requires a separate validity analysis.

Layer maps depend on their parameters rather than query positions, so separating map construction from propagation is conceptually possible. The inspected code interleaves them. This is an optimization opportunity, not evidence for an already implemented separated pipeline.

## 7. Fair comparison with the improved CPU direct reference

The local [DirectTutteLayer](../../src/qcopt/neural_bijection/tutte/direct.py) uses Equation (2) with row-softmax supported probabilities. One scipy.sparse.linalg.splu call per sample with a nonempty interior creates a SuperLU factor for both coordinate columns. The autograd graph retains it and calls the same object's solve with trans="T" in backward.

For \(Y=A^{-\mathsf T}G_{\mathcal I}\), the normalized-system gradients are

\[
\frac{\partial\mathcal L}{\partial p_{ij}}=Y_i^\mathsf T X_j,\qquad
\frac{\partial\mathcal L}{\partial C}
=G_{\mathcal B}+P_{\mathcal I\mathcal B}^\mathsf T Y. \tag{10}
\]

PyTorch differentiates the surrounding softmax, giving

\[
\frac{\partial\mathcal L}{\partial z_{ij}}
=p_{ij}\left(g_{ij}-\sum_kp_{ik}g_{ik}\right),
\qquad g_{ij}=Y_i^\mathsf T X_j. \tag{11}
\]

These establish a factor-lifetime difference, not an end-to-end timing advantage.

| Dimension | Inspected official code | Improved direct reference | Fair interpretation |
|---|---|---|---|
| Control topology | Center-split; N=11 gives V=221, I=181, F=400 | Tested structured_rectangle, two triangles per cell | A local 11-by-11 vertex grid has V=121, I=81, F=200; equal labels do not mean equal systems |
| Parameters | Bounded sigmoid weights; midpoint-tied learning or directed fitting | Interior directed row-softmax logits | Different ranges and redundancies; not equal model capacity |
| Boundary | Network/parameter angle increments mapped to square sides | Supplied ordered boundary with geometric validation | Does not reproduce the official boundary predictor |
| Sparse solver | CPU float64 KLU | CPU double-internal SuperLU; float32/64 interface | Different packages/orderings confound timing unless controlled |
| x/y RHS | One shared forward factor | One shared forward factor | Multi-RHS reuse already exists in the official backend |
| Backward factor | Fresh LU of transpose | Retained LU's transpose solve | One factorization avoided per sample/layer for a single backward, not a measured 2× speedup |
| Cross-forward symbolic cache | None identified | None exposed | Neither establishes cross-iteration symbolic caching |
| Batch | Sequential backend loop | Sequential Python loop | Neither is a batched GPU sparse factorization |
| Graph-lifetime memory | Saves matrix/RHS/solution; frees LU in each solve | Retains LU and an independent immutable double primal | Trades saved-factor memory for avoided work; peak memory unmeasured |
| Dense queries | Path-dependent location and 24-layer propagation | Direct layer returns control vertices; fixed-query interpolation is separate | One 2D solve is not the complete official 3D model |
| Numerical validation | Full official checks not reproduced here | Rejects nonfinite inputs, zero supported probabilities, invalid boundary, and nonpositive post-cast face areas | Validation overhead belongs in the workload; no production-readiness inference |
| Gradient scope | First-order sparse adjoint inspected | First-order only; nonsymmetric tests | No higher-order or training-convergence comparison |

A fair solver ablation holds \(A,R,G\), precision, ordering policy, threads, and measurement scope fixed while varying factor retention. A fair model comparison must additionally match topology, parameter distribution, boundary, batch, rotations, query workload, and all 24 propagated stages. Historical paper timings are not baselines measured in this environment and cannot establish a speedup for the local module.

### 7.1 Executed local verification

The implementation turn ran [test_phase5_tutte_direct.py](../../tests/test_phase5_tutte_direct.py): 18 tests passed after the alias fix. Its combined run with test_phase5_dense_warp.py passed 23 tests without warnings. These are recorded test outcomes, not benchmark rows or throughput measurements. Tests cover an independently assembled nonsymmetric dense reference, both-input gradcheck/directional finite differences, boundary permutation, batching, invalid inputs, and factor-call instrumentation.

Instrumentation wraps the real splu and its real solves. With batch size two, one forward plus backward records two factorizations and four two-RHS calls: factor 0/N, factor 1/N, factor 0/T, factor 1/T. The output in-place regression initially reproduced a factor-of-two logit-gradient discrepancy; an independent read-only primal copy prevents output storage from corrupting the backward state. A separate example has positive double-precision faces but zero-area faces after float32 rounding and is rejected by the existing post-cast certificate.

Relevant test functions are test_one_factor_per_sample_and_reused_transpose_multirhs, test_inplace_output_scaling_does_not_corrupt_saved_primal, and test_float32_cast_collapse_rejected_despite_valid_double_solution. This is builder-side local verification, not independent approval or an official backend benchmark.

## 8. Conclusions and unresolved evidence

Static source inspection and executed topology counts establish that official TutteNet separates small control systems from dense evaluation, that its learning and fitting paths differ materially, and that the inspected KLU wrapper refactors in backward despite already sharing factors across coordinate RHS. The local reference changes factor lifetime and provides targeted first-order correctness evidence. Unless explicitly matched, it also differs in topology, parameterization, boundary interface, and workload.

Still unverified are exact historical dependencies; full official installation/training/checkpoint reproduction; published timing reproduction; current transfer and locator profiles; matched official-versus-reference accuracy/time/memory; locator edge cases; arbitrary-domain or arbitrary-topology guarantees; and higher-order differentiation. Historical numbers and focused green tests establish neither a production GPU layer nor a universal bijectivity result.
