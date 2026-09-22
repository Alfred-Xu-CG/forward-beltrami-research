# Fine-grid positive-conductance layers: Woodbury, Schur, and implicit gradients

This note analyzes Route C on the **same final triangular control mesh** as the output. A coarse latent can determine weights, but the returned map is never obtained by merely interpolating a coarse solution. All numerical comparisons below are from Phase VI and are conditional on the stated mesh, boundary, precision, and validation.

## 1. Discrete problem and its topology scope

Let \(G=(V,E)\) be the undirected one-skeleton of a planar triangular disk. Separate vertices into interior \(I\) and ordered boundary \(\partial V\). Fix the boundary coordinates \(Y_b\in\mathbb R^2\). For every edge \(e=(a,b)\), choose a symmetric conductance \(c_e>0\). If an edge joins two boundary vertices, its value does not affect the unknown Dirichlet extension and may be omitted from the learnable edge set. Let \(Y_i\in\mathbb R^2\) be unknown for \(i\in I\). The strictly convex quadratic energy is

\[
\mathcal E(Y_I;c)=\frac12\sum_{(a,b)\in E}c_{ab}\|Y_a-Y_b\|_2^2.
\]

Its stationarity equation at each interior vertex is

\[
\sum_{j:\{i,j\}\in E}c_{ij}(Y_i-Y_j)=0.
\]

Let \(B_I\in\mathbb R^{E\times |I|}\) be the oriented edge-vertex incidence matrix restricted to interior columns, and \(B_\partial\) its boundary columns. With any consistent edge orientation,

\[
A(c)Y_I=b(c,Y_\partial),\quad
A(c)=B_I^\mathsf T\operatorname{diag}(c)B_I,\quad
b=-B_I^\mathsf T\operatorname{diag}(c)B_\partial Y_\partial.
\]

Every connected component of interior vertices touches the fixed boundary in the complete rectangular grid. Therefore \(B_I\) has full column rank and \(A(c)\) is symmetric positive definite: for nonzero \(v\), \(v^\mathsf TA(c)v=\sum_ec_e(B_Iv)_e^2>0\). Thus the linear system has a unique exact solution. This is a statement about algebraic solvability, not by itself a global-topology theorem.

The topological conclusion uses the precise convex-combination theorem, not positive definiteness alone. The normalized interior equation is \(Y_i=\sum_j(c_{ij}/\sum_kc_{ik})Y_j\), with **strictly positive** coefficients on every incident edge. [Floater's one-to-one PL mapping theorem (2003, Theorem 6.1)](https://doi.org/10.1090/S0025-5718-02-01466-7) states, for a triangulated topological disk whose boundary maps homeomorphically onto a convex polygon, that this convex-combination map is injective if and only if no *dividing edge* (an internal edge joining two boundary vertices) maps into the target boundary. The theorem permits the collinear intermediate vertices on each side of our square; a strict-convex-vertex version cannot simply be substituted. On the canonical \(257^2\) triangulation with its fixed identity square boundary, the graph has two dividing edges; both fail the forbidden boundary-containment condition. The constructor checks the ordered simple weakly convex boundary and dividing edges before solving. Thus, **in exact arithmetic**, every strictly positive conductance vector in this implemented family yields a full-grid P1 homeomorphism. This is a parameterization-wide statement for this mesh/boundary, not a theorem for arbitrary imported meshes or nonconvex boundaries. In represented floating point, the positive-weight theorem alone does not certify the rounded returned coordinates: Phase VI additionally checks every face determinant and the fixed boundary. That is a checkpoint-specific PL certificate, whereas a small linear residual alone would not be one.

## 2. Positive edge updates and Woodbury without a dense inverse

Let \(c^0_e>0\) be a fixed fine-grid baseline with matrix \(A_0\) and full map \(Y^0\). Select \(k\) fine edges and add \(\delta_r>0\) to their conductances. Denote by \(u_r\in\mathbb R^{|I|}\) the restricted incidence column for selected edge \(r\): it has entries \(+1,-1\) at its interior endpoints, or one nonzero entry for an interior-boundary edge. Let \(U=[u_1,\ldots,u_k]\), \(D=\operatorname{diag}(\delta_1,\ldots,\delta_k)\), and let row \(r\) of \(d\in\mathbb R^{k\times2}\) be the **full** baseline edge difference \(Y^0_{a_r}-Y^0_{b_r}\), including the boundary coordinate when relevant.

Write \(Y_I=Y_I^0+\Delta\). The updated equilibrium equation is exactly

\[
A_0\Delta+UD\bigl(d+U^\mathsf T\Delta\bigr)=0.
\]

Set \(W=A_0^{-1}U\) and \(G=U^\mathsf TW\). Eliminating \(\Delta\) gives

\[
Y_I=Y_I^0-W(D^{-1}+G)^{-1}d.
\]

For numerical stability, the code instead solves the equivalent positive-definite system

\[
\bigl(I+D^{1/2}GD^{1/2}\bigr)\beta=D^{1/2}d,\qquad
Y_I=Y_I^0-WD^{1/2}\beta.
\]

The baseline sparse factorization computes \(W\) once, as a block of \(k\) solves. It does **not** materialize \(A_0^{-1}\). Each later latent-dependent call performs a \(k\times k\) solve and an \(|I|\times k\) matrix product. The output is still on the original \(N^2\)-vertex control mesh. Omitting the full-edge difference \(d\) would give a wrong boundary right-hand side for interior-boundary updates; tests deliberately include such an edge.

The cost is not free: stored \(W\) has \(|I|k\) scalars, setup includes a fine-grid sparse factorization, and the small dense solve grows cubically with \(k\). On \(N=257\), \(|I|=65{,}025\). At \(k=256\), float32 \(W\) alone uses about 66.6 MB before other tensors. This is not a universal fast solver for all conductance fields.

## 3. Why one coarse variable need not mean rank one

Suppose a single scalar latent changes *all* edge conductances inside one connected patch by positive factors. Although the parameter count is one, the matrix update is

\[
\Delta A=B_{\mathrm{patch},I}^\mathsf T
\operatorname{diag}(\delta_e)
B_{\mathrm{patch},I}.
\]

Its rank equals the rank of the patch's reduced incidence matrix because every \(\delta_e>0\). For a connected changed-edge component containing no fixed boundary vertex, incidence vectors have one constant-vector relation, so their rank is \(\#V_{\mathrm{component}}-1\). If the component touches a fixed boundary vertex, restricting to interior columns removes that constant null vector and the rank equals the number of participating interior vertices. Hence an interior \(16\times16\)-cell patch contains \(17^2=289\) graph vertices and produces rank 288 when its changed edges connect all of them, despite being controlled by one latent. A coarse latent changing a macroscopic area or all edges has rank growing with the fine-grid vertex count. Parameter dimension and matrix-update rank must not be conflated.

This rank result explains the measured Woodbury expression limit: the fast experiment altered only 16, 64, or 256 isolated selected edges out of 196,094 active edges on the \(257^2\) grid. It cannot stand in for a smooth coarse-to-fine change to every conductance. A coherent horizontal-row cut at the same rank was subsequently tested in Section 6; it remains a restricted family, though other useful low-rank support patterns are not ruled out.

## 4. Exact patch-interior Schur elimination

When many fine edges change, partition the interior unknowns as \(I=P\sqcup Q\). The set \(P=\bigsqcup_b P_b\) consists of vertices strictly inside disjoint \(p\times p\)-cell patches; \(Q\) contains the remaining interior interface vertices. With the chosen rectangular triangulation, no edge joins different \(P_b\), including across cell diagonals. Thus \(A_{PP}\) is block diagonal. Write

\[
\begin{bmatrix}A_{PP}&A_{PQ}\\A_{QP}&A_{QQ}\end{bmatrix}
\begin{bmatrix}Y_P\\Y_Q\end{bmatrix}
=\begin{bmatrix}b_P\\b_Q\end{bmatrix}.
\]

Eliminating each patch interior *exactly* yields

\[
S=A_{QQ}-A_{QP}A_{PP}^{-1}A_{PQ},\qquad
S Y_Q=b_Q-A_{QP}A_{PP}^{-1}b_P,
\]

then \(Y_P=A_{PP}^{-1}(b_P-A_{PQ}Y_Q)\). As a Schur complement of an SPD matrix, \(S\) is SPD. The code factors each patch block and the complete filled interface matrix; it does not truncate fill or stop an approximate multigrid iteration. Recovered \(Y_P,Y_Q\) together solve the *original* \(N^2\)-control-vertex system.

Let \(N=mp+1\). The original number of interior vertices is \((mp-1)^2\); patch interiors contain \(m^2(p-1)^2\) vertices; therefore

\[
|Q|=(mp-1)^2-m^2(p-1)^2.
\]

At \(N=257\), the interface sizes are 7,425 for \(p=16\), 3,521 for \(p=32\), and 1,521 for \(p=64\). A smaller interface does not automatically mean a faster solve: larger patch factors and Schur fill may dominate. The code records all three.

## 5. First-order implicit gradient

Let \(L(Y)\) be a scalar loss that can include image resampling. The exact solution satisfies \(A(c)Y_I=b(c)\). Define the interior adjoint by

\[
A(c)\lambda_I=\frac{\partial L}{\partial Y_I},
\]

using the same symmetric factors as the forward solve; set \(\lambda_b=0\) at fixed boundary vertices. Differentiating one edge's equilibrium contribution gives

\[
\frac{\partial L}{\partial c_{ab}}
=-(Y_a-Y_b)\cdot(\lambda_a-\lambda_b).
\]

This single formula covers interior-interior and interior-boundary edges. It includes the right-hand-side derivative for the latter because the *full* primal edge difference contains \(Y_b\). If \(c_e=c_{\min}+\operatorname{softplus}(z_e)\), then \(\partial L/\partial z_e=(\partial L/\partial c_e)\operatorname{sigmoid}(z_e)\). The encoder's coarse/fine interpolation gradients follow by the usual chain rule. The Schur implementation saves its factors for one backward pass and does not retain Krylov iteration trajectories. It currently supports first-order gradients on CPU float64; it is not a GPU-batched production layer.

## 6. Evidence at the dense scale

All dense Route C results below use a \(257\times257\) control mesh, 66,049 vertices, 131,072 triangles, and 196,094 active edges. The map-error workload and image-training workload are different and must not be merged.

| Method and workload | Precision/device | Forward | VJP | Other result |
|---|---|---:|---:|---|
| Woodbury, 16 isolated edges, batch 1, map loss | float32/AI A6000 | 1.18 ms repeated median | 1.42 ms | 5.65 s setup; full direct oracle max coordinate difference about \(3\times10^{-8}\) |
| Woodbury, 64 isolated edges, batch 2, map loss | float32/AI A6000 | 1.09 ms | 7.79 ms | 5.95 s setup; 59.0 MB peak CUDA allocation |
| Woodbury, 256 isolated edges, batch 2, map loss | float32/AI A6000 | 2.88 ms | 7.89 ms | 7.34 s setup; 108.3 MB peak CUDA allocation |
| Exact Schur, all edges, \(p=16\), batch 1, map loss | float64/Element CPU | 0.620 s | 0.059 s | separate complete direct forward 0.439 s; max difference \(3.45\times10^{-14}\) |
| Exact Schur, all edges, \(p=32\), batch 1, map loss | float64/Element CPU | 0.592 s | 0.060 s | separate complete direct forward 0.509 s |
| Exact Schur, all edges, \(p=64\), batch 1, map loss | float64/Element CPU | 0.868 s | 0.055 s | separate complete direct forward 0.515 s |

The previous matrix-free CG baseline did not converge in the backward solve at \(129^2\), float32 tolerance \(10^{-5}\), even after 3,000 iterations (final true relative residual approximately \(1.93\times10^{-5}\)); it raised an error. At \(257^2\), float64 tolerance \(10^{-8}\) similarly missed after 4,000 iterations (approximately \(1.48\times10^{-7}\)). Relaxing to \(10^{-6}\) allowed a forward/VJP call, but it took 15.92 s/2.43 s and is **not** an equal-accuracy comparison with the exact methods. CG may benefit from a suitable preconditioner, but a preconditioned approximate solve would still need a true-residual and geometry contract.

On the 512×512 high-frequency image-only task, batch 1, the all-edge exact Schur encoder reduced image MSE from 0.02610 to 0.01818 in 50 Adam steps; two latent-scale head gradients were nonzero, and the last full-system relative residual was at machine precision. The 256-edge Woodbury encoder reduced MSE only from 0.02614 to 0.02612 in 50 steps; its first head-gradient norms were about \(9\times10^{-7}\), consistent with weak influence. These are different update families, not a solver-only ablation. The 300-step all-edge run and same-quality wall-time comparison are analyzed separately in the final report.

At identical rank 256, we then compared 256 diagonal edges scattered over a 16×16 cell lattice with 256 diagonal edges crossing one horizontal row of fine cells. Both were tested on the original 257² mesh. The float64 CPU Woodbury outputs agreed with newly assembled independent full sparse solves to at most \(1.55\times10^{-14}\) per coordinate, with residuals below \(5\times10^{-15}\), and all represented faces positive in the untrained benchmark. One-time setup cost was about 13.1–13.3 s and the stored precomputed columns occupied 134.8 MB in float64. For the same fixed-pair 512² high-frequency image task, batch 1, 300 Adam updates on Element CPU, the measured values were:

| Updated-edge support | Initial image MSE | Final image MSE | Final query-map RMSE | Minimum observed face-area ratio | Median full forward/VJP |
|---|---:|---:|---:|---:|---:|
| scattered 16×16 lattice | 0.026145 | 0.025989 | 0.014399 | 0.00897 | 11.08/12.77 ms |
| one horizontal cut | 0.026137 | 0.025805 | 0.014344 | 0.00819 | 11.00/12.99 ms |

The cut gives only a small task-specific improvement; both barely reduce the image loss and approach very small face-area margins. Positive conductances provide the exact-arithmetic theorem, and the measured returned faces were positive, but this is poor *trained* deformation quality for the target, not a success hidden by fast algebra. The two update patterns cover different edges, so even their small initial-loss difference matters. The all-edge Schur model's previously measured 300-step image MSE was 0.008438 on this same fixed-pair family, at far higher wall time. These are **different representation families**, not an isolated solver speed comparison. Raw records, including sample trajectories, are in `raw_results/routeC_{cut,lattice}_image257_cpu300.json` and `raw_results/routeC_cut_vs_lattice257_cpu.json`.

## 7. Current conclusion, not a stop rule

Woodbury has a compelling repeated-call speed when a small set of **physically useful** fine edges changes and its one-time setup is amortized. The currently selected isolated-edge lattice is too restricted for the image task. Exact Schur supports arbitrary positive all-edge conductances and a correct implicit VJP at 257², but its CPU factorization is presently slower than a fresh complete direct solve and far slower per step than the explicit two-layer candidate. This does not justify abandoning the conductance route: structured cutsets, batched GPU patch factors, or an exact reusable-interface formulation deserve further targeted tests. Any subsequent acceleration must be compared at matching map quality, not only matching algebraic residual.
