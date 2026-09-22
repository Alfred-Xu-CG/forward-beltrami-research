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

The all-edge Schur model was also trained on the **first item** of the 32-sample high32 training set, using 257² control vertices, 512² image pixels, CPU float64 solve, batch 1 and 300 image-only Adam steps. This is a *one-pair training* diagnostic, not the eight-item held-out evaluation used for A/AB2/A2. Image MSE fell from 0.004186 to 0.001778; query-map RMSE was 0.004604, and the smallest represented face-area ratio was 0.0811. On source-face centroids its Beltrami RMSE against the analytic target was 0.1594, while the sampled-target P1 discretization floor was 0.0212. The observed maximum predicted \(|\mu|=0.649\), versus 0.167 for that target item. A representative full forward was about 0.57 s; peak observed process RSS was 1.69 GB. Thus the all-edge family is more expressive than the rank-256 sparse-edge families on the earlier fixed-pair target, but this implementation is much slower than the explicit candidates and still gave poor QC recovery on high32 after 300 steps. The exact full-system residual and face checks were retained. Raw data: `raw_results/trainpair_C_high32_schur257_512_element_cpu300.json`.

## 7. Reusable-interface candidate under test

An exact restricted-family alternative fixes every edge touching a strict patch-interior vertex and learns only edges whose interior endpoints lie on the patch interface (boundary endpoints are permitted). Then the patch-interior matrix \(A_{PP}\), its sparse factors and its coupling \(A_{PQ}\) are independent of the latent. Let \(S_0=A_{QQ}-A_{QP}A_{PP}^{-1}A_{PQ}\) for the unit-conductance baseline, let \(U_Q\) contain the reduced incidence columns of the selected interface edges, let \(D=\operatorname{diag}(c-c^0)\), and let \(d^0\) contain their *full* baseline coordinate differences. Because no selected edge touches \(P\), the exact update satisfies

\[
(S_0+U_Q D U_Q^\mathsf T)\Delta Y_Q=-U_Q D d^0,
\qquad \Delta Y_{P_b}=-A_{P_bP_b}^{-1}A_{P_bQ}\Delta Y_Q.
\]

The update entries of \(D\) may be negative when a positive learned conductance drops below baseline one; the full graph and hence the reduced Schur matrix remain SPD because *all actual* conductances stay positive. This is not the small-rank Woodbury formula. Each forward factors the current complete interface Schur matrix, but reuses all patch factors and inverse couplings; each backward uses that current interface factor and the fixed patch factors for the exact adjoint. The output remains the original fine-grid P1 map under the same positive-weight/boundary theorem and represented face checks. A 17² independent full sparse solve agreed to \(2.3\times10^{-15}\), the full relative residual was below \(10^{-12}\), and a directional VJP matched central differences within \(10^{-7}\). The 257² speed/expressivity measurement is pending; the method should not be called an all-edge fast solver because it cannot vary patch-interior edges.

The 257² CPU float64, batch-one repeated map-loss benchmark is now available. Every row uses the same 66,049 vertices, 131,072 faces and 196,094 active base edges; `variable edges` counts the subset the neural latent may change. The full sparse-direct comparator independently reassembles and refactors the complete system. Times are repeated medians after construction; the one-time precomputation is separate. Results are from the same Element host, although its shared load can vary.

| Patch width in cells | Variable edges | Interface vertices | One-time precompute | Forward | VJP | Independent full direct forward | Max coordinate discrepancy | Peak process RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 8190 | 7425 | 12.23 s | 0.2205 s | 0.0266 s | 0.4410 s | \(5.14\times10^{-14}\) | 800 MB |
| 32 | 3710 | 3521 | 12.22 s | 0.1835 s | 0.0196 s | 0.5123 s | \(6.62\times10^{-14}\) | 816 MB |
| 64 | 1566 | 1521 | 12.46 s | 0.0852 s | 0.0216 s | 0.4619 s | \(6.68\times10^{-14}\) | 921 MB |

The full relative residuals were \(5.9\)–\(8.4\times10^{-15}\); every benchmark map had positive represented face areas. At p64, interface factorization took about 54 ms and recovery plus full residual/face validation about 30 ms. This is a real exact-solve repeated-call speed improvement over full direct and the all-edge Schur prototype, **conditional on accepting interface-only conductance variation** and amortizing the 12.5 s setup. It remains slower than A2+'s full CPU image forward on the common task, and the control family is smaller.

On the same **one training pair** of high32 images used for the all-edge 300-step experiment above, p16/p64 interface models gave:

| Conductance family | Variable edges | Initial image MSE | Final image MSE | Query-map RMSE | Face-Beltrami RMSE | Smallest face-area ratio |
|---|---:|---:|---:|---:|---:|---:|
| all 196,094 active edges, refactor every step | 196,094 | 0.004186 | 0.001778 | 0.004604 | 0.1594 | 0.0811 |
| p16 reusable interface | 8190 | 0.004196 | 0.001997 | 0.004543 | 0.1539 | 0.0703 |
| p64 reusable interface | 1566 | 0.004196 | 0.002416 | 0.004684 | 0.1145 | 0.2165 |

This target's sampled-P1 Beltrami floor was 0.0212, so none of the trained C variants came close. p64 trades away image fit for repeated-call speed; the lower *mean* \(\mu\) error does not mean uniformly gentle geometry (its maximum observed \(|\mu|\) was 0.740, versus target 0.167). The models have different numbers and locations of learnable conductances and slightly different initial losses; the table is a family comparison, not an isolated solver ablation. The experiment does establish that exact hierarchical elimination can yield a forward/VJP advantage when the update support is structurally restricted, without interpolating a coarse map or weakening the fine-grid topology theorem. Raw benchmark and training records are in `raw_results/routeC_reusable_interface257_p*.json` and `raw_results/trainpair_C_high32_interface257_p*_cpu300.json`.

## 8. Current conclusion, not a stop rule

Woodbury has a compelling repeated-call speed when a small set of **physically useful** fine edges changes and its one-time setup is amortized. The currently selected isolated-edge lattice is too restricted for the image task. Exact Schur supports arbitrary positive all-edge conductances and a correct implicit VJP at 257², but its CPU factorization is presently slower than a fresh complete direct solve and far slower per step than the explicit two-layer candidate. The reusable-interface variant improves repeated-call speed for a restricted family; it does not resolve the all-edge cost. This does not justify abandoning the conductance route: structured cutsets or batched GPU patch factors deserve further targeted tests. Any subsequent acceleration must be compared at matching map quality, not only matching algebraic residual.

## 9. Fixed-patch harmonic representation limit

The reusable-interface family also has an exact **representation obstruction**. At every strict patch-interior vertex \(i\), all incident conductances remain one, so any output \(Y\) obeys \(\sum_{j\sim i}(Y_i-Y_j)=0\), regardless of the learned interface weights. For sampled target vertices \(F_{\star,i}\), put \(r_i=\sum_{j\sim i}(F_{\star,i}-F_{\star,j})\) and let \(d_i\) be graph degree. If an output had Euclidean vertex error everywhere at most \(\varepsilon\), then \(\|r_i\|_2\le2d_i\varepsilon\); hence \(\varepsilon\ge\max_{i\in P}\|r_i\|_2/(2d_i)\). This is a strict exact-matching obstruction but **not** a lower bound on mean image error or global coordinate RMSE. An independent complete-mesh edge enumeration found 57,600 strict patch-interior vertices for p16 and 63,504 for p64 at 257² controls. Across eight held-out base targets the maximum-error lower bound ranged from 0.0000205 to 0.0000566; across eight high32 targets it ranged from 0.000218 to 0.000541. The p16 and p64 maxima happened to coincide because the maximally violating vertices lie in both strict-interior sets, not because the families have identical expressive power. These are weak lower bounds compared with observed C image-trained map errors, but they prove that merely improving the interface linear algebra cannot exactly reproduce this high32 family. Raw values and the independent edge construction are in `raw_results/interface_harmonic_obstruction257_p{16,64}_{base,high32}.json` and `tools/phase6_interface_harmonic_obstruction.py`.
