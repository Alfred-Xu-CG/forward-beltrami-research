# A solve-free multiscale latent layer with a fixed-grid P1 topology theorem

This note introduces a Phase VI candidate, called A2 here, whose output is a map on the **original** \(N\times N\) rectangular vertex grid and its standard southwest-to-northeast diagonal triangulation. It is not an arbitrary vertex displacement followed by a topology-preserving line search. Latents choose positions of *new* edge vertices in a recursive convex subdivision; all old vertices remain where the previous level placed them. The construction itself determines every new cell center. The hypothesis and conclusion below are in exact arithmetic, followed by a separate represented-float contract.

## 1. Grid, orientation, and one subdivision

The domain is \(\Omega=[0,1]^2\). At level \(k\), \(n_k=2^k+1\) source grid vertices lie at \((i/2^k,j/2^k)\), \(0\le i,j\le2^k\). Its cells are closed axis-aligned squares. The output vertices \(Q^k_{ij}\in\Omega\) are arranged so that every image cell

\[
C^k_{ij}=
[Q^k_{ij},Q^k_{i+1,j},Q^k_{i+1,j+1},Q^k_{i,j+1}]
\]

is a *strictly convex* quadrilateral in counterclockwise order. Neighboring cells share their full image edge. Initially \(n_0=2\) and the four output vertices are the four corners of the unit square.

Choose one split fraction \(t_e\in(0,1)\) for each **global** current-grid edge \(e\); use the same \(t_e\) from both incident cells. At the boundary, set \(t_e=1/2\). A latent logit \(z_e\) on an interior edge produces

\[
t_e=\tfrac12+s\tanh z_e,\qquad 0<s<\tfrac12.
\]

The implementation uses \(s=1/4\), hence even a saturated finite logit leaves the represented intended fraction between \(1/4\) and \(3/4\), before rounding.

Consider a cell with ordered image corners \(q_{00}\) (bottom-left), \(q_{10}\) (bottom-right), \(q_{11}\) (top-right), and \(q_{01}\) (top-left). Their four edge points are \(B\) (bottom), \(R\) (right), \(T\) (top), and \(L\) (left). Join \(B\) to \(T\), and \(L\) to \(R\). Because the endpoints alternate in cyclic order on a strictly convex quadrilateral, these chords cross once in the cell interior. Write \(d_1=T-B\), \(d_2=R-L\) and the oriented planar cross product \([a,b]=a_xb_y-a_yb_x\). Their common point is

\[
u=\frac{[L-B,d_2]}{[d_1,d_2]},\qquad
C=B+u\,d_1,\qquad 0<u<1.
\]

The denominator cannot vanish in exact arithmetic: the chords cross transversely. One **shared** tensor entry stores each global edge point, and one center entry is created per old cell. The new \(n_{k+1}=2n_k-1\) vertex grid places old vertices at even-even indices, horizontal edge points at even-odd indices, vertical edge points at odd-even indices, and centers at odd-odd indices.

The two crossing chords partition the convex parent cell into four child quadrilaterals:
\[
[q_{00},B,C,L],\quad[B,q_{10},R,C],\quad
[C,R,q_{11},T],\quad[L,C,T,q_{01}].
\]
Each child is an intersection of the convex parent with two chord-bounded halfplanes, so it is strictly convex and has nonzero area. Their interiors are disjoint and their union is the parent. This holds for **all** interior edge fractions, not just small perturbations.

## 2. Global P1 homeomorphism theorem

Inductively assume level \(k\) cell images form a conforming convex quadrilateral tiling of the square and the boundary is fixed pointwise. Adjacent cells use the same edge-split vertex, so the level \(k+1\) child tiling is conforming. The four children partition each parent without overlap; hence the new cells partition the same square. Boundary edges are split at their exact source midpoint, so every new boundary vertex retains its source coordinate. Induction starts from the identity square.

Triangulate **every final source cell** along the fixed southwest-to-northeast diagonal, and use the corresponding diagonal in its convex image quadrilateral. Both image triangles have strictly positive orientation. Adjacent affine pieces agree on common edges because they have shared vertices. The resulting continuous PL map sends a conforming source triangulation bijectively to a conforming image triangulation of \(\Omega\), with inverse given by the inverse affine map on each image face. Thus the final P1 map is a homeomorphism of \(\Omega\) onto itself. For \(k=8\), the final control grid has \(N=257\), 66,049 vertices and 131,072 P1 triangles. No global linear system is solved.

The theorem does **not** assert that every homeomorphism on this grid is representable. Every dyadic cell image remains convex; old level vertices and old subdivision chords cannot later bend. This is a real expressivity restriction to measure, not a numerical defect. The final output is P1 on the canonical triangulation, unlike an exact composition of P1 maps, whose full partition generally refines that triangulation.

## 3. Differentiability, cost, and numerical scope

For finite logits away from a degenerate represented cell, edge interpolation and the rational chord-intersection formula are differentiable. Reverse-mode automatic differentiation gives a VJP through the hierarchy; no sparse adjoint or stored iterative-solver trajectory is needed. At level \(k\), the number of cells and edge latents is \(O(4^k)\). Summing geometric levels through \(N=2^L+1\) gives \(O(N^2)\) arithmetic and \(O(N^2)\) saved activations for a first-order backward pass, up to tensor-implementation constants. There are seven non-root learnable split levels at \(N=257\); the initial \(2\to3\) subdivision is deterministic because all four root edges are boundary.

Exact-arithmetic convexity is not by itself a finite-precision certificate. A cell can become poorly conditioned after many subdivisions; the chord denominator and final triangle areas may be small. The default implementation checks finite coordinates, checks that all represented coordinates lie in the unit square, checks every final P1 triangle's **represented** signed area in float64 with an absolute roundoff margin \(64\epsilon_{64}\) at that explicitly checked coordinate scale, and checks exact dyadic boundary values. It raises rather than returning a failed map. Strictly positive exact face orientations plus fixed simple boundary give a global PL certificate for that realized output; the double-precision margin is a numerical screen for those signs, not a formally verified adaptive-exact predicate. The check does not claim an interval-arithmetic proof of the latent-to-float computation, and backpropagation through a rejected output is undefined. float32 and float64 margin/gradient measurements are therefore required at 257².

## 4. Free-center extension and its stronger expression set

The chord intersection is sufficient for convexity, but not necessary. In a convex parent, the four edge points \(B,R,T,L\) in cyclic order form a strictly convex *inner* quadrilateral \(K\). Let the center be **any** \(C\in\operatorname{int}K\). Each of the four children in Section 1 is still strictly convex. For example, the bottom-left child \([q_{00},B,C,L]\) has positive turns at its inherited corner and two edge points because \(C\) lies inside the convex parent, while its turn at \(C\) is positive because \(C\) lies to the left of the inner edge \(L\to B\). Cyclic rotation proves the other three. Their boundaries remain nonintersecting segments from one interior point to four cyclic boundary points, so they tile the parent. This removes the chord-collinearity restriction while retaining the same final fixed-grid P1 theorem.

One convenient differentiable interior parameterization uses two fractions \(u,v\in(0,1)\):

\[
C=(1-v)\bigl((1-u)B+uR\bigr)
  +v\bigl((1-u)L+uT\bigr).
\]

All four coefficients of \(B,R,T,L\) are strictly positive and sum to one, so \(C\in\operatorname{int}K\). The implementation uses \(u=1/2+s_c\tanh z_u\), \(v=1/2+s_c\tanh z_v\), \(s_c=0.35\). This adds two center logits per current cell, including the root cell; the root center may now move despite the boundary remaining fixed. The updated arithmetic avoids chord division and remains \(O(N^2)\). It is a distinct tested variant, not a post-hoc topology repair. The edge split and numerical certification conditions remain unchanged.

## 5. Relation to prior work and experimental obligations

Subdivision of convex quadrilateral meshes is established geometric practice; see, for example, [Li and Zhang, *Optimal Quadrilateral Finite Elements on Polygonal Domains* (2017)](https://doi.org/10.1007/s10915-016-0242-5), which studies convex shape-regular quadrilateral refinement for finite elements. [Liu et al., *Neural Subdivision* (2020)](https://www.dgp.toronto.edu/projects/neural-subdivision/) predicts refined mesh geometry using a network, but addresses a different surface-modeling task and does not establish this fixed-grid planar homeomorphism layer. We do not claim novelty from a targeted search.

The decisive Phase VI tests are: asymmetric one-cell geometry; random and saturated latent topology at 5², 17², 257²; central-difference VJP; direct-latent fit of the independent high-frequency target; full 257²/512² image-to-latent training; and fair comparison with A, AB2, and positive-conductance Route C. A construction theorem alone is not a performance or accuracy result.

## 6. First numerical evidence and its limits

The focused test suite exercised both center rules at 5², 17² and 257² with two random float32 batches, one extreme float64 chord-center batch, a moving root center, and central-difference VJPs. All ten focused tests passed. At 257², 1000 Adam steps fitting the same independent high-frequency analytic map at **control vertices** gave:

| Variant | Learnable scalar latents | Vertex map RMSE | Maximum point error | Smallest source-normalized face area | CPU train wall |
|---|---:|---:|---:|---:|---:|
| Chord intersection center | 44,196 | 0.005396 | 0.05848 | 0.00130 | 31.0 s |
| Free bilinear center | 87,886 | 0.000496 | 0.01178 | 0.09190 | 40.3 s |

Both rows use the same local Windows CPU, float32 and Adam learning rate 0.05. The existing one-axis A direct fit had RMSE about 0.01062; exact AB2 composition reached about \(6.9\times10^{-6}\) on this map but is not P1 on the same original triangulation. The direct fit uses target-map supervision and is **not** image-to-latent evidence. Its nonzero maximum error suggests remaining coarse-cell expressivity constraints or optimization effects; the table alone does not identify which.

A preliminary 65²-control/128²-image held-out experiment (16 training, 4 test textures/maps, 300 image-only updates) with the free-center layer reduced test image MSE from 0.03472 to 0.00543 and test query-map RMSE to 0.00656. It ran on a different local CPU than the earlier A/AB2 medium tests, so its timing is not a fair speed ranking; the later common-host 257² result is reported next.

## 7. First image-to-latent result on the required control scale

The \(257^2\)-control, \(512^2\)-image, 32-train/8-held-out experiment was then run for all three explicit candidates on **the same** idle Element L40 GPU 1, float32, batch 2, 1000 Adam updates, image-only loss, the same synthetic samples and seed schedule. All networks use an eight-channel two-convolution body; their prediction heads and latent organizations differ. The decoder topology contracts also differ as stated earlier. The measured results are:

| Layer | Encoder parameters | Held-out image MSE | Held-out query-map RMSE | Training wall | Median forward/VJP | Peak allocated CUDA | Minimum represented face-area ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| A, one-axis fixed P1 | 754 | 0.01273 | 0.01260 | 4.09 s | 1.04/2.25 ms | 251 MB | 0.196 |
| AB2, two-factor exact PL | 772 | 0.01188 | 0.01215 | 7.37 s | 2.24/3.67 ms | 336 MB | 0.174 and 0.694 per factor |
| A2+, free-center fixed P1 | 1006 | **0.002387** | **0.004196** | 17.13 s | 4.75/10.70 ms | 269 MB | 0.0486 |

Forward includes encoder, decoder, dense query evaluation, image resampling and image loss; VJP is the complete first-order backward. Peak CUDA allocation includes resident training data but not GPU driver/context memory. One-time dataset generation and query-table setup are excluded from training wall. A2+ gains substantial quality after an equal update count but is slower per step. It is **not** a decoder-only ablation: the A2+ encoder has multilevel heads at every dyadic scale, whereas A and AB2 use coarse/fine heads. Its better result must therefore be attributed to the complete encoder–decoder design until a common-head ablation is performed. The directly optimized latent oracle above gives a separate, more limited expression comparison.

We also checked the saved **image-trained** A2+ model's actual P1 face geometry against the analytic target on all 131,072 source triangles of each of eight held-out maps. Every source face has equal area, so the source-area-weighted Beltrami RMSE is the square root of the arithmetic mean of \(|\mu_{\rm predicted}-\mu_\star|^2\) over samples and faces. It was **0.1009**; maximum predicted \(|\mu|\) was **0.9115**, compared with target maximum **0.3155**. The minimum face Jacobian determinant was 0.04855. Thus the output remained nonfolded and accurately aligned images, but its QC distortion was much worse than the target. This is an important non-success on the geometry axis.

We reran the identical image-only schedule with a target-independent edge-strain penalty, the mean squared deviation of horizontal and vertical image-edge vectors from the undeformed grid edges. It does not use the target map or target \(\mu\), so the analytic geometry remains held-out evaluation only. Each row uses 1000 steps, batch 2 and the same Element L40 GPU 1; \(\lambda=0\) is the preceding baseline.

| Edge-strain weight \(\lambda\) | Held-out image MSE | Query-map RMSE | Face Beltrami RMSE | Maximum predicted \(|\mu|\) | Minimum face Jacobian determinant | Training wall |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.002387 | 0.004196 | 0.10092 | 0.91152 | 0.04855 | 17.13 s |
| 0.001 | 0.002398 | 0.004192 | 0.08622 | 0.82138 | 0.10269 | 17.35 s |
| 0.01 | 0.002461 | 0.004206 | 0.07141 | 0.52922 | 0.32911 | 17.55 s |

This improves the observed distortion margin with a modest image-error increase, but it neither recovers the target Beltrami field exactly nor proves all future samples will have a given \(|\mu|\) bound. The hard homeomorphism argument comes from the decoder, not from this loss term. The saved states and raw evaluation records are in `checkpoints/` and `raw_results/`.

An independent checker separately reviewed the exact convex-child and fixed-grid P1 arguments, searched 50,000 random convex-parent/split/center cases without a counterexample, and compared a 257² directional VJP to central differences (relative discrepancy about \(6.7\times10^{-8}\)). It also identified that the standalone numerical checker relied on a unit coordinate scale without testing that condition. The checker now explicitly rejects coordinates outside the unit square; its test suite includes that failure case. These numerical checks support implementation correctness but are not substitutes for the exact subdivision proof.

## 8. Shared-head ablation: decoder benefit versus encoder capacity

The original A2+ encoder predicts independent edge/center logits at every dyadic level. A controlled ablation ties its three small 1×1 convolution heads across all seven non-root levels, retaining the identical two-convolution image body, root head, A2+ decoder, dataset, GPU, optimizer and 1000-step schedule. Parameters drop from 1006 to 790, close to AB2's 772. The shared heads are evaluated on each level's downsampled image features, so the latent tensors retain the same shapes and the fixed-grid P1 theorem is unchanged.

| A2+ heads | Encoder parameters | Held-out image MSE | Query-map RMSE | Face Beltrami RMSE | Maximum \(|\mu|\) | Minimum face area ratio | Train wall | Median full forward/VJP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent per level | 1006 | 0.002387 | 0.004196 | 0.10092 | 0.91152 | 0.04855 | 17.13 s | 4.75/10.70 ms |
| shared across levels | 790 | 0.004099 | 0.005320 | 0.08156 | 0.37289 | 0.48932 | 16.85 s | 4.80/10.63 ms |

Thus independent heads improve image and map accuracy, but on this split they also permit much worse local distortion. The shared-head A2+ still substantially outperforms the 772-parameter AB2 encoder–decoder's 0.01188 held-out image MSE in the equal-update run. This supports a contribution from the decoder's cross-coupled fixed-grid representation, but is not a fully controlled decoder-only comparison: AB2 still uses a different latent geometry and network head layout. All \(\mu\) values here are computed from the **actual A2+ P1 face Jacobians** against the analytic target at source-face centroids; the target's image pixel field was not used to supervise \(\mu\).

## 9. Near-control-scale detail is not recovered by the current image encoder

A second independent target family changes the fine term from \(\sin(16\pi x)\sin(16\pi y)\) to \(\sin(64\pi x)\sin(64\pi y)\): 32 periods across the square, eight 257² control cells per period. The low- and fine-term amplitudes are reduced so the *continuous* displacement's global Lipschitz bound is below 0.895, hence the boundary-fixed target is a homeomorphism. We separately checked the sampled target P1 face signs at coefficient-range corners; the continuous guarantee alone was not used as a P1 claim. The same texture generator, 32/8 train/test split, 512² image queries, batch 2, Adam rate 0.003 and 1000 image-only updates were used for A, AB2 and A2+ on the same Element CPU, float32.

| Method | Held-out image MSE | Held-out query-map RMSE | Median full forward/VJP | Training wall | Smallest reported factor face-area ratio |
|---|---:|---:|---:|---:|---:|
| A, one-axis fixed P1 | 0.003990 | 0.006271 | 15.25/21.75 ms | 38.57 s | 0.343 |
| AB2, exact two-factor PL | 0.004100 | 0.006424 | 33.90/44.71 ms | 80.07 s | 0.576, 0.849 |
| A2+, fixed-grid P1 | **0.000613** | **0.002022** | 23.47/35.16 ms | 60.48 s | 0.777 |

The image number alone is misleading: the image-trained A2+ face-Beltrami RMSE is 0.1395 against the analytic target, whereas a map made by P1-interpolating the **true target vertices** already has a 0.0383 RMSE on these faces. The predicted maximum \(|\mu|\) is only 0.117 while the target reaches 0.462. Projecting each predicted displacement onto the known 32-cycle sine product gives x/y coefficients of order \(10^{-7}\), compared with true sample coefficients of order \(10^{-3}\). Thus this model nearly omits the fine oscillation. A diagnostic using **true** low-frequency coefficients but omitting the fine term has image MSE \(6.65\times10^{-5}\), much lower than the trained A2+'s \(6.13\times10^{-4}\): the issue is not just an intrinsically invisible fine image signal; the current amortized training has not reached even the known low-only optimum. That diagnostic is evaluation-only and is not a registration method.

To separate decoder expression from image inference, we directly optimized A2+ latent tensors against one independent high32 target's **vertex coordinates**. This is an oracle that cannot be compared as image-only training. After 1000 Adam steps it reached vertex-map RMSE 0.000444 and projected fine amplitudes about 0.00178 in x/y versus the target 0.0025. Warm-starting the latent values for another 2000 steps at a lower rate reduced vertex-map RMSE to 0.000356 and raised the projected amplitude to about 0.00199. The corresponding analytic face-Beltrami RMSE was 0.130 and then 0.122, while the *sampled target P1* discretization floor for this worst-case single target was 0.0703. The experiment shows meaningful fine representational capacity, but not exact reproduction or a fast inverse from images. Its optimizer may still be limiting; the data do not prove an absolute architecture lower bound.

For comparison, an AB2 direct-latent oracle on the same 257² vertex targets improved from RMSE 0.000883 after 1000 steps to 0.000205 after 2000 additional warm-started steps; its maximum vertex error fell to 0.000883. That is a more accurate query-value fit than A2+'s 0.000356 RMSE and 0.00608 maximum error, but AB2 remains an **exact two-factor PL composition**, not a P1 map on the original regular triangulation. The image-trained AB2 above still failed to realize this oracle advantage, making representation and amortized inference separate questions.

A final diagnostic removes the encoder entirely: for one held-out high32 image pair whose true fine amplitude is 0.002096, initialize all 87,886 A2+ latents at zero and optimize **only image MSE**, never the map or \(\mu\). After 1000 Adam steps, image MSE fell from 0.01213 to \(1.18\times10^{-5}\); the projected x/y fine amplitudes were 0.00124/0.00157. Thus the image data and A2+ latent optimization can indeed drive some fine recovery, whereas the amortized encoder's projection remained near zero. But the image-only direct solution had query-map RMSE 0.00116, face-Beltrami RMSE 0.202 versus the sampled-target P1 floor 0.0594, maximum \(|\mu|=0.934\), and minimum area ratio 0.0268. This is a clear image-to-geometry ambiguity: excellent registration does not imply accurate QC geometry, even within a hard-homeomorphic family. It also prevents claiming that direct per-pair latent optimization is a satisfactory neural layer; its 90.5 s per-pair training wall is a different workflow and much slower than one amortized inference call.

### Direct high32 image fit with a geometry prior

We reran that one-pair, zero-initialized, 1000-update direct-latent experiment on the **same Element CPU**, with only the target-independent fine-edge strain weight \(\lambda\) changed. For each horizontal or vertical control edge, multiply its mapped edge vector by 256, subtract the corresponding identity unit vector, square its Euclidean norm, and average equally across horizontal and vertical edges to obtain \(E_{\rm strain}\). The optimized objective is image MSE plus \(\lambda E_{\rm strain}\); the target map and target Beltrami coefficient are used only after training. All rows use float32, 257² control vertices, 131,072 triangles, 512² image queries, Adam rate 0.05, and held-out pair index one with true projected fine amplitude 0.002096 in each output coordinate.

| \(\lambda\) | Image MSE | Query-map RMSE | Face-\(\mu\) RMSE | Maximum predicted \(|\mu|\) | Minimum normalized face area | Fine x/y projection | 1000-step wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00000310 | 0.001115 | 0.20157 | 0.9340 | 0.0269 | 0.001238 / 0.001567 | 18.1 s |
| 0.0001 | 0.00000231 | 0.000815 | 0.16847 | 0.9193 | 0.0220 | 0.001026 / 0.001293 | 19.5 s |
| 0.001 | 0.00001327 | 0.000860 | 0.17200 | 0.8661 | 0.0473 | 0.000691 / 0.000884 | 19.4 s |
| 0.01 | 0.00009112 | 0.001001 | 0.19698 | 0.4486 | 0.4283 | 0.000226 / 0.000300 | 19.4 s |

The analytic face-\(\mu\) RMSE floor of the sampled true target P1 map for this pair is 0.05944, which no row approaches. Stronger strain suppression raises the minimum area and lowers maximum distortion but erases the very high-frequency component sought here; a weak weight can improve map error yet leaves near-folding faces. This simple prior is therefore **not** an adequate joint solution to high-frequency accuracy and controlled quasiconformal geometry. The different wall time from the earlier local-CPU baseline reflects a different host; comparisons *within this table* are the controlled ones. Reproduction JSON and latent checkpoints have matching `a2_imageonly_high32_pair1_strain*_cpu1000` names under `raw_results/` and `checkpoints/`.

### Exact-representation obstruction and control-resolution ablation

The A2+ theorem makes each **dyadic cell image convex** and puts every newly split edge vertex on the *straight segment* joining its two parent-edge endpoint images. These are necessary conditions for exact reproduction of target control vertices. We evaluated eight independently generated held-out analytic targets in float64 on the 257² control grid. Every tested dyadic cell in both base and high32 families was strictly convex; the minimum normalized corner cross product over all 256² finest cells was 0.5921 for base and 0.3506 for high32. Thus convexity itself is not the observed obstruction for this family.

Straight-edge alignment does fail. For every dyadic parent edge with target endpoint images \(a,b\), let \(m\) be its target midpoint image and compute \(\delta=\operatorname{dist}(m,[a,b])\). The maximum \(\delta\) at the 4-cells-per-side level was 0.01170 for base and 0.005102 for high32; at 128 cells per side it was 0.0000689 and 0.0005969 respectively. These are coordinate distances in the unit square, not pixel distances. If a candidate A2+ map has Euclidean error at most \(\varepsilon\) at each of \(a,m,b\), its exact segment constraint implies \(\delta\le2\varepsilon\) by the triangle inequality. Hence one high32 target in this held-out set necessarily has maximum control-vertex error at least 0.002551 under **any single A2+ layer**; this is not a lower bound on global RMSE or on exact PL compositions. The full per-scale audit, including positive target P1 face signs, is in `raw_results/target_quad_edge_audit257_{base,high32}.json`. It identifies a genuine structural approximation limit in addition to the encoder failure.

To test whether simply doubling control resolution addresses the high32 encoder failure, we retained the 512² images, 32/8 split, batch two, 1000 Adam image-only updates, and eight-channel encoder, changing A2+ control side from 257 to 513. On the same Element CPU, the 513² run had 263,169 control vertices and 524,288 triangles; held-out image MSE was 0.0006084 and query-map RMSE 0.002013, versus 0.000613 and 0.002022 at 257². Its face-Beltrami RMSE was 0.13943 versus 0.1395 at 257², while the sampled-target P1 discretization floor improved from 0.0383 to 0.01925. The maximum predicted \(|\mu|\) was 0.1152, minimum normalized face area 0.7886, and all eight predicted 32-cycle x/y projections remained around \(10^{-8}\) to \(10^{-7}\), despite true amplitudes between about \(-0.00111\) and 0.00210. Training took 131.8 s; median end-to-end sampled-step forward/VJP were 0.0500/0.0812 s, with sampled process RSS 968 MB. Thus more *control* vertices alone did not recover the target detail; this is a one-run ablation, not a universal resolution result. The checkpoint is `checkpoints/a2_high32_heldout513_cpu1000.pt`, and its face-level evaluation is `raw_results/a2_high32_heldout513_cpu1000_face_eval.json`.

We also tested whether the 257² high32 encoder simply needed longer training. Loading its 1000-step checkpoint and running 3000 further image-only Adam updates at rate 0.001 reduced held-out image MSE from 0.0006132 to 0.0004641 and query-map RMSE from 0.0020225 to 0.0018597. However, face-Beltrami RMSE worsened from about 0.1395 to 0.1601, maximum predicted \(|\mu|\) rose to 0.9060, and minimum normalized area fell to 0.0591. The largest x/y 32-cycle projections across eight held-out items were only about \(4.3\times10^{-7}\) and \(6.2\times10^{-6}\), versus true amplitudes up to 0.00210. Longer optimization of this unchanged local encoder did not fix its high-frequency failure. The continuation required 183.0 s on Element CPU, with sampled median forward/VJP 0.0237/0.0351 s and process RSS 816 MB; see `raw_results/a2_high32_cpu_warm3000_train_summary.json` and `raw_results/a2_high32_cpu_warm3000_face_eval.json`.

Giving the encoder three pooled convolutional context scales, while leaving the A2+ decoder, image-only loss, 257² controls and 1000-step protocol unchanged, also did not help. Its 3342 parameters versus the original 1006 produced held-out image MSE 0.0006205 and map RMSE 0.0020316, compared with 0.0006132 and 0.0020225 for the local body. Face-Beltrami RMSE remained about 0.1393 and all predicted 32-cycle amplitudes remained below \(4.0\times10^{-8}\). Its median training-step forward/VJP were 0.0283/0.0404 s, sampled process RSS 858 MB, and 1000 steps took 71.6 s on Element CPU. Merely increasing local image context is therefore not a sufficient explanation or fix; explicit displacement matching, positional information, objective identifiability and/or optimization may matter. This negative result is in `raw_results/a2_high32_context_cpu1000_{train_summary,face_eval}.json`.
