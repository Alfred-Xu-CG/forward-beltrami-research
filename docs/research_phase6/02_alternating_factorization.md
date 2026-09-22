# Two triangular homeomorphisms as a dense forward layer

This note states the mathematical contract behind the Phase VI two-layer candidate. It distinguishes an exact continuous factorization, its fixed-grid piecewise-affine (P1) approximation, and a sampled image grid. It does **not** claim that every quasiconformal map is representable by two layers or that a sampled composition is automatically P1-bijective on the original triangulation.

## 1. Definitions and notation

Let $Q=[0,1]^2$. Write a map $F:Q\to Q$ as $F=(F_1,F_2)$, with $x$ horizontal and $y$ vertical. A *homeomorphism* is a continuous bijection whose inverse is continuous. On a compact domain such as $Q$, a continuous bijection onto $Q$ is automatically a homeomorphism. An orientation-preserving $C^1$ diffeomorphism has an invertible derivative $DF$, with determinant $\det DF>0$, and a continuously differentiable inverse.

A vertical triangular map has the form

\[
V(x,y)=(X(x),v(x,y)),
\]

where $X$ is strictly increasing from 0 to 1 and, for each fixed $x$, $y\mapsto v(x,y)$ is strictly increasing from 0 to 1. A horizontal triangular map has the form

\[
H(u,v)=(h(u,v),Y(v)),
\]

where $Y$ is strictly increasing from 0 to 1 and, for each fixed $v$, $u\mapsto h(u,v)$ is strictly increasing from 0 to 1. Both are homeomorphisms of $Q$. The implemented layer uses the diagonal triangulation of every rectangular cell and realizes each map as a continuous affine function on every triangle, determined by its fine-grid vertex values. The variable $N$ denotes the number of vertices along one side, so the control mesh has $N^2$ vertices and $2(N-1)^2$ triangles. A query grid with $M^2$ points is a separate object.

## 2. Exact two-factor theorem

**Theorem.** Let $F=(F_1,F_2):Q\to Q$ be an orientation-preserving $C^1$ diffeomorphism that fixes the boundary pointwise. Suppose

\[
\partial_y F_2(x,y)>0\quad\text{for every }(x,y)\in Q.
\]

Then $F=H\circ V$, where $V$ is a vertical triangular homeomorphism and $H$ is a horizontal triangular homeomorphism. One canonical choice is

\[
V(x,y)=(x,F_2(x,y)),\qquad
H(u,v)=\bigl(F_1(V^{-1}(u,v)),v\bigr).
\]

**Proof.** Because $F$ fixes the bottom and top edges, $F_2(x,0)=0$ and $F_2(x,1)=1$. By the strictly positive derivative, for each $x$, the function $y\mapsto F_2(x,y)$ is strictly increasing and covers $[0,1]$. Hence $V$ is a bijection of $Q$, continuous with continuous inverse, and is vertical triangular. Define $H=F\circ V^{-1}$. Its second coordinate is $v$, so it is triangular in the other direction. At fixed $v$, write $y=y(u,v)$ for the inverse vertical fiber. Differentiating $F_2(u,y(u,v))=v$ with respect to $u$ gives $\partial_u y=-F_{2,x}/F_{2,y}$. Consequently

\[
\partial_u H_1
=F_{1,x}-F_{1,y}\frac{F_{2,x}}{F_{2,y}}
=\frac{\det DF}{F_{2,y}}>0.
\]

The two endpoints of each horizontal fiber map to 0 and 1 because $F$ fixes the left and right boundaries. Thus $H$ is a horizontal triangular homeomorphism. Its definition gives $H\circ V=F$. For the fixed canonical choice $V=(x,F_2)$, $H$ is uniquely determined. □

This is a sufficient representation theorem, not a theorem that *all* orientation-preserving maps satisfy the hypothesis. In fact, a two-layer vertical-then-horizontal composition of the implemented family necessarily has $F_{2,y}>0$ wherever its affine derivatives exist: $F_2=Y(v(x,y))$, and both factors have positive fiber derivatives. A map with $F_{2,y}<0$ on an open set is therefore outside this two-layer family, even if it is orientation-preserving. More layers or a different construction would be needed for such a target.

## 3. Fine-grid P1 discretization

Take uniform vertices $x_i=i/(N-1)$ and $y_j=j/(N-1)$. For one vertical layer, output vertices have coordinates

\[
V_h(x_i,y_j)=(X_i,v_{ij}),\qquad
0=X_0<X_1<\cdots<X_{N-1}=1,\quad
0=v_{i0}<v_{i1}<\cdots<v_{i,N-1}=1.
\]

On both triangles of every cell, $\partial_x(V_h)_1>0$ and $\partial_y(V_h)_2>0$, so the affine determinant is strictly positive. More importantly, every vertical fiber of the whole P1 extension is strictly increasing and covers $[0,1]$, which proves global bijectivity without inferring it from local determinants alone. Swapping $x$ and $y$ gives the horizontal layer proof. The neural decoder uses a softmax of finite logits plus a fixed positive spacing floor, followed by cumulative sums. It checks that strict inequalities survive in the represented floating-point dtype. This is a structural guarantee for each layer, conditional on successful represented checks, not a post-hoc fold repair.

If $F$ in the theorem is $C^2$, sample the canonical $V,H$ at the grid vertices and use their P1 interpolants $V_h,H_h$. Strict fiber inequalities hold at the sampled vertices, so each P1 factor is a homeomorphism. On a shape-regular uniform triangulation, ordinary affine interpolation of a $C^2$ function has a uniform function-value error of order $h^2$, where $h=1/(N-1)$. Because $H$ is Lipschitz on compact $Q$,

\[
\|H_h\circ V_h-H\circ V\|_\infty
\le \|H_h-H\|_\infty+\operatorname{Lip}(H)\|V_h-V\|_\infty
=O(h^2).
\]

This is an **existence/approximation** statement. It does not say that finite-step image-only training recovers those interpolants, nor that a fixed spacing floor represents arbitrarily small target derivatives. The implemented floor fraction is 0.01; each normalized interval is at least $0.01/(N-1)$. Targets with stronger compression may require a smaller floor or more layers, balanced against floating-point resolution.

## 4. What exact composition means computationally

The continuous composition $H_h\circ V_h$ is piecewise affine because the plane can be subdivided by the preimages under $V_h$ of edges of the triangulation used by $H_h$. That subdivision generally cuts through the **original** triangles. The layer stores two P1 control maps and evaluates a query $q$ by first locating $q$ in the first grid, applying $V_h$, then locating $V_h(q)$ in the second grid and applying $H_h$. No intermediate image is resampled; only the final coordinate is used to sample the moving image. The chain-rule VJP is defined almost everywhere. A query exactly on a P1 edge receives the derivative of one selected affine branch.

If one samples $H_h\circ V_h$ only at the original $N^2$ vertices and P1-interpolates those samples on the original mesh, one has constructed a **different map**. The two-layer homeomorphism theorem does not certify this exported map. For a particular checkpoint, checking all original-grid face orientations and the fixed boundary may certify it; this is a checkpoint-specific result, not a parameterization-wide guarantee. The main neural-layer output remains the exact composition representation.

## 5. Beltrami and error measurements

On an affine piece, write $J=DF=\begin{psmallmatrix}u_x&u_y\\v_x&v_y\end{psmallmatrix}$. The complex derivatives are

\[
f_z=\tfrac12[(u_x+v_y)+i(v_x-u_y)],\qquad
f_{\bar z}=\tfrac12[(u_x-v_y)+i(v_x+u_y)],\qquad
\mu=f_{\bar z}/f_z.
\]

The evaluation code locates each pixel-center query in each structured triangle, computes that triangle's **exact affine Jacobian**, and multiplies layer Jacobians in the composition order. Its reported map RMSE is $\sqrt{\frac1{2Q}\sum_{q=1}^Q\|F(q)-F_\star(q)\|_2^2}$. Its Beltrami RMSE is $\sqrt{\frac1Q\sum_{q=1}^Q|\mu_F(q)-\mu_\star(q)|^2}$. Uniform pixel-center sampling approximates an area integral; it is not an exact integral over every refined affine piece. The maximum $|\mu|$ and minimum Jacobian determinant in that evaluation are likewise **sampled** quantities. Individual P1-layer face orientations are checked exactly from their vertex coordinates in represented arithmetic.

## 6. Current evidence boundary

On a 257×257 control mesh, a two-layer latent optimized directly against an independently generated smooth target with local 16-cycle detail reached a vertex map RMSE of about $6.93\times10^{-6}$ after 1,000 Adam steps. On 512×512 pixel-center queries, the same saved checkpoint has map RMSE $1.35\times10^{-5}$, sampled Beltrami RMSE 0.00836, sampled minimum exact-composition Jacobian determinant 0.5366, and maximum sampled $|\mu|=0.3574$ versus target maximum 0.3605. The particular exported original-grid P1 interpolation has zero nonpositive faces and minimum signed area ratio 0.5465. Those numbers are direct-map-supervised *expressivity* evidence, not image-only generalization. Separate image-to-latent training and timing must be evaluated on the same data and scale.

## 7. Overlay growth and an explicit sampled-export counterexample

The exact common-refinement partition has a representational cost beyond two original grids. For each source triangle, we counted how many open *horizontal row strips* of the second factor meet the positive-area interior of its first-factor image. Each such source-triangle/row-strip incidence requires a nonempty geometric overlay cell before accounting for the second grid's vertical and diagonal edges. This is a lower bound on overlay incidences, not on the number of intrinsically distinct affine formulas (neighboring pieces can sometimes merge). For 257² factors, there are 131,072 source triangles. Identity has exactly 131,072 incidences; a seeded random vertical factor gave 303,869, with 98.1% of source faces crossing a row line; the saved directly fitted two-factor oracle gave 271,013, with 93.8% crossing. The fitted sample had at most three row strips per original face. The full exact overlay would require additional vertical/diagonal cuts. Thus a compact *factor representation with dynamic queries* is materially cheaper than naively expanding all affine pieces, but the latter's exact size was not enumerated.

More decisively, fixed-grid P1 export can actually fold despite both factors being valid. On a 17² mesh, with float64 logits drawn from `torch.manual_seed(20260923)` and multiplied by 0.5, the first trial produced a vertical factor with minimum normalized P1 face area 0.08214 and a horizontal factor with minimum 0.14985. Their exact continuous composition is a homeomorphism by the factor theorem. Sampling it at the original vertices and P1-interpolating there produced minimum normalized signed area **−7.2355**. A separately evaluated factor-by-factor point query agreed with the composition routine at all original vertices to below \(10^{-12}\), and a 10,000-point branch-Jacobian test found positive determinants for the exact composition. The saved exact latent tensors and reproducible regression test are in `checkpoints/ab2_resample_fold17.pt`, `raw_results/ab2_resample_fold17.json`, and `tests/test_phase6_alternating.py`. This is not a failure of exact composition or repeated image resampling; it is a counterexample to silently replacing that composition with a single P1 map on the old triangulation. At some benign checkpoints, such as the one in Section 6, sampled export remains valid, but that is a measured instance, not a structural guarantee.

## 8. Fine-detail geometry of a directly fitted exact composition

For the saved high32 **coordinate-supervised oracle** after the additional 2000 warm-start steps, we also evaluated the exact two-factor chain Jacobian at all 131,072 original source-face centroids. The factors' minimum normalized face areas were 0.5119 and 0.5372. The chain determinant minimum at those sampled points was 0.4002; the exact continuous composition is a homeomorphism by the factor theorem, independently of this finite sample. Its source-centroid map RMSE against the analytic target was 0.000301 and its complex Beltrami RMSE was 0.10854, versus a 0.07034 analytic-to-sampled-target-P1 discretization floor; maximum sampled \(|\mu|\) was 0.4721. On the *same* target and centroid convention, the saved A2+ direct oracle had map RMSE 0.000432, Beltrami RMSE 0.12202, maximum \(|\mu|=0.7687\), and minimum face determinant 0.1326. The AB2 result is therefore not merely a good vertex fit: it retains useful local geometry in this one direct-fit case. These numbers do **not** establish a fast or accurate image-trained AB2 encoder, nor do centroid samples equal an exact area integral over the composition's refined PL partition. Reproduction records are `raw_results/ab2_high32_mapfit257_cont2000_exact_qc_centroids.json` and `raw_results/a2_high32_mapfit257_cont2000_beltrami_eval.json`.

## 9. Coarse-to-fine convex-cell composition

To test a cascade closer to the user's multilevel proposal, `CoarseFineConvexQuadComposition` first evaluates an A2+ P1 homeomorphism on a 17² control grid and then evaluates another on a 257² grid at its **mapped coordinates**. No inverse, global linear system or intermediate image resampling is used. The exact continuous composition is a PL homeomorphism because each factor is one; its affine partition generally refines both input grids and must **not** be replaced by P1 interpolation on the original 257² grid without a separate certificate. A 5²+9² identity/topology test and directional finite-difference VJP passed. The default 17²+257² direct oracle has 88,252 latent scalars, only 366 more than a single 257² A2+ (87,886), because the extra factor is coarse; the output still requires two dynamic coordinate evaluations.

With target-map supervision and 1000 Adam updates, the base target gave original-vertex RMSE 0.000268 for coarse+fine versus 0.000496 for single A2+ under the same step count and rate. At all 131,072 source-face centroids, the exact chain map/Beltrami RMSE were 0.000269/0.06954, compared with single A2+'s 0.000462/0.07244. Factor minimum normalized areas were 0.6626 and 0.0918, and sampled chain determinant minimum was 0.1331; maximum sampled \(|\mu|\) was 0.8012. Thus a cheap coarse factor improved this **one smooth/detail target** in both map and sampled geometry, albeit far above the sampled-target-P1 Beltrami discretization floor 0.00554.

The same architecture was less successful on high32: original-vertex RMSE was 0.000769 after 1000 updates and 0.000601 after another 2000 lower-rate updates, versus 0.000444/0.000356 for a single fine A2+ on its corresponding schedules. The final exact chain source-centroid map/Beltrami RMSE were 0.000622/0.16169, versus 0.000432/0.12202 for the single factor; its two factors were still homeomorphic with minimum normalized areas 0.7880/0.1230. A 33² coarse factor worsened the 1000-step high32 vertex RMSE further to 0.001063, though its maximum pointwise error was smaller. Therefore the cascade is a genuine, inexpensive expressive extension but **not** a general accuracy improvement under simultaneous Adam. These are all direct-map oracle results, not image-to-latent evidence. Raw checkpoints and evaluations use the `convex_coarse*_fine257_*` names in this phase's `checkpoints/` and `raw_results/` directories.

A staged 17²+257² oracle then spent 1000 updates on the coarse factor alone, 1000 on the fine factor alone, and 1000 jointly, starting from identity and using the exact composed output in every stage. Its high32 original-vertex RMSE went 0.001358→0.000718→0.000649. At the last checkpoint, source-centroid map/Beltrami RMSE were 0.000657/0.17847, maximum sampled \(|\mu|=0.5648\), and minimum chain determinant 0.1988. Compared with simultaneous 3000 updates (0.000601 vertex RMSE, centroid-\(\mu\) RMSE 0.16169, maximum sampled \(|\mu|=0.7771\)), staging improves worst-case local distortion and maximum pointwise error but worsens mean coordinate and Beltrami accuracy. This does not rescue high32 expressivity in the measured budget; optimization order changes the tradeoff rather than giving a clear win. The staged checkpoints and `raw_results/cofi17_257_high32_stage_*.json` retain every phase.

The cascade was also trained as an **image-to-latent neural layer** on the same 257² fine controls, 512² image queries, 32/8 disjoint sample split, batch two and 1000 Adam image-only updates as A2+. Each factor has its own small image encoder; together they have 1868 learned parameters, versus 1006 in the original single A2+ encoder, so this is not yet a parameter-matched causal comparison. Both factors passed the full represented P1 certificate (finite coordinates, fixed boundary, positive faces); the output is their exact PL composition. On the same Element CPU:

| Target family and method | Held-out image MSE | Query-map RMSE | Face-centroid Beltrami RMSE | Maximum sampled \(|\mu|\) | Minimum factor area | Median forward/VJP per sampled training step | Training wall | Sampled RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base, A2+ single P1 | 0.002387 | 0.004196 | 0.10095 | 0.9116 | 0.0485 | 22.2/31.2 ms | 55.8 s | 816 MB |
| base, A2+ width 12 (single P1) | 0.002713 | 0.004344 | 0.07451 | 0.6042 | 0.2737 | 23.0/34.9 ms | 60.9 s | 822 MB |
| base, AB2 exact composition | 0.011867 | 0.012147 | 0.30003 | 0.9718 | 0.1711 | 32.1/41.6 ms | 75.0 s | 973 MB |
| base, CF2 exact composition | 0.001854 | 0.003629 | 0.06069 | 0.3117 | 0.6135 | 44.6/55.8 ms | 103.4 s | 956 MB |
| high32, A2+ single P1 | 0.000613 | 0.002022 | 0.1395 | 0.117 | 0.7773 | 23.5/35.2 ms | 60.5 s | 816 MB |
| high32, CF2 exact composition | 0.000499 | 0.001831 | 0.13917 | 0.1138 | 0.8063 | 43.4/55.9 ms | 101.7 s | 964 MB |

The base-family improvement includes a substantial reduction in unwanted quasiconformal distortion, not only lower image error; the CF2 chain determinant minimum at sampled source centroids was 0.5709, and both factor minimum normalized areas exceeded 0.61. The same-host AB2 image encoder (772 parameters) was faster but much less accurate in this setting; its maximum sampled \(|\mu|\) reached 0.972 despite every factor being a homeomorphism. AB2 also starts from a slightly nonidentity zero-logit map (initial held-out image MSE 0.03729 versus 0.03665 for A2/CF2), so this comparison changes parameterization, encoder size and initialization, not only composition type. To probe predictor capacity, we widened the single A2+ encoder to 12 channels, yielding **1926** parameters—3.1% more than CF2's 1868—without changing the decoder or training protocol. It improved face-Beltrami error relative to width 8 but had *worse* image/map errors than both width 8 and CF2, and its maximum \(|\mu|\) remained 0.604. Under this one controlled width ablation, CF2's gain is not explained simply by more encoder parameters; different architecture/optimization could still matter. The high32 result is more limited: despite modest image/map improvement, the eight predicted 32-cycle coordinate amplitudes remained only about \(10^{-7}\)–\(10^{-6}\), versus true magnitudes around \(10^{-3}\). Its face-Beltrami RMSE scarcely moved, and the target-P1 discretization floor was 0.03832. The forward/VJP timings include the encoders, both decoders, dynamic query composition, image sampling and loss on CPU; the sampled RSS is process memory, not model-only memory. Per-run summaries, independently sampled exact-chain geometry, and checkpoints are saved under the corresponding `cf2_*_heldout257_cpu1000`, `ab2_base_heldout257_elementcpu1000` and `a2_*_base_heldout257_*cpu1000` names.
