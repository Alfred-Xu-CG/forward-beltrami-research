# Route A3: latent-controlled local motion with a fixed-grid P1 homeomorphism

## 1. The contract and the obstruction being removed

Let \(\Omega=[0,1]^2\) and let \(\mathcal T_N\) be the fixed triangulation of the regular \(N\times N\) vertex grid, with every square split along its southwest-to-northeast diagonal. A vertex map assigns \(Y_v\in\mathbb R^2\) to each grid vertex \(v\); its **P1 extension** is the unique continuous map affine on every triangle. An orientation test is made on *all original triangles*, not only sampled image pixels. A valid layer must fix the square boundary and make this P1 extension a homeomorphism of \(\Omega\) onto itself.

The earlier A2+ hierarchy builds this contract without a linear solve, but every newly introduced edge midpoint lies on the straight segment joining the images of its parent edge endpoints. This collinearity can exclude a curved target even when all target grid cells are convex. A3 retains A2+ as a coarse-to-fine base and adds fine-grid latent freedom **after** the full grid is constructed. It does not move vertices by taking a step along \(-\nabla_Y L\), perform a line search, or repair a fold after it occurs. The latent vector itself parameterizes a safe family of maps.

## 2. Exact construction

Write \(Y^{(0)}\) for any boundary-identity, orientation-preserving P1 map on \(\mathcal T_N\); our implementation obtains it from A2+. For an oriented triangle \(t=(a,b,c)\), define its signed double area

\[
A_t(Y)=\det(Y_b-Y_a,Y_c-Y_a).
\]

Assume \(A_t(Y^{(0)})>0\) for every triangle. Color each strict interior grid vertex \((i,j)\) by the pair \((i\bmod2,j\bmod2)\). The only mesh edge displacements are horizontal \((1,0)\), vertical \((0,1)\), and diagonal \((1,1)\) up to sign. Therefore no two vertices with the same color share an edge, and no triangle has two vertices of the same color.

Process the four colors in a fixed order. At the beginning of one pass, take a vertex \(v\) of that color and one of its six incident triangles. Cyclically write that triangle as \((v,q,r)\), so its edge opposite \(v\) is oriented from \(q\) to \(r\). Define the positive signed altitude

\[
h_{v,t}(Y)=\frac{\det(Y_r-Y_q,Y_v-Y_q)}{\|Y_r-Y_q\|_2}
=\frac{A_t(Y)}{\|Y_r-Y_q\|_2}>0,
\qquad r_v(Y)=\min_{t\ni v}h_{v,t}(Y).
\]

The input latent at this vertex is \(z_v\in\mathbb R^2\). With a fixed \(\gamma\in(0,1)\), A3 updates it by

\[
Y_v^{\mathrm{new}}=Y_v+\Delta_v,\qquad
\Delta_v=\frac{\gamma r_v(Y)}{\sqrt2}\bigl(\tanh z_{v,1},\tanh z_{v,2}\bigr).
\]

Every other vertex, including all boundary vertices, stays fixed during this color pass. Same-color updates are simultaneous. The next color recomputes all altitudes from the map *after* earlier passes; it does not reuse stale margins. The code stores six cyclic opposite-edge pairs per strict interior vertex and gathers them in four vectorized PyTorch passes.

## 3. Why every output is globally one-to-one

For finite \(z_v\), \(\|\Delta_v\|_2<\gamma r_v<r_v\). In an incident triangle, moving only \(v\) changes its signed double area to

\[
\begin{aligned}
A_t(Y^{\mathrm{new}})
&=A_t(Y)+\det(Y_r-Y_q,\Delta_v)\\
&\ge \|Y_r-Y_q\|_2\bigl(h_{v,t}(Y)-\|\Delta_v\|_2\bigr)>0.
\end{aligned}
\]

No triangle contains two moving vertices of the same color, so this inequality applies simultaneously to every affected triangle; unaffected triangles keep their area. Induction through four colors preserves positive orientation on **every** original triangle. Because the outer boundary remains the identity, the planar PL global-inversion theorem then gives a homeomorphism. In particular, [Lipman, *Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing*, Theorem 1](https://arxiv.org/pdf/1310.0955) states that a nondegenerate orientation-preserving simplicial map of a compact mesh is bijective onto the target when its boundary map is bijective. Its degree proof counts positive preimages of interior regular points. The global conclusion uses both positive orientation *and* the injective fixed boundary; positive sampled Jacobians alone would not justify it.

The theorem is in exact arithmetic, conditional on the base map and finite latent values. In floating point we separately check finite coordinates, the pointwise fixed boundary, range, and every original-grid triangle's signed area using `certify_convex_quad_output`. Its name predates A3; its implementation checks the represented **P1 triangles**, not convexity of every final quadrilateral. A small positive numerical margin is a verification filter, not a proof that roundoff cannot ever defeat the exact formula.

## 4. Differentiation, work and memory

Away from equal-altitude ties and query triangle-switch boundaries, all operations are ordinary differentiable tensor arithmetic. At a tie, the minimum radius has a valid piecewise subgradient but is not classically differentiable; a first-order VJP exists almost everywhere, as for many ReLU-based neural layers. Backpropagation passes through A2+, the four sequential altitude/minimum/update operations, and final P1 query interpolation. It does not store a Krylov history, factor a sparse matrix, or call an inverse solver. The vertex work and saved first-order activations scale as \(O(N^2)\) for one color cycle; evaluating \(Q\) final image coordinates on the fixed structured P1 mesh adds \(O(Q)\). No dynamic point location between factors is needed because the **output itself remains P1 on \(\mathcal T_N\)**.

The four-color radius couples each vertex only to its six incident faces. If a map already has a tiny face altitude, local freedom shrinks automatically; one cycle is therefore not a universal parameterization of all homeomorphisms. A2+ supplies broad displacement, and A3's local latent resolves detail around that base. The range of each local move is \(O(1/N)\), not a coarse-field replacement.

## 5. Tests and measured scope

At 17², random base A2+ maps followed by large random A3 logits preserved every oriented original face and the pointwise boundary. An independent test enumerated actual triangles from `structured_rectangle` and matched every one of the six arithmetic opposite-edge pairs for every interior vertex; a double-precision directional finite difference matched the autograd VJP. A 9² witness moved one dyadic parent-edge midpoint off the line through its mapped endpoints while all faces stayed positive. These tests establish the indexing and local mechanism, not a quantitative expressivity advantage.

On Element CPU, float32, four threads, batch one, fixed 512² image queries, two warmups and five repeats, a synthetic random-latent image-warp/loss benchmark measured:

| Control grid | Original triangles | Latent scalars | Median forward incl. query/warp/loss | Median VJP | Sampled process RSS | Minimum normalized face area |
|---:|---:|---:|---:|---:|---:|---:|
| 257² | 131,072 | 217,936 | 13.2 ms | 23.7 ms | 559 MB | 0.516 |
| 513² | 524,288 | 872,784 | 26.9 ms | 48.4 ms | 660 MB | 0.419 |
| 1025² | 2,097,152 | 3,493,200 | 84.5 ms | 160.5 ms | 1,062 MB | 0.444 |

Full samples are in `raw_results/a3_scaling_{257,513,1025}_element_cpu.json`. Process RSS is sampled after each repeat and includes runtime overhead, so a transient peak may be missed. One-time setup, including opposite-edge indexing, is excluded from forward/VJP and recorded separately. The 1025² control-grid test still uses only 512² queries; it must not be read as a 1025² image-training result.

For the deterministic high32 target, direct **target-coordinate-supervised latent** optimization at 257² (not image-to-latent training) obtained original-vertex coordinate RMSE 0.000357 after 1,000 Adam updates and 0.000305 after another 2,000 lower-rate updates. Independent all-face-centroid Beltrami RMSE was 0.11543 then 0.10719; the predicted 32-cycle coordinate amplitude at 3,000 steps was (0.002072, 0.002064) versus true (0.002500, 0.002500). The represented minimum normalized face area remained 0.1804 and maximum centroid \(|\mu|\) was 0.699. The single A2+ 3,000-step comparator had vertex RMSE 0.000356 and Beltrami RMSE 0.12202; the two-fine exact-composition oracle had 0.000191 and 0.10274 but is not fixed-grid P1. Thus A3 improves the measured high-frequency fit/geometry of the fixed-grid layer at roughly doubled forward/VJP cost, while an exact two-factor composition still fits that particular target better. This one target is not a claim about arbitrary deformations or generalization.

The 257²/512² **image-only** high32 test used a shared-feature convolutional encoder with 1,024 learned parameters and one extra two-channel fine local-logit head. On the same 32/8 disjoint split, batch two and 1,000 Adam updates as A2+, held-out image MSE was 0.000610, query-map RMSE 0.002020, all-face-centroid Beltrami RMSE 0.13947 and minimum represented normalized area 0.7676. Median full encoder+decoder+query/image-loss forward/VJP was 34.8/57.4 ms on Element CPU, sampled RSS 893 MB and total training 93.3 s. These numbers are essentially the single-A2 high32 result (image 0.000613, map 0.002022, Beltrami 0.1395), but at higher cost. All eight predicted 32-cycle displacement projections were only around \(10^{-7}\), whereas target amplitudes were around \(10^{-3}\). Thus the new *decoder* can reduce direct-oracle high-frequency geometric error, while this image encoder still fails to use its extra local degrees of freedom.

On the base family with the same split and schedule, A3 held-out image MSE was 0.002330, query-map RMSE 0.004180, centroid Beltrami RMSE 0.10308, maximum sampled \(|\mu|=0.8665\), and minimum face area ratio 0.0788. Compared with A2+'s 0.002387/0.004196/0.10095 for image/map/Beltrami, the image/map gains are small and the *mean* Beltrami error is slightly worse; CF2's exact composition remained considerably better on this base task, though it is not original-grid P1. A3 full training forward/VJP was 33.1/51.5 ms, sampled RSS 875 MB, and wall 88.6 s versus A2+'s 22.2/31.2 ms and 55.8 s. Hence A3 is a defensible fast fixed-P1 expressive extension, but not yet a superior image registration layer under this encoder/protocol. Both A3 results and their face-level evaluations are saved in the `a3_{base,high32}_heldout257_cpu1000*` artifacts.

Code, focused tests, saved checkpoints and raw face evaluations are in `src/qcopt/neural_bijection/dense/colored_vertex_relaxation.py`, `tests/test_phase6_colored_vertex_relaxation.py`, `tools/phase6_fit_dense_map_oracle.py`, and the `a3_high32_*` artifacts.

## 6. A4 directional radial safety: a larger latent-feasible region

A3 inscribes a Euclidean disk in the intersection of the six incident face-positive halfplanes. That sufficient disk is sometimes unnecessarily small in a particular direction. A4 changes only the local motion parameterization; the A2+ base, four-color schedule, mesh, fixed boundary, P1 representation, and global-inversion argument remain the same.

Let \(h=1/(N-1)\), choose fixed \(\alpha>0\) and \(0<\gamma<1\), and let the latent at interior vertex \(v\) be \(z_v\in\mathbb R^2\). For each currently incident oriented face \(t=(v,q,r)\), put \(e_t=Y_r-Y_q\) and \(A_t=\det(e_t,Y_v-Y_q)>0\). First propose the direction-and-magnitude vector \(d_v=\alpha h\tanh(z_v)\), where the hyperbolic tangent acts coordinatewise. The decrease in signed double area caused by the *unscaled* proposal is \(g_{v,t}=\max\{0,-\det(e_t,d_v)\}\). Define

\[
\tau_v=\min_{t\ni v:g_{v,t}>0}\frac{A_t}{g_{v,t}},\qquad
s_v=\min\{1,γ\tau_v\},\qquad
Y_v^{\mathrm{new}}=Y_v+s_vd_v,
\]

with the minimum over an empty set interpreted as \(+\infty\). Thus the latent proposal is restricted only by incident faces for which it is actually adverse; it is **not** the gradient of the image or map loss, and no loss-dependent line search occurs. For an adverse face, \(s_vg_{v,t}\le \gamma A_t\), so \(A_t(Y^{\mathrm{new}})=A_t-s_vg_{v,t}\ge(1-\gamma)A_t>0\). For a nonadverse face, area does not decrease. Same-color vertices never share a face; after four passes every original-grid face remains positive. Together with the injective fixed boundary, the planar PL theorem cited in Section 3 certifies that the **represented original-grid P1 map** is a homeomorphism in exact arithmetic. This does not assert a uniform float32 margin across arbitrary logits; all-face numerical checks remain necessary. The implementation uses \(\alpha=2\) and \(\gamma=0.85\) by default, and the test suite covers indexing, finite-difference VJP, high-logit positive faces and noncollinear parent-edge motion.

The direct target-coordinate-supervised high32 oracle at 257² control vertices and 1,000 Adam updates of learning rate 0.05 attained vertex RMSE \(1.748\times10^{-5}\), then \(1.373\times10^{-5}\) after 2,000 more updates of rate 0.01. These are decoder/latent-fit measurements, **not** image-only neural inference. The independent 131,072-face centroid evaluation of the 3,000-step checkpoint gave map RMSE \(1.834\times10^{-4}\), complex Beltrami-coefficient RMSE 0.070522 against the analytic target, and RMSE 0.004715 against the target sampled as a P1 map on this same mesh. The analytic-to-sampled-P1 Beltrami discrepancy is already 0.070343, a discretization floor for this target/mesh/evaluation. The predicted 32-cycle displacement projections were 0.00249890 and 0.00249810, versus true 0.00250000; minimum normalized signed face area was 0.39067 and maximum face-centroid \(|\mu|\) was 0.54779. These numbers show strong representation on **one synthetic target**, not universal expressivity or image inference.

For batch one, random latent logits of standard deviation 0.08, float32, Element CPU with four threads, fixed 512² queries, two warmups and five repeats, A4 median full decode/query/image-loss forward and VJP were 13.4/24.9 ms at 257², 29.5/51.9 ms at 513², and 83.2/175.6 ms at 1025². Sampled process RSS was 562/690/1119 MB. The minimum normalized face-area ratios were respectively 0.00350/0.00227/0.00240: strictly positive, but significantly smaller than A3's random-logit margins. In the trained high32 oracle the margin was much healthier (0.39067). The radial freedom therefore comes with a genuine near-degeneracy risk for unconstrained random outputs; numerical robustness and output-logit calibration require further testing. The timings include only one query grid, not the cost of generating all 1025² image pixels.

On the shared 32/8 high32 image-only split with 512² images, batch two and 1,000 Adam updates, the existing 1,024-parameter local image encoder plus A4 produced held-out image MSE 0.0005368, pixel-query map RMSE 0.0019768, face-centroid Beltrami RMSE 0.14059, minimum normalized face area 0.27874, and 32-cycle projections only around \(10^{-5}\), despite true amplitudes around \(10^{-3}\). Full training forward/VJP medians were 32.6/52.4 ms and sampled process RSS 880 MB. This is an image/MSE improvement over A2+ but **does not recover the fine deformation**. Direct latent representability and amortized image-to-latent learnability are different questions.

Raw samples: `raw_results/a4_radial_scaling_{257,513,1025}_element_cpu.json`, `a4_radial_high32_mapfit257_3000_face_eval.json`, and `a4_high32_heldout257_cpu1000_face_eval.json`; checkpoints: `checkpoints/a3_radial_high32_mapfit257_{1000,3000}.pt` and `a4_high32_heldout257_cpu1000.pt`. The `a3_radial_*` checkpoint prefix records chronological development; its stored method is `convex_quad_radial`.

### Follow-up: base family, random margins, and image identifiability

The matching A4 **base-family image-only** training gave held-out image MSE 0.002169, query-map RMSE 0.004144, face-centroid Beltrami RMSE 0.10393, and minimum area ratio 0.0792. Its 33.0/52.6 ms full forward/VJP and sampled 886 MB RSS are slower than A2+; the image-fit gain over A2+ (0.002387) is modest. It does not reverse the earlier conclusion that the richer decoder is not yet a decisively better image-to-latent layer.

Random-logit margin depends strongly on both the latent scale and the A2+ base, not only the A4 radius. At 257² with Gaussian latent standard deviation 0.08, changing radial raw-span \(\alpha\) from 2 to 1 to 0.5 changed the smallest normalized face area from 0.00350 to 0.23570 to 0.45141 in one common-seed sample. With standard deviation 0.25 or 0.5, the minimum became nearly independent of \(\alpha\) and very small, indicating that the base map itself can be close to degenerate. These are **one-seed stress samples**, not probabilistic guarantees. The \(\alpha=1\) high32 direct oracle after 1,000 steps had vertex RMSE 0.000148, worse than \(\alpha=2\)'s 0.0000175 under the same schedule; simply shrinking \(\alpha\) trades expressive reach for margin. All represented face areas were still positive. Further work should measure base-versus-local margin separately and seek a local calibration that preserves fit.

To test whether the high32 images contain usable high-frequency information at all, a deliberately *non-general* diagnostic used the known synthetic three-function displacement basis. It estimated its three coefficients from each of the eight held-out fixed/moving image pairs, without feeding the target map or true coefficients into optimization. Linearized brightness constancy gave high-frequency coefficient RMSE 0.000237, image MSE 0.000204, and map-coordinate RMSE 0.001186; the three-column normal matrices had condition numbers 2.37–2.76. Initializing 200 exact differentiable image-sampling optimization steps from that estimate reduced coefficient RMSE to about \(3.2\times10^{-8}\) overall (fine coefficient \(6.6\times10^{-9}\)), image MSE to \(1.3\times10^{-13}\), and map RMSE to \(3.0\times10^{-8}\). This establishes identifiability **within the declared synthetic parametric family** and implicates the present amortized encoder/optimization. It is not evidence that arbitrary medical/texture registration is identifiable or that the three-basis estimator is a general deformation layer. A 12,448-parameter context encoder trained with image-only loss also failed to recover the 32-cycle projection (around \(10^{-7}\)); simply increasing receptive-field context and width did not fix the problem in 1,000 steps. Code and raw estimates are in tools/phase6_identify_high32_images.py and raw_results/high32_image_identifiability_refine{0,200}.json.

A separate 1,000-step **target-map-supervised** diagnostic using the small A4 image encoder also produced near-zero 32-cycle projections (about \(10^{-7}\)) on held-out pairs, with map RMSE 0.002027. Because this objective directly supplies the desired map, the failure is stronger evidence that the current translation-equivariant local architecture or its optimization cannot synthesize the globally phase-aligned high-frequency field. It is not an image-only registration result; its checkpoint and all-face evaluation are labeled mapsupervised.
