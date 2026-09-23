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
