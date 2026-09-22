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

Exact-arithmetic convexity is not by itself a finite-precision certificate. A cell can become poorly conditioned after many subdivisions; the chord denominator and final triangle areas may be small. The default implementation checks finite coordinates, every final P1 triangle's **represented** signed area in float64 with an absolute roundoff margin \(64\epsilon_{64}\) on this unit-square coordinate scale, and exact dyadic boundary values, and raises rather than returning a failed map. Strictly positive exact face orientations plus fixed simple boundary give a global PL certificate for that realized output; the double-precision margin is a conservative numerical screen for those signs, not a formally verified adaptive-exact predicate. The check does not claim an interval-arithmetic proof of the latent-to-float computation, and backpropagation through a rejected output is undefined. float32 and float64 margin/gradient measurements are therefore required at 257².

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

We also checked the saved **image-trained** A2+ model's actual P1 face geometry against the analytic target on all 131,072 source triangles of each of eight held-out maps. Every source face has equal area, so the source-area-weighted Beltrami RMSE is the square root of the arithmetic mean of \(|\mu_{\rm predicted}-\mu_\star|^2\) over samples and faces. It was **0.1009**; maximum predicted \(|\mu|\) was **0.9115**, compared with target maximum **0.3155**. The minimum face Jacobian determinant was 0.04855. Thus the output remained nonfolded and accurately aligned images, but its QC distortion was much worse than the target. This is an important non-success on the geometry axis. A target-independent deformation-gradient regularizer is being tested; no target map or Beltrami label enters training.
