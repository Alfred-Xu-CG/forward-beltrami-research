# Current 257² learned P1 registration layer: precise formulation and limits

This document describes the **measured 1000-minibatch, 12-channel recurrent model**, not a proposed future architecture. Its purpose is to distinguish the mathematical map guarantee, the neural computation, the external image-matching preprocessing, and what the experiments actually measure. The executable definitions are `tools/digital_frozen_residual_predict.py`, `tools/digital_mind_recurrent257.py`, `tools/digital_image_force_feature257.py`, and `src/qcopt/neural_bijection/dense/digital_q1.py`. Experimental scores and their selection limitations are in `EXTENSION_IMAGE_REGISTRATION.md`.

## 1. Domain, coordinate convention, input and output

Let \(\Omega=[0,1]^2\), and let \(h=1/256\). The **control vertices** are
\[
  x_{ij}=(ih,jh),\qquad 0\le i,j\le256.
\]
There are \(257^2=66{,}049\) vertices and \(2\cdot256^2=131{,}072\) fixed reference triangles. Each square is cut along the \((i,j)\)–\((i+1,j+1)\) diagonal. This is the fixed triangulation throughout prediction: later passes update the same vertex table, not a sequence of separately triangulated maps resampled after composition. A table \(u_{ij}\in\mathbb R^2\) defines a continuous **P1** map \(u\) by barycentric interpolation separately on these triangles. The four-corner **Q1** determinant test is a stronger local audit on the same vertex table; it is not a claim that image values themselves are Q1-interpolated.

The deployed predictor takes (i) a fixed grayscale 512×512 image \(I_f\), (ii) a moving grayscale 512×512 image \(I_m\), (iii) an externally computed affine map \(A(y)=My+b\) with \(\det M>0\), and (iv) at least 16 externally selected machine-image correspondence pairs \((s_k,t_k)\) in the **affine-aligned** unit-square coordinate frame. The direct SuperGlue/RANSAC preprocessing in these experiments estimates a similarity, but the predictor checks positive determinant rather than the similarity identity \(M^TM=s^2I\). \(s_k\) lies in the fixed image; \(t_k\) lies in the moving image *after* affine prewarping. These correspondences are not manual evaluation landmarks. The output is \((M,b,\{u_{ij}\})\); the physical moving-side deformation represented by the archive is
\[
       F(x)=A(u(x))=M u(x)+b.
\]
Thus a positive determinant of \(M\) is part of the output guarantee. The raw moving image is prewarped as \(J_A(x)=I_m(A(x))\), using differentiable image sampling **once the affine is fixed**. The SuperGlue/RANSAC match selection and direct-affine estimation are external, generally nondifferentiable preprocessing and are excluded from the reported neural-layer time and VJP. Continuous coordinates of the *selected* matches, image pixels, and network parameters do have a VJP through the map decoder.

## 2. Frozen starting map and spatial evidence

A previously trained image-only global network maps 128² image features to a 257² base table. A previously trained one-head 257² network then applies one safe F1 update. We call the resulting fixed-boundary, positive-orientation table \(u^{(0)}\), or the **frozen starting map**. “Frozen” here means its checkpoint is held fixed during the residual-student training; it is still part of the complete inference computation. It already improves over the external affine on several cases, so every residual experiment compares against **this same starting map**, not only against \(A\).

From \(I_f,J_A,u^{(0)}\), the predictor constructs a 256² feature tensor \(P\) with 88 channels: 43 bilinearly upsampled 128² **input image feature** channels (not hidden global-network activations), eight fixed local self-similarity descriptor channels, eight moving descriptor channels, 25 descriptor-correlation channels, two components of starting-map displacement at cell centers, and two normalized cell-center coordinates. At each 256² cell center \(c_{ij}=((i+\tfrac12)h,(j+\tfrac12)h)\), the map feature uses the **P1-exact** value on its diagonal, \(u^{(0)}(c_{ij})=(u^{(0)}_{ij}+u^{(0)}_{i+1,j+1})/2\). The descriptor validity mask \(m(c_{ij})\in\{0,1\}\) excludes low fixed-image intensity or degenerate descriptor scale. The local descriptors and correlation calculation are implemented in the files cited above; “MIND-like” in the result table names this particular eight-channel self-similarity construction, not an abstract clinical similarity measure.

The selected machine matches are rasterized at each cell center \(c\). For \(\sigma\in\{0.04,0.12\}\), set \(w_{k,\sigma}(c)=\exp[-\lVert c-s_k\rVert^2/(2\sigma^2)]\) and \(d_k=t_k-s_k\). Three channels per \(\sigma\) are
\[
  R_{\sigma,x}(c)=512\frac{\sum_k w_{k,\sigma}(c)d_{k,x}}
                                 {\sum_k w_{k,\sigma}(c)+10^{-8}},\quad
  R_{\sigma,y}(c)=512\frac{\sum_k w_{k,\sigma}(c)d_{k,y}}
                                 {\sum_k w_{k,\sigma}(c)+10^{-8}},\quad
  R_{\sigma,m}(c)=\log(1+\sum_k w_{k,\sigma}(c)).
\]
The factor 512 expresses displacement in canvas-pixel scale. These six fields are **features**, not a guaranteed or directly accepted deformation. Their Gaussian evaluation is separable in the grid axes, but still costs work proportional to selected match count; it is not a grid-sized linear solve.

At each recurrent pass, six further **image-force features** are recomputed using the *current* table \(u^{(p)}\). Let \(D_f(c),D_m(c)\in\mathbb R^8\) be the descriptors of \(I_f,J_A\); let \(D_m(u^{(p)}(c))\) denote P1-map sampling of the moving descriptor. Write \(r(c)=D_f(c)-D_m(u^{(p)}(c))\), and estimate the two moving-descriptor spatial derivatives by central differences on the 256² descriptor grid. Componentwise soft signs \(q_a=r_a/\sqrt{r_a^2+0.05^2}\) give a two-component force
\[
  g_x(c)=-m(c)\frac18\sum_{a=1}^8 q_a(c)\,\partial_xD_{m,a}(u^{(p)}(c)),
  \qquad
  g_y(c)=-m(c)\frac18\sum_{a=1}^8 q_a(c)\,\partial_yD_{m,a}(u^{(p)}(c)).
\]
If \(\rho\) is the root mean square of both components over the grid, the returned six channels are \(\tanh(g_x/(3\rho+10^{-5}))\), its \(y\)-analogue, their 7×7 local averages, mean absolute descriptor mismatch, and \(m\). This is an approximate descent cue for the chosen descriptor loss. It is **not** a ground-truth anatomical correspondence vector; the wrong-direction examples in the report show why that distinction matters. It is recomputed once per pass, so it does not become stale when the map moves. Central descriptor differences keep first-order reverse differentiation through the cue available without invoking a grid-sampling second derivative.

## 3. Four recurrent safe updates

For pass \(p=0,1,2,3\), form the 256² input tensor by concatenating \(P\) (88 channels), the static match raster \(R\) (six), the freshly evaluated force (six), and the two-component current-map feedback \(u^{(p)}(c)-u^{(0)}(c)\). A four-scale width-32 U-Net maps these 102 channels to hidden features; a pass-specific 1×1 convolution produces two latent logits per cell, bilinearly upsampled to interior 257² vertices. All passes share the U-Net trunk but have distinct output heads. The logits are suggestions only; the subsequent geometric operator determines accepted motion.

For an interior vertex \(i\), write \(H_i=(u_{i+e_x}-u_{i-e_x})/2\) and \(V_i=(u_{i+e_y}-u_{i-e_y})/2\) in the **current**, possibly irregular image of the grid. If its two logits are \(z_i\), the raw proposal is
\[
            r_i=8[H_i\tanh z_{i,x}+V_i\tanh z_{i,y}].
\]
This current-edge formula is why the step scale does not assume that a deformed mesh still has regular physical spacing \(h\). Interior vertices are visited in four row/column parity classes \((i\bmod2,j\bmod2)\in\{0,1\}^2\); vertices of the same class do not share the immediately incident cell constraints. Boundary vertices are never changed, so \(u|_{\partial\Omega}=\mathrm{id}\) in the aligned frame.

At one active vertex there are 12 incident oriented Q1-corner area constraints. For each, let \(a_k>0\) be its current doubled signed area, and let \(e_k\) be the opposite oriented edge. If only this vertex moves by \(d\), then exactly \(a_k(d)=a_k+\operatorname{cross}(e_k,d)\). With \(a_{\min}=0.05h^2\), define
\[
 B_k=\min\{0.75a_k,\max(a_k-a_{\min},0)\},\quad
 M_i=\max_k\max\{0,-\operatorname{cross}(e_k,r_i)/B_k\},\quad
 d_i=\frac{r_i}{1+M_i}.
\]
If a margin is invalid or a quantity nonfinite, the implementation uses zero displacement for that vertex. When all \(B_k>0\), any adverse loss in an area is strictly smaller than \(B_k\) for finite \(r_i\); hence exact arithmetic keeps all 12 areas positive and above the chosen floor when the input already meets it. This is a **per-vertex differentiable analytic safety scale**, not a shared line search or a post-hoc fold repair. The four color classes are processed sequentially, and the next pass starts from the actual accepted map. The scale depends on current geometry and on logits and is *not detached* in reverse-mode differentiation. At max/clamp/tie surfaces the layer is piecewise differentiable and uses the framework's chosen subgradient; it is not globally smooth.

The positive-area argument is an exact-arithmetic property under its input hypotheses. Stored binary32 output may round, so every reported deployment map is independently re-read and checked with exact binary-coordinate signs for **all four corners of all 256² cells** (262,144 values), exact ordered boundary, and positive post-affine. On the chosen fixed diagonal, the two P1 triangle orientations are among these corner signs. Positive triangles, consistent shared edges and the one-to-one ordered boundary imply a global P1 homeomorphism of the rectangle onto its image. The post-affine composition preserves orientation and injectivity. The empirical exact certificate is for each saved artifact, not a theorem that every possible binary32 rounding of arbitrary network parameters succeeds.

### 3.1 Cell-to-vertex readout: formal rank versus usable conditioning

The final interpolation deserves a precise capacity qualification. In one coordinate, let the 256 cell-head values be \(x_0,\ldots,x_{255}\) and retain only interior vertex indices \(j=1,\ldots,255\). With the actual `align_corners=False` convention, their logits are
\[
 y_j=a_jx_{j-1}+b_jx_j,\qquad
 a_j=(j+\tfrac12)/257,\quad b_j=1-a_j.
\]
The matrix \(U\in\mathbb R^{255\times256}\) has full *mathematical* row rank: its columns 1–255 form a lower bidiagonal square submatrix with positive diagonal \(b_j\). Therefore the separable two-dimensional readout \(U\otimes U\) is onto all \(255^2\) interior logit values when its 256² cell-head values are treated as independent real variables. At the identity map, the current-edge safe update has first derivative \(d_i=8h z_i+o(\|z\|)\), and the four parity sweeps visit every interior vertex once. Consequently one pass is locally onto the interior *vertex-displacement tangent space* with respect to freely chosen cell-head values. This is neither a claim about the restricted outputs of a fixed trained U-Net nor a global approximation theorem for large homeomorphisms.

Full rank conceals an extreme conditioning defect. A right-vector residual would **not** prove this, since \(U\) has a one-dimensional exact kernel. Instead take a nonzero *left* vector \(v\in\mathbb R^{255}\) with \(v_{128}=1\), and recursively choose its other entries to satisfy \(b_kv_k+a_{k+1}v_{k+1}=0\) for \(k=1,\ldots,254\). Then \(U^\top v\) has only two nonzero entries, at cell columns 0 and 255. Direct evaluation of these finite products gives \(\|v\|_2\approx3.77201\) and equal-magnitude endpoint entries \(1.56865406\times10^{-75}\). Because \(U\) has full row rank, the minimum in the left-vector variational characterization is the smallest positive singular value, so
\[
 \sigma_{\min}(U)\le\frac{\|U^\top v\|_2}{\|v\|_2}
 \approx5.88125\times10^{-76}.
\]
Direct binary64 singular-value computation cannot resolve this scale: it reports 254 singular values above \(10^{-12}\) out of 255, with the last at its numerical floor near \(5\times10^{-17}\). The two-dimensional readout inherits the conditioning problem through \(U\otimes U\). The left witness is an alternating mode concentrated near the center; a binary64 computation reports the next-smallest one-dimensional singular value near 0.0882, but this numerical spectrum alone is not a certified bound on every other mode. Thus current evidence does **not** establish the readout as the cause of anatomical errors. A zero-initialized **additive** direct-to-vertex residual head, rather than a replacement of the original bilinear path, is an isolated architecture test. The finite-precision *computed* logits should not be described as an exactly onto linear map.

## 4. Training, gradients, and what the numbers mean

The best currently retained matched checkpoint used 63 ACROBAT training IDs and 18 disjoint validation IDs among cases for which the direct image-only affine, at least 16 selected matches, frozen start and image/match-optimized pseudo-teacher all existed. Thus the split is case-disjoint **within this eligible subset**, not a complete evaluation of all 102 selected IDs. In 1000 minibatches of two cases, only the recurrent student's parameters were updated, with
\[
L=1000L_{\rm map}+0.1L_{\rm image}+0.01L_{\rm match}+0.2L_{\rm strain}.
\]
Here \(L_{\rm map}=\operatorname{mean}_{ij}\lVert u_{ij}-u^{\rm teacher}_{ij}\rVert^2\); \(L_{\rm image}\) is the mask-normalized mean absolute difference of the eight fixed descriptors and moving descriptors sampled through the P1 map; \(L_{\rm match}\) is the mean of \(\sqrt{\lVert512(u(s_k)-t_k)\rVert^2+4^2}-4\) over selected machine matches; and \(L_{\rm strain}=\tfrac12[\operatorname{mean}\lVert256\Delta_x(u-x)\rVert^2+\operatorname{mean}\lVert256\Delta_y(u-x)\rVert^2]\). Pseudo-teachers were 25-step per-case optimizations from image-only starts and are **not** evaluation ground truth. No manual anatomical landmark loss entered this training. A zero-map-loss experiment still loaded teacher archives for cohort eligibility/diagnostics, so it must not be described as entirely teacher-independent.

The map-table-to-loss automatic-differentiation path traverses four accepted F1 passes, current-map image-force recomputation, U-Net parameters, differentiable selected-match rasterization, and image sampling. The external SuperGlue/RANSAC *selection* and similarity-estimation step is not in this VJP. Tests check nonzero finite selected-match/image/parameter gradients; the original double-precision directional finite-difference test used a **reduced two-pass, width-four synthetic recurrent decoder**. A later optional production-path probe in `tools/digital_frozen_residual_predict.py` takes a linear functional of the **map output itself**, eliminating the image-loss direct-pixel-path ambiguity. It checked two cases on the complete base→one-head→four-pass architecture after casting the trained float32 weights to float64. For selected-target-coordinate, moving-image-pixel and fixed-image-pixel directions, small-step central differences were within about 0.01–0.2%, 0.09–0.49% and 0.29–1.08%, respectively, of automatic differentiation at the most favourable tested step. Larger steps sometimes disagreed substantially; one fixed-pixel direction retained a roughly 1.08% discrepancy as the step shrank, consistent with unresolved nonsmooth feature/safety branches. This is local directional evidence on two cases, not a complete Jacobian proof or a direct float32 production-map finite-difference test. The full predictor's recorded “forward plus VJP” remains one forward and one backward of its proxy objective, and no claim is made about the external matcher/RANSAC/affine or every discrete cell-index transition and active safety-constraint tie.

The 257² control mesh is **not** a 257² query-only resampling of a 17² or 49² map: there are 66,049 output control vertices, and each strict interior vertex can move through its own latent logits at each recurrent pass. A 512² input image is a separate sampling resolution. On eight confirmation image pairs outside all 102 selected training-source IDs, the retained matched model lowered its dimensionless image proxy on 8/8, preserved 8/8 exact saved P1 certificates, and had indicative per-pair neural forward about 62 ms, forward-plus-VJP about 246 ms, and maximum PyTorch VJP allocation about 1.55 GB on the measured GPU. The external affine/matcher, decoding, files and sign audit are outside that neural timer. The 20-direction native-pixel anatomical TRE result is meaningful **development evidence from one previously viewed specimen**, not blind independent patient validation. See §5.15–5.17 of `EXTENSION_IMAGE_REGISTRATION.md` for every denominator and negative comparison.
