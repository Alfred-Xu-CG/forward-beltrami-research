# A5: image-conditioned local latent with fixed-grid P1 topology

## 1. Scope and definitions

Let \(\Omega=[0,1]^2\), let \(\mathcal T_N\) be the standard \(N\times N\) vertex grid triangulated with one fixed southwest-to-northeast diagonal in each cell, and let \(I_f,I_m:\Omega\to\mathbb R\) be fixed and moving images. For a deformation \(F\), registration here means that the warped moving image \(I_m(F(x))\) approximates \(I_f(x)\). The desired output contract is a map represented by the \(N^2\) vertex positions and affine interpolation on **every original triangle** of \(\mathcal T_N\), with fixed identity boundary and no flipped face.

The existing A4 decoder takes multiscale learnable logits \(z\), first generates a base map \(Y^{(0)}\) by convex-quad refinement, then applies one four-color local radial motion. Section 6 of the colored-vertex document defines that motion and proves its exact-arithmetic orientation preservation. A5 does not replace that topology mechanism. It adds an *image-conditioned latent hint* \(H(I_f,I_m,Y^{(0)})\) before A4's local radial step, so the final local latent is \(z_{\rm local}+\eta H\). The map is still P1 on \(\mathcal T_N\). Its learnable encoder is the same 1,024-parameter image-to-multiscale-latent network as A4; \(\eta\), patch width and ridge were held fixed during the reported training.

This is not an arbitrary optical-flow field added directly to the map, and it is not a loss-gradient step \(F-t\nabla_F L\). The brightness-constancy normal equation generates a bounded latent proposal; the radial decoder maps that proposal into a face-positive P1 geometry. The exact-topology statement is conditional on finite inputs, a positively oriented base and the fixed-boundary hypothesis. Float32 outputs are separately checked face by face.

## 2. Local image-derived proposal

At original control vertex \(x_v\), the base map gives \(y_v=Y_v^{(0)}\). Define the image residual \(r_v=I_f(x_v)-I_m(y_v)\) and moving-image gradient \(g_v=\nabla I_m(y_v)\), in physical unit-square coordinates. The gradient is approximated by centered pixel differences scaled by image-side-minus-one, then bilinearly sampled at \(y_v\); \(I_m(y_v)\) is sampled by the same bilinear image convention. The fixed image is bilinearly sampled at the regular control vertices. This distinction matters: taking a gradient of the already warped image would multiply by \(DY^{(0)}\) and correspond to a different increment coordinate.

If a small correction \(\delta_v\in\mathbb R^2\) were approximately constant within a local control-grid window \(W_v\), first-order brightness constancy would give

\[
I_m(y_u+\delta_v)-I_m(y_u)\approx g_u^\mathsf T\delta_v\approx r_u,\qquad u\in W_v.
\]

For a ridge \(\lambda>0\), A5 computes

\[
G_v=\frac{1}{|W_v|}\sum_{u\in W_v}g_ug_u^\mathsf T+\lambda I_2,\qquad
b_v=\frac{1}{|W_v|}\sum_{u\in W_v}g_ur_u,\qquad
\widehat\delta_v=G_v^{-1}b_v.
\]

This is a local \(2\times2\) solve in each window, not a global mesh linear system. Since \(G_v\succeq\lambda I_2\), it is invertible for finite image values. Window averages are computed by local pooling, and the two-coordinate inverse is evaluated by its scalar determinant formula. To express the correction as a finite radial logit, set \(h=1/(N-1)\), raw span \(\alpha=2\), and

\[
H_v=\operatorname{atanh}\left[
\operatorname{clip}\!\left(\frac{\widehat\delta_v}{\alpha h},-0.95,0.95\right)
\right].
\]

The clip bounds the logit input even if the local optical-flow estimate is inaccurate. The final A4 radial step still restricts the actual vertex motion by all six incident original-triangle area inequalities. The outer boundary has no local logits and stays fixed.

## 3. Differentiation and guarantee

The encoder, bilinear image sampling, local means, positive-ridge \(2\times2\) inverse, clipping, hyperbolic arctangent, A4 radial safety, fixed-grid P1 interpolation, final image warp, and image loss form one first-order differentiable computational graph almost everywhere. Nonsmooth switching occurs at grid-sampling cell boundaries, clipping thresholds, and minimum-area active-set changes; VJPs remain well defined through PyTorch's chosen subgradients. A 17² double-precision directional finite difference checked the path through the image-derived hint, base-map coordinates and radial decoder. No factorization of an \(N^2\times N^2\) matrix or saved Krylov trajectory is involved.

For topology, each A4 color pass uses the current positive signed area \(A_t\), opposite edge \(e_t\), proposed displacement \(d_v\) from its finite logit, and adverse area change \(g_t=\max(0,-\det(e_t,d_v))\). Restricting the scalar motion to \(s_v\le\gamma A_t/g_t\) for every adverse face gives \(A_t^{new}\ge(1-\gamma)A_t>0\). Same-color vertices do not share faces; four passes preserve every original face. The fixed injective boundary plus positive oriented PL faces then supplies the global planar homeomorphism theorem stated in the A4 document. A5's image-derived hint changes *which* finite logits are supplied but not this theorem.

## 4. Common 257²/512² image experiment

All figures below use **66,049 control vertices and 131,072 original triangles**, not a sparse control mesh with only dense image queries. Synthetic high32 pairs were generated from varying texture phases/centers and three bounded deformation coefficients. The train/test seeds were 55101/99317, with 32/8 disjoint pairs, one-channel 512² images and batch two. Training used only image mean-squared error (MSE) and 1,000 Adam steps, learning rate 0.003; target maps/coefficients were used solely for evaluation. The A5 run was initialized from the already trained A4 encoder, then all 1,024 encoder parameters were further optimized. Window width 5, ridge 10 and hint gain 1 were chosen using only the training split, with a minimum train-face-area ratio of 0.05 in a frozen-encoder screening run. This selection is exploratory, not a broad hyperparameter search.

| Candidate | Held-out image MSE | Held-out query-map RMSE | All-face centroid Beltrami RMSE | Minimum original-face area ratio | Maximum face-centroid \(|\mu|\) |
|---|---:|---:|---:|---:|---:|
| A4 image-only trained | 0.00053678 | 0.00197683 | 0.14059 | 0.27874 | 0.55118 |
| Frozen A4 encoder + one image-derived safe hint | 0.00007071 | 0.00115973 | 0.14956 | 0.11116 | 0.80678 |
| A5, another 1,000 image-only updates | 0.00006000 | 0.00122204 | 0.15245 | 0.06488 | 0.85276 |

The hint improves image and coordinate-map errors substantially on this held-out synthetic set, but worsens Beltrami error and maximum distortion. Continuing image-only training improves image MSE but makes map RMSE and distortion slightly worse than the frozen hint. Thus image fit alone is not an adequate surrogate for quasiconformal geometry. The A5 run's median full encoder+decoder+hint+query+warp/loss forward and VJP were 38.1/60.9 ms on Element four-thread CPU, with sampled process RSS 894 MB and 104.6 s for 1,000 updates. The corresponding A4 medians were 32.6/52.4 ms, RSS 880 MB and 88.0 s. These are batch-two CPU training measurements, not bare decoder microbenchmarks.

The high32 target's true fine displacement amplitudes in the eight test pairs have magnitude around \(10^{-3}\). Frozen A4 predicted projections were around \(10^{-5}\); A5's photometric branch made them around \(10^{-4}\), still well short of the target. The analytic target versus its *own sampled P1 interpolation* already has Beltrami RMSE 0.03832 at face centroids; A5's 0.15245 is not near this discretization floor. One known-basis image estimator can recover the synthetic coefficients nearly exactly, so the remaining high-frequency deficit is architectural/optimization-specific, not a proof of intrinsic image ambiguity.

## 5. Numerical limits and repeated passes

One might recompute the photometric residual after a safe pass and apply several more passes. In exact arithmetic, each pass remains an original-grid P1 homeomorphism. In float32, however, the original radial rule can drive areas toward zero so quickly that roundoff defeats the hypothesis of the next pass. With the same frozen A4 encoder and train-selected window 5/ridge 10/gain 1, train-set minimum area ratios for one through four sequential passes were 0.09353, 0.000631, \(5.96\times10^{-7}\), and **negative \( -8.34\times10^{-6}\)**. The four-pass map is invalid and is recorded as a failure, even though image MSE kept falling. Choosing pass count by train image MSE **subject to area ratio at least 0.05** retains only one pass. This illustrates why the exact-arithmetic theorem cannot substitute for a finite-precision topology check.

An experimental base-relative area-floor variant picks \(m=\rho\min_t A_t(Y^{(0)})\), with \(\rho=0.2\), once from the initial base P1 map and holds the same \(m\) across all local passes. At each vertex motion, the allowed adverse area loss in face \(t\) is restricted to

\[
\ell_t=\min\{\gamma A_t,\max(A_t-m,0)\},\qquad
s_v=\min\!\left(1,\min_{t:g_t>0}\frac{\ell_t}{g_t}\right).
\]

If the input faces have area at least \(m\), then each output face has \(A_t^{new}\ge\max((1-\gamma)A_t,m)\); induction gives an exact-arithmetic absolute area floor for any finite number of color/local passes. The global base minimum is differentiable almost everywhere and is not detached from the VJP. A small independent four-pass double-precision test checks the floor and finite gradient. The implemented ratio is evaluated as a clipped quotient to avoid inactive zero-adverse divisions poisoning gradients.

At 257² float32 on the same 32/8 split, the floor variant kept train minimum area ratios above 0.0628 for one to four passes and all observed faces positive. Two passes gave held-out image MSE 0.0000333, query-map RMSE 0.001106, Beltrami RMSE 0.1610, maximum \(|\mu|=0.938\), minimum area ratio 0.0732; four passes gave image MSE 0.0000192, map RMSE 0.001178, Beltrami RMSE 0.1922, maximum \(|\mu|=0.964\), minimum area ratio 0.0732. This cures the observed numerical fold but **not** the geometric overdistortion. Four passes are also roughly twice as expensive as one in the local Windows screening timing. The area-floor variant is not yet a trained neural layer or a finite-precision theorem.

## 6. GPU memory and the remaining gap

On the AI host's then-idle RTX A6000 GPU 2, float32, random latent standard deviation 0.08, fixed 512² image query, two warmups and 20 timed repeats, batch eight at 257² controls, A4's median complete decode/query/image-loss forward was 13.84 ms and VJP 21.02 ms; peak CUDA allocated/reserved were about 309/403 MB. A2+ on the same device, batch and repeat protocol was 12.72/15.78 ms and 166 MB allocated; A3 disk was 14.13/25.08 ms and 279 MB allocated. Separate A4 scaling runs reached 513² controls at batch two and 1025² controls at batch one with 512² queries; the latter used 515 MB peak allocated, but these shorter five-repeat measurements should not be ranked as though they used the 20-repeat protocol. GPU times exclude one-time mesh/table setup and are random-latent throughput tests, not A5 training or held-out accuracy tests. A separate A5 GPU training run is reported below.

The main remaining scientific issue is now clear: a fast dense fixed-P1 decoder with a correct topology theorem and usable gradients exists, but the available image-conditioned latent rule trades QC distortion against image fit and still under-recovers the true fine deformation. Further improvement should target a better data-consistency/latent inference mechanism with a geometric distortion control, not merely stack more unsafe-looking local passes or report lower image loss alone.

A direct 1,000-step A5 training repetition on that idle GPU 2, starting from the same CPU-trained A4 encoder, took 52.3 s total. Its batch-two median full training forward/VJP were 13.7/38.9 ms, peak CUDA allocated 342 MB, and sampled host-process RSS 1.53 GB. Held-out image MSE 0.0000601, query-map RMSE 0.001223, Beltrami RMSE 0.15247, maximum \(|\mu|=0.8527\), and minimum face-area ratio 0.0650 agree closely with the CPU A5 run. The GPU training result is stored separately in raw_results/a5_high32_gpu_warm_heldout257_1000_train.json and its all-face evaluation. This establishes that the A5 image-conditioned hint and encoder can backpropagate at the real 257² control scale with modest GPU allocation; it does not resolve the quality tradeoff.
