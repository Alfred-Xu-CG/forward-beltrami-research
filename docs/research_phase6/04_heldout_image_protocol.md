# Image-conditioned latent inference: held-out protocol

The fixed-pair experiments establish that gradients reach a dense decoder and can reduce an image loss. They do **not** establish that an encoder learns to infer deformation from unseen images. This protocol separates those questions. It is implemented in [the multi-sample experiment](../../tools/phase6_train_multisample_image.py). All records below use the same final control triangulation for both methods: \(257\times257=66{,}049\) vertices and \(2(256)^2=131{,}072\) faces. Each sample has a 512×512 moving scalar image, a fixed image, and 512² evaluation queries.

## 1. Data generation and independent target

For each sample, a local pseudorandom generator draws four texture phases, four Gaussian centers, four texture amplitudes, and three deformation amplitudes. The moving image is a sum of two oblique sinusoidal waves, two Gaussian blobs, and a high-frequency product of sinusoids; the full formula and seed are in the script. Train items 0–31 use seed 55101; held-out items 0–7 use seed 99317. Thus neither image tensor nor deformation coefficients are shared between splits.

Let \(x,y\in[0,1]\), \(b(x,y)=\sin(2\pi x)\sin(2\pi y)\), and \(h(x,y)=\sin(16\pi x)\sin(16\pi y)\). The independently specified target map is

\[
F_\star(x,y)=\bigl(x+a_x b(x,y)+a_f h(x,y),\;
                  y+a_y b(x,y)+a_f h(x,y)\bigr),
\]

with \(a_x\in[0.012,0.035]\), \(a_y\in[0.025,0.055]\), and \(a_f\in[-0.002,0.003]\). Write \(F_\star=\mathrm{id}+u\). Since \(\|\nabla b\|_2\le2\pi\sqrt2\) and \(\|\nabla h\|_2\le16\pi\sqrt2\),

\[
\|Du\|_2\le
\sqrt{a_x^2+a_y^2}\,2\pi\sqrt2
+\sqrt2|a_f|\,16\pi\sqrt2
<0.882<1.
\]

Therefore \(\|F_\star(p)-F_\star(q)\|_2\ge(1-0.882)\|p-q\|_2\) for any points in the square. The sine products vanish on the boundary, so \(F_\star\) fixes the boundary. In exact real arithmetic it is an orientation-preserving bi-Lipschitz homeomorphism of the square onto itself. The fixed image is generated once as \(I_{\rm fixed}(x)=I_{\rm moving}(F_\star(x))\), using bilinear image sampling. Numerical raster sampling is not an exact continuum identity.

The true map is never an encoder input or a training target. The optimizer sees only \((I_{\rm fixed},I_{\rm moving})\), and minimizes the mean squared image difference after warping the moving image by the predicted map. A code path computes target-map error only in evaluation mode, outside the training forward graph.

## 2. Methods and measurements

Method A is the one-layer vertical monotone decoder. Its output is P1 on the stated original grid and is a homeomorphism for every valid represented positive-increment latent. Method AB2 is one vertical then one horizontal full-grid monotone decoder, evaluated by exact dynamic-coordinate composition; it is a PL homeomorphism, but need not be P1 on the original grid. Both use the same two-convolution image encoder width, coarse latent side 33 plus fine latent side 257, Adam rate 0.003, 1000 updates, batch 2, and the same CPU. Independent encoder output heads reflect their different latent spaces; parameter counts should be reported separately in the final comparison.

Image MSE is the average squared intensity difference over every pixel and held-out sample. Query-map RMSE is the square root of the average squared coordinate-component error against \(F_\star\); it is evaluation-only. The reported minimum signed-area ratio is the minimum output-face signed double area divided by the source-face signed double area, checked **per represented layer**. A positive minimum plus the analytic layer construction establishes each layer's represented topology; it does not certify an AB2 map reinterpolated as P1 on the original grid. Forward timing covers encoder, decoder, query evaluation, one image resampling and loss; backward timing covers the first-order VJP. One-time data and query-table construction and held-out evaluation are excluded from per-step times. CPU peak is sampled resident-process memory, not an allocator-exact peak.

## 3. Clean 1000-step 257² results

The first two exploratory runs computed, but did not backpropagate, an unused target-map metric during training forward. The script was corrected so training never reads the true map; both methods were then rerun from the same deterministic initialization and samples. The following are those clean reruns on Element CPU, 32 train and 8 held-out samples:

| Method | Held-out image MSE initial→final | Held-out map RMSE final | Train seconds | Median forward/backward | Sampled peak RSS |
|---|---:|---:|---:|---:|---:|
| A | 0.03713→0.01262 | 0.01259 | 33.56 | 12.9/19.9 ms | 821 MB |
| AB2 | 0.03729→0.01187 | 0.01215 | 82.16 | 35.2/45.7 ms | 980 MB |

AB2 has modestly better held-out quality after the same number of updates, but its time and memory are higher. The unmodified per-sample initial loss is different because independently initialized A and AB2 maps differ slightly; the *data* are identical. This is one synthetic family and one seed pair, not evidence of generalization to medical images. A final quality-vs-wall-time comparison must also consider that a 1000-step AB2 budget permits more A updates. Full machine-readable results are in [A](raw_results/heldout_A_257_512_element_cpu_1000.json) and [AB2](raw_results/heldout_AB2_257_512_element_cpu_1000.json).
