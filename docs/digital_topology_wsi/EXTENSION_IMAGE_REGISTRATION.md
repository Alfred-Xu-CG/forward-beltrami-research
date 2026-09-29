# Twelve-hour extension: image registration with a certified dense P1 decoder

Status: ongoing research record, 2026-09-30 (China time). All numerical results in this document are **exploratory**. The seven anatomy-labelled pairs have been opened repeatedly during development; they are not a blind test set. Every saved result cited here is on D: under `D:\QC_optimization_data\digital_topology_wsi\ACROBAT_train_subset\scaleup102\imageonly_extension`.

## 1. Exact objective and what the layer returns

The requested object is a learnable map from an image-dependent latent state to a **piecewise-affine homeomorphism on a fixed, dense triangulation**, with a usable vector-Jacobian product (VJP). In this extension the reference domain is the closed unit square \(\Omega=[0,1]^2\), sampled at \(v_{ij}=(i/256,j/256)\), \(0\le i,j\le256\). Every one of the \(256^2\) squares has the same top-left-to-bottom-right diagonal, giving 131,072 reference triangles. A predicted vertex array \(Y\in\mathbb R^{257\times257\times2}\) defines a unique continuous P1 map \(F_Y\): on each triangle, barycentric interpolation of its three vertex values. The output is the **full 257² vertex table**, not merely a 65² control map or a dense image sampling grid.

For a real image pair, the network first receives an external orientation-preserving similarity affine \(A(x)=Mx+b\) inferred from image-only SuperPoint/SuperGlue correspondences and RANSAC. The certified residual \(F_Y:\Omega\to\Omega\) is composed as \(A\circ F_Y\). The network branch after this supplied affine is differentiable in its weights and continuous match coordinates, and its output is P1 on the fixed triangulation. The *entire raw-image-to-deformation pipeline is not yet end-to-end differentiable*, because correspondence selection and RANSAC are discrete and external. Claiming otherwise would conflate two different notions of a differentiable layer.

## 2. Why every reported saved map is homeomorphic

For each reference vertex on the square boundary, the residual decoder keeps \(Y(v)=v\). Interior vertices are updated in four row/column parity classes: \((i\bmod2,j\bmod2)\in\{(0,0),(0,1),(1,0),(1,1)\}\). Within one class no two vertices share a triangle, so each affected oriented area is affine in that vertex's own displacement. At an interior vertex, let the current geometric central edges be

\[
e_x=\tfrac12(Y_{i+1,j}-Y_{i-1,j}),\qquad
e_y=\tfrac12(Y_{i,j+1}-Y_{i,j-1}).
\]

A latent \(z_i=(z_i^x,z_i^y)\) proposes \(r_i=8[e_x\tanh z_i^x+e_y\tanh z_i^y]\). For each incident oriented Q1-corner area \(a_k>0\), the area after moving only this vertex by \(t r_i\) is \(a_k+t\,d_k(r_i)\), where \(d_k\) is a known linear functional. The implemented smooth radial safety factor is based on the most restrictive negative area derivative and a local positive area budget (safety fraction 0.75; nominal minimum normalized Jacobian 0.05). There is **no global line search and no post-hoc fold repair**. The four classes are processed sequentially on the *current* deformed vertex table; the local edges are recomputed after earlier classes and passes. For an arbitrary proposed goal map, four passes steer toward that goal through this same safe operator.

This construction supplies a mathematical positive-area argument in exact arithmetic under finite inputs, positive initial map, fixed ordered boundary and the operator's stated area-budget conditions. The implementation also independently rechecks the **saved float32 bits**: all 262,144 Q1 corner determinants are strictly positive, the outer boundary equals the reference rectangle exactly, and the post-composed affine has positive determinant. The chosen-diagonal P1 triangles are therefore strictly positive, and the fixed-boundary planar disk map is globally homeomorphic, not merely locally fold-free. A nominal Jacobian floor is an implementation steering parameter, not a claim that every saved triangle satisfies the floor exactly after floating-point rounding. The certificate is geometry/topology, **not** evidence that anatomical landmarks are correctly registered.

## 3. Two different implementations: amortized neural output and per-pair multilevel fitting

### 3.1 Frozen image-only neural baseline and fast students

The common frozen baseline \(B\) uses a 512² fixed image and a moving image prewarped by the direct image-only affine. A 128² image feature extractor predicts coarse 17²/33²/65² safe levels, exactly refines to 257² and applies an independently proposed safe 257² head. The pretraining affines for the **new** baseline in this document were also generated directly from images; no DHR-derived initial affine or anatomical label is used by this baseline. Its frozen output, together with 88 image-feature channels and optionally a six-channel raster of machine-match offsets/density, enters a residual network. A four-pass residual decoder returns a 257² P1 map. One-shot inference does not run a per-pair optimizer.

The four student variants compared here have the same frozen baseline, 63 training and 18 case-ID-disjoint ACROBAT pseudo-teacher pairs, 1,000 AdamW minibatches, and four safe passes. The local trunk has three 3×3 convolutions; the global trunk downsamples 256→128→64→32→16 and upsamples with skips. A third design uses this global trunk to predict only interior latent controls at 17², 33² and 65², prolongs their zero-boundary fields, and safely steers the baseline toward their sum. It outputs the same 257² P1 map, but the learned geometric proposal has only 10,310 scalar controls instead of four independent full-grid latent fields. These are controlled architecture tests, not claims that U-Net or multiscale fields are new.

### 3.2 Pair-specific low-dimensional multilevel fitting

The strongest registration result so far in this extension is **not a one-pass learned network**. It freezes the same \(B\) and optimizes three two-component interior tensors \(c^{(17)},c^{(33)},c^{(65)}\), with

\[
N_c=2(15^2+31^2+63^2)=10{,}310.
\]

Let \(P_s\) zero-pad an interior \((s-2)^2\) tensor and bilinearly prolong its values to the 257² reference vertices. The unconstrained *goal*, not the output, is

\[
G_c=B+P_{17}c^{(17)}+P_{33}c^{(33)}+P_{65}c^{(65)}.
\]

The actual output is \(Y_c=T_4(B,G_c)\), where \(T_4\) means four sequential current-edge safety-steering passes. Consequently even if \(G_c\) folds, \(Y_c\) remains a certified fixed-grid P1 homeomorphism. This is a **forward sum of multiscale proposals followed by safe updates**; it neither inverts a 257² matrix nor composes separately triangulated maps and resamples them. Controls start at zero, so \(Y_0=B\). One diagnostic fit to a saved teacher map on ACROBAT case 233 reduced the baseline-to-teacher vertex RMSE from 0.003262 to 0.000764 unit after 100 steps; that supervised capacity check is not an anatomical result.

The actual landmark-blind registration objective uses a 256² eight-channel normalized self-similarity descriptor \(D\), a fixed tissue/texture mask \(w\), image-only aligned machine matches \((p_n,q_n)\), and a first-difference strain penalty:

\[
\begin{aligned}
L_{\rm image}(Y)&=\frac{\sum_{x,k}w(x)|D_F^k(x)-D_M^k(F_Y(x))|}
                         {8\sum_x w(x)+10^{-8}},\\
L_{\rm match}(Y)&=\frac1N\sum_{n=1}^N
  \left(\sqrt{\|512(F_Y(p_n)-q_n)\|_2^2+4^2}-4\right),\\
L_{\rm strain}(Y)&=\tfrac12\left[\operatorname{mean}\|256\Delta_x(Y-v)\|_2^2+
                              \operatorname{mean}\|256\Delta_y(Y-v)\|_2^2\right],\\
L(c)&=L_{\rm image}(Y_c)+0.05L_{\rm match}(Y_c)+L_{\rm strain}(Y_c).
\end{aligned}
\]

Here \(D_M\) is computed on the affine-prewarped moving canvas; \(q_n\) is correspondingly in that aligned frame. The image term samples the *P1 map* at 256² pixel-center queries, then bilinearly samples the descriptor. The match term is a smooth robust error in 512-canvas pixels. No anatomical landmark or DHR full field enters this objective. Adam learning rate is 0.002; the best iterate is selected by \(L\), never by anatomy. We predeclared 25 versus 100 steps to examine convergence. The same objective/step budgets were also used to compare zero-control starts with a neural-student map as a warm start.

## 4. Data, evaluation and provenance

The teacher-student training source is 81 ACROBAT image pairs, split deterministically by case ID into 63 train and 18 validation. The pseudo-teachers are 25-step per-pair multilevel fits from the frozen image-only baseline. All 81 teacher maps passed the saved topology certificate. Their supervision is *generated by the same images and machine matches*, not by anatomical landmark labels; teacher-map RMSE on the 18 validation cases measures amortization, not true registration accuracy.

Anatomy is measured on seven previously inspected development pairs: four stains (Cc10, CD31, Ki67, proSPC) against a single He image of one lung-lesion_3 specimen (80 corresponding points per stain); BIRL lesions (78), rat kidney (69), and HistoReg CD4/CD68 (77). The lung annotations were supplied at 50% image scale and converted to the provided 5% JPEG coordinates with the nominal center rule \(p_5=(p_{50}+0.5)/10-0.5\). Pixel-center conventions have subpixel uncertainty. For every fixed landmark \(x_n\), the predicted moving point is obtained by fixed-triangle barycentric interpolation of the 257² P1 map, application of the stored positive affine, and inversion of the recorded moving-canvas resize/padding. Target-registration error (TRE) is the mean Euclidean distance to its moving-image landmark in **native moving JPEG pixels**. The four lung errors can be averaged within that one common-scale specimen; the three BIRL/Histo errors must not be pooled into one physically meaningful mean because their source image scales differ.

These labels have been observed repeatedly in earlier rounds and in this extension. They support a feasibility diagnosis but not a clean generalization claim. The match archives do not all embed their affine metadata; the generation paths and baseline/teacher affine equality were checked, but machine-match frame consistency is not independently certified from each archive alone. External SuperGlue/RANSAC matching remains discrete and adds substantial upstream time.

## 5. Landmark results: full table, not best-case selection

The following table reports mean TRE in native moving JPEG pixels. Every row is one fixed method applied to all seven pairs; lower is better. "Lung mean" is the mean of the four lung-stain means, not an average over independent patients.

| Method | Lung mean | Cc10 | CD31 | Ki67 | proSPC | Lesions | Kidney | Histo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct image-only affine | 13.778 | 11.644 | 9.230 | 19.269 | 14.970 | 7.914 | 11.787 | 41.159 |
| Frozen image-only neural baseline | 12.865 | 10.596 | 7.595 | 18.455 | 14.813 | 7.806 | 9.869 | 32.596 |
| One-shot global-trunk full-grid student | 12.543 | 10.144 | 7.302 | 18.251 | 14.477 | 7.486 | 9.332 | 31.932 |
| One-shot global-trunk 17/33/65 latent student | 12.670 | 10.365 | 7.403 | 18.360 | 14.553 | 7.629 | 9.748 | 31.312 |
| Per-pair 17/33/65 fit, 25 steps from frozen | 12.345 | 10.312 | 6.822 | 18.134 | 14.112 | 7.208 | 8.282 | 23.228 |
| Per-pair 17/33/65 fit, 100 steps from frozen | 12.356 | 10.312 | 6.887 | 18.191 | 14.035 | 7.112 | 7.929 | 22.240 |
| Per-pair 17/33/65 fit, 25 steps from global neural student | 12.417 | 10.198 | 6.954 | 18.361 | 14.156 | 7.291 | 7.881 | 20.852 |

The 25-step fit beat its *matched frozen baseline* and the initial affine on 7/7 pairs. It beat the one-shot global student on 6/7 pairs (Cc10 is slightly better with the student). The 100-step objective was lower on all seven pairs than the 25-step objective, but CD31 and Ki67 TRE **worsened** by 0.064 and 0.058 px, respectively. Neural warm-start plus 25 steps also lowered the unlabeled objective relative to cold 25 steps on all seven, yet improved anatomical TRE on only three of seven. These are direct signs that objective quality and anatomical accuracy are not monotonically equivalent; more optimization or a larger network alone is not the whole answer.

### 5.1 What the network did and did not learn

On the 18 ACROBAT validation IDs, the frozen baseline's mean vertex RMSE to the 25-step pseudo-teacher was 0.003068 unit. A map-plus-image/match/strain-trained global full-grid student reduced that to 0.002828, while the global low-dimensional 17/33/65 student reached 0.002838. These are real case-disjoint improvements but recover only a small fraction of the per-pair teacher deformation. The global student's validation image term changed from 0.221995 to 0.219403 and robust match term from 0.941311 to 0.647425; its teacher's corresponding image and match terms were about 0.200981 and 0.057974. This large remaining gap is the core amortization bottleneck.

A controlled stronger machine-match training weight (0.01→0.1 with all else fixed) reduced the global student's validation match term from 0.647 to 0.614, but worsened its teacher-map RMSE from 0.002828 to 0.003018 and its lung anatomy mean from 12.543 to 12.888 px; Histo improved from 31.932 to 29.474 px. The same direction held for the low-dimensional head (validation match 0.642→0.601, teacher-map RMSE 0.002838→0.002960, lung 12.670→12.884, Histo 31.312→28.683). Thus simply increasing sparse-match supervision trades off tissue groups; it is not a universal fix.

Two additional global networks were trained **directly** against the pair-registration objective \(L_{\rm image}+0.05L_{\rm match}+L_{\rm strain}\) for 2,000 minibatches, with no teacher-map MSE term. On validation, full-grid and low-dimensional total objectives were 0.26457 and 0.26270, respectively, versus 0.26367 and 0.26218 for their 1,000-step pseudo-teacher-trained counterparts; both remained below the same frozen baseline 0.27889. Direct training therefore did not close the amortization gap. It also did not reliably improve anatomy: full-grid lung mean was 13.059 px (proSPC 15.209, worse than its affine 14.970); low-dimensional lung mean was 12.875, though Histo improved to 25.289. These negative or mixed outcomes are retained, not hidden by selecting the best pair.

### 5.2 Fast dual sparse-match multilevel proposal

A separate explicit proposal fits multilevel controls to machine-match residuals with a small dual system. Let \(\Phi\in\mathbb R^{N\times5155}\) contain the **exact fixed-diagonal 257-P1 interpolation weights** of the three prolonged interior hat bases at the \(N\) fixed match points, and let \(d_n=q_n-F_B(p_n)\). The two-component ridge solution is

\[
c=\Phi^T(\Phi\Phi^T+\lambda I_N)^{-1}d.
\]

The resulting proposed goal is passed through four current-edge safety steps, so solving the ridge problem does not assert that the unconstrained goal itself is homeomorphic. The 18 case-disjoint ACROBAT validation pairs had 18--94 matches each. For \(\lambda\in\{0.001,0.01,0.1,1,10\}\), the frozen baseline's teacher-map RMSE was 0.003068 unit; the dual method yielded 0.002715, 0.002715, 0.002718, 0.002775, and 0.002966. When 20% of matches were withheld from each fit, the baseline robust held-out match loss averaged 1.055 canvas pixels; the five dual values were 0.881, 0.881, 0.881, 0.906 and 1.004. This small-system method fitted the pseudo-teacher map at least as well as the tested amortized networks on that metric, but the validation teacher is not an anatomical ground truth.

All five ridges were applied to all seven already opened anatomy pairs, avoiding a landmark-selected ridge. Their lung mean TREs were 12.664, 12.662, 12.646, 12.624 and 12.756 px; Histo TREs were 30.090, 30.036, 29.595, 28.243 and 30.472 px. The dual method beat the initial affine on every pair at each ridge but did not approach the 25-step image-driven optimizer's 12.345 lung mean or 23.228 Histo. The 35 maps all passed saved exact Q1 certification and had finite match-coordinate VJPs. The fast branch is therefore a plausible **differentiable sparse-match initialization layer**, not a demonstrated one-shot solution to complex cross-stain registration. A 10-step image fit initialized from the validation-favoured \(\lambda=0.1\) dual map yielded lung mean 12.399, kidney 8.348 and Histo 24.941 px, versus 12.370/8.470/28.101 for a 10-step fit from the frozen map. Again, gains are tissue-dependent.

## 6. Forward/VJP cost, memory and what the timer excludes

All reported layer times are GPU measurements for batch one at 257² control/output resolution. The one-shot global full-grid student needed about 48 ms forward; its forward-plus-parameter-and-selected-match VJP was about 150--290 ms depending on the pair/run, with 159 MiB forward and 1.18 GiB VJP peak `torch.cuda.max_memory_allocated` for Cc10. The one-shot 17/33/65 latent student needed about 46--48 ms forward and about 150--287 ms forward-plus-VJP, with 150 MiB forward and 784 MiB VJP peak for Cc10. Its lower VJP allocation did **not** translate to a clear anatomical accuracy gain.

The 25-step per-pair multilevel fit required 1.66--2.09 s across the seven pairs; 100 steps required 6.68--7.22 s. The 25-step Cc10 PyTorch peak allocation was 395 MiB. This makes it substantially faster than the earlier full-grid, 300-step, four-pass fit (18.9--21.8 s/pair, approximately 389 MiB PyTorch peak), but it remains a test-time iterative method, not a 50-ms neural layer. A 10-step fit took roughly 0.68--0.96 s, but did not consistently match the 25-step anatomy. The fitting timer starts after input loading, affine prewarp and descriptor creation; the one-shot layer timer includes its feature computation from decoded tensors, but not image decoding, external matcher/RANSAC, saving or exact post-save certification. PyTorch allocated-memory peaks are not total GPU residency and exclude external matcher memory. A separate Cc10 direct affine matcher/RANSAC measurement was about 0.56 s plus approximately 0.60 s one-time model loading; the additional aligned machine-match process is also not included in the decoder numbers.

For the dual ridge layer, after the initial warm-up outlier, the measured GPU forward was about 29--31 ms per pair and its backward alone about 45--54 ms, including the N-by-N solve and safe steering but not external correspondence generation, file I/O or saved-map certification. These **must be added** for a forward-plus-VJP cost of roughly 75--85 ms; reporting the backward alone as the whole gradient cost would be misleading. The one first-call Cc10 measurement was 231 ms forward/163 ms backward due to initialization effects, so this is not a cold-start latency claim.

## 7. Independent checks and current interpretation

An independent checker recomputed all 14 cold multilevel25/100 anatomy scores directly from raw landmark CSVs and saved maps (maximum discrepancy below \(5.4\times10^{-14}\) px), rechecked exact saved-binary Q1 signs and fixed boundaries on every map, and confirmed positive, bitwise-matched post-affines. All 14 maps had 262,144/262,144 positive Q1 corners. The warm/student rows have passed the same saving certificate in their own execution paths but are still awaiting an equally separate full score audit. The dual ridge method's mathematics and ridge-0.1 seven-pair scores are under a separate independent check. A direct numerical finite-difference test agreed with the low-dimensional neural decoder's analytic VJP to within 3% on a representative 257² direction; this is useful evidence but not a proof of correct gradients at all non-smooth safety boundaries. The teacher generation and seven-pair optimizer paths read images, image-derived matches and saved image-only baselines; they do not read anatomy or DHR full fields. The capacity-only ACROBAT teacher-map fit is explicitly separate from anatomical evaluation.

The clearest measured bottleneck is now **amortization of pair-specific geometric evidence**, not an inability of the fixed 257² P1 safety decoder to express useful deformations. The per-pair multilevel representation can fit a target deformation and improves all seven existing cases at 2 s/pair, while 45--50 ms students trained on only 63 ACROBAT IDs recover a much smaller fraction of that improvement. The present image/machine-match objective also has documented mismatches with anatomy. The task is therefore **not finished**: we have neither shown strong blind cross-specimen anatomical generalization nor made the entire raw-image pipeline differentiable, and the best anatomy results still require per-pair iterations.

## 8. Reproducibility pointers

- Per-pair multilevel optimizer and seven-pair driver: `tools/digital_multilevel_safe_pair_optimize.py`, `tools/digital_multilevel_seven_batch.py`.
- Dual sparse-match layer and validation: `tools/digital_dual_multilevel_match_safe257.py`, `tools/digital_dual_multilevel_validation.py`, `tools/digital_dual_multilevel_seven_batch.py`.
- ACROBAT pseudo-teacher generation: `tools/digital_multilevel_acrobat_batch.py`; student training: `tools/digital_matchopt_distill257.py`.
- Safe CNN decoder and one-shot inference: `tools/digital_mind_recurrent257.py`, `tools/digital_frozen_residual_predict.py`, `tools/digital_frozen_residual_seven_batch.py`.
- Landmark scoring: `tools/digital_lung_lesion3_score.py`, `tools/digital_birl_three_score.py`.
- Tests: `tests/test_digital_multilevel_safe_pair_optimize.py`, `tests/test_digital_frozen_residual_start.py`, `tests/test_digital_dual_multilevel_match_safe257.py`.
- Raw maps, JSON reports, student checkpoints and qualitative Cc10/Histo overlays are stored under the D: path stated at the top. Large inputs and maps are not embedded in Git. The Git repository contains the source, formulations and the compact result summary.
