# Independent BIRL saved-result review

2026-09-29. Read-only review of both newly acquired BIRL development pairs, DHR saved fields/configuration/metadata, the O and affine-H NPZ maps, teacher targets, and evaluation JSONs. Critical calculations used independent NumPy/SciPy interpolation, cell-wise numerical inversion, and arbitrary-precision integer orientation signs; no production evaluator was invoked. No registration was rerun and no implementation was edited. **No coordinate-direction or reported-TRE discrepancy was found for these artifacts.**

## Pairing, 5% coordinates, and evaluation denominator

The pinned [BIRL pair manifest](https://raw.githubusercontent.com/Borda/BIRL/f1648b293664e3bea50736280a19b846d60841f1/data-images/pairs-imgs-lnds_histol.csv) specifies HE as fixed/target and proSPC or PanCytokeratin as moving/source. The maintainers' [landmark documentation](https://borda.github.io/dataset-histology-landmarks/) describes ImageJ X,Y coordinates, a top-left origin, and same-stem images/CSV files grouped by scale. These CSVs belong to the supplied **5%-scale JPEGs already**; multiplying their coordinates by .05 again would change the evaluation frame incorrectly.

| Pair | Moving JPEG size xy | Fixed JPEG size xy | Landmark file counts moving/fixed | Evaluated matched IDs |
|---|---|---|---:|---:|
| `lesions_` proSPC to HE | 891x735 | 890x733 | 78 / 78 | 78 |
| `rat-kidney_` PanCytokeratin to HE | 1123x724 | 1164x787 | 69 / 71 | 69 |

I independently parsed the original CSVs, rejected duplicate IDs in the audit, and checked all coordinates against their own decoded image bounds. Kidney IDs 70 and 71 exist only in the fixed CSV. The explicit ID intersection and disclosed missing IDs are appropriate for this development diagnostic; they are not two failed registrations or a 71-point evaluation. Use the same 69 matched points for identity, O, DHR and H. Bounds alone cannot prove a CSV has the correct scale, so the paired manifest and matching scale directories matter.

All TRE numbers below use supplied fixed-JPEG pixels, not scanner-level pixels, micrometers or an official ANHIR score. These two public sets do not establish a patient/animal-independent split or held-out generalization.

## DHR coordinate chain, independently derived from the actual resampler

The installed DHR package and checked-out source both use `F.interpolate(scale_factor=1/r, recompute_scale_factor=False, align_corners=False)` in `dhr_utils/utils.py:64`. `basic_preprocessing` sets r, and `full_resolution.py:261` saves `current_displacement_field` directly; the saved field has not been expanded back to the padded original-resolution image. Therefore a loaded-image pixel-center coordinate p with loading scale s and left/top padding vector P reaches field coordinates by

`z = ((p + 0.5) * s + P) / r - 0.5`.

Its inverse is `p = ((z + 0.5) * r - P) / s - 0.5`. The saved field represents `B(z_fixed)=z_fixed+displacement(z_fixed)`, so a **moving** landmark is first converted to z_moving, then B must be inverted, then the fixed-coordinate inverse conversion is applied. Adding the backward displacement at a moving landmark is not this inverse.

- Lesions: s_m=s_f=1, r=1, moving P=(0,0), fixed P=(0,1), saved field shape `(2,735,891)`.
- Kidney: s_m=s_f=1, **r=787/768=1.0247395833333333**, moving P=(20,31), fixed P=(0,0), saved field shape `(2,768,1135)`. The dimensions are consistent with the padded 1164x787 canvas and the actual scale factor; replacing r by a ratio inferred independently from rounded width would be incorrect.

Example kidney ID 1: moving `(62,290)` becomes field point `(80.0082592122,313.2382465057)`. Independently inverting the field gives fixed field point `(58.2243513153,304.3770780552)`, hence fixed JPEG prediction `(59.6771672984,311.9196099342)` and TRE **4.423272469** pixels. Kidney ID 69 predicts `(143.4917849330,394.8243624393)` and TRE **.968429649** pixels. Lesion ID 1 predicts `(34.0345749277,78.0608642977)` and TRE **25.690592151** pixels. This also checks padding-axis order.

For inversion I built each cell from the saved coordinate nodes, selected candidate cells by their four-corner bounding boxes, and solved the bilinear equations with SciPy using five starting locations per candidate. All evaluated points yielded one distinct accepted root and agreed with the saved evaluator predictions. This numerical exercise is not a proof of global injectivity for the folded DHR fields.

## Independently recomputed results

| Pair | Method | Mean TRE px | Median px | P95 px | Maximum px |
|---|---|---:|---:|---:|---:|
| Lesions, 78 points | Normalized identity | 75.5799412150 | 64.6697312572 | 136.2698863454 | 161.8044169961 |
| Lesions | O, saved safe Q1 | **64.7495658982** | 58.3410435288 | 133.8906546891 | 166.8623596012 |
| Lesions | DHR saved field | **8.0683228205** | 6.4886487089 | 21.0971431051 | 34.4000201354 |
| Lesions | H, affine after safe Q1 | **8.0778409795** | 6.4856644711 | 21.1078873043 | 34.3317640628 |
| Kidney, 69 points | Normalized identity | 14.1068516090 | 13.6594088583 | 25.7052557285 | 36.7036442977 |
| Kidney | O, saved safe Q1 | **23.4056230901** | 21.6971612426 | 49.1677697933 | 63.3732531503 |
| Kidney | DHR saved field | **5.5762336883** | 4.1465464169 | 18.7607943989 | 21.0511637697 |
| Kidney | H, affine after safe Q1 | **5.5686627369** | 4.0190479087 | 18.6018671629 | 21.2080177316 |

Maximum prediction discrepancy from the saved JSONs was `5.69e-13` pixel across these checks. DHR inverse reconstruction residuals were at most `3.42e-13` field pixels. For H, I inverted the affine first and then the residual, rather than invoking the evaluator's composite-table materialization. Independently sampling DHR at all 257² teacher sites with the full r/padding chain agreed with the stored raw teacher arrays to maximum **1.10e-7** unit for lesions and **1.86e-7** for kidney, consistent with their float32 preparation.

## Stored topology and exact affine signs

I converted every binary32 residual coordinate to a common dyadic integer scale, then used Python arbitrary-precision integer determinants for all four corner triangles. Every O/H residual had **262,144 positive / 0 nonpositive** corners, and every residual boundary was independently checked against the complete ordered identity rectangle. Common denominators were 2^34 (lesion O), 2^35 (kidney O), and 2^36 (both H residuals).

The two stored float32 H affine determinants are exactly

- Lesions: `4369753492216013 / 4503599627370496 > 0`, approximately **.970280187799**.
- Kidney: `282349415848567581 / 288230376151711744 > 0`, approximately **.979596320202**.

Thus O is a Q1 homeomorphism of the unit square onto itself; H's factored exact interpretation is a Q1 homeomorphism onto the affine image parallelogram. H does not have the same output-boundary contract as O. In float64 materialization, **5,199 lesion H control nodes** and **436 kidney H control nodes** lie outside the moving unit square; this is not a fold count, and an image warp must declare its outside-image behavior. The exact residual-plus-affine certificate does not automatically certify a separately rounded/exported composite array.

Independent determinant calculations on the original DHR field found **473/653,260 lesion cells** and **140/869,778 kidney cells** with at least one nonpositive corner. These match the reported failing-cell counts. The local corner order used here was SW,SE,NE,NW, whereas the DHR JSON uses top-left,top-right,bottom-left,bottom-right; the last two counts consequently appear swapped when comparing vectors, not when comparing cells. The accurate DHR landmark results do not provide a global Q1 guarantee.

## Deliberate failure fixtures

1. **Omit the kidney initial-resolution factor.** I deliberately used r=1 for both entering and leaving the otherwise unchanged real saved field. Only **68/69** labels then had a unique found inverse; the mean over those survivors was **6.3451** pixels rather than the correct complete-set 5.5762. This survivor mean must not be reported as a valid 69-point result. The lesions case, where r actually equals 1, remains unchanged and acts as a control.
2. **Use backward displacement as a forward transform.** I deliberately sampled B's displacement at each moving field point and added it instead of solving B(z)=z_moving. Mean TRE became **148.5741** pixels for lesions and **40.9583** pixels for kidney, versus the correct 8.0683 and 5.5762. This catches a direction error that positive local determinant values would not reveal.

The current production coordinate chain avoids both attacks. These deliberate computations altered no files or registered maps.

## Verdict and limits

The saved-result arithmetic and conditional factored-map topology claims pass this independent check. O improved mean TRE on lesions but worsened it on kidney relative to the appropriate normalized identity. H closely reproduces each image-derived DHR teacher's landmark accuracy while providing the different, explicitly declared safe residual-plus-affine representation. This is useful teacher-assisted evidence, not an independent network or a same-initialization comparison attributing an accuracy change solely to geometry. Include DHR cost in H inference, retain all matched labels, keep unmatched-ID counts visible, and do not call these public development cases a formal ANHIR leaderboard or generalization result.

No correction to the current saved metrics was required. NCC rankings, frozen-N results, and other datasets were outside this review's requested recomputation.

## Focused geometry merge appendix: image network and positive affine head

Independently inspected `q1_image_network.py`, its inherited feature encoder, the Q1 pyramid and composite archive reader. For raw affine logits z, the head implements `M=L diag(sx,sy) R`, with `L=[[1,0],[l,1]]`, `R=[[1,u],[0,1]]`, positive `sx=exp(alpha*tanh(z0))`, `sy=exp(alpha*tanh(z1))`, and bounded shears. Direct expansion cancels the shear products and gives **det M=sx*sy>0**. For finite logits and default alpha=.3, the exact determinant lies within `[exp(-.6),exp(.6)]`. With finite accepted bounds up to 2, it lies within `[exp(-4),exp(4)]`. The translation does not affect this sign.

The network returns residual vertices and affine factors separately; `apply_affine` has the correct row/column convention for `M U+b`. Conditional on a valid residual U, affine postcomposition remains Q1 on each original source cell and is homeomorphic onto `A([0,1]^2)`. It need not remain inside or cover the moving image square. This is an exact-arithmetic representation theorem, not a blanket proof about every stored float32 factor or materialized vertex table. Finite **raw affine logits** are a premise; bounded tanh outputs do not repair NaNs produced earlier in the CNN. Continue checking the actual residual, complete boundary and exact determinant of saved affine data.

**Confirmed defect and independently checked correction.** The first inspected constructor accepted `max_log_scale=1000`, `max_shear=1e30`, and `max_translation=inf`. With finite unit bias logits and finite zero features, these produced infinite matrix entries or offsets. The builder subsequently restricted every bound to finite `(0,2]`. I reran all three attacks plus a NaN bound; all now raise `ValueError`. Testing all 16 sign combinations of extreme finite geometric logits +/-1000 gave finite float32 factors with exact rational determinants positive: minimum .54881160385 at default scale/shear .3 and .01831563640 at the permitted upper bound 2. These sample checks confirm the repair; they are not an exhaustive floating-point proof.

**Gradient reachability.** On a separate float64 9→17 control-grid experiment with random 16² grayscale inputs, all four seed heads, the refinement head and the affine output head had finite nonzero gradients initially. The shared encoder stem and affine hidden layer had exactly zero gradient on that first pass because the output-head weights start at zero; this is expected chain-rule behavior, not permanent detachment. After one SGD update they had nonzero gradient sums approximately `1.97e-6` and `2.57e-7`. A combined direction through affine bias, seed/refinement biases and encoder weights matched central differences at h=`1e-4,1e-5,1e-6`, with absolute errors `1.21e-10,1.20e-12,5.43e-13`. This demonstrates reachable derivative paths on the tested branch, not conditioning, absence of tanh saturation, or learning benefit. Hard safety min/max operations remain only almost-everywhere differentiable.

**Actual stored-file failure fixtures.** Temporary NPZ fixtures were created and removed after the check, without changing production or permanent tests:

1. Identity residual plus stored `M=diag(-1,1)` has exact determinant -1. The composite reader reports residual-valid=true but affine-positive=false and composite-valid=false. This is an orientation-preservation failure; a reflection itself is still bijective.
2. A fixed-boundary 3x3 residual on `[0,1]^2` with its center moved from `(.5,.5)` to `(.125,.125)` has corner determinant **-1/8**, although its stored affine is identity. The reader reports one nonpositive corner and composite-valid=false.
3. Identity residual plus `M=diag(0,1)` collapses one coordinate. The residual alone passes, but the composite reader rejects the singular factor. This reinforces why residual-only validity must never stand for the whole represented map.

After the bound-validation repair, no blocking defect was found in the reviewed positive-affine formula or the tested default network composition/gradient paths. The initial combined check could not start in the base Anaconda interpreter because of duplicate OpenMP runtimes; all reported checks ran in the existing clean DHR environment without setting an unsafe duplicate-runtime override. No production code was edited by this checker.

## Independent three-example leave-one-out pilot check

This section records the **final report-bearing rerun**, whose `*_train.json`, `*_infer.json`, `*_eval.json`, weight archives and maps share the three `q1_n_*train_*eval` prefixes. The first-run results (HistoReg 76/77 at conditional 713.276 px, lesions 72.453 px, kidney 66/69 at conditional 54.084 px) were independently reproduced but are **superseded**; they do not describe the current saved maps. The rerun changed both errors and coverage, so these few runs cannot support a stability estimate.

Independently inspected the group-training, checkpoint/inference and landmark-evaluation paths. Training builds a fresh network and chooses its checkpoint only by training teacher-coordinate MSE; it loads the two image pairs and **raw** teacher vertex tables, not affine-residual teacher coordinates. Neither training nor frozen inference reads landmark CSVs, and frozen inference accepts no teacher. The teacher targets and network output both run fixed-unit to moving-unit; label evaluation must invert that map. Pixel centers use `(p+.5)/size` and the inverse conversion `u*size-.5`, with the original supplied JPEG dimensions. No direction, size-axis or pixel-center discrepancy was found. Kidney evaluation still requires the explicit 69-ID intersection of 71 fixed and 69 moving labels; the unmatched fixed IDs are 70 and 71.

I did **not** call the production evaluator or inverse solver for these critical calculations. I inverted each stored affine first, solved the residual bilinear cell equations with SciPy from five starts per candidate, and converted back to fixed-JPEG coordinates. All accepted residuals were below `1.6e-16` unit; maximum disagreement with the saved evaluation predictions was `2.73e-12` fixed pixels. The three final saved held-out maps reproduce the reported results:

| Frozen evaluation pair | Unique / matched | Complete mean TRE px | Conditional mean TRE px | Identity mean on the **same surviving IDs** px |
|---|---:|---:|---:|---:|
| HistoReg, trained on lesions+kidney | 77/77 | 604.1889615985 | 604.1889615985 | 464.1772114076 |
| Lesions, trained on HistoReg+kidney | 78/78 | 72.6702573079 | 72.6702573079 | 75.5799412150 |
| Kidney, trained on HistoReg+lesions | 68/69 | undefined | 55.2098535732 | 14.0223007624 |

Thus the text's worse-than-identity statement holds even with identical survivor subsets, not merely by comparing a conditional network mean to a complete identity mean. For spot checks, HistoReg ID 1 predicts `(5579.89344806,5983.99280593)` with TRE `675.41383781`; lesion ID 1 predicts `(13.52376541,223.34564954)` with TRE `158.03035641`; kidney ID 1 predicts `(80.86666274,228.74473315)` with TRE `82.21998233` pixels.

**The remaining missing inverse is a genuine coverage failure.** Using exact decimal CSV coordinates, exact integer image sizes and exact rational interpretations of the stored binary32 affine coefficients, I found `A^{-1}(moving landmark)` outside the residual square for kidney ID 57: its coordinates are approximately `(.556657960289,-.0156676541864)`. The negative sign was decided with rational arithmetic, not a solver tolerance. Since the residual is onto the square, this point provably has no inverse in the represented domain. Reporting the kidney full-set TRE as undefined, while exposing the surviving-point mean, is correct; neither dropping the point nor clamping it to the boundary would be an honest complete-set evaluation. Unlike the superseded first run, all HistoReg points and kidney IDs 26/41 now have an inverse.

**Stored topology independently passes.** Common dyadic integer denominators for final HistoReg/lesion/kidney residuals were `2^34/2^34/2^36`. Every residual had 262,144 strictly positive exact-integer corner determinants and the complete ordered identity boundary. The exact stored affine determinants were respectively `564066045915690525/576460752303423488`, `140350143392379345/144115188075855872`, and `4278944311371567/4503599627370496`, all positive. Independent float64 materialization counted `537/219/4046` control vertices outside the moving square, agreeing with the report. The certified object is the stored factored Q1 map onto its affine parallelogram; a rounded materialized table and an image-domain coverage guarantee are different claims.

**Actual fold isolation.** The original lack of saved training stdout was resolved by the report-bearing rerun. I read all three final training reports and checked their actual `train_pairs`: lesions+kidney, HistoReg+kidney, HistoReg+lesions, with matching teacher names and correct original image sizes. Their paired inference reports name only the respective omitted image pair and matching weights; no omitted teacher or label is an inference input. Every training report records batch 2, seed 290929, 800 steps at .002 and 800 finite gradient steps. Their best vector RMSEs are `.0060102713/.0032200362/.0071538003`, and median steps `83.3607/75.3871/82.0376` ms. These records plus the inspected source support computational fold separation; they are ordinary run evidence, not protection against arbitrary falsification. Because all three label sets were already inspected and authoritative patient/block relationships are unavailable, the correctly isolated run remains a leave-one-**example**-out development diagnostic, not a blind patient-held-out result. The reported negative transfer and responsiveness-versus-utility distinction are supported; architecture-wide or population-level conclusions are not.

## DHR initial-only and affine-only output check

The three saved initial-only configs differ from their respective full-DHR configs only in `run_nonrigid_registration=false`, case name and log path. No changed matcher, loading or nonrigid hyperparameter was hidden in this config comparison. Runtime records confirm zero nonrigid-stage time.

Using the actual saved MHA fields and postprocessing parameters, I independently rebuilt fixed-field-to-moving-field Q1 nodes and inverted their cells with SciPy, then reversed the loading/padding/resampling chain. Separately, because each saved safe residual is **exactly the full identity grid**, I inverted its stored affine directly by a 2x2 linear solve; neither calculation calls the production evaluator. This checks all matched labels, not a sample:

| Pair | Unique / matched, both outputs | Initial-field mean TRE px | Affine-only mean TRE px | Signed change px |
|---|---:|---:|---:|---:|
| HistoReg | 77/77 | 46.3463982652 | 46.3479377514 | +.0015394862 |
| Lesions | 78/78 | 7.6650061966 | 7.6650266525 | +.0000204559 |
| Kidney | 69/69 | 10.3527624380 | 10.3526967474 | -.0000656906 |

Maximum disagreement with either saved evaluation JSON was `1.82e-12` fixed pixels. The kidney chain retains `r=787/768`, moving padding `(20,31)` and fixed padding `(0,0)`; HistoReg retains loading scale `.1`, moving padding `(0,1)` and fixed padding `(15,0)`. These matter before inversion. For two explicit spot checks, HistoReg ID 1 predicts `(5490.74522503,6677.11804972)` from the initial field versus `(5490.75581242,6677.11623459)` from the affine; kidney ID 1 predicts `(52.61145346,315.01447458)` versus `(52.61226170,315.01436132)`. The maximum individual prediction displacement between representations was `.02300/.001185/.001237` pixels; a tiny *mean TRE change* is not a bound on every point.

Independent bilinear sampling of the original fields at all 257² fixed-unit sites agreed with each stored raw teacher within `1.47e-7/1.14e-7/1.82e-7` unit. Re-fitting ordinary unweighted least squares to those raw teachers reproduced the stored affine matrices/offsets within `5.89e-8`; this confirms fixed-to-moving orientation and image-only fitting. Pullback RMS depends slightly on whether the rounded saved matrix or the pre-rounding fit is inverted, but stays about `1.9e-6/8.6e-7/7.3e-7`. These are finite-precision closeness statements, not exact affine identities for the original fields.

**Topology:** float64 determinants on the original fields reproduce zero failing cells, with minimum corner determinants `.948312/1.002638/.911870`; this alone is not their global-boundary certificate. For the safe outputs, every stored residual vertex equals `(i/256,j/256)` exactly, so all 262,144 corner determinants equal `1/65536 > 0` exactly and the ordered boundary is identity. Exact rational determinants of the stored affine matrices are respectively `8222057118378987/9007199254740992`, `281163053409409/281474976710656`, and `296154008153803767/288230376151711744`, all positive. Thus the factored outputs are globally homeomorphic onto their affine parallelograms, not necessarily onto the moving image square.

**Cost and conclusion:** `runtime.json` gives total `50.8539/16.3534/16.8935` seconds and library initial-stage `18.4793/16.1745/16.6284` seconds. The HistoReg total-minus-stage difference is not independently profiled and must not be attributed specifically to matching. The current experiment text keeps those distinct and explicitly charges the safe affine representation for DHR initialization; it does not mislabel this as 19-ms independent neural inference. No coordinate, metric or saved-topology discrepancy was found. This supports a useful image-only global initializer on these inspected examples, not broad generalization or a causal isolation of every preprocessing/optimization difference versus zero-start O.

## GPU DHR initial-only comparison

I compared **all three** CPU/GPU config pairs recursively: the only differences are the top-level/stage device settings, the initial-stage CUDA flag, and case/log-path metadata. The saved postprocessing parameters are identical for each pair, including kidney's `787/768` initial resampling factor. The config/runtime copies beside the experiment report match those in the copied D: result folders.

From each actual GPU-produced MHA field I independently formed Q1 coordinate nodes, evaluated all four corner determinants in float64, selected inverse candidate cells by their bounding boxes, and solved their bilinear equations with SciPy. No production evaluator or its topology booleans were used for these calculations.

| Pair | Cells / nonpositive cells | Minimum raw field-corner determinant | Unique / matched | Independent mean TRE, fixed-JPEG px |
|---|---:|---:|---:|---:|
| HistoReg | 738,540 / 0 | .9474552991 | 77/77 | 46.3341412337 |
| Lesions | 653,260 / 0 | 1.0043570702 | 78/78 | 7.6277104672 |
| Kidney | 869,778 / 0 | .9060802661 | 69/69 | 10.4615716016 |

These reproduce the reported means and join policy; maximum disagreement with saved predicted coordinates was `1.82e-12` pixels and maximum inverse reconstruction residual `2.28e-13` field pixels. They are full saved-field local-sign and finite-label numerical checks, **not** a global boundary audit or an all-input DHR homeomorphism theorem. CPU/GPU mean proximity does not imply bitwise or pointwise equivalence.

Two convention attacks on the actual GPU kidney field reinforce the direction/scale distinction. ID 1 correctly predicts `(51.02504571,313.00159285)` with TRE `12.62585742`; wrongly setting `r=1` predicts `(51.94487322,313.56824213)`, while wrongly adding the backward displacement at the moving point predicts `(111.41742679,328.87103691)` with TRE `52.33646267`. The wrong-ratio fixture happens to lower this one point's error, demonstrating why label improvement cannot validate a coordinate convention.

Runtime records support GPU total/initial-stage times `39.9498/3.2977`, `4.1073/3.4871`, and `4.6376/3.9714` seconds. The comparison correctly distinguishes cold pipeline time from the approximately 19-ms frozen network and discloses different machines/OS/dependency builds and absent peak-memory/warm-throughput measurements. **One attribution needs care:** HistoReg's measured total-minus-initial remainder is `36.6521` seconds, but it was not subdivided; a statement that large-JPEG/full-field processing or output specifically *dominates* is an unverified explanation, not a measured result. Report it as unprofiled non-initial overhead unless a separate profile establishes the cause. No GPU job was rerun, and no production file was modified.

## Matcher-cache pilot review

Compared `digital_dhr_cached_matcher.py` directly with the installed DHR 1.0.1 `superpoint_superglue.py` and `multi_feature.py`. The adapter preserves defaults, the hard-coded outdoor SuperGlue configuration, tensor conversion, match filtering, transform direction/helper and return tuple; the change is eval-mode model/weight reuse. The configured two rotations and six sizes imply 12 calls. Two existing mocked tests passed, and separate mocked checks reproduced both rigid and affine return values. Changing each of the five explicit model hyperparameters or either weight path independently caused a rebuild. Rotation/resolution, display flags and transform type need not key the network: they affect inputs or postprocessing, which are still recomputed.

Readback of all three original/cached MHA pairs found **byte-identical float32 displacement arrays**, equal shapes, equal ITK size/spacing/origin/direction, and identical postprocessing parameters. Both lesion repeat-0 and repeat-1 arrays also matched the original exactly. Cached configs differ only in case/log names. Therefore the previously independently checked coordinates, local topology observations and landmark metrics transfer exactly to these outputs; there is no overlooked numerical change in the inspected fields.

**Cache scope is important.** The key contains two path strings, five model hyperparameters and one device string, not a complete identity of mutable process state. Two deliberate mock attacks confirmed that replacing weight contents at an unchanged path leaves a stale model, and changing the current GPU while retaining the device alias `cuda` reuses a model tied to the old GPU. Relative paths similarly presume stable path resolution. Neither event occurred in this fixed-device, unchanged-weight pilot. The experiment text now records these premises and corrects the setting count; use a fresh cache when such state changes rather than treating this adapter as a general cache-invalidation system.

The copied runtime reports reproduce cached cold full/initial seconds **38.6990/1.3287**, **1.8289/1.3146**, **2.1142/1.4659**, each with one construction and 12 matcher calls. The persistent lesion command records **1.9496/1.4343** then **.8985/.7147** seconds, with builds **1 then 0** and 12 calls each. Instrumented inference includes prediction-to-CPU conversion but excludes input transfer and subsequent transform estimation; these host-wall counters are not a complete GPU-kernel time decomposition. The text fairly notes changing preprocessing/warmup, separate non-randomized runs, only two persistent observations, absent peak-memory data, and unresolved HistoReg non-initial overhead. No general throughput, pure causal speedup or fast independent image-to-Q1 inference follows. No GPU workload, package edit or production edit was performed by this review.
