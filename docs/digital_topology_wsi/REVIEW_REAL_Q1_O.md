# Independent review of the saved real-pair Q1 O result

2026-09-29. Read-only review of `digital_q1_real_eval.py`, `digital_compare_appearance.py`, the optimizer, `EXPERIMENTS.md`, the stored Q1 map, DHR field/metadata, original JPEGs and public landmark CSVs. No registration was rerun and no production code or tests were changed. **The reported Q1 TRE deterioration and opposing NCC ranking reproduce independently. No map-direction or coordinate-order error was found for these files.**

## Independent landmark computation

The saved Q1 map sends fixed unit coordinates to moving unit coordinates. Therefore a moving landmark must be inverted through that map before converting to fixed pixels. The implemented conversions `(p+0.5)/image_size` and `u*image_size-0.5` agree with the declared whole-image pixel-center normalization. Image sizes were moving `(7470,9890)` and fixed `(7171,9916)` in x,y order.

I loaded the stored vertices directly, selected candidate cells from independent corner bounding boxes, and solved the two bilinear interpolation equations with SciPy's numerical root solver from each cell center. This did not call the production inverse solver, displacement converter, or map-area helper. Every landmark had exactly one accepted in-cell root. Two explicit checks:

| Landmark | Moving pixel xy | Moving unit xy | Inverse fixed unit xy | Predicted fixed pixel xy | Labeled fixed pixel xy | TRE px |
|---|---|---|---|---|---|---:|
| 1 | (5805.3, 6111.6) | (.777215528782, .618008088979) | (.799055289147, .625857874213) | (5729.525478475, 6205.506680699) | (5502.0, 6654.9) | 503.708446214 |
| 77 | (4530.7002, 1307.2998) | (.606586372155, .132234560162) | (.593941409124, .107497782549) | (4258.653844829, 1065.448011757) | (4761.9, 1587.6) | 725.189210841 |

All 77-point Q1 metrics independently reproduced: mean **503.7609906223452**, median **503.7084462138763**, P95 **915.7123989387086**, maximum **1028.0642408837375** pixels. Maximum inverse reconstruction residual was `1.5700924586837752e-16` in unit coordinates; maximum difference from the saved JSON's predictions was `4.55e-12` pixels. Independent normalized-identity mean/median/P95 were **464.1772114076415 / 496.85602174186454 / 633.2721180556088** pixels. No missing points were removed from these aggregates. These are supplied-JPEG pixel diagnostics, not physical distances or official ANHIR metrics.

## Stored topology and validator semantics

The exact binary32 coordinates in `q1_real_O_257_float32.npz` all have rational denominators dividing **2^34**. I scaled them to that common denominator and evaluated the four oriented triangle determinants using arbitrary-precision Python integers, independently of the filtered validator. **All 262,144 signs were strictly positive; none was zero or negative.** I also independently checked every boundary value against the exact ordered identity rectangle. Combined with the reviewed Q1 theorem, this certifies this stored coordinate map's homeomorphism on its declared control-grid domain. Numerical inverse agreement is supporting evidence, not the topology proof.

The newer `validate_q1_map` checks all cell corners in row chunks, finite coordinates, exact boundary agreement and an ordered axis-aligned rectangular reference boundary. Its own docstring correctly calls this a tensor audit, not an outward-rounded sign certificate. That distinction must remain. The separate saved-binary sign certificate and this integer recomputation substantiate the particular saved sample; they do not prove that every possible layer output survives floating-point rounding. The later single-patch scheduling fix and explicit F2 preconditions also supersede those two implementation findings in the initial `REVIEW.md`.

## Independent common-thumbnail appearance comparison

I re-read both JPEGs with the declared draft/512-square grayscale preprocessing, sampled the saved vertex table and DHR displacement using SciPy bilinear interpolation, and computed centered correlation, MSE and threshold Dice in NumPy float64. No production map-query, warp, or loss function was used. DHR's fixed-pixel coordinates were transformed to its scaled/padded canvas, the displacement was added there, and the resulting moving coordinates were unpadded and converted back to the moving unit square.

| Candidate | Independent 1-NCC | Independent MSE | Independent Dice at .08 | Moving queries outside unit square |
|---|---:|---:|---:|---:|
| Normalized identity | .265013092281 | .002196227136 | .825645188021 | 0 |
| Saved safe Q1 O | **.092967531726** | **.001076741133** | .916013227037 | 0 |
| Saved DHR | .104813088613 | .001255854889 | **.923385834773** | 65 |

These agree with the float32 report to expected numerical precision. The Q1 map's much worse landmark TRE therefore cannot be explained away by a mistaken reported NCC ranking.

One small boundary convention should be explicit: DHR has a 991x747 field, but the last row of thumbnail queries reaches fixed-canvas y=`990.131640625`, beyond the last field-node y=990. Thus **512 fixed query sites use border extension of the displacement**, in addition to the 65 moving-unit out-of-square queries already reported. This is a defined extrapolation convention, not interpolation strictly inside a stored Q1 cell. A sensitivity check restricted all methods to the same 261,568 sites lying inside both the DHR node domain and its mapped moving square; losses remained identity **.265329594699**, Q1 **.093088216507**, DHR **.104947012572**. The ranking survives this boundary exclusion. This exploratory mask is a sensitivity check, not a replacement official metric.

## Interpretation and limits

- The evidence demonstrates disagreement between this global grayscale objective and anatomical landmark accuracy on one development pair. Optimizing and then scoring the same image objective is expected to improve that objective; the independent landmarks expose why this is not registration success.
- The mismatch **does not isolate initialization, fixed-boundary restrictions, or decoder capacity**. A safe output establishes topology, not that topology constraints played no role in its accuracy. Avoid reading “not insufficient topology” as an ablation proving those restrictions irrelevant.
- DHR and Q1 used different initialization/preprocessing/optimization configurations. Their common-thumbnail rescore is a legitimate objective-ranking diagnostic, not a controlled geometry-layer comparison. The approximately 35-fold TRE difference is descriptive of these saved development outputs, not an attributable effect of the safety constraint.
- The optimizer reads no landmarks, and its best-map selection uses image loss. Nevertheless the pair and labels are now observed development material with unknown specimen grouping; neither this review nor code separation creates a held-out test. No G2–G4 claim follows.
- Later 128/64/32 area-average diagnostics were not independently recomputed in this review.

No blocking defect was found in the current saved-result calculations. Keep the boundary-extension convention and causal limitations above explicit when using this negative result to choose the next experiment.

## Addendum: DHR-teacher H-mode distillation

Independently inspected `digital_q1_dhr_distill.py`, `dhr_teacher_257_float32.npz`, `q1_h_dhr_distill_257_float32.npz`, and the H-mode evaluation JSON. The target is explicitly derived from DHR's image-only backward field; no landmark enters preparation or latent fitting. This is teacher-assisted inference/representation evidence, not an independent Q1 image optimizer or learned-network benefit.

For fixed unit query u, I independently computed the DHR canvas coordinate `c=u*(7171,9916)*.1-.5+(15,0)`, sampled its displacement with SciPy bilinear interpolation and border extension, then converted to moving unit coordinates `(c+displacement-(0,1)+.5)/((7470,9890)*.1)`. Across all 257² nodes, the result differed from the stored float32 teacher by at most **1.46125e-7** unit. Examples: u=(0,0) maps approximately to `(-.000941781474,.030979227851)`; u=(.5,.5) to `(.501279395518,.476706296730)`; u=(1,1) to `(1.002781212537,.913561510292)`. Endpoint control queries use the explicitly defined border extension where outside the DHR node domain.

Independent arbitrary-precision integer determinants again found **262,144 positive / 0 nonpositive** corners for each of the saved teacher and H maps. Common exact coordinate denominators were respectively 2^42 and 2^31. The teacher has **256 vertices outside the unit square** and maximum coordinate-wise identity-boundary mismatch **.10365675389766693**; positive local corners do **not** make it a certified fixed-canvas homeomorphism. H has exact identity boundary and no out-of-unit vertices, so the previous global theorem applies to its stored map. Its independently recomputed interior vertex-vector RMSE against the stored teacher is **.008187497235** unit.

The separate cell-wise numerical inverse routine used above produced exactly one root for every one of the 77 landmarks for both maps:

| Saved field/map | Mean TRE px | Median px | P95 px | Maximum px |
|---|---:|---:|---:|---:|
| Original DHR field, earlier independent field review | 14.2325 | 11.7720 | 35.2819 | See prior review |
| Raw sampled 257² DHR teacher, independently recomputed here | **14.4974227845** | 12.0431438079 | 35.4317444229 | 63.1973379417 |
| Safe H fit, independently recomputed here | **32.5511753523** | 18.0496800116 | 85.4824437851 | 256.1729695972 |

For H, maximum inverse residual was `1.57009e-16` unit and maximum prediction disagreement with the saved JSON was `3.64e-12` pixel. Landmark 1 predicts `(5503.707477861,6651.806588052)` with TRE **3.533366402** pixels; landmark 77 predicts `(4729.152644057,1540.365516853)` with TRE **57.475957752** pixels. Direction and pixel-center conventions are consistent with the O-mode evaluation.

Thus sampling DHR onto 257² nodes changes mean TRE only **14.2325 → 14.4974**. Most of the remaining increase to **32.5512** occurs during safe latent fitting. This does not yet isolate identity-boundary restrictions, finite optimization, or decoder expressivity. Conversely, the marked gain over O-mode's 503.761 pixels demonstrates that this safe representation can express a much more useful map when supplied this image-derived teacher.

The error locations do not support a simple claim that the largest landmark errors lie at the frame. The two largest H errors are IDs 59 and 63, **256.17/208.11 pixels**, whose true fixed unit coordinates are respectively **.2521/.2579** from the nearest square edge; the median edge distance among all 77 labels is .1829. ID 60 has 156.66 pixels at edge distance .1163. Dense vertex-to-teacher errors tell a different story: vector RMSE is **.01747** for interior nodes with edge distance below .05, **.004682** for [.05,.1), and **.003623** for [.1,.5). The worst interior fitting error is .08537 unit at row255,column1, one grid step from the edge. These are descriptive diagnostics, not proof that boundary effects caused a particular landmark error.

No directional or metric defect was found. Retain DHR's computation as part of H's inference cost, retain teacher dependence in the method label, and retain the raw sampled teacher row to distinguish resampling from safe-fitting error. No registration or training was rerun for this addendum.

## Addendum: affine-postcomposed H mode

Independently reviewed the updated affine factorization/preparation and `load_effective_vertices` paths, the stored `q1_h_affine_dhr_257_float32.npz` arrays, the factored teacher, and the new evaluation JSON. **The reported mean TRE 14.55877 pixels is reproduced, and the stored factored representation has a sound exact-interpretation homeomorphism certificate.** This is a changed boundary contract, not another identity-boundary map onto the moving square.

The file contains float32 residual vertices U, its float32 identity reference, float32 matrix M and offset b. Independently converting the stored M coefficients to exact binary rationals gives

`det M = 2037155625597927 / 2251799813685248 = approximately .9046788321133934 > 0`.

The numerical condition number of this particular M is approximately 1.084. Repeating the teacher's affine least-squares fit independently agrees with stored M/b to float32 rounding (maximum matrix difference `2.77e-8`, offset difference `1.04e-9`). This affine factor uses teacher coordinates, not landmarks.

Using a common denominator **2^36** and arbitrary-precision integer determinants, I independently found **262,144 strictly positive residual corners and no nonpositive corners**, and verified every residual boundary coordinate equals the ordered identity rectangle. Thus U maps the unit square homeomorphically onto itself. On each source cell, `A(U)=M U+b` remains bilinear; its corner determinants are exactly `det(M)` times U's corner determinants. Since A is an invertible global affine map, A composed with U is a homeomorphism onto `A([0,1]^2)`.

The target parallelogram's cyclic vertices are approximately

`(-.03710802272,.03136340901)`, `(.89929179475,-.03847354278)`, `(1.04330537841,.91691041365)`, `(.10690556094,.98674736544)`.

It is not the moving image's unit square. A float64 materialization has **493 control vertices outside the moving square**; this is consistent with the parallelogram contract, not a fold count. Image intensities outside the original moving domain require a declared extension/crop policy.

For an independent inverse computation I first applied the explicit 2x2 inverse of the stored affine to each normalized moving landmark, then solved **U alone** with the separate SciPy cell-wise inverse routine. This avoids the evaluator's rounded composite-vertex materialization. All **77/77** landmarks had one accepted root; reconstructed composite residuals were at most `3.51e-16` unit, and predictions differed from the evaluator by at most `2.73e-12` pixel.

| Independently recomputed affine-H statistic | Pixels |
|---|---:|
| Mean TRE | **14.5587716374242** |
| Median TRE | 12.0659581307384 |
| P95 TRE | 35.2842972199625 |
| Maximum TRE | 63.0697741900828 |

For two explicit points, landmark 1 predicts `(5496.310983349,6636.062262867)` with TRE 19.67804 pixels, and landmark 77 predicts `(4761.975686511,1572.083411886)` with TRE 15.51677 pixels. Independent interior residual-to-teacher vector RMSE is `.0009322390624` unit; after applying the stored affine, interior composite-to-raw-teacher RMSE is `.0008789544510` unit. These findings place this result close to the sampled teacher's 14.49742-pixel mean, under the changed boundary contract.

**Certificate semantics and one concrete reporting defect.** The evaluator correctly checks the residual certificate plus the exact sign of the stored affine determinant, and explicitly states that its float64 materialization is only for numerical inversion. This certifies the factored mathematical representation, not arbitrary later rounded/cast composite arrays. A fresh materialized export requires its own signs and boundary proof/check; the current axis-aligned-boundary validator cannot simply be applied to a general parallelogram reference. In contrast, at review time the distillation fit CLI's `saved_binary_valid` still forwards `certify_q1_binary_map`'s **residual-only** result even when affine metadata exists. A singular affine attached to a valid residual would leave that boolean true while making the represented composite non-injective. The current stored M passes the independent exact check, so the present result is unaffected; rename this field to residual validity or include affine validation before treating it as a composite certificate. Also reject unpaired affine metadata rather than silently dropping it during fit-file copying.

The affine-preparation threshold `det>1e-8` alone is not a general conditioning bound, although the saved matrix here is well conditioned. No claim about all possible affine inputs follows from this sample. No direction, transpose, or metric error was found for the actual artifact; no production code was edited by this review.
