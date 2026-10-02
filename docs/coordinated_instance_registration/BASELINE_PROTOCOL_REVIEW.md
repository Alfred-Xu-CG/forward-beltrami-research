# Baseline and data protocol review — extension audit, 2026-10-02

Current reading guide: this document preserves the INITIAL restart findings
followed by their corrections and measurements. It is not a queue of still-open
tasks. Released native STANDARD DHR has now run on all25 directions with its own
initializer AND separately with the same saved affine. Initializer provenance,
canvas/native coordinates, original labels and exported fields have been
independently checked. The current retained method uses shared-affine MIND-like
features and SG+MA fusion, not every historical feature-frame choice described
below. See [RESTART_SYNTHESIS](RESTART_SYNTHESIS.md) for the current comparison
table and [REPORT sections 1--5](REPORT.md#1-what-is-implemented-and-what-is-not)
for the actual retained mathematical formulation. The original SG1-era F2
comparison below is distinct from the final current-fusion F2 control.

The existing A-versus-F2 comparison is a useful matched development ablation.
The archived column called “Native DHR” is **DHR shared-affine reduced-512**:
it does not reproduce the released full fast preset, the standard preset, or
the published RegWSI evaluation. Correcting that baseline is necessary before
claiming application competitiveness. Independent numerical agreement with the
existing scorer establishes implementation consistency, not publication equivalence.

## Comparisons that answer different questions

| Comparison | Held common | Important difference | Legitimate inference |
|---|---|---|---|
| A versus matched F2 | Same images/canvases, same saved positive affine, raw MIND and frozen image matches, P1 output, regularizers and label set | Decoder/update class, schedule, actual runtime; F2 uses its declared .95 reserve | Geometry/optimizer comparison at the disclosed gradient budget; not automatically equal-time |
| A/F2 versus archived DHR shared-affine reduced-512 | Same 512 inputs, exact initializer and evaluation labels | DHR native CLAHE/NCC/diffusion, affine intensity prewarp, unconstrained field, 5×30 updates | Application diagnostic under a common initialization; cannot isolate geometry |
| A/F2 versus released DHR fast/standard on native images | Same original cases and all available paired labels, evaluation units | Native initialization, native higher-resolution evidence, native regularization, full schedule, resource cost | Complete application comparison with cost/resolution disclosed; not a matched-geometry ablation |
| Our three MIIT directions versus publication tables | Only dataset family/context | Cohort, number of subjects, units, aggregation, annotations, release protocol | No numerical ranking against published table values |

The lung 20 ordered directions contain ONE previously viewed lung specimen.
MIIT 2→3, 7→8 and 10→11 contain ONE previously used prostate specimen. They are
development observations; neither 20 directions nor 328 available paired labels
are independent subjects. Retain initial failed experiments and all amended
initializer attempts. No independent confirmation or SOTA is established.

## Highest-risk findings and fixes

1. **The baseline was substantially reduced.**
   `tools/coordinated_dhr_common.py:36` replaces the native initial stage;
   lines 65–66 set 512, five levels and 30 updates per level. Its inherited
   preset name is consequently misleading if used as a reproduction label.
   The locally installed DHR 1.0.1 fast preset uses 2048/7×100; standard uses
   4096/8 levels with seven 100-update stages and one 200-update stage.
   Native scale/angle initialization is also retained by those presets.
   Minimal decisive correction: run both released configurations with native
   images and report complete costs; preserve the old diagnostic separately.
   [Released parameter source](https://github.com/MWod/DeeperHistReg/blob/main/deeperhistreg/dhr_pipeline/registration_params.py).

2. **The .690 s number excludes shared input preparation.**
   `tools/coordinated_miit_transfer.py:185` times the DHR wrapper only; canvas
   construction and shared initialization are recorded separately. Recomputing
   the archived manifest gives mean preparation 1.374686 s and raw correspondence
   extraction .273757 s. Allocating those actually measured dependencies gives
   A 5.866927 s, F2 14.860636 s and reduced DHR 2.064744 s, rather than optimizer
   calls 4.218484 s, 13.212193 s and .690058 s. These are arithmetic reconstructions
   of single serial calls, not new repeated end-to-end timings. Full DHR timings
   must include its own initial registration, not the shared initializer.

3. **Our support mask is not the released tissue mask.**
   `tools/coordinated_real_case.py:100,246` uses original inverted grayscale >.04;
   no released MIIT mask is loaded. A direct CPU check of original masks, resized
   nearest-neighbor using the saved canvas layout, finds no released tissue
   pixel excluded but 23.7168%, 23.1925%, 24.7128% of loss support outside the
   released fixed tissue for the three pairs. Thus background contributes
   materially. This is a concrete optimization hypothesis, not proof of a
   RegWSI protocol violation: released DHR does not require these masks.
   Smallest next test: change only the common fixed support to released tissue
   for A and F2, keeping all evaluation labels and the denominator fixed.
   Do not introduce warped-overlap filtering or choose labels by prediction.

4. **Changing raw MIND into “native DHR objective” requires more than CLAHE.**
   A/F2 use PIL luminance inversion, transported MIND, fixed support, original
   moving-feature sampling, P1 ARAP and a corner-shape term; see
   `tools/coordinated_lung_all20.py:35` and `tools/coordinated_real_case.py:145`.
   DHR uses its own normalization/grayscale/inversion/CLAHE, NCC and relative
   diffusion. It prewarps intensity bicubically by the initializer, optimizes
   a residual field, and composes fields. A shared image input does not make
   those objectives or interpolation paths equal. The paper itself presents
   a combined initialization and intensity-registration method and resolution
   ablations. [RegWSI methods](https://arxiv.org/html/2404.13108v2).

5. **Public benchmark units and aggregation do not match these canvas scores.**
   Current scores are Euclidean moving-domain native pixels plus explicitly
   declared 512-canvas pixels, equal-pair mean and mean pair-p90. There is no
   verified MIIT physical spacing or official CSV origin in the local protocol.
   ANHIR publication rTRE divides by image diagonal. ACROBAT 2023 averages
   per-landmark distances across annotators in micrometers, computes pair p90,
   then averages pair p90; absent submitted landmarks receive the official
   fallback rather than disappearing. Our missing MIIT labels mean absent
   annotations, a distinct issue. Do not call our canvas mean-p90 an official
   ACROBAT score. [Official evaluation](https://acrobat.grand-challenge.org/evaluation-ranking-and-prizes/).

## Coordinate, image and label audit

The present common-canvas conversion is internally coherent. PIL reads width,
height; points are x,y, while arrays index y,x. Aspect ratio is preserved before
padding, modulo explicit integer resize rounding. `digital_birl_pair_canvas.py:31`
uses white RGB padding and bilinear resize; `digital_birl_landmark_score.py:22`
uses `((p+.5)*scale+pad)/512` with exact axis-specific rounded resize scales.
The inverse includes the matching -.5. EXIF orientation other than 1 is rejected.
Unknown MIIT spacing is explicitly replaced by no physical-units claim.

The initializer rotates only the moving matching view. Its keypoints are
unrotated into the original image before fitting (`coordinated_rotation_initializer.py:25,94`).
Selection uses image matches and spatial support; the amended four-quarter-turn
initializer is not RegWSI's native multi-scale initialization. It was introduced
after direct image matching failed on two pairs; the amendment must stay visible.

Despite case labels “moving_to_fixed”, saved sampling maps go **fixed→moving**.
This is correct for backward image sampling and current landmark scoring.
P1 maps are evaluated under their AC diagonal, with the affine applied once.
`digital_compare_appearance.py:40` handles the saved DHR field as displacement
in field pixels, unequal padding and source/target loading ratios. Unit queries
must use each original image's own width and height, never force native TIFFs
into a synthetic 512 square during scoring.

Installed DHR `dhr_utils/utils.py:60–64` calls interpolation with explicit scale
factor and `recompute_scale_factor=False`. Therefore its saved nominal
`initial_resample_ratio` is the correct coordinate conversion, even if rounded
output dimensions yield a slightly different effective size ratio. The canvas
PIL resize above instead requires its saved dimension-derived scales. These
two conventions must not be interchanged. An independent rectangular/padded
source-level check is assigned to the separate checker context.

The existing MIIT scorer (`coordinated_miit_score.py:31,138–158`) preserves all
124 declared IDs and excludes only released (+inf,+inf) absent annotations.
It applies the same eligible set to every method: 123, 107 and 98 paired labels.
Prediction failures remain failures and incomplete cases cannot receive an
all-three summary. `successful_only` is diagnostic, never the cohort score.
The [version-v4 data description](https://zenodo.org/records/14931377) confirms
nine serial sections, matching labels and supplied semi-manual tissue masks;
it does not itself settle subpixel CSV origin or physical pixel spacing.

## Executable correction and remaining work

`tools/coordinated_dhr_released_baseline.py` now calls the actual installed
`default_initial_nonrigid_fast()` or `default_initial_nonrigid()` and the
unmodified `DeeperHistReg_FullResolution` class. It does not accept our affine,
feature matches, masks or annotations. The only adaptations are device/thread
selection, PIL loading at ratio 1, output/log paths, disabled intermediate/final
image raster exports, and retained displacement export. Ratio 1 preserves
the original ~3k MIIT TIFF evidence; the preset still controls initial and
nonrigid resolution. The default .2 loading ratio would prematurely reduce
these already reduced images to ~600 pixels.

The native algorithm settings are saved beside the exact released preset.
Every pair is attempted; failures remain in `predictions.json`. Original sizes,
padding/postprocessing path, field path, actual tensor shapes, version/source,
timing and allocated GPU memory are recorded. Subsequent runner versions also
record the small native initial affine, its frame, determinant and finiteness.
The first fast process had already loaded the earlier runner without that
optional affine metadata; do not rerun or overwrite it just for metadata.

Local installed package: DeeperHistReg **1.0.1**, located in
`D:/QC_optimization_data/digital_topology_wsi/dhr_clean_venv/Lib/site-packages/deeperhistreg`.
Local source mirror Git revision: `42e7c9ddedb5932fbcbdf598fbc9b3a47baa47b6`.
These are distinct locations; the mirror commit is not asserted to identify
the installed wheel. The runner records the execution host's package version
and actual source paths.

Focused author checks: four tests pass, verifying preserved algorithm fields,
both preset routes, direct native image arguments and three-case failure
retention. Independent code review is required before treating the new
runner/scorer as independently verified. Coordinator owns GPU execution and
scoring; this reviewer has not run a GPU experiment or inferred its outcome.

Next decisions are limited to: complete native baseline/scorer verification;
run a matched released-mask support ablation if warranted; retain development
scope until a genuinely independent specimen/protocol exists.

## Prepared fixed-support ablation, not yet an efficacy result

The coordinator authorized a separate support-only experiment after the native
baseline and simultaneous-image-scale diagnoses. `tools/coordinated_tissue_support.py`
prepares the original fixed-section 3, 8 and 11 masks using each saved image
layout: native binary support, nearest-neighbor resize, zero padding. Its three
reusable arrays and ordinary provenance record are at
`D:/QC_optimization_data/digital_topology_wsi/miit_released_fixed_tissue_support`.
No manual correspondence coordinates or candidate deformations are read.

`coordinated_real_case.py --fixed-mask <prepared.npz>` replaces only the fixed
support; coarse levels continue using area-weighted fractions. The default
grayscale threshold, feature construction, point evidence, priors, topology
conditions and complete evaluation label set remain unchanged.
`coordinated_miit_multiscale_pilot.py --fixed-supports <support.json>` applies
the same per-pair support to both A/F2 and requires `--image-objective continuation`.
It explicitly reports supplied tissue annotations and does not rerun DHR.
This is images-plus-semi-manual-ROI evidence, not a pure-image-only claim or a
new optimizer mechanism.

Nine focused checks pass: native axes/padding and categorical resize, fractional
area weights, malformed input rejection, unchanged default image tensors/mask,
masked value/map-gradient finite difference, unchanged priors, and exclusion of
simultaneous MS plus support changes in this pilot. The author did not run a
real GPU support experiment; the coordinator owns the decision and execution.

## Native standard initializer versus final field

The completed native standard run permits a posthoc decomposition without
rerunning registration or transferring its initializer to our optimizer.
`tools/coordinated_dhr_initial_score.py` evaluates only the stored initial
theta through native loading/padding/preprocessing pixel-center coordinates.
It reads no dense field. The same 123/107/98 paired labels yield equal-pair
mean/p90 of **4.887064/8.282913** canvas pixels at native initialization, versus
the common initializer's **4.904193/8.897884** and native final field's
**3.672192/6.432444**.

The native initializer changes aggregate mean by only .017129 pixels; its
nonrigid stage then lowers mean by 1.214872 pixels. Thus the principal measured
mean benefit of the full native run occurs during nonrigid registration. This
is a sequential decomposition, not the counterfactual result of running that
stage from our common initializer. Pair 7→8 retains an adverse tail: its native
maximum rises from 50.448319 at initialization to 53.497967 after nonrigid.

Initial scores are saved in
`outputs/coordinated_instance_registration/miit_dhr_released_standard_t19/initial_landmark_scores.json`.
The analytic scorer evaluates saved float32 theta in float64; it does not claim
bitwise equality to rasterizing an initial displacement in float32. Three
focused tests pass, including hand-calculated rectangular, unequal-padding,
nonunit-loading coordinates and an independent Torch affine-grid/displacement
path. Separate-context formula review remains the coordinator's responsibility.

## Shared landmark-tail diagnosis: resolution is not the leading explanation

This is a read-only diagnosis of the completed shared-affine MIND control and
native standard DHR, not a new training or model-selection experiment. Inputs
are `miit_multiscale_control_t19/landmark_scores.json` and
`miit_dhr_released_standard_t19/landmark_scores_complete.json` below the common
outputs directory. Their equal-pair A/F2/DHR mean TREs are
3.548752/3.554327/3.672192 canvas pixels.

For pairs 2→3, 7→8 and 10→11, the A-versus-native-DHR per-label Spearman
correlations are .7864/.8977/.7642. Worst-decile label overlap is 10/13, 9/11
and 8/10. All 13 A landmarks exceeding 10 pixels also exceed 10 pixels under
native DHR; DHR has one additional such label. A's worst decile contributes
29.9%/42.3%/44.6% of its pair's summed errors. This is a shared adverse tail,
not a uniform subpixel accuracy deficit unique to the 512-pixel optimizer.

The clearest example is pair 7→8, Pt121: A/native-DHR errors are
53.0265/53.4980 pixels; Pt122 is 30.5261/30.7747. The existing historical
`miit_7_to_8_oracle_tail_context_t17.png` shows the Pt121 discrepancy at a
gland/boundary scale already visible in the 512 overview. That plot's predictor
overlay predates the native-standard run and must not be relabeled as current
DHR. The current errors above come only from the current score files.

Native standard uses the available ~3k source evidence yet preserves these
large errors. Together with the visible scale, this weakens the claim that
512-pixel raster resolution alone causes the principal tail. It does not prove
that a fixed-257-P1 1024/2048 image experiment cannot improve finer errors:
different descriptors/optimizers can exploit detail differently. Nor does it
prove annotation error, absent tissue or an impossible correspondence. These
remain ambiguous serial-section cases. The next higher-information intervention
is optimizer/correspondence diagnosis, given the coordinator's independently
observed persistent update-bound hits; a resolution-only experiment is secondary.

## Historical NCC ablation versus actual released DHR

Installed DeeperHistReg `cost_functions.ncc_local` and released registration
presets use squared local NCC with a 7×7 zero-padded window and denominator
epsilon 1e-5. Our `RealProblem.image_terms` uses the same local sums and
algebraically the same centered cross/variance expressions. Its `1-NCC` versus
DHR's `-NCC` differs by an objective constant if aggregation is identical;
our variance clamp can additionally alter roundoff-edge cases. The historical
ablation therefore did **not** mistakenly compare global NCC with local NCC,
or use a different nominal NCC window.

However, it was not a DHR-faithful objective pipeline. Our NCC was averaged on
the raw fixed-image threshold support, while the released preset supplies no
mask and averages the whole frame. Our pyramid independently area-resizes each
level; DHR recursively applies Gaussian smoothing and bilinear downsampling.
The historical NCC run directly samples the original moving evidence using the
composed affine/residual query; DHR first bicubically affine-prewarps and then
bilinearly residual-warps the raster. Boundary conditions, priors, shape and
machine-point terms, iteration schedule and final resolution also differ.
In particular the historical saved NCC configuration omits `strain_model`
and therefore uses the then/default displacement-gradient prior, not the later
ARAP control. DHR uses `diffusion_relative`.

The saved native-preprocessing NCC experiment used image512, grid257,
32/64/128/256/512 continuation, 30 steps per coordinate/stage, strain weight3,
shape weight1e-4 and machine-point weight.1. Its three development-case means
(.794/3.936/2.945) versus corresponding MIND (.808/3.713/2.468) establish no
consistent benefit under **that** matched ablation. They are not current MIIT
results and do not reject DHR-style evidence as a class. `PROGRESS.md` already
records the principal pipeline differences; the source audit confirms that
qualification rather than finding a new NCC-window bug. This alone does not
justify another parameter sweep.

## Existing22 native STANDARD expansion: setup and pre-run review

Question: does the corrected released baseline change the comparison beyond the
three correlated MIIT directions? Exact claim under test: full native STANDARD,
with its own image-only initialization and unmodified algorithm preset, can be
evaluated in the SAME existing512 moving-canvas coordinates without interpreting
its native field as a512 field. Assumptions: original JPEGs, saved layouts and
existing annotation conventions are correct. Falsifier: unequal native sizes,
padding or resampling changes the coordinate fixture, or any prediction failure
vanishes from the denominator. Smallest decisive test: rectangular padded native
field with nonunit initial resampling, followed by one synthetic complete22
cohort with the actual dataset readers. Prior implementation: the independently
checked MIIT released runner and `digital_compare_appearance` native evaluator.

Approved scope is exactly20 ordered lung-lesion3 stain directions, Histo fixed
CD4→moving CD68, and kidney fixed HE→moving PanCytokeratin. The old BIRL
`lesions_` crop is a different image/annotation frame and is NOT a23rd job.
All are previously viewed development specimens; twenty directions do not add
twenty patients. Lung keeps80 labels/direction, Histo77, kidney69 with fixed-only
IDs70/71 explicitly disclosed. Lung's50pc→5pc conversion remains
`(coordinate+.5)/10-.5`, with its existing subpixel uncertainty.

`coordinated_dhr_released_baseline.py --input-rows` now accepts only name,
fixed and moving paths plus a scope string. MIIT `--source-data` defaults are
unchanged. Relative image paths resolve next to the manifest. Every case is
attempted serially, with native initialization, loading ratio1, unchanged preset,
per-case config, sizes/modes, timing, and GPU free/peak memory metadata. There is
no reduced-resolution fallback, inherited common affine or hidden timeout.

`coordinated_dhr_existing_score.py` requires all22 attempts to be terminal before
opening labels. It uses native image dimensions with the existing saved-field
evaluator, then the original native→512 layout conversion. It reports field
scalar ranges and all saved-field bilinear-cell corner signs, not a global
homeomorphism certificate. Failed cases remain failures; no complete lung20 or
equal-specimen aggregate is emitted when its required cases fail. It never
averages native-pixel errors across specimens.

The image-only preparation is at
`D:/QC_optimization_data/digital_topology_wsi/dhr_existing22_inputs`:
`local_inputs.json`, `remote_inputs.json`, and `transfer_list.json`. The last is
an ordinary copy list of9 unique JPEGs totalling93,025,647 bytes, not a new
integrity framework. Copy the remote manifest and its listed images under
`images/` beside it; manual CSVs are not in the transfer list. At the preparation
milestone this worker had not copied images remotely or launched registration.

Execution after coordinator approval:

```text
python -m tools.coordinated_dhr_released_baseline --input-rows <remote_inputs.json> --output <new-output> --preset standard --device cuda --threads 2
python -m tools.coordinated_dhr_existing_score --predictions <completed-local-output> --data-root D:/QC_optimization_data/digital_topology_wsi --output <new-score.json>
```

The native lung/kidney rasters are below4096 and the released preset does not
upsample them. Histo's native ~74M-pixel RGB images are copied to GPU before
preprocessing; the two padded float32 inputs alone need approximately1.8GB,
with substantial additional workspace. Check host RAM and free VRAM before
launch and keep GPU jobs serial. A~20min reservation for22 cases is a planning
estimate, not a measurement; native MIIT STANDARD took31–33s/pair but is not
the same resolution. Author checks:31 focused tests pass, including existing
MIIT regressions. The bounded independent review in
`BASELINE_PROTOCOL_INDEPENDENT.md` passes with no production fix required.
Its separate NumPy border-bilinear oracle uses nonuniform fields, unequal
rectangular dimensions, asymmetric padding, nonintegral initial-resampling
ratios and unequal loading ratios; maximum differences are7.99e-6 native and
1.44e-5 canvas pixels. The probe is saved at
`outputs/coordinated_instance_registration/check_sources/independent_native22_probe_20261002.py`.
This validates coordinate implementation, not anatomical accuracy. The later
authorized execution and observed outcomes are recorded below.

## Executed22: full native comparator and shared-frame transfer control

After independent review, native STANDARD ran serially on AI GPU5; all22 cases
completed in683.392s. GPU5 was idle and host available RAM was~976GB before
launch; before Histo, available RAM was~975GB and GPU5 had~48.2GB free. Histo
completed at5437×4096 in77.684s, kidney at787×1164 in17.517s. Maximum native
allocated GPU memory was7,709,164,544bytes. Native MHA exports total279,916,595
bytes and remain local data on D, not committed code artifacts.

The separately approved transfer control ran the SAME22 analytic300 recipes
on idle AI GPU1, changing ONLY `mind_frame="shared_affine"` and output paths.
Original images, affines and frozen raw matches were reused; no new matching,
seed, preconditioner, support or parameter choice. Independent real-config
comparison confirmed those are the only semantic configuration differences.
All22 finished300 gradients with zero failed trials in99.344s. All actual saved
P1-ac residual/boundary/positive-affine certificates pass; minimum actual corner
ratio across the22 maps is.0069463903, above the.001 floor. Maximum allocated
GPU memory was213,791,744bytes. These single-call timings are not an ABBA speed
claim: native DHR estimates its initializer, while A reuses the stored affine
and machine correspondences.

Manual scoring began only after BOTH22-case prediction batches were terminal.
Every method retains the same1746 paired queries:20×80 lung,77 Histo and69
kidney; kidney fixed-only70/71 remain disclosed. Values below are mean / p90
in the SAME512 moving-canvas pixels; for lung these are equal-direction mean
of means / mean of within-direction p90s, not a pooled quantile.

| Existing cohort | Original-frame A | Declared F2 | Reduced512/common-affine DHR | Shared-frame A | Native STANDARD DHR |
|---|---:|---:|---:|---:|---:|
| Lung20 |4.520228 /9.555206|4.551344 /9.657840|5.133593 /11.552831|4.568541 /9.593679|6.046615 /13.914145|
| Histo77 |.793449 /1.544649|.787888 /1.532709|.942615 /2.288375|.851903 /1.586005|.712328 /1.618927|
| Kidney69 |2.351586 /4.507429|2.394303 /5.096757|3.431687 /7.110328|2.364866 /4.756907|1.908163 /3.178382|

Lung's primary F2 source is `lung_all20_f2reserve_t134/all_id_scores.json`, with
explicit `f2_floor_safety_fraction=.95`, accepted gain1 and all20 complete300.
The old `lung_all20_fixedrecipe_t123` F2 mean4.599464 is a DIFFERENT arm:
HE→Ki67 completed242 steps/2 failed trials; Ki67→HE158/5. It remains historical
evidence and is not substituted for the corrected F2. Original lung A and
reduced DHR are unchanged archived references in the corrected cohort.
Histo/kidney A/F2 come from `development_p1arap3_stage30_t88`; their reduced
DHR scores come from `histo_dhr_common_score.json` and
`rat_kidney_dhr_common_score.json`, respectively.

The descriptor-frame benefit observed on rotated MIIT does NOT transfer as an
observed mean improvement here: all three cohort means are slightly worse than
the original A recipe, and only7/20 lung directions improve. Do not assert a
general accuracy gain from that correction. Conversely native STANDARD is
better on Histo mean and kidney mean/tail, but worse on lung mean/tail. The
three-specimen mean2.595104 for shared A versus2.889035 native DHR is driven by
lung and is not universal method superiority. All22 saved native DHR fields
have nonpositive local bilinear-cell corners; this is a representation-specific
local diagnostic, not a global-boundary certificate or exclusion rule.

### Saved-initializer decomposition, not a rerun

`coordinated_dhr_existing_score.py --initial-only` reuses the independently
checked `map_initial_native`, plus the SAME22 dataset dispatch/IDs/layouts.
It reads saved theta and postprocessing parameters, never a dense field. Every
theta and its explicit normalized preprocessed target-to-source frame are
retained in `native22_dhr_standard_t20/initial_only_scores.json`.

Lung native initial mean/p90 **6.291708/11.390313** becomes
**6.046615/13.914145** after native nonrigid. Our shared initializer is
**6.661222/12.234678**, becoming **4.568541/9.593679** under shared-frame A.
Thus worse native initializer MEAN does not explain the aggregate lung gap;
native nonrigid improves mean only.245092 while its mean directional p90 worsens
2.523832. Native nonrigid worsens7/20 direction means and14/20 direction p90s.
Shared A beats native final mean in16/20 directions.

Histo native initialization2.339111 is close to our2.332173, followed by native
final.712328 versus shared A.851903. Kidney is different: native initialization
4.322930 is substantially better than our6.117413, then native reaches1.908163
and shared A2.364866. Do not attribute kidney's advantage solely to its nonrigid
optimizer. None of these stage decompositions supplies the unrun counterfactual
of exchanging initializers, or isolates resolution, objective and boundary class.

Reproducible per-case comparison, exact source score/configuration paths,
unchanged ID-set checks and the historical F2 version distinction are in
`outputs/coordinated_instance_registration/native22_comparison_t20.json`.
Its one-off read-only generator is
`outputs/coordinated_instance_registration/check_sources/native22_comparison_t20.py`.
New raw scores are `native22_dhr_standard_t20/landmark_scores.json` and
`existing22_shared_affine_a300_t20/landmark_scores.json`. Nine bounded A-control
tests pass;23 focused tests including the initializer extension pass. Independent
integration AND actual-output checks pass in `BASELINE_PROTOCOL_INDEPENDENT.md`.
The checker reaggregated all1746 per-label records in each arm, matched IDs and
recomputed the corrected F2 cohort. A separate float64 NumPy border-bilinear
oracle checked all226 labels in native HE→CC10/Histo/kidney: maximum difference
was3.287e-5 canvas pixels (.0006362 native pixels), consistent with the scorer's
float32 field sampling. Saved-initial-only dataset dispatch, helper/scales and
means also pass. No GPU rerun, label-based selection or new dataset is implied.

## Approved supporting counterfactual: native STANDARD with shared initializer

Research card, 2026-10-02, before implementation. This is a supporting baseline,
not a new research mechanism and not an unmodified released complete pipeline.

**Question.** How does the released STANDARD nonrigid stage behave when it starts
from the exact same previously frozen positive affine used by the safe methods, while
retaining its native images, preprocessing and complete nonrigid configuration?
The existing STANDARD25 uses its own native initializer. The existing shared
affine DHR uses reduced512 evidence and five levels of30 steps. Neither answers
this question. Compare the new run to BOTH of these existing references; do not
replace the unmodified-initializer STANDARD result or select per-case baselines.

**Exact claim and coordinate conversion.** All maps below act on COLUMN vectors.
Let p_i denote native zero-based pixel-center coordinates for fixed f or moving m.
The already saved512 layout defines the affine

    C_i(p_i) = (diag(layout_scale_i)*(p_i + .5*1) + layout_pad_i)/512.

The actual DHR loader/padded/preprocessed frame defines

    N_i(p_i) = 2*diag(1/(r*extent_i))
                  *(rho_i*(p_i + .5*1) + d_i) - 1,

where extent_i=(Wpre_i,Hpre_i), r is the saved initial_resample_ratio,
rho_f/rho_m are target/source loader ratios (both1 in this reproduction), and
d_f=(pad_2[x,left],pad_2[y,top]), d_m=(pad_1[x,left],pad_1[y,top]). The source
code calls bilinear interpolation with scale_factor=1/r and
recompute_scale_factor=False, so use this DECLARED r, not a rounded-size ratio.
The full-resolution loader pads source and target to a common size; validate
that the resulting preprocessed extents agree rather than assuming it silently.

For the stored original unit-canvas affine H(q)=Aq+b, the normalized DHR affine is

    theta_homogeneous = N_m * inverse(C_m) * H * C_f * inverse(N_f).

Compute this composition in float64 from unchanged stored A,b/layouts, then cast
the2x3 theta once to float32, matching the installed release. The composition is
the SAME geometric initializer in real arithmetic; native-frame rounding means
we must not claim bitwise identical coefficients or zero numerical error after
the float32 cast. No affine re-estimation, dense-field teacher, matcher rerun,
annotation access, resizing to our512 canvas, or pixel-content rewrite occurs.

**Assumptions.** Same original source files/native dimensions and EXIF orientation;
correct saved two-axis integer resize/padding layouts and positive stored affine;
installed DHR1.0.1 STANDARD. Only a subclass run_initial_registration override
sets initial_transform, initial_displacement_field and current_displacement_field.
The existing released runner's device/I/O adaptation is reused. Leave all native
preprocessing and nonrigid algorithm fields unchanged: initial_resolution4096,
registration_size4096, eight levels with iterations[100,100,100,100,100,100,100,200],
NCC window7, diffusion_relative and all released learning rates/alphas. The
subclass does not alter the installed package or the archived native-init runner.

**What would falsify the implementation claim.** A wrong source/target order,
half-pixel shift, padded-axis error, use of rounded extent ratios, theta applied
twice, nonrigid/preset difference, or sampled initializer inconsistent with the
declared transformed supplied affine. Wrong initialization does not become acceptable
because final TRE looks good.

**Smallest decisive tests.** Non-square synthetic native frames with unequal
resize scales, odd padding, nontrivial positive affine and r>1; independently
roundtrip original512 query centers through the converted saved theta and back
against Aq+b. Check float64 algebra and actual float32 theta separately. Verify
the generated initial field at its full preprocessed grid against an independent
analytic affine evaluator. Check native field sampling's expected boundary clamp:
the native field does not represent the extra512 letterbox or affine extrapolation
outside its pixel-center domain. Keep this domain distinction explicit; no manual
evaluation landmark is dropped. Compare every STANDARD algorithm parameter to its
released counterpart; only device/save/I/O fields are adapted. A tiny injected
runner checks per-case failure retention, no native matcher call and all25 terminal
before preparing the two existing scorer manifests. Independent verification of
conversion and actual output is required before interpreting accuracy.

**Experiment and costs.** Run all25 existing directions with the same fixed source
ordering and all available labels only after every prediction terminates. Reuse
the original MIIT/native22 field scorers and exact original data/CSV transforms.
Keep native field local-corner diagnostics; never repair or exclude folded cases.
Archive original affine/layout provenance, computed theta, runtime config, final
full MHA, postprocessing metadata and image-only conversion diagnostics. Do not
create an artifact-integrity layer or duplicate full initial fields by default.
Observed existing full STANDARD25 complete-call time is777.7915s, of which
588.5642s is native initialization and121.0433s is nonrigid. Subtracting only the
skipped native initializer suggests roughly189.23s before conversion overhead;
allow5–10minutes serial as an estimate, not a performance guarantee. The prior
maximum allocation is7,709,164,544bytes and all25 final fields total491,480,185bytes.
Record supplied initialization as a shared historical cost, not a free new
algorithm; separately report complete replay and nonrigid costs.

**Hypothesis and interpretation.** A change versus native-init STANDARD measures
initializer sensitivity under otherwise unchanged DHR native processing. The
comparison to the safe method matches the initial geometric affine, not image
resolution, normalization/CLAHE, objective, regularizer, boundary constraints or
iteration budget. It cannot alone isolate a topology-layer effect, prove global
optimality, or turn this modified pipeline into the published complete method.
If the shared initializer hurts DHR, report that result as well; no baseline
selection by labels. Existing released DHR source/configuration is the prior-work
reference, and the validated initial/native-field readers above supply the frame
conventions.

Implementation is `tools/coordinated_dhr_native_shared.py`; the original native
own-initializer runner and installed DHR package are unchanged. Eleven focused
tests (new shared-native tests plus original released-runner tests) pass. A tiny
CPU execution of the actual installed `tc_transform_to_tc_df` through the new
override checked all5,673 nodes of a synthetic61x93 frame: maximum normalized
field error3.2444e-7 and maximum original512-canvas error2.3767e-6pixels. Across
all262,144 accepted512 query centers, analytic float64 conjugacy error was
3.4577e-13pixels and float32-theta casting error1.4638e-5pixels. This deliberately
small native-support fixture also exposes large out-of-lattice border-clamp
error130.551pixels, versus2.37294e-6pixels within support; reporting only the
inside subset would conceal a real domain difference. These are correctness
fixtures, not actual25-case registration or accuracy results. The production
runner reports both domains without dropping queries and stores no second
full initial field. Independent review and the actual native25 run remain
separate work; no GPU experiment was launched by this implementation task.

## Native STANDARD shared-initializer counterfactual: actual all25 result

All25 predictions completed on AI GPU5 before local CPU annotation scoring.
The unchanged native scorers produced25/25 scores for the same2,074 available
landmarks across own-init STANDARD, shared-init STANDARD and retained fusion300.
MIIT uses `miit_three_rotations_t153/*_layout.json`: its original dimensions,
integer resize/padding and exact two-axis scales were checked against all three
new saved accepted512 layouts before scoring. Existing22 uses the unchanged
native data-root readers. No label selected an initializer, model or output.
The saved released STANDARD preset exactly equals both original baseline
presets; all25 saved preprocessing, nonrigid, loading and saving parameter
sections also exactly equal their own-init counterparts. The later33-test
focused native-runner/scorer and refresh-comparison regression collection passes.

Values below are mean-of-direction mean / mean-of-direction p90 errors in the
same512 moving-canvas pixels, not native pixels or independent-patient inference.

| Development specimen | Native own-init STANDARD | Native shared-init STANDARD | Retained fusion300 |
|---|---:|---:|---:|
| MIIT,3 directions |3.672192 /6.432444 |3.712720 /6.260251 |3.534718 /5.858170 |
| Lung,20 directions |6.046615 /13.914145 |6.240578 /14.626331 |4.471007 /9.342538 |
| Histo,1 direction |0.712328 /1.618927 |0.712976 /1.523942 |0.842064 /1.580138 |
| Kidney,1 direction |1.908163 /3.178382 |2.464310 /5.245805 |2.243606 /4.643549 |

Shared initialization worsens mean error in17/25 directions versus native
own-init (8 improve), p90 in16/25 (9 improve), and worst landmark in16/25.
Against fusion300 it worsens mean and p90 in21/25 directions (4 improve) and
worst landmark in17/25. Per-specimen mean regressions versus own-init are
2/3 MIIT,13/20 lung,1/1 Histo and1/1 kidney. Histo is the important adverse
comparison for our retained method: native shared-init is better in mean/p90
and worst error. Own-init native DHR also remains better than fusion300 in
kidney mean/p90. Do not choose the stronger native initialization per case or
replace the original baseline. This result says the common supplied initializer does
not improve this native DHR package globally; it is not evidence that its native
initializer was unfair, or that the safe optimizer is universally superior.

Recorded complete-call sums are777.791515s own-init versus190.383174s shared-init;
the new batch wall time is190.551014s. Native initial-stage time is588.564189s
versus0.071818s, native nonrigid time121.043309s versus125.345216s, and preprocessing
1.967333s versus1.687901s. Shared initialization audits add4.780044s inside its
complete-call scope. Fusion300's archived optimizer calls total114.162324s, but
omit its historical affine/SG/MA preparation. Shared-native also omits the
historical supplied-initializer cost; only own-init includes its own native initializer.
These are measured component scopes, not a cold end-to-end speed comparison.
Peak allocated memory maxima are7,709,164,544bytes own-init,
5,677,745,664bytes shared-init and214,255,616bytes fusion300; do not add peaks or
ignore differing reset/setup scopes.

The actual initializer audit checks61,434,097 preprocessed native nodes and
all6,553,600 accepted512 query centers. Maximum analytic theta64 conjugacy
error is2.8422e-13canvas pixels; theta32 casting error1.7922e-5; actual native-node
field error1.0024e-4; literal sampling within native-node support1.5007e-4.
There are1,560,510 accepted512 queries outside that native pixel-center domain;
their literal border-clamped field discrepancy reaches27.725942pixels. This is
the predicted domain/clamping distinction, not an affine-coordinate conversion
failure. All queries and all available evaluation labels were retained. The
same geometric initializer is established; bitwise equal raster evaluation
through extra512 letterbox pixels is not claimed.

All25 native outputs in BOTH arms contain nonpositive local saved-field cell
corners. Shared-init has2,039,097/245,494,952 checked corners with minimum ratio
-1.084299; own-init has1,984,908 with minimum ratio-0.975030. These checks concern
the native saved bilinear displacement-grid cells, not the safe method's257-grid
P1 class. Outer-boundary injectivity was not checked, and neither native arm has
a global-homeomorphism certificate. No fold repair or case exclusion was used.
Shared initialization still leaves native resolution/preprocessing, NCC,
diffusion regularization, boundary behavior and900-step optimization different
from fusion300; this experiment cannot isolate a topology-layer advantage.

Reproduction: existing `coordinated_dhr_released_score` and
`coordinated_dhr_existing_score` write the new directory's `miit_scores.json`
and `existing_scores.json`. `check_sources/native_shared_comparison_t27.py`
checks the exact per-direction landmark-ID sets and writes
`native_standard_shared25_t27/comparison.json`, including every direction,
tail regression, initializer audit, phase cost and local topology count.
The independent checker re-read all25 final MHA fields and original CSVs:
all2,074 errors agree within5.84e-5canvas pixels (0.0008811native pixels), and
all245,494,952 corner diagnostics agree exactly. Source A/layouts, native
dimensions/bytes, padding/resampling frames, configurations and the comparison
tables also pass. `native_standard_shared25_t27/independent_check.json` records
this check. The online initial-field audit remains a recorded production audit;
the independent final-field re-read does not pretend that an unexported initial
field was separately re-read.

## Additional specimen reconnaissance (metadata only, 2026-10-02)

The local MIIT release is ONE actual sample, not nine independent specimens.
The opening "Note on the test data" in the
[author's example notebook](https://github.com/mwess/miit/blob/master/examples/notebooks/04_analysis_from_paper.ipynb)
explicitly describes test data for one sample. Its next dataset-description
cell lists serial-section IDs `1,2,3,6,7,8,9,10,11`, exactly matching
`D:/QC_optimization_data/miit_v4/extracted/test_data/test_data/source_data/`
and the [current Zenodo release](https://zenodo.org/records/14931377).
Unused sections or new directions can test within-specimen robustness, but do
not supply independent-specimen confirmation. Only notebook Markdown and file
metadata were inspected; no additional target landmark coordinates were read.

The author-owned [Borda landmark repository](https://github.com/Borda/dataset-histology-landmarks)
contains other tissue annotation folders but only the already-used
`dataset/lung-lesion_3/scale-5pc` images. Its linked original CIMA image page
returned HTTP404 during this check. The [CBICA HistoReg repository](https://github.com/CBICA/HistoReg/tree/master/Data)
contains only the already-used CD4/CD68 image/landmark pair. Neither repository
therefore supplies another immediately usable specimen in its current file tree.

[HyReCo's official publication endpoint](https://doi.org/10.21227/pzj5-bs61)
does document independent labelled histology: nine consecutive-section cases
plus additional H&E/PHH3 re-stained cases. TIFF images and CSV landmarks are
paired; CSV rows correspond across a case, and `(x,y,z)` coordinates are in mm
with an upper-left image origin. However, the endpoint explicitly requires
login, the base ZIP is233.36GB, and no account or download was initiated.
The title specifies CC-BY-SA4.0 while its JSON-LD license field specifies
CC-BY4.0; this discrepancy needs resolution before acquisition/reuse. Individual
image dimensions and TIFF physical-coordinate conversion remain unverified.

The [Warpy example record](https://zenodo.org/records/5675686) has accessible
metadata, CC-BY4.0 API licensing, and a1,074,844,732-byte example project ZIP.
Its listing does not establish independent evaluation landmarks; an included
example registration is not independent ground truth. No ZIP was downloaded.
This bounded check found no new specimen with both unrestricted acquisition and
verified independent registration labels ready for the present benchmark; it
does not assert that no such public dataset exists.

## Initializer provenance correction (2026-10-02, exact archive audit)

The shared initializer must be called the **saved image-only positive affine**,
not uniformly "our SG similarity". All25 cases use the same numerical saved
`A,b` across the compared shared-initializer arms, but they do not all use the
same initializer recipe. The raw SG correspondence term is a separate later
input and is indeed SG-derived on all25; it does not identify the origin of A.
This correction changes no map, score, case denominator, or initializer.

| Cases | Provenance of the actual saved A,b | Scope |
|---|---|---|
| MIIT3 | Existing four-quarter-turn image-only SG matching, fixed RANSAC and positive-similarity selection | Our direct initializer recipe |
| Lung20 | Existing single-orientation image-only SG matching, fixed RANSAC and positive-similarity fit | Our direct initializer recipe |
| HistoReg1 | Historical CPU DHR initial-only native-image field, sampled/factored and conjugated into the physical512 canvas | Not the direct SG/RANSAC recipe; saved canvas A is not an exact similarity |
| Kidney1 | Historical GPU DHR initial-only run directly on the accepted physical512 canvases, sampled/factored at257 vertices | Not the direct SG/RANSAC recipe; exactly identified below |

For HistoReg, the original local archive is
`D:/QC_optimization_data/digital_topology_wsi/DHR_CD68_CD4_initial_only/`.
Its configuration disables nonrigid registration. CPU resampling of its saved
`HistoReg_CD68_to_CD4_initial_only/Results_Final/displacement_field.mha`
with the archived postprocessing parameters, followed by the existing
`factor_affine_teacher`, reproduces the native affine in
`docs/digital_topology_wsi/q1_dhr_initial_affine_histo_257_float32.npz`
exactly. `tools/digital_histo_canvas_affine.py` and the saved
`birl_anhir_dev/canvas/histo_initial_affine.json` specify its exact native-to-canvas
conjugation; rounding its result to float32 reproduces the actual current A,b.
The canvas matrix is
`[[.9684233069419861,.11542077362537384],[-.11578276753425598,.9676365256309509]]`
with offset `[-.03893144428730011,.0453045628964901]`. Its unequal diagonal
entries and unequal-magnitude opposite off-diagonals prohibit calling this
stored matrix an exact similarity. Positive affine is the valid general scope.

For kidney, the exact original source remains on AI under
`/home/ET/zhxu/codex_runs/digital_topology_wsi_20260929/`:

- `birl_rat_initial/config.json`, `runtime.json`, and `deeperhistreg.log`;
- `birl_rat_initial/birl_rat_initial/Results_Final/displacement_field.mha`
  and its `postprocessing_params.json`;
- `birl_rat_initial_affine.npz`, whose five arrays are bitwise identical to
  the current project's `data/rat_kidney_initial_affine.npz`.

The original fixed/moving `rat_kidney_*512.png` pixels are also bitwise equal
to the current accepted canvases. A read-only CPU replay of only field sampling
and affine factorization, using `dhr_map_at_unit_queries` on the257-square
identity grid and then `factor_affine_teacher`, reproduces **all five NPZ arrays
exactly**, maximum absolute difference0: raw sampled map, residual teacher,
reference, matrix and offset. This is stronger provenance evidence than a
filename or an approximate matrix match. The saved matrix is
`[[.9442551732063293,-.03775688260793686],[.03775688260793686,.9442551732063293]]`
and offset `[.046108126640319824,.02435700222849846]`.
The historical configuration explicitly has `run_nonrigid_registration=false`,
runtime has `initial_only=true` and `nonrigid_seconds=0`, and the log ends
after initial registration and field export. The helper's generic
"teacher" array names therefore do **not** imply use of a full nonrigid teacher
in this initializer. The archived whole-call time4.06799258s is historical,
not a current end-to-end measurement.

Both DHR initial-only recipes use their multi-feature initial stage with
SuperPoint/SuperGlue enabled, SIFT/SP-RANSAC disabled, feature sizes
150/200/250/300/350/400, angle step180, keypoint threshold.005, matching
threshold.3 and50 Sinkhorn iterations. HistoReg uses original images at
loader ratio.1 on CPU; kidney uses physical512 canvases at ratio1 on GPU.
Thus SG is an underlying component but these are not interchangeable with the
direct initializer used in the other23 cases. Both identified paths are
image-only initial registration, not anatomical-landmark or full-nonrigid-field
inputs. A current raw-raster-to-map timing must reconstruct and charge each
actual case's recipe, or explicitly declare a changed initializer; replaying
one common direct-SG recipe would not reproduce the frozen experiment.

Earlier shorthand in this review (including the opening comparison table and
native shared-initializer sections) is superseded by this provenance correction.
The native shared-A counterfactual still compares the same mathematical A,b;
its scientific conclusion does not require identical initializer algorithms
across specimens. Neither that counterfactual nor the conditional optimizer
times alone constitute full-pipeline timing.
