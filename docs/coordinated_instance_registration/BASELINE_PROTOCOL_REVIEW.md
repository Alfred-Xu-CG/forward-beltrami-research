# Baseline and data protocol review — extension audit, 2026-10-02

The existing A-versus-F2 comparison is a useful matched development ablation.
The archived column called “Native DHR” is **DHR shared-affine reduced-512**:
it does not reproduce the released full fast preset, the standard preset, or
the published RegWSI evaluation. Correcting that baseline is necessary before
claiming application competitiveness. Independent numerical agreement with the
existing scorer establishes implementation consistency, not publication equivalence.

## Comparisons that answer different questions

| Comparison | Held common | Important difference | Legitimate inference |
|---|---|---|---|
| A versus matched F2 | Same images/canvases, shared positive similarity, raw MIND and frozen image matches, P1 output, regularizers and label set | Decoder/update class, schedule, actual runtime; F2 uses its declared .95 reserve | Geometry/optimizer comparison at the disclosed gradient budget; not automatically equal-time |
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

## Existing22 native STANDARD expansion: prepared, not executed

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
`images/` beside it; manual CSVs are not in the transfer list. No images have
been copied remotely and no GPU registration has been launched by this worker.

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
This validates coordinate implementation, not anatomical accuracy; no real22
registration outcome exists yet.
