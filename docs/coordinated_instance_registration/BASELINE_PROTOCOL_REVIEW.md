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
