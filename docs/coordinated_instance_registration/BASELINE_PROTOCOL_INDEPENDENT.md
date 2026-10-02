# Independent experimental-protocol check, 2026-10-02

Independent Astra checker; actual local images, masks, maps, CSVs and installed
DHR source inspected. No optimizer, main scorer, labels or historical result was
changed. New focused tests are in `tests/test_coordinated_dhr_native_coordinates.py`.

## Native DHR coordinates and released-preset scorer

**No consequential coordinate or denominator bug found in
`tools/coordinated_dhr_released_score.py`.** The declared map is the saved MHA
displacement interpreted bilinearly, with preprocessing pixel-center coordinates.
The native-query routine is not restricted to the old 512-square case.

Let `s_f,s_m` be loading ratios, `p_f,p_m` left/top padding vectors, `r` the
initial preprocessing reduction ratio, and `x` an original fixed zero-based
pixel-center coordinate. Its displacement-field coordinate is
`z=((x+.5)*s_f+p_f)/r-.5`. For sampled saved pixel displacement `d(z)`, the
original moving coordinate is `y=((z+d(z)+.5)*r-p_m)/s_m-.5`.
This agrees with `digital_compare_appearance.py:41` and with installed DHR's
`dhr_utils/utils.py:60` and `:673` plus `dhr_preprocessing/general_preprocessing.py:37`.
The installed package root is
`D:/QC_optimization_data/digital_topology_wsi/dhr_clean_venv/Lib/site-packages/deeperhistreg`.

Crucial rounding detail: DHR uses `F.interpolate(scale_factor=1/r,
recompute_scale_factor=False, align_corners=False)`. **Use nominal `r`, not
original extent divided by the floored new extent.** Actual resampled coordinate
ramps verify this; the wrong extent rule differs by more than 0.5 original pixel
in the 127-by-91, `r=1.37` fixture. The saver scales normalized displacement by
actual `W/2,H/2`, producing `(2,H,W)` rather than channel-last arrays.

Verification: **10 passed in 9.65s**, comprising eight new independent tests and
the scorer's two existing tests. New fixtures use actual installed resampling,
padding, `DisplacementFieldSaver`, and SimpleITK readback. They cover unequal
rectangles, `r=1.37/1.19`, loading ratios `.61/.73`, identity and nontrivial
native affine maps (error below `3e-5` native pixel), and an end-to-end three-pair
manifest with six native TIFFs and 124 nominal CSV IDs. The latter scores all
122 available paired labels, retains a deliberately reflected map's accuracy
while reporting its local folds, and retains a deliberate failed pair with
denominator three and no complete aggregate. Chunked corner formulas were also
read independently; their reference determinant is one on the saved lattice.

Command (PowerShell, from active worktree):

```powershell
$env:PYTHONPATH='src;.'
& D:/QC_optimization_data/digital_topology_wsi/dhr_clean_venv/Scripts/python.exe -m pytest tests/test_coordinated_dhr_native_coordinates.py tests/test_coordinated_dhr_released_score.py -q --basetemp=D:/QC_optimization_data/tmp/native_coordinates_independent_endtoend_20261002
```

**Scope limit:** this establishes the declared saved-field/preprocessing map,
not equivalence to every optional native rendering/evaluation path.
Installed `dhr_utils/warping.py:212` uses SciPy `map_coordinates` with its default
cubic order for `warp_landmarks`; its landmark preprocessing divides coordinates
without half-pixel adjustments. `dhr_deformation/apply_deformation.py` and
`warp_pyvips` expand a field to loaded image extents, potentially using rounded
extent ratios. Final full-image exports are disabled in the new runner. Keep
the explicit bilinear scoring convention; do not label it an official DHR score
or assert equivalence to a separately rendered full-image output without checking.

## Actual preprocessing gap: MIIT mask includes background

`coordinated_real_case.py:254` constructs the image-loss mask from inverted gray
`>.04` (original gray below 244.8). On all six actual MIIT TIFFs it selects
essentially the entire resized image rectangle. Supplied `masks/tissue_mask.tif`
is binary, in exactly the original image dimensions. Resizing this mask with the
saved layout gives the following background mass inside the current loss mask:

| Fixed section / pair | Eligible raster pixels | Background fraction of eligible mask |
|---|---:|---:|
| 3 / 2-to-3 | 226080 | 23.7168% |
| 8 / 7-to-8 | 239112 | 23.1820% |
| 11 / 10-to-11 | 217592 | 24.7115% |

No supplied tissue mass is excluded by this threshold. Native background median
RGB values range from 217 to 238, so the threshold failure is expected. This is
an **evidence/weighting mismatch**, not proof that replacing the mask improves
registration, nor evidence of adaptive overlap cropping. The existing fixed
denominator and retained out-of-bounds loss do work as declared. A smallest
decisive follow-up is one explicit supplied-tissue-mask control at unchanged
initialization, weights and budget, with all evaluation landmarks retained.

## Actual target-class restriction: lung fixed boundary

For the saved common positive similarities, compute every native target in
residual coordinates `t=A^{-1}(target-b)`. An identity-boundary residual maps
the square onto itself, so `t` outside the square is unattainable. The exact
Euclidean lower bound here is `512*||A(t-clip(t,0,1))||`, since A is a similarity.
This differs from merely asking whether the source landmark is near a boundary.

| Direction | Unattainable label IDs | Maximum error lower bound, canvas px |
|---|---|---:|
| he-to-cd31 | 80 | 3.5084 |
| he-to-ki67 | 52,56,80 | 15.6535 |
| he-to-prospc | 52,56,80 | 19.8657 |
| cc10-to-cd31 | 52,56,80 | 5.6780 |
| cc10-to-ki67 | 80 | 1.0490 |
| cc10-to-prospc | 52,56,80 | 14.2282 |

Thus **14/1600 observations, six of twenty directions** are outside the permitted
image. The equal-direction mean lower bound over all twenty is only
**0.07059 canvas pixel**, so this finding can explain a hard tail limitation but
does not explain the entire mean error. A native free-boundary DHR comparison
is a useful application baseline, not an isolated optimizer/geometry comparison.
Do not remove these points or relax evaluation to hide the restriction.
All 328 available MIIT targets are inside the corresponding affine square;
minimum residual-boundary distances for its three pairs are 17.8877,31.7836,
27.1289 canvas pixels. The same impossibility argument does not apply to them.

## Checks that did not reveal a consequential error

- All six MIIT TIFF series have axes `YXS`, shape `(height,width,3)`; PIL RGB
  arrays equal tifffile arrays exactly. Rebuilding each saved 512 canvas from
  the TIFF and saved resize/padding metadata gives maximum pixel difference zero.
  No resolution/orientation tags were present; physical units remain unknown.
- Existing raw machine matches use `(pixel+.5)/512`, and full target coordinates
  are `A*p+b`; eligibility is static. None of the eligible actual MIIT machine
  targets lies in the artificial canvas padding. The label denominator remains
  124 nominal IDs and 123/107/98 paired finite coordinates, independent of method.
- Published Borda conversion code uses plain scale multiplication
  (`handlers/utilities.py:359`, `run_generate_landmarks.py:128` in the local
  `external_lung_lesion3_borda` source), whereas `digital_lung_lesion3_score.py:40`
  uses `(p+.5)/10-.5`. This is a disclosed nonofficial convention difference of
  0.45 native 5%-image pixel on both axes. Re-scoring the *same* lung maps using
  publisher scaling changes aggregate canvas mean from 4.520228 to 4.519870
  (analytic), 4.599464 to 4.599592 (old F2), and 5.133593 to 5.137651 (DHR).
  It does not change that ranking or explain the observed errors.
- MIIT origin sensitivity was checked without changing maps: applying a common
  -1 native-pixel correction to both CSVs preserves the aggregate analytic-versus-
  DHR mean ranking, although the very close pair10 mean comparison reverses.
  Do not elevate small per-pair advantages above the unverified origin convention.

Reproduction scripts (read-only scientific calculations, no report overwrites):
`D:/QC_optimization_data/tmp/independent_protocol_probe_20261002.py` and
`D:/QC_optimization_data/tmp/independent_protocol_mask_boundary_20261002.py`.
Inputs are `miit_three_rotations_t153`, `lung_all20_fixedrecipe_t123`, the
existing `lung_lesion3_eval` native annotations/canvases and its saved all20
affines. F2 numbers above refer to that older fixed-recipe run, not the corrected
reserve run. No new claim about current F2 ranking is made.

## Minimum actions

1. Complete the released native fast/standard DHR comparison using the now-tested
   explicit saved-field convention and the existing fixed available-label sets.
2. Before redesigning optimization, test the supplied MIIT tissue-mask control;
   label the existing threshold as an intensity mask rather than verified tissue.
3. Carry the six-direction lung range restriction into interpretation of tails
   and boundary comparisons; any changed domain must be a separately declared arm.
