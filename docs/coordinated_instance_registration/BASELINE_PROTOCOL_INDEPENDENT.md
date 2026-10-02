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
`outputs/coordinated_instance_registration/check_sources/independent_protocol_probe_20261002.py`
and `outputs/coordinated_instance_registration/check_sources/independent_protocol_mask_boundary_20261002.py`.
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

## Subsequent bounded MS objective check

No production-code defect found in `coordinated_multiscale_evidence.py` and its
`coordinated_real_case.py` hook. An independent literal P1 evaluation, five image
rasters, fractional/zero masks, float32 evidence and float64 geometry reproduce
the complete scalar to `5.55e-17` and every vertex gradient to `4.44e-16`.
Separately adding each term's VJP differs by at most `2.22e-16`. Active
`3*ARAP + 1e-4*shape + .1*matches + finest OOB` occurs exactly once. Coarse OOB
is computed as an unused intermediate, not added to the objective or its VJP.
An instrumented tiny optimizer gave 18 complete objective calls in either arm:
continuation raster counts `{8:6,16:12}`, MS `{8:18,16:18}`. Each MS stage's
accepted scalar equals its full accepted scalar; final selection uses the same MS.

Work accounting: five scales mean five times the image-term call count, not
five times measured runtime. For the actual 300-gradient/332-objective recipe,
MS samples 115,929,088 raster queries versus continuation's 25,493,504,
**4.547397 times**. The backward raster-query count is five times. Legacy
`query_count=512**2` is not the total MS query count per objective, which is
`32**2+64**2+128**2+256**2+512**2=349184`.
Source: `outputs/coordinated_instance_registration/check_sources/independent_multiscale_probe_20261002.py`.
This is a correctness check, not evidence that MS improves anatomy; its observed
negative registration result remains separate.

## Subsequent bounded NGF objective check

Current `coordinated_ngf.py`, Evidence hook and pilot selector match the stated
postwarp-intensity functional. Explicit finite-difference matrices independently
verify central interiors and both one-sided endpoints, including two-pixel axes;
maximum transpose inner-product error is `5.33e-15`.
With `c=a dot b`, `A=|a|^2+eps_f^2`, `B=|b|^2+eps_m^2`, the independent derivative
is `-2*m/Z*(c*a/(A*B)-c^2*b/(A*B^2))`. Passing this through an explicit `G^T`
and the actual image sampler/P1 map reproduces the implementation's entire vertex
VJP to `3.33e-16`; literal image scalar difference is zero. A directional check
gives AD `0.0865108486398` versus finite difference `0.0865108484849`.

A nonuniform, identity-boundary spatial-map fixture separates `G(warped I)`
from `warp(G I)` by `0.0349583`; the implementation follows the former.
Independent initial-affine epsilon calibration agrees, epsilon stays frozen
across candidates, fixed-epsilon contrast sign reversal is symmetric, and a
flat moving image has loss one and zero image gradient. No zero identity-loss
or zero identity-gradient assumption was imposed with finite epsilon.

Runner review confirms raw moving intensity is sampled once through
`A*f_P1+b`, then differentiated. Original raw fixed-mask support remains; no
supplied tissue support or MS mode is accepted in this pilot. Stage acceptance
uses that stage's NGF; prefix/best-full/final selection uses complete NGF512.
ARAP3, shape1e-4, match.1 and OOB remain single-counted in the shared complete
objective. No MIND-based final selector leaks into the NGF arm.
Source: `outputs/coordinated_instance_registration/check_sources/independent_ngf_probe_20261002.py`.
No production edits or repeated broad test suite were needed for this check.

## Same-functional stiffness optimizer: independent pre-run check

The batch-one, unit-square actual `p1_arap_energy` equals half the area integral
of the squared distance to proper rotations. Freezing the current face rotations
gives a touching upper quadratic. For a unit scalar direction its fine-interior
Hessian is the standard P1 stiffness `K`, not `2*K`: diagonal four and four
axis-neighbor entries minus one. The factor three comes only from the existing
ARAP weight. Multiple-batch averaging or a nonunit direction would change this
normalization and is outside the proposed specialized implementation.

Independent assembly from triangle shape gradients, not the production stencil,
checks fine grids with 2, 3, 4 and 8 intervals. Maximum touching-value error is
`1.1e-18`, touching-gradient error `2.13e-16`, frozen-Hessian/FE error `1.78e-15`,
and FE/five-point error `1.78e-15`. These are arithmetic checks of the stated
majorizer, not a Hessian claim for the complete nonlinear image objective.

The proposal prolongation is raw tensor-bilinear interpolation with zero coarse
boundary, NOT nested P1 interpolation of a map. For integer refinement `r`,
the independently constructed one-dimensional hat matrix satisfies
`S=B^T*T_f*B=T_c/r` and `M=B^T*B`, with diagonal `(2*r*r+1)/(3*r)` and adjacent
entry `(r*r-1)/(6*r)`. Consequently `P^T*K*P=S tensor M+M tensor S`.
Orthonormal sine modes diagonalize both factors; this gives exactly the spectral
denominator stated in `OPTIMIZER_REDESIGN.md`. For 2, 3 and 4 coarse intervals
and refinements 1, 2 and 4, explicit dense Galerkin error is at most `2.67e-15`,
and the weighted-three spectral inverse differs from a dense solve by at most
`2.22e-16`. Every nonzero test gradient has negative gradient/direction pairing.

The existing analytic safety scale with theta `.95` reaches the closed
contracted polytope `Q >= eta+.05*(Q0-eta)`. The proposed strict-interior search
is not a larger deformation class; it excludes exact saturation of that face.
The floor must remain tied to the original directional-stage anchor, not be
refreshed after accepted inner steps. This subsection currently verifies the
mathematical proposal and the subsequent implementation checks below.

The production `DirichletGalerkinStiffness` inverse agrees with the independent
dense FE/Galerkin solve to `1.67e-16`, its operator action to `1.78e-14`, and its
raw prolongation with independently sampled hat functions to `3.89e-16`.
A nonidentity-anchor fixture computes corner determinants independently using
2-by-2 matrix determinants: the first active bound is `0.4203232650592689`
versus production `0.4203232650592685`. Eight accepted steps preserve the
ORIGINAL contracted floor, physical-coefficient reconstruction and exact fixed
boundary. Literal recorded Armijo inequalities pass. A deliberately high-curvature
nearby target, with zero backtracking allowance, produces `line_search_exhausted`
and preserves the original anchor and coefficients; no rejected trial is exported.

An instrumented tiny complete application builds the old analytic and new
stiffness per-raster Evidence on identical inputs. At a common nonidentity map,
all objective parts, the complete scalar and every vertex gradient are bitwise
equal at raster sizes 8 and 16. The new path makes 15 observed complete objective
calls, exactly its reported count: four gradients, four accepted steps and one
backtrack, plus initial/prefix/final calls. Final selection equals the minimum
full-raster objective across the initial map and all accepted directional-stage
prefixes; the exported binary certificate passes. The same frozen matches loader
and prior weights are retained by direct integration inspection (this tiny fixture
has no point-match term). No production correction was required by this bounded
check. GPU performance and anatomical usefulness are not established here.

Reproducible source:
`outputs/coordinated_instance_registration/check_sources/independent_stiffness_probe_20261002.py`.

### Actual optimizer experiment: bounded independent postrun check

All six saved maps (Adam900 and stiffness300 on the three MIIT directions) have
finite float64 vertices, exactly unchanged saved boundary and positive affine
determinant. Independently evaluated normalized corner minima are
`.363924/.327949/.309700` for Adam900 and `.652874/.750933/.678257` for stiffness300,
all above `.001`. Independent triangle selection plus explicit barycentric solves
reproduce every available landmark error: 123/107/98 labels, 328 per arm, with
maximum discrepancy `1.21e-13` canvas pixels and `9.65e-13` native moving pixels.
Equal-pair means independently reproduce `3.55935383115` and `3.58415412845`.
The complete stiffness score file is authoritative; the earlier score file's
archived-path failure is not an analytic-prediction failure.

Adam logs contain 900 gradients and 910 evaluated inner iterates per case;
stiffness logs contain 300 gradients and 300 accepted steps per case. Every
stiffness initial alpha is one: the feasible-bound fraction is not limiting
these initial trials. Backtracks are genuinely complete-objective Armijo
rejections, not geometry failures: independently recounting trials gives
1265/1355/1274 halvings and 1877/1967/1886 complete objective evaluations.
Each trial alpha is the recorded power of one half, earlier trials fail Armijo,
and the final trial passes. Best-full prefix selection agrees with each saved
report. Independently taking the median directional secant ratio
`2*(Eaccepted-Ecurrent-alpha*g_dot_d)/(alpha^2*(-g_dot_d))` across all 60 accepted
level-257 steps gives `149.3379108/179.8851258/118.4886769`; accepted-alpha medians
are `1/128,1/128,1/64`. These are finite-step remainder ratios, not Hessian
eigenvalues. The weighted-ARAP contribution is at most one by its touching
quadratic bound, using its own component-gradient subtraction. The author's
separate fresh-final-map component probes were not independently rerun here.
Prediction configurations contain only image, affine and machine-match
input paths; reviewed prediction code has no evaluation-annotation reads.
The manifest's label-free prediction flag does not erase the explicitly declared
label-informed development history. These checks support the negative reported
comparison, not anatomical equivalence or a general impossibility conclusion.
Run the same probe with `--postrun`; no GPU or prediction rerun is needed.

### Native DHR existing-22 expansion: bounded pre-run review

No blocking coordinate or cohort defect found in the new existing-input/scoring
wrapper and the small released-runner extension. The 20 lung rows enumerate all
ordered distinct fixed/moving stain pairs once; the two other directions remain
fixed CD4 to moving CD68 and fixed HE to moving PanCytokeratin. Nine native images
are used, with no external affine, masks, machine matches or annotation inputs.
The runner still invokes DHR as `(moving/source, fixed/target)`, preserves its
released STANDARD algorithm settings, and keeps the original MIIT default rows.
All 22 attempts must be terminal before the scorer opens any manual labels.

Independent NumPy border-bilinear sampling of a nonuniform saved field checks
the native center conversion, unequal rectangular images, asymmetric padding,
noninteger initial ratios 1.37/1.19 and unequal load ratios .73/.61. The formula is
`z=((p_fixed+.5)*s_fixed+pad_fixed)/r-.5`, followed by
`p_moving=((z+d(z)+.5)*r-pad_moving)/s_moving-.5`.
The wrapper uses native image extents here, not 512; only the final comparison
uses the existing moving-image per-axis 512 layout. Maximum oracle differences
are `7.99e-6` native pixels and `1.44e-5` canvas pixels on these small fixtures.
The previously verified nominal-ratio resampling convention remains unchanged.

Inspection confirms the original 80/77/69 paired-ID policies, including kidney's
two fixed-only IDs, and the existing lung 50%-to-5% center convention. Failure
rows remain in denominators, lung is summarized over all 20 correlated directions,
and equal-specimen aggregation weights lung/Histo/kidney once each only when
all succeed. There is no cross-specimen native-pixel average or topology-based
accuracy exclusion. This is the declared bilinear saved-field evaluation, not
proof of equality with DHR's optional cubic landmark/image-export path.
Source: `outputs/coordinated_instance_registration/check_sources/independent_native22_probe_20261002.py`.
No registration, broad duplicate test suite or real-label access was performed.

The subsequent corrected-frame analytic22 control was also checked against its
ACTUAL archived configurations: reserve-t134's original analytic references
and the t88 Histo/kidney reports. All 22 ordered case names and fixed/moving
canvas basenames match the intended directions. Per-key comparison confirms
only `mind_frame=shared_affine` and output paths change, apart from equivalent
Path conversion; identity initialization, original affine/machine matches and
all optimizer/objective settings remain. Its scorer reuses the same native
readers/layouts, evaluates the saved affine-composed P1ac map and retains the
same all-terminal-before-labels policy and specimen aggregation. This check is
included in the native22 probe, without opening real annotations or running A22.

After BOTH actual 22-case prediction batches completed, the bounded postrun check
reaggregated all 1746 per-label records per arm, verified identical 80/77/69 ID
sets and all successful denominators, and confirmed actual A configurations
still change only frame/output with 300 gradients and zero failed trials.

|Existing specimen|Native DHR initial mean|Native DHR final mean|Shared-frame A300 mean|
|---|---:|---:|---:|
|Lung, equal20-direction|6.29170770|6.04661547|4.56854112|
|Histo|2.33911053|0.71232802|0.85190347|
|Kidney|4.32293049|1.90816268|2.36486646|

Units are the existing moving 512-canvas pixels. Equal-specimen final means
reproduce `2.88903539` for native DHR and `2.59510368` for shared-frame A, not
22-independent-patient estimates. The correct complete lung F2 reference is
reserve-t134: independently reaggregated `4.55134379393`, not old t123's result.
Stored diagnostics mark all 22 A exports valid and all 22 DHR fields locally
nonpositive somewhere; DHR accuracy rows are nevertheless retained.

One ACTUAL nonuniform native field per specimen (he-to-cc10, Histo and kidney;
226 labels total) was also sampled by the independent NumPy float64 bilinear
oracle. Maximum discrepancy from the float32 production-coordinate evaluator
is `0.0006362` native pixels, or `0.00003287` canvas pixels. This rules out a
material native/512-unit mismatch for these checked cases. The new saved-native-
initial-only scorer preserves the same dataset dispatch, ID sets and per-axis
canvas scaling, reuses the already checked affine-frame evaluator, and reads
no dense field or performs new registration. Its table aggregates above were
checked separately. Run the native22 probe with `--postrun` to reproduce.

### Coupled finite-displacement seed: initial independent card check

The announced exact card's normalization is consistent. Differentiating its
UNWEIGHTED coupling `c*sum||z-u||^2/Z` and its frozen-identity-rotation physical
quadratic `3*u^T*G*u/(2*128^2)` gives
`[2*c*I+3*Z*G/128^2]u=2*c*z` for each component. The factor two belongs to the
unhalved coupling square. The discrete block minimizes `w_i*C_i(k)+c||k-u_i||^2`;
putting the fixed mask on the coupling instead would invalidate that screened
DST solve. Both components remain in 128-raster pixel units until division by128.
`G` is the already checked raw-bilinear Galerkin stiffness, not nested-P1 transfer.

The fixed node coordinates, nine patch offsets and immutable mask denominators
define a nodal/patch surrogate, not exactly the original 128-raster image sum.
Labels are residual displacements: their original-moving change is `A*k/128`.
Eight fixed x/y safe-construction cycles start at identity, refresh legal geometry
after each step and retain one frozen desired target. This guarantees neither
target reachability nor an original-objective decrease during initialization;
the card explicitly permits a higher-energy legal seed and retains identity in
the final original-E512 candidate set.

The upstream [coupled_convex implementation](https://github.com/multimodallearning/convexAdam/blob/main/src/convexAdam/convex_adam_utils.py#L83-L97)
directly confirms the six borrowed constants and local-average update. The card
correctly distinguishes its own global quadratic solve and L1 costs from that
method; the borrowed constants do not imply identical cost normalization.
The subsequent bounded independent production checks also pass:

- A pure NumPy zero-padded bilinear sampler independently reconstructs fixed
  vertex/patch queries, eight-channel L1, rotated-affine original OOB, fixed-mask
  denominators, empty patches and label order. Maximum cost error is `1.11e-16`
  with float64 features and `1.44e-8` with production-like float32 features.
- Independently assembled fine triangle stiffness and raw hat interpolation
  give a dense screened solve. The entire 13-solve/13-label sequence agrees to
  `8.89e-16` in final nonzero displacement and `4.45e-16` across recorded block
  energies. Both blocks decrease the same fixed-c objective; zero-cost ties
  preserve zero displacement.
- A raw one-axis checkerboard has 384 nonpositive target corners, yet all 16
  returned construction states have exact fixed boundary and actual ratios
  above `.001`, with minimum `.0010000000390242`; the desired target stays fixed.
  A stronger two-axis checkerboard instead triggers the declared explicit
  rounded-margin failure before output. Repeated `.95` steps can exhaust finite
  precision: there is no uniform numerical-margin or unconditional-completion
  guarantee. Such a real-case failure would concern this 16-step constructor,
  not refute coupled matching. No floor relaxation, repair or silent fallback
  was added.
- The actual 257-map/512-image application hook was instrumented with one
  injected legal seed whose full objective is HIGHER than identity. The first
  coarse-stage evaluation sees that seed, not identity; final selection includes
  identity, seed and accepted prefixes. The four-gradient test makes exactly
  19 observed/reported objective calls, including one extra seed call, and saves
  the supplied initializer separately. Core construction is tested above;
  this injected-seed fixture specifically isolates integration/selection.

Source: `outputs/coordinated_instance_registration/check_sources/independent_coupled_seed_probe_20261002.py`.
No production correction was needed. The author subsequently confirmed READY/
FROZEN; the only later edits were a diagnostic-key correction and an explicit
`.001` floor guard, both inspected. No GPU or GT was used in this check.

### Coupled seed: actual three-case postrun check

The actual coupled exports pass the reused independent P1/landmark check on all
123/107/98 available labels. Maximum error discrepancy is `5.69e-14` canvas pixels
and `4.53e-13` native pixels. Equal-pair mean/p90 reproduce
`3.54760907862033 / 5.91743722699801`; this near-neutral mean and worse p90 do not
establish a useful improvement over the original corrected Adam300.

All three final maps have exact fixed boundary and independently evaluated
minimum corner ratios `.342235/.345279/.305014`. Saved 65-square pixel
coefficients independently reconstructed with raw tensor-bilinear hats recover
the archived initializer vertices to `5.73e-17`. Their minimum ratios are
`.382466/.466193/.390679`; all 48 construction scales are one, so these actual
seeds suffered neither folds nor construction contraction.

The first two seeds have higher E512 than identity. Fresh CPU evaluations of
their saved initializers under the original full512 and first-stage32 Evidence
agree with the stored seed objective and first-stage anchor/step-zero objective
within `1.49e-8` (float32 evidence arithmetic). Thus the higher-energy seeds
actually entered refinement; they were not silently replaced with identity.
All three runs have 300 gradients, 333 objective calls, zero failed trials and
selected stage9. Final original-E selection agrees with the minimum over identity,
seed and accepted prefixes. Run the coupled probe with `--postrun`; no prediction
rerun or additional GPU experiment is involved.

### Fixed contrast-calibrated H proxy: independent preproduction check

The approved single-recipe card and implementation pass a bounded independent
check. This does not predict registration accuracy or establish real-slide
stain invariance. The checker uses the scalar cofactor expression
`cH=(.7095*OD_R-.0249*OD_G-.2274*OD_B)/.377799` and manually sorted linear order
statistics, not the implementation's matrix inverse or quantile call.

- Random-RGB normalized float32 output agrees exactly; ideal mixed Beer--Lambert
  output agrees within `2.98e-8` when calibration support is held fixed.
  Pure E/DAB leaves at most `3.99e-10` normalized floating-point residue, not
  a new threshold or fallback. White, zero/saturated RGB, negative H, empty
  calibration support, and sparse-positive q99=0 follow the declared formula.
- Default raw loading is bitwise unchanged. Original grayscale masks and their
  area weights are identical. Independent area averaging and a separate NumPy
  bilinear sampler verify normalized512 -> area pyramid -> frozen affine ->
  descriptor order. Under a strongly rotated, partially outside affine, the
  captured pre-descriptor intensity agrees within `5.97e-8`; no renormalization
  or support-based query removal occurs after the original calibration.
- All 25 actual completed control configurations differ only in preprocessing
  and output. A nonterminal case25 blocks scorer-input creation. A deliberate
  all-failed fixture retains the full 3+22 denominators and all original MIIT
  geometry, match and archived F2/DHR path targets. No manual labels were read.

Source: `outputs/coordinated_instance_registration/check_sources/independent_stain_proxy_probe_20261002.py`.
No production correction was needed. The adapter's relative-path operation
requires a common filesystem volume on Windows; the first checker fixture
exposed this when its temporary output was on C: and inputs on D:. The fixture
now uses D: and passes; the approved Linux same-tree production is unaffected.

### H proxy: actual all25 postrun check

All 25 predictions completed before scoring. The actual source controls are
`miit_multiscale_control_t19` and `existing22_shared_affine_a300_t20`; direct
configuration comparison confirms only preprocessing/output changed. Every
case has 300 gradients, 332 objective calls, zero failed trials and minimum-H-
full-objective candidate selection. Every stored float64 P1ac map has its exact
identity rectangle boundary and positive affine. Independent four-corner
calculations give an overall minimum ratio `0.00686149536872144 > .001`.

Independent raw CSV parsing and literal triangle barycentric linear solves
reproduce all 2,074 paired-label errors (328 MIIT, 1,600 lung, 77 Histo, 69
kidney), with maximum discrepancies `1.14e-13` canvas pixels and `1.04e-12`
native pixels. All 15 unique original canvases reproduce the saved grayscale
support counts, H99 scales and zero fractions under the separate scalar oracle.
Every per-level fixed-mask denominator equals its original support count times
`(side/512)^2`; shared-affine setup reports one intensity warp per scale.

Direct per-row aggregation also reproduces the coordinator's comparison:

| Cohort | H mean | H-minus-gray mean | H-minus-gray mean pair-p90 |
|---|---:|---:|---:|
| MIIT, 3 directions | 3.56822147 | +0.01946957 | +0.03437110 |
| Lung, 20 directions | 4.56755994 | -0.00098118 | +0.04273073 |
| Histo | 0.92517151 | +0.07326804 | +0.07859755 |
| Kidney | 2.35532726 | -0.00953920 | -0.20395749 |

Units are 512-canvas pixels; these are four already-viewed specimens, not 25
independent patients. The mixed/near-neutral mean changes and worsening tails
in three cohorts do not establish a useful general improvement from this proxy.
H and gray objective totals remain different functionals.

The weak `miit_7_to_8` moving channel retains H99 `0.127710095311532` and zero
fraction `0.687592053022541`. Its normalized512 MIND variance median is
`0.00358927669003606`, but `0.273033261299133` of original support weight remains
at or below epsilon. A separate NumPy shift/patch-average implementation checks
all five reported variance medians to `3.00e-9` and reproduces weighted epsilon
incidences. Scaling boosts contrast but cannot restore the clamped-zero signal.
Run the same independent probe with `--postrun`; no optimization or GPU rerun
was performed.

### MatchAnything point substitution: independent preproduction review

The bounded review found and adjudicated one concrete source/card mismatch
BEFORE production. The pinned released `CoarseMatching` reads `mtd_spvs=True`
and takes every coarse pair strictly above `.1` after two-cell border removal.
Its configured `FORCE_NEAREST=True` is unused by this class. A direct small
confidence-matrix fixture retains two deliberately non-mutual neighbors, while
rejecting an exact-threshold entry and a border entry. The original card and
`force_mutual_nearest` metadata incorrectly inferred behavior from the unused
flag. The coordinator chose to preserve the released algorithm; the author
corrected the card and metadata to distinguish configured versus active policy.
No nearest filter, deduplication, parameter change or altered forward was added.
The official source lineage is linked by the
[author repository](https://github.com/zju3dv/MatchAnything) to the
[released inference package](https://huggingface.co/spaces/LittleFrog/MatchAnything/blob/6a7bcb589ec8da3a9e861e799122beaa5eba2193/imcui/third_party/MatchAnything/README.md).

The actual fine branch uses TOPK=1 plus unmasked local regression and retains
coarse `mconf`; it is not newly calibrated fine confidence. This also means
retained observations can share source coordinates. Multiplicity diagnostics
are reporting-only. Independent safe CPU reading finds 447 finite checkpoint
state tensors. The corrected actual strict-load report confirms 16,025,216
parameters, no missing/unexpected keys, NPE `[832,832,512,512]`, threshold `.1`,
FP32 and the active MTD branch. The author's actual MIIT2-to3 forward table
contains 2,910 finite retained pairs and 2,908 positive eligible pairs, independently
reproduced with the original saved affine; it has 2,903 unique source positions.
The checker did not rerun this model forward or use a GPU.

Independent injected-output/application checks pass:

- Ordinary PIL grayscale agrees exactly. A separate NumPy bilinear BORDER
  sampler under a rotated, partly outside affine matches captured model input
  within `1.24e-7`. No inversion, resize, second affine application or clipping
  of retained coordinates occurs.
- The 30-output fixture retains 28 in-domain pairs and 23 positive static
  eligible pairs; pixel-index endpoints `-.5/511.5` map to unit `0/1`. NaN,
  infinity and invalid confidence fail even in would-be discarded entries.
  There is no extra confidence threshold or source tissue selection, and fewer
  than eight positive eligible pairs fail. Source-support lookup is explicitly
  a containing-pixel diagnostic. The old loader's `eligible_matches` counts
  geometric eligibility including zero-confidence entries; the adapter's count
  is positive eligibility. The actual smoke has no zero-confidence entries.
- Literal P1 barycentric solves and an analytic robust-loss derivative agree
  with ordinary and frozen point samplers to `8.89e-16` in value and `1.34e-15`
  in VJP; a directional finite difference agrees to `4.43e-10`. A nonsymmetric
  affine and nonzero offset isolate the required moving-canvas units: offset
  cancels from the point residual but not static eligibility. Tripling all
  observations preserves total confidence-normalized point strength.
- All 25 actual source configurations change only `matches/output`. Batch
  inspection and author tests retain extraction failures without SG fallback,
  complete all extraction attempts before optimization, release the model,
  and create scorer inputs only after every prediction is terminal. New MIIT
  point provenance is explicit while the old SG table remains archived context.

Source: `outputs/coordinated_instance_registration/check_sources/independent_matchanything_probe_20261002.py`.
After the source-fidelity correction, no numerical/coordinate blocker remains.
These checks verify a point-evidence substitution, not anatomical correctness
or expected registration improvement. Subsequent scoring uses P1ac triangles;
the reused certificate's legacy Q1 representation wording describes its
four-corner residual check, not a switch to Q1 landmark interpolation. With the
unchanged rectangle boundary those checks also certify the scored P1ac map.

### MatchAnything: actual all25 postrun check

The completed `matchanything_all25_t22` run passes the bounded independent
postrun check. All 25 extraction and optimization attempts succeeded, with one
model setup, no SG fallback and no lost scoring denominator. Each saved map has
300 gradients, 332 objective calls, zero failed trials, exact identity rectangle
boundary and the unchanged positive affine; the smallest independently computed
corner ratio is `0.34688274641299965`. Actual configurations differ only in
`matches/output`, raw-gray preprocessing and all-scale fixed-mask/prewarp
metadata equal their matched controls, and selected maps minimize the new full
functional over the reported candidate set.

Reusing the independent raw-CSV/P1 barycentric checker reproduces every one of
the 2,074 errors: maximum discrepancy `1.28e-13` canvas pixels and `1.74e-12`
native pixels. All 25 new point tables are finite and in domain, with 1,932--3,169
retained pairs and 1,929--3,158 positive eligible pairs. Independent application
of the saved affine reproduces eligibility, confidence mass and loader metadata;
the optimization reports identify the new tables and the same counts/masses.
Retained source multiplicity reaches three; target multiplicity is one. There
is no deduplication or extra confidence/tissue rejection.

Independent aggregation reproduces the paired comparison:

| Cohort | New mean | New-minus-SG mean | New-minus-SG mean pair-p90 |
|---|---:|---:|---:|
| MIIT, 3 directions | 3.61640501 | +0.06765311 | +0.23303690 |
| Lung, 20 directions | 4.83976232 | +0.27122120 | +0.21376458 |
| Histo | 0.87385295 | +0.02194948 | +0.09111757 |
| Kidney | 2.23683445 | -0.12803201 | -0.04630521 |

Units are 512-canvas pixels. Kidney improves, but the other three specimen
cohorts worsen in both mean and mean pair-p90; this does not support a general
gain at the predeclared point weight. Extraction scarcity is not the failure
mechanism in this run. The experiment measures the released matcher substitution
within this fixed registration, not the quality of all possible uses of the
matcher. Runtime totals reproduce: setup `1.052606s`, summed extraction calls
`3.237218s`, summed optimization calls `110.825728s`, batch wall `115.669845s`.
Download/dependency preparation is outside those inference-run numbers.

Run the MatchAnything independent probe with `--postrun`; it reuses the prior
saved-map/CSV checker and adds point-table and comparison checks. No optimization,
network inference or GPU rerun was performed by the checker.

### SG-preserving complement: independent preproduction check

The two-arm fusion/control implementation passes the bounded algebra and
configuration review. For each original table the denominator is the confidence
sum over STATIC original-moving `A*p+b` eligibility, computed with the same
float32 stored affine and float64 point coordinates as the existing loader.
Every row, including ineligible and zero-confidence rows, is retained. Thus
each table contributes eligible mass one; the combined loader divides by two,
and global weight `.2` gives exactly `.1 P_SG + .1 P_MA` in real arithmetic.
The denominators are fixed evidence, so the same identity holds for vertex VJPs.

A separate literal-triangle/analytic-derivative oracle checks 116 retained rows
with unequal eligible masses `5.8176339540` and `37.8112623064`, a nonsymmetric
affine, nonzero offset, boundary queries and static ineligibility. The combined
loader mass is `1.9999999999999998`; weighted value and vertex VJP errors are
`1.78e-15` and `1.12e-16`, for both ordinary and frozen P1 sampling. Fusing two
copies of SG also agrees with `.2 P_SG`. A valid subunit eligible mass is accepted
when normalized confidence stays within one; an incompatible high-confidence
ineligible row is rejected rather than clipped. Zero/nonfinite masses or data,
coordinate/raster disagreement and changed affine are rejected.

All 25 actual configurations preserve the originals except
`matches/match_weight/output` for fusion and `match_weight/output` for SG2. The
SG2 arm keeps its exact original point path, both arms use `.2`, and raw-gray
MIND plus the remaining optimizer/geometry settings are unchanged. Independent
inspection of the runner and its injected failure/order test confirms all 25
composition attempts precede optimization; both 25-case manifests remain
incomplete until all 50 attempts are terminal. A failed fusion composition does
not silently become SG, but does not prevent the distinct valid SG2 control.
All failures remain in their original arm denominator. Scorer adapters are
created only afterward. No production correction was necessary.

Source: `outputs/coordinated_instance_registration/check_sources/independent_match_fusion_probe_20261002.py`.
This verifies the proposed functional and matched-strength control, not that
the two matchers supply complementary anatomical truth.

### SG-preserving complement: independent all-50 postrun check

The completed `match_fusion_all50_t23` experiment passes the bounded independent
postrun check. Both 25-case arms are terminal before scoring; all 50 attempts
succeed with 300 gradients, 332 objective calls and the original best-full
selection. Literal recomputation from the original CSV coordinates and saved
P1 maps covers 4,148 landmark errors. Maximum discrepancy is `1.14e-13`
512-canvas pixels and `1.82e-12` native pixels. The original affine and rectangle
boundary, float64 257-by-257 P1 representation and stored binary certificates
are unchanged/valid. Minimum normalized corner determinant is `0.00532227`
for fusion and `0.00258231` for SG2, both strictly above `.001`.

All 25 fused tables are independently reconstructed from their two original
sources: exact source/target row concatenation, including static ineligibility,
and confidence `c/D` with independently summed eligible masses. No original
row is removed or clipped. Retained rows range from 2,180 to 3,704, and combined
loader mass is two to rounding. SG2 uses its exact original SG table. Actual
configuration changes have precisely the declared three/two keys; raw-gray
image preparation and every scale's fixed-mask/frame metadata equal the
original controls. MIIT scorer provenance points to the current arm's table.

| Cohort | Fusion mean | Fusion - SG1 mean | Fusion - SG2 mean | Fusion - SG1 mean pair-p90 | Fusion - SG2 mean pair-p90 |
|---|---:|---:|---:|---:|---:|
| MIIT, 3 directions | 3.53471836 | -0.01403354 | -0.02942371 | -0.04978710 | -0.09528598 |
| Lung, 20 directions | 4.47100714 | -0.09753398 | -0.04601779 | -0.25114079 | -0.32800184 |
| Histo | 0.84206433 | -0.00983914 | -0.10489162 | -0.00586717 | -0.13433669 |
| Kidney | 2.24360647 | -0.12125999 | -0.13861742 | -0.11335725 | -0.05855000 |

Units are 512-canvas pixels. SG1 is the archived `.1 P_SG` control. All cohort
means and mean pair-p90 values improve versus BOTH SG1 and the matched-strength
SG2 control. This supports a modest complementary-evidence effect in this
fixed experiment, not a pure increase in point strength. It is not uniform
per-direction or worst-tail improvement: MIIT 2-to-3 slightly worsens in mean
and p90 versus SG1; kidney worst error rises by `0.03730` versus SG1; fusion
worst errors exceed SG2 by `0.14739` on MIIT and `0.07214` on lung. These are four
previously viewed specimens, not 25 independent patients or a held-out result.
A global decision to retain this one fused recipe is distinguishable from
choosing the best arm separately for each direction.

Every comparison row, cohort aggregate and equal-specimen delta is independently
reproduced from the score tables. Summed complete optimizer calls are
`114.162324s` for fusion and `106.531195s` for SG2; table composition is
`1.095903s`, and current batch wall is `222.466018s`. The earlier MA setup
`1.052606s` and extraction `3.237218s` remain explicit additional costs.
These are single serial calls, not a controlled speed claim. The checker did
not rerun optimization or matching. No production correction was needed.

Reproduction: run the same independent fusion probe with `--postrun`; it reuses
the existing literal saved-map/CSV checks with strictly declared configuration
deltas, then adds table, comparison and cost checks.

### Joint-pose candidate: independent schedule review

The approved schedule in `JOINT_POSE_FORMULATION.md` has no mathematical
blocker: the six chart columns are normalized by their original-moving-canvas
RMS displacement under the CURRENT residual and rebased pose. The lower-bound
chart derivative must include `ds/dr=sigmoid(r)`. Scales are frozen for the ten
pose steps and recomputed at the next block; a nonfinite or degenerate scale
is a reported numerical failure. The learning rates
`2.048, 1.024, .512, .256, .128` are local coordinate calibration, not fixed
actual displacement, monotone descent or equal runtime. Frozen250 and
joint50-plus250 share the residual step budget; the archived frozen300 arm
answers the separate total-gradient-budget comparison.

The mathematical review identified and the formulation now records: uniqueness
of the affine function rather than its periodic angle coordinates; only
piecewise differentiability of the image functional; fixed physical B during
residual differentiation rather than fixed softplus coordinate r; and the
absence of an informative-support guarantee even with the full determinant
floor and fixed denominator. The all-zero prewarp/all-ones descriptor example
does not prove a lower COMPLETE objective because points and OOB still count.

Independent probe source:
`outputs/coordinated_instance_registration/check_sources/independent_joint_pose_probe_20261002.py`.
Its reference uses NumPy eigendecomposition for the symmetric exponential,
literal triangle solves, bilinear pixel weights and an independent descriptor
implementation. Oracle-only composition/corner scaling and the two support
examples pass.

### Joint-pose implementation: independent tiny checks and correction

The corrected `tools/coordinated_joint_pose.py` passes the independent probe
with `--implementation` (CPU, about six seconds; no registration or matching
rerun). At identity pose, the full objective and vertex VJP exactly equal the
existing Evidence implementation, both for all-float64 inputs and for the
production float32-image/float64-geometry convention. Under nonzero translation,
rotation, scale and shear, the independently implemented image, robust point,
OOB and prepared descriptor values agree. All six physical-pose derivatives
agree with the NumPy objective finite differences to `1.14e-8`; halving the
step changes those estimates by at most `1.78e-8`. This fixture avoids the
piecewise-differentiable kinks rather than asserting differentiability there.

Changing the residual minimum changes the lower chart bound without moving
the recovered physical pose. All six mask-weighted RMS scales, including the
softplus scale derivative, agree with independent chart perturbations to
`1.33e-8`. Cached and uncached PHYSICAL-pose-frozen residual VJPs agree, while
a trainable pose with a cached descriptor is rejected. Original point targets,
weights and queries remain unchanged, including when a large pose change
moves predictions outside the original image. The point formula continues
to use the original static eligibility and original A.

The original-raster reconstruction is checked on a separate translation case:
96 raster locations lie outside the old aligned canvas but inside the original
moving image. The implementation retains their nonzero original intensities
and matches the literal rebuilt-descriptor objective. Affine/P1 commutation
agrees to `2.22e-16`, with the expected corner determinant scaling.

Claim/finding/fix: the export must represent a float64 unit-square-boundary
residual, not merely some ordered rectangle. Independent review found that the
first validator delegated this restriction to the generic binary certificate,
which permits other rectangle references, and normalized by the supplied
reference. The author confirmed a shifted-reference failing regression and
corrected the validator to require the exact generated float64 unit reference
and use its corner determinants. The independent final probe now rejects
shifted and rescaled references and float32 residuals. It also rejects a
positive affine whose full normalized floor is below `.001`, a changed boundary,
inconsistent pose/composite metadata and a negative combined affine. The valid
stored-affine floor agrees with the literal determinant ratio. The certificate
concerns the stored residual and exact binary affine parameters, not a newly
rounded materialized composite table.

The optimizer source review confirms that each residual block freezes the
physical pose, stage/final selection saves both G and Y, current/full objectives
rebuild descriptors for the corresponding G, and the joint runner records all
50 outcomes before exposing scorer adapters. Actual prewarped texture
diagnostics were also added by the author; original moving-image variance alone
could not detect the zero-prewarp example. These tests establish the stated
functional, derivatives and representation checks, not anatomical gain.
