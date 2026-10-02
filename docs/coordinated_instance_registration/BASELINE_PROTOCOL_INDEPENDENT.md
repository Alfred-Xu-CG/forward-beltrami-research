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

### Joint pose: independent all-50 saved-result check

`joint_pose_all50_t24` passes the independent saved-result check. The outer
manifest and both 25-case manifests are complete before any evaluation CSV is
read. Frozen250 uses 250 residual gradients and 282 objective calls; Joint300
uses 50 pose plus 250 residual gradients and 347 objective calls. All 50
complete without failed trials. Configuration differences are precisely the
declared budget/output/pose fields. Original affines, fixed/moving input paths,
fused point-table paths, point eligibility/weights and raw-gray mask source
remain unchanged. Joint exports separately retain the original affine.

All 4,148 landmark errors are recomputed from original CSV coordinates and
literal source-triangle interpolation followed by the STORED affine. Maximum
differences are `1.14e-13` canvas pixels and `1.72e-12` native pixels. Exact unit
boundaries, float64 source P1 tables, actual saved binary residual certificates
and exact binary affine determinant signs pass. Minimum normalized determinant
is `0.00577563` for Frozen250; Joint300 has residual minimum `0.00395089` and
original-affine-normalized complete-map minimum `0.00359920`, all above `.001`.

The saved physical pose equals the pose in the selected stage, its residual
minimum agrees with that stage, and the selected full objective is the minimum
over initial and all accepted paired states. All joint cases select stage 14,
the final accepted pair. Every pose block has the predeclared RMS-normalized
rate and rebased floor; its following two residual blocks retain exactly that
physical pose and adjusted floor. Independently recomputing the final complete
objective from the saved pair, original images and fused points agrees to
`2.50e-8` (float32 image arithmetic); the static point term agrees to `2.50e-16`.
This check uses literal image/descriptor interpolation and SVD-based ARAP,
not the production objective implementation.

One finite-arithmetic qualification is retained: the saved affine agrees with
the implemented `torch.matrix_exp` composition, while an independent symmetric
eigendecomposition/exponential differs by at most `8.79e-11` in the combined
affine coefficients; the ideal `exp(2s)` determinant identity differs by at
most `9.32e-11`. A worst-case check finds NumPy eig and SciPy expm agreeing to
`2.22e-16` while the implemented Torch exponential differs by `8.12e-11`.
This is not machine-precision reproduction of the ideal parameter formula.
The checked floors and topology concern the authoritative stored binary affine
and residual, so this small approximation does not invalidate those checks.

Actual final outside fractions independently agree to `7.21e-9`; all 25 final
nominal descriptor-footprint fractions agree exactly. Aligned intensity and
descriptor variances agree to `6.85e-9`. Learned affine area ratios range from
`0.910985` to `1.128654`, actual outside fractions from `0` to `0.149946`, nominal
fixed-mask-weighted footprint coverage from `0.839406` to `0.997289`, and aligned
descriptor channel variances from `0.068426` to `0.090340`. These observations
exclude a global all-zero descriptor-collapse explanation here; they do not
certify valid support at every deformed query or anatomical correspondence.

| Cohort | Joint mean | Joint - Frozen300 mean | Joint - Frozen250 mean | Joint - Frozen300 mean pair-p90 | Joint - Frozen250 mean pair-p90 |
|---|---:|---:|---:|---:|---:|
| MIIT, 3 directions | 3.56510797 | +0.03038961 | +0.02859604 | +0.09441900 | +0.05001610 |
| Lung, 20 directions | 4.29937213 | -0.17163501 | -0.16607247 | -0.31976672 | -0.30465065 |
| Histo | 0.86401889 | +0.02195455 | -0.00339447 | +0.01789019 | -0.11306335 |
| Kidney | 2.23728354 | -0.00632293 | -0.03337851 | +0.32294918 | -0.13187285 |

Units are 512-canvas pixels; both frozen controls use the same fused evidence.
All comparison rows, cohort/equal-specimen aggregates, support deltas and cost
summaries reproduce independently. Compared with the archived Frozen300 arm,
two cohort means and three mean pair-p90 values worsen; kidney worst error
rises by `2.52603` pixels despite its slight mean improvement. Lung improves
against BOTH controls, so reducing the residual budget alone does not explain
that gain. This remains four previously viewed specimens, not independent
held-out evidence or grounds for per-case best-arm selection.

Complete calls total `93.259012s` for Frozen250, `191.325626s` for Joint300 and
the archived `114.162324s` for Frozen300. Fresh batch wall is `285.155555s`.
These single-run costs do not establish a controlled speed ratio, but the
joint recipe provides no uniform accuracy/cost advantage here. Retaining the
frozen300 fused recipe globally while preserving the lung-specific pose result
as a research observation is consistent with these data.

Reporting erratum, production revision `42a599c`: all 25 saved joint reports
inherit `image_preprocessing.original_moving_features_no_affine_prewarp=True`
from the raw loader. That flag is stale: actual moving descriptors ARE built
after the original-raster A G prewarp, as the objective description, source
review, identity test and independent image recomputation establish. Saved
production reports are left unchanged; the future reporting flag can be fixed
without a numerical rerun. No map, score or evaluation choice was changed.

Reproduction: the independent joint-pose probe with `--postrun`; it shares the
previous original-CSV loader and literal P1 oracle, with joint-specific saved
affine, budget, paired selection and diagnostic checks.

### Terminal detail: independent preparation and narrow-hook precheck

The bounded CPU precheck passes; this is not a production accuracy result.
All 25 directions' original-image identities and accepted layout roles were
checked, including MIIT moving/fixed direction, Histo/kidney roles and the
HE layout reused consistently across lung pairs. The actual non-square lung
example is 892 by 661, resized to 512 by 379 with padding (0,66). Doubling
resized dimensions and padding preserves unit coordinates exactly, including
the pixel-center relation c'=2c+1/2. Direct rendering agrees exactly with an
independent explicit RGB-resize/paste/grayscale calculation. The lift agrees
exactly with an independently written separable half-pixel bilinear formula.
Mask two-by-two replication and normalized mass agree exactly; altered
accepted RGB is rejected rather than tolerated. Native dimensions,
upsampling axes and unchanged 512-point units remain explicit.

Actual-source testing first FAILED despite correct coordinates: current
Pillow12.3/JPEG8 decoding does not reconstruct the old lung RGB canvases.
A boundary probe passing raw full-resolution RGB from the existing
Pillow10.3/JPEG9 environment into the current resize routine reconstructs
both initially tested lung roles exactly. This isolates JPEG decoding, not
the BILINEAR resize/layout. The attempted all-nine historical decoder bridge
then FAILED on kidney. Checking every source under both existing decoders
establishes mixed provenance: five lung and two HistoImages accepted canvases
reconstruct exactly under Pillow10.3/JPEG9, whereas both kidney accepted
canvases reconstruct exactly under current Pillow12.3/JPEG8. The mismatching
decoders have maximum RGB-component differences 6--14 (lung), 1 (HistoImages)
and 6--13 (kidney) after accepted-size rendering. These are not numerical
tolerances or registration errors.

With the parent's explicit authorization, full-resolution lossless RGB
PNGs and one decoder metadata manifest were produced in
`D:/QC_optimization_data/digital_topology_wsi/historical_rgb_decode_cache`.
The initial seven historical caches passed local checks, but the parent's
remote smoke passed 24 directions then rejected kidney: that remote environment
uses Pillow10.4/JPEG9, not the local current JPEG8 decoder. Following explicit
parent authorization, the cache was extended to all nine sources. The two
kidney images are decoded by the existing local Pillow12.3/JPEG8 executable in
a PIL-only subprocess; their rows carry explicit decoder overrides while the
seven historical rows retain the global decoder metadata. Every one of the
nine cached decodes was independently resized under current Pillow and shown
to reconstruct accepted512 RGB exactly. No extra decoder version,
annotations, accepted-image rewrite, fallback tolerance or changes to the
original sources were introduced. Completing/verifying the initial seven-image
cache took 6.78 seconds after two HistoImages PNGs had already been written by
the failed first attempt. The later nine-image extension/verification took
7.58 seconds. Neither number is fresh-from-empty total cache cost: the earlier
failed attempt and diagnostic setup are separate and not fully timed. This bridge is
preparation cost, not added image information. A fresh remote all-direction
reconstruction check must still pass there. Execution note: after the actual reconstruction failure,
the checker read `systematic-debugging/SKILL.md` once; no subsequent skill was
invoked and no extra workflow gate was added. The cache change followed explicit
parent authorization, not an implicit expansion of that skill.

The literal static-point oracle matches point loss and vertex VJP, and using
512/8 versus 1024/16 gives bitwise-equal values and gradients. Three tiny CPU
optimizer runs show: the same-raster explicit override reproduces default
objectives, updates and saved arrays bitwise; changing terminal evidence keeps
lower-stage intensities, masks, descriptors and accepted maps exactly unchanged;
the control grid is unchanged while terminal query count changes; and the
full-resolution selector uses the changed terminal objective. An independent
NumPy image/ARAP/shape/outside computation of every accepted terminal state
agrees with its reported full objective to `1.5649e-8`. The saved tiny binary
certificate is valid. The focused probe finishes in 7.97 seconds on CPU.

Reproduction sources are
`outputs/coordinated_instance_registration/check_sources/independent_terminal_detail_probe_20261002.py`,
`independent_pillow_decode_probe_20261002.py`,
`independent_decoder_all9_probe_20261002.py` and
`produce_historical_rgb_cache_20261002.py` in that same directory. This precheck
does not establish 50-case completion, GPU cost, terminal anatomical benefit,
or the correctness of not-yet-reviewed production scores.

### Terminal detail: independent all-50 saved-result check

`terminal_detail_all50_t25` passes the independent saved-result check, with
the finite-arithmetic qualifications below. Before reading original CSVs,
the checker verifies the outer completion record, both complete 25-case
manifests, all 50 saved map/report paths and the all-50 scoring declarations.
Source review confirms that scoring adapters are created only after all
attempts terminate. Every run uses 300 gradients, 332 objective evaluations,
zero failed trials, 257-square controls and 1024-square queries. Original
affine parameters and frozen fused match evidence remain unchanged. All saved
boundaries equal the unit-square reference exactly; binary certificates and
positive exact-binary affine determinant signs pass. Independently computed
minimum normalized corner ratios are `0.00557258179` (direct) and
`0.00573967989` (lift), both above `.001`.

All direct-original prepared intensities reconstruct pixel-exactly from the
declared original TIFFs or per-source decoded RGB caches, with the recorded
double dimensions/padding. All masks are exact two-by-two repetitions of the
original512 support. Seven JPEG9 and two kidney JPEG8 source decodes are
explicit in the saved metadata, including the original-image dimensions and
lung upsampling. Across every saved lift image, an independent separable
half-pixel bilinear oracle differs by at most `1.19209290e-7` in intensity;
a separate local Torch replay has the same maximum discrepancy from the remote
saved float32 result. Therefore cross-runtime lift reconstruction is NOT
claimed bitwise exact. No new RGB/grayscale quantization is present.

The first four levels retain exactly the declared settings, input-construction
path, image-loss traces and outside-fraction traces. Their actual GPU numerical
trajectories are NOT bitwise identical to the archive: maximum differences
are `3.2793e-10` in trace total, `1.6268e-10` in static-point loss,
`8.3686e-10` in corner-shape loss and `5.9716e-8` in a geometric margin
diagnostic. Accepted-stage current objectives differ by at most `4.7043e-12`.
There are no saved intermediate maps proving bitwise prefix-map equality.
This does not retract the earlier controlled tiny CPU equality check; it limits
the separate production claim to unchanged construction and schedule plus the
reported numerical comparisons.

All 50 final full1024 objectives were independently recomputed from the SAVED
maps, SAVED prepared images, literal descriptor/interpolation calculations,
SVD-based ARAP and frozen machine points. Maximum objective discrepancy is
`3.5237e-8`, static-point discrepancy `5.56e-17`, and outside-fraction discrepancy
`4.803e-9`. Every arm/case selects stage9, whose full objective is the minimum
over the initial and accepted-stage values. All 4,148 landmark errors were
recomputed from original CSV coordinates and literal P1 interpolation followed
by the original affine; maximum discrepancies are `1.661e-13` canvas pixels
and `2.594e-12` native-moving pixels. Every required label, reported mean/p90,
paired delta, cohort/equal-specimen aggregation and cost summary agrees.

| Cohort | Frozen512 mean | Direct mean | Lift mean | Frozen512 mean pair-p90 | Direct mean pair-p90 | Lift mean pair-p90 |
|---|---:|---:|---:|---:|---:|---:|
| MIIT, 3 directions | 3.53471836 | 3.56196634 | 3.54753996 | 5.85817028 | 5.82631649 | 5.86928026 |
| Lung, 20 directions | 4.47100714 | 4.49272836 | 4.47937148 | 9.34253837 | 9.42861498 | 9.40106849 |
| Histo | 0.84206433 | 0.86431318 | 0.84390350 | 1.58013825 | 1.66040791 | 1.67062241 |
| Kidney | 2.24360647 | 2.27098771 | 2.25145235 | 4.64354946 | 4.68220463 | 4.87430180 |

All units are 512-equivalent canvas pixels. Direct-original worsens all four
specimen mean errors versus BOTH controls. Relative to Frozen512, direct
worsens 19/25 pair means, 17/25 pair-p90s and 9/25 maxima; lift worsens 15/25,
19/25 and 8/25 respectively. Direct versus lift worsens 20/25 means, 14/25
pair-p90s and 11/25 maxima. Every regression's case name and numerical delta,
not just selected examples, is retained in
`outputs/coordinated_instance_registration/terminal_detail_all50_t25/independent_check.json`.
The equal-specimen mean changes are `+0.02464982` (direct versus Frozen512),
`+0.00771775` (lift versus Frozen512) and `+0.01693208` (direct versus lift).
Tails remain mixed: direct improves MIIT mean pair-p90, but the MIIT worst error
remains `52.83107` pixels. This is negative evidence for extra original-image
detail helping this fixed recipe, not an impossibility result for other uses
of the original images.

Complete optimizer calls total `171.158316s` direct and `172.744234s` lift,
versus archived Frozen512 `114.162324s`. Shared per-pair rendering totals
`9.070214s`; current batch wall is `354.444963s`. Current allocated peak reaches
`520508928` bytes. Historical setup/matching and the incompletely timed
one-time mixed-decoder cache creation are excluded, not free. These are
single-run costs rather than synchronized speed repeats. Neither1024 arm
provides a mean-accuracy or cost advantage here, and this reviewed run does
not justify automatic2048 escalation or choosing arms per case using labels.

Reproduction: `check_sources/independent_terminal_postrun_20261002.py` under
the same output parent. It performs no optimization and shares only the
previous independent literal P1, descriptor, prior and original-CSV oracles.

### Data-aware metric: independent formula, operator and timed-wrapper precheck

The approved data-metric card and new core/application/runner/scoring paths pass
the bounded independent precheck. This is a preparation result, not a claim
about production GPU performance, anatomical accuracy or optimizer convergence.
The checker did not author the new metric or production wrapper, did not edit
them, and used no installed skill workflow. The new independent probe is
`outputs/coordinated_instance_registration/check_sources/independent_data_metric_probe_20261002.py`.

The metric is not the Hessian of the original nonsmooth MIND objective. With
fixed-axis residual motion and fixed precomputed moving descriptors, ordinary
bilinear sampling is piecewise affine along the moving axis; absolute residuals
are piecewise linear away from sampling/sign knots. The positive image block
is an IRLS search model. Its exact production AD definition includes the
float64-query to float32-grid cast, zero padding and align_corners=False.
No second affine factor belongs in this image slope, since those features
already occupy the shared affine-aligned frame. OOB still uses the original
frame and thus retains Ae, factor2 and the original fixed-mask denominator.

For an individual point, differentiating
lambda*w*(sqrt(1+|z+t*a|^2)-1) twice gives
lambda*w*(|a|^2/s-(a dot z)^2/s^3), s=sqrt(1+|z|^2).
The two-dimensional identity
|a|^2*|z|^2-(a dot z)^2=(a1*z2-a2*z1)^2 gives exactly the stated
nonnegative stable numerator. The existing normalized fused weights and total
coefficient .2 are retained; no second confidence normalization is introduced.
The Dirichlet five-point block is positive definite on the interior unknowns,
and each remaining weighted sampling block is positive semidefinite, so the
sum is SPD in exact arithmetic. This is a metric claim, not a guarantee about
finite-precision PCG or global optimization.

Independent numerical checks use hand-assembled fine-ac P1 rows and a literal
five-point matrix, not the production interpolation/stiffness implementations
as their own reference. Boundary query rows have zero interior norm where
appropriate. Forward/transpose products, explicit H, symmetry, positive
eigenvalues, boundary-excluded trace gamma and the shifted sine-transform
preconditioner all agree. Tiny H errors are at most `4.45e-16`; minimum
eigenvalues are `7.50195` and `8.52317`; the shifted-inverse comparison error
is at most `1.39e-17`. This verifies the preconditioner for 3K+gamma*I, NOT an
exact inverse of spatially varying H.

A full per-output sampler Jacobian confirms zero cross-query coupling and
reconstructs D_I within `5.69e-14`. A separately written zero-padded bilinear
slope formula at the rounded grid agrees with the actual AD slopes to
`3.74e-7`; its resulting D_I differs by at most `1.36e-7` after scaling by
max(1,|D_I|), reflecting float32 arithmetic rather than a claimed bitwise
classical derivative through quantization. Both coordinate axes, boundary
sampling, far-outside zeros, nonuniform masks and nontrivial affine coupling
are exercised. Point curvature agrees with the alternate closed formula to
`2.85e-14` and with the complete production point Hessian after hand-built L
assembly. The OOB Hessian agrees with the literal masked factor2 construction;
the production zero derivative convention at exactly0and1 is checked.

For the tiny actual metrics, PCG takes four iterations and independently
recomputed relative residuals are `.0486194` and `.0628311`, agreeing with the
reported recursive residuals to floating-point precision. One-step capped
solves remain explicitly nonconverged and yield finite descent directions.
Production reports now label the residual as recursive; these tests do not
prove an arbitrary ill-conditioned finite-precision solve meets its true
residual tolerance. A cap or stop reason is not silently replaced by another
solver or equated with stationarity.

The physical-metric acceptance test correctly rejects a constructed rounded
equal-objective case despite a negative AD slope and rounded-equal Armijo
right-hand side: all13 trials fail, exactly12 halvings occur and the unchanged
last accepted map is returned. Successful tiny steps satisfy actual strict
contracted floors, original-boundary equality and actual strict E decrease.
An incoming map exactly at the original .001 floor is rejected. Conversely,
the original Adam active-contraction case is retained: its original minimum
Q is `.05095000000000027` with contracted slack only `2.71e-16`. Imposing
strict contracted slack on Adam would change the control; the amended card
and metadata correctly distinguish original Adam Q>.001 from the new metric's
Q>Q_floor. This remains an optimizer-package comparison.

Completed work remains visible after an injected fourth-channel VJP failure:
one attempted refresh, one descriptor forward and three completed VJPs.
`trial_evaluations` counts objective-called trials; geometry-only rejected
attempts remain in the trial trace. An injected three-second setup exhausts
the nominal two-second metric clock before any gradient and reports the
overrun. A separate actual CPU-clock smoke with a .02-second nominal budget
finishes its in-progress iteration at `.031449s`, reporting `.011449s` overrun;
this is not a two-second GPU performance measurement.

Production configuration checks require the unchanged257 controls,512 raster,
frozen fused objective and original learning rate. An isolated tiny wrapper
test bypasses that SIZE guard only for its small synthetic case, using injected
clocks; it does not relax production. Its raw eighth accepted map is exactly
the ordinary optimizer's eighth map, distinct from the separately retained
best-prefix role. Both timed arms receive exactly the same saved numerical
start and initial E. Every complete objective call is counted, the final Adam
rate is preserved, and selection is the minimum over common best prefix and
accepted suffix states with valid exports. Source review confirms that shared
prefix cost, suffix loading/features and per-axis setup/gradient/solve/trial
time have explicit scopes; arm order alternates by case parity. All50 attempts
terminate before scoring adapters are written. Early stops remain visible and
are not mislabelled as fixed300-gradient completion.

The independent probe completes its checks in `2.64s` of CPU computation.
A fresh focused run of the core, application and batch tests passes `12/12`
in `7.89s` after setting the normal src/tests import paths. No GPU production
optimization or manual annotation scoring was performed by this precheck.

### Actual omitted-configuration failure and corrected-wrapper check

The first production attempt, `data_metric_all50_t26`, retained 46 successful
outputs and four explicit pre-optimization failures: both Histo and kidney arms
encountered the missing optional `mind_order` attribute in the wrapper. This
was not a metric-solver failure. The preceding synthetic wrapper precheck did
not exercise these actual legacy configurations; it must not be read as complete
production-configuration coverage. The failed batch remains unscored.

The independent postrun checker subsequently recomputed the full objective and
actual saved geometry for all46 successful outputs without opening annotations.
Maximum total-objective discrepancy was `3.27e-8`; both accepted final axes were
reconstructed from each saved fixed-axis map, and actual original/contracted
corner-floor checks passed with the distinct Adam/metric rules above. Both arms
used identical numerical prefix starts. All PCG iteration caps remained marked
nonconverged; a finite descent direction and accepted Armijo step do not certify
linear-solve convergence. Details remain in that batch's
`independent_geometry_check.json`.

After the wrapper adopted the original optional-field defaults, a separate
read-only CPU regression loaded the ACTUAL archived Histo and kidney settings,
their512 images/fused evidence and257 raw prefix maps. For each case, omitted
`mind_order` and explicit `mind_order='transport'` produced bitwise-identical
objectives AND full vertex VJPs, without mutating the input configuration. The
two-case check passed in7.71seconds wall time. It performed no optimization,
GPU work, annotation access or accuracy-based selection. The corrected complete
rerun is kept separately as `data_metric_all50_t26r`.

### Corrected timed-metric all50 independent postrun

The corrected collection completed all50 predictions before scoring. Independent
saved-output review passed for every common240-gradient prefix and both timed
suffix arms. Literal NumPy P1 interpolation, descriptors, physical point loss,
ARAP and shape terms reproduce all final full objectives with maximum discrepancy
`2.74e-8`; point/prior discrepancies are at most `1.12e-16`. All original
affines, float64 identity perimeters and binary certificates agree. Each final
map selected the second axis, permitting independent reconstruction of both
accepted axis-endpoint maps; all100 actual floor checks pass. This does not
claim independent reconstruction of every unsaved trial map.

Adam's minimum actual corner ratio is `.00610995598`, with192–332suffix
gradients per pair and7532in total. The metric arm's minimum is `.00985886070`,
with95–130gradients per pair and2865in total. All100stages stop at their nominal
time budget, with in-progress iteration overruns explicitly retained. The
metric trace records2865refreshes,22920descriptor VJPs,30193PCG iterations and
92backtracks. Of2865PCG solves,2542meet the RECURSIVE residual criterion and323
hit the20-iteration cap; the largest reported recursive relative residual is
`.2483896`. Caps remain nonconverged. Every accepted metric trial satisfies the
recorded strict decrease/Armijo test and the actual reconstructed stage-endpoint
contracted floors; this is neither stationarity nor a proof of true-residual
convergence for all finite-precision solves.

Original annotation CSVs and a separate literal P1 evaluator reproduce all4148
per-label errors across both arms: maximum discrepancy `1.14e-13` canvas pixels
and `1.76e-12` native-moving pixels. Independent reaggregation confirms all
per-case deltas, four specimen aggregates, equal-specimen deltas, objective
records, gradients, elapsed calls and prefix/suffix peak summaries. A reporting
error that summed per-pair memory peaks was corrected to `total=null`; maxima
are not additive memory. Prefix cost is92.5803seconds, with110.6925seconds Adam
suffix calls and110.8216seconds metric suffix calls. Historical preparation is
additional, and the recorded reset scopes do not certify cold-pipeline peaks.

The metric arm worsens8/25mean,7/25p90 and11/25maximum errors against paired
Adam, and worsens18/25mean,18/25p90 and16/25maximum errors against frozen300.
Its equal-specimen mean delta is `+.001962` versus Adam and `+.001518` versus
frozen300; p90 deltas are `-.136207` and `+.020694`, respectively. Thus its
smaller p90 deterioration than timed Adam is not a useful improvement over the
cheaper incumbent. These are four repeatedly viewed development specimens,
not25independent patients or evidence of generalization.

Reproduction sources are `independent_data_metric_postrun_20261002.py` and
`independent_data_metric_comparison_20261002.py` in the ordinary check-sources
folder; their combined result is `data_metric_all50_t26r/independent_check.json`.

## Native STANDARD with supplied SG: bounded independent coordinate check

The column formula `N_m C_m^-1 H C_f N_f^-1` correctly converts the stored
fixed-to-moving canvas affine into DHR's normalized sampling transform. Source
review confirms loader scaling precedes common padding and that installed DHR
uses declared `scale_factor=1/r, recompute_scale_factor=False` during initial
resampling. The normalized pixel-center affine therefore uses declared r and
actual preprocessed width/height, not a rounded padded/preprocessed extent ratio.

`independent_native_shared_probe_20261002.py` independently evaluates scalar
native/canvas/normalized conversions without using the production frame helper.
It tests odd unequal pads, non-square frames, nontrivial affine/offset, r=1 and
r=1.82373046875, unequal loader ratios, and off-canvas coordinates. Maximum
real-arithmetic composition discrepancy is `2.23e-16`. It calls the INSTALLED
DHR transform-to-field helper and compares every tiny native lattice node to
the scalar oracle; maximum float32 field discrepancy is `6.29e-7` normalized.
An actual installed resampler applied to a coordinate ramp verifies its
nonintegral scale-factor center convention independently.

A separate literal border-bilinear sampler checks all512canvas query centers,
with native-lattice inside/outside counts kept separate. The off-lattice field
does not reproduce unrestricted affine extrapolation; large synthetic outside
errors are expected and explicitly reported, not omitted or misclassified as
in-domain conversion errors. The float64 conjugacy, float32 coefficient cast,
and actual sampled field remain separate diagnostics.

The subclass overrides only `run_initial_registration`; it sets the initial
transform/field and passes that field once to the inherited nonrigid path.
The released STANDARD preprocessing/nonrigid parameters remain unchanged apart
from the existing device/save adaptations. Six fresh focused tests pass,
including an injected failed case retained among25terminal attempts before
either scorer manifest is written. Existing scorer path/shape/frame fields are
compatible. No native GPU optimization or anatomical scoring is part of this
precheck; actual-run conversion/cost metadata still require postrun inspection.

## One MA refresh: bounded independent coordinate and incoming-map check

The point conversion `p=f0(s)` is correct: the original-moving residual is
`F_Y(q)-F0(s)=A(f_Y(q)-p)`. The affine offset cancels, and neither an inverse
map nor a second affine application belongs in the point term. The checked
implementation preserves direct fine P1-ac evaluation and tests original-world
eligibility after p, not at s. All-raw coordinate/confidence validation occurs
before finite unit-domain filtering; transformed invalid p fails without clip
or repair. Confidence and SG observations remain unchanged.

`independent_refreshed_probe_20261002.py` confirms exact torch equality between
the identity257 rendering and the old affine-prewarp helper on the same CPU
runtime. A legal nonidentity257fixture is independently evaluated by NumPy P1
interpolation and a literal border-bilinear sampler of the ORIGINAL raster.
Maximum intensity discrepancy is `1.29e-7`. A deliberately affine-prewarped
then deformation-sampled raster differs by `.451734`, so the check distinguishes
the prohibited double image interpolation. The prescribed float64 P1 followed
by float32 affine arithmetic is retained; this is not a claim of exact
real-arithmetic image composition.

For nontrivial affine coupling, unequal confidences, duplicate source points,
and a target rendered ineligible only after its P1 transform, a hand-built
barycentric matrix and analytic robust-loss scatter gradient reproduce the
production point loss to `7.11e-15` and its full vertex VJP to `3.56e-15`.
The physical residual identity is separately verified. Q1 interpolation is
not substituted for the stored diagonal-P1 map.

A scoped smooth-objective fixture places an exact minimum at a legal
NONidentity incoming map. The production optimizer first evaluates that exact
map, performs its tiny four-gradient schedule, and retains the incoming map
bitwise with `selected_stage=None`; identity remains only the reference.
This is a selector regression test, not image-objective or anatomical evidence.
The ordinary image-based tests also confirm explicit identity versus omitted
incoming-map behavior bitwise, a legal displaced start, and invalid affine,
reference, boundary and floor rejection.

Fresh focused refresh/core-batch tests pass8/8in8.31seconds. Source review and
injected extraction failures confirm all25extractions terminate and the model
is released before suffix optimization; frozen control still runs for a failed
refresh, and both scorer manifests are deferred until all50attempts terminate.
Every arm receives the same declared incumbent, uses only its own complete
objective for candidate selection and retains the existing independent
SG/MA normalization. Different refreshed/frozen objective values are not
cross-arm anatomical evidence. No model inference, GPU optimization or manual
annotations were used in these bounded prechecks.

### Refresh comparison helper review before real scoring

The bounded independent review of `refreshed_match_comparison_t27.py` found
no aggregation or cost-scope correction necessary. Eight fresh synthetic tests
pass. The helper checks all25ordered extraction attempts and all50terminal
suffixes before opening new score files, the archived full incumbent paths,
and300new gradients for successful suffixes. Failed extraction/prediction/scoring
remains in the denominator; the affected full-cohort and equal-specimen values
become undefined, rather than silently averaging successful cases. Per-case
mean/p90/maximum deltas, unavailable contrasts and adverse tails are retained.

Only within-functional final-minus-initial objectives are reported. Incumbent
construction and failed calls remain charged; one new matcher setup is separate
from extraction calls, and their nested phase times are not added again. The
required historical initializer/SG work stays explicitly unmeasured where its
compatible cost is unknown. Per-pair and phase memory peaks are not summed.
This source/synthetic check does not validate forthcoming actual refresh outputs.

### Native shared-initializer all25 actual-output independent postrun

`independent_native_shared_postrun_20261002.py` rechecks every successful native
prediction after all25attempts terminated. Original stored affines and exact
accepted512layouts agree with the retained fusion300 inputs. Native image
dimensions, orientation and byte counts, loader ratios, odd floor/ceil padding,
declared initial resample ratio and actual preprocessed extents agree. Every
saved native configuration matches its own-initializer counterpart except the
logging path; preprocessing and all900nonrigid iterations remain unchanged.

An independent scalar native/canvas/normalized-coordinate construction verifies
all25theta64 matrices with maximum coefficient discrepancy `6.67e-16`; stored
theta32 is exactly their once-cast float32 value. All512query-center conjugacy
and inside/outside lattice counts are rechecked. Maximum analytic residual is
`3.22e-13` canvas pixels and the cast-only bound is `1.793e-5` pixels. The actual
initial GPU field was deliberately NOT exported. Its recorded online all-node
and sampling diagnostics were inspected; this postrun does not falsely claim
to re-read an unavailable initial field. The large off-lattice letterbox
sampling discrepancy remains explicitly distinct from in-domain conversion.

Every actual final MHA was independently read, totaling491,480,185file bytes.
A separate float64 border-bilinear oracle and ORIGINAL annotation CSVs recompute
all2074errors with maximum differences `5.832e-5` canvas pixels and `.0008811`
native-moving pixels from the ordinary float32 scoring path. All ID sets,
per-label summaries and frozen/own-initializer comparison rows agree.

Independent complete native-grid corner computation exactly reproduces all
245,494,952diagnostics:2,039,097nonpositive corners, all25cases affected, and
minimum ratio `-1.0842985846`. These genuine local failures are not repaired,
excluded or converted into a global-homeomorphism claim. Independent
reaggregation also verifies cohort values, every adverse-tail list, phase/call
costs, peak summaries and initialization metadata. Shared initialization worsens
17/25mean and16/25p90errors versus native own initialization; versus fusion300
it worsens21/25means and21/25p90s. Thus the supplied-SG initializer does not
explain away the previously observed native nonrigid disadvantage on these
development specimens, while this comparison still does not isolate topology,
objective, evidence resolution or optimization mechanism.

Combined reproduction results are saved in
`native_standard_shared25_t27/independent_check.json`; the checker supports
`--comparison-only` for inexpensive reaggregation after its all-field check.

### One-refresh all50 actual-output independent postrun

The complete `refreshed_match_all50_t27` collection passes independent review
after all25extractions and all50suffix predictions terminate. The saved model
revision and complete inference configuration exactly match the old MA setup.
For all66,159retained refreshed matches, an independently written NumPy P1-ac
evaluator reproduces `p=f0(s)` with ZERO observed discrepancy. All114changes
between eligibility computed at s and at p are correctly handled using p.
Raw retained confidence counts/masses, transformed-target multiplicities,
coverage bins and both independently normalized source weights agree. The
entire SG prefix, including its coordinates, ineligible rows and normalized
confidences, is bitwise unchanged from the old fused table; the declared
effective coefficients remain `.1 P_SG + .1 P_MA`, not an added third term.

Every saved map has the original affine, float64 identity perimeter, valid
binary certificate and actual corner ratio above .001. Minima are
`.00404258727` for frozen suffix and `.00328914881` for refreshed suffix.
Every successful suffix has300NEWgradients,310traced evaluations and332full
objective calls, with the unchanged17/33/65/129/257raw levels and32/64/128/256/512
evidence continuation. Reported stage minima match their31candidate trace
values and final selection matches the minimum of the incoming map and own
accepted full-objective states. Six frozen suffixes retain the original
incumbent BITWISE; the other19and all25refreshed outputs select final stage9.
Unsaved intermediate maps are not falsely described as independently reread.

Literal NumPy descriptor, P1 point, ARAP, shape and OOB calculations verify
each arm's incoming and final complete OWN functional. Maximum total/image
discrepancy is `3.542e-8`; point/prior errors are at most `1.12e-16`. Both arms
start from the exact same full saved Y0 and its unchanged dense/prior terms;
only refreshed point evidence changes the refreshed initial functional.
Original CSVs and the separate P1 evaluator reproduce all4148per-label errors
within `1.16e-13` canvas pixels and `1.63e-12` native-moving pixels. Original,
frozen-suffix and refreshed-suffix label ID sets are identical.

The independently reaggregated per-case, four-specimen, equal-specimen,
adverse-tail, cost and timing records all agree. Refresh versus matched frozen
suffix worsens16/25mean,16/25p90and13/25maximum errors. Equal-specimen deltas
are `+.0110512` mean and `-.0764623` p90; versus the cheaper fusion300 they are
`+.0154672` mean and `+.0361343` p90. The p90 gain against an extra-iteration
control is therefore not an improvement over the retained lower-cost recipe.
No per-case arm selection, anatomical filtering or fold repair occurred.

Both historical incumbent construction and new suffix calls remain charged.
One matcher setup is counted separately from25extraction calls; extraction
phase subtimes are nested, not added again. Actual batch time bounds all
recorded calls, memory peaks remain nonadditive, and historical unknown costs
remain null. Reproduction scripts are `independent_refreshed_postrun_20261002.py`
and `independent_refreshed_comparison_20261002.py`; combined results are in
`refreshed_match_all50_t27/independent_check.json`. These are instance-registration
experiments on repeatedly viewed development specimens, not network training
or held-out neural-layer validation.

### Incumbent-distortion-budget card: independent mathematical check

Read the complete `OPTIMIZER_REDESIGN.md` card beginning with "Conditional next
mechanism: incumbent-distortion-budget continuation", including the revised
`alpha=.99*min(1,alpha_max)` rule. The scalar-fiber majorizer, full-node RMS
metric, common DST diagonalization, multiplier monotonicity and finite-bracket
existence exception are correct under the stated batch-one, aligned-grid,
unit-direction and positive-P1 hypotheses. The cap-active `h=0,r=0` case is
indeed the singleton feasible set; it must not enter an infinite multiplier
search. The always-contracted proposal provides exact-real quadratic slack,
but does not replace the strict rounded actual ARAP-budget check.

Independent probe `independent_distortion_budget_probe_20261002.py` assembles
full-node tensor-hat prolongation and the literal five-point interior matrix
without the production prolongation or spectral constructor. Five grid pairs
`(5,3),(7,4),(9,3),(9,5),(5,5)` cover refinement1/2/4 and odd/even interior
sizes. Stiffness/spectral errors are at most `1.78e-15`; mass/spectral errors
are at most `2.78e-17`, using `fine_side**2`, including the zero perimeter.
Actual prolongation agrees with the hand-built full-node matrix.

A separate literal source-triangle-inverse evaluator and NumPy SVD proper
rotations give the frozen ARAP Hessian equal to UNWEIGHTED K within `2.78e-16`.
Value/gradient touching holds, with gradient discrepancy at most `1.50e-16`.
All40 nonidentity-map perturbations satisfy the stated nonlinear majorizer;
the literal frozen quadratic identity differs by at most `3.77e-18`. No
additional factor3,2,cell-area or inverse-mass belongs in that Hessian.

Dense multiplier solves independently compared with SciPy SLSQP constrained
solutions pass inactive, zero-slack active, positive-slack active, zero-r and
near-Pareto cases; model minima differ by at most `3.01e-13`. Inactive physical
RMS is `.004` to rounding. The singleton case returns zero. These checks verify
the formula, not the forthcoming production solver or anatomical usefulness;
production core/wrapper and actual-run checks remain separate.

The same independent probe additionally verifies the actual wrapper's direct D
value and complete vertex VJP against separately summed non-ARAP Evidence parts
(both bitwise equal); `D+3R` differs from the original total only by `5.55e-17`
operation-order rounding. A clearly synthetic, scoped solver fixture verifies
one fixed runtime cap across stages with decreasing actual R, continued
execution after legal early stops, actual4 rather than nominal120 gradients,
unchanged physical-rate schedule, full-D best-endpoint retention, ordinary
callback-count aggregation, and immediate schedule termination with an explicit
failure label after a reported numerical failure. It is a wrapper semantics
test, not experimental accuracy or production-core evidence. The inherited
preprocessing metadata incorrectly described only one original512 affine
preparation; the author was asked to clarify that this pyramid independently
prepares each area-reduced raw raster once. The actual computations already do
that and do not double-warp moving data.

The production core is now independently checked, not just its card. Nine
active/inactive/zero-r/near-Pareto directions agree with the separate dense
solver to `4.12e-18`. An actual six-gradient legal-map run passes literal NumPy
corner floors, the original-anchor reconstruction, strict computed data/Armijo
decrease and actual fixed R cap. Deliberately finite no-descent callbacks cause
13 rejected trials and12 backtracks followed by a valid retained-map stop;
a nonfinite trial causes one immediate numerical failure and no backtracking.
A returned ARAP value only ONE binary64 ulp above B is rejected on all13
attempts: there is no hidden cap tolerance. Zero-budget identity stops after
one real outer iteration rather than reporting30 completed gradients.

Review caught a diagnostic mismatch in exhausted dual bracketing: the record's
lower multiplier was the final positive probe but its residual was still the
initial mu0 residual. The author changed it to the final probe residual; the
accepted trajectory is unaffected. Both this correction and the pyramid
metadata correction were reread in the saved files. A fresh combined run of
core, wrapper and batch tests passes35 tests. The first test invocation lacked
the local `src` import path; rerunning with that path fixed test collection,
without modifying source or dependencies.

The batch and both scoring adapters were read through their ordinary map
validation paths. They require the complete25-attempt denominator before
scoring, preserve failed attempts, compare actual cap-mode accounting with the
saved report, and no longer require an artificial300 completed gradients for
the capped arm. The archived frozen-table control still requires its actual300
gradients and the exact same incumbent/configuration. These are implementation
prechecks only; no new real-case accuracy claim is made before the full run.

The downloaded actual257-grid MIIT2-to-3 cap smoke subsequently passes the new
label-free `independent_distortion_budget_postrun_20261002.py --smoke` audit.
Literal original/saved endpoint ARAP agrees within `4.34e-19`; complete D/E
within `1.38e-8` from independently implemented float32 image sampling. Saved
binary geometry, exact identity perimeter/original affine and literal minimum
corner ratio `.27053576924` pass. The296 recorded outer steps,294 accepted
steps,560 candidate callbacks,266 finite rejections and264 backtracks reconcile
exactly, including two valid line-search-exhausted stages. All296 directions
have active dual caps (maximum recorded multiplier `3.53899608`). The saved
endpoint R is `.003418679633662371`, strictly below its fixed runtime budget
`.0034189056020324637`; D changes `.23861893719244703` to `.23777583429492904`.
Only incoming/final maps were independently reevaluated. Intermediate trial
maps and spectral vectors were not exported, so their audit is explicitly
record arithmetic/acceptance consistency, not an independent map reevaluation.
No evaluation labels were read for this smoke check.

### Actual all25 distortion-budget outputs: independent post-run PASS

After every attempt terminated and ordinary scoring was authorized, all25
saved float64 maps pass independent original-affine, exact identity-perimeter,
binary-certificate and literal-corner checks. Minimum saved normalized corner
determinant is `.01549217378489276`. Both incoming and final full R/D/E are
recomputed with the independent NumPy triangle/sampler/descriptor formulas;
maximum R discrepancy is `3.47e-18`, D/E `3.459e-8`. Every output has strictly
lower independently recomputed R than its incumbent and satisfies its exact
reported runtime cap; no budget inflation or tolerance acceptance occurred.
All25 select endpoint9 by their own full D. D improvements range `.000622414`
to `.001723179`; R decreases range `1.846e-7` to `5.694e-5`.

All2074 original-CSV errors are recomputed using the separate literal P1
evaluator: maximum discrepancy `1.118e-13` canvas pixels and `1.475e-12` native
moving pixels. Full25 denominators and the exact same anatomical ID sets are
preserved, with no failures. Recorded intermediate checks have the limited
scope stated above. They reconcile7307 outer gradients,7307 data VJPs and7307
ARAP VJPs;7272 accepted steps;13581 candidate callbacks;6309 finite rejections;
6274 backtracks;250448 dual evaluations. Including300 wrapper callbacks gives
21188 objective evaluations. There are215 budget-ended stages and35 legitimate
line-search-exhausted stages, with zero numerical failures. Actual per-case
gradients are278--300; unused work is not reported as completed. All7307 dual
proposals have active caps; this does not imply constrained stationarity.

Independent reaggregation verifies every case/cohort/regression/objective/cost
entry of `comparison.json`. Versus original fusion300, equal-specimen mean/p90
deltas are `-.0147991/-.000397950` canvas pixels, with11/25 means,10/25 p90s and
13/25 maxima worse. The small aggregate mean gain is dominated by kidney
(`-.0642202`), whose maximum worsens `.435152`; MIIT and lung aggregate means
and p90s both worsen. Against the same-incumbent frozen-table300 suffix, deltas
are `-.0192151/-.112995`, with9/25 means,11/25 p90s and13/25 maxima worse.
Suffix calls total225.223s; adding the required historical incumbent calls gives
339.386s, versus224.106s for incumbent plus frozen suffix. Maximum allocated
suffix memory is184566272 bytes; peaks remain nonadditive and old preparation
costs are not silently counted as free. This is a modest development tradeoff,
not a held-out, neural-training or general anatomical improvement claim.

Reproduction: `independent_distortion_budget_postrun_20261002.py --scores` and
`independent_distortion_budget_comparison_20261002.py`. Combined output is
`distortion_budget_all25_t28/independent_check.json`.

### Full-confidence posthoc diagnostic: independent precheck and actual PASS

Read the complete bounded card and released local coarse/fine matching source.
The coarse centers are literal pixel indices8i, flattening is x-fast, and the
saved `conf_matrix` is not modified by threshold/border removal: those operations
modify a separate boolean mask. Fine matching stores `conf_matrix_f` separately.
Literal independent two-axis tests verify source-row/target interpolation,
closed native-center support, no clamping, strict all4096-cell ranks and ties.
Four-footprint maxima intentionally include zero-weight neighboring corners
at exact centers; they are a coarse-localization sensitivity, not bilinear
equivalence. Five focused author tests also pass.

Precheck found a blocking scorer error: a batch-one saved map was passed to an
unbatched P1 evaluator. The author added validated unbatching before scoring.
It also corrected cohort aggregation so a failed or zero-supported case cannot
silently reduce expected denominators or cause `mean(None)`; case/ID denominators
and computed/support counts are now distinct, with unavailable cohort means null.
Both corrections were reread before validating the actual output.

`independent_confidence_probe_20261002.py --actual` reads all25 saved4096-square
float32 C matrices (1,677,724,800 file bytes), verifies finite[0,1] entries and
bitwise equality of every newly extracted original MA point/confidence/affine
array. It then reloads original CSVs and uses a separate vectorized P1 formula,
explicit2-by-2 affine inverse and literal coarse-grid sampler. All2074 IDs,
coordinates, current TREs, both confidence scores, strict ranks/orderings,
support flags and case/cohort/full aggregates agree. Maximum coordinate,
confidence-score and TRE differences are `2.22e-16`, `1.166e-14` and `3.60e-14`.

Exactly2054 IDs are supported;20 unsupported IDs remain explicitly present.
Coordinate-specific unsupported counts are8 source,15 true-target and8
prediction (overlapping sets). Bilinear scores prefer truth707 times and the
current prediction1347 times, with no ties. Footprint maxima prefer truth74,
prediction333 and tie1647 times. Every cohort's mean rank advantage is
nonpositive under both definitions; Histo's footprint value is exactly zero.
The1682 ordering disagreements include newly tied footprint scores and must
not be described as1682 preference reversals. This particular posthoc probe
does not supply robust positive evidence for an unused anatomically preferable
alternative; it does not prove that all possible assignments or features fail.
Saved result: `full_confidence_all25_t28/independent_check.json`. No network
training, optimizer run, map edit or new GPU work occurred during this review.
Original full C was not historically saved, so matching point tables do not
prove historical bitwise equality of that formerly unavailable full matrix.

### Histo initializer provenance: separate numerical reconstruction

`independent_histo_initializer_provenance_20261002.py` confirms the actual Histo
common A,b are the positive affine factored from the historical DHR INITIAL-ONLY
field, then conjugated into the512 canvas. That saved CPU field reproduces all
66049 archived raw sampled targets bitwise. A separate literal float64
center-frame/border-bilinear sampler differs by at most `1.468e-7` from the
historical float32 path. Independent NumPy all-vertex least squares reproduces
the saved float32 native A,b bitwise; manual row/column scale-and-pad conjugation
reproduces the current canvas A,b and fusion output's stored affine bitwise.
The original configuration has CPU, source/target loader ratios `.1`, initial
resolution768 and multi-feature SuperPoint/SuperGlue initialization; nonrigid
registration is disabled and runtime nonrigid time is zero. This is numerical
provenance, not filename inference. Calling all25 initializers a uniform direct
SG similarity is incorrect; the valid shared-map statement is unchanged exact
image-only positive A,b within each comparison. No benchmark map changed.
