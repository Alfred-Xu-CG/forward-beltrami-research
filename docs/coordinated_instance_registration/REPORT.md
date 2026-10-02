# Coordinated registration: self-contained living research report

This is a working synthesis, NOT the final review. The authorized window
started2026-10-01 11:26:23UTC. On 2026-10-02 the user resumed the paused goal
and extended the deadline by five hours to 2026-10-02 16:26:23UTC. Historical
sections retain their dated scopes; newer restart measurements are added below.
The detailed derivations and experiment definitions are in FORMULATION.md;
chronology and negative interventions are in PROGRESS.md. This report introduces
the main objects without requiring knowledge of those earlier discussions.

### Complete native baseline and one refreshed-observation experiment

The full native STANDARD DeeperHistReg baseline has now also been replayed
with our same image-derived SG similarity, changing ONLY its initializer.
It keeps native preprocessing, the4096 registration-size setting, NCC,
diffusion regularization and its released eight-level900-iteration schedule.
This is a modified-initialization counterfactual, not the untouched published
pipeline and not an isolated topology-layer ablation.

| Development specimen | Native own initializer mean / p90 | Native shared initializer mean / p90 | Retained fusion300 mean / p90 |
|---|---:|---:|---:|
| MIIT /3 | 3.672192 /6.432444 | 3.712720 /6.260251 | 3.534718 /5.858170 |
| Lung /20 | 6.046615 /13.914145 | 6.240578 /14.626331 | 4.471007 /9.342538 |
| HistoReg /1 | .712328 /1.618927 | .712976 /1.523942 | .842064 /1.580138 |
| Kidney /1 | 1.908163 /3.178382 | 2.464310 /5.245805 | 2.243606 /4.643549 |

All25 directions and2,074 same annotation IDs are retained. Independent
re-reading of every final field and CSV verifies the result. Shared initialization
worsens17/25 means versus native own initialization. The two native pipelines
must both remain visible: our method does not universally outperform native
DeeperHistReg, particularly on HistoReg and the own-init kidney comparison.
The shared-native calls total190.383s, versus777.792s own-init, primarily because
the latter includes588.564s of initialization. The historical SG preparation
is not included in shared-native timing, so this is not a full-pipeline speedup.
Maximum allocated peaks are5.68GB shared-native and7.71GB own-init, versus
the retained method's recorded .214GB optimizer scope, which excludes earlier
feature-setup transients. All25 native fields have some nonpositive local
corners; these are actual saved-field diagnostics, not a global certificate.
The current safe257 P1 outputs have a separate boundary-and-corner certificate.
See BASELINE_PROTOCOL_REVIEW and native_standard_shared25_t27 for details.

The ONE new observation experiment renders the original moving image through
the retained absolute map F0=A f0+b and applies the SAME frozen MA matcher.
If it matches fixed coordinate q to warped-image coordinate s, the new aligned
target is p=f0(s), evaluated in the exact declared fine P1 interpolation.
The residual for a candidate is A(fY(q)-p); no inverse, double affine, second
image interpolation or resampled geometric composition is introduced.
Only the MA table changes. SG, both .1 coefficients and all dense/prior terms
stay fixed. Both comparison arms start the identical retained fusion300 map
and receive300 NEW gradients, so extra optimization has an explicit control.

| Development specimen | Frozen-table suffix mean / p90 | Refreshed-table suffix mean / p90 |
|---|---:|---:|
| MIIT /3 | 3.540120 /5.893797 | 3.538318 /5.872131 |
| Lung /20 | 4.482110 /9.329755 | 4.486130 /9.454880 |
| HistoReg /1 | .849139 /1.587516 | .850989 /1.599316 |
| Kidney /1 | 2.237691 /5.063714 | 2.277828 /4.642606 |

All25 extractions and50 suffixes complete in229.110s without failure. Refresh
versus frozen continuation worsens16/25 means,16/25 p90s and13/25 maxima;
versus the cheaper original fusion300, allfour specimen means worsen. This
does not justify adopting the refresh globally. No repeated rematching or
per-case winner selection follows. Matching the incumbent can self-confirm
its errors; low refreshed loss or many matches are not anatomical accuracy.

Frozen/refreshed suffix calls average4.398/4.433s, ADDITIONAL to the historical
4.566s incumbent. Refresh extraction averages.234s including table/fusion work,
plus one1.086s model setup per25-case batch. Extraction peak is1.019GB;
suffix recorded peaks are at most.215GB, not additive to extraction peaks.
Historical initialization and SG/old-MA preparation remain required dependencies.
Full postrun independent verification passes: all50 exports, all4,148 original
CSV errors, every own objective/selector and all66,159 refreshed targets are
recomputed. The latter agree exactly with independent NumPy P1 interpolation;
114 targets change world eligibility after p=f0(s), and are handled correctly.
The SG prefix remains bitwise unchanged and the effective coefficients remain
.1+.1. Maximum objective/error discrepancies are3.55e-8 and1.16e-13canvas
pixels. Minimum saved corner ratios are.00404259 frozen/.00328915 refreshed.
Six frozen-control maps retain the incumbent exactly; all25 refresh maps select
their final stage. None of these correctness findings changes the negative
anatomical conclusion. The integrated44 focused tests also pass.
The experiment's complete equations are in REFRESHED_MATCH_FORMULATION;
all scores, failures, costs and maps are in refreshed_match_all50_t27.

### Same-prefix timed optimizer: lower loss is not a registration breakthrough

This experiment keeps the frozen512 SG+MA objective, affine, images, support,
257-square P1 control grid and all point units. Compute one common240-gradient
prefix through coefficient levels17/33/65/129. From its SAME saved last map,
compare the final257x/y stages with two seconds peraxis: original analytic-latent
Adam versus a physical scalar update using a stiffness-plus-data sensitivity
matrix and PCG/Armijo. The detailed operator, gradient, stopping and feasibility
rules are defined in DATA_AWARE_METRIC_FORMULATION. This compares optimizer
packages, not just their matrices. Both retain the common best prefix as an
own-objective candidate and select only by full512 loss, never by landmarks.

| Development specimen | Original fusion300 mean / p90 | Timed Adam mean / p90 | Timed data metric mean / p90 |
|---|---:|---:|---:|
| MIIT /3 directions | 3.534718 /5.858170 | 3.544819 /5.872471 | 3.541050 /5.864543 |
| Lung /20 directions | 4.471007 /9.342538 | 4.490432 /9.417400 | 4.479757 /9.363577 |
| HistoReg /1 | .842064 /1.580138 | .828476 /1.591787 | .839882 /1.594773 |
| Kidney /1 | 2.243606 /4.643549 | 2.225893 /5.170345 | 2.236778 /4.684280 |

Units are512-equivalent moving-canvas pixels, with means/p90 first computed
within each direction. The data metric improves MIIT/lung means versus timed
Adam but worsens HistoReg/kidney means; versus the cheaper original300 recipe,
its MIIT/lung means and allfour p90s worsen. Its pair mean/p90/max regressions
are8/7/11 of25 versus timedAdam and18/18/16 versus original300. These are
mixed small changes, not evidence for replacing the retained global recipe.

Timed Adam lowers the unchanged full loss versus original300 on25/25pairs
(mean change-.00170144). The metric does so on24/25 (mean-.00107103). Both
reduce loss from their common suffix start, yet anatomical gains do not track
these decreases. This strengthens the observed loss--anatomy mismatch without
proving either optimizer converged, the loss has an incorrect global minimizer,
or all other optimizers are ineffective. A new solver has not solved the problem.

The common prefix costs3.703s perpair on average; the complete timed suffixes
cost4.428s Adam and4.433s metric. Required prefix+suffix is therefore about8.13s,
not4.43s, before historical matching/initialization costs. The old300 optimizer
averaged4.566s in its archived run. New suffix peaks are153--155MB/158--159MB;
the maximum RECORDED prefix/suffix peaks are176--181MB. Prefix/old300 counters
reset after feature construction and omit earlier setup transients; these are
not verified complete cold-pipeline memory peaks and must not be added together.

All corrected50attempts finish in315.173s; all100axis stages reach their nominal
time budget with recorded overruns. Adam uses192--332suffix gradients perpair;
the metric95--130. Across25metric calls:2,865accepted steps,22,920descriptorVJPs,
30,193PCGiterations and92backtracks. Of2,865linear solves,323hit the20-step cap
and remain explicitly NONCONVERGED (maximum reported recursive residual.24839);
finite descent and actual feasibility/decrease still govern their accepted steps.
No solve-cap label is changed to convergence. This is instance optimization,
not a trained neural encoder or a backward pass through the whole trajectory.

Independent reconstruction checks all50exports,100axis endpoint floors, full
objectives and4,148original-CSV errors. Maximum discrepancies are2.74e-8 in
the full objective and1.14e-13canvas pixels. Minimum saved corner ratios are
.00610996 Adam/.00985886 metric, both>.001. The first attempted batch is retained
separately: two older optional-field omissions caused FOUR wrapper failures,
not numerical divergences. Restoring the original `mind_order=transport`
default was independently tested before the complete corrected rerun; no
images, configurations, points, loss weights or labels were altered.

Primary corrected outputs: `data_metric_all50_t26r`; failed first attempt:
`data_metric_all50_t26`. Next: one deformation-conditioned refresh of the SAME
MA model's observations, with a same-incumbent/same300-suffix frozen-table
control. REFRESHED_MATCH_FORMULATION states its exact coordinate conversion,
self-confirmation risk and cost before accuracy testing. A supporting native
STANDARD DHR replay with the same frozen initializer separately tests the
initialization confound; it is not relabelled as the unmodified released pipeline.

### Original-image terminal detail: tested, no mean-accuracy benefit

With the SAME257-square control map, frozen affine, fused points and300gradient
schedule, change only the last image level/full-objective selector to1024.
Compare direct rendering from the original files against a deliberately enlarged
float32 version of the old512 input. Both use the same replicated original
support mask; the firstfour image levels remain constructed from the old512
images. Machine points remain in512/8pixel/robust-scale units. This isolates
available source information from merely adding raster samples, although
resampling/quantization differences prevent a pure frequency-only interpretation.

| Previously viewed specimen | Original512 mean / p90 | Direct-original1024 mean / p90 | Enlarged512-information mean / p90 |
|---|---:|---:|---:|
| MIIT /3 directions | 3.534718 /5.858170 | 3.561966 /5.826316 | 3.547540 /5.869280 |
| Lung /20 directions | 4.471007 /9.342538 | 4.492728 /9.428615 | 4.479371 /9.401068 |
| HistoReg /1 | .842064 /1.580138 | .864313 /1.660408 | .843904 /1.670622 |
| Kidney /1 | 2.243606 /4.643549 | 2.270988 /4.682205 | 2.251452 /4.874302 |

All error units are STILL512-equivalent moving-canvas pixels. Both1024arms
worsen every specimen mean versus512; direct-original also worsens every mean
versus the enlarged control. Direct worsens19/25pair means versus512 and20/25
versus enlargement. Some tails improve, notably MIIT p90 and kidney maximum,
but this is not an average accuracy advance. Do not escalate automatically to
2048, select different resolutions by specimen, or call this a map-capacity
failure. It is negative evidence for this particular use of additional detail.

All50attempts complete before scoring with300gradients/332objective calls and
zero failed trials. Independent full-objective reconstruction agrees within
3.53e-8; all4148label errors agree within1.67e-13canvas pixels. Minimum actual
corner ratios are.00557258direct and.00573968enlarged, both>.001. Every output
selects the final stage9 using its own declared full objective, not labels.

Per-pair calls average6.846/6.910s (direct/enlarged) versus archived5124.566s;
allocated peaks are.517--.521GB versus.210--.214GB. Shared rendering takes9.070s,
whole50batch354.445s. Original matching/initialization and full one-time decode
cache setup are excluded, not free. This is not a controlled repeated speedup
measurement. The representation is257controls, NEVER1024controls.

A consequential preprocessing issue was found and resolved BEFORE production:
old JPEG canvases used two decoder environments. Seven lung/HistoReg sources
require Pillow10.3/JPEG9, while the two kidney sources require12.3/JPEG8. Lossless
full-resolution RGB caches now preserve these decodes, and all accepted512RGB
canvases reconstruct EXACTLY on the remote host. No source, accepted image or
layout was rewritten. Cache creation was interrupted by two informative failed
attempts; a full fresh-from-empty setup time is unmeasured. The recorded6.78s
and7.58s partial completion/verification timings must NOT be called its total.

Numerical scope also matters: lower-level IMAGE terms and outside fractions
match the old traces exactly, but geometry/total traces differ at roundoff
(max total3.28e-10, normalized margin5.97e-8). Across runtimes, enlarged float32
rasters differ from an independent formula by at most1.19e-7 absolute. These
are disclosed numerical differences, not bitwise production-prefix equivalence.
Details, every regression and reproducible checks are in
`terminal_detail_all50_t25`, TERMINAL_DETAIL_FORMULATION and the independent report.

The next controlled question concerns a data-Jacobian-aware SEARCH direction
for the unchanged512objective. DATA_AWARE_METRIC_FORMULATION records the
mathematics and matched final-stage wall budgets before implementation; no
accuracy benefit from that new optimizer is yet claimed.

### Joint positive global pose: lung benefit, not a uniform replacement

The next controlled experiment changes the map from F(x)=A f_Y(x)+b to
F(x)=A G(f_Y(x))+b. Here A,b are the same saved image-only initializer,
f_Y is the same boundary-fixed257-square P1 map, and
G(z)=o+t+R(theta) exp(S)(z-o), o=(.5,.5), with symmetric2-by2 S.
Its six learned parameters permit translation, rotation, scale, anisotropy
and shear while keeping its determinant positive in exact arithmetic. The
actual saved affine and residual are checked numerically and with binary
sign certificates. Both residual and complete-map corner floors remain>.001.
JOINT_POSE_FORMULATION gives every parameter, derivative, floor and output
convention. This changes the allowed boundary polygon and regularization
model; it is not just a faster solver for the identical old problem.

Use the same fused point evidence throughout. Joint300 alternates50global
pose gradients with250local residual gradients. Frozen250 is a fresh control;
Frozen300 is the previous retained recipe. Neither control receives labels.

| Previously viewed specimen | Frozen300 mean / p90 | Frozen250 mean / p90 | Joint300 mean / p90 |
|---|---:|---:|---:|
| MIIT /3 directions | 3.534718 /5.858170 | 3.536512 /5.902573 | 3.565108 /5.952589 |
| Lung /20 directions | 4.471007 /9.342538 | 4.465445 /9.327422 | 4.299372 /9.022772 |
| HistoReg /1 | .842064 /1.580138 | .867413 /1.711092 | .864019 /1.598028 |
| Kidney /1 | 2.243606 /4.643549 | 2.270662 /5.098371 | 2.237284 /4.966499 |

Units remain512moving-canvas pixels; p90 is computed per pair then averaged.
Lung improves against both controls (mean about3.84% versus Frozen300).
Compared with Frozen300, however, two specimen means and three p90s worsen.
Kidney's worst error rises from8.97058to11.49661pixels despite its small mean
gain. Therefore keep Frozen300 globally as the current development default;
do not choose pose mode separately per specimen from these labels. The lung
result is useful evidence about the boundary/pose restriction, not a uniform
registration breakthrough or held-out patient-level result.

All50fresh attempts finish before scoring, with no failed trials. Independent
literal interpolation of all4148errors agrees to1.14e-13canvas pixels. Joint
outputs have minimum actual residual ratio.00395089 and complete normalized
ratio.00359920. Independently rebuilt final image/point/prior objectives agree
to2.50e-8; paired(G,Y) selection, masks, point eligibility and support diagnostics
also agree. The floating-point matrix exponential is not exact: its affine
coefficients differ from an independent exponential by at most8.79e-11.
Topology checks use the actual stored coefficients, not the ideal formula.

Fresh50wall time is285.16s. Frozen250 calls average3.730s; Joint300 averages
7.653s (7.331--7.890s), allocated peaks.244--.249GB. Archived Frozen300 averages
4.566s, peaks.210--.214GB. These exclude already computed initialization and
matching and are single batches, not a repeated controlled speedup estimate.
There is no uniform accuracy/cost advantage. All maps, scores and contrasts
are in `joint_pose_all50_t24`.

One reporting erratum is preserved: production revision42a599c inherited a
stale `original_moving_features_no_affine_prewarp=True` flag in joint reports.
Actual descriptors are computed AFTER the differentiable A G prewarp of the
original moving raster, independently verified from the saved results. Old
reports remain unchanged; future metadata is corrected. This is not a changed
map or a reason to rerun/select results.

### Preserving sparse anchors plus pretrained points gives a modest gain

The follow-up keeps the original SuperPoint/SuperGlue point term rather than
replacing it. Write P_SG and P_MA for the independently confidence-normalized
robust point losses. Compare the original .1P_SG, the matched-total-strength
control .2P_SG, and the complement .1P_SG+.1P_MA. No point is rematched,
deduplicated, manually filtered or used to refit the initializer. The same
images, masks, priors,257-square P1-ac decoder and300-gradient optimizer remain.
Independent unequal-mass loss/vertex-gradient tests check the exact algebra
of concatenating the two frozen tables; more points do not implicitly set
their relative strength.

| Previously viewed specimen | .1 SG: mean / p90 | .2 SG: mean / p90 | .1 SG + .1 MA: mean / p90 |
|---|---:|---:|---:|
| MIIT /3 directions | 3.548752 /5.907957 | 3.564142 /5.953456 | 3.534718 /5.858170 |
| Lung /20 directions | 4.568541 /9.593679 | 4.517025 /9.670540 | 4.471007 /9.342538 |
| HistoReg /1 | .851903 /1.586005 | .946956 /1.714475 | .842064 /1.580138 |
| Kidney /1 | 2.364866 /4.756907 | 2.382224 /4.702099 | 2.243606 /4.643549 |

All units are512moving-canvas pixels. The complement improves all four
specimen means and mean-pair-p90s against both controls. Mean reductions from
.1SG are about.4%,2.1%,1.2%,5.1%, respectively. Merely doubling SG does not
produce this pattern. This supports useful complementarity for THIS tested
recipe, not a generally better matcher or a breakthrough in the geometry.
We retain the complement as ONE global exploratory recipe for subsequent
work; we do not pick the best matcher or weights separately for each case.

The result is not uniform at the direction/landmark level: five of25pair means
worsen versus .1SG, including MIIT2-to3 by.00149pixels and cd31-to-ki67 by
about.44pixels. MIIT2-to3 p90 also worsens by.01483pixels. Kidney's worst error
increases by.03730pixels. The complement does not yet beat full native DHR on
HistoReg or kidney. These are repeatedly viewed DEVELOPMENT specimens, not
independent validation or a patient-level statistical result.

All50attempts finish before scoring, with300gradients/332objective calls and
no failures each. Independent literal P1/CSV evaluation reproduces all4148
label errors to1.14e-13canvas pixels. Original affines, identity residual
boundaries and declared output connectivity are retained. Minimum actual
corner ratios are.00532227(complement) and.00258231(.2SG), both above.001.

Current two-arm wall time is222.47s, including1.096s of frozen-table composition.
Complement optimization calls take4.06--6.47s, allocated peaks.210--.214GB;
.2SG calls take4.00--4.83s, peaks.209--.213GB. These exclude the previously
computed initialization and matching. A deployment comparison MUST add the
already measured MA extraction3.237s for25pairs and cold setup1.053s for its
batch; its extraction peak is about1.03GB, not the optimizer's.21GB. We do not
call cached evidence free or infer a speedup from these single timing batches.
Results and all paired contrasts are in `match_fusion_all50_t23`.

### Cross-modality pretrained point evidence: kidney gain, no general gain

We replaced the frozen SuperPoint/SuperGlue correspondence table with the
released MatchAnything ELoFTR model. It receives the existing512-square image
canvases in ordinary grayscale, with the moving image prewarped once by the
same saved affine. Its pixel-index correspondences become unit-square point
observations via `(pixel+.5)/512`. The original shared-gray MIND term, point
weight.1, robust scale8pixels, priors, initialization,257-square P1-ac map,
300-gradient optimizer and full-objective output selection stay unchanged.
The matcher is frozen: this experiment does not train a new neural registration
network or reproduce MatchAnything's published affine/B-spline pipeline.

| Previously viewed specimen | Original SG points: mean / p90 | MatchAnything points: mean / p90 |
|---|---:|---:|
| MIIT /3 directions | 3.548752 /5.907957 | 3.616405 /6.140994 |
| Lung /20 directions | 4.568541 /9.593679 | 4.839762 /9.807444 |
| HistoReg /1 | .851903 /1.586005 | .873853 /1.677123 |
| Kidney /1 | 2.364866 /4.756907 | 2.236834 /4.710601 |

Units are512 moving-canvas pixels; p90 is first computed within each direction
and then averaged. Kidney's mean improves about5.4%, but the other three
specimen means worsen. This does not establish a generally better pipeline.
More machine correspondences are not necessarily better anatomical evidence.
There are1929--3158positive eligible matches per pair, all16source bins occupied;
these are correlated observations, not thousands of independent landmarks.
No manual labels, dense competitor fields, new RANSAC filter or per-case
parameter selection enters prediction. Independent literal interpolation/CSV
recalculation reproduces all2074landmark errors to1.28e-13canvas pixels.

All25attempts complete, each with300gradients/332objective calls and no failed
trials. Every actual exported residual retains its identity boundary; the
minimum four-corner determinant ratio is.34688275, above.001. The independent
checker caught and resolved a source/documentation mismatch BEFORE production:
the official active coarse branch keeps thresholded candidates, not mutual-
nearest pairs, despite an unused configuration flag. We preserve that release
behavior and report it, rather than silently modifying the model.

On the idle RTX A6000, one cold setup takes1.053s. All25extraction calls total
3.237s (individual.077--1.062s, including the first cold forward); optimizer
calls take4.15--4.89s each. Combined pair calls take4.23--5.85s, excluding that
one-time setup and the previously frozen affine initialization. Whole serial
batch wall time is115.67s. Allocated GPU peaks are.971--1.030GB(decimal) during
extraction and.210--.216GB during optimization; the model is freed between
phases, so these peaks are not additive. These are actual call measurements,
not warmed-throughput or statistically repeated speedup claims. Source,
configuration, point tables, maps and paired scores are in
`matchanything_all25_t22`; checkpoint/source caches are on D and the research
host. The next question concerns evidence weighting/spatial support and its
conflict with the original objective, not another unmotivated model swap.

### Fixed shared-stain evidence: stable execution, no useful accuracy gain

The single H-proxy experiment applies fixed H/E/DAB color deconvolution to
each original512 RGB canvas, clamps negative H concentration, converts it to
bounded darkness and divides by its own frozen99th percentile on the ORIGINAL
gray support. The exact formula and zero/weak-signal rules are in
OPTIMIZER_REDESIGN. Only image preprocessing changes versus the already-run
shared-affine gray controls: not masks, affine, machine matches, optimizer,
priors,257P1ac representation,300-gradient budget or final selector.

All25 directions complete before scoring; all300/332 gradient/objective budgets
complete without failure. Mean / mean-pair-p90 TRE in512canvas pixels:

| Previously viewed specimen | Shared-affine gray | H proxy |
|---|---:|---:|
| MIIT /3 directions | 3.548752 /5.907957 | 3.568221 /5.942328 |
| Lung /20 directions | 4.568541 /9.593679 | 4.567560 /9.636410 |
| HistoReg /1 | .851903 /1.586005 | .925172 /1.664603 |
| Kidney /1 | 2.364866 /4.756907 | 2.355327 /4.552949 |

Kidney's tail improves, but its mean change is small; lung's mean is essentially
unchanged with a worse tail, and MIIT/Histo worsen. This is not a useful general
advance, and no matrix/percentile/epsilon sweep follows. The known weak-H MIIT7
image retains68.76% zero H values; calibration raises its512-scale median
descriptor variance to.003589, so its adverse outcome cannot simply be blamed
on leaving the low-amplitude channel below the descriptor's1e-4 stabilizer.
That does not make the proxy a faithful physical stain measurement.

Complete calls take4.20--5.86s, allocated GPU peaks198--204MiB, and the serial
25-case run takes121.26s; timings exclude frozen initialization/matches and
include the new CPU preprocessing/diagnostics. Minimum actual corner ratio
across outputs is.0068615, above.001. The H and gray total energies belong to
different functionals and are not compared as convergence scores. Outputs,
unchanged label denominators and paired deltas are in `stain_proxy_all25_t21`.

### Additional real cohorts: benefit is dataset-dependent, not universal

Full released STANDARD DeeperHistReg and the shared-affine analytic300 control
have now both completed ALL22 predictions before evaluation:20 ordered stain
directions of the same lung specimen, one HistoReg direction and one kidney
direction. These are THREE previously viewed specimens, not22 patients or an
untouched validation set. The table uses the same512 moving-canvas coordinates
and every declared80/77/69 paired landmark; each entry is mean TRE / p90 TRE,
with the lung entries averaging the20 separate direction statistics.

| Specimen / directions | Original-frame A | Shared-affine-frame A | Released STANDARD DHR |
|---|---:|---:|---:|
| Lung /20 | 4.520228 /9.555206 | 4.568541 /9.593679 | 6.046615 /13.914145 |
| HistoReg /1 | .793449 /1.544649 | .851903 /1.586005 | .712328 /1.618927 |
| Kidney /1 | 2.351586 /4.507429 | 2.364866 /4.756907 | 1.908163 /3.178382 |

The new shared-frame control changes ONLY descriptor frame and output path
relative to each original A case: no rematching, changed rates/priors, additional
seed or landmark-dependent choice. Its MIIT benefit does NOT transfer here:
all three specimen means worsen, and only7/20 lung means improve. Consequently
"feature-frame correction" should not be read as a universally better or
mathematically equivalent replacement. The prewarp changes the image functional
and introduces resampling; original and shared-frame outcomes remain separate.

Saved native initial affines reveal an important confound. Lung native-initial
mean/p90 is6.291708/11.390313, versus our common-initial6.661222/12.234678.
Native nonrigid registration then only improves its mean to6.046615 and worsens
its tail to13.914145;7/20 direction means and14/20 tails worsen. Thus our lung
advantage is not explained by a worse aggregate native initializer. Histo native
initial mean2.339111 is close to our2.332173. Kidney differs materially:
native initial4.322930 is better than our6.117413. Its final advantage cannot be
assigned solely to the nonrigid algorithm. This is an observed sequential
decomposition, NOT the counterfactual result of exchanging initializers.

All22 native calls return successfully, but success status is not anatomical
success. Every saved native field contains some nonpositive bilinear-cell
corner determinants. This is a full-field local diagnostic, potentially including
background/borders, not a claim that all tissue folds or that a native continuous
model was globally certified. Our actual257-square P1-ac outputs pass their
declared boundary/four-corner checks. No native field is repaired or used as a
target for our methods.

Single complete calls: shared-frame A4.05--5.96s, allocated peaks200.13--203.89MiB;
native STANDARD17.52--77.68s, observed allocated peaks1.47--7.71GB(decimal).
The native calls include native initialization and native-resolution processing;
A reuses frozen image-only affine/matches and512 rasters. These different scopes
do not establish a matched end-to-end speedup. Equal-specimen means2.595104(A)
and2.889035(DHR) are dominated by the lung difference and do not imply universal
superiority. The exact per-case sources/configurations are in
`outputs/coordinated_instance_registration/native22_comparison_t20.json`;
native and A outputs live in `native22_dhr_standard_t20` and
`existing22_shared_affine_a300_t20`. Independent coordinate/aggregate checks pass.

### Coupled finite-displacement initialization: no material accuracy gain

The precise surrogate and construction are defined in OPTIMIZER_REDESIGN.md.
In short,1089 possible2D shifts at each65-square proposal-grid node are coupled
through a quadratic spatial term. Exact screened Galerkin solves produce one
image-only proposal;16 existing safe coordinate steps construct a legal initial
map on the ACTUAL257-square output grid. The proposal is not a certified map.
This seed may raise the original energy before ordinary300-gradient refinement;
the final selector still compares identity, seed and accepted prefixes using the
original complete512 objective, never anatomical landmarks.

All three MIIT runs complete300 gradients/333 objective calls without failure.
The first two seeds genuinely have higher E512 than identity and nevertheless
enter refinement. The final mean/p90 is3.547609/5.917437 versus ordinary
Adam300's3.548752/5.907957: a negligible mean difference and worse tail, not a
breakthrough. All48 construction scales equal1; raw proposals already have no
folds, so safety contraction cannot explain this outcome. Seed displacement RMS
is1.672/1.594/2.642 aligned-canvas pixels. Final-map RMS differences from Adam300
are only.217/.080/.083 pixels: refinement largely returns near the old solution.
The new seed construction costs about.3--.4s for cost-volume preparation alone;
complete wrapper calls take7.20/4.55/4.48s and allocated peaks202--206MiB.
Independent P1 evaluation reproduces all328 errors to5.69e-14 canvas pixels,
and independently reconstructed seed coefficients/energies agree with execution.
The exact recipe supplies no useful accuracy improvement; no range/coupling sweep
follows. The outcome does not refute all discrete matching or certify global
optimality of the original objective. Full evidence is in `miit_coupled_seed_t20`.

### Restart correction: scope of the earlier DHR comparison

The tables below retain their original measured values, but the method previously
called "native DHR" is more accurately **DHR shared-affine reduced-512**. Its
objective and nonrigid implementation are DHR's; its full configuration is NOT
the released complete pipeline. `tools/coordinated_dhr_common.py` substitutes our
supplied affine and sets a 512 raster, five levels and 30 iterations per level.
The released fast preset uses 2048 and seven levels of 100 iterations; the
standard preset uses 4096 and eight levels (seven times 100 plus 200). The old
0.690-second timing excludes native image-based initialization. Thus the earlier
comparison is useful for a shared-initialization diagnostic, not evidence of
superiority over recommended full RegWSI/DeeperHistReg. The restart adds genuine
released-preset runs, with native preprocessing and all configuration differences
disclosed. The first complete runs and the corrected MS experiment are below.

### T+19h restart experiments: full baseline and one failed objective hypothesis

Both released presets were actually run on AI RTX A6000, original native TIFFs,
loading ratio1, the preset's own preprocessing/initialization/nonrigid schedule,
and no supplied affine. All three directions terminated successfully as software
calls. Fast nevertheless failed anatomically on2-to3 and10-to11; successful call
status is not registration success. Every available123/107/98 annotation pair
was scored after prediction. The same512canvas conversion as before is used
only for the comparison units, not to reduce the native baseline's input.

| Method / objective | Mean pair mean TRE | Mean pair p90 TRE | Actual complete registration call |
|---|---:|---:|---|
| Shared-affine coordinated A, previous continuation | 3.548752 | 5.907957 | 4.14–6.13s |
| Shared-affine F2, previous continuation | 3.554327 | 5.942943 | 12.48–14.20s |
| Coordinated A, simultaneous five-scale objective | 3.608180 | 6.060186 | 7.53–8.85s |
| F2, simultaneous five-scale objective | 3.612537 | 6.062816 | 14.60–16.74s |
| Released DHR fast, native full pipeline | 119.044162 | 173.029174 | 8.18–9.79s |
| Released DHR standard, native full pipeline | 3.672192 | 6.432444 | 30.56–32.95s |

TRE is Euclidean target-registration error in common512canvas pixels, not
micrometers. Means weight the three pairs equally. Safe-method call times above
exclude the already-computed shared initialization/matches; native DHR times
include its own initializer. Therefore this table is NOT an apples-to-apples
end-to-end speed ratio. Initial parallel control/MS calls used separate idle
GPUs5/6; corrected MS calls were serial onGPU5. These are single-call descriptive
timings, not repeated hardware benchmarks. Allocated peaks were approximately
201–202MiB A-control,211–212MiB A-MS,461–462MiB F2-control and488–490MiB F2-MS.

The MS hypothesis failed this fixed test: every one of six method/pair means
became worse; aggregate mean and p90 also worsened while cost increased. It
averages32/64/128/256/512 normalized image terms at EVERY stage, keeping priors,
points and512OOB once, so it changes objective and continuation, not geometry.
All twelve primary safe runs completed300gradients with no failed trials;
actual corner floors remain strictly positive. Independent mixed-precision
value/gradient recomputation agrees to5.55e-17/4.44e-16 on a small separate test.
The selected MS recipe is not promoted; this is not a proof that all multiscale
metrics fail. No neighboring weight sweep follows this negative result.

One first MS run used an algebraically equivalent subtraction/addition of image
scalars; after a mixed-precision arithmetic cleanup only MS was rerun, BEFORE
primary scoring. The first output remains in `miit_multiscale_sum_t19`; the
primary result is `miit_multiscale_sum_direct_t19`. Control continuation was
unchanged and independently reproduces the earlier scores. Formula and test
details are in OPTIMIZER_REDESIGN.md. Full332objective calls require4.5474times
the prior continuation's raster queries; the300gradient evaluations require
five times its image raster work. Neither count is a measured runtime ratio.

Native standard has mean TRE3.087571/3.280505/4.648501 on the three pairs
and still a roughly53.5pixel worst error on7-to8. Scoring only its stored native
initial affine gives mean/p90=4.887064/8.282913, compared with the original
common initializer4.904193/8.897884. Its native nonrigid stage therefore reduces
mean by1.214872pixels (about24.9%), whereas the mean difference between initializers
is only.017129pixels. This is an observed stage decomposition, not the unrun
counterfactual of giving every method exactly the native initializer. The native
7-to8 maximum actually worsens from50.448319 to53.497967. Its actual saved bilinear
field has nonpositive local corners in all three cases; no post-hoc repair was
applied. The scorer checks its own pixel-center field cells and does not claim
a global boundary theorem or an equivalent257P1 representation. Independent
rectangular/resampling/saver fixtures verify the declared field evaluator,
not equivalence to DHR's optional cubic landmark utility or rendered-image path.

The audit also found approximately23–25% of our MIIT intensity-threshold support
outside the provided tissue mask. That is a modeling issue worth a separate
fixed-support test, not a demonstrated violation of DHR's mask-free preset.
The completed mask arm uses images PLUS released semi-manual tissue masks and
does not remove any evaluation landmark. It also shows no material improvement:
A mean/p90=3.565586/5.933243; F2=3.556160/5.926603. All six runs complete300
gradients with valid outputs. Thus the background-support observation is real,
but it is not supported as the main cause of the registration plateau. Keep the
image-only threshold baseline; do not require extra annotations without benefit.

The next six-run pilot substitutes postwarp-intensity NGF for MIND, using the
same geometry, frozen affine/matches, priors, original support, continuation and
300 gradients. It also fails to improve registration: A mean/p90 becomes
4.060915/6.914149 and F2 becomes3.982225/7.187874, versus the shared-affine MIND
control3.548752/5.907957 and3.554327/5.942943. All six runs finish without failed
trials and all exported maps pass the strict stored-binary certificate. A takes
5.60–7.06s and179–180MiB allocated peak; F2 takes12.82–15.95s and438–441MiB.
These are descriptive complete optimizer calls excluding the frozen common
initializer, not clean repeated end-to-end speed ratios. Outputs and all328
posthoc point errors are in `miit_ngf_t19`. The NGF derivative is independently
verified, but correctness of a derivative is not evidence of anatomical benefit.
This exact recipe is retired without an epsilon sweep. Its finite epsilon can
favor larger warped gradient magnitudes, and flat moving regions have zero
image derivative; no global-capture or true-map optimality theorem was claimed.

After these three distinct negative interventions (simultaneous multiscale loss,
released tissue support, NGF), further small objective substitutions are not the
next priority. The next decision separates lost high-resolution image evidence
from the local optimizer's inability to establish coherent correspondences.
Native standard DHR's stage decomposition above makes a purely better affine
initializer an insufficient explanation for its nonrigid gain on these cases.

### T+20h: same-functional convergence control and stiffness preconditioner

This experiment changes the optimizer, not the corrected shared-affine MIND
objective. All original masks, affine, machine points, priors, source P1 grid,
coefficient/raster levels and full512 prefix selection stay fixed. The new
direction uses the inverse of the frozen-rotation ARAP majorizer's exact scalar
Galerkin stiffness, followed by analytic feasible-step bounds and complete-loss
Armijo acceptance. A sine transform applies this particular inverse exactly in
real arithmetic; it IS a global structured linear solve, not a new solve-free
decoder or a learned network. Its full definition and independent dense-matrix
checks are in OPTIMIZER_REDESIGN and BASELINE_PROTOCOL_INDEPENDENT.

| Optimizer | Mean pair mean / p90 TRE | Actual gradients | Complete objective calls per pair | Complete optimizer call |
|---|---:|---:|---:|---|
| Corrected-frame Adam control | 3.548752 / 5.907957 | 300 | 332 | 4.14–6.13s |
| Same Adam with threefold budget | 3.559354 / 5.904981 | 900 | 932 | 11.48–12.80s |
| Exact-stiffness physical-fiber descent | 3.584154 / 5.936919 | 300 | 1877 / 1967 / 1886 | 9.27–11.26s |

Each new arm actually finishes all three cases before manual-coordinate scoring.
All six maps pass the saved topology check. Longer Adam lowers the complete
objective on every pair but does not improve aggregate mean anatomy. The
stiffness method does not even beat Adam300's final complete objective in this
budget. Its allocated peaks are about142MiB versus200–203MiB for Adam900, but
the lower-memory implementation does not make this an accuracy/speed success.
These are single complete calls excluding the common initializer/matches,
not repeated end-to-end benchmarks or intermediate time-to-accuracy curves.

All900 stiffness directions start with feasible trial scale1: geometry never
reduces that initial scale. Nevertheless complete-objective Armijo requires
1265/1355/1274 halvings, with accepted scales down to1/128. All stages finish
their gradient budget; none claims a stationary point. Thus the observed cost
is associated with objective acceptance, not a folding-bound failure. The
prior-only majorizer is not the curvature of the entire image/point objective;
more detailed attribution requires further evidence. No neighboring regularizer
or initial-step sweep is promoted from this failure.

Artifacts: `miit_adam900_t20/landmark_scores.json` and
`miit_stiffness300_t20/landmark_scores_complete.json`. The first stiffness score
retains an archive-only DHR read failure: uncollapsed nested relative paths
exceeded the Windows path limit although the canonical files existed. Resolving
artifact paths before opening fixes this;58scorer tests pass, and rescoring
recovers the unchanged archived DHR values without changing ANY new map or A/F2
score. The failed first score is preserved, not reported as registration failure.

## 1. What is implemented, and what is not

Implemented: a differentiable local forward decoder which takes learnable
displacement coefficients and an already valid vertex table, and returns another
valid P1 vertex table on the SAME dense source triangulation. Its geometry uses
local determinants and parallel reductions, not inversion of a global mesh
matrix. A sequence of such updates is used for image-driven per-instance
optimization, with actual257-by257 control vertices in the main experiments.

Not established: a newly trained image-to-map neural network, inference-time
anatomical superiority over state-of-the-art registration, clinical validity,
or universal approximation of every digitally admissible homeomorphism by a
particular finite schedule. Local decoder gradient checks do not establish
that a new neural encoder has learned complex registration. Optimizing one
image pair's latent coefficients is a different task from training an encoder.

## 2. Domains, coordinates and the output function

Let Omega=[0,1]^2. Let N be the number of vertices along each axis, with source
vertices X_ij=(j/(N-1),i/(N-1)),0<=i,j<N. The N=257 production lattice has66049
control vertices,65536 quadrilateral cells and131072 source triangles. These
are map degrees of freedom, not merely dense queries of a25-square control map.

For each cell its source vertices are called a,b,c,d in top-left,top-right,
bottom-right,bottom-left order. Fix the a-c diagonal. A P1 map f_Y is the unique
continuous function that maps X_ij to Y_ij and is affine on each of these source
triangles. P1 means degree-one polynomial on each triangle. The connectivity
and source triangulation are fixed even when Y becomes irregular.

The complete image-sampling function is

    F(q)=A f_Y(q)+b0,

where A is a frozen orientation-preserving2-by2 image-derived similarity and b0
is a frozen translation. The residual boundary is Y_ij=X_ij on all four sides.
Hence f_Y maps the rectangle to itself; F maps it onto its affine image, not
necessarily the entire moving-image canvas. The saved archive retains the
residual table and original affine separately. Evaluators apply A exactly once.

An image of width W,height H defines intensities at ((j+.5)/W,(i+.5)/H), its
pixel centers. Map vertices instead lie at endpoints j/(N-1),i/(N-1). Map
sampling is declared P1; intensity/feature sampling is bilinear with explicit
zero padding outside the image. These two interpolation operations are distinct.

## 3. The exact discrete feasibility constraints

For vectors v,w in R^2 define det(v,w)=v_x w_y-v_y w_x. For a mapped cell protect

    q1=det(b-a,d-a),  q2=det(b-a,c-b),
    q3=det(c-d,c-b),  q4=det(c-d,d-a).

Each q is twice a signed triangle area. Normalize by the source value
q_ref=1/(N-1)^2, so the identity has q/q_ref=1. Production requires all four
ratios greater than eta=.001 and an exactly fixed, ordered rectangle boundary.

The four inequalities ensure strict orientation for the triangles of BOTH
diagonal choices. Together with continuous edge sharing and an injective outer
boundary, they imply global P1 homeomorphism for the declared a-c map. A
homeomorphism is a continuous bijection whose inverse is continuous. The same
table also defines a legal quadrilateral-bilinear map, but that is a different
function and is not silently substituted into scoring.

The eta floor excludes some legal maps with tiny positive determinants. In real
arithmetic the formula below preserves that floor; implementation additionally
checks actual rounded candidates and the exported table. Returning the previous
valid anchor after a numerical failure is an explicitly recorded fallback, not
post-hoc fold repair. Native DHR receives no such repair and no global guarantee.

## 4. Coordinated forward updates and their gradients

Fix a current accepted table Y, a direction e equal to(1,0) or(0,1), and scalar
amplitudes u_i at the vertices, with zero boundary amplitudes. Update all interior
vertices simultaneously by Y'_i=Y_i+u_i e. For a triangle(i,j,k),

    det(Y'_j-Y'_i,Y'_k-Y'_i)
    =det(Y_j-Y_i,Y_k-Y_i)
     +(u_j-u_i)det(e,Y_k-Y_i)
     +(u_k-u_i)det(Y_j-Y_i,e).

The quadratic term vanishes because det(e,e)=0. Thus every protected normalized
constraint is EXACTLY s_k+(C_Y u)_k>0, where s_k=q_k(Y)/q_ref-eta is its current
positive slack and C_Y is a sparse local linear operator assembled implicitly
from current edges. No finite-difference approximation of determinant change
is used. The matrix need not be materialized or inverted.

At scale l a learnable scalar coefficient table z_l is interpolated as a RAW
proposal r=P_l z_l on the fixed fine vertices. P_l denotes ordinary scalar
proposal interpolation plus boundary zeroing; it is not resampling an accepted
map or composing maps on an unrelated control grid. Levels are17,33,65,129,257.
The actual output table remains257-square throughout these production stages.

Two entry operators were tested. Define the nonnegative feasibility gauge

    g_Y(r)=max(0,max_k[-(C_Y r)_k/s_k]).

The radial decoder uses u=r/(1+g_Y(r)); then each new slack is strictly positive
for every finite proposal in real arithmetic. Its gauge is positively homogeneous.
For unrestricted scalar amplitudes, this radial transformation reaches every
feasible single-direction displacement u: its inverse is
r=u/(1-g_Y(u)), because feasible u has g_Y(u)<1. This is a statement about the
single-direction feasible set, not a theorem that a fixed finite alternating
network reaches every2D homeomorphism. Coarse P_l may restrict its proposal space.

The production analytic-step operator uses

    sigma=min(alpha_trial,theta/g_Y(r)),  u=sigma r,

with theta=.95 and theta/0 interpreted as infinity. Here alpha_trial=1 for the
ordinary proposal. It leaves at least(1-theta)s_k slack whenever the geometric
bound is active. Both operators are differentiable away from max/min ties;
autograd includes the derivative of the active scale. Detaching sigma would be
a different and generally incorrect gradient. Finite-precision checks and image
accept/reject decisions are separate discrete branches, not claimed globally
smooth operations. Local value/gradient and deliberate failure tests cover the
decoder; a gradient through300optimizer steps is not claimed or retained.

For completeness, with the anchor held fixed and a scalar loss whose derivative
with respect to u is v, the coefficient vector-Jacobian product is

    dL/dz_l = P_l^T [sigma v+(r^T v) grad_r sigma].

P_l^T is the transpose interpolation accumulation. For an active constraint row
c_k of C_Y, radial sigma has grad_r sigma=sigma^2 c_k/s_k. The geometry-limited
analytic scale has grad_r sigma=theta s_k c_k/(c_k r)^2; when the trial limit is
strictly active, that gradient is zero. These formulas apply away from ties;
anchor derivatives additionally differentiate C_Y and s, which ordinary AD
can retain in a neural-layer usage. In the instance optimizer accepted anchors
are intentionally detached between stages. A vector-Jacobian product means
multiplying a scalar-loss derivative backwards through a function's Jacobian,
without storing or forming that entire dense Jacobian.

This is coordinated motion: adjacent vertices may move a long common distance
while their DIFFERENCES, not original mesh spacing, determine admissibility.
A worst active cell can still limit the global scalar scale. This limitation is
measured rather than denied. F2 provides a local-patch comparator under identical
input evidence and objective; it is not automatically worse by construction.

Successive accepted tables are on the same source triangulation. Geometrically,
one can define an affine update h on each current deformed triangle taking its
old vertices Y to its new vertices Y'; then f_Y'=h composed with f_Y EXACTLY.
There is no generic continuous-flow integration followed by uncertified sampled
P1 interpolation. This statement relies on current nondegenerate triangles,
shared-edge consistency and the preserved connectivity.

## 5. What optimization actually consumes

Inputs for one case: fixed and moving images, one common positive image-only
affine, frozen machine-generated correspondence points/confidences, and the
fixed reference mesh. Manual evaluation landmarks are not inputs. The output is
the saved P1 map, its affine factor, explicit geometry certificate, runtime,
gradient/evaluation counts and diagnostics. There is no trained CNN in this phase.

The retained objective is

    E = I +3 R +.0001 S +.1 M + O.

Here are definitions of every term. Convert images to inverted grayscale
intensities G in[0,1]. At each raster scale, use the eight integer offsets
(2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2). For each offset a compute
D_a(x), the3-by3 local average of(G(x)-G(x+a))^2, with replicated edge values
for shifting and available-neighbor averaging at pooling edges. Set

    Phi_a(x)=exp[-(D_a(x)-min_b D_b(x))/(mean_b D_b(x)+.0001)].

These eight values are the local self-similarity descriptor (MIND-like here;
not a claim of reproducing every published MIND variant). Compute Phi_f and
Phi_m separately before warping. Let m(x) be the fixed foreground mask: threshold
G_f>.04 at full resolution, area-reduced to lower raster scales. Let Z=sum_x m(x)
on the fixed pixel centers; this denominator is unchanged by the candidate map.
With bilinear zero-padded feature sampling B, define

    I=sum_x m(x) mean_a |Phi_f,a(x)-B(Phi_m,a,F(x))| / Z.

For a source triangle t=(i,j,k), define edge matrices
B_t=[X_j-X_i,X_k-X_i],D_t=[Y_j-Y_i,Y_k-Y_i],J_t=D_t B_t^{-1}.
Only a2-by2 SOURCE edge inverse is involved. The uniform mesh gives equal source
areas, and R=.5 mean_t ||J_t-R_t||_F^2. R_t is the closest orientation-preserving
rotation. Explicitly put a_t=J00+J11,b_t=J10-J01,s_t=sqrt(a_t^2+b_t^2);
then R_t=[[a_t,-b_t],[b_t,a_t]]/s_t. Positive detJ ensures s_t>0. Identity has
zero R; a rigid rotation also has zero R, whereas spatially varying rotations
usually induce stretch. This is an elastic prior, not the topology safeguard.

Let K_l be each of the four2-by2 cell-corner derivative matrices, with first
column equal to the relevant horizontal mapped edge divided by source spacing
and second column the vertical mapped edge divided by spacing. Define

    S=mean_l [||K_l||_F^2+||K_l^{-1}||_F^2-4].

For a2-by2 K with positive determinant, ||K^{-1}||_F^2=||K||_F^2/det(K)^2;
the implementation uses that identity rather than a mesh-system solve. This is
four-corner quadrature, not an exact quadrilateral integral; identity has S=0.

Frozen machine points are(q_j,p_j,c_j): fixed-unit keypoint, affine-aligned
moving-unit keypoint and original confidence. Their original moving location
is A p_j+b0. Statically discard only machine targets outside the original moving
rectangle, never difficult MANUAL evaluation points. Normalize eligible c_j
to weights w_j with sum1. With rho(r)=sqrt(1+||r||_2^2)-1 and kappa=8pixels,

    M=sum_j w_j rho(512 A(f_Y(q_j)-p_j)/kappa).

The translation cancels; A remains in the metric. There is no extra kappa^2
multiplier. Machine confidence is not anatomical ground truth, and no derivative
through keypoint selection/matching is claimed. Finally define componentwise
outside excess t_d(v)=max(-v_d,0)+max(v_d-1,0) and

    O=sum_x m(x) sum_d t_d(F(x))^2 / Z.

Thus out-of-bounds image queries are both zero-padded and penalized, not dropped.
I and O use the current raster scale; M always uses the full512pixel metric;
R and S use the actual fixed fine control mesh. A stage compares its COMPLETE
objective at that stage's raster scale, with all terms and the coefficients
above; it does not compare image loss alone. Accepted stage maps are additionally
scored by the complete512objective for final selection. Consequently the512
objective need not decrease at every lower-resolution stage; best-full selection
retains the best full-resolution accepted output rather than asserting monotonicity.

Each stage freezes its accepted anchor Y while Adam optimizes that stage's z_l.
Trial candidates do NOT accidentally accumulate within the latent solve. Only
a geometry-valid candidate that improves the complete STAGE-objective comparison
becomes the next anchor. The final result is selected by the full512-resolution
objective among accepted maps, never by manual landmark error.

Production analytic: five scalar scales, x/y alternating stages,30Adam steps
per stage,300gradient steps total,310decoder trials,332complete objective calls.
Learning-rate calibration is.004*16/(level-1), in pre-affine normalized units.
F2 uses two five-scale cycles with30steps each for the same300gradients and
has the corrected.95strict-floor reserve. Both start from the same affine plus
identity residual. Native DHR has its own objective/preprocessing and is an
application comparator, not an equal-objective geometry ablation.

## 6. Evidence available aroundT+15h

The lung comparison covers20ordered stain directions from ONE previously viewed
physical specimen, with all80matched manual IDs per direction. Directions and
landmarks are correlated, not20patients or1600independent observations. Errors
below are Euclidean target-registration errors in512canvas pixels, computed
after prediction by an independent P1 evaluator. Mean-p90 means the arithmetic
mean of each direction's90th-percentile error, not the pooled90th percentile.

|Method|Mean direction mean TRE|Mean direction p90 TRE|Output topology|
|---|---:|---:|---|
|Common affine|6.661222|12.234678|Positive affine|
|Coordinated analytic|4.520228|9.555206|Certified P1-ac/all-four corners|
|Corrected full-budget F2|4.551344|9.657840|Certified P1-ac/all-four corners|
|Native DHR|5.133593|11.552831|No hard global guarantee|

Analytic improves mean TRE versus affine in all20directions, but native DHR
wins six direction means, including all fourproSPC-source directions. Analytic's
aggregate advantage over corrected F2 is only0.68%mean and1.06%tail: do not claim
strong anatomical superiority. Native DHR remains much faster in these small
canvas experiments. Clinical competitiveness and unseen-patient generalization
are not established.

Warm complete-call engineering comparisons, with repeatedABBA order, reduce
coordinated runtime by compiling existing ARAP/shape work (about4.05to2.95s on
one known pair) and then by precomputing fixed machine-point triangle indices
and barycentric weights (2.95to2.70s on a second known pair). Map/objective
differences stay at the scale of measured repeat variation, not bitwise identity.
Cold compiler setup is not a speedup. Typical allocated GPU peaks for the main
analytic optimizer are about205--229MB; F2 about495--500MB; native DHR about70--73MB.
These are measured allocated peaks, not whole-process RAM/driver/compiler peaks;
setup/load/export inclusion is specified for each timing experiment.

Strong adverse interventions are retained: more iterations/budget redistribution
can lower the objective while worsening landmarks; removing dense appearance
worsens mean TRE in all20directions; a finite independent-label displacement
prefix improves appearance but raises the complete objective and is rejected
in all six attempted axes. Neither lower proxy loss nor valid topology alone
proves anatomical benefit. Those recipes are retired rather than swept nearby.

Additional prostate transfer now completes after a DISCLOSED image-only
initializer amendment: four right-angle moving views, with original-frame
keypoints and fixed inlier/support/fit ranking, specified before reading manual
coordinates. The original raw-image initializer's two failures remain recorded.
The original assumption that all124labels are finite is also false: the release
uses(+inf,+inf) for absent annotations. Author metric code excludes absent labels.
An explicit source-availability correction, before computing errors, scores ALL
123,107,98both-source-available IDs respectively, identical across methods; all
missing labels and original strict-scoring failures are reported separately.
Finite-point bounds and invalid-prediction guards are not relaxed.

|Method|Prostate equal-pair mean TRE (512px)|Mean pair-p90 TRE (512px)|
|---|---:|---:|
|Common affine|4.904193|8.897884|
|Coordinated analytic|3.897540|6.289528|
|Corrected F2|3.964294|6.381702|
|Native DHR|4.050615|7.181205|

Independent native-frame/CSV/generic-triangle evaluation agrees within1.13e-12px;
actual six safe maps have1,572,864positive corners, minimum ratio.2015743,
exact boundaries/shared affines and all300gradient budgets complete. Native
fields have11506,5271,15158nonpositive corners, retained in scoring. Analytic
improves affine mean/p90 on all three pairs, but loses to F2 on7-to8mean/p90
and to DHR on7-to8p90. Its maximum error is worse than DHR on two pairs; the
7-to8worst error worsens versus affine and remains52.71px. It is not an outlier
cure. Single mean complete calls are4.218s analytic,13.212s F2,.690s DHR;
this is not a paired timing estimate or equal-objective application comparison.
This additional previously used specimen supports bounded development transfer,
not unseen-patient/clinical/general superiority. Details and actual files are in
FORMULATION47, miit_three_rotations_t153/predictions.json and its separately
saved available_landmark_scores.json.

As a bounded post-hoc diagnostic, all123/107/98available targets lie inside
the common affine image rectangle. Their distance-to-range necessary lower
bounds are zero. The worst7-to8target is38.98canvas pixels inside its closest
range boundary despite52.71pixels of registration error. Its error is therefore
not forced by a target outside the decoder's image. This does not establish
that boundary constraints have no other effects or that image evidence is adequate.

## 7. Current interpretation, not final closure

The strongest established result is a scalable, differentiable, no-global-solve
safe update operator plus a useful instance optimizer on the known lung cohort.
The geometry problem is not equivalent to solving arbitrary prescribed
facewise Beltrami coefficients. No Beltrami-recovery objective is required here.
The main unresolved tasks are robust image-derived initialization/evidence,
cross-specimen anatomical usefulness and eventual amortization into a learned
network. The24-hour goal remains active; this document does not mark it complete.

An additional engineering test consolidated detached trial diagnostics into one
CPU transfer while preserving guards and gradients. Despite CPU bitwise checks
and valid28actualGPUoutputs, its paired real timing was not robust: HECC warm
median2.906to2.738s, HEK2.731to2.757s (regression), with~.4swithin-arm ranges.
The predeclared two-pair condition fails; the default remains unchanged. This is
a retained negative result, not a speed benefit. See FORMULATION48.

### 7.1 Why more iterations are not an automatic solution

Six unchanged-recipe MIIT replays save ten accepted maps each. All60snapshots
pass strict geometry checks. Prefix outputs are selected by the complete512
objective E1, never landmarks. On2-to3 the analytic E1 falls.382334to.374758
from90to300gradients, but mean TRE increases3.34619to3.55292pixels. On10-to11,
F2 correctly chooses stage4 by E1 although terminal stage9 has better anatomy.
This is an observed objective-anatomy disagreement, not a prefix-selection bug.
It does not prove optimization globally adequate or image evidence impossible.
Snapshot times include copying overhead; no clean speed claim uses them.
Replays are not bitwise originals: the largest F2 vertex difference is7.72e-5
unit coordinates, although final landmark scores differ by only~4e-8pixels.

![Full E1 and anatomical error along unchanged-recipe development paths](../../outputs/coordinated_instance_registration/results/miit_trajectory_t16/trajectory_summary.png)

Each column is one correlated pair from the same sample. Horizontal axis is
cumulative gradient evaluations, NOT time. Top: complete512E1 of accepted maps.
Middle/bottom: mean/p90TRE; solid is the accepted state, dashed the E1-selected
prefix. All available123/107/98points remain. Different stages use different
coefficient/image resolutions, but the plotted objective is ALWAYS full512E1.
These curves do not authorize TRE-based stopping or early-checkpoint selection.

### 7.2 A feature-frame limitation with an exact controlled example

Our eight MIND-like channels measure directional patch self-similarity.
Under a known affine image transform, both offset directions and patch windows
transform. Merely moving a precomputed descriptor field leaves channel identities
unchanged and is generally not equivariant. For exact90-degree rotation the
necessary correction is a known channel permutation. On an existing512texture,
the uncorrected true-map mean channel error is.26229, while the corrected maximum
is2.22e-16; identity error is zero. A separate independent pixel-loop calculation
confirms the formulas. This isolates a mechanism, not real MIIT causality.
Generic affine transforms need more than channel permutation. Sparse landmarks
also do not define true neighborhood warps. Consequently no registration objective
has been changed on the strength of this control. See FORMULATION50.

A follow-up OFFLINE fixed-affine-frame comparison retains all328available
centers. Raw descriptors prefer the annotated center over the current mapped
center for160IDs; affine-prepared descriptors do so for174. Mean cost ranking
margin changes-.002027to+.008449, but7-to8's mean margin gets worse. All
nominal support flags are valid, and an independent implementation reproduces
every ranking sign. This is weak/mixed representation evidence, not a registration
improvement. It is not a true-neighborhood ground-truth experiment.
The one six-configuration pilot is now complete and independently checked.
Analytic aggregate mean/p90 improves from 3.897540/6.289529 to
3.548752/5.907957 canvas pixels; F2 changes from 3.964294/6.381702 to
3.554328/5.942943. However, BOTH methods worsen 7-to-8 mean, whose maximum
stays around 53 pixels; 120/328 analytic and 118/328 F2 individual errors worsen.
Under the predeclared mixed-result condition the EXACT variant is retired from
main-line adoption, while its positive aggregate evidence is retained. These
same labels influenced choosing it: no untouched confirmation is claimed.
All six 257-square outputs pass strict topology checks, minimum corner ratio
.228070. Serial optimizer calls are about 4.05-5.84 seconds analytic and
12.45-13.55 seconds F2, with allocated peaks about 219-223 and 479-480 MiB;
no paired speed claim follows. See FORMULATION52 for definitions, support
limitations, all pair scores and the archived metadata correction.

The next bounded diagnostic is explicitly a manual-label ORACLE, not registration:
fit ALL 107 available 7-to-8 centers with the same fixed-boundary/affine/257-square
P1 class and exact safe operator, starting from the archived original map.
It tests simultaneous sparse representational capacity, which individual
affine-range inclusion cannot establish. Its map may not be used as production
initialization or a training teacher. The 24-hour goal remains active.

### 7.3 A rigorous forward-expressivity statement and an actual capacity witness

The earlier theoretical claim assumed a legal vertex-motion path existed.
For strictly positive FOUR-corner tables with a fixed rectangular boundary,
that assumption can now be proved: represent a target by positive directed
four-neighbor mean-value weights, suppress the graph's four degree-two corners
to apply the appropriate weak-convex-boundary Tutte theorem, and restore those
corners with explicit area calculations. Interpolating positive weights gives
a continuous path in the same four-corner class. A sufficiently fine finite
partition can be realized exactly by alternating full-nodal x/y forward updates.
This is an EXISTENCE bridge to classical convex-drawing/morphing theory, not
an online linear solve. It does NOT give a bounded depth or prove connectivity
above the preset `.001` floor. Full definitions and proof are in
[THEORY_FORWARD_EXPRESSIVITY.md](THEORY_FORWARD_EXPRESSIVITY.md).

Separately, the one actual 7-to-8 label-oracle fit succeeds at the IMPLEMENTED
floor: all 107 centers have error <=.026773 canvas pixels, mean .015168,
p90 .021693, with minimum corner ratio .0130772 and the original boundary/affine.
It uses 300 additional gradients after the archived image registration, not a
new image-only inference result. The original E1 rises .379627 to .589959;
87.18% of this particular cost increase is weighted ARAP. That does not prove
every accurate map must be costly or justify a prior change. Independent
recomputation confirms the map, all point errors and objective decomposition.
See [ORACLE_CAPACITY.md](ORACLE_CAPACITY.md) for its complete inputs, equations,
units, budget, timing and interpretation. A separately predeclared matched F2
oracle control now tests motion efficiency; no oracle map enters production.

The declared warmup+ABBA oracle collection is now complete, including three
FAILED/incomplete F2 attempts. Analytic first certifies all107 errors<=1pixel
after150 gradients (measured1.0783/1.1226s including snapshot certification),
then completes300 gradients and reaches maximum.026773pixel. F2 retains legal
outputs but reaches only165/188 measured gradients before strict rounded-floor
rejections; maximum remains37.48/37.04pixels. This is an implemented-protocol
advantage on one sparse labelled task, NOT a completed equal-budget comparison,
finite speedup ratio, general F2 impossibility or automatic image registration.
Independent recomputation confirms all nine endpoint/threshold maps, including
exact binary-rational near-floor F2 corners. ORACLE_CAPACITY Section8 gives
the full method, counts, timing/memory scopes and failure interpretation.

### 7.4 One actual initializer-restriction check, with no downstream sweep

The original 'affine' initializer is in fact a four-parameter positive
similarity, not an unrestricted six-parameter affine. A one-shot label-free
probe fits both model families to the SAME frozen machine correspondences,
original confidences/world eligibility and eight-pixel pseudo-Huber loss.
The predeclared4-by4 spatial parity split holds each parity out once on all
three pairs. All12 fits converge and remain positive, but full affine improves
only4/6held-out losses. The failed directions are2-to3 train-parity0
(.173494to.188222) and10-to11 train-parity1(.390509to.399405).
The exact branch stops, with no all-point refits or downstream registrations.
Independent coordinates/losses/gradients and a separate optimizer confirm the
result. It is not explained by fitting failure, and it does not establish that
every full-affine initializer is unsuitable. See
[INITIALIZER_STUDY.md](INITIALIZER_STUDY.md) for the complete equations/data scope.
