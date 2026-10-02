# Coordinated instance registration — progress

Started: 2026-10-01 11:26:23 UTC (Asia/Shanghai 19:26:23).
Initial deadline: 2026-10-02 11:26:23 UTC. Goal ACTIVE.
Branch codex/coordinated-instance-registration; base 4ca9f09.

## Initial facts

- User approved review amendments, 24h execution and idle remote compute.
- Existing 17 transfer bundles remain preserved, unrelated untracked files.
- All three SSH probes succeed with ClearAllForwardings=yes.
- Turing and element GPUs occupied: use CPU-only pilots there.
- AI GPUs 6/7 initially 13 MiB, 0% utilization, no compute processes; recheck before jobs.
- No installed skills used; no new large network training required.

## First decision

Observed failure: prior learned-map proxy and anatomy rankings conflict.
Two plausible causes: geometry/optimization limitations; evidence/objective limitations.
Smallest discriminating test: radial vs analytic coordinated vs matched F1/F2 on
independent known maps, in parallel with a shared-objective real baseline.
Decision: implement both inexpensive entries; run early RegWSI-style evidence comparison.
Synthetic-only benefits cannot establish an evidence bottleneck.

## Actual agents

Geometry builder actually dispatched as /root/coordinated_geometry_builder through
runtime tools with gpt-6.1-sol/high and a bounded fresh context.
Second builder dispatch hit the runtime agent-thread limit (completed historical
contexts also remain). Coordinator temporarily owns application/baseline work; no
unavailable second agent/model is claimed. Reuse checked formulas and independent
numerical paths for review until an authorized checker slot can be dispatched.
Calling coordinator model is the active runtime; Markdown does not switch it.

## Geometry research card (builder, before core implementation)

Question: Can a shared direction provide exact full-grid corner safety and useful gradients?
Exact claim: q(Y+u e)=q(Y)+C(Y,e)u for all four corners; radial and analytic-step entries
with positive normalized slack preserve the declared margins in real arithmetic.
Assumptions: common direction, valid anchor, positive reference determinants, boundary
constraints; sliding axes additionally protect every tangential boundary gap.
Falsifiers: independently recomputed corners disagree; unique-active directional finite
difference fails; actual float32 candidate violates margins.
Smallest test: non-square perturbed 3x4 anchor, direct four-triangle recomputation, zero
preservation and directional finite difference. Prior: existing digital_q1 conventions
and convex radial parameterization; no novelty claim for the denominator formula.

## First geometry and real-data milestone (within first hour)

New shared-direction module: radial and analytic entry, fixed/sliding boundary,
all four corner constraints, true safety-scale gradients. Author focused suite initially
34 passed; coordinator independently recomputed triangle determinants through generic
NumPy 3x3 determinants and checked original-image identity/OOB behavior: 6 passed.
Application was independently read/recomputed by geometry builder (not its author):
half-pixels, A(Q1Y), fixed mask denominator, cumulative strain, stage anchors and final
trial evaluation are consistent. Full radial-objective directional FD error ~9.96e-12.

Equal-budget 65-square capacity fitting (240 objective evaluations) does NOT establish
coarse-revisit superiority: radial single vs revisit RMSE .000379/.000527 shear,
.000420/.000531 rotation, .000360/.000711 coarse+fine. Unregularized oracle fits can
produce thin cells although targets are well conditioned. Margin .05 excludes legal
compression target min normalized corners .0339; this is an extra restriction.

Coordinator flagged inactive reciprocal differentiation; builder reproduced actual
analytic near-zero NaN at proposal 1e-305 and fixed it with branch-guarded, mathematically
equivalent feasible-step evaluation. No detached safety derivative. New focused suite
20 passed; root independent reproduction remains to run after integration.

Real HistoReg is a repeatedly viewed DEVELOPMENT specimen, not independent confirmation.
Fixed/moving shared 512 canvases, frozen SAME prior image-only positive affine; Q1
residual with 257-square output (66,049 vertices, 262,144 corner constraints), float32,
batch1. Original MIND evidence, fixed gray-inversion>.04 mask; outside queries retained
with zero padding and explicit quadratic excess penalty. Total=image+.05 cumulative
strain+OOB. No labels or competing dense fields read by optimizer. Saved maps certified.

| Method | Mean TRE, 512-canvas px | p90 TRE | Gradient steps | Optimizer seconds |
|---|---:|---:|---:|---:|
| Common affine | 2.33217 | 3.34554 | 0 | excludes initializer |
| Radial, equal physical rate | .77027 | 1.50234 | 72/100 | 2.05787 |
| Analytic, equal physical rate (pre-nearzero fix) | .81928 | 1.62535 | 96/100 | 2.11692 |
| Radial, edge-calibrated rates | .79804 | 1.50100 | 100/100 | 2.32074 |

Uncalibrated radial had 7 rejected trials and near-floor slack ~2.74e-6; calibrated
lr=.004*16/(level-1) completed all120 evaluations, no trial failures, final min slack
.001273 above floor. Calibrated optimizer peak allocated144MB; median forward+objective
6.92ms, VJP3.90ms. Times include staged optimizer setup, exclude loading/features,
initializer and final exact-sign certificate; no precise superiority ratio yet.
Its lower objective but slightly worse TRE preserves the proxy/anatomy caveat.

Native DHR on same canvases executed3.159s (initial1.328s/nonrigid1.313s) with declared
development config. It has a newly estimated initial affine, so it is NOT a strictly
matched initializer comparison. A native DHR NONRIGID run supplied the exact common
affine is now implemented separately; it does not consume a competing dense teacher.
Native DHR appearance/regularizer still differ from shared-geometry objective.

First Windows scoring hit duplicate OpenMP runtime. Setting MKL_THREADING_LAYER=SEQUENTIAL
resolved it; no unsafe duplicate-runtime override was used. A failed pre-certificate
smoke NPZ remains preserved as incomplete; only complete reports enter the table.

Next: matched F1/F2, NCC-versus-MIND early comparison, further specimens and independent
score/frame checks. Do not conclude real-data competitiveness from this one specimen.

## First-hour continue/change decision

Common-affine native DHR is independently verified (all512² centers max2.84e-14px
coordinate error; stored A,b exact match; no padding/resampling frame change).
All three common-affine native runs completed ~1.97–2.04s excluding the prior initializer.
The complete shared-evidence MIND and squared-local-NCC matrices (3methods×3specimens×2losses)
are saved under outputs/coordinated_instance_registration/development_{edge,ncc_edge}.

| Development pair | Affine | Radial MIND | Analytic MIND | F1 MIND | Native DHR/common affine |
|---|---:|---:|---:|---:|---:|
| HistoReg | 2.332 | .837 | .869 | .798 | .943 |
| Lung lesion | 4.900 | 4.809 | 5.548 | 4.420 | 4.347 |
| Rat kidney | 6.117 | 5.595 | 6.051 | 5.527 | 3.432 |

Values are mean512-canvas TRE, not official scores. Strongest adverse result: kidney
DHR p90=7.110px versus radial10.750/F1 12.378, while DHR's runtime is comparable to
the new optimizers. The new coordinated construction has NOT demonstrated general
real-data competitiveness. MIND generally beats localNCC means here, but tails can
rank differently; do not declare an evidence bottleneck or tune by these labels.

Repeated same-config radial Histo TRE varies .798/.837 (.837 is fresh matrix), consistent
with a sensitivity/nondeterminism issue that needs measured reproduction, not dismissal.
Analytic near-zero gradient fixture independently passed after the fix;7rootchecks passed.
Remaining rejected trials are recorded as actual rounded η-margin failures, NOT NaN gradients.

F2 overlap calibration independently gives the requested identity physical JVP.
First257² F2/gain1 runs reached only32–35gradient steps;17/20 stages rejected candidates
touching the extraηfloor. Actual saved maps remain topology-certified. Gain .75 retained
slack and increased completedsteps to56–58, but11rejections remained. This is NOT a proof
of F2 capacity failure. Next isolate cumulative regularization and roundoff.

Independent known-target controls:48cases at33/65,120trials/100grad each, zeroillegal
candidates. With strain0, F2 best finalRMSE on all three65targets (.000106–.000134).
Analytic reached .001 sooner in that CPU setup. Strain.05 biases allmethods to .013–.019
RMSE, so representation and regularized optimization must remain separate conclusions.

Decision: continue explicit coordinated main line, change from one GLOBAL constraint
scale to tested nonconflicting REGIONAL supports as a main-line variation. A thin-region
fixture shows216–999× more distant motion under identical smooth tapered proposals,
without losing any corner constraint. It proves isolation of the worst-cell bottleneck,
not anatomical improvement. Regional module author34affectedtests; coordinator adds
independent full-grid determinant/reconstruction tests:9rootchecks passed. Actual257
registration benefit remains NOT TESTED. No alternative research branch has been opened.

## Second-hour discriminating questions

129/257 oracle fits now distinguish local-step attenuation from topology rejection.
At257, radial/analytic RMSE stays around .00045--.00253 across the three targets;
F1/F2 at the same120-trial budget is around .008--.016. Correcting F2 accepted gain
to .75 eliminates its float64 rejections but not its257 error. This is a finite-budget
comparison, NOT a theorem that F1/F2 cannot approximate the targets. An independently
checked finite-amplitude proposal loses increasing motion with resolution under F1/F2,
despite exactly calibrated identity derivatives. Coordinated transfer remains stable.

Real-image F2 float64/gain.75 now completes100 gradients on all3 specimens, whereas
float32/gain.75 still rejects floor-contact trials. Its anatomy has yet to be scored.
The geometry builder independently verified that gain1 only protects a CLOSED floor;
gain gamma<1 retains at least(1-gamma) slack per pass in exact arithmetic. Four passes
retain at least(1-gamma)^4, which is NOT a floating-point sign certificate.

Question: can geometry float64 and evidence float32 remove rounded-floor rejection
without doubling all descriptor/sampling storage? Smallest test: finite decoder and
image VJP, same saved-map certificate, then matched3-case precision comparison. Falsifier:
nonfinite/incorrect gradient or continuing floor rejection. This is a precision change,
not a new geometric mechanism; output and raster sampling precisions must be disclosed.

Question: mean membrane strain is cheap for a very small region to collapse. Does an
optional local reciprocal-stretch energy prevent thin-cell trapping under the SAME
objective for all methods? At each actual Q1 corner let J have normalized derivative
columns and define D(J)=||J||_F^2+||J^{-1}||_F^2-4. In2D this equals
||J||_F^2(1+det(J)^(-2))-4 for positive det(J). Average over all four cell corners,
using a common small weight; retain the existing exact feasible decoder. This is a
corner quadrature regularizer, NOT exact integrated Q1 energy and NOT a new topology
certificate. Prior work: [SLIM](https://igl.ethz.ch/projects/slim/SLIM2017.pdf), symmetric
Dirichlet distortion. Smallest test: independent matrix inverse comparison, identity,
rotation, compression and FD. Then one common-weight matrix, not endless weight tuning.
Falsifier: it merely biases maps or spends time without reducing observed thin-cell
failure/optimization stalls. Do not choose the weight or iterates using landmarks.

The first shape-weight1e-4 matrix removes all17 prior mixed-precision F1 rounded-floor
rejections (all100 gradients), but does NOT improve anatomy consistently: kidney
radial/analytic/F1/F2 meanTRE5.776/5.794/5.522/5.386 versus5.642/5.916/5.582/5.227
without shape. Analytic image optimization improves, but not general competitiveness.
Keep this as a conditioning intervention, not claim a registration breakthrough.

Two global/regional support interventions do not show a consistent anatomical gain:
all-level windows and global-coarse/regional-fine both remain worse than DHR on kidney.
Diagnosis now moves to image capture/convergence, not a third nearby patch-size sweep.
Known independently legal image warps have37--61px peak movement. True complete MIND
objective .058--.074 is below all current .113--.144 outputs, yet held-out map RMSE
remains12--23px. This establishes a feasible better-objective solution was not reached;
it does NOT prove the correspondence signal is impossible. The matched original-image
pyramid is the next discriminator. Acceptance is monotone WITHIN each scale's complete
objective; changing scales can increase the full512 objective, explicitly recorded.

Correction from the actual saved native DHR config: its NCC window is7, NOT3 as a
previous descriptive sentence claimed. Executed baseline configs/results unchanged.
Its CLAHE/normalization, diffusion-relative regularizer and boundary freedoms still
prevent calling it an isolated shared-objective geometry comparison.

## Capture-budget discriminator and actual P1 output

Known-texture image pyramid,300 nominal gradients, all else unchanged: coordinated
query RMSE remains4--13px, better than100 steps but not accurate recovery. Analytic
coarse/fine reaches5px at1.616s; F1/F2 need much longer for10px. These thresholds are
computed only AFTER image-selected optimization from accepted-map snapshots. Snapshot
clone/copy overhead is included in runtime; no target participates in iteration choice.
Actual rotation accepted stages produce qmin about.001 despite true target qmin.991;
analytic global scales fall to6.8e-5. A common reciprocal-stretch conditioning experiment
is now justified by this observed failure, not another unmotivated safety-coefficient sweep.

First image matrices were honestly Q1. New explicit P1(ac/bd) query support evaluates
the SAME fixedsource triangles, not a Q1 result resampled onto another mesh. Strong four
corner certificate and fixed boundary support BOTH diagonal interpretations. Queries,
archive interpolation tag and independent NumPy barycentric scorer agree on the chosen
function.28affected application/sampler tests pass, including both diagonal/batch VJPs,
source corners, nonlinear Q1/P1 distinction and protected target-free snapshot callback.
Real P1 comparison still pending; do not retroactively relabel the old Q1 experiments.

Git milestone122004d pushed. First push encountered LFS lock-API timeout with NO new LFS
objects; per-command lfs.locksverify=false allowed ordinary code/docs sync. No persistent
configuration, occupied network ports, SSH aliases or other users' jobs were changed.

## T+2.9h: actual P1 and image-capture comparison (2026-10-01 14:22 UTC)

The real P1(ac) matrix uses257² CONTROL vertices/131072triangles,512² queries,
float64geometry/float32evidence,shape1e-4,100gradients per method. All12 runs
complete without rejected trials and pass the saved-binary four-corner certificate.
Independent NumPy triangle interpolation scores every shared manual landmark.
Means in512canvas pixels, in radial/analytic/F1/F2 order:

| Development specimen | Actual P1, full512 evidence | Q1 image pyramid, shape0 | Common affine | Native DHR |
|---|---|---|---|---|
| HistoReg CD4/CD68 | .785/.877/.819/.834 | 3.176/4.214/1.847/1.784 | 2.332 | .943 |
| Lung lesion HE/proSPC | 4.708/4.892/4.382/4.739 | 6.098/7.446/5.147/4.998 | 4.900 | 4.347 |
| Rat kidney HE/PanCK | 5.634/5.753/5.520/5.322 | 4.677/5.480/3.622/3.461 | 6.117 | 3.432 |

These columns differ in interpolation/continuation/conditioning and are NOT one
isolated geometry ablation. Within each column all four methods share the setup.
The kidney pyramid F2 result is close to native DHR; the same intervention degrades
the other two specimens. This is not a general registration breakthrough. No required
moving landmark lies outside the fixed affine image of the unit square (0/77,0/78,
0/69). The necessary range lower bound is zero; this does NOT establish that the
whole correspondence is representable with our fixed boundary.

Actual P1 warm optimizer times are about1.7s coordinated,3.3--3.6sF1,7.2--7.4sF2;
allocated peaks236/371/528MB respectively. First cold Histo coordinate jobs are2.0--2.5s.
These times exclude input loading and final exact-sign certificate, separately recorded.
Naive actual P1 gathering adds roughly36MB relative to Q1 here, not an unmeasured
claim of lower memory. No graph through the instance optimizer history is retained.

Known-texture300-gradient pyramid/shape1e-4 MIND recovery improves11/12final query
errors versus shape0, but errors still3.54--17.47px. Native DHR on EXACTLY the same
generated images and identity affine gives11.70/9.52/3.82px RMSE for shear/rotation/
coarse-fine at2.00/1.95/2.10s. Its CLAHE/NCC7/diffusion/free boundary are different,
and it is NOT digitally certified. Our analytic MIND errors12.22/7.82/3.54px at
roughly5s do not establish dominance: faster native baseline and adverse shear remain.

Changing only the requested raw localNCC loss in the conditioned300 matrix worsens
ALL12query errors. Native-style NCC alone is not a rescue. Its true-warp loss around
.49 is also found on the identical-image self-pair: stabilization and low-texture
windows dominate that floor, not PNG quantization. A low loss floor alone does not
prove a geometric optimizer defect; diagnose its map-dependent effects independently.

Focused affected-suite verification:80tests passed before the next selection change.
Next discriminators: cross-resolution output selection by the common full objective
(without changing optimizer trajectory), and fewer resets/longer stage solves under
the SAME total gradient budget. Image landmarks remain evaluation-only. No third
regional patch-size sweep or unrelated alternative mechanism is opened.

## Reset allocation and image-generated correspondence discriminator

Question: does cold-resetting a stage every5Adam steps limit coordinated capture?
Exact experiment: SAME known rasters/geometry/objective,300total gradients, change
cycles6/inner5 tocycles2/inner15 (joint controls twice cycles). Falsifier: no map or
objective gain after counting ALL evaluations/time. Result: analytic query RMSE
shear12.219->6.067px,rotation7.819->4.460,coarse-fine3.537->2.358. Radial improves2/3;
F1/F2 worsen all6. All12savedmaps valid,zero failures;362 rather than482total E calls.
This is a schedule/convergence finding, not proof of uniform method superiority.
Real matched P1 short/long300 runs are underway; no label-driven iterate selection.

Question: is there independent image-generated information about displaced regions,
or do proposed sparse matches only cover stationary texture? Reuse installed frozen
[SuperGlue](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf)
and [DHR](https://github.com/MWod/DeeperHistReg) code/weights; do not train a new matcher.
SIFT has3/4/11retained real matches with very narrow kidney coverage. Global-similarity
RANSAC SuperGlue gives114/140/68retained realmatches; knowntexture retainedmatches
have meanerrors .62/.45/.74px, but only63/19/62 cover>5px true movement. SIFT rotation
matches coverZERO>5px movement. Those are posthoc diagnoses, not ground-truth filters.
Global-affine RANSAC can reject genuinely nonrigid matches; a raw-confidence probe
now preserves the original match assignment, records its exact affine and .5pixel
coordinates, and reads NO manual label/maptruth. Seven focused tests pass.

Potential next decisive test, only after raw accuracy/coverage diagnosis: common
robust machine-match objective added to existing image/shape/strain evidence. With
fixed source q_j and aligned-moving p_j, complete moving error is
512 A(f_Y(q_j)-p_j). Define rho(r)=sqrt(1+||r||²)-1 with r=error/8px and average
with fixed normalized matcher confidences. This is a conventional robust landmark
energy on MACHINE-generated points, not manual anatomical training supervision.
Predeclare one common weight and all eligible failures, compareagainstweight0; no
new affine is fitted and no competitor's dense output is loaded. Falsifier: weak
coverage, inaccurate raw correspondences, or image-match gains without true map/
anatomical gains. No claimed novelty for SuperGlue or robust point fitting itself.

Git milestone298e433 pushed;83affected tests pass at that milestone. Research goal
remains active; modelcapacity interrupted ONE bounded builder followup, not the
remote computation or research. No installedskill/port/sharedjob was changed.

Stage30 allocation (cycles1/inner30, SAME300gradients) further improves all6coordinate
fits: analytic shear1.478px,rotation3.437,coarse-fine1.974. F1/F2 again worsen all6.
Every stage still gains objective during its last5steps; no gradient-magnitude history
was recorded. Shear/coarse-fine finalE fall BELOW their targetE, reminding us that the
synthetic truth is a feasible comparator, not the objective's global minimum.

The corresponding REAL P1 pyramid stage15 test fails to transfer this advantage:
Histo coordinate means3.37/2.80px,lesion7.76/8.38,kidney5.66/8.52, versus affine
2.33/4.90/6.12. F2 kidney3.35is useful, but othercases do not beat baseline consistently.
This is strong adverse evidence for coarse cross-stain image capture; fewerresetssolve
one R2bottleneck but are not by themselves a solution of R3. No manual-label selection.

RawSG knownmatches cover322/316/317points, means .660/.676/.773px,95--97%within2px.
There are142/97/119matches with>20px TRUE movement, versus much weakerglobal-RANSAC
coverage. Truth was only loaded AFTER extraction. This justifies one COMMON optional
robust point term λ=.1,κ8px, with static original-moving-domain exclusions reported.
Independent checker uses different NumPy bilinear/barycentric formulas and FD:
values agree<5e-16,VJP<5e-10, and point term is identical across16/32rasterpyramids.
Twenty focused point/extraction tests pass;56affected app/known/point tests pass.
Small actual257 GPU smoke passes before matched medium matrices. No ground-truth
or DHR-dense teacher is introduced, and no new matching network is trained.

Temporary-file correction: three coordinator pytest directories from this turn
mistakenly used Windows' default C: TEMP. Exact current-test targets3359/3360/3361
were verified and moved intact to D:outputs/relocated_test_temp; older tests untouched.
Subsequent tests explicitly set D:TEMP/TMP/basetemp and disable Python bytecode writes.

## T+3.7h: machine evidence helps, but real competitiveness remains unresolved

Known SAMEstage30/300-gradient matrix, λmatch0->.1, analytic query RMSE .824/1.209/.717px
versus1.478/3.437/1.974, all12methods/targets improve query AND raster errors. All300
gradients/332complete E calls,zero failed trials,allbinarycertificates valid. Raw/static
eligible322/322,316/316,317/317; no point exclusion. Analytic optimizer4.76--6.15s;
matcher setup/inference is separate about1s/pair here, not hidden in a zero-cost encoder.
Sparse machine-point fitting below the truth's pointE reflects subpixel matcher noise,
not an exactly reproduced teacher. Unaugmented image/prior E worsens for F2 despite
better correspondence, explicitly saved; augmented totals cannot be compared toλ0
totals as if the objective were unchanged.

REAL matched P1(ac),same300-gradient stage30,pyramid,shape,init,boundary,precision,
only λmatch changed. Mean TRE512canvas pixels (native baseline unchanged):

| Development specimen | Radial λ0->.1 | Analytic λ0->.1 | F1 λ0->.1 | F2 λ0->.1 | Native DHR |
|---|---|---|---|---|---|
| HistoReg | 3.193->1.849 | 2.516->1.587 | 1.352->1.179 | 1.198->1.182 | .943 |
| Lung lesion | 7.606->5.423 | 8.015->5.082 | 4.538->4.453 | 4.651->4.518 | 4.347 |
| Rat kidney | 9.022->4.990 | 11.009->5.259 | 3.878->3.698 | 3.622->3.388 | 3.432 |

All12real runs complete300gradients withzero failures and certified actualP1outputs;
allrawmachinepoints125/192/111remain statically eligible. Allsharedmanual IDs77/78/69
are scored. Coordinator's new read-only scoring wrapper stores explicit CSV/map/affine
provenance; it does not invoke optimization or select iterates. Analytic kidney p90
falls28.108->11.297px but remains worse than native7.110. F2 kidney mean3.388is close
to native3.432, yet costsabout15s vsnative~2s. Coordinate optimizersabout6--7s at~241MB
peak do not establish competitive accuracy. Matcher confidence is NOT ground truth.

Decision: preserve the meaningful known-texture coordinated result; do NOT rename it
real-data success. Next examine EXACTnative preprocessing/pyramid versus raw-NCC
approximation, and source-compatible P1 layer scaling. No more adjacent patch-size
or safety-coefficient trials. Focused affected milestone suite103tests passed; existing
legacy test archive is not repeatedly rerun. Goal remains active until real24h window.

## T+4h: native residual regularization and isolated frozen preprocessing

Research question: does weak residual regularization or raster preprocessing
explain the real/known transfer gap? Native installed source was traced independently:
affine-prewarp source, optimize ZERO-initialized residual at each pyramid level,
regularize that residual, then compose with frozen affine AFTER optimization.
It does NOT regularize the total affine-composed field. For the SAME physical
affine residual r(q)=Cq+t, native diffusion(2r)=||C||_F^2, our nodal strain
=.5||C||_F^2. Thus native alpha1.5 corresponds to lambda3 on this fixture.
An earlier factor2*(S/(S-1))^2 compared the same ARRAY reinterpreted on
different coordinate grids; it is NOT the same physical-field calibration.
The independent nontrivial-affine fixture records .0046 residual, .00498853
A-metric increment, and .09014053 total-field diffusion, distinguishing them.
One paired lambda.05->3 intervention is running on known and real matrices.
Different quadrature/boundary and image objectives still preclude equivalence.

Frozen native PIL/normalization/grayscale/CLAHE extraction now calls the actual
installed loader and preprocessing function. Independent pipeline capture is
bit-exact on L/RGB fixtures and a real RGB512 pair; focused helper tests6pass.
Sigma.1 gives kernel1 at ratio1: this is EXACT identity, not a meaningful blur.
Undefined constant-channel normalization and unsupported RGBA are reported,
not repaired. A separate preprocessing option preserves the ORIGINAL inverted
grayscale foreground mask and original moving coordinates. Default evidence is
unchanged. Fixed matches, raw posthoc raster errors and all manual landmarks
remain unchanged. It does not claim native NCC/pyramid/free-boundary equivalence.
Falsifier: preprocessing/regularization yields worse or unchanged held-out
correspondence despite decreasing its own objective; save that negative result.
Native raster variant will not be combined silently with new geometry or masks.

### First calibrated-prior results (development, not independent confirmation)

All settings are the previous stage30/300-gradient MIND+rawSG .1 matrix, changing
ONLY strain weight .05->3. Actual P1(ac) real outputs, mixed precision, best-full
image-only selection, fixed boundary and ALL evaluation landmarks are retained.

| Specimen | Radial mean .05->3 | Analytic mean .05->3 | F1 mean .05->3 | F2 mean .05->3 | Native DHR mean |
|---|---|---|---|---|---|
| HistoReg | 1.849->.878 | 1.587->.801 | 1.179->.815 | 1.182->.796 | .943 |
| Lung lesion | 5.423->3.769 | 5.082->3.689 | 4.453->4.000 | 4.518->3.704 | 4.347 |
| Rat kidney | 4.990->2.506 | 5.259->2.450 | 3.698->4.073 | 3.388->2.574 | 3.432 |

Analytic p90=1.585/7.133/5.007px, optimizer6.39/6.68/6.54s, peak239--243MB.
All12real runs300gradients/zero failures, allactual exported certificates valid.
Native takes about2s and is a DIFFERENTobjective/nohardcertificate. This is a
promising DEVELOPMENT result, not patient generalization or speed superiority.
F1 kidney becomes worse; don't turn a method-dependent prior effect into a
universal statement. The coefficient was derived BEFORE these labels were scored.

Known-texture Q1 query RMSE exposes bias: analytic shear .824->.533px,
rotation1.209->7.719px, coarse-fine .717->.769px. Radial2.886->.915,
1.989->7.853,.983->1.354. At truth the strong-prior rotation objective is
GREATER than the fitted wrong map; this is an explicit accuracy/prior conflict,
not inability to decode a legal rotation. All12knownmaps remain legal/300gradients.
Keep both objectives; don't select a different lambda for each evaluated target.
Next isolated native-preprocessing MIND matrix uses lambda3 for allmethods/cases;
only frozen features change, ORIGINALmask and rawSG stay unchanged.

## T+4.5h: evidence isolation, dense layer scaling, native topology

Observed: calibrated residual prior helps real development, but biases legal
rotation; frozen native preprocessing makes little difference under MIND.
Two causes: native raster contrast versus descriptor/prior mismatch.
Smallest test: native-preprocessed NCC7 versus native-preprocessed MIND under
the SAME lambda3/point.1/stage30/geometry/mask; known-texture test also retained.
Decision: no new safety formula; separate the image objective from geometry.

Real analytic meanTRE: rawMIND .801/3.689/2.450; native-preprocessed MIND
.808/3.713/2.468; native-preprocessed NCC .794/3.936/2.945. NCC is not a
consistent improvement and costs6.36--6.94s with215--218MB allocated peak.
All12NCC maps complete300gradients/zero failures and actualvalidcertificates.
Native-pipeline masks, recursive pyramid, affine prewarp and free boundary
are STILL different; this is an isolated evidence comparison, not exact DHR.

Same-function actualP1 dense geometry benchmark, one nonregular real257anchor
with qmin.00983719, uniformintegerrefinement to513/1025, float64, batch1,
RTX A6000, warmup1/three repeats. The proposed .02sin^2(pi x)sin^2(pi y)
horizontal field has identical physical units and boundary0. Query agreement
afterrefinement<=2.22e-16. Proposal-only VJP uses same random upstream pergrid;
coordinatedscalar vsF1/F2two-component parameter counts differ. Times include
decode/interpolation/geometry constraints, exclude images/setup/refinement.

| Nodes | Method | Forward/VJP milliseconds | Motion/request RMS | Total peak allocated MiB |
|---|---|---|---|---|
|257^2|Radial|3.918/2.098|.3860|46.07|
|257^2|Analytic|4.070/2.379|.5972|46.08|
|257^2|F1|4.570/11.661|.2434|97.85|
|257^2|F2|7.663/24.662|.2488|176.86|
|513^2|Radial|4.359/2.450|.3860|184.18|
|513^2|Analytic|4.235/2.406|.5972|184.18|
|513^2|F1|7.536/14.528|.1554|384.23|
|513^2|F2|13.893/29.131|.1856|718.33|
|1025^2|Radial|12.434/4.182|.3860|736.26|
|1025^2|Analytic|12.608/4.033|.5971|736.26|
|1025^2|F1|13.114/44.031|.0934|1542.33|
|1025^2|F2|27.010/93.397|.1218|2837.27|

All12outputs actualqmin>.001/finiteVJP. At1025 coordinatedresident72.08MiB,
increment664.18; F1resident96.04/increment1446.29; F2resident184.90/increment2652.37.
F2coldconstructor2.124s excluded. Identity infinitesimal calibration is NOT equal
finite motion, and kernel timing is NOT full-registration runtime. No new neural
training or low-memory million-query image claim follows from these numbers.

Independent native audit corrected a tempting frame error: MHA stores PIXEL
displacements, although native internal tensors use[-1,1]displacements. Native
actual512center table usesY=((j+.5+d_x)/512,(i+.5+d_y)/512), trimdomain
[1/1024,1023/1024]^2, no endpoint padding. All6native maps have negative AC AND
BD faces; some background zeros are exact. Static original-mask AND-tissue
negativecorner counts real25/54/1191; known17/28/34. Excluding32outercellbands
stillleaves25/46/22 and17/28/26. Histo boundary has37exact nonadjacent intersections;
otherssimple. Sixnegative minima independently recomputed by coordinator's
Fraction homogeneous3x3 area formula AND exactsavedpixel-frame reconstruction.
These results qualify topology, NOT anatomicalTREfailure. No repair is used.
Native remains an accuracy/runtime baseline, not a guaranteed-layer output.

## T+5h: fixed-query caching, objective tradeoffs and first-order adjoint

The authorized window is still 2026-10-01 11:26:23 UTC through
2026-10-02 11:26:23 UTC. About nineteen hours remain; the improvements below
do not constitute completion, independent anatomical confirmation or CNN training.

Research question: reduce repeated work while keeping the SAME actual P1 function
and complete objective. Frozen source-triangle indices and barycentric weights
replace repeated source-query lookup, not moving-query composition. Thirteen
independent value/transpose/finite-difference tests pass, including batch,
non-square grids, edge queries and rejection of query gradients. Application
integration keeps original masks, affine, frozen matches and geometry checks.
An independent review found no changed evidence term; affected clean-env suites
passed61tests. The new output-slack-only calculation reuses reference determinants
WITH their graph, eliminating zero-proposal change calculations. Coordinator
independently reran38update/patch tests; all six public output fields and
current/proposal/reference gradients match the original helper in new regressions.

| Scope | Existing -> frozen | Qualification |
|---|---|---|
|512^2 fixed queries,257^2 vertices,B1,float64,A6000|forward .6394->.1874ms; VJP .7829->.2134ms|sampler ONLY, warm3/repeat10|
|Same query table,B4|forward1.6728->.4583ms; VJP1.2428->.5020ms|not complete registration|
|Large B1 total allocated peak|74->48.017MiB|resident increases11.025->20.017MiB|
|Complete analytic development runs, three cases|6.21/5.66/5.56s; peaks217.94/216.89/218.99MiB|one paired run each; not a robust speed claim|

Cache setup is counted separately, all pyramid caches occupy16,760,832 bytes,
and their storage counts in total allocated peak. Small4096-query cache increases
peak slightly; it is not a universal memory reduction. Complete cached meanTRE
.800718/3.689299/2.450187px matches prior uncached results up to numerical
optimizer/CUDA differences. All twelve cached runs complete300gradients,
zero failures, and valid actual exported-map certificates. A four-pair same-
process AB/BA registration timing replay is now running without evaluation labels.

Known-texture native-preprocessed NCC under the SAME strong prior improves
analytic queryRMSE to .54/1.33/.50px for shear/rotation/coarse-fine; MIND gives
.51/7.62/.66px. In real specimens NCC is NOT consistently better than MIND
(.794/3.936/2.945 versus .808/3.713/2.468 meanTRE). Therefore an image-objective
and deformation-prior mismatch explains some errors independently of decoder
expressivity; no per-evaluation-target coefficient selection is introduced.

Target-free recording saved accepted maps for all twelve lambda3 development
runs; manual landmarks were read only by a separate offline scorer. At every
prefix, the selected map minimizes the COMPLETE full-resolution image objective
among initialization and accepted maps, NOT landmark error. Offline time until
that image-selected prefix first matches native final meanTRE is roughly
2.2s for analytic lesions/kidney and6s for histo. These are descriptive curves,
NOT a deployable landmark-based stopping rule, and include CPU snapshot overhead.
F1 kidney never reaches the native threshold. Final means agree with the prior
matrix. More accurate native timing comparisons require consistent cold/warm
setup, not substituting these offline thresholds for an algorithm.

A full512 cached MIND+point+shape trial on a real257 rat-kidney anchor has
median decoder-forward5.719ms, evidence-forward3.147ms and combinedVJP6.514ms
(ten warmups/ten repeats, float64 geometry/float32 evidence). Split-chain
derivatives agree within2.08e-17; separate decoderVJP2.045ms suggests geometry
is approximately half the cost. The split is diagnostic, not perfectly additive.
Total allocated peak136.664MiB, resident49.273MiB. This small proposal has
scale1 and does not stress an active safety bound. Thus geometry engineering is
justified, but cannot alone imply twofold complete-application acceleration.

FORMULATION Section14 now states the candidate-only explicit first-order adjoint,
full current-Y and latent derivatives, exact maximum tie averaging, clamp-zero
convention, analytic branch equality and scope limitations BEFORE implementation.
An independent mathematical context checked signs/unique constraints/ties/zero
proposals and finite differences. Builder implementation is isolated; it will not
enter application benchmarks until coordinator checks pass. No sliding-boundary,
reference-gradient or higher-derivative support is claimed by that new API.

Bounded independent-cohort reconnaissance did NOT obtain a new usable labeled
pair: ACROBAT training lacks paired evaluation landmarks; validation targets are
hidden; HyReCo requires very large login-mediated archives; the ordinary public
Warpy archive probe returned403. No bypass, submission, purchase or massive
download was attempted. Existing three cases remain DEVELOPMENT evidence; this
limitation does not stop algorithmic work or justify a generalization claim.

Counterbalanced replay completed: rat-kidney analytic, SAME output-slack-optimized
decoder on both paths, one full warmup for each cache variant, four measured AB/BA
pairs. Median optimizer5.73448->4.99883s (12.8% reduction); reported elapsed
including loading/features/export/certificate5.83366->5.10373s. All ten runs
finish300gradients/zero failures/valid certificates; final complete objectives
differ by less than4e-8. This supports a modest warmed speed benefit, not a new
algorithm or a cold-start claim. Optimizer-phase peak allocated memory
240,536,064->214,782,464 bytes; setup temporaries are outside that peak interval.
Independent checker confirmed configuration preservation, no label reads and
no surviving GPU tensor ownership. Historical end_to_end_seconds stops before
final shape diagnostic/report writing; future timing replays additionally measure
the complete optimize() call. Coordinator's combined affected suites102pass.

## T+6h: actual P1 recovery and two distinct acceleration scopes

Known-image extension keeps prepared Q1-generating PNGs/matches/truth unchanged,
but explicitly optimizes and evaluates P1(ac) estimates. No Q1 reinterpretation
of a fitted P1 map enters query or raster metrics. Independent numerical/scoping
review passed; the mocked optimizer-interface test now exercises all three
interpretations, forwarding cache flags while never passing truth. Twenty tests
pass. Actual P1 analytic queryRMSE=.542089/1.335958/.502327px for the three
targets, all twelve methods/targets300gradients, zero failures, valid certificates.
Same-target-vertex P1-versus-Q1 discrepancies=.000493/.001180/.000483px are
reported separately, NOT best-approximation lower bounds. PNG truth residual is
the generating-map quantization residual, NOT a proved minimal image loss.

Candidate-only full-current-Y/proposal adjoint: retained active-row storage
avoids full constraint trajectories. The first local-autograd-stencil variant
is slower when bounds are active at257/513; retain that negative measurement.
Replacing ONLY local stencil graphs with direct q1-edge/triangle differentials
now gives the following same-process candidate-only operator comparison:

| Float64 grid,B1,A6000, default checks on BOTH | Existing -> explicit manual forward+VJP ms | Existing -> explicit peak MiB |
|---|---|---|
|257^2, active radial|10.639->8.153|52.05->26.02|
|257^2, active analytic|11.249->8.550|52.05->26.02|
|1025^2, active radial|25.437->13.823|816.31->400.27|
|1025^2, active analytic|25.661->13.943|816.31->400.27|

Both Y and z require gradients; same integer-refined nonregular anchor and
upstream, fresh independent output corners, warm3/repeat10. All nine manual
benchmark cases have bit-identical candidate values; relative derivative L2
errors<=1.75e-16(Y)/1.02e-16(z). Absolute errors/RMS scales are also saved.
Unique active constraint per batch; many-tie speed NOT tested on GPU, while
CPU >4096-tie correctness is tested. Retained unique storage, including input
Y/z, falls432.031->24.047MiB at1025. First-order/fixed-reference/fixed-boundary
candidate-only scope remains; no diagnostic-gradient or higher-order claim.
Coordinator independently read the direct stencil and reran82affected tests.
No Triton dependency was needed. This is NOT full image-registration speed.

Measurement correction: the first saved-tensor probe accidentally retained
differentiable tensors in hook handles, producing reference cycles and inflated
resident/peak measurements. It is preserved and explicitly marked INVALID for
memory. A failing lifetime regression led to detached storage-only handles;
all corrected CUDA probes return EXACT resident baseline after counting. Corrected
and original numerical/timing records remain separate, not silently overwritten.

Constant-anchor stage cache is a DIFFERENT scoped engineering optimization:
cloned fixed Y/reference plus unnormalized determinant slopes computed once per
stage; no trainable-anchor differentiation claim. All public proposal-dependent
diagnostics and actual output checks remain connected. Independent11core tests,
33integration/profile/timing tests passed; coordinator44affected tests include
two-stage actual-image optimizations with identical exported maps and traces.

Complete rat-kidney analytic comparison, frozen P1 cache on BOTH, ten full runs
(two warmups+four AB/BA measured pairs), SAME image objective/300 gradients:
median optimizer4.88570->3.58688s; complete optimize() call4.99664->3.66890s.
All runs zero failures/valid saved certificates. Stage cache constructor costs
are INSIDE optimization time. Optimizer-phase peak216,908,800->219,011,072bytes
slightly INCREASES; this is speed engineering, not a memory saving claim. Each
cache retains approximately10.5MB at257. Component trial timings alone varied
under shared-host scheduling, so full same-process repeated comparisons take
precedence. Only one development specimen has this repeated backend comparison
so far; other specimens and larger complete-instance scales remain to be measured.

Important diagnosis: in the calibrated MIND/point/strain3 analytic development
matrix, all310evaluated trials in EACH of three cases have scale1; maximum gauges
about.52/.49/.70, belowtheta.95. Thus the guard did not limit these trajectories.
It still guarantees feasibility for other latent proposals, and other active
benchmarks exercise its nontrivial derivative. Do not attribute remaining
development error solely to conservative step scaling or claim a guard-free
network would always be legal. Independent-cohort and image-to-neural training
evidence remains absent; the approved phase's main results are instance optimization.

## T+6.5h: repeated three-case acceleration and full latent cascade

Two additional same-process four-pair AB/BA comparisons complete the declared
three development specimens. Frozen P1 queries and stage budget remain unchanged;
constructor costs are inside optimization. Complete optimize() calls include
loading, setup, final diagnostics and serialization, but not process startup.

| Development specimen | Existing complete call, median s | Stage cache complete call, median s | Existing/cache optimizer-phase peak MiB |
|---|---:|---:|---:|
|Histo CD4/CD68|4.45033|3.65558|206.83/208.00|
|Lung lesions|5.48506|3.82141|204.85/212.85|
|Rat kidney|4.99664|3.66890|206.86/208.87|

Each specimen has two warmups plus eight measured runs; all thirty full runs
complete300gradient steps with zero failed trials and valid saved certificates.
Warm complete-call savings17.9/30.3/26.6percent are engineering observations,
not independent-cohort anatomical evidence. Stage caching slightly increases
resident/peak memory; no memory-reduction claim. All maps/reports are preserved
on D, selected compact timing records versioned. Anatomical equivalence requires
separate scoring/map comparison rather than lower objective values alone.

Multilevel decoder now accepts one(B,2,L-2,L-2)latent tensor at EVERY level and
an initial(B,N,N,2)map, returning a final vertex table on that same material mesh.
Levels17,33,65,129,257 (plus513,1025 for1025controls), bounded raw proposals,
horizontal then vertical updates; no intermediate geometry detachment. Finest
latent equals control size:172,618 and2,787,918 scalar parameters, respectively.
Full initial-map and ALL latent first derivatives agree across ordinary/manual/
checkpointed paths. Coordinator independently read implementation and timing;
tiny finite differences, non-square batch2, zero and float32-chain tests pass.

| Float64,B1,A6000 geometry-only cascade | Ordinary AD | Explicit manual | Checkpointed manual |
|---|---:|---:|---:|
|257controls,10stages, forward+VJP ms|73.09|44.29|69.76|
|257controls, total peak MiB|300.60|42.09|34.69|
|1025controls,14stages, forward+VJP ms|366.70|163.16|309.39|
|1025controls, total peak MiB|6488.76|736.84|555.15|

Warm3/repeat10, same calibrated real anchor with initial qmin=.20129084,
deterministic synthetic smooth/noisy latents, actual rounded checks at EVERY
stage. Values bit-identical; maximum full-variable relative gradient error
4.40e-15. All six detached saved-storage probes release handles and return
allocated memory to resident baseline. Setup/reference clone~.53-.79ms,
input leaf copies, CPU exact refinement and transfer separately recorded.
Active analytic constraints0/10 at257 and2/14 at1025; final normalized qmin
.09576993/.00110285. The latter is close to extra eta=.001, so this is not a
uniformly comfortable distortion margin. Independent boundaries/corners pass.
Checkpoint saves~25percent of manual peak at1025 but takes~1.90times manual
forward+VJP time. No claim of checkpoint speedup, encoder training, registration
quality or fixed-depth universality follows from this geometry benchmark.

New actual-control coarse-to-fine variant is under verification (FORMULATION18).
Its objective ALWAYS materializes the final fine P1 map before ALL evidence and
priors. Independent review found a rounding edge case: a float32 coarse normalized
minimum.5000000596 can refine to exactly.5. With eta=.5 this is still positive
and homeomorphic but violates the additional STRICT margin. New initial/refined/
accepted-fallback margin checks reject it; no repair or tolerance loosening.
A deterministic mock-objective fixture also forces an earlier coarse-stage best
iterate and confirms exact final-size export. These are protocol tests, not image
accuracy experiments. Fine materialization/check costs may erase coarse savings;
the forthcoming full comparisons will decide.

Full actual-control comparisons are now complete: three specimens, two warmups
and four AB/BA pairs EACH, stage_cache and frozen P1 sampling on BOTH, same300
gradients and fine-grid objective. All30runs have zero failed trials and valid
saved certificates. Complete-call fixed/nested medians in seconds:
Histo3.56257/4.00583, lesions3.75995/4.05491, kidney4.07104/4.24504.
Optimizer-phase peak fixed/nested MiB208.00/236.34,210.88/235.86,210.98/235.83.
Thus this first fully materialized nested implementation is SLOWER and uses
MORE peak memory at257. This is retained as a negative result, not erased by
the positive geometry benchmark or by the exact-refinement theorem.

Separate saved-map manual scoring keeps ALL77/78/69shared landmark IDs. One
predeclared measured run (repeat0, not selected by labels) gives fixed/nested
mean canvas TRE px .800718/.807616,3.689299/3.700581,2.450187/2.450550;
p90 1.584832/1.585150,7.133147/7.209355,5.006613/4.994422. All ten saved maps
per specimen were scored, not only this displayed run. Means are slightly worse,
while kidney p90 is slightly better; no consistent accuracy benefit. For the
preceding stage-cache-equivalent comparison, separate scoring of all30 maps
confirms means effectively unchanged (.800718/3.689299/2.450187), including
small CUDA differences rather than an exact-bit anatomical-equivalence claim.
These repeated runs are not independent specimens/statistical replications.

Independent nested integration review:59tests passed, including the new strict
floor and forced earlier-stage-winner fixtures; coordinator's combined cascade/
nested/application/timing subset75tests passed. Cached refinement values match
dynamic diagonal-aware P1 evaluation exactly in fixtures; adjoints have normal
scatter summation differences up to~4.8e-7(f32)/1.1e-15(f64), independently
checked by finite differences, not incorrectly required to be bit-identical.
No default fixed-control behavior is changed. Next decision is based on a
component profile of actual refinement/fine-prior/check costs, not another nearby
coarse-grid safety-factor sweep. All present remote jobs completed; a bounded
new diagnostic profile is delegated on idleGPU7. Goal remains active.

## T+7.1h: optimization budget diagnosis and exact-prior derivation

Three development cases and four mechanisms complete both600-gradient schedules,
all24runs zero failed trials/valid binary outputs. Same evidence/physical rates/
initialization/fine257 control grid, fixed residual boundary, label-free best_full.
One coordinate cycle with60 inner steps is compared with two cycles of30 steps;
F1/F2 joint controls have twice the cycle count to match gradients, not geometric
pass counts or runtime. No anatomical labels enter either optimization.

| Analytic method, canvas TRE mean/p90 px | Original300 | One cycle,60steps (600) | Two cycles,30steps (600) |
|---|---:|---:|---:|
|Histo|.800718/1.584832|.785460/1.577051|.800718/1.584832|
|Lesions|3.689299/7.133147|3.683485/7.154315|3.689299/7.133147|
|Kidney|2.450187/5.006613|2.394437/4.866506|2.450187/5.006613|

Thus more inner optimization gives SMALL mean improvements, not uniformly
better tails. Repeating the image continuation fails to improve the selected
radial/analytic output on ALL cases: best_full remains first-cycle stage9.
All analytic scales remain1, so conservative topology scaling is not the
observed cause. Second-cycle initial coarse-image stages increase full objective
(.168639->.176216, .308407->.310256, .217678->.222312). This justifies the
bounded first-cycle-only IMAGE continuation intervention of FORMULATION19,
now running for R/A only: later coarse COEFFICIENT levels optimize full512
evidence. Independent config review and two new actual-image tiny tests passed;
coordinator50affected tests passed. Do not extend this schedule comparison to
F1/F2 without accounting for their different initial-continuation exposure.
The existing all_cycles default is unchanged. No dataset/test-label pivot.

Other controls remain informative: one-cycle-equivalent longer-inner F2 is best
Histo mean.776990 but costs25.45s versus analytic6.93s; its kidney mean2.532293
is worse than analytic2.394437. Two-cycle-equivalent F1 improves kidney4.0727
to3.4401 but still trails analytic2.4502 at19.72s versus7.37s. Objective rankings
do not fully match anatomy (e.g.F2kidney lower proxy thananalytic in longer-inner
run but worse TRE). All per-method/case results remain available, not just winners.

Actual nested-trial profile: synthetic legal anchors, real frozen512 kidney
Evidence, fullfine257 objective. It is NOT an optimizer trajectory comparison.
Nested17/65 add exact materialization+.4msVJP and ~.6-.7ms fresh fine check;
even nested257 duplicates the fine check. Fine Evidence forward/reverse dominates
roughly52-69percent of whole trials. Split/full coefficient gradients agree
relative<=1.23e-15. Separate parts-VJP medians image1.26ms,strain1.14ms,
shape1.93ms,match1.41ms,OOB1.23ms, versus combined4.66ms. These are NOT additive
cost attribution: shared graph paths and separate synchronization matter.
Weighted sum of part gradients matches combined relative2.03e-16. This is
enough to justify an isolated exact coarse-prior benchmark, not promise a full
application speedup. Original profiles and distinct perterm probes are preserved.

FORMULATION20's candidate exact quadrature is independently confirmed. Twenty-four
CPU checks use independent integer-cell barycentric refinement, two diagonals,
factors1..4, nonaffine convexquad/B2random grids/near-small determinants. Ordinary
shape value/full-Y derivative discrepancies<=1.95e-16/1.34e-15, strain<=2.1e-17/
1.25e-16. Near-det.001 shape cancellation amplifies ABSOLUTE errors (~1e-7value,
3.1e-4gradient), relativegradient<=1.7e-13. Actual production strain creates a
float32 reference before casting, so non-dyadic factor3 differs by up to7.44e-9
value/7.45e-9gradient from ideal-double-reference algebra. Do NOT silently change
the reference. Initial implementation is restricted to dyadic source/fine grids;
keep actual rounded fine checks and numerical comparisons. Reproducible checker
and JSON are saved. A bounded isolated prior implementation/benchmark is delegated;
no app integration or claim of measured whole-instance benefit yet.

First-cycle-only image-continuation experiment completes all six R/Acase runs:
600gradients each, zero failed trials, valid binary maps, best_fullstage19.
Analytic full objective is nonincreasing throughout cycle2 on all cases (it is
now the SAME functional at every second-cycle coefficient level). Scales remain1.
Analytic mean/p90 canvas TRE: Histo.779500/1.512891, lesions3.743505/7.071469,
kidney2.392966/4.908347; times7.02/7.44/7.01s, peaks~211MiB.
Radial means.779985/3.705722/2.405640, p90 1.506978/7.023034/5.120602.
This resolves the repeated-coarse-surrogate optimization issue, but NOT the
proxy/anatomy issue: lesions analytic objective improves.308407->.307126 while
meanTRE worsens3.689299->3.743505, even though p90 improves. The response is
not to silently pick schedules per case by labels. Preserve complete comparison
and distinguish objective convergence from anatomical progress. No new blind
cohort or clinical competitiveness evidence is obtained by these extra cycles.

Coordinator's current application/sweep/nested/profile/timing affected subset
passes68tests. The independent quadrature checker also includes directional
finite differences (maximum reported relative discrepancy4.51e-7); its production
reference mismatch remains restricted to the documented non-dyadic case.

## T+7.4h: decision card — direct coarse evaluation of the SAME fine functional

Question: Can nested P1 instance stages avoid a full fine differentiable map
without changing the image evidence, fine regularizers or output topology?
Exact claim: on dyadic nested grids with one unchanged diagonal, direct coarse
P1 pixel/point evaluation equals fine-refinement P1 evaluation in real arithmetic.
Section20 count-weighted priors equal the existing fine strain/shape functional.
Assumptions: no Q1 interpretation, no diagonal changes, original fixed queries,
same affine/features/masks/matches/weights, dyadic reference convention.
Falsifier: a complete objective or full-coarse-Y VJP disagrees beyond explained
floating-point accumulation, or actual rounded fine output fails its margin.
Smallest decisive test: both diagonals, NCC/MIND, matches+OOB, nonaffine coarse
maps, full value/gradient and directional finite differences; tiny pipeline then
same-process warmed AB/BA registrations. Fine output is still materialized and
freshly checked per trial without a gradient graph; accepted/full-best comparison
continues to use the original full Evidence. Prior: exact P1 refinement and
Section20 independently checked quadrature, not a new coarse loss or a proof
that the earlier negative nested application result is overturned.

Independent reduced-Evidence review identifies a rounding acceptance issue: choosing
the reduced winner and merely reporting full loss is insufficient. Implemented
comparison of original complete STAGE loss against original anchor, with fallback
if worse/nonfinite; full-resolution best_full remains separate and label-free.
Explicit bilinear-knot fixture explains one central-FD failure (image sampler is
nondifferentiable there), while coarse/fine selected AD gradients agree. Smooth
fixture passes strict full-objective FD. No production affine/features altered.

Isolated priors GPU7 result: all12dyadic AC/BD cases pass. Fine257 coarse17/65
only saves about.2ms forward+VJP (old4.03–4.20,new3.81–3.98ms); factor1 is
slightly slower and higher peak. Fine1025 coarse17 old17.59–17.71→3.57–3.72ms,
coarse65/257 old15.91–16.02→3.87–3.93ms. Total allocated peak old528–531MiB
versus new.136/2.075/34.005MiB, isolated prior graph only. Maxfloat64 relative
value9.79e-15/full-YgradientL2 2.98e-11; no bitwise equivalence. Old-then-new
sameprocess order is NOT AB/BA. First untimed257fine-check coldoutlier13.08ms
is retained, normalchecks~.7ms. F32thinBDshape gradient relative3.26e-5 retained.
This motivates full reduced-Evidence comparison, not claimed registration gains.

Question (next bounded optimization variant): does joint latent optimization of
one x-then-y coordinated stage recover interactions missed by long alternating
axis solves? Exact claim: each safe substep is legal, composition on the SAME
source vertex table stays legal and differentiated through intermediate Y.
Assumptions: frozen accepted stage anchor, x/ylatent jointly optimized, no detach
between substeps, no cached geometry of a changing intermediate map. Falsifier:
full both-latent VJP disagrees or matched objective budget has no useful benefit.
Smallest test: ordinary full AD and an independently FD-checked cached-first/
explicit-second decoder, then same3development cases with300gradient evaluations
and unchanged objective/initialization. Layer passes and wall time differ and
will be reported; this is block-coordinate versus joint parameter optimization,
not a new universal map family, matcher or proof of anatomical improvement.

Full reduced-Evidence AB/BA result (T+7.6h): three specimens, both warmups and
four counterbalanced pairs EACH, same300 gradients, fine257CONTROL/512query,
all30runs zero failedtrials/valid exported certificates. Whole-call full_fine→
coarse_exact medians seconds Histo4.35672→4.00609, lesions3.94320→3.94751,
kidney4.24468→4.04012. Thus about8percent/NOgain/5percent, not the large
isolated-prior speed ratio. Peak bytes245826048→242474496,
245836288→245566976,247297024→241362432, all optimizer-phase CUDAallocated.
Original-stage comparisons included, cache/setup included in wholecall. No
rounding fallback on inspected repeat0; acceptanceguard remains mandatory.

Offline independent P1 scoring of ALLten maps per specimen retains77/78/69IDs.
Predeclaredrepeat0 full_fine/coarse_exact meanTRE canvaspx
.8076159940494/.8076159940499,3.7005811666926/3.7005811667569,
2.4505503677188/2.4505503674917; p90unchanged to~1e-12. This is engineering
equivalence evidence, not new anatomical improvement/independent patients.
The reduced nested path is still not faster than the prior fixed-control
stagecached baseline overall at257; retain the earlier negative comparison.
A bounded1025CONTROL/512image replay uses the SAME five stages except last
control level257→1025 (17,33,65,129,1025), same300gradients, dyadic refinement.
This investigates scaling; no extra image information or accuracy conclusion
is claimed simply by increasing control resolution. Four AB/BA pairs running
on idleGPU6 after resource check. Coordinator80affected tests and17targeted
latest replay/reduced tests pass; independentchecker59focused tests pass.

Actual1025CONTROL scaling replay completes tenHisto runs,300gradients each,
ALLzero failedtrials/validcertificates. Whole-call full_fine→coarse_exact median
8.83987→7.05739s (~20percent saving), optimizer8.03113→6.23949s;
peak1944990208→1816321024bytes (~1.811→1.692GiB). Same512image queries,
levels17/33/65/129/1025, fullfine output has1050625vertices/2097152triangles.
Both methods include setup, actual fine checks, originalstage acceptance and
binarycertification (~.8s final overhead). This is ONEdevelopmentcase scaling
result, not a speed theorem or a competitive evaluation at1025image resolution.

Independent scoring ALLten maps: repeat0mean/p90full_fine.9102249745/1.68551849,
coarse_exact.9102180978/1.68551925canvaspx. Minor CUDA/scatter/optimizer path
differences are real: pairedmaxvertexunit discrepancy2.60e-5–4.44e-5 andRMS
6.42e-7–6.63e-7; do NOT claim identical optimizer trajectories from arithmetic
equivalence. More importantly both1025means are WORSE than nested257mean.807616
under this same300gradient budget/five-stage schedule; greater control density
does not automatically improve correspondence. No scale-up anatomical success
claim. Same data, original evidence and manual denominator, no casewise choice.
Coordinator's complete affected application/sweep/replay/reduced/priors subset
111tests passed11.24s. GPU6job finished and all maps/logs copied toD.

## T+8h: joint-coordinate optimization, gradient check and adverse outcomes

Joint frozen-anchor x-then-y module implemented. Ordinary two-layer AD versus
cached-first/manualFULLintermediate-Y adjoint agree on both latent gradients,
including unique-active directions, ties/tiny fields, float32B2 and non-square
meshes. Actual rounded margins for BOTH substeps are reported even validateFalse;
application rejects invalid x intermediate even if final is valid. Independent
checker41focused tests pass; author101core tests; coordinator75integration and
68known-interface/application tests pass. A deliberately injected zero x-margin
valid-final fixture confirms rejection/anchor retention. Zero backward calls now
report VJP time asNA/None rather than NaN/empty-median warnings.

Same three real cases, joint5levels*60steps=300gradient evaluations. TWOfields
and610safe substeps versus alternating310, plus305EXTRA no-grad diagnostics in
cached_manual. Thus not equalcompute. All6runs valid outputs/zero failedtrials.
Analytic mean/p90/time: Histo.775726/1.508054/8.23s,
lesions3.701831/7.111217/6.83s,kidney2.426533/4.899550/6.95s.
Comparedwith alternating300(.800718/3.689299/2.450187), means improve on two
cases, slightly worsenlesions; costs roughlytwice. Against600full-after schedule
(.779500/3.743505/2.392966), advantage remains MIXED, not a general winner.
Radialmeans.855653/3.731731/2.494603, time8.90/8.87/7.26s. All77/78/69manualIDs
scored offline; no labels enter accepted/full-best selection.

Untuned known-map control, original three prepared512histology targets, P1(ac)
estimates and unchanged Q1 generating function, all300gradients/zero failures:
analytic alternating/joint queryRMSEcanvaspx shear.53138/.52524,
rotation7.72023/7.78397,coarse-fine.76104/.52119. Joint costs~6.8–7.0s versus
~3.4–3.6s; no warmABBA timeclaim from these single optimization runs. Both use
same best_full objective selection, original frozenSGpoints+MIND+strain3+shape1e-4.
Allmaps/outputlogs onD. Rotation adverse result retained: jointlowerproxy
.188103→.186724 but queryerror INCREASES. Extra joint freedom/optimization
is not uniformly transferred to actual correspondences.

Next diagnostic question: edge-calibrated Adam steps impose a chosenO(h) scale
even though the coordinated decoder has no such intrinsic displacement cap.
At final257/1025 LR=.00025/.0000625 versus existing physical=.004 option.
Independent rank-one determinant bound (FORMULATION23) covers ALLfour corners:
g<=rho/(rho-eta)*max current-triangle amplitude-gradient norms ifQ>=rho>eta.
No unconditional mesh-independent depth/expressivity claim. Small discriminating
experiment: EXISTING physical-vs-edge calibration, fixed257R/A, same300gradients/
evidence/init/weights/boundary, logscales/gauges. Not a largerLRsweep, not aclaim
that finer gradients alone explain1025error; Adam normalization matters. Preserve
allcases, including iflargerproposalsactivateglobalworst-cell restrictions.

### T+8.4h bounded proposal-conditioning card

Question: does smoothing RAW fine-level coefficient proposals improve the
physical-rate experiment without changing geometry or image evidence?
Exact claim: zero-ghost stencil K=.5I+.125*(four-neighbor adjacency) is symmetric
SPD on finite interior arrays; applying K^4 BEFORE boundary padding and safe
decoding leaves topology controlled by the existing four-corner operator. Its
VJP is K^4. Assumptions: fixed zero residual boundary, finite f32/f64 arrays,
ordinary AD through the filter, unchanged acceptance objective and eta.
Falsifier: independent dense/VJP tests fail, or the fixed three-case trial has
no useful correspondence/time improvement. Smallest decisive test: four passes
ONLY on coefficient levels>=129, physical rate .004, alternating R/A, fixed257
control/512query, same300 gradients as the just-completed UNFILTERED physical
trial. No additional rate/pass-count sweep. Compare all cases/landmarks offline.
Prior work: standard explicit diffusion/damped Jacobi conditioning; no novelty
or solver-free expressivity breakthrough is asserted.

Independent checker: K^4 is algebraically invertible, but at interior255 its
smallest eigenvalue is about2e-18 and condition number about5e17. Consequently
unbounded exact range equality is NOT practical expressivity/conditioning.
Four passes smooth approximately one coefficient cell per-axis, not a fixed
physical width across scales; zero ghosts attenuate constants near boundaries.
Filtering need NOT decrease the current-geometry gauge. Record raw/filtered
proposal RMS and accepted displacement RMS to avoid confusing attenuation with
a demonstrated incoherence mechanism. No accepted map is smoothed or repaired.

Author28operator tests, coordinator80filter/integration tests and139affected
milestone tests pass. Independent checker108filter/joint/application tests pass,
verifies both joint gradients and original acceptance/certification remain.
Filter finite-check GPU synchronization and diagnostic RMS scalar transfers are
INCLUDED in forward timing, not presented as pure stencil costs.

The preceding physical-rate trial is NEGATIVE in meanTRE on all three cases:
analytic .996012/3.740509/2.601344 versus edge .800718/3.689299/2.450187.
All300 gradients/case valid, no failed trials;120/310 analytic trial scales
active per case, minimum .01077/.01252/.00738. These are ALL trial candidates,
not just accepted maps. Changing rate affects optimization as well as safety;
this does not alone establish the causal explanation of registration error.

Bounded filter result: all12 fresh none/four-pass R/A runs finish300 gradients,
zero failed trials and valid binaries. Radial meanTRE none->filter H1.02767->
.79010,L3.79364->3.70465,K2.57130->2.44338; analytic .99601->.86401,
3.74051->3.79752(WORSE),2.60134->2.35910. Analytic p90filtered1.60383/7.19930/
4.52240. Single-run optimizer seconds analytic3.96/3.50/3.44->4.35/4.13/4.03,
not warmABBA. No general advantage over existing edgebaseline is established.
Fine-level analytic minScale .01077/.01252/.00738->.12855/1/.07756;
maxGauge88.17/75.89/128.76->7.39/.906/12.25. Mean candidate-anchor RMS rises
roughly3.2--4.7x while filtered coefficientRMS falls. This is consistent with
alleviating incoherent worst-corner restriction, but coupled Adam trajectories
and amplitude attenuation prevent a unique causal claim. Accepted winning
diagnostics are retained separately from ALL trial extrema. No more filter
or rate sweep: modest utility and adverse lesion outcome are both preserved.

### T+8.5h decision card: rotation-aware regularization

Question: is displacement-gradient regularization excluding useful rotations
even when image/machine-point evidence supports them? Independent diagnosis on
known rotation: truth total .224444 > estimated .188103 despite truth image
.071159 < estimated .110828 and weighted match .000613 < .019766. Weighted
strain truth .152651 > estimated .057501. best_full selects TERMINAL stage9,
so an early-output-selection hypothesis is falsified. Truth diagnostic currently
uses Q1 generating map; declared P1 comparator will be separately quantified.

Exact candidate: replace ONLY the displacement-gradient penalty by
S(Y)=.5*mean_actual_P1_faces min_{R in SO(2)} ||J_t(Y)-R||_F^2, on the same
uniform material triangulation, retaining weight3 and existing shape/image/
matches/boundary/geometry. For detJ>0, nearest rotation is analytic and smooth;
no global solve/local-global ARAP iterations are used. Safety remains the
four-corner decoder, NOT this energy (ARAP alone does not prevent flips).
Assumptions: declared AC/BD P1, valid positive triangles, normalized rectangular
coordinates; interpret energy in residual-affine frame as before.
Falsifier: independent SVD/polar value or full-Y FD/VJP disagrees, or the bounded
same three real cases and known targets do not improve useful accuracy/time.
Smallest test: offline truth/current energy comparison first; then tiny operator
tests and one existing300gradient matrix, no weight grid search.
Prior work: standard ARAP/corotational distortion (SLIM2017 equations1--2,
https://igl.ethz.ch/projects/slim/SLIM2017.pdf); this is a conventional objective
ablation prompted by observed evidence/prior conflict, not novel geometry.

Read-only decisive comparison succeeds for rotation: on SAME stored truth/
alternatingA/jointA vertex tables, actual-AC ARAP prior .0259849/.0107068/
.0102986 replaces membrane .0508837/.0191669/~.018; new weight3 completeE
.149747/.162723/.162050 reverses truth/final ranking. Shear totals .116988/
.114927/.114305 and coarse-fine .101761/.104075/.100003 remain mixed. No
optimization or truth-based output selection occurred. Q1/P1 truth comparator
gap is only1.41e-7 objective and .00118px query discrepancy; neither explains
the7.7px rotation error. ALL rotation analytic scales inactive (maxg~.442),
so active safe scaling is NOT its demonstrated bottleneck. Independent checker
derives polar formula, actualface weights and collapse limitation (Section25).
Proceed one conventional ARAP objective ablation, not another LR/filter sweep.

ARAP author31independent-face/SVD/FD/second-derivative tests and coordinator84
affected tests pass; independent checker101operator/app/sweep/known tests pass.
Checker also supplies compressed-shear negative curvature example (Section25):
rotation invariance does NOT imply convexity. p1_arap rejects Q1 and coarse_exact
membrane quadrature rather than silently changing their mathematics. Known
Q1-generating truth is unchanged; additional same-declared-P1 objective comparator
is labeled separately. All pyramid stages receive identical chosen strain model.
IdleGPU6real12-run andGPU7known12-run matrices started after resource check;
no filter/rate change, allR/A/F1/F2 use shared prior3/MIND/machine-point evidence.

Both matrices complete all24runs,300gradients each,zero failed trials,allvalid.
Real analytic meanTRE membrane->ARAP: H.800718->.793449,L3.689299->3.657262,
K2.450187->2.351586; p90ARAP1.544649/7.080791/4.507429. These are modest
DEVELOPMENT gains, not independent validation. ARAP radialmeans.882463/3.700171/
2.328260; F1.808723/4.031140/4.028482; F2.787888/3.669633/2.394303. Single-run
analytic optimizer3.85/4.35/3.94s, peak225--229MB; F1~9s/341--343MB,
F2~15.5s/499--500MB. No warmed timing superiority claim from these single runs;
allactual257CONTROL/512query/P1ac/mixedprecision, same300gradient budget but
scalar/vector field and geometry-pass counts differ explicitly.

Known analytic queryRMSE membrane->ARAP shear .53138->.68548(WORSE),
rotation7.72023->3.39263 (~56percent lower), coarse-fine .76104->.59565.
RadialARAP .88251/4.04729/.87792. Known F1 20.69492/14.15570/15.45156 and
F2 14.10876/8.99728/9.30272 remain behind R/A at this budget. No new raster/
matcher/GTinputs, Q1-generating truth retained. Allsnapshots used ONLY posthoc;
known tests are warped textures of one existing specimen, not heldout patients.
RotationARAP finalE.152433>truth-table .149747 leaves optimization/evidence
questions unresolved; prior conflict was alleviated, not all errors solved.

### T+8.9h bounded evidence-evaluation card

Question: does descriptor TRANSPORT create an avoidable optimization bias under
large local rotation/stretch? Existing MIND-like pipeline compares Phi(If)
with sampled Phi(Im). In general Phi(W_F Im) != W_F Phi(Im), where W_F samples
the original moving raster through currentdeclaredmap and Phi is the EXISTING
eight-offset self-similarity descriptor. Independent read-only knownrotation
check: unquantized truth transported image cost .070843; recomputed-after-warp
image cost .008584 against the PNG fixed raster. Thus quantization alone is
not the transported-descriptor error. This is diagnostic, not general anatomy.

Exact candidate changes ONLY descriptor order: compare Phi(If) with Phi(W_F Im)
under unchanged mask/denominator/ARAP3/shape1e-4/frozenpoints/init/geometry/Adam.
Backpropagate through raster sampling AND all descriptor operations; never
cache/detach candidate descriptors. Assumptions: same P1 query convention,
zero raster OOB and replicated descriptor boundaries, originalmovingraster,
same offsets3x3patch scale1e-4. No learned descriptor/newmatching network.
Falsifier: independent operator/FD fails, time/memory is prohibitive, or one
predefined same3case and3known-target matrix yields no useful accuracy/time
tradeoff. Smallest decisive test: literal tiny reference computation and full-Y
FD away from bilinear/absolute/minimum knots, then same300gradient matrices.
No descriptor parameters or prior coefficients tuned per case.
Prior work: MIND's self-similarity descriptor is established (Heinrich2012,
https://pubmed.ncbi.nlm.nih.gov/22722056/); our eight-offset implementation is
MIND-LIKE, not a faithful new reference implementation or novelty claim.
Noncommutation is an operator fact; it does NOT prove this revised objective
will perform better on different stains/noncorresponding tissue.

Both descriptor-order matrices completed: all24runs,300gradients each,
zero failed trials and valid actual exported maps. Independent checker45tests
and coordinator60affected tests passed. After-warp analytic known queryRMSE
shear/rotation/coarse-fine .622750/1.417840/.549527 improves transport-ARAP
.685480/3.392633/.595647. But real analytic means .906167/3.730921/2.472741
are ALL WORSE than transport-ARAP .793449/3.657262/2.351586. All four methods
have worse meanTRE in every real case. After-warp analytic single-run7.68/7.05/
7.09s,296--299MB versus transport3.85/4.35/3.94s,225--229MB; concurrent jobs
and non-ABBA timing preclude precise speed claims. Decision: retain transport
ARAP as the current main objective; retain this negative real result, do not
continue descriptor tuning merely because the synthetic rotation improved.

### T+9.4h original-image resolution research card

Question: can actual original-image detail, rather than a512-thumbnail ceiling,
improve registration? Candidate: render1024 canvases directly from original
JPEGs at EXACTLY twice the stored512 resized dimensions and padding. With
u=((x+.5)*scale+padding)/side, doubling scale/padding/side preserves every
normalized source coordinate. Copy positive affine arrays and frozen normalized
machine matches unchanged. Label match prediction resolution512 honestly; no
new matcher or manual landmarks enter preparation or optimization.
Assumptions: integer scale factor; source dimensions agree with layout; target
resized dimensions do not exceed native source dimensions; same RGB/BILINEAR
renderer as before. Histo and kidney satisfy these; lesions' local originals
would need upsampling and are excluded from native-detail1024 validation.
Falsifier: normalized frame invariance, affine/match bit identity or provenance
fails; actual1024 evidence produces no useful time/accuracy tradeoff. Smallest
test: synthetic non-square padded source and source-mismatch guards, then kidney
and histo with257controls/1024queries, same300gradient budget, ARAP3 transport.
Point robust scale doubles8->16canvaspixels to preserve normalized penalty.
Report native-moving-pixel errors and512-equivalent canvas error, not raw1024
pixel numbers as comparable512 errors. Descriptor pixel footprint and image
continuation levels change; this is NOT a pure same-functional timing test.
Prior work: conventional image pyramid and pixel-center coordinate conversion;
this is an evidence-resolution experiment, not a novel geometry construction.

Prepared both exact1024 families from native originals after focused33tests;
independent checker29tests plus constant-DHR-translation unit test passed.
First remote sweep stopped before optimization: Linux Path did not strip Windows
backslashes from transported raster basenames. Corrected cross-host basename
parsing (still checks side and affine), retained failed log and started a new
named sweep. Cross-host point-loss/value/VJP invariance tests pass; combined48
focused tests pass. No input record, initialization, port or existing job changed.

### T+9.6h exact nested ARAP reduction research card

Question: can denser actual output controls avoid fine-grid regularizer AD while
preserving the CURRENT useful ARAP objective? On uniform globally aligned dyadic
AC/BD P1 refinement, each coarse source triangle subdivides into m^2 triangles
with identical affine Jacobian and total unchanged material area. Thus actual
face ARAP mean is identical to its coarse-face mean in real arithmetic. Shape
remains the existing exact four-corner count quadrature, not coarse shape mean.
Candidate: extend only ExactNestedP1Priors with explicit strain_model, default
old membrane; ARAP branch uses actual declared coarse P1 face ARAP. Root handles
application integration after independently checked value/full-Y gradient tests.
Assumptions: same global diagonal, square uniform dyadic source grids, exact
P1 refinement, positive current coarse faces, no detached changing vertices.
Falsifier: direct fine materialization+face computation or full-Y pullback VJP
disagrees; nonaligned/non-P1 inputs must not be silently accepted. Smallest test:
both diagonals/non-affine valid vertex tables, batch>1, factors1/2/4/8, independent
triangle/polar reference and finite differences. Rounded fine differences and
actual fine topology/margin checks remain explicit; this is not a certificate.
Decision changed: permits paired257/513/1025-control ARAP timings with identical
functional rather than reverting to less useful membrane for scale experiments.
Prior work: elementary P1 subdivision/integration identity; no novel solver claim.

### T+9.7h native-detail outcome and matched native baseline card

All8real1024runs completed300gradients,zero failed trials,valid actualP1outputs.
Same257control mesh, now1024query/image; machine points still predicted512.
Analytic mean512-equivalentTRE H .793449->.756767, K2.351586->2.287044;
p90H1.544649->1.617151 WORSENS, K4.507429->4.153198 improves. Radialmeans
.761823/2.303790, F1.739403/4.252492, F2.744972/2.406358. Native-moving-pixel
analyticmeans14.64588/5.20260. All77/69 manual IDs retained, no labels in optimizer.
Single-runanalytic4.22/4.72s,~520MiBpeak; query evidence dominates increased memory.
No precise speed claim, no independent specimens, no controlled same-footprint
descriptor comparison. Added image evidence helps modestly, not a breakthrough.

Question: are these native-detail1024 results useful against an executable
native DHR at the SAME1024 images and SAME affine? Candidate: generalize only
coordinated_dhr_common's declared image_side (default512 unchanged), guard square
unalteredpreprocessing/noinitialresampling/noextraDHRpadding, registration_size
image_side, initial_resolution sufficient to retain image_side. Native fivelevel
30iterationNCC/regularizer remains different; it is an application baseline,
not isolated geometry ablation. No new matcher, affine estimation or teacher.
Falsifier: supplied normalized affine/grid conversion differs at512/1024,
preprocessing silently resizes/pads, or saved field parameters use wrong frame.
Smallest test: mocked nativepipeline shape/affineidentity/translation, defaults
unchanged; then actualH/kidney1024 baseline and independentall-ID scoring.
Prior work: existing DeeperHistReg implementation, no new algorithm claim.

Independent checker recomputed all77/69 required IDs with literal barycentric
and f64 native-field interpolation: new257-control analytic512eq means.756767/
2.287044 versus SAME1024/nativeDHR .844177/3.614272. DHR wholecall2.46/2.39s
versus coordinated optimizer-only4.22/4.72s: no competitive-time claim. Native
saved1024Q1 fields have4532/6507 NEGATIVE corners and0/2 ZERO corners, in2362/
3120 cells containing negatives; neither has a hard topology guarantee. Exported
DHR MHA is pixel-unit displacement, not its internal normalized tensor; checker
caught and corrected that metadata distinction. Scorer tests include known
translation at512/1024 and a deliberate all-cell fold,20tests pass.

Actual1025CONTROL/1024query ARAP nestedABBA: all10maps valid,300gradients/zero
failures. Same-process warmed complete-call median11.07981->8.61442s (22.25percent
less), optimizer10.32498->7.85332s; allocated optimizer peaks2427569152->2294671360
bytes (5.47percent less). AllHmean512eq .7817015--.7817018, WORSE than257A .756767.
Paired component-wise nodal differences up to1.86e-5 normalized, despite almost
identical landmark means; real-arithmetic equivalence is not rounded bit identity.
Independent actualall18map sign/boundary recomputation succeeds, minimum normalized
fourcorner .00690158>.001. Root120affected nested/operator/app tests, checker98
operator and36app/DHR tests passed. Cost improvement is established here; denser
anatomical superiority is not. No amortized CNN training claim in this instance phase.

### T+9.8h diagnosed dense-scale locality card

New failure evidence: actual1025 final vertical trial has global analytic scale
.11307/gauge8.4016 despite edge-scaled Adam rate6.25e-5, whereas257 main stages
had largely inactive scales. Its dense-control mean error is worse, not better.
Question: can local independent exact AFFINE constraints help this NEW dense
active-scale regime without erasing the useful global coarse stages?
Candidate: preserve globalcoarse updates; only final dense-level proposal uses
four sequential nonconflicting coordinated patch passes with offsets(0,0),
(P/2,0),(0,P/2),(P/2,P/2), the SAME common direction and proposal. Existing
sin-squared patch windows vanish exactly on patch perimeters. Each pass has
independent per-patch radial/analytic scales, updates the SAME vertex table,
and next pass uses updated geometry with all gradients connected. No blend of
accepted maps, output resampling, geometry line-search, QP or inverse is introduced.
Assumptions: validfixed boundary anchor, evenP>=2 and enough gridcells, correct
material reference per level, unchanged fullobjective/margin/precision. Four
passes must be counted explicitly, not called equal-geometry work. Smallest
test: fullcorner recomputation, all-interior seam coverage, deliberate thinpatch
and full-Y/proposal FD; then one fixed-config1025H run against existingglobal.
Falsifier: connectivity/VJP/margin fails, or additional cost buys no meaningful
objective/registration benefit. This is a main-line support variant prompted
by actualdense conditioning, not a third257patch-size sweep after prior failures.
Prior work: existing disjoint-support coordinated primitive and sequential
domain-decomposition-style updates; no universal approximation/newnovelty claim.

Cascade author67 and independentchecker83focused tests pass; root97affected
tests pass. Deliberate invalid intermediate margin with valid final map is
rejected, and all4passes are counted. All-interior support requires per-axis
cellcount divisibleP/2; otherwise tails remain frozen. Complementary sin-squared
windows sum1 away from truncated edges and at most1 elsewhere, so there is no
automatic4x amplitude; defaulttrial1scales<=1. Pfixed has physicalsupportP*h,
not resolution-independent macro support.
First1025pilot configuration inspection caught an unintended CLI difference:
directcase defaults lr_calibration=physical whereas the comparison record used
edge. That pilot is retained but NOT a matched-support comparison. All300gradients
complete,496coordinatedpasses,valid output; finalE.182906 vsglobal.175765 cannot
be attributed to patch supports. Rerun uses explicit edge calibration .004.

Matched edge rerun differs from original1025 record only support flag/output:
300gradients,496passes,zero failures,allintermediate/final margins positive.
Hmean512eq .781702->.767542,p90 1.594350->1.576050; stillWORSE mean than257A
.756767. Objective .175765->.175117 improves, but corner shape .02986->.45425
and minimum normalized determinant .00100138 is very close to configuredfloor.
Final patch MINscale .001357 is NOT global multiplier: meanpatchscale .998969,
displacementRMS8.55e-5 vs global2.26e-5. Decoupling works, not uniformly good
conditioning. Cost is adverse: optimizer17.72s/end-to-end18.94s, peak6418621952
bytes (~5.98GiB) vsglobal~7.85s/2.14GiB. Do not promote ordinary patch backend
as a fast winner. Defaultmain stays global; no patch-size/rate sweep follows.

### T+10.2h patch-adjoint engineering card

Question: is the four-geometry ordinaryAD graph causing the measured6.4GB peak,
and can the EXISTING exact first-order ALL-Y/proposal stencil VJP remove it?
Candidate: reuse coordinated_explicit_vjp's active-row adjoint on each gathered
patch; scatter derivatives normally and keep changing intermediateY connected.
Expose candidate plus NONDIFFERENTIABLE numerical diagnostics from the SAME
forward invocation to avoid recomputing a second decoder for scales/margins.
Preserve old candidate-only API/default behavior. Optional patch/cascade backend
manual supports first derivatives of vertices/proposal only; no auxiliary
diagnostic gradients, trainable reference/trial or second-derivative claim.
No new safety formula, no map approximation or Krylov history is introduced.
Falsifier: candidatevalues/active-tie/full-Y+proposal VJPs/FD disagree; actual
intermediate check is lost; paired unchanged-config realcost/memory is no better.
Smallest test: ordinary/manual all4passes B2/non-square/bothmodes, uniqueactive
and ties, boundary/sourcepatchnormalization, guardunsupported derivatives; then
one matched1025 nativeH case before any ABBA. Geometry work remains496passes,
not silently doubled for diagnostic reconstruction. Prior work: already checked
local-stencil adjoint in Section17; this is memory engineering, not a new decoder.

### T+10.4h manual patch candidate implementation and first actual pilot

Builder135 focused tests pass; independentchecker139 pass8.92s with no blocking
defect; root application/operator/joint/ARAP affected95 pass13.33s. Old API and
ordinary default preserved. All-Y/proposal gradients stay connected through
four passes; diagnostics are explicitly nondifferentiable; trainable reference/
trial and second derivatives unsupported. One forward provides candidate and
numerical diagnostics, no second ordinary decode. A real float32 rounded floor
contact is retained as matched strict rejection, not tuned away.

After checking GPU6 was idle and shared GPUs/processes untouched, ran ONE
matched1025control/native1024H pilot: P32, same edge LR.004, same frozen evidence,
ARAP3/shape1e-4,300gradients/496geometrypasses,0failedtrials, saved certificatevalid.
Manual optimizer16.9306s versus priorordinary17.7181s; end-to-end18.1841s versus
18.9364s. Peak2146297344 versus6418621952 bytes (~66.6%lower). This is an UNWARMED
singlepilot; no time superiority claim. Objective .1751168185 versus .1751168524,
shape .45425377 versus .45425381: the adverse thin-conditioning result persists.
Candidate-based manual backend is optional, not selected globally by default.
Next: exact-config warm AB/BA ordinary/manual and independent all-ID scoring;
memory improvement must not be advertised as an anatomical improvement.

Independent saved-map scoring of all77H IDs gives mean512equivalentTRE
manual .76754214040 versus ordinary .76754214028; p90manual1.57604957963 versus
ordinary1.57604958024. No anatomy change from differentiation engineering.

### T+10.4h second native-detail specimen, fixed-support transfer question

Question: does the modest Histo1025 patch-support gain recur on the OTHER
available native-detail specimen, kidney, or is it a single-case benefit?
Compare global versus finalP32/manual on identical17/33/65/129/1025 controls,
native1024 rasters, frozen affine/machine points, same ARAP3/shape1e-4 objective,
300gradient budget, edge LR.004 and best_full image-objective selection.
No patch-size/rate/weight change and no manual-landmark input to optimization.
The pre-existing257control result is context, not an extra tuned candidate.
Falsifier: patch worsens all-ID anatomy/tails or merely improves proxy while
approaching floor and raising distortion; keep all eligible cases and failures.
Smallest test: one full global and one full manual-patch kidney registration,
actual saved certificate, independent all69-ID scoring. Single cold calls do
not establish timing superiority. This extends the already implemented main
support variant; it is not a second active alternative mechanism or blind test.

### T+10.5h actual dense results and focused mathematical decision

Warm SAMEPROCESS4 AB/BA pairs (two initial whole-run warmups excluded) give
ordinary/manual medians completecall18.72699/15.98874s, optimizer18.00509/15.25910s,
peak6412377088/2148538368bytes. Manual is ~14.62%faster in completecall and
~66.49%lower optimizer-phase allocated peak. All10runs300gradients/0failures,
actual saved certificatesvalid. Still slower than global1025~8.61scompletecall
and no evidence of anatomical change from changing adjoint implementation.

Second native-detail specimen: all69kidney-ID mean512equivalentTRE global1025
2.28789809 versus fineP32/manual2.28180269, p904.25500850/4.25459788; essentially
no gain. Global257mean2.28704418,p904.15319765 remains a cheaper competitive
reference. Patchshape2.90937 versusglobal.0733454 is adverse; global/patch cold
optimizer9.0222/16.9307s, both300gradients,0failures,validsavedmaps. Fixed-affine
range lower bound0 for all69IDs: this necessary bound does not explain error.

Decision: stop support-size/safety-rate tuning. A real GPT-6 Astra high context
read the actual application/formulation and identified an EXACT degeneracy:
when analytic clipping is active, g(tp)=tg(p) and T(tp)=T(p) along that ray.
Changing Adam to L-BFGS on the same raw latent would keep this radial null mode.
Investigate physical scalar-fiber optimization instead, preserving map class.

### T+10.5h research card: fixed-fiber feasible L-BFGS diagnostic

Question: is final-stage dense convergence constrained by the existing latent
chart/Adam combination, rather than inadequate resolution or unavailable anatomy?
Exact variable u is actual scalar nodal displacement along ONE fixed coordinate
direction e from a frozen incoming map Ybar; boundaries0. Y(u)=Ybar+u*e.
For allfour actualcornerrows, s(Y(u))=s0+A*u EXACTLY, with A fixed atYbar.
Preserve the analytic decoder's CLOSED reachable stage set A*u>=-theta*s0,
theta=.95, not merely the larger positivity domain. Starting0, keep strict
contracted slack theta*s0+A*u>0 by fraction-to-boundary. This is a constrained
instance optimizer diagnostic, NOT a new differentiable neural decoder or
claim of gradient through solver iterations. Existing decoder remains intact.

Use first-order compact L-BFGS history5 on physical u, skip unreliable curvature
pairs, descent fallback. For direction d use exact alpha_max=min_(A*d<0)
(theta*s0+A*u)/(-A*d), and alpha=min(initial_physical_trial,.99*alpha_max).
Image-objective Armijo backtracking is separate, finite-budget, evaluates the
UNCHANGED complete objective; no geometric line search/projection/foldrepair.
All rounded intermediate maps still checked. Fixedanchor/reference/e remain
constant within the coordinate solve; history reset at stage/direction changes.

Smallest decisive test: focused quadratic/fiber/curvature/float32-rounded tests;
then saved129-prefix finalx/y replacement on H/K at257 and1025, baselineanalytic
Adam versusphysical-fiber L-BFGS,30gradient evaluations/coordinate, capforward
trials and reporttotalcounts/time/memory. Same savedprefix WITHIN eachgrid;
crossgrid bitidentity is not assumed. Same image evidence/affine/prior/boundary
and best_fullobjective selection; no evaluationlabels until saved-map scoring.
This comparison changes chart AND optimizer and cannot isolate the two causes.
Physical-fiber gradient control may follow only if the combined intervention
shows useful improvement and that attribution matters. No learningrate sweep.

Falsifiers: tinyfeasible steps despite usefuldescent; excessiveArmijo/curvature
failure; noobjective gain; objectivegain withoutanatomygain; adverse distortion
or actualtopologyfailure. A failedmethod does not prove constrainedstationarity
(no tangentcone/QP solver). No guarantee of global nonconvex convergence. Prior
work is standard L-BFGS/Armijo applied to the existing exact scalar feasible
fiber, not a newtopologytheorem. IndependentAstra review will check implementation.

Independent saved-map checker additionally examined12actual1025maps: allfinite,
exactfixedboundary,50331648cornerdeterminants,0nonpositive/nonfinite; minnormalized
determinant .0010001746655. WarmH pairs agree in77-landmarkmean to5.54e-9pixels,
but wholemaps are NOT bitwise equal: maximum pairednodal vectordifferences are
.04001/.04404/.04087/.03915 native1024canvaspixels. Do not equate sparse landmark
agreement with entire-map equality. All69K landmarks independently reproduce
scoring within2.2e-13canvaspixels. This strengthens the measured engineering
conclusion only; no anatomical breakthrough or adjoint proof from savedscores.

### T+10.9h fixed-fiber implementation and actual four-case result

Physical scalar-fiber L-BFGS module29focusedtests pass; independentAstra initially
23tests plus3targetedfiniteguards pass. Root84affectedoperator/app tests and
83suffix/core/nested tests pass. Independent dense4pair inverse-BFGS action
agrees with two-loop recursion to5.55e-17. Derivedreciprocal/gamma overflow,
extreme direction norm and initialRMSunderflow corrected with explicit tests;
no constraint/margin tolerance relaxed. Wrapper34tests include realtiny solve,
allactualfailurecounts, correct16factor edgecalibration and1ULP materialization
differences. Exactold129nodes reconstruct the same actualbaseline transition;
frozenfinecallback equality is informational, not a false fairness prerequisite.

Checked GPU6/7idle; ran H257pilot then H1025/K257/K1025 ontheseauthorizedidle
GPUs, no otherprocess/portchanges. Every baseline andreturnedfiber output is
actuallycertified. All77H/69K manualIDs read ONLYafter saving, kept indenominator.
Native1024images/query side shared throughout. Baselinehere is NESTEDcontrol
notpreviousfixed257coarse-latent implementation; only currentwithin-row contrasts
are claimed. Finalx/y30gradientlimit includesinitial; directhasatmost29moves
percoordinate, whereasAdamhas30moves pluslasttrial; actualworkcounts retained.

| Case/control side | Baseline mean512eqTRE | Fiber mean512eqTRE | Baseline completeE | Fiber completeE | Fiber gradients | Fiber termination |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H/257 | .75581973 | .75454328 | .174978263 | .174495014 | 60 | both budget |
| K/257 | 2.29071518 | 2.28292079 | .226442168 | .225166896 | 60 | both budget |
| H/1025 | .78170155 | .79551918 | .175764610 | .176939103 | 32 | objective backtrack limit; minimum step |
| K/1025 | 2.28789814 | 2.29949922 | .228191741 | .230817092 | 34 | minimum step; rounded contracted margin reject |

257p90 H1.59524039→1.58616028,K4.21161219→4.21920088 (Ktailworse despite
tinymeangain). 1025p90 H1.59435009→1.57644553,K4.25500827→4.27122036.
No practical dense accuracybreakthrough. Directsuffixselection includesbest
precedingfull-objectiveprefix; it is not forced to choose its terminaliterate.
257fiber completeobjectivecalls64H/63K inclselection/final;1025calls43H/37K,
NOT60gradients completed. H257oneArmijohalving;K257none. No curvature skips or
descentfallbacks occurred. All stages/stops/rejectedobjective trials preserved.

Cold suffix times baseline/fiber H2571.2060/1.1202s,K2571.3068/1.3033s;
H10252.7467/1.4595s,K10252.7574/1.4406s. The apparently shorter dense solve is
EARLYTERMINATION, not a speed-to-accuracy win. Baselinepeak covers fullrun,
directpeak onlysuffix and separateevidence setup; cannotclaimmatchedpeakgain.

Diagnostic: K1025xordinaryslack remains~.0244412 while fixedtheta-contracted
slack approaches3.06e-14 andalpha~5.47e-12. K y rejectedactualcontractedslack
-6.34e-15 despiteordinaryslack~.004540855>0. Thus this is a numerical approach
to the ARTIFICIAL fixedstagepolytope face, NOT anactualphysicaltriangleflip.
Do not attribute it to the original eta bound or repair/relax it silently.
Do not claim stationarity: the direction may point out of an active face while
tangential descent remains possible. A focusedAstra adjudication is inspecting
thisactualtrace before any nextimplementation; no LR/rho/backtrack sweep.

### T+11.1h research card: bounded inverse-Hessian-metric tangent rescue

Question: can tangential descent bypass the ONE nearly active contracted row
observed independently in each of the four failed 1025 coordinate stages?
Keep the SAME frozen scalar fiber, theta=.95, objective, boundary and budgets.
With positive-definite L-BFGS inverse action H, gradient g and p=-H g, collect
rows B of A with contracted slack/(theta*s0)<=1e-6. Cap count at8; exceeding
the cap stops this diagnostic rather than spawning a general active-set QP.
For outward p, use d=p-H B^T (B H B^T)^+ B p. With independent exact rows,
B d=0 and g^T d<=0; a rank-deficient Gram requires explicit numerical rank,
tangent residual and descent checks. No unconstrained Euclidean projection
of an arbitrary quasi-Newton direction is claimed to preserve descent.
Recompute ALL exact affine bounds after correction; retain fraction .99,
ordinary Armijo and strict actual rounded geometry checks. Count extra H
actions, projection work and objective evaluations. No geometry tolerance
relaxation, post-hoc repair, neural-decoder or global convergence claim.

Smallest decisive tests: independent small dense SPD action/constraints;
single/multiple/dependent active rows; inward, outward, nonfinite, cap and
rounded-feasibility fixtures. Then ONLY H/K1025 from already SAVED129 prefixes,
same final x/y30-gradient budget and label-free best-full-objective selection.
Labels are read after output export. Falsifiers: no objective/anatomical gain,
still collapsing steps, many active rows, excessive work or adverse tails.
Failure is not a stationarity certificate; equality tangents restrict the
feasible cone. This is standard constrained-optimization geometry applied
to our exact scalar fiber, not a novelty claim. Independent Astra review
precedes medium/large experiments; no new parameter sweep or QP framework.

Root separately loaded the exact native1024 fixed rasters through the SAME
raw-inverted preprocessing. Limiting material cells Hx(783,381), Hy(178,640),
Kx(506,808), Ky(183,856) all have fixed foreground weight1; their9x9 foreground
fractions are1/1/1/.9876543. Thus these observed limits are not simply excluded
white-background cells. This checks only source support, NOT correctness of
cross-stain evidence or biological correspondence. No masks/objective changed.

### T+11.3h actual bounded tangent result: bypassing faces is not enough

Finalized root129affectedtests pass11.59s; wrapper45pass, geometry45pass,
independentAstra16targetedtests pass4.42s. Separate Cholesky-whitened nullspace
projection agrees to2.14e-14 including dependent rows. Replay threads were
explicitly restored from saved config; a new red/green regression caught the
bypassed optimize() setup, not a geometry error. No constraint tolerance changed.

After checking GPU6/7idle, ran ONLY H/K1025 suffixes from already SAVED129prefixes.
No baseline/upstream rerun or manual-label inference. All actual prefix/output
certificates valid; source bestprefix/initial E re-evaluated (2setupcalls).

| Case | Original baseline E | Plain fiber E | Tangent fiber E | Baseline mean512eqTRE | Plain fiber mean | Tangent fiber mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H | .1757646105 | .1769391025 | .1766470683 | .7817015492 | .7955191786 | .7934268002 |
| K | .2281917408 | .2308170920 | .2305527743 | 2.2878981413 | 2.2994992191 | 2.2992546663 |

All77H/69K scored after maps saved. Tangent p90 H1.58647909 versusbaseline
1.59435009, plainfiber1.57644553; K4.27138231 versusbaseline4.25500827,
plainfiber4.27122036. Thus small mean changes do not imply improved tails.
H26+30gradients,54acceptedmoves,65solverforwards+3selection/report=68completeE;
K30+30gradients,58moves,61solverforwards+3=64. Plus2setupE calls each.
H x stopsobjective_backtrack_limit, ybudget; Kbothbudget. Zero geometry
rejections/tangent failures/curvature skips/descent fallbacks. H15rescues/15
extraHactions/15Gram solves; K29rescues/50extraHactions/29Gram solves. Near-face
row cap never hit. Cold suffix2.7133/2.9093s, peaks1909636608/1884417536bytes;
historicalbaseline suffix2.7467/2.7574s has DIFFERENT timer scope, no speedwin.

Decision: this fixes one specific outward-direction stopping mechanism but
does NOT produce competitive dense optimization at this budget. Retire the
physical-fiber extension as a performance candidate for now; keep its module,
counterexamples and evidence. Do not expand to a general QP, threshold sweep
or larger active set. Continue mainline diagnostics rather than proclaiming
the entire coordinated map class infeasible. Independent saved-map check pending.

Independent saved-map review now confirms every source/replayed prefix array
BITIDENTICAL, including actual129oldnodes;16,777,216actualfourcorners across
fourprefix/final maps have0nonpositive/nonfinite and exactfixedboundaries.
Final minnormalizedJ H.01583257/K.01141455. LiteralindependentP1AC/manualIDs
reproduce allscores. Including2setupcalls totalcompleteE70H/66K. The genuine
conclusion is bypassed contracted-face stops, NOT superior registration.

### T+11.3h research card: source-query conditioning, not a new optimizer

Question: does the declared source-grid interpolation weaken observation of
fine nodal modes when image side=control side-1? For AC triangles with one
pixelcenter per cell, query map is exactly .5*(Y_a+Y_c). A boundary-zero diagonal
chain of L interior nodes has sampling Gram .5diag+.25offdiag. Its eigenvalues
are .5+.5cos(k*pi/(L+1)); no exactnullmode, but condition=cot²(pi/[2(L+1)]).
For L1023 this is424971.18. IndependentAstra derives the same spectrum.
Smallest decisive test: compare actual FrozenP1Evaluator to literal midpoint,
explicit SMALL boundary-eliminated matrix eigenvalues, and two samples/axis
percell at quarter/threequarter points. Main sizes257/1025 use closed chain
formula plus actual evaluator weights, NOT a giant eigenproblem.
Falsifier: actual query weights or boundary chain disagree. Limits: this is
UNWEIGHTED source evaluation, NOT full raster/MIND/mask/ARAP/shape/SG Hessian;
positive regularizers and image gradients change it. Do not infer anatomical
bottleneck, exact latent nullmode, or benefit of denser image quadrature from
this diagnostic alone. No registration objective or output safety changed.

Diagnostic8focusedtests pass4.16s; actualdyadic17/65/257/1025 evaluator probes
have0midpointerror. IndependentAstra verifies quarterlocalGram eigenvalues
((3-sqrt5)/4,1/4,1/4,(3+sqrt5)/4), and globalfixedboundary condition<=
(7+3sqrt5)/2=6.854101966 independentgrid. Masks/imagegradients/priors remain
outside this bound. Code+compactresults and tangentnegative milestone pushed
as d568668; no giant Gram was constructed atmain resolution.

### T+11.5h research card: one image quadrature ablation, original rasters

Question: does weak source observation atcenterqueries materially limit the
existing dense analytic-Adam suffix? Retirephysicalfiberperformance extension;
do not combine its chart/Hmetric with this next intervention. Use original
analyticlatentAdam, same saved129H/K prefixes, final1025control/1024raster,
same30gradients+lasttrial percoordinate and nominaledgephysicalLR6.25e-5.
ONLY change MIND-transport IMAGE term to4quarter-position sourcequeries in
eachoriginalpixel; descriptorfields computedonce on ORIGINAL1024rasters,
fixed/movingdescriptorvalues bothbilinearlysampled withzero padding and
align_corners=False. Eachfourquery inherits the ORIGINAL pixel's fixedmask
weight, denominator4*sum(mask). No maskinterpolation/adaptiveoverlap/dropped
points; no fabricatedextraimageinformation or descriptor recomputation.

Keep ORIGINAL CENTER-based OOB penalty exactly unchanged, plus ARAP3,
shape1e-4,frozenmachinepoints. Additionalquarteroutsidefraction may be a
diagnostic only. Bothobjectivearms share candidateprefixfamily(initial,
storedE1-bestprefix,incoming129prefix) and acceptedx/y. Selection uses each
arm's OWNcompleteobjective, never landmarks. Cross-evaluate BOTHoutputs
under E1 and E4; rawtotals across objectives cannot establish improvement.
No claim that the sourceGram bounds transfer to fullE4Hessian.

Smallestdecisive tests: tiny literal4quadrature reference/full-Y VJP FD away
fromknots; identitysamefeatures, oldcenterOOB/priors/matches exactlyretained,
staticmaskdenominator/nonfinite/inputguards. Then pairedH/K dense suffixes
from SAMEsavedprefixes, no upstream rerun, all77/69labels scoredONLYafter
exports. Count imageinterpolations4vs1 (pluscenterOOB mapquery), setupmemory,
grads/forwards/checks/crossE/timing/certs. No new feature/kernelparameters.
Falsifiers: excessivecost/peak, no objective/anatomygain, adverse tails or
actualtopologyfailure. A negative result ends this quadraturevariant; no
newfeaturesearch/preconditioner based solelyonsourceGram. Primarypriorart:
Pierré etal2016FE-DIC quadrature DOI10.1016/j.optlaseng.2015.07.008;
finiteelementquadrature is established, not a proposed novelty.

Independent Astra review and root42focusedtests passed before the real jobs.
The first two GPU starts stopped BEFORE optimization/export: the new input
guard compared unindexed cuda to the actual cuda:0 tensor device as unequal.
Device-only normalization now resolves current CUDA index and still rejects
explicit mismatched indices; finite/immutable/dtype guards are unchanged.
Root rerun43pass/1CUDAfixture skip8.89s; Astra separately approves the fix.
Failure logs t117 are retained. Standalone arm reports now explicitly identify
E1/E4, rather than inheriting an ambiguous configuration.loss='mind'.

### T+11.8h research card: predeclared all20 known-specimen direction coverage

Question: do current coordinated-instance gains survive stain/direction changes,
or do the selected three development pairs hide adverse tails? This is a broader
DEVELOPMENT stress test, not an independent-patient confirmation. One existing
lung-lesion3 specimen has five stains and exactly20 ordered distinct pairs;
all20 labels were viewed in earlier work. Keep every pair/failure in the cohort.
Existing original512canvases, image-only positive affines (det.838--1.184), and
native frame layouts are available. Do not call these20patients or blind tests.

Exact intervention: NO new geometry/optimizer/loss. Freeze current257control,
P1ac/MINDtransport/ARAP3/shape1e-4/match.1/edgeLR.004/300gradient recipe,
identity residual plus the SAME saved image-only affine for analytic/F2/nativeDHR.
Analytic1cycle30steps per scalar direction, F2two vectorcycles30steps, same
five17..257coefficient levels and32..512imagecontinuation, mixedprecision.
F2/nativeDHR are controls, not relabelled same-runtime/functional algorithms.
NativeDHR keeps its existing nativeobjective/preprocessing; export is not repaired.

Existing all20 aligned-match archives contain selected RANSAC inliers WITHOUT
confidence or raw assignments. Therefore they cannot silently stand in for the
current raw-confidence evidence. Use20passes of the ALREADY existing frozen
image matcher and its current raw-record provenance, with no matcher training,
new filtering, invented confidence or anatomical labels. Preparation/setup cost
is separate and counted. Existing positive affines are reused, not refitted.

Smallest decisive test: tiny wrapper/case-count/shared-affine/provenance guards,
known independent P1/native-field/frame scoring fixtures; then exactly20pairs
on idle remoteGPU, nominal15--30minute budget, no nearby parameter sweeps.
All80sharedIDs/direction scored ONLY after prediction exports, using original
annotation scale conversion. Report equal-direction mean, p90/max, worse-than-
affine directions, failures/topology, per-direction time/allocatedpeak and setup.
Never pool1600landmarks as1600independent subjects. Failure counts remain explicit;
successful-only summaries, if shown, cannot replace denominator20 disclosure.
Decisionchanged: robustness of currentinstance gains, not architecture selection
by labels or a formalSOTA claim. No second algorithmic alternative is opened.

### T+11.9h actual quadrature result: no practical transfer

GPU6tinydevicecheckpassed (same losses, maxfullYgrad atomic-scatter difference
1.39e-17, NaNfieldsstillrejected); remotevenvlacks pytest, no install attempted.
RootCPU43pass/1CUDA skip; separateAstrafixreviewapproved. Afterfreshidle6/7
inventory, pairedH/K1025 jobs t118finished. Allarms60gradients/73completeE,
zero failedtrials, y-stage selectedby OWNfullE;150E/case includingcross4.

| Case | Center mean512eqTRE | Quarter mean | Center p90 | Quarter p90 | Center suffix s | Quarter suffix s |
|---|---:|---:|---:|---:|---:|---:|
| H77 | .7817015492 | .7831290582 | 1.5943500880 | 1.5796892244 | 3.3747 | 4.2300 |
| K69 | 2.2878981413 | 2.2926002741 | 4.2550082692 | 4.2702678071 | 3.4154 | 4.2860 |

Quartermaps improveE4(.1675794921->.1672776439 H,
.2162816450->.2158488187 K) and worsenE1(.1757646105->.1763116963 H,
.2281917408->.2293928716 K). Thus smaller rawE4 is NOT an E1 improvement.
Meansadverseboth/tailsmixed; moregradientsampling is not bettercorrespondence.
Shapeandminscale improve, but anatomydoesnot. Addedquartercaches640MiB;
allpairedcachesresidentbeforeBOTHarms. Absolutepeakscenter2752487424bytes,
quarter3020957696H/3020924416K; incremental1850296832versus~2101727744bytes.
No historical/isolation/warmratio claim. Eacharm75E inclcross ⇒75vs300moving
descriptorinterpolationequivalents,8fixedsetupinterpolations. Binaryactual
certsvalid; independentarray/IDreviewpending. FORMULATION34 containsfullscope.

Decision: STOPquadraturevariant, keeporiginalcenterfunctional. Do not turn
sourceGram theorem into fullimage/anatomyclaim or rescuewithnearbyparameters.
All20fixedrecipe DEVELOPMENT coverage is now the next executable test; no new
geometry/optimizer/matchertraining and no independentpatient claim.

Independent Astra actual-array review completes:6prefix/center/quarter maps,
4,194,304cornerrows EACH (25,165,824total),0nonpositive/nonfinite, allnormalized
corners>.001, exactboundary andpositivecorrectaffines. BothprefixesBITIDENTICAL
t109prefixes and129oldnodes; bothcentermapsBITIDENTICALt109originalbaseline.
SeparateCSV/frame+literalAC calculations (no productionmap/scorer calls) agree
with ALL77/69ID scores within1.14e-13pixel512. Thus the negativeanatomyresult
is not a sampler/frame/export comparison bug. Criticalclaims checked separately.

Root finalaffected quadrature/sampler/nestedpriors/stagecache suite:
108passed/1localCUDAfixture skip9.21s. GPUalias smoke was separately executed
on the actual remoteA6000. Code,pairedcomplete reports andall-IDscores form
the milestone; largeactualnpz/maps andfailure/processlogs remain preservedonD.

### T+12h CPU portability/scaling measurement, unchanged existing operator

Question: are compact first-order full-currentY/proposal VJPs useful outside
the GPU application? Reuse the ALREADY verified explicit_vjp benchmark and
EXACTsame savedreal257P1 anchor as the earlier GPU test, refinedexactlyto1025.
No newgeometry/loss/optimizer, no image/landmarks in the timed operation.
Measure existingAD versus compact torch_manual, float64, validate=True,
radial/activeanalytic/inactiveanalytic,3warmups/10repeats; batch1 at257/1025,
batch4only257. KeepfullY andproposal gradients enabled withsame randomupstream.
Recordmedianforward/VJP andretainedsavedstorage separately; CPUtotalRSS/peak
is NOT inferred from graphstorage or GPUpeak. Candidate equality and BOTHVJPs
must pass existingstrictchecks, freshactualcorners remainpositiveaftertiming.
Element resourceprobe:128CPUs, load~2.1,~918GiBavailable; useONLY2CPUthreads,
no GPUallocation or packageinstall. Confirmed existingbase_cp Torch2.5.1cu124.
Smallestdecisivetest: oneexistingboundedbenchmarkcall, nominalunder10min,
no threadcount/autotuning sweep. This changesdeployment/scaling evidence only,
notanatomy or the all20predeclared method recipe.

ActualCPUbenchmarkfinished;9cases,3warmups/10rep each, sameY/z/upstream,
candidatevalueerror0, maxrelativeY/zVJPerror3.65e-16/1.76e-16. At1025B1,
ordinary→compacttorch_manual forward/VJPms:radial338.10/372.40→238.63/18.26,
activeanalytic266.62/370.79→263.31/19.60, inactive290.76/349.91→212.10/16.45.
Saveduniquegraphstorage432.031→24.047MiB includesY/z; NOTCPUtotalRSS/peak.
Allfreshqmin>.001. IndependentAstra recomputesall18storedmedians exactly,
checksactualbenchmarkcode/counts/scope, noindependenthardware re-probe/rerun.
FORMULATION35 hasall9rows; oldbenchmarkGPU6/integrationcaveats explicitly stale.

All20predictor10tinytests/scorer34tests; rootjoint44pass21.62s andindependent
checker44pass21.92s. Scorer raw-pending andmissingaffine-catch fixesverified,
no actualannotationsread duringthese syntheticfixtures. ResourcecheckGPU6idle,
thenall20fixedrecipe t123started with externaltimeout1800s,2CPUthreads,
ClearAllForwardings=yes; original5canvases/20positiveaffines onlytransferred,
no annotations/densemaps/teachers. All20rawmatchattempts completedfirst;
currentmanifest has11/20directions all3methodsok,0failures. No scoresselected.
ActualF2accepted_gain1 isverifiedfrom original512ARAPt88config, not silently
changedtoearlierdiagnosticgain.75. FORMULATION36 is the self-contained protocol.

### T+12.5h all20 completion and measured next decision

All20 predictions and post-prediction scores complete. Allmethods exported20/20;
all80IDs retained. Primary meanTRE/mean-p90,512canvaspx: affine6.66122/12.23468,
analytic4.52023/9.55521,F2 4.59946/9.70151,nativeDHR5.13359/11.55283.
Native196916nonpositivecorners over20maps, no foldedcase removed fromscoring.
Analytic300grads/0failures each; F2 two incompletebudgets:HE→Ki67 242grads/2
marginrejections,Ki67→HE158/5. Allrejectedextra margins are roundoff-negative
aroundeta=.001, notorientationflips; exportedacceptedmapsvalid. This corrects
the earlier informal "0failures":0job/export failures doesNOTmean0trialfailures.
IndependentAstra reaggregates per-ID errors within1.25e-14. On descriptive
complete-budget18subset, A/F2mean4.1795866/4.1850395,p90mean9.0243651/9.0171700.
The2incompletecases explain93.8%meanadvantage/104.4%tailadvantage; keepprimary20.
DoNOTclaim substantialanatomicalsuperiorityovercompletedF2fromthisevidence.
Actualcompletecall meansA4.79099/F2 15.07797/DHR.72903s,rawmatchermean.37460s
separate. Ainner4.68476s≈97.8%call. A6.57xslowernativeDHR; this matters.

Research card — unchanged complete-application runtime attribution:
Question: which measured component prevents a competitive time-to-accuracy?
Claim: profiler can locate the dominant CURRENT execution cost; no speedup is
claimed before measurement. Assumptions: exact frozen HE→CC10 firstdirection,
same300-gradient analytic recipe/rawmachinepoints/affine/objective/guards.
Falsifier: instrumentation changes maps/counters/functionals, or profiler overhead
dominates sufficiently that device/host attribution is uninformative.
Smallest test: unprofiled→oneCPU/CUDAprofile→unprofiled complete runs on sameidle
GPU, <10min, annotationsabsent. Reportoverhead, nestednonadditive ranges, kernel/
host selfevents andscalar/sync events, ordinaryend-to-endtimings separately.
Prior work: existingstagecache/FrozenP1/compactVJP alreadyimplemented; doNOT
rebrandrecachingasnewgain. SelectONEpure-tensor speed intervention onlyafter
measured attribution; doNOTstartnewCNN/distillation ononeviewedspecimen.
Decision independently recommended by Astra; builder owns boundedprofiletool,
root executes, raw-arraychecker works separately. Remainingwindow~11.5h.

### T+12.6h concrete F2 floor/strict-acceptance mismatch

Question: can the existing F2 safetyfraction.75 nevertheless hiteta exactly?
Independent Astra source derivation and executable side3/patch2 fixture sayYES.
Identityboundary, center(.001,.5), centerlogit(-1,0), sf=.75,rawspan=.5,gain1:
inputminimumratio.002 → outputcenter(.0005,.5), minimumratio EXACT.001.
Thus positiveorientation/thehistorical nonstrictfloor theorem holds, but the
instanceoptimizer'sSTRICTextra-floor guard rejects. Realfixed257reference is
exactdyadic1/65536; no resampling/refinement occurs in the two failedcohortcases.
The actual failed tensors were not saved, so those exactsevenroundedtrials are
not independently replayed from logs. This is a reproducedstructural mismatch,
not evidence that exportedF2maps folded or a missing-factor-of-two convention.

Research card — minimal explicit strict-floor reserve, not a newregistrationroute:
For a protectedcorner, q(t)=q0+t*L+t²*Q. Let B=max(-L,0)+max(-Q,0),
f=eta*h², q0>f, and choose sigma<=1 with sigma*B<=theta_f*(q0-f),
0<theta_f<1. For0<=t<=sigma, q(t)>=q0-t*B>=f+(1-theta_f)*(q0-f)>f.
Retain existingfraction-of-total-area bound aswell, acceptedgain<=1, fixedpatch
perimeter/nonconflictcoverage, originalactualrounded-outputstrictguard unchanged.
Candidatechange: optional namedfloor_safety_fraction withhistoricaldefault1;
explicitresearchvariant.95 uses min(sf*q0,theta_f*(q0-f)) allowance. Noη
relaxation, no tolerance-basedacceptance, no repair or addedgeometrylinesearch.
Falsifier: linearfixture stillhitsfloor, nonlinearpath violatesbound, localVJP
disagrees, existingdefault changes, or the preservedroundedguard stillrejects
meaningfullyoften inactualmatchedcases. Tinyfixtures first, then threefixed
directions HE→CC10(control),HE→Ki67,Ki67→HE, same300gradientrecipes/rawpoints.
Onlythen decidewhetherall20comparisonneeds a corrected F2 version. This is a
baseline strictnesscorrection alongsideperformanceprofile, not aparameter sweep.

Independent literal all20 actual-array/frame check completes (separate formulas,
no production scorer/map/certifier helper):40safemaps/10,485,760cornerdets,
0nonpositive, exactunitreference/boundary andidenticalpositiveaffines. All3200
safeper-IDcanvaserrors agree≤1.27e-13px; originalmovingpixelerrors≤2.53e-13.
Nativefield affine alreadyincluded independentlytraced ininstalledbaseline;
literalfloat64 bilinear sampling HE→CC10 andHE→Ki67 agrees≤2.60e-5canvaspx
with productionfloat32queries. A mistaken secondaffine giveslarge discrepancies,
not the recorded scores. All20nativecornercounts independentlytotal196916.
MinratioA.00655735/F2.01479939. Noindependentpatient/globalnativecertificateclaim.

CPUbenchmark and completeall20 code/reports/tests synchronizedas ba79a55.
Largeactualmaps/rawmatches/archives remainonD, no files deleted. Profilingtool
underimplementation; concreteF2strictreserve authorizedafterindependentproof,
separatebounded3casecomparison, originalcohort preserved.

### T+12.8h complete application profile observed on idleGPU6

Rootprofilefocused5testsPASS10.98s; deploymentonlynewprofiletool, GPU6idle
48GiBfree verified beforejob, external600stimeout,2CPUthreads. Actualprofile
complete with CUDAcapture; threeunchanged300gradient/0failure registrations.
Unprofiled5.44376/7.47894s,profiled9.39967s,session16.3213s; bracketvariation
37.4%notnegligible. Mapsnormalrepeatmax4.91e-9/profile7.52e-10unit; allcounter/
frame matches, finalEorder1e-12difference. ActualCUDAevents190271including
168105kernel launches; deviceeventdurations939.648ms(includesmemcpy/memset),
CPUlinked940.034msSEPARATE. CPUself13266ms>wall viaoverlappingthreads/scopes.
Backwardinclusive4713.76ms,Evidence2150.68,decoder1224.69,matchprior623.77,
ARAP415.41,shape374.52. Scalarinclusive114.11ms,streamsyncself28.48,devicesync17.27;
doNOTlabelallwaitasavoidablecost. RangeintervalsNESTED/nonadditive.
FORMULATION38 explicitlystates limitsandexactv2.5.1primarysources. Actualall3
maps/reports/profileJSONfetchedtoD. Independentchecker reviewsactualprofile
beforechoosingONEpuretensor fusion test, coldcompilecostmustbecounted.

IndependentAstra raw-profile-map review:all3literalcornerminima.294420,
exactboundary/affine, mapmax/RMS differences reproduceexactly. Profiler backward
"self"4.684s is callerthreadenvelope, NOTexclusivecomputation; work onautograd
threads overlaps. FullCUDA totals cannotberesummedfromcompacttop25JSONalone;
summationcodeverifiedagainstexactPyTorch2.5.1source. No duplicateprofilingjob
foranartifact-onlyaudit. Allcounts/objective/framecomparisonverified.

Research card — ONE joint pure P1 prior compilation:
Question: can fusion reduce genuineexisting ARAP/shape forward/VJP dispatchcost
without changingtheregistrationfunctional? Closure returns existingp1_arap_energy
(validate=False,ac) ANDexistingcorner_symmetric_dirichlet; weightedVJPuses3/1e-4.
UseexistingTorch2.5.1 Inductor/defaultmode, fullgraph=True/staticshape; noautotune
sweep, noCUDAgraph first, no weakercorners/precision/loss. Same257² inputshape
acrossallimagelevels. Firstforward ANDfirstbackward compilelatency counted
separately, then3warmups/10pairedrepetitions; comparevaluesandfullvertexVJP on
actualsavedHE→CC10map (andactualgeometry-hardmap, selectedwithoutlandmarks).
Falsifier: fullgraphcapturefails, values/gradientsdisagree, coldcostcannot
amortize, or measuredwarmgainfails totransfer tooneunchanged300stepapplication.
After successfullocal/operatorchecks, optionalappbackendintegration ONLYfor
thisjointprior, keepimage/matches/decoder/optimizer/guards unchanged; rootowns
integrationafterF2builderfile isstable. No completeEvidence/decodercompile
untilthisoneinterventionisdecided. Primarysource exactTorch2.5.1compilefunction
andprofiler_util.py; no newmathematicalmethod orNNgeneralizationclaim.

### T+13h F2 reserve completed; measured joint-prior fusion positive at operator level

NewF2focused25tests(root9.22s) and90prioraffectedtests(root24.99s) pass;
independentAstra23coretests/proofreview plus statusguardreadcheck pass. Original
rootinvocation usedonewrongtestfilename/no testsran; correctedaffectedrun is
the recorded90pass, notthefailed invocation. Noinvalidfixtures countedvalid.
Actualpredeclared3pairs×2arms6validexports, independentfraction.95 each300grad/
0failedtrials. HistoricalHEK217grad/3fails,KH252/2; earlierprimarycohort242/158
differs because CUDAnear-floor trajectories/earlyexits are notdeterministic.
ControlHECC300both. Primary20unchanged; post-prediction80-IDscoringpending.
The strictreservefixedbudgetbehavior, not yetanatomy. CorrectedALL20F2 next
forfairbaselinecoverage, notjustsubstitute2favorablecaseoutputs.

ActualGPU7 purejointpriorbench complete2real257maps,3warm/10pairedrep:
H F+VJP5.01571→1.22706ms; geometry-hardHEK4.96715→1.08166ms. All14comparisons
permap agreevalues≤6.11e-16/fullvertexgradient≤9.77e-15. IndependentAstra
recomputes12medians exactly; factory.6332s ALSOcosts beyondfirstF6.5341/VJP2.9703s.
Secondmapsharedcodecache, notindependentcold. Noimage/neural/application
speedclaim fromthis4.09--4.59xoperator result. FORMULATION39 fullscope.

Rootoptionalapplicationintegration retains eagerdefault/completefunctional/
actualguards and sharesONEcompiledpriorcallable amongimagelevels; root86
affectedtestsPASS14.27s, Astrareadreviewapproved. Test compilerbackend'eager'
capture is declared correctness-only, notInductortiming. Next14freshfullcase
optimizations: observedcoldE/compiled, then3E/C/C/Ewarmgroups; cachepolicyand
firstconstant-input/gradient-enabledspecializations counted, nohiddenwarmup.
Two builders ownonlynewcomparison/scorerfiles, rootownsproductionintegration.

Post-prediction3pairscore complete, root12scorerfixturesPASS6.87s. All80IDs
retained perarm, validmaps, nostatisticdrop: meanTRE historicalrepeat7.24058→
reserved7.15148canvaspx; meanofpairp90 13.84972→13.87533 (slightly WORSE).
PerpairmeansHECC5.76437→5.76437,HEK8.18505→8.00298,KH7.77230→7.68710;
no anatomysuperiorityclaim fromsmall mixed result. The successfulstrict-budget
correction warrants ALL20fairbaselineclosure irrespectiveoflabelperformance;
newall20F2usesonlyfrozenimages/matches/affines andoriginal300budget.

### T+13.5h measured whole-application fusion, next bounded dispatch question

Actual14freshHE-to-CC10applications completed on idleGPU6: all300gradients,
310trials,332completeobjectiveevaluations,zero failures,validsavedP1-acmaps.
Warm complete-callmedians eager4.04864s/Inductor2.95093s (27.1%less); three
ABBA meanratios1.36144/1.38280/1.40429. Observedcold6.49489/13.93556s counts
factoryandconstant/gradient specializations; noone-shotwin. Warmcrossbackend
mapmax1.21e-8/unitRMS8.13e-11 versus samebackendrepeatmax8.25e-9/RMS5.49e-11;
finalEcrossdifference<=5.28e-12. Independent actual14mapreview pending.
Rootnewharness/integration/F2all20 tests28PASS38.83s; initialGPUsetup rejected
unindexedcuda beforeanymapwork, correctedvisibleindex0 (or suppliedindex)
and9harness testsPASS20.02s. No invalid execution counted completed.
F2all20 correctedrunner ongoing separatelyGPU7; originalcohort unchanged.

Research card — frozen machine-point P1 evaluation (ONE follow-on dispatch test):
Question: can immutable point-query indices/weights remove repeated point
location and duplicate gather work without changing image/match evidence?
Exactclaim: forfixed source queries q_j andfixed source triangulation,
f_Y(q_j)=sum_(k=1)^3 w_jk Y_(i_jk), withweights/indices independentofcurrentY.
Weightedrobustmatchloss andfullvertexVJP remain unchanged inreal arithmetic;
rounded summation neednotbebitwise identical. Assumptions: detachedfixedquery
buffers, declaredfixed grid/diagonal, same affine/targets/confidence/robustscale,
newsetup fornewcontrolshape; doNOTfreeze deformedgeometry or detach gradients.
Falsifier: value/fullvertexVJP disagreement, stalegrid/diagonalbuffer use, or
setup+warmfullapplication costfails toshowusefulgain. Smallesttest: optional
FrozenP1Evaluator reuse inside ImageCorrespondences on actualHECC frozen203
points and257²map, bothdiagonals/edgequeries tested before application.
Priorwork: ordinary barycentricP1 finite-elementevaluation andexistingrepo
FrozenP1Evaluator (already usedforrasterqueries); no newgeometry/noveltyclaim.
Do notcompilefullEvidence/decoder orchange matcher/robustfunctional forthis.

IndependentAstra completed14-map/91-comparison review: all3,670,016 actual
corners exceedeta, exactidentityboundary/commonpositiveaffine/P1ac,300/310/332/0
each. Warm27.1129%reduction andthreeABBA ratiosreproduceexactly. Crossmapmax
1.215e-8 is sameorder BUTlargerthanwithinrepeat8.251e-9; no bitwiseclaim.
Warmpeak217.854→194.393MiB, coldcompiled364.564MiB; excludescompilerhost/RSS.
Coldextra7.44067s predictssevenadditionalwarmcalls underconstantmedianmodel,
NOTmeasuredcrossover. FORMULATION39.1 definesactualinput/output/clocks/caveats.
Independent3pairF2 score review also complete:1,572,864cornersstricteta,
80IDs eachliteralCSV/frame/P1matches<=2.27e-13px; difficultpairp90bothworsen.

### T+13.6h corrected complete F2 cohort and next accuracy decision

CorrectedF2all20 onGPU7 completes20/20,300gradients/zero failedtrials each;
newonlyF2elapsed290.810s, meancompletecall14.5334s. OriginalA/DHR/rawarchive
references retained, notrerun orretimed. Postprediction productionall20score:
meanTRE A4.52022762/F2.95 4.55134379/DHR5.13359287canvaspx; meanofdirectionp90
9.55520600/9.65783971/11.55283127. A beatsF2 mean14/20,p9012/20, butmeanadvantage
.0311px tiny. Bothmethods improveall20means overaffine; CC10-to-CD31tailworse
thanaffine. IndependentAstraactual20maps/5,242,880corners allstricteta,
boundary/commonaffine/P1ac verified, completeCSV/P1scoresagree<=2.51e-13.
Allarchivedoriginalmetrics unchanged, configsONLYnewoutput/.95floorreserve.
Usecorrectedcohortforfairbaselineclaims; oldprimaryremainsavailable.

Read-onlygeometrybuilder tracedALL20 analytic trials:257/6200cappedtrials
concentratedHEK113/KH85/HEProSPC30/CD31Ki6729; other16entirelyscale1.
197/200acceptedstageslowerfull512E; threeincreases areearlycoarse-image
surrogatetransitions. All40finalfine x/y stagesstillstrictlydecreaseE inlast5
trials; all20best_fullselectterminalstage9. Medianlate5gain .00027162,
uncapped .00027623/capped .00021157, roughly7%oftotalfine-stagegain.
Noevidenceofcohortwidegeometryblockade orhardfine-stall. Allfine stages
initiallyovershoot afterfirstAdamstep, so doNOTassume arbitrarilylargerLRhelps.

Research card — ONE same300-gradient allocation test:
Question: does shiftingexistinggradientbudget towardstill-improvingfine
stages improveactualregistration, ratherthanmerelyaddingiterations?
Exactalgorithm: onecycle, two scalardirections perlevel17/33/65/129/257,
innercounts [30,20,20,30,50] versusoriginal[30,30,30,30,30]. Bothsum150
perdirection,300gradients/10stages/310trials/332E. Stageanchors, actualfixed257
map, pyramid32/64/128/256/512, physicalrates, evidence/weights,strictfloor/
checks unchanged. No new universality/convergenceclaim followsfromallocation.
Assumptions: fullbudgetcompleted, no landmark-drivenallocate/earlystop;
all20directionsretained. Smallestdiscriminatingtest: tinyunequalbudgetcounts
anduniformdefaultmapregression, thenpairuniform/redistributed ALL20 maps in
oneprocessusingSAMEcompiledpriorbackend; separatecoldcachecost/orderrecords.
Falsifier: lowerproxywithoutbetteranatomy orworse tails; retainnegativecases,
doNOTsweepnearbyallocations. RuntimeNOTequalbecausefineimageiterationscostmore.
Priorwork: standardmultiresolutionregistration iterationallocation/subspace
optimization; this ismainlineoptimizationdiagnosis, notnewmechanism/novelty.
Earlier600-cycle/joint/regional/fiber/filter/quartermixed/negative results stay
retired; doNOTreopen themundernewnames.

### T+14h paired allocation NEGATIVE, frozen-point dispatch transfers

Root optionalperlevelbudget/defaultregression+point/prior tests31PASS8.37s,
pointapp/fullharness39PASS25.93s, finaldispatch/budget30PASS27.78s;
allocationrunnerroot14PASS39.57s (author14PASS41.01s, independentAstraread).
No actual GPU speed was inferred from capture/mock fixtures.
IdleGPU6actual40allocationmaps complete, both300/310/332/0 percase; same
compiledpriorbackend andonlyallocation/outputdifferent. Postpredictionall20
scoring: uniformmean/p90 4.52022762/9.55520600; redistribution4.52727187/
9.57804700, WORSE despiteLOWERfullE20/20 (meanEchange-.000873024).
Meanwins5/20,p90wins11/20; worstmeanproSPC-to-HE+.07949/tailHEK+.22767.
IndependentAstra40maps10,485,760strictcorners/exactboundary/affine/P1ac and
40x80literalCSVscoreagreement<=2.54e-13. Uniformretained; retireallocation,
noadjacentsweep. Firstuniformcold7.5554sconfoundsmeantime; medians2.94133/
2.94025 essentiallysame, noallocation-speedclaim.

GPU7actual14point-dispatchapplications withBOTHjointpriorsInductor:
warm2.944472→2.695455s,8.4571%less, ABBA1.09336/1.09127/1.08644.
RootpartialAPIs setupcountverified; independentAstraall14maps/counters/
comparisonrecomputation passes (3,670,016strictcorners). Warmcrossmapmax
1.61410e-8<within1.78035e-8, E4.68e-12<within5.73e-12,imagepartsidentical.
Coldexisting7.52208includesfirstpriorruntimecompile, secondfrozen2.69539
reusescaches; doNOTpresentaspointcold-speedgain. Prior27.1% andpoint8.46%
are separateboundedtests, notmeasuredcombinedall20speed. FORMULATION43/42.1.

Next scientificdecision escalatedtoactualAstrahigh: objective-anatomyalignment,
notanotherglobalcap/iterationremedy. ConsiderONEphysicalbending-priorhypothesis
onlyafterdiagnosis; no newCNN/matcher/nearbyparameterforest. Currentuseful
instance result isreal32%meanimprovementoveraffine onknownspecimen, but
strongF2anatomicalsuperiority/independentclinicalgeneralization NOTestablished.

### Around T+14h actual-Astra scientific pivot: test evidence causally, not another prior

The proposed bending explanation is NOT supported. IndependentCPU five-point
diagnostic on the40savedmaps: r=Y-X withbinary-exact257identity,
L=(r_left+r_right+r_up+r_down-4*r_center)*256², E_b=mean(sum_components L²)
over255²interiors; pre-affine map, no padding/newprior/classicalP1Hessianclaim.
Redistribution lowersE_b18/20 (median24.07%decrease) butmeanTREworsens15/20;
onlyHEK/KH E_b increases. MINDdecreases20/20, ARAP16/20. Therefore doNOT
implementbending asifroughmotionwerea diagnosedcohortcause. No priortuning.

Research card — ONE dense-MIND knockout:
Question: doesdenseMIND addusefulanatomicalsignal beyondexistingfrozenmachine
points underCURRENTpriorbalance? Exactfunctional E_beta=beta*MIND+3*ARAP+
1e-4*cornerSD+.1*rawmatch+OOB, betaONLY1or0. Implementoptionalimage_weight
defaulthistorical1; E0candidateuses0, nointermediateweight/sweep. Same original
affine/rawmatches/confidences/robustscale/foregroundmask/fixeddenominator,
257P1controls/uniform300budget/strictcorners/fixedboundary/compiledpriorbackend.
KeepMINDcomputedandreporteddiagnostically evenwhenitsweightzero, so this
experimentisnotaninference-speedshortcut; geometry/pointgradientsremainlive.
Outputselection usesE_beta ofeacharm, neverlandmarks; doNOTcompareunlikeE0/E1
totalsascommonobjective. All20failures/80IDs retained, labelsreadafterprediction.
Assumptions: rawmachinepoints provideimperfectimage-derivedcorrespondence,
NOTgroundtruth; inredistributionmatch improves11cases,10withworseTRE.
Smallestdecisivetest: completevalue/VJP E0=E1-MIND onactualfixedEvidence,
thentinyoptimizer+counters/certificates, thenONE20-directionknockoutcohort.
Falsifier: meanORtailworse=>rejectcandidate/keeporiginalhybrid,noadjacentweight
sweep. Bothbetterwithoutnewfailure=>densecontributiondetrimental underTHIS
fixedbalance/knownspecimen ONLY; independentconfirmationstillrequired.
Priorwork: featureassignment correspondence (SuperGlueprimaryarXiv1911.11763)
andstandardregularizedlandmarkregistration. Notnewmatcher/neuraltrainingclaim.

Separatecheapposthocboundary-capacity diagnostic (AFTERcompletedpredictions):
fixedresidualboundary impliesimageA([0,1]²). Projectmovingannotationtargets
ontoitsfourline segments tolowerboundunreachableTRE.14/1600targets across6
directions outsidepolygon, maximumbound19.8657pxHE-to-ProSPC;14/20directions
boundzero inclALLproSPC-source whereDHRwins. This raretailrestriction doesNOT
explaincohortwideerrors; boundaryslidinginsamepolygonwouldnotremoveit.
No boundary/affinebranch opened orlandmark-driveninitializerchange.

### T+14.25h dense-MIND knockout NEGATIVE; retain hybrid

Root20focused tests passed58.11s; independentSol verifiedall20 source recipes
and180rebasedartifact references, with ONLYbeta changed. IdleAI GPU6actual
20calls finish300/310/332/0 each; elapsed63.7652s. PostpredictionALL80-ID scoring:
beta0mean/p90 5.02422311/10.02019965 vsbeta1 4.52022762/9.55520600.
MeanTREworsens20/20; tailwins5/20. proSPC-to-HE+1.041997pxmean isstrongadverse
case. Rejectknockout, keeporiginalhybrid; NO intermediateweight sweep.
This intervention showsdensecontributionuseful underTHISfixedbudget/balance,
notMIND'sglobalanatomicalcorrectness. UnlikeE0/E1totalsNOTcompared.
Minimumactualcornerratio.0073506453, allocations202001408--206005248bytes;
firstcall7.45846s/median2.95394s NOTpairedspeedclaim. Independentactualmap/
literalCSVreview inprogress. FORMULATION44definesallterms/selection/VJP/scope.

Independentchecker alsofoundreturnedNaNdiagnostic couldabortJSONpersistence:
builderfixedONLYnewrunner withnonfinitepaths/null+failedstatus+continue19;
14focusedtests60.72s. Actualfinitecohort/results unchanged. Rootaffectedcheck
pending. No newauditframework oroldrunner rewrite.

Next: actualAstra decidingONE evidence/optimization move afterknockout;
parallelboundedengineeringcheck willtestknownfrozen-point speedtransfer on
geometry-difficult HE-to-Ki67, notclaimnewanatomicalgain orindependentdata.
24hgoalremainsACTIVE withapproximately9.75havailable, notcomplete.

IndependentSol closure: literal5CSVs/all80IDs/generictriangleaffine solves on
both40maps/3200queries maxstoredscoreerror1.1413e-13px;10,485,760strictcorners,
exactboundaries/commonaffines/P1ac/counterspass. IndependentARAP/shapevalues
agree1.73e-18/9.71e-16; E_beta reconstruction6.94e-18. F2/DHR140archiverefs/
allper-IDmetricsunchanged. Rootpostfix21tests59.09s; checkerreadNaNfailurefix
andFORMULATION44 approves. Denseknockoutimplementation/researchresultclosed,
NOT24hgoalclosed.

Research card — ONE finite-displacement MIND capture prefix:
Question: can existingfeatures proposefinite motion beyond a local derivative,
entering a better basin without changing the accepted objective or topology?
Exactalgorithm: existing128Evidence;33nodeqi=(j/32,i/32),9patchsamples
s=(-1,0,1)^2/128,81labels k=(-4,...,4)^2 ORIGINALmoving128px. Cost_i(k)=
sum_s m(qi+s)*[mean8ch abs(Phi_f(qi+s)-Phi_m(A(qi+s)+b+k/128))+
sum_components[(relu(-target_d)+relu(target_d-1))^2]]/sum_s m(qi+s). SampleORIGINALmoving
features, zero padding/bilinear/align_corners=False; denominatorSTATICalllabels,
nooverlapdropping. Emptyfixedpatch=>zero; zero-firstthenlexicographicties;
coarseboundaryzero. p_i=A^{-1}(k*/128); prolongRAWscalarcomponents33→257,
notmaps. Prefixtryxtheny withoriginalanalyticeta.001/theta.95/trial1;
actualroundedstrictgeometry+unchangedFULL512E1decrease required toaccept
eachaxis;refreshconstraintsonacceptedY, noalpha ladder/repair/smoothing.
Thenoriginaluniform300gradientsfromacceptedprefix; extraE/search/costsreported,
NOTsame332calls/equalcompute. DiscreteargminnotclaimedneuralVJP.
Assumptions: residualcorrespondencewithin±16canvaspx/axis, discriminative
existingfeatures, locallytranslatedpatch andcoarsebasiscanfindbetterbasin.
Falsifier: fixedsyntheticcapturefixture noimprovement/extra failure=>retire;
actualall20 ifjustified mustimprovebothmeanandtailwithoutnewfailure, otherwise
retainoriginalhybrid. No radius/window/weight/feature sweep.
Priorwork: Siebert/Hansen/Heinrich arXiv2112.03053 Sec2(fullprimaryread)
costvolume+coupledconvex+Adam; ourL1/33basis/strictprefix is LIMITEDadaptation,
NOTConvexAdamreplication/globalconvexity/globaloptimality theorem.
Builderownsisolatedproposalmodule/tests; rootownscurrentoptionalcoreintegration;
independentchecker willverifyunits/denominators/actualsafeacceptance.

### T+14.6h discrete proposal correct; second-pair speed transfers

Capturemodule20tests5.33s; roothelper5tests4.02s, defaultcore40tests8.73s;
rootwholecachedzero-prefix+capture+prior affected42tests8.09s. Earlier2guard
fixturefails were correctedtoactuallyreachcapturevalidation, no production
guard weakened. Independentchecker25tests5.63s/module+helperread passes;
itfoundunnecessaryexisting-onlybackendrestriction, correctedBEFOREdeployment
so stage_cache originalstagesremainunchanged. FORM45self-contained search/
affine conversion/costdenom/topology/acceptance/noargminVJP/overhead/criteria.
ActualAstra confirms pilot meansTHREEsyntheticrealtexturetargets, notH/L/K:
freshsameknownARP300baseline+prefix, atleast2/3coordinateRMSEimprove+equalcase
meanlower+nonewfailuresbeforeall20. Search16pxcannotcoverwhole37--61pxtruth;
it tests finitecapture followedbyoriginaloptimizer, notteacher recovery alone.

Pair-selector root25tests48.11s then idleAI GPU7actualHEK14applicationscomplete.
Independent14NPZ/reportread3,670,016corners/1,835,008P1trianglesstricteta,
exactboundaries/commonpositiveaffine/300/310/332/0/10, selected9. Frozen157
queriesprepareoncepercall/7536bytes, existing0. Warmmedians2.951007409→
2.698016612s,8.5730%less, ABBA1.094499/1.087646/1.115014. Crossmapmax
8.23423e-10vswithin7.92953e-10, E4.38427e-13vs3.68594e-13 SAMEORDER,
NOTbitwise/notstrictbelowrepeat. Cold7.40655vs2.66280 asymmetricpriorcache
reuseNOTpointcoldgain;peaks204834304--205915136bytes. FORM46scopes/actualdata.
This issecondknownpairengineeringtransfer, notnewanatomicalgain/heldoutpatient.

### T+14.9h discrete capture rejected; frozen-recipe transfer next

All three fresh baseline/prefix synthetic applications completed. Six proposed
axis updates are feasible (scale.474525, minimum corner ratio.05095) but raise
the unchanged complete objective by.06532--.12143, so all are rejected.
Appearance improves, but weighted ARAP increases. Equal-case coordinate RMSE
1.557919913341->1.557919913420px is numerical variation, not progress.
Independent final-map and reconstructed-candidate checks close geometry,
direction, full-E acceptance and query errors. FORMULATION45.2 retains the
negative result. Retire this exact recipe; no nearby parameter sweep/all20 run.

Read-only availability audit finds MIIT_v4 on D:, nine released prostate serial
sections from ONE sample,124rows in each matching-label CSV. Publisher metadata
resolves correspondence semantics. Previous QC project already evaluated these
pairs, and prior synthetic work used these images as textures: NOT blind,
historically untouched, nine patients or an external clinical cohort. Pixel
coordinates are documented; physical spacing/pixel-center convention uncertain.
Use only explicitly declared pixel conventions, no invented micron values.

Research card — frozen additional-specimen development transfer:
Question: does the retained lung recipe improve correspondence on a different
real tissue specimen, with common image-only initialization and matched controls?
Exact design: metadata-selected moving->fixed2->3(HE/HES),7->8(HES/MTS),
10->11(IHC/HES); both directions NOT added after viewing scores. Original TIFFs,
512square aspect-preserving canvases with saved exact two-axis resize/padding,
existing positive image-only SuperGlue similarity and raw machine matches.
No manual coordinates before all prediction attempts finish. Analytic P1-ac257,
original uniform300gradient recipe/E1; corrected F2 with strict-floor reserve
.95 and same300gradients; native DHR with common affine, no topology repair.
Report common-affine baseline, every pair mean/p90 TRE in moving native pixels
and512canvas pixels, equal-pair mean and mean pair-p90, all failures, actual
budgets/topology/time/memory. No micrometer or official challenge score claim.
Assumptions: matching publisher labels; CSV x/y are native-image pixel locations,
treated explicitly as zero-based centers for resampling (uncertainty disclosed).
Falsifier: failure or affine-relative deterioration is retained; no landmark-based
initializer/regularizer tuning. Result changes transfer claim, not frozen recipe.
Smallest decisive test: three pairs, one prediction per method, then posthoc
all-ID score and independent frame/geometry recomputation. Not three patients.
Prior work: official MIIT Zenodo14931377 and author analysis notebook04.

### T+15.1h first additional-specimen prediction: two initializer failures

Root runner/scorer31tests7.16s; independent review30tests6.49s closes an actual
missing P1 diagonal argument before production scoring. NativeCUDA peak fixture
then added; scorer successful124point fixture covers both P1 arms and no doubleA.
Only six native TIFF images (102MB), no manual coordinates, are uploaded to AI.
Fresh idleGPU6 call21.8052s completes all three attempt records. Two pairs have
image-only initializer failure:2->3 rawSG5matches,10->11 rawSG3matches, both below
existing8point eligibility. All three methods skipped on those pairs, failures
remain in denominator.7->8 completes300/310/332/0 for both safe arms, actual
mincorner.369447/.205039, completecalls4.73055/12.42606s, allocated225289216/
495478272B. NativeDHR.71344s/70539264B; timing is single-call, not pairedspeed.

No real coordinate or individualID has yet been opened. Thus no anatomical
conclusion is available. Frozen pipeline fails to cover all3cases because of
INITIALIZATION, not an observed decoder/topology failure. Before scoring, bounded
Astra decision asks whether one standard image-only four-right-angle search is
appropriate; it would be a disclosed NEW initializer protocol, never silently
overwrite the frozen failure or call it unchanged confirmation. No image-loss,
regularizer, decoder, annotation-based selection or nearby matcher threshold
sweep is opened. Actual saved inputs/outputs are fetched to D.

Research card — ONE initializer amendment, before any real labels:
Question: do standard four right-angle moving-image views repair image-only
matching eligibility without changing any deformable-registration recipe?
Exact algorithm: k=0,1,2,3 torch.rot90 views of moving512canvas; unchanged
SuperPoint/SuperGlue settings, confidence.3, same6pxRANSAC/minimum8inliers.
Unrotate moving target keypoints to ORIGINAL pixel frame BEFORE existing RANSAC
and positive-similarity fit. For k1 inverse(x,y)=(511-y_rot,x_rot); k2=(511-x_rot,
511-y_rot); k3=(y_rot,511-x_rot). Use fixed keypoints unchanged. Select eligible
candidate lexicographically: most RANSAC inliers, most occupied fixed4x4bins,
lowest final positive-similarity fitRMSE, smallestk tie-break. Retain all four
diagnostics. No labels, downstream image objective, TRE or manual orientation.
Apply identical policy to ALL three pairs including previous successful7->8;
samechosenfloat32positiveA,b shared byanalytic/F2/nativeDHR. All subsequent
settings and raw point extraction unchanged. Original frozenfailures retained.
Assumptions: orientation mismatch is at least partly matcher-limiting; right-angle
views sufficient to provide an eligiblematch set. No anatomicalcorrectness claim.
Falsifier: no eligiblecandidate retains failure, not identityfallback; no extra
angle/threshold/loss sweep. Completion improvespipelinecoverage only; anatomical
evaluation follows all terminalpredictionattempts under explicitamendedprotocol.
Smallest decisive test: exact quarter-turn keypoint fixtures/ranking/defaultpath,
then threeamended predictions. Priorwork: standardmulti-viewrigidinitialization,
existing DHR SuperGlue/RANSAC; no new matcher/theorem/learnedmodel claim.
Astra independentdecision approves thisoneboundedresponse to actual5/3 rawmatch
failures. It is NOTunchanged-recipeconfirmation and no realcoordinates read yet.

### T+15.4h all amended predictions complete; annotation availability correction

Four-view initializer rescues coverage: allthree commonaffines/rawmatches valid,
allnine registrations exported, allsixsafe300gradient/0failure. Root56focused
tests7.65s; independentrotation39tests6.87s. Only AFTER allpredictionattempts
complete did separate scorer FIRST openmanualcoordinates. Initial all124finite
assumption is falsified; saved strict scoring report retains all3failures.
First commentary incorrectly suspected out-of-native-scale; corrected immediately
on independent diagnostic. ALLfinitecoordinates fit nativeTIFFbounds; NOscaling
or origin correction is justified. Missingcoordinates are exclusively(+inf,+inf).

Both independentcontexts verify native124nominalIDs identical inallsixsections;
section2/3/7/8/10/11missing counts0/1/11/17/24/26. Authorprimarymetrics.py
merges semanticlabels, replacesinf withNaN anddrops unavailableannotations before
distances; nativeimage/Pointset loaders applyNOcoordinate scale. This is legitimate
sourceannotation availability, not difficult prediction exclusion.
Explicit scorer option --allow-missing-inf now requiresALL124nominalIDs but
evaluates EVERYboth-source-finite label withSAME eligible set forALLmethods:
123,107,98points for thethreepairs. Report unavailable labels/counts, all3pair
denominator and EVERYpredictionfailure. NaN/partialinf/negativeinf/duplicate/
finiteout-of-bounds stillfail; no map/parameter/annotation/source edits, no outcomes
used tochoose eligibility. Oldstrictscore preserved; correctedscore saved separately.
This amends scoring-input interpretation, NOTregistration; anatomicalnumbers still
unavailable atthisdecision. Physicalspacing and exactcenter convention stayunverified.
Prior source: https://raw.githubusercontent.com/mwess/miit/master/miit/utils/metrics.py

### T+15.6h additional-specimen development transfer independently closed

Explicitmissing-annotationcorrectedscore retains123/107/98availablelabels,
all124nominalIDs/missinglists/all3pairs/ALLmethods. Equal-pair512pxmean/p90:
affine4.90419347/8.89788359,analytic3.89754022/6.28952850,
F23.96429371/6.38170185,nativeDHR4.05061461/7.18120484.
Independentgenerictriangleaffines+literalCSV/nativeframequerychecksperID1.13e-12px;
1,572,864actualsafecorners/min.20157431038/exactboundaries/sharedfloat32A,
all300/310/332/0/10. Nativeactualnonpositivecorners11506/5271/15158retained,
localdiagnosticnotglobalclinicalclaim.12initializerrecords: selectedk3/0/3,
inliers142/144/110,support16/16/13; higherinliersselectk0on7->8despitelowerk3fit.
AI GPU6RTXA6000,CUDA_VISIBLE_DEVICES6 fromactualSSHlaunch/deviceinventory;
savedartifactconfigsrecordCUDAbutnotphysicalGPUordinal. No othersjob/portchanged.

Strongadverse: analyticlosesF2on7->8mean/p90,DHRon7->8p90; maxima worseDHR
on2->3and10->11.7->8worst52.713426px worsensaffine51.551922. Nooutliercure.
Observedmeanscomplete4.218/13.212/.690s(A/F2/DHR), single serialcallsnotwarmABBA;
allocatedbaselinepeaks~216/474/67MiB notprocessGPU/RSS. IndependentSolsource/
artifactreviewclosed, not24hgoalclosed. Mainaccuracyrecipekept, correctedknown
sourceavailabilityandimageonlyinitializerextensions fullydisclosed. REPORT.md
nowlivingSELF-CONTAINED definitions/VJP/objective/pipeline/results/limitations,
notfinalreview; independentmathcheckcorrectedstageacceptancevsbestfullwording.

Research card — ONE detached trial-diagnostic transfer consolidation:
Question: can avoidable individual GPUscalar->CPU transfers reduce complete
application time without changing the mathematical optimizer or feasibility?
Exact claim: at the SAME existingtrialdecision boundary, concatenate ALREADY
computed detached diagnostics/parts/total/finiteflags into one CPUtransfer,
unpackidenticalvalues; allobjective tensorarithmetic/reductions/gradientpaths,
guardpoints/bestmapselection/iterationcounts unchanged. Legacydefaultpreserved.
No guard isdeferred to a latertrial, no newEcall, no logs/failedtrial discarded.
Only mainfixed-fine/nonjointanalyticstage_cache recipe initiallysupported;
unsupported combinations explicitly rejected ratherthan quietlychanged.
Assumptions: hosttransfer/dispatch overhead is measurable; CUDAwaitingunder
scalaritems mayinsteadbe genuineGPUwork, so profiledCPUtime is only an upperbound.
Smallestdecisivetest: CPUliteralvalue/gradient/fulltinytrajectory and deliberate
NaN/illegalfixturetests, then TWOknownHECC/HEK pairedwarmABBA actualapplications,
maps/budgets/certificates and within-backendvariation independentlyrecomputed.
Falsifiers: anydecision/count/certdifference or mapdifference beyondknownrepeat
scale; speedgainnotbeyondordinarywithinbackenddispersion or regressiononepair.
If falsified retire EXACTintervention, no nearbybatching/syncpolicyforest.
Packing/unpack/guardcost insidecompletetime. No claimednewanatomicalgain, neural
training, registrationnovelty or newalgorithm. PrimaryPyTorchdetach/stack/cpu
semantics plusactualpriorprofiler; independentAstraapproves boundedengineering.

### T+15.8h bounded range diagnosis and unchanged-optimizer engineering

Independent MIIT range diagnostic: zero targets outside the shared affine
image polygon among123/107/98available labels. Every distance-to-range
necessary lower bound is zero; worst7-to8Pt-121lies38.98canvas pixels inside
the polygon but has52.71px analytic TRE. This excludes ONLY an outside-range
explanation for that failure, not other boundary/optimization/evidence effects.
No boundary retune. Existing ABBA comparison runner now also tests detached
trial diagnostics while holding Inductor priors/frozen machine queries fixed.
Root14tests31.22s, including14complete tiny CPU applications with bitwise maps
and identical objective trajectories. Independent core/helper/runner review
pending before production GPU pilot. Complete timers include pack/unpack cost.

Research card — unchanged six MIIT accepted-stage replays:
Question: do decreasing full-resolution E1 and the E1-selected prefix improve
actual anatomical mean/tail errors along the already tested trajectories?
Exact experiment: replay each of the original three amended MIIT analytic/F2
configurations WITHOUT changing coefficients, initial affine, machine matches,
iteration budget, evidence, proposal, objective or backend. Reuse existing
coordinated_replay_trajectory.record to copy10accepted maps to CPU; predictions
recording reads no labels. Then offline score the SAME123/107/98available paired
IDs under original declared P1/ac/half-pixel frames. At each stage show accepted
map and prefix selected ONLY by minimum complete512 E1, including identity.
Align by cumulative gradients/trials/E evaluations, not just stage number.
Assumptions: instrumented replay endpoints match original maps/counters within
known GPU repeatability; all actual exported and snapshot geometry remains valid.
Falsifiers: endpoint/selection/counter mismatch prevents interpreting a replay as
the original trajectory. Falling full E1 with flat/worsening TRE supports ONLY
observed objective-anatomy discordance, not global optimizer adequacy or inherent
evidence insufficiency. If final selected E1 exceeds a visited prefix, fix the
selection bug rather than claiming mechanism failure. No TRE stopping/selection,
weight/schedule/initializer tuning or new alternative mechanism follows silently.
Smallest decisive test: existing prefix-selection unit fixtures plus whole tiny
record; then six actual unchanged257-grid replays and independent snapshot/frame
scoring. Snapshot CPU copies and callbacks contaminate elapsed time; disclose,
do not replace clean paired timing by instrumented time-to-accuracy claims.
Prior work: existing repository replay/score_arrays, ordinary optimizer diagnostic
curves. Independent Astra decision approves this bounded explanatory experiment.

Live saved MIIT diagnostics additionally show930/930analytic scales exactly1:
there was no ACTIVE uniform feasibility clipping on these observed trials. This
does not remove fixed-boundary/representation constraints or show simultaneously
attainable target positions. ALL328available targets inside affine range is only
a necessary range check, not proof of a globally feasible desired correspondence.

### T+16h packed extraction retired; unchanged trajectory replay started

Independent actual28callaudit closes correctness, not performance: all7,340,032
corners>.001, exact boundaries/affines, budgets300/310/332/0/10/selected9.
HECC warm existing2.905629 vs packed2.737756s, but lastABBAregresses;
HEK2.731447 vs2.757498s, pairedmedianratio.996016. Within-arm dispersion~.4s.
This FAILS predeclared two-pair nonregression; retire exactintervention, legacy
default remains. No syncpolicyretune. Actual same-scale GPU map differences
fully disclosed, CPU bitwise fixtures/latefailureguards pass. FORMULATION48.
Root47packed/replaytests31.28s. The independently approved next diagnostic
reuses existing recording for ALLsixMIIT analytic/F2configs unchanged. First
record2-to3analytic completes300/10,10.57MBsnapshots;remainingfive running on
idleAI GPU6. Manual labels are not on remote/record inputs. Independent reader
will compare snapshot anatomy ONLY offline after terminal recording.

Research card — directional descriptor coordinate-frame diagnostic:
Question: does channelwise transport of our eight directional MIND-like channels
give a nonzero error even at a KNOWN exact affine image correspondence?
Exact claim: for I_f(x)=I_m(Ax+b), patch SSD with offset r/window W transforms
to offset Ar/window AW. A quarter-turn preserves the eight offsets and square
3x3window, but permutes their CHANNEL identities; arbitrary rotation/scale does
not generally have an exact eight-channel permutation. This is a limitation
of the declared transport objective, not automatically an implementation bug.
Assumptions: same scalar image under an exact quarter-turn, identical finite
boundary/pooling conventions; normalization min/mean is permutation invariant.
Smallest decisive test: identity and k=1,2,3exact torch.rot90 of seeded anisotropic
texture, then one existing512real texture; independent literal channel indices.
Compare fixed descriptors with spatially rotated original moving descriptors
WITH and WITHOUT the mathematically required permutation. No image interpolation,
OOB ambiguity, optimizer, landmarks, learned model or registration retuning.
Falsifier: corrected channels fail to match within numerical reduction rounding,
identity fails, or non-permuted directional mismatch is absent on the chosen
non-isotropic texture. Mechanism evidence does not establish responsibility for
MIIT errors. Sparse manual centers do NOT supply true local Jacobians/neighborhood
warps, so cannot define a true-map after_warp descriptor experiment. Any shared-A
local-neighborhood probe would require explicit approximation wording. Prior
mixed/negative after_warp optimization results remain contrary evidence.
Prior work: MIND self-similarity descriptor (Heinrich etal2012,
doi10.1016/j.media.2012.05.008); actual repository implementation OFFSETS and
pooling definitions. Independent Astra approves ONLY diagnostic before any
new objective pilot; no after_warp optimizer experiment authorized by this card.

Research card — ONE frozen-affine-frame feature-ranking diagnostic:
Question: does computing the moving descriptor AFTER the known shared affine
give better truth-versus-current ranking on the same328MIIT available centers?
Define T_A(p)=Ap+b, fixed descriptor D_f=D(I_f), original D_m=D(I_m),
affine-frame D_A=D(I_m composed with T_A), computed once on the512pixel raster.
Costs C_R(q,p)=mean_channels|D_f(q)-D_m(T_A(p))| and
C_A(q,p)=mean_channels|D_f(q)-D_A(p)|. Evaluate BOTH at known landmark residual
p_truth=A^{-1}(y_truth-b) and saved analytic p_current=f(q). No optimizer call,
map/affine/weight/matcher change, or ground-truth local warp/Jacobian claim.
Main observable: margin C(current)-C(truth), fraction strictly preferring truth,
paired per-case distributions, not simply lower absolute C_A. Preserve ALL328
IDs and original123/107/98availability. Flag descriptor/bilinear neighborhood
support separately: center inclusion alone is insufficient. Common-support
subset, if shown, needs explicitdenominator alongside retained full cohort.
This changes fixed affine orientation/scale AND raster interpolation/smoothing/
boundary effects; not a pure rotation causal ablation. Exactquarterturn controls
already establish channel mechanism but do not prove MIIT responsibility.
Falsifier: no better truth-vs-current ranking, or benefits confined to invalid/
border support, removes rationale for escalation. Lower absolute cost alone
does NOT trigger an optimizer pilot. Even positive ranking is diagnostic, not
proof of superior anatomical registration; earlier negativeafter_warp remains.
Smallest decisive test: exactquarterturn/identity rank and support fixtures,
then offline328center calculation and independent literal-frame recomputation.
Independent Astra approves this single diagnostic, not a new optimizer branch.

### T+16.4h frame ranking: weak/mixed signal, one exploratory pilot conditionally approved

Offline probe retainsALL328IDs andall328nominalcommon-support neighborhoods:
raw vs frozen-affine margin means-.00202690/+.00844876, stricttruthpreferences
160/328vs174/328. Perpair2-to3margin-.00260777to+.01361308,truth61to72;
7-to8-.00243468to-.00298207(WORSE),truth49to50;10-to11-.00085264to+.01444760,
truth50to52. This is weak/mixed development evidence, not a robust classifier
or anatomy improvement. Nominalf64support flags are not an exactf32access proof.
Independent17tests6.35s plus separateNumPypatch/sampling implementation agree
costs5.96e-8/margins1.19e-7 andallsyntheticranking signs;400literal enumerated
footprints match. Actual328source/recompute audit pending.

Research card — ONE exploratory shared-affine frozen-descriptor pilot:
Question: does the specific known-frame correction improve actual registration,
not merely sparse descriptor ranking, under unchanged safe optimizers/priors?
Definition: at each raster scale prewarp moving INTENSITY once byoriginal T_A,
using bilinearzeros/alignfalse, constructD_Aonce, then compare D_f(q)withD_A(f(q)).
Residual P1 andcompleteF=T_A composed withf, originalmachine-pointterm, fullfixed
foreground mask/OOBpenalty, ARAP/shape weights,300gradientbudget andE1-prefix
selection stayunchanged. Existing transport/original remainsdefault/primary.
No masked-descriptor repair: zero prewarp intensity mayyielddescriptorones,
unlike original descriptor zero padding; freeze anddisclose thissupport difference.
Report full-domain/perforeground andper-scale support, notjust328landmarkflags.
ALLthreepairs andbothanalytic/F2, six fresh configs, no method-specifictuning.
This is explicitly exploratory ANDlabel-informed: the same328IDs influenced
variant choice and cannotprovide untouchedconfirmation. It is motivated by
the exact frame identity, not the weak53%preference statistic. IndependentAstra
conditionallyapproves AFTERpendingactualprobe audit andimplementationchecks.
Falsifier: mixed/negative mean/tail/worst harm =>retire thisEXACTvariant, no
nearby masking/weight/descriptorforest. Positive meansretainhypothesis only,
not newgeneralization/SOTA/finalneuraltraining success. New E_A and old E_R
are DIFFERENTfunctionals; doNOTcompare theirnumeric totals asmatchedE.
Smallestdecisivetest: identity/quarter-turn true-imageobjective+finite-difference
mapgradient, default-bitwise regression andwhole tinyoptimizer; then sixreal
unchanged-budgetregistrations. Featureprewarp/per-scaleconstruction INCLUDED
incompleteapplication timings. No TREiterate/stoppingselection or newNN.

Actual328probe nowindependentlyclosed: separateliteralCSV/PIL/NumPydescriptor/
sampling/genericP1coordinate audit agreesfourcosts8.94e-8, margins1.3411e-7,
coordinates2.22e-16, ALLranking signs andnominalsupportflags. Everyfootprint
pixel enumerated independently; all328commonflags true. Root28trajectory/frame
tests40.01s; newprobe author40PASS7.06s andindependent17PASS6.35s. FORMULATION51
recordsweak/mixed results andconditions. Git5d6d844already pushed includes
trajectory selfcontained figure andexactquarterturn probe; currentfeatureprobe
milestone follows. Buildernowowns ONLYminimaloptional shared_affine Evidence
path+focusedtests; rootwillown sixexploratoryconfigs/remotejobs. No objective
pilot hasstarted yet andoriginaldefault/results haveNOTchanged.

### T+17h: exploratory frame pilot completed; next capacity question

The six shared-affine-frame runs completed with 300 gradients, 310 trials,
332 objective calls and no failed trials each. Aggregate mean/p90 TRE improve,
but both methods worsen the 7-to-8 mean; its worst error remains approximately
53 canvas pixels. The predeclared mixed-result condition therefore retires this
EXACT variant from main-line adoption. Positive aggregate findings are retained,
not hidden or reclassified as an independent confirmation. Original-frame
results remain primary. A metadata-only fix distinguishes the unwarped loader
raster from the affine-prepared descriptor; original archived reports retain
their stale preprocessing flag and must be read with this explicit correction.

Research card — one labelled sparse-capacity witness, NOT registration:
Question: can the actual fixed 257-by-257, fixed-boundary, P1-ac, four-corner
eta=.001 class simultaneously fit ALL 107 available anatomical centers in 7-to-8?
Individual targets being inside the affine output range does not answer this.
Start from the archived original analytic map; freeze its original A and b.
For native annotations converted using the saved pixel-center canvas layout,
minimize J(Y)=(1/107) sum_k ||512*(A*f_Y(q_k)+b-y_k)||^2.
Use the existing analytic safe operator, 17/33/65/129/257 raw proposal levels,
x then y at each level, 30 Adam gradients per stage: exactly 300 gradients.
No image or landmark term changes in production; this separate oracle consumes
manual labels explicitly and may never become an initializer or teacher.
Assumptions: annotation IDs/frame are meaningful; frozen boundary and affine
are retained; actual saved four-corner ratios remain strictly above eta.
Smallest decisive test: identity/exact small target and gradient fixtures,
then this ONE real labelled fit. Report every point, mean/p90/max, distortion,
displacement and unchanged ORIGINAL image objective components afterward.
A valid witness with every error <=1 canvas pixel proves only simultaneous
sparse capacity at these centers. If original E1 also falls, it is an observed
same-objective optimization opportunity; if E1 rises, it quantifies this
witness's anatomical-fit cost, not unavoidable incompatibility of all good maps.
Falsifier/stop: after the one fixed budget, failure to fit is inconclusive;
do not infer impossibility or sweep mesh, boundary, weights or optimizer budgets.
Prior work: the repository's analytic coordinated update and P1 evaluation;
this is a diagnostic application, not a new registration or approximation theorem.

The one actual capacity witness succeeds: ALL 107 errors are <=.026773 canvas
pixels, mean .015168, p90 .021693, with final minimum corner ratio .0130772.
Counts are 300 gradients / 310 trials / 322 oracle loss calls / zero failures.
Fit time is 2.8979 seconds; this is explicitly manual-label fitting, NOT
image-only registration. Original E1 rises .379627 to .589959; most of this
particular witness's cost increase is its cumulative ARAP distortion. It does
not prove every accurate map has that cost or that a regularizer must be changed.

Research card — ONE matched analytic versus corrected-F2 oracle comparison:
Question R2: removing correspondence ambiguity, do coordinated updates realize
this large actual sparse motion more efficiently than the existing F2 mechanism?
Freeze the original 7-to-8 incoming analytic map, original affine, ALL 107
available centers, squared oracle loss, 257-square P1-ac, fixed boundary and
eta=.001. Analytic uses five coefficient levels x/y, 30 gradients per stage;
F2 uses the ORIGINAL two five-level cycles, 30 vector gradients per stage,
patch_cells=8, accepted_gain=1 and floor reserve=.95. Both total 300 gradients
and 310 trials; parameter counts and internal geometry passes differ and must
be reported, not hidden as equal mechanisms. No priors in fitting; original E1
is evaluated only afterward. This is a NEW controlled R2 question after the
successful capacity fit, not a continuation or retuning of that fit.
Measure first CERTIFIED accepted stage with maximum error <=1 canvas pixel,
final all-ID mean/p90/max, actual gradient/loss/decoder-pass counts and costs.
Use one warm-up each then one fixed ABBA block (analytic,F2,F2,analytic), retaining
every output. No rate, patch, budget, objective or label subset sweep. Timings
are descriptive paired measurements on one known task, not general GPU statistics.
Decision: reaching the same threshold at lower cost, or reaching it when the
control does not within budget, supports a task-specific motion-efficiency
advantage; F2 matching/beating analytic narrows that claim. Either outcome
does NOT improve production image registration or justify an image-loss change.
Falsifier/stop: finish this ONE paired block and stop regardless of result;
no oracle map becomes a production initializer, training teacher or inference score.

### T+17.7h: paired oracle collection retained a real F2 protocol failure

One warmup each and the declared ABBA collection finished. Analytic completes
300/310/322 and reaches the first CERTIFIED all107<=1px accepted state at stage4,
150 gradients/155 trials. The two measured threshold times are1.0783/1.1226s.
F2 attempts stop individual inner loops at actual rounded eta-floor rejections:
warm180/190/202, measured165/175/187 and188/198/210 gradients/trials/J calls.
Their retained endpoints are legal, but maximum errors remain37.0--37.5px;
no threshold is attained. All three incomplete budgets are marked FAILED,
not discarded or rerun with a weaker floor. This is an implemented-protocol
task-specific motion-efficiency result, NOT completed equal-budget comparison,
general F2 incapacity, image-only registration, or a finite speedup ratio.
Independent endpoint/threshold and failure-source checking is underway.

Research card — ONE label-free initializer restriction diagnostic:
Question: does the current four-parameter positive SIMILARITY force global
shear/anisotropic scaling into the fixed-boundary residual and its ARAP prior?
All current saved 'affine' initializers are similarities by source construction.
Use ONLY the same original frozen image-machine correspondences, not manual
labels, oracle maps, or a rematcher. Convert targets to original moving units
y_j=A_old*p_j+b_old; freeze original confidence and original-domain eligibility.
Fit a similarity and an unconstrained six-parameter affine with the SAME loss
sum_j w_j [sqrt(1+||512*(A*q_j+b-y_j)/8||^2)-1]/sum_j w_j.
This is a small convex robust regression in4 or6 coefficients, not a large
mesh solve. Verify rank, stationarity and positive rounded determinants; a
nonpositive affine fails rather than being projected or silently replaced.
Smallest decisive test: exact similarity/full-affine synthetic fits, analytic
gradient/IRLS stationary checks and degenerate/failure fixtures, then ONE fixed
two-fold spatial split parity=(floor(4*qx)+floor(4*qy)) mod2 across all3pairs.
Train each model on one parity and evaluate all eligible opposite-parity points;
reverse once. No split/grid/ridge/loss/threshold sweep. Machine confidences are
not anatomical truth and this is not independent-patient validation.
Advance only if full-affine held-out robust loss improves BOTH directions on
ALL three pairs, with full-rank fits and positive float32 stored determinants.
Otherwise stop this EXACT branch. Training loss alone cannot justify it.
If supported, compare RE-FITTED similarity versus full affine, both fitted once
on all same eligible points, across analytic/F2/nativeDHR with the original
transport features, weights,300safe-gradient recipe, declared geometry and
identical initializer within each arm. Re-express the SAME world correspondences
under the new affine, with no rematching, point dropping, or manual selection;
report any residual-target-domain incompatibility rather than silently excluding
points. This would change initialization and residual-prior interpretation,
not be an equal-functional geometry ablation. Original results remain archived.
Prior work: ordinary positive Procrustes similarities/full-affine robust fitting;
new contribution, if any, would be useful integration, not a novel affine solver.
This is the only next alternative application branch, approved by the independent
decision agent after excluding already retired neighboring interventions.
