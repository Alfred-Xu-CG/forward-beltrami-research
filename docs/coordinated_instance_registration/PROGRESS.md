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
