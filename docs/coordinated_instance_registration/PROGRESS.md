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
