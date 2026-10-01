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
