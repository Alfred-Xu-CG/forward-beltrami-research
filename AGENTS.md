# AGENTS.md — Coordinated Instance Registration

Current state: PAUSED BY USER after handoff, recorded2026-10-02 16:11:57 UTC.
Do not launch new research jobs until the user explicitly resumes or authorizes
a next phase. This is not a claim that the scientific objective was achieved.

## User-approved restart and extension — 2026-10-02

This section overrides older model routing and duration text, including the
archived phases. The user resumed the paused goal and added FIVE hours to the
original window: start 2026-10-01 11:26:23 UTC, extended deadline
2026-10-02 16:26:23 UTC (29 hours total; 2026-10-03 00:26:23 Asia/Shanghai).
Do not restart the old clock or silently add agent time to elapsed wall time.

ALL newly launched or resumed research agents must actually use `gpt-6-astra`.
Use medium for bounded extraction/routine implementation, high for protocol,
numerical implementation and independent checking, and xhigh for the present
unresolved optimizer redesign. Max requires a specific additional justification.
The coordinator cannot change its own runtime model through a role prompt; do
not claim that it did. Old Sol agents must not resume under this instruction.

The restart priority is: define comparison methods and metrics; verify the
actual preprocessing/evaluation against original methods; then implement and
measure a justified optimizer improvement on real images. Separate matched
mechanism ablations from full native-pipeline comparisons. A 512-pixel reduced
configuration is not automatically a faithful RegWSI reproduction. Keep this
review bounded and actionable; do not substitute further oracle diagnostics or
an audit framework for real registration work. The pending nullspace oracle is
deferred. Existing negative results remain evidence, not permanent prohibitions
on scientifically justified redesign. Report accuracy/cost tradeoffs and paired
case results; tiny per-case changes alone do not justify a universal rejection.

## Current objective

Read `docs/coordinated_instance_registration/PLAN.md` as the current project plan.
Legacy documents are evidence, not a queue of unfinished work.

The immediate objective is accurate, fast **per-case image registration** whose
actual exported map satisfies the declared digital-topology conditions. A new
neural network, inverse consistency, or a new infrastructure framework is NOT a
prerequisite for this phase.

## Priorities

Hard topology is a mandatory feasibility condition.
1. Correct correspondence / anatomical accuracy on real images.
2. Hard topology of the declared output, including boundaries and interpolation.
3. Time-to-accuracy and usable memory.
4. Understand whether remaining error is representation, optimization, or evidence.

Start real-data execution alongside geometry work; do not postpone it until all
proofs or synthetic experiments are complete. Reuse available data and working
baselines. Do not silently treat previously viewed specimens as blind tests.

## Active work

Main: multiscale coordinated feasible instance optimization; compare radial latent
and analytic feasible-step entry on identical geometry.
Controls: existing F1/F2 instance optimization with the SAME input evidence,
initialization, objective, boundary class, and comparable compute budget.
Supporting baseline: an executable, correctly configured RegWSI/DeeperHistReg.
At most ONE alternative mechanism may be active when a diagnosed bottleneck
justifies it. F1/F2 remain useful local/block components, not mandatory winners.

## Non-goals

Do NOT optimize inverse/cycle consistency, undertake inverse-field generation,
train a new large network, build a new SLAM-like matcher, develop Stripe/
Progressive/Yee/DEC frameworks, or perform enormous-resolution demos by default.
Do not add more metrics just because they are customary. Existing cheap metrics
can be retained, but cannot create a new research branch.

## Anti-rabbit-hole rule

Before spending substantial time, answer in a few sentences:
What decision can this task change? What is the smallest discriminating test?
What result would redirect the work? If no answer, do not do the task.

After two materially different failed interventions on the same hypothesis,
diagnose and escalate; do not repeat nearby hyperparameters indefinitely.
Every 90–120 minutes, or after three informative experiments, make a brief
continue/change/pause decision. No meeting-style report is required.

## Models and delegation

Use only `gpt-6-astra` for delegated research work under the current restart.
Routine implementation/extraction: medium; protocol and complex implementation:
high; the current unresolved optimizer redesign: xhigh; max is exceptional.
Verify the actual role configuration once. Never pretend that a prompt switched
the model or created an independent agent. Do not silently use another family.

At most two builders plus one temporary checker. No recursive delegation.
Give each worker a bounded question and relevant files, not the full archive.
Checkers do not approve their own code or merely repeat the author's conclusion.

## Checking that matters

Independently check new topology formulas, boundary/seam coverage, gradients of
new operators, map direction and coordinate units, and final empirical claims.
Run focused tests after edits and a final affected-suite check; do not repeatedly
run the entire legacy suite. Agreement between agents is not a proof.

The production map and evaluator must specify P1 or Q1. Four-corner positivity
supports both interpretations, but does not make the two functions identical.
Never assume arbitrary resampling preserves the guarantee.

## Experiments and optimization

Fix the accepted map as a stage anchor while optimizing that stage's latent.
Only commit a candidate after applying the declared safe operator; then refresh
geometry/evidence as needed. Do not accidentally compound trial candidates during
one latent solve. Per-case optimization does not need a graph through its entire
optimization history. Test the local decoder gradient separately.

Keep the same regularized objective for acceptance comparisons. A lower image
proxy is not proof of better anatomy. Never select a blind-test iterate by its
manual landmarks. Include all eligible failures and timeouts.

## Time, compute, and safety

Use the user's latest authorized duration: the extended 29-hour window above.
Record real UTC start/deadline/end in an existing simple run record. Do not invent
T+24 labels, add token times to wall time, sleep to fill the window, or mark an
interrupted session complete. A Markdown file cannot extend a host's execution
limit: report interruption honestly and leave the next command.

Use local small experiments if remote VPN/SSH fails. Retry briefly, then return
to useful local work. Do not disrupt other users' GPUs, ports (including 18082),
processes, or VPN settings. Never bypass approval or network/security controls.
No extra paid model/compute service beyond the user's authorized environment.

## Minimal process

Use Git, normal tests, one result table, and short decision notes. No new hashes,
manifest system, frozen-contract framework, benchmark framework, or per-run audit
forest without a concrete failure ordinary tools cannot address. Do not delete
existing safety mechanisms. A comparison method is welcome; a new 'baseline
framework' is not. Keep existing necessary data/model artifacts.

Do not read, invoke or follow installed skills/workflows in this round, per user.
Use direct code, derivations, targeted primary sources and ordinary experiments.
Higher-priority tool, security and workspace rules still apply.

## Handoff

Separate: geometry established; useful instance optimizer; evidence bottleneck;
real-data competitiveness; formal SOTA not established. Never equate tests passed,
wall time consumed, or a theorem alone with successful registration. Finish with
actual results, the strongest adverse result, and at most three next actions.



# Approved revisions — 2026-10-01

This section is authoritative over conflicting inherited wording below.
The user authorized actual execution for 24 hours, with possible later extension.
Active checkout: D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan.
Branch: codex/coordinated-instance-registration, initial base 4ca9f09.
Window start 2026-10-01 11:26:23 UTC; deadline 2026-10-02 11:26:23 UTC.

1. Main line is multiscale coordinated feasible INSTANCE OPTIMIZATION. Compare radial
   latent and analytic feasible-step entry using identical exact four-corner constraints.
   For proposal p with slack s>0, alpha_max=min_{(Cp)_k<0} s_k/(-(Cp)_k), infinity if none.
   Choose alpha=min(alpha_trial,theta*alpha_max), 0<theta<1, and candidate=Y+alpha*p*e.
   Image-objective acceptance is separate. No inner geometry line search or QP is needed.
   Measure the extra margin restriction; radial surjectivity does not ensure good conditioning.
2. Normalize coefficient scales or calibrate physical proposal displacements per level.
   Equal raw Adam rates are not automatically fair: P_l^T aggregates different supports.
   A diagonal P_l^T W P_l preconditioner is optional, not a new required framework.
3. Run a small existing-objective versus RegWSI-style-objective comparison early, alongside
   geometry, because prior proxy/anatomy conflicts have already been reported. Every method
   within each geometry ablation shares objective/evidence/initialization/physical units.
4. Synthetic benefit without real benefit means "geometry benefit not yet transferred".
   It is NOT automatically an evidence bottleneck. First distinguish target-class mismatch,
   directions, initialization, boundary, image resolution, regularization and actual convergence.
5. Specify tissue/background masks, out-of-bounds queries and noncorresponding tissue.
   Do not lower loss by dropping difficult tissue or shrinking effective overlap. Preserve
   fixed eligible-case and required-landmark evaluation denominators. Reuse existing masks.
6. Predeclare dataset, metric/units, pair/specimen aggregation, tail/failure measure and budgets.
   ACROBAT 2023 official-compatible score is mean of per-pair 90th-percentile TRE.
   Missing hidden labels mean official competitiveness NOT TESTED. Existing untouched labelled
   independent specimens can be confirmation; new downloads are not mandatory.
   Thresholds 2x/25%/10%/5% are guides; small means alone do not establish noninferiority.
7. Local/global supports, normalization, limited extra directions, schedules, radial versus
   analytic-step are ordinary main-line variants. At most one DIFFERENT mechanism is active
   at once after diagnosis; sequential replacement is allowed.
8. Hard topology is a mandatory feasibility condition. q_ref means the reference determinant,
   twice signed triangle area, so identity q/q_ref=1. Declare interpolation and validate
   the actual exported map; nonaligned resampling does not inherit an old certificate.
9. Roles: coordinator integrates evidence and controls changed variables; geometry/optimizer
   builder owns operators/stages/scales/supports; application/baseline builder owns datasets,
   groups/common evidence/RegWSI/real experiments/performance; temporary checker independently
   derives/recomputes critical results read-only. At most two builders plus one checker,
   no recursive delegation or overlapping file edits. Use real runtime model selection.
10. All local code, logs, caches, outputs and reports stay on D:. Research SSH MUST use
    ClearAllForwardings=yes for each existing alias; preserve occupied port 18082 and others'
    jobs. Recheck GPU processes/capacity before a job. No installed skills are read/invoked/
    followed in this round, per the user; use direct code, derivations and primary sources.
11. Time tables are adaptable; don't abandon a decisive high-information test merely to meet
    an intermediate slot. Use actual elapsed time; no premature closure after a successful pilot.

---
## Archived phase instructions
The material below records completed stages. Current priorities, skills, roles
and goal timing are governed by the coordinated plan above; applicable safety remains.

# AGENTS.md — F1/F2 Digital-Topology WSI Research
+
## Mission and source of truth

Current plan: `docs/digital_topology_wsi/PLAN.md`.
Goal: useful high-resolution 2D pathology registration with a digital-output-consistent homeomorphic map, not merely a zero-fold toy decoder.
Both F1 and F2 remain primary. Their schedule and the usefulness of learning must be experimentally established.

## Scope and honesty

Separate G1 geometry, G2 benefit of learning, G3 competitive real-data performance, and G4 dataset/protocol-specific SOTA. Passing tests, using the time budget, or returning identity does not establish G2–G4.
No claim of exhaustive literature coverage or error-free agents. Cite verified sources and distinguish new derivations from established results.
Do not revert to broad legacy Beltrami/Tutte/BHF route exploration unless a specific current bottleneck justifies one targeted alternative.

## Geometry

Default deployed map is Q1 on a declared rectangular grid. Protect all four corner determinants of every affected cell, not only the original SW–NE triangles.
Require consistent boundary, shared-edge values, physical coordinates, and saved-output validation.
F1 uses all incident constraints and independent colors; F2 uses all affected-cell quadratic bounds and nonconflicting patch passes.
Never blend accepted maps or silently resample them under an old certificate. Features/raw proposals may be blended before a valid update.
Report almost-everywhere differentiability, active scaling, rejections and fallbacks. Never detach a safety scale while claiming the true gradient.

## Experiments

Start real pathology baseline/data work in parallel with tiny geometry tests.
Compare no-network optimization, network prediction, and hybrid refinement using the same safe layer.
Use patient/block/physical-slide grouping; never split patches from one physical slide across training and test.
Never use test landmarks, true target boundary, or target maps for inference or model selection.
Use official evaluation conventions. Keep failure cases in the denominator. Do not call an unavailable hidden-test comparison SOTA.

## Models and delegation

GPT-6 only. Default coordinator/coding: `gpt-6-sol` medium; complex implementation/checking: Sol high; extraction: `gpt-6-luna` low/medium; core math and independent math review: `gpt-6-astra` high.
Use xhigh/max only for a focused unresolved decision-critical conflict after two distinct attempts. Never claim unavailable model routing or fake independent agents.
At most three concurrent focused agents. Return short evidence summaries, not whole log dumps.

## Independent checks

Independent review at geometry merge, formal evaluation setup, and final scientific claim. Checker uses separate critical formulas/code paths and at least two deliberate failure fixtures.
The author is not their own only reviewer. Agreement between agents is not proof; mathematical derivation and independent computation are both needed.

## Autonomy and resources

Autonomously edit, test, measure, retry and stop your own over-budget experiments within permissions. Preserve other users' files, jobs, ports and security rules.
VPN failure: two brief retries, then local/theory/CPU work; recheck at natural transitions, not continuously.
Use real elapsed time for the default 24h window; do not fabricate progress timestamps, sleep to fill the window, or equate elapsed time with scientific success. If the environment ends, report a resumable partial state.

## Keep the process light

Do not add hashes/manifests, frozen contracts, duplicate auditing platforms, per-run permanent folders, or speculative infrastructure by default. Use Git, ordinary configs/tests, concise results and saved scientific outputs.
Do not delete necessary existing safety measures. Respect instruction hierarchy.
Optional `ars/experiment-agent` and premature `academic-paper` workflows are not appropriate; targeted literature and real-bug debugging are. Disable optional workflows only through supported project-local configuration, never pretend or override mandatory controls.

## Final decision

Read actual saved maps and evaluation outputs. Report G1/G2/G3/G4 independently as PASS/FAIL/NOT TESTED, strongest counterexample, best viable architecture, baseline coverage, and at most three next tasks.
A strong alternative outperforming F1/F2 is a legitimate research finding, not a result to suppress.

## Archived Phase VII and Phase VI instructions

The material below records prior stages only. The current phase above and `docs/digital_topology_wsi/PLAN.md` govern this work; retain prior safety requirements where compatible.

# AGENTS.md — Phase VII Forward P1 Homeomorphism Research Mode

> Phase VII research window ran from 2026-09-23 17:52:45 UTC to 2026-09-24 14:52:56 UTC. Authoritative scope: `docs/research_phase7/PLAN.md`; clock: `docs/research_phase7/START_TIME.json` and `docs/research_phase7/END_TIME.json`. The technical report is `docs/research_phase7/REPORT.md`. Phase VI material below remains archived evidence.

Phase VII prioritizes a latent-to-fixed-grid-P1-homeomorphism decoder built by coarse-to-fine forward updates. Separate all-latent topology proofs from approximation and image-training evidence. The 21-hour window has no route-specific hour quotas; at its end, stop new large jobs and synthesize honestly rather than declaring technical success by time. Keep all artifacts on D:, preserve unrelated files and shared host resources, and synchronize meaningful Git milestones.

## Archived Phase VI instructions

> Phase VI's 18-hour window ended on 2026-09-23 at 12:59:58 UTC; see `docs/research_phase6/END_TIME.json`. The instructions below record the completed Phase VI run, including its user-specific no-installed-skills rule. They do not silently define a new research phase.

## Mission

The completed Phase VI research objective was to investigate a fast, memory-efficient, differentiable neural layer that maps multiscale latent variables to a piecewise-affine homeomorphism on a dense triangulated rectangle. The authoritative plan and timing are:

```text
docs/research_phase6/PLAN.md
docs/research_phase6/START_TIME.json
docs/research_phase6/END_TIME.json
```

Explore all three routes: A, explicit dense-grid monotone construction; B, local patch maps and exact PL composition; C, multiscale positive conductances and structured solves. Woodbury is one candidate within C. Theory and counterexamples guide implementation but do not replace it. At least two distinct mechanisms should reach actual 257×257 control-vertex forward/VJP tests. Do not stop other routes after one succeeds.

## User override: no installed skills

Do not read, invoke, or follow any installed skill or its workflow during this Phase VI run. This user instruction overrides earlier skill-routing advice in Phase V documentation. Use direct reasoning, repository code, targeted primary literature, and actual experiments.

## Model and time

The user selected GPT-6 Sol as the execution model. Use medium reasoning for routine implementation and experiments, high for hard topology, sparse adjoints, and algorithm design. Record actual wall-clock progress from the Phase VI start time. Do not reuse Phase V start or completion files. At around 16.5 hours independently check major results and write the self-contained final report. Do not label the phase complete before 18 hours unless the user explicitly stops or the goal is otherwise terminated by system controls.

## Scientific claims

Distinguish a fixed-grid P1 homeomorphism from an exact composition of PL homeomorphisms: the composition need not be P1 on the original grid. Resampling its vertices and reinterpolating does not inherit the composition theorem. Separate theorem, implementation, finite-precision evidence, and untested hypothesis. SPD, positive conductances, small residual, positive sampled area, and global injectivity are different claims. State boundary and triangulation hypotheses. Never count post-hoc repair as the primary topology guarantee.

## Research method

Every experiment answers a named question. Begin with a precise construction and a small correctness test, then implement a full forward/VJP, then scale to 129×129 and 257×257 control grids. Include an actual multi-step image-to-latent training run at the main scale for at least one candidate. Compare against the existing symmetric-CG and sparse-direct baselines using identical quality requirements. Measure complete forward, VJP, dense query, training step, and peak memory. Label control vertices and image pixels separately. Preserve failures and negative results.

Keep work focused on implementable solutions. After two reasonable fixes to a local failure, analyze and redirect effort; do not spend the phase only deriving obstructions. A route is not closed from literature alone.

## Remote compute and ports

Use idle CPUs and GPUs on `turing-codex-mihomo-codex`, `element-codex-mihomo-codex`, and `ai-codex-mihomo-codex` for meaningful medium/large experiments. Check active GPU processes, free VRAM, and CPU load before jobs. Do not preempt or kill unrelated processes. The existing aliases configure `RemoteForward 18082`; research SSH calls must specify `ClearAllForwardings=yes` so they do not bind that occupied port. Keep the original SSH configuration unchanged. Retry an SSH failure at most twice, then continue elsewhere and recheck later.

## Files and communication

Keep all new code, results, logs, and reports on D drive in this repository. Use ordinary Git commits and sync meaningful milestones to GitHub. Do not add checksum systems, per-run artifact trees, speculative infrastructure, or duplicate audit gates. Keep a compact shared results table and a self-contained report defining formulas, experimental setups, metric calculations, observed results, and limitations. Give the user a concise evidence-backed progress update about every 2–3 hours of ongoing work.
