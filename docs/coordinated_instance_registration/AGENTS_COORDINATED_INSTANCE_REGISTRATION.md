# AGENTS.md — Coordinated Instance Registration

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

Use only `gpt-6.1-sol` and `gpt-6-astra` when actually supported by this runtime.
Coordinator and routine implementation: Sol medium.
Mechanical extraction: Sol low. Complex implementation/debug: Sol high.
New load-bearing geometry and ambiguous scientific pivots: Astra high.
Use Astra xhigh only for a specific unresolved decision; max is exceptional.
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

Use the runtime's actual goal duration; default to the PLAN's 24-hour window.
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


