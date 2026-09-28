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
