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
