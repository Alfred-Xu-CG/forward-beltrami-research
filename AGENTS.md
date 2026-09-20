# AGENTS.md — Computational QC Research Mode
## Mission

This repository is in **research convergence mode**.

The goal is a fast, memory-efficient, differentiable neural layer that maps a learnable latent representation to a **guaranteed-bijective discrete deformation**, while retaining quasiconformal / Beltrami geometry when useful.

Authoritative plan:

```text
docs/research_phase2/PLAN.md
```

Legacy route documents and artifacts are evidence and reusable code, not the current agenda.

## Current priorities

Only these are primary:

1. Mixed-Boundary Modulus Linear Beltrami Solver (MBM-LBS).
2. M-matrix / positive-convex-combination conditions and mixed-boundary monotonicity.
3. Primal-dual / electrical-network QC rectangle mapping.
4. Directed positive Tutte weights as a hard-bijective latent coordinate system.
5. Fixed-P1 compatibility/projected-Beurling only as supporting exactness theory.

Do not expand old A–N routes merely because they are marked partial.

## Research method

Prefer:

1. precise mathematical formulation;
2. proof or smallest counterexample;
3. small decisive numerical experiment;
4. implementation;
5. medium benchmark;
6. large benchmark only when earlier steps justify it.

Negative results are progress.

When a core idea fails twice independently, escalate to a frontier reasoning model before abandoning it.

## Anti-bureaucracy

By default, DO NOT add:

- checksums or SHA manifests;
- custom artifact-integrity systems;
- frozen contracts or schema freezes;
- new baseline frameworks;
- workflow/completion/release gates for ordinary research;
- duplicate audit layers;
- per-run artifact directories;
- speculative infrastructure unrelated to a concrete failure.

Only add one when there is a concrete failure scenario and Git/versioning, types, ordinary tests, and normal constraints are insufficient.

Do not remove existing safety mechanisms merely to simplify the repository.

Workflow gates belong only at irreversible, destructive, cross-system, security, production-data, or formal-release boundaries.

Preflight checks must not crowd out actual code execution, simulation, or measurement.

## Skills

If these skills exist:

- Do **not** use `ars/experiment-agent` during this open-ended research phase.
- Do **not** use `academic-paper` until conclusions are stable.
- Use `academic-research-suite` only lightly for theorem/prior-art verification.
- Use `ars/deep-research` only when a core theorem/prior-art problem cannot be resolved with targeted search.
- Use `verification-before-completion` only at real phase milestones/final handoff.
- Use `systematic-debugging` only for actual software bugs/test failures.
- Do not repeatedly invoke `writing-plans` or `brainstorming`; the current plan already exists.
- Keep `using-superpowers` to minimal routing.

## Experiments

Every experiment must answer a named research question.

Scale order:

- tiny/local first;
- medium only after tiny correctness;
- 512²+ only after medium correctness and accuracy.

Default budgets:

- tiny: ~2 min;
- medium: ~10 min;
- large: ~30 min unless a written high-information reason justifies more.

Use one compact results table. Do not create one audit directory per run.

## Remote compute

Three remote hosts may be available through a VPN.

- Use remote compute for meaningful medium/large jobs, not tiny sanity checks.
- On SSH failure: retry at most twice over ~2 min, mark host temporarily unavailable, and continue locally.
- Do not pause the objective waiting for VPN restoration.
- Recheck at phase transitions or after ~45–90 min.
- Long jobs may use `tmux`/`screen`/`nohup` and concise logs.
- Never kill unrelated processes or modify shared services/ports.

## Model routing

Minimum model baseline: **GPT-5.6**.

- routine implementation: GPT-5.6 Sol medium;
- complex sparse/adjoint/DEC implementation: GPT-5.6 Sol high;
- routine literature extraction / experiment summary: GPT-5.6 Terra medium;
- mechanical log parsing / utility tests: GPT-5.6 Luna medium;
- hard mathematical reasoning / route redesign: GPT-6 Astra high;
- genuine conceptual impasse after two attempts: GPT-6 Astra xhigh/max.

Do not automatically fall back to models below GPT-5.6.

Do not use maximum reasoning for routine work.

## Git and files

Use Git as normal version history. No extra hashing layer.

Meaningful milestone commits are enough.

Do not refactor legacy code merely for cleanliness.

Prefer isolated new modules and minimal corrections to confirmed bugs.

Temporary logs may remain untracked.

## Scientific integrity

Always distinguish:

- theorem vs numerical evidence;
- continuum guarantee vs fixed-mesh guarantee;
- local positive Jacobian vs global homeomorphism;
- exact Beltrami reproduction vs metric/harmonic approximation;
- solver residual vs induced Beltrami error.

Do not claim `|mu|<1` alone guarantees a sampled P1 map is fold-free.

Do not claim a continuous diffeomorphism implies sampled P1 interpolation is fold-free.

Do not use post-hoc fold repair as the primary topology guarantee.


