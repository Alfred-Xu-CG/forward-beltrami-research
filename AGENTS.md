# AGENTS.md — Phase V Sequential Deep-Research Mode

## Mission

This repository is in **Phase V sequential deep-research mode**.

Authoritative plan:

```text
docs/research_phase5/PLAN.md
```

Goal: develop a fast, memory-efficient, differentiable neural layer producing a guaranteed-bijective discrete deformation.

Routes must be explored **sequentially and deeply**:

1. Fast Tutte neural layer.
2. MVC canonical coordinates + geometric incremental optimization.
3. Primal-dual / RT / Whitney / Hodge.

Do not start the next route until the current route passes its independent checker.

## Real wall-clock requirement

This is an 18-hour real wall-clock phase.

Use:

```text
docs/research_phase5/START_TIME.json
tools/check_phase5_elapsed.py
```

Do not simulate `T+Nh` labels.

Do not mark complete before real elapsed time reaches 18 hours.

A successful result is not a stop criterion.

## Research depth

Before a core theorem, solver design, or experiment, state:

```text
Question:
Precise claim/hypothesis:
Assumptions:
What falsifies it:
Smallest decisive test:
```

Prefer:

1. precise formulation;
2. proof/counterexample;
3. independent check;
4. minimal decisive code;
5. realistic benchmark;
6. engineering optimization.

## Independent checker

Each route requires an independent checker.

High-risk words require review:

```text
unique
iff
necessary
sufficient
universal
exact
guaranteed
canonical
bijective
```

The builder may not be the sole reviewer.

## Model routing

Minimum baseline: GPT-5.6.

- coordinator: GPT-5.6 Sol high;
- core implementation: GPT-5.6 Sol high;
- utilities/tests: GPT-5.6 Sol medium;
- hard mathematics: GPT-6 high if available;
- independent theorem checker: GPT-6 high/xhigh if available;
- conceptual impasse after two attempts: highest available GPT-6 reasoning, otherwise GPT-5.6 Sol maximum effort.

Do not automatically fall below GPT-5.6.

## Route order

Route I Tutte must close before Route II MVC.

Route II MVC must close before Route III primal-dual.

No parallel route exploration.

Independent checker work for the current route is allowed.

## Experiments

Use one shared benchmark.

Mandatory realistic scale:

```text
256×256 image/query
```

512×512 preferred when resources allow.

Always distinguish control mesh resolution from dense image/query resolution.

Every route must execute code and produce numerical evidence; literature-only exploration is insufficient.

## Tutte rules

Audit TutteNet paper, official code, and `torch_sparse_solve`.

Implement and benchmark:

- reference direct solve;
- improved direct/factor reuse where possible;
- matrix-free iterative solve;
- custom implicit backward;
- fixed-query dense-warp precomputation.

Do not call CPU reference code production-ready.

## MVC rules

MVC is primarily a canonicalization / optimization-geometry route, not a separate hard decoder.

Distinguish:

```text
D(E(Y)) = Y
```

from the false general claim:

```text
E(D(p)) = p.
```

Analyze redundancy, decoder Jacobian nullspace, covariance lift, canonical re-encoding, trainability, and incremental correction.

Treat covariance lift as one latent gauge/right-inverse, not the derivative of MVC unless proved.

## Primal-dual rules

Do not stop at “DEC/RT is promising”.

Implement at least one concrete compatible reference operator and one positive hard-valid approximation/hybrid.

Do not equate:

- conservation with bijection;
- SPD with M-matrix;
- positive wide stencil with planar graph;
- fixed-direction failure with route failure.

Quantify anisotropy/accuracy vs positivity/topology.

## Anti-bureaucracy

Do not add by default:

- hashes/checksums;
- manifests;
- frozen contracts;
- completion ledgers;
- per-run artifact trees;
- speculative frameworks.

Allowed gates:

- route checker;
- real 18h finalization clock;
- destructive/security/release boundaries.

## Skills

- do not use `ars/experiment-agent`;
- do not use `academic-paper` before final synthesis;
- use research/deep-research only for targeted prior-art/theorem questions;
- use systematic debugging only for real code failures;
- use verification-before-completion at each route closure and final review.

## Remote compute

VPN failure does not pause the objective.

Retry briefly, then continue locally.

Use remote compute for meaningful medium/large runs.

Avoid high-frequency polling.

Never interfere with unrelated processes or ports.

## Scientific integrity

Always distinguish:

- theorem vs evidence;
- exact vs approximate;
- expressivity vs trainability;
- control-map bijection vs dense-query evaluation;
- continuous diffeo vs sampled PL homeomorphism;
- directed positive Tutte weights vs symmetric conductance family;
- discrete exactness/conservation vs spatial injectivity.

Do not use post-hoc fold repair as the primary topology guarantee.

## Git and files

Use Git as normal version history without extra hashing infrastructure. Work on the
Phase V feature branch in the D-drive worktree. Prefer isolated modules and minimal
corrections to confirmed legacy bugs; do not refactor unrelated code. Keep temporary
logs under `tmp/`, and make meaningful milestone commits only after fresh verification.

