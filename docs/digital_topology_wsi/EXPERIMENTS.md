# E0/E1 experimental record

This is a live record, not a G1–G4 verdict. The current branch is `codex/digital-topology-wsi`; the real-time window began at the UTC value in `START_TIME.json`. All persistent results remain in this D: repository. The plan and independent proof/code review are in `PLAN.md`, `TOPOLOGY.md`, and `REVIEW.md`.

## Named question E0: Do old P1 or center-only checks certify the deployed Q1 map?

No. The exact rational fixed-boundary fixtures in `TOPOLOGY.md` have respectively (A) all old SW–NE P1 triangles positive while one Q1 corner is negative, and (B) positive center checks while Q1/P1 have negative corners. A separate checker reproduced the arithmetic without calling the production determinant helper. These are logical counterexamples to transferring an old certificate; they do not indicate a defect in the old P1 theorem.

## Named question E1: Do the first F1-D/F2-D implementations preserve Q1 corners and propagate their actual gradients on tiny grids?

The newly added `digital_q1.py` implements four-corner Q1 evaluation, exact dyadic Q1 nodal refinement, F1-D 12-constraint colored updates, and F2-D four-corner patch updates. A single patch across the domain now runs one pass because no shifted seam exists; larger grids use four staggered passes. `validate_q1_map` audits the supplied actual tensor in bounded row chunks and never silently substitutes an identity map. It does **not** provide an outward-rounded IEEE sign proof.

On local Windows Python 3.12.4 / PyTorch 2.5.1+cpu, the command below returned **12 passed** (2026-09-28 UTC):

```powershell
$env:PYTHONPATH=(Resolve-Path -LiteralPath 'src').Path
python -m pytest tests/test_digital_q1_geometry.py tests/test_digital_q1_independent_checker.py tests/test_digital_q1_benchmark_smoke.py -q
```

The independent checker compared a separate NumPy Q1 derivative enumeration with the torch output, confirmed float64 F1-D/F2-D directional VJPs by three central-difference step sizes, and rejected a deliberately detached F2 safety scale. An additional pre-existing regression command (with `src;tools` on `PYTHONPATH`) returned **25 passed** for the old F1/P1 and F2/P1 suites; the new code does not reinterpret old checkpoints.

These tiny tests support an implementation claim on the tested cases only. The exact-arithmetic theorem assumes an initially valid Q1 map and a simple ordered boundary. Current update classes assume a **square, uniform unit source grid**. Their optional minimum Jacobian floor is conditional on the input already meeting it. Nonfinite inputs, saved/cast outputs, and uncertain floating-point corner signs require separate deployment checks.

## Next medium-scale measurement protocol

`tools/digital_q1_benchmark.py` creates an identity map on an actual `N×N` control-vertex grid, one batch of deterministic Gaussian latent proposals (seed 290929, logit standard deviation 0.15), and a fixed random cotangent. Mode `f1` applies four sequential vertex colors to **one** latent field; mode `f2` applies four shifted patch passes with **four** latent fields. These are distinct workloads, not a same-parameter architecture ranking. The script warms once, then reports synchronized medians of pure geometry forward and VJP-to-all-latents separately, minimum of all `4(N-1)^2` unnormalized Q1 corner determinants, nonpositive/nonfinite count, boundary error, gradient finiteness/magnitude, and CUDA allocated/reserved peaks. It excludes image encoder, I/O, final warp, saving, and any true registration metric.

At the initial remote read-only check, `ai-codex-mihomo-codex` was reachable with `ClearAllForwardings=yes`; GPUs 0, 2, 6, 7 were RTX A6000 49,140 MiB total with 48,660 MiB reported free and 0% utilization, while GPUs 1, 3, 5 were busy. This was an instantaneous snapshot, not a standing reservation. GPU process IDs were read and no other process was stopped. The `cuda124` environment provides PyTorch 2.5.1+cu124. Occupancy will be rechecked immediately before launching medium jobs.

## Real-data status

The data scout found no verified local HyReCo, ACROBAT, or ANHIR pair. The local MIIT-v4 archive contains single-image records but no association/landmark/grouping manifest; it cannot be called a registered source–target pair. Dataset and baseline availability are detailed in `EVIDENCE.md`. **No real pathology registration or external baseline has run yet**; G2–G4 remain untested. This absence cannot be filled by the existing synthetic Phase VII experiments.
