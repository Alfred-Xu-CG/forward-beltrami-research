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

### Actual remote geometry-only results, batch 1 float32

The first remote call failed before executing the geometry because the benchmark constructed its index-buffered layer on CPU while the tensors were on GPU. A side-9 CUDA reproduction gave the same error; existing image-training code showed the required `.to(device)` pattern. After adding an optional CUDA smoke test and fixing **only that device placement**, both side-9 CUDA modes returned finite VJPs and positive corners, then the following separate 257² and 1025² jobs completed on the then-idle AI GPU0. They used three synchronized warm repeats after one warm-up; times exclude construction, input creation, corner audit, images, CNN, I/O, and training optimizer.

| Control grid | Mode | Latent scalars | Median forward | Median VJP | Minimum raw corner q | Nonpositive / nonfinite corners | Peak CUDA allocated / reserved |
|---|---|---:|---:|---:|---:|---:|---:|
| 257² | F1-D | 130,050 | 7.09 ms | 10.72 ms | 7.627e-7 | 0 / 0 | 59.1 / 86.0 MB |
| 257² | F2-D | 520,200 | 9.87 ms | 29.17 ms | 7.627e-7 | 0 / 0 | 93.9 / 121.6 MB |
| 1025² | F1-D | 2,093,058 | 6.71 ms | 28.64 ms | 4.764e-8 | 0 / 0 | 938.4 / 1094.7 MB |
| 1025² | F2-D | 8,372,232 | 13.96 ms | 50.82 ms | 4.763e-8 | 0 / 0 | 1546.8 / 1688.2 MB |

The source cell area is `1/(N-1)^2`, so the raw q minimum shrinks with resolution; these q values are not normalized Jacobians. All boundary errors were exactly zero and every measured gradient tensor was finite with nonzero maximum magnitude. F2 has four independent dense latent fields versus F1's one, so these numbers **do not** show an intrinsic four-to-one cost/accuracy tradeoff or prove training usefulness. The full JSON-derived scalar values are in `results.csv`. The remote clone used milestone `dafc704` plus the benchmark's subsequent device-placement-only fix copied from the D: worktree; that fix must be committed at the next milestone. A system-level benchmark with image encoder, map query, export audit, and matched mapping quality remains open.

### Saved-output check and the serialization issue

The first export attempt used `torch.save` and failed inside the remote PyTorch serializer (`_pickle.Pickler.persistent_id` read-only) under its Python 3.13.0 / PyTorch 2.5.1+cu124 environment; local Python 3.12.4 did not reproduce it. The export path was changed to NumPy `.npz` after a failing format-specific test. The remote job then wrote a **257² float32 F1-D map**, reloaded it, and found all 262,144 corners positive, zero nonfinite corners and exactly fixed boundary. That artifact was copied to [q1_257_f1_float32.npz](q1_257_f1_float32.npz) on D:. A separate local NumPy test recalculates all corners directly from the **saved binary32 coordinates converted to binary64** rather than using the production helper; it passed. This is evidence for this one exported sample, not yet a rigorous floating-point sign certificate for every possible latent or an actual pathology registration output.

An additional independent checker evaluated every one of the saved map's 262,144 corner signs as exact integers representing the saved binary32 coordinates. The filtered validator uses outward-expanded binary64 intervals and exact rational fallback for ambiguous signs; its interval path was cross-checked against rational arithmetic on 900 adversarial/random binary64 triples, including overflows and subnormals. This is a certificate of **this stored sample** under correctly rounded IEEE binary64 operations, not a proof that all network outputs will survive binary32 rounding. A deliberately twice-wrapped annular map showed that four positive corners per cell, even for every cell, do not imply global injectivity if the boundary is not simple. Both the tensor and saved-binary validators now require an explicitly ordered, axis-aligned rectangular reference boundary as well as equality to that boundary; they reject empty batches. The narrow boundary condition matches the current fixed-canvas decoder but does not certify arbitrary curved boundary data.

## E1 extension: mixed Q1 pyramid and pixel-center sampling

`HybridPatchSeedVertexQ1Pyramid` applies F2-D to a coarse seed and F1-D to only the new vertices at each dyadic level. `q1_dyadic_refine` represents the previous Q1 function exactly in ideal arithmetic; the decoder edits one final Q1 vertex table, without interpolating a composition of separately sampled maps. Local tests cover identity, non-affine refinement, strict four-corner positivity for finite example fields, fixed boundary and VJP to both seed and level latents. An image encoder's zero-initialized heads feed the mixed decoder; a loss on a new fine vertex has a nonzero gradient to the corresponding head. A loss placed only on an old even/even vertex gives zero gradient to that level by design, which the test initially exposed.

For real image use, `q1_image_sampling.py` distinguishes the fixed-image pixel centers `((j+1/2)/W,(i+1/2)/H)` from control-grid vertices `((j)/(N-1),(i)/(N-1))`. It evaluates the map by bilinear Q1 interpolation (`align_corners=True` on the vertex table), then evaluates moving-image intensity by bilinear pixel-center sampling (`align_corners=False`). Identity and analytic-cell tests pass. This is a defined image-sampling convention, not a digital-topology certificate for a regridded map. Earlier Phase VII synthetic results used endpoint image samples and `align_corners=True` throughout; their numbers are not silently reinterpreted under the new convention.

## Real-data status

The data scout found no verified local HyReCo, ACROBAT, or official ANHIR package. The local MIIT-v4 archive contains single-image records but no association/landmark/grouping manifest; it cannot be called a registered source–target pair. An author-provided HistoReg CD68→CD4 real pathology pair with 77 matched landmarks was subsequently acquired for **development only**; its patient/block identity is unknown, so it is not a leakage-safe test set. An image-only centroid translation and identity diagnostic have been run, but neither is a strong external registration baseline; separate protocol review is pending. Dataset availability and those provisional numbers are detailed in `EVIDENCE.md`. No learning benefit, competitive real-data accuracy, or SOTA claim is established by this pair alone; G2–G4 remain open.
