# Phase V worklog

UTC time: 2026-09-21T16:10:26Z
Route: setup
Question: Can the 18-hour phase be anchored to an immutable real wall clock and a clean verified baseline?
Precise claim/hypothesis: The checked-in start receipt is created once, elapsed time is computed from the system UTC clock, final review is denied before 17h15m, and completion is denied before 18h.
Smallest decisive test: exercise the guard immediately and at injected timestamps bracketing both thresholds.
What falsifies it: overwriting the receipt, accepting a pre-threshold timestamp, or using handwritten elapsed labels.

Result: Phase V began from commit `748aa84`; clean baseline is `317 passed, 1 skipped, 1 warning in 123.84s`. The optional skip is the absent official surface-data package and the warning is Paramiko's external Blowfish deprecation.
Checker needed: yes, before treating the clock gate as authoritative.
Next: test-drive `tools/check_phase5_elapsed.py`, then audit the existing Tutte implementation and official TutteNet sources.

UTC time: 2026-09-21T16:18:31Z
Route: I — Fast Tutte neural layer
Question: Can one fixed control mesh support a reusable batched Tutte solve and precomputed 256²/512² dense queries without repeating point location or sparse factorization in backward?
Precise claim/hypothesis: Fixed source topology permits one neighbor table and one pixel-to-face/barycentric table; a direct SuperLU backend can reuse a single numeric factor for x/y right-hand sides and transpose adjoints, while matrix-free directed and symmetric backends can expose the same map and VJP contract.
Assumptions: one simple triangulated disk; a positively oriented weakly convex rectangle boundary; strictly positive valid-neighbor weights; all query pixels lie in the source control domain.
Smallest decisive test: on an 11×11 control-vertex grid, compare direct output/VJP against an independently assembled reference and compare precomputed dense warp against analytic identity at 256².
What falsifies it: direct/reference disagreement, adjoint finite-difference failure, per-forward point location, negative control-face orientation, or iterative residual failure.
Prior work: Phase III directed Tutte implicit layer; official TutteNet paper/code audit is running independently.

Result (2026-09-21T16:35:25Z): Independent adversarial review found four concrete failures in the legacy directed layer: boundary-loop order was mistaken for global vertex order on a boundary-only mesh; a weakly convex collinear boundary ear could collapse a face; finite extreme logits could underflow a supported probability to zero; and an unused isolated source vertex was accepted. Regression tests now cover all four. The implementation scatters/gathers the boundary permutation correctly, rejects unused vertices and zero supported probabilities, and accepts a solve only when every original oriented face has strictly positive numerical signed area. The focused directed suite is `9 passed`; this is a finite-precision certificate for the returned P1 map, not yet a proof that every admitted generic source mesh is a simplicial disk.

Dense-query result (2026-09-21T16:35:25Z): `StructuredDenseQueryTable` analytically precomputes source triangle indices and barycentric weights for `structured_rectangle`. At 512 squared queries its affine-map error is at most `6.66e-16`; 104 independently located triangle/corner queries agree to `1.30e-15`; `torch.autograd.gradcheck` passes. The implementation explicitly prepares device/dtype data outside the forward path. An independent review exposed a `cpu:0`/indexless-device cache-key mismatch; a regression test and actual-allocation key normalization fixed it. Final independent verdict: PASS, `5 passed`; prepared interpolation contains gather, multiply, reduction, and reshape operations but no transfer or point location. CUDA key semantics were inspected but not executed locally because no local CUDA device exists.

Checker needed: yes. The generic source-disk hypotheses and direct-solver factor-reuse path remain open and are being checked separately.
Next: close the source topology certificate, implement the one-factor direct oracle, and only then compare matrix-free backends.

UTC time: 2026-09-21T16:38:00Z
Route: shared benchmark, first used by Route I
Question: Can all three routes be evaluated on the same valid deformation and image data without confusing a forward geometric map with the backward sampling grid used by image registration?
Precise claim/hypothesis: A deterministic suite of orientation-preserving analytic maps, supplemented later by positive-Tutte random maps, can be sampled on every structured control mesh and accepted only after an independent P1 injectivity audit. For images, the learned PL homeomorphism is defined as the backward coordinate map from fixed-image coordinates to moving-image coordinates; therefore `grid_sample(moving, F_theta)` directly produces the warped moving image and no differentiable inverse is hidden in the benchmark.
Assumptions: align-corners pixel coordinates on the closed unit square; moderate deformation parameters with explicit positive continuous Jacobian bounds; structured source disks; image intensities normalized to `[0,1]`.
What would falsify it: a named analytic map with nonpositive analytic determinant, a sampled P1 face with nonpositive orientation, nondeterministic data, an inconsistent corner/pixel convention, or an image pair that can be fit without calling the decoder and dense warp.
Smallest decisive test: check every named map on 11-by-11 control vertices, independently audit the returned P1 map, verify identity sampling exactly at pixel centers, and verify each of four image families is finite, nonconstant, deterministic, and normalized.
Prior work: Phase V common benchmark specification; PyTorch `grid_sample` backward-coordinate convention; existing independent `audit_injectivity` routine.

UTC time: 2026-09-21T16:50:00Z
Route: I — rectangle boundary and learnable modulus
Question: Can boundary freedom be exposed to optimization while preserving the target rectangle boundary as an orientation-preserving homeomorphism by construction?
Precise claim/hypothesis: For every side, softmax-normalized positive segment lengths and cumulative sums place all noncorner boundary vertices in strict side order; sharing the four exact corners closes the polygon. A positive height `H = H_min + softplus(m)` supplies a differentiable learnable modulus without allowing a collapsed rectangle.
Assumptions: source is a certified structured rectangle; the source boundary loop follows its oriented four sides; width is fixed to one to remove global scale; logits are finite and do not underflow to zero in the returned dtype.
What would falsify it: a corner mismatch, duplicate/reversed adjacent boundary points, a zero segment after dtype conversion, wrong loop-to-side indexing, nonpositive height, or a failed finite-difference modulus/boundary-logit gradient.
Smallest decisive test: on unequal 4-by-3 and 1-by-3 grids, check identity logits recover uniform side spacing, random logits preserve exact corners and strict order, and autograd/finite differences agree for every side plus modulus.
Prior work: Phase V Route I-B specification and TutteNet's cumulative boundary-angle construction.

Direct-solver result (2026-09-21T16:52:06Z): `DirectTutteLayer` is a CPU correctness/reference backend supporting unbatched, batched, and singleton-broadcast directed row-softmax systems. Each sample calls SuperLU factorization exactly once, solves the two coordinate columns as one multiple-right-hand-side call, retains that factor with the autograd graph, and uses the identical factor's `trans='T'` solve for the two-column adjoint. A batch-two instrumented forward/backward therefore records two factorizations and the solve sequence `N,N,T,T`; an independent variable-degree dense reference agrees within `4.45e-16`. Non-symmetric gradcheck and joint logits/boundary directional finite differences pass. The first independent review found a real float64 alias: the returned tensor shared NumPy storage with the primal saved for backward, so an in-place user update silently doubled selected gradients. The saved primal is now a separate read-only copy; independent recheck found zero in-place/non-in-place gradient discrepancy for unbatched and batch-one/two cases. A returned-float32 collapse that remains positive in the hidden float64 solution is also explicitly rejected. Final focused result: `18 passed`; independent verdict: PASS. Limitations remain explicit: sequential CPU only, first derivatives only, no cross-forward symbolic cache.

Shared-benchmark result (2026-09-21T16:52:06Z): Eight deterministic analytic orientation-preserving maps have stated positive continuum determinant lower bounds and pass independent P1 audits on every required 11/17/25/33/49 control-vertex grid. Four deterministic image families are finite, nonconstant, normalized, and pass 256-squared float32/float64 checks. The convention is fixed-to-moving backward coordinates with `align_corners=True`. Initial review found two silent validity holes: border padding hid out-of-domain maps, and NaN coordinates could return finite border intensities. `make_registration_pair` now rejects a map leaving the sampled unit square unless a future explicit extended-domain/mask protocol is provided, and the warp rejects all nonfinite images/maps. Independent closure recheck: PASS, `6 passed`. At present only local compression and boundary sliding from the analytic suite are admitted as in-domain registration pairs; the remaining maps remain valid geometric expressivity tests.
