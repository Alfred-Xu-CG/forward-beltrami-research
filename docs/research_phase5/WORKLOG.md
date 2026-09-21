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
