# Genus-2 native multi-chart registration implementation plan

Date: 2026-08-28

## Acceptance gate

The work is complete only when unit/integration tests pass from a clean command,
the exact genus-2 topology path passes, all selected real-data runs emit
machine-readable metrics and visualizations, and an independent audit verifies
the reported topology and map-quality claims.

## Task 1: Data provenance and topology inventory

- Ignore downloaded data in Git.
- Implement an OBJ loader that accepts `v`, polygonal `f`, and optional texture
  indices without confusing geometry and texture connectivity.
- Test Euler characteristic, boundary count, manifoldness, connectedness,
  orientation, and genus on known meshes.
- Inventory every official A/B and common-refinement pair, retaining the archive
  URL, byte size, SHA-256, vertex/face counts, and measured genus.

## Task 2: Native PL atlas

- Write failing tests for barycentric round trips, across-edge transport,
  multi-edge walking, vertex tie cases, and closed-surface rejection rules.
- Implement `SurfacePoint(face, barycentric)` and oriented face adjacency.
- Implement local tangent frames, barycentric derivatives, edge unfolding, and
  deterministic face walking.
- Add transition consistency and length-preservation tests.

## Task 3: Geometry descriptors and smooth tangent fields

- Write failing tests for constant curvature signals on planar patches and
  stable finite values on closed meshes.
- Implement vertex areas, normals, Gaussian curvature, cotangent mean curvature,
  robust normalization, and scalar smoothing.
- Implement screened graph-Laplacian vector-field smoothing with cached sparse
  factorization and tangent projection.
- Test force scattering/interpolation and rigid-scale invariance of normalized
  descriptors.

## Task 4: Reversible surface flow

- Write failing identity, constant-field, face-crossing, and forward/inverse
  round-trip tests.
- Integrate a global tangent field using adaptive substeps and chart transitions.
- Record every accepted incremental field so the inverse applies negated fields
  in reverse order.
- Add crossing-budget and trust-radius diagnostics.

## Task 5: Landmark and curvature registration

- Write a failing controlled genus-2 recovery test before the optimizer.
- Assemble landmark and curvature forces at mapped source samples.
- Add backtracking, objective accounting, fixed compute budgets, and convergence
  reasons.
- Implement landmark-only, curvature-only, and combined modes.

## Task 6: Numerical homeomorphism audit

- Write failing tests containing deliberate collapse, orientation reversal, and
  bad inverse history.
- Implement local orientation margin, collapse count, coverage/degree proxy,
  forward-inverse round trip, and trajectory health checks.
- Refuse to label a map certified unless every required check passes.

## Task 7: Genus-2 benchmark

- Generate a triangulated double torus, verify `chi=-2` and `genus=2`, and make
  deformed targets with identical connectivity.
- Select deterministic curvature-aware/farthest-point landmarks.
- Run multiple perturbation strengths and seeds for all objective ablations.
- Save numerical metrics and visualizations.

## Task 8: Official real-data benchmark

- Validate common-refinement connectivity for genus-3, genus-5, and pretzel
  pairs.
- Treat the published correspondence as held-out ground truth, perturb only the
  initialization, and prevent ground-truth coordinates from entering forces.
- Run at least three deterministic seeds per pair under equal budgets.
- Save per-case and aggregate results; retain all failures.

## Task 9: Reproducibility and independent audit

- Add one-command runners with explicit interpreter/environment assumptions.
- Record configuration, source revision, dataset hashes, timings, and seeds.
- Generate comparison tables and map/trajectory/distortion figures.
- Independently reload outputs and recompute topology, errors, and certificate
  predicates rather than trusting in-memory summaries.
- Run the full existing and new test suite, inspect Git diff, and publish a
  limitations report before making any completion claim.
