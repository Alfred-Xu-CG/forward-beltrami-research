# Native multi-chart high-genus registration: independent audit

## Outcome

- Primary matrix: 9 trials; complete=False.
- Reload-and-recompute audits passed: 9/9.
- Numerical homeomorphism certificates: 9/9.
- Held-out dense registration improvements: 5/9.

## Per-case, per-objective aggregate

| case | mode | certified | dense successes | mean initial | mean final | mean improvement | landmark improvement | curvature improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| official_pretzel_genus3 | combined | 3/3 | 2/3 | 0.2178 | 0.2185 | -0.5% | +70.8% | +2.1% |
| official_pretzel_genus3 | curvature | 3/3 | 0/3 | 0.2178 | 0.2524 | -16.5% | -22.7% | +3.5% |
| official_pretzel_genus3 | landmark | 3/3 | 3/3 | 0.2178 | 0.1992 | +8.6% | +94.4% | -0.1% |

## What is and is not established

The state is a global target-surface point `(face id, barycentric coordinates)`; face ids change only through deterministic edge walking. The optimized generators are edge-compatible quadratic fields in the native PL atlas, so there is no fixed source-chart to target-chart assignment and no cut seam.

The numerical certificate combines a topology-valid common-refinement base map, projection checks, a shared Lipschitz-CFL policy, target-chart orientation probes, and forward/inverse replay. It is floating-point evidence, not an exact-predicate proof for arbitrary smooth maps.

Official mesh landmarks are synthetic samples from the published common map; they are not manual semantic annotations. The published map is held out for error measurement and initialization perturbation. Consequently these experiments validate robust refinement from a known topology class, not automatic initialization from unrelated raw meshes.

Curvature-only optimization is accepted only when its own residual decreases, but correspondence success is reported separately. On geometrically dissimilar official pairs, curvature residual reduction can move away from the held-out map; this negative result is retained.

The official archive contains genus-3 and genus-5 cases, not genus-2. Exact genus-2 coverage is supplied by the generated smoothed double torus and is not presented as published real data.
