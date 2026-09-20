# Native multi-chart high-genus registration: independent audit

## Outcome

- Primary matrix: 9 trials; complete=False.
- Reload-and-recompute audits passed: 9/9.
- Numerical homeomorphism certificates: 9/9.
- Held-out dense registration improvements: 9/9.

## Per-case, per-objective aggregate

| case | mode | certified | dense successes | mean initial | mean final | mean improvement | landmark improvement | curvature improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_genus2 | combined | 3/3 | 3/3 | 0.2708 | 0.2355 | +13.1% | +27.6% | +11.8% |
| synthetic_genus2 | curvature | 3/3 | 3/3 | 0.2708 | 0.2484 | +8.3% | +5.2% | +11.7% |
| synthetic_genus2 | landmark | 3/3 | 3/3 | 0.2708 | 0.2255 | +16.7% | +91.9% | +2.3% |

## What is and is not established

The state is a global target-surface point `(face id, barycentric coordinates)`; face ids change only through deterministic edge walking. The optimized generators are edge-compatible quadratic fields in the native PL atlas, so there is no fixed source-chart to target-chart assignment and no cut seam.

The numerical certificate combines a topology-valid common-refinement base map, projection checks, a shared Lipschitz-CFL policy, target-chart orientation probes, and forward/inverse replay. It is floating-point evidence, not an exact-predicate proof for arbitrary smooth maps.

Official mesh landmarks are synthetic samples from the published common map; they are not manual semantic annotations. The published map is held out for error measurement and initialization perturbation. Consequently these experiments validate robust refinement from a known topology class, not automatic initialization from unrelated raw meshes.

Curvature-only optimization is accepted only when its own residual decreases, but correspondence success is reported separately. On geometrically dissimilar official pairs, curvature residual reduction can move away from the held-out map; this negative result is retained.

The official archive contains genus-3 and genus-5 cases, not genus-2. Exact genus-2 coverage is supplied by the generated smoothed double torus and is not presented as published real data.
