# Phase V final independent review

Review date: 2026-09-22, Asia/Shanghai. Checker context:
`/root/phase5_final_reviewer`, fresh and independent of the route builders and
the three route-closure authors. Reviewed baseline:
`5193b2949ca829c465ef60493ac145e7540d66b3`; the working tree was clean when this
review began. This checker changes only this review document.

## Decision

**QUALIFIED PASS: Phase V is ready for the T+18h bounded research synthesis.**
The three routes contain executable implementations, independent derivative
and geometry checks, realistic synthetic instances, preserved negative
results, and enough evidence to make a technical decision. There is no newly
identified correctness blocker to that bounded synthesis.

**FAIL for the stronger claim that the overall fast, accurate, memory-efficient,
generally trainable production neural-layer objective has been achieved.**
This is an actual unachieved objective, not a qualification that disappears
after the clock expires. Neither an unbiased optimizer ranking nor a universal
finite-precision topology/solver/gradient guarantee follows from the evidence.
The eventual decision must retain the failed and untested items below.

The start receipt contains `2026-09-21T16:10:26Z`, epoch `1790007026`.
The first system-clock observation in this checker context was
`2026-09-22T17:48:45+08:00`, after the review boundary
`2026-09-22T17:25:26+08:00`. A fresh invocation of
`python tools/check_phase5_elapsed.py --gate review` returned epoch
`1790070588`, elapsed `63562` seconds, `FINAL_REVIEW_ALLOWED`, and
`completion_allowed=false`. The independent decision boundary remains
**2026-09-22T18:10:26+08:00**. This review neither writes
`15_final_decision.md` nor marks the phase complete. Readiness to synthesize
does not authorize early finalization.

## 1. Evidence and fresh verification

Reviewed the authoritative Phase V plan, particularly sections 45--58 and the
route closure requirements; START_TIME; WORKLOG research cards, corrections
and route transitions; chapters 01--12; the common CSV and its extractor/tests;
the directed/direct/symmetric solvers, MVC encoder/retraction, positive-Hodge
student, Whitney reference and relevant benchmark paths. Older builder headers
in chapters 02/06/07/09/10 still say that a checker or application is pending.
For closure chronology those are superseded by the dated 04/08/12 verdicts,
not evidence that an old failure was silently erased.

The official temporary source clones were independently checked at the stated
revisions: TutteNet `cb9f91969ba012f5670c6066298760314d8d31a5` and
torch_sparse_solve `89c227c0e7771b5c86dca0f25b78c14b4edc85b1`. Direct inspection
of the latter's Python/C++ confirms a sequential sample loop, shared RHS factor,
fresh transposed factorization in backward, and freed forward KLU factors.
The local improved direct solver instead retains one SuperLU factor per sample
and calls its transposed solve. This verifies mechanisms, not official training
or historical benchmark reproduction.

Fresh Windows CPU verification used `PYTHONPATH=src`,
`MKL_THREADING_LAYER=SEQUENTIAL`, `OMP_NUM_THREADS=1`, and a NumPy LAPACK
initialization before importing pytest/Torch. No duplicate-OpenMP safety bypass
was used. The first independent selection completed with **219 passed,
7 skipped in 32.05 seconds, exit 0**. The skips explicitly require unavailable
CUDA. The selected files were:

```text
tests/test_phase5_elapsed.py
tests/test_phase5_tutte_direct.py
tests/test_phase5_tutte_iterative.py
tests/test_phase5_tutte_symmetric.py
tests/test_phase5_tutte_boundary.py
tests/test_phase5_dense_warp.py
tests/test_phase5_tutte_composition.py
tests/test_phase5_mvc.py
tests/test_mvc_retraction.py
tests/test_phase5_mvc_lift_comparison.py
tests/test_phase5_mvc_incremental.py
tests/test_phase5_whitney_hodge.py
tests/test_phase5_positive_hodge.py
tests/test_phase5_cross_route_results.py
tests/test_phase5_route3_pref_common_benchmark.py
```

These tests include independently assembled nonsymmetric dense systems,
finite differences, factor-call instrumentation, boundary/batch derivatives,
MVC local and global identities, and Whitney energy/orientation checks. Fresh
CPU tests do not replace the recorded remote CUDA runs.

A second, disjoint application/harness selection completed with **122 passed,
2 CUDA-unavailable skips in 59.60 seconds, exit 0**. It covered
`test_phase5_tutte_reference`, `test_phase5_tutte_decoder`,
`test_phase5_tutte_instance`, `test_phase5_mvc_instance_optimization`,
`test_phase5_mvc_o4_trust`, `test_phase5_shared_benchmark`,
`test_phase5_metrics`, `test_phase5_route1_validity_suite`,
`test_phase5_route1_engineering_benchmark`,
`test_phase5_route2_mvc_instance_benchmark`,
`test_phase5_route2_mvc_layer_benchmark`, `test_phase5_route2_mvc_roundtrip`,
and `test_phase5_route2_incremental_benchmark` (all `.py` files under `tests/`).
Thus this final reviewer freshly ran **341 passing tests and 9 explicitly
unavailable-CUDA skips**, without claiming a full-repository regression or a
new GPU performance run. At `17:59:01+08:00`, a second live review-gate check
reported elapsed `64115` seconds and completion still disallowed.

A separate checker calculation, outside the production metric and solve
helpers, gave the following results. The CSV was regenerated in memory and
compared directly with the committed bytes; no receipt/table was rewritten.

| Independent check | Fresh result |
|---|---|
| Common CSV regeneration | 10 rows, 57 columns, byte-identical |
| P1 map final state: independently build sorted mesh edges, assemble scalar Laplacian from saved conductances and solve Dirichlet system | Maximum coordinate difference `5.085754e-11`; minimum Jacobian determinant `0.46498088` |
| P1 image final state, same independent construction | Maximum coordinate difference `4.449585e-11`; minimum determinant `0.50599549` |
| Independent edge-Jacobian map/Beltrami metrics for those states | Map-task RMSE `0.0029898871`, mu RMSE `0.0760834653`; image-task RMSE `0.0150900167`, mu RMSE `0.1566943050`; agree with saved final metrics |
| P1 complete trace and timing reconstruction | Each task has 82 observations; observation `k` has `2k+1` completed solves; exact83 timing sums agree independently |
| Formal target regeneration | N25/R256, seed `20260922`, strength `.25`, height `.9`/`1`; control/dense summaries reproduce Route-II authority; thresholds match `1e-4`/`1e-3` |
| New prescribed nodal target, independent of a harmonic target generator | On a 7-by-5 control grid, affine-plus-sinusoidal displacement has minimum determinant `0.83656417`; Whitney recovery error `2.220446e-16` |
| Same manufactured target, joint tensor and boundary directional derivative | Central difference at `h=1e-6` versus implicit VJP absolute discrepancy `1.950204e-10` |
| Git provenance | `74b452c`, `35a1878`, `0da8ffa`, `f0fa66d` are ancestors of reviewed HEAD |

For the manufactured recovery check, face tensors were constructed directly as
`det(F) inv(F) inv(F).T` from independent source/target edge matrices. The target
was prescribed nodally before any PDE solve. This avoids validating a solver
only against a target manufactured by the same solver.

The two P1 final-state checks above concern the final 163-solve state, **not**
the intermediate exact83 state. Intermediate coordinates were not serialized;
the latter's scalar metrics were checked against trace/slice consistency, not
independently reconstructed geometry. The two kinds of evidence must remain
separate.

## 2. The twelve section-55 questions

| # | Requirement | Finding and verdict |
|---|---|---|
| 1 | Which route has a hard topology theorem? | **PASS, conditional theorem.** Directed Tutte, MVC when its final map is a Tutte decode, and positive symmetric P1 have the positive barycentric embedding mechanism. It requires an oriented embedded triangulated disk, positive supported rows, cyclic orientation-preserving convex boundary homeomorphism, and the weak-convex dividing-edge condition. The fixed structured rectangle construction supplies the intended boundary incidence. Exact mathematics and floating output screens are distinct. |
| 2 | Which has only observed no-fold behavior? | **PASS as classification.** Full Whitney/P-ref has no general embedding theorem. Its particular recovered target has positive faces and the applicable a posteriori boundary/degree justification, but arbitrary full-Hodge inputs do not inherit that result. All finite-precision success populations are observations under numerical screens, not universal exact-predicate guarantees. |
| 3 | Which solver is actually GPU/batched? | **PASS, bounded engineering scope.** Directed BiCGStab and symmetric CG use device tensors and vectorize distinct batch/RHS recurrences. The positive student inherits symmetric CG. M1/M2 use these available backends. CPU geometry copies and scalar synchronizations remain; this is not a fused, zero-transfer or CUDA-graph-ready kernel. Fixed-topology batching is not batching arbitrary different meshes. |
| 4 | Which is CPU reference only? | **PASS.** Official inspected KLU, legacy local reference, improved SuperLU direct, and Whitney/P-ref are CPU paths. Multi-RHS reuse is not GPU batching. Whitney has element matrix-free action, but its actual Dirichlet solver is assembled sparse factorization. |
| 5 | Is Tutte/MVC prior art overstated? | **PASS for the bounded attribution.** Classical MVC and positive barycentric maps, differentiable cage coordinates, TutteNet and Generative Escher Meshes are acknowledged. Neither MVC differentiability nor a positive-Laplacian neural layer is new by itself. An explicit canonical section and its optimization use are candidate contributions, not established priority. Official training/timing replication was not performed. |
| 6 | Is conservation confused with bijection? | **PASS.** `B1 B0=0` is incidence exactness; `(B0.T q)_I=0` is equilibrium balance. The disk stream is face-valued and concerns a variational cochain, not RT0 physical normal conformity or a nodal conjugate. Neither identity proves spatial injectivity. Whitney is exactly ordinary P1 on exact gradients and does not improve that nodal conjugacy error. |
| 7 | Is the benchmark common-input? | **QUALIFIED PASS for T1/T2/P1 evidence; FAIL for an all-four identical-protocol ranking.** The same builder, mesh, deterministic target, medical phantom, warp convention and thresholds are used. Optimized rows use AI A6000 float64. P-ref is target-informed CPU one-shot inference and explicitly `comparable=False`. Equal hardware model is not proof of identical load. P1 and T1/T2 have different initialization geometry, solver tolerances and tuning opportunity. |
| 8 | Does timing include dense warp and backward? | **PASS with explicit scopes.** Optimized wall time encloses repeated solves, dense losses, reverse passes, updates and audits. Stage sums are cumulative through the selected observation. P1 row 41's later backward is correctly excluded from exact83. Projector training is outside method-only time but charged separately; wrapper and setup-inclusive threshold times must accompany cold-start claims. P-ref time is a different one-forward/one-adjoint workload. |
| 9 | Is memory measured? | **QUALIFIED PASS.** Process HWM and CUDA allocated/reserved peaks are measured. They are different scopes; CPU HWM includes runtime/audits and CUDA peaks include the declared setup/workload. Explicit array bytes are not peak memory. No measured network-training peak, universal memory ranking or million-control-vertex inference follows. |
| 10 | Do formulas and code agree? | **PASS for first-order fixed-geometry claims.** Directed transpose gather/scatter, positive softmax VJP sign, symmetric negative edge-difference product, boundary terms, MVC covariance lift, Whitney symmetrized tensor derivative and factor reuse agree with derivations and fresh tests. Custom solve backwards are `once_differentiable`: higher-order derivatives are unsupported. |
| 11 | Is expressivity confused with optimization? | **PASS after retaining negative results.** `D(E(Y))=Y` is a representation result, not fast convergence. O4's image failure and a bounded-budget threshold miss do not prove an unattainable map. The finite-direction theorem concerns local tensor cones; it does not prove an arbitrary global positive graph cannot realize a particular map. Small image error does not prove small mu error. |
| 12 | Are directed and symmetric weights distinguished? | **PASS.** Directed rows need a nonsymmetric transpose adjoint. Symmetric conductances give an SPD reduced Laplacian and detailed balance after row normalization; normalized rows themselves need not be symmetric. The operator-family restriction alone is not a proof of strict separation of attainable coordinate maps. Official midpoint-tied learning and independently directed fitting are distinguished. |

The source/topology distinction is substantive: target area and boundary
predicates are floating-point screens, while source validation contains exact
represented-coordinate predicates. A positive face screen with an arbitrary
unvalidated boundary is not the theorem. Likewise, an exact composition is PL
on a refined partition; resampling it on the original mesh does not preserve
the original-triangle P1 guarantee automatically.

Targeted fresh primary-source checks confirm the attribution boundaries:
[TutteNet](https://arxiv.org/abs/2406.12121) already composes differentiable
Tutte layers; [Lipman's global inversion result](https://arxiv.org/abs/1310.0955)
requires the boundary condition in addition to orientation;
[Generative Escher Meshes](https://arxiv.org/abs/2309.14564) is relevant prior
art; and [FEEC](https://arxiv.org/abs/0906.4325) concerns compatible numerical
structure rather than a general spatial embedding theorem. This targeted
review does not establish novelty by absence of search results.

## 3. Explicit success criteria and route closure audit

`PASS` below means the Phase V research deliverable exists and has appropriately
scoped evidence. It does not convert a negative experimental answer into an
algorithmic success.

| PLAN section 57 item | Verdict | Evidence and limit |
|---|---|---|
| Tutte: official mechanism reproduction | **QUALIFIED** | Source audit, official mesh execution/counts and local corresponding mechanisms are present. Full official extension installation, trained checkpoint, published timing or training reproduction is absent. Sections 7/14 explicitly require the audit, which is satisfied. |
| Tutte: fast backend | **QUALIFIED PASS** | One-factor CPU direct, GPU matrix-free directed/symmetric backends and fixed-query interpolation exist. CPU factor reuse has a bounded measured benefit. A universally faster GPU backend is not demonstrated. |
| Tutte: correct implicit gradient | **PASS, first order** | Nonsymmetric dense oracle, gradcheck/FD, boundary/modulus/batch and composition checks; finite-precision gradient accuracy still depends on conditioning/tolerance. |
| Tutte: 256-square map optimization | **PASS** | Actual latent optimization, N17/N25, Adam/LBFGS, dense decoder calls and threshold records. Selected 512-square cases also exist. |
| Tutte: 256-square image optimization | **PASS** | Same required resolutions/control sizes and real warp/backward calls; successes and misses retained. Synthetic instances do not establish clinical/real-data performance. |
| Tutte: hard topology | **PASS, conditional** | Explicit exact theorem and numerical accepted-map checks; no post-hoc fold repair. No universal return-for-every-finite-latent guarantee. |
| Tutte: solver/memory benchmark | **PASS** | Full control `11/17/25/33/49`, batch `1/4/8`, layers `1/2/4`, queries `256/512` matrix; stress and failure records. |
| MVC: canonical encode/decode | **PASS** | MVC affine precision plus solve uniqueness proves `D(E(Y))=Y`; 54 bounded realistic-control round trips and fiber/nullspace tests. General `E(D(p))=p` is rejected. |
| MVC: differentiable canonicalization | **PASS, first order** | M1 implementation and CPU/GPU runs; float64 correctness evidence. Large tested float32 derivative errors remain a negative result. |
| MVC: differentiable covariance update | **PASS as prototype** | M2 chain includes encoder, lift and final decode, with derivative tests. Weighted-minimum-norm lift is not MVC's derivative or universal Euclidean pseudoinverse. |
| MVC: 256-square instance optimization | **PASS as investigation; O4 image success FAIL** | O1/O2/O3 map/image and O4 map run; O4 image fails at attempted step 18, before exact83. Separate trust variant also fails and cannot replace that row. |
| MVC: raw versus canonical optimization | **QUALIFIED PASS** | Same-target O1--O4 robustness/solve-budget comparison exists. Equal tuning effort and a parameterization-fair winner are not established. |
| MVC: incremental solver study | **PASS** | Cold/warm/exact-correction comparisons and bounded local Woodbury cases on two hosts. Global updates are not called a Woodbury speedup. |
| MVC: optional tiny network | **NOT DONE, OPTIONAL** | No CNN integration. This is explicitly not a route-closure requirement. |
| Primal-dual: compatible reference | **PASS** | Explicit full Whitney Hodge, independent local energy and P1 operator equality, conservation/dual diagnostic. RT0 physical conformity is absent. |
| Primal-dual: positive hybrid or obstruction | **PASS** | Positive frozen-projector/Tutte student plus strict local finite-direction obstruction and three planar coverage families. Canonical NNLS near-transition coefficient limitation is disclosed. |
| Primal-dual: autograd | **PASS, first order** | Tensor/boundary VJP, symmetric edge VJP and latent-to-map path; fresh manufactured joint derivative check. |
| Primal-dual: 256-square experiment | **PASS** | Two clean N25/R256 formal positive-student tasks, common-target P-ref, retained target-informed supplements excluded from ranking. |
| Primal-dual: what it contributes | **PASS, bounded answer** | Differentiable full-tensor teacher and hard-valid approximate QC-informed student; not improved same-mesh nodal conjugacy or exact arbitrary prescribed mu. Outcomes A/B are supported; C is not claimed. |

Sections 14/31/44/49 independent closures are documented by `e4085a1`,
`4a01311` and `5193b29`. WORKLOG places the Route I verdict at
`2026-09-21T22:04:59Z`, the Route II verdict at `2026-09-22T03:06:41Z`,
and Route III start at `03:11:21Z`. The Route III minimum therefore ends at
`08:56:21Z`; its closure occurred afterwards. The current review starts after
both that closure and the independent-review clock gate. Cross-route table
construction uses already closed routes as references and does not constitute
premature reopening/exploration of a future route.

The section-5 metric wish list is not fully populated in the compact CSV:
inverse consistency is not measured, some P1 intermediate extrema are absent,
and several setup/iteration/gradient/latent-size diagnostics remain in route
receipts rather than common table cells. **FAIL for a claim that every named
metric was measured in every common row.** Missing values must remain missing.
These omissions limit comparisons but do not invalidate the delivered
section-57 investigations or require inventing substitute metrics.

## 4. Fairness and decision-critical negatives

The common rows are useful evidence, not a controlled causal ablation of
parameterization. T1/O1 and T2/O3 share a numerical rate `.001` selected for
O1--O4 robustness; P1 uses `.03` selected on separate seed `20260921`. This
avoids formal-target tuning leakage, but does not equalize tuning opportunity.
P1's uniform tensor latent also decodes a different initial map from uniform
directed rows. Its solver tolerance differs. Exact target boundary values are
provided for all cross-route tasks, so they are boundary-informed synthetic
registration, not arbitrary image-only boundary discovery.

The final synthesis must not drop O2 merely because the compact cross-route
table selected O1 as T1. From the clean O2 receipts, at exact83 its map/image
objectives are `4.9662126e-5` / `1.2159964e-3`, versus O1's
`7.0055138e-5` / `1.8133391e-3` and O3's `6.7622546e-5` /
`1.7500772e-3`. O2 first reaches map/image thresholds at 23/101 global solves
and 8.207/38.406 seconds. O4 reaches map threshold at 23 solves and 10.877
seconds and attains `4.8073904e-7` at exact83, but its image run has no such
observation. These support different bounded recommendations for map fitting
and robust image fitting, not a universal O4 or MVC winner.

P1 has lower exact83 objectives and recorded times in this cohort, but its image
mu RMSE `.158852` is worse than O1's `.151093`. It is therefore not a blanket
accuracy winner. Its method-only threshold times are 1.373/4.283 seconds;
including projector setup gives about 5.383/8.470 seconds. The offline projector
is a trained small MLP, so “no network was trained” would be false; what is
absent is end-to-end image-to-latent CNN training through the deformation layer.

The following are concrete unclosed claims and the exact evidence needed to
make a stronger claim. They are not hidden blockers to an honest negative or
qualified synthesis.

| Unclosed claim | Status and required remediation |
|---|---|
| Parameterization-fair overall winner | **NOT ESTABLISHED.** Give each method the same separate-seed tuning budget, match objective/initial-map protocol where feasible, retain failures, and compare multiple held-out cases/time-to-threshold. Until then report cohort observations only. |
| Uniformly reliable/accurate float32 GPU layer | **FAIL on tested cases.** Directed N49 normal failures and strong-latent failures remain; MVC N49 float32 FD discrepancy is about `.1999`. Improve numerical accuracy/availability and demonstrate both forward and VJP accuracy at declared tolerances, not just finite gradients or low residual. |
| Universal fast GPU winner | **NOT ESTABLISHED.** Current CUDA kernels include host checks/synchronizations; GPU is slower on several matched bounded profiles. An isolated matched workload, kernel/profile accounting and repeat measurements are required for that ranking. |
| P1 at 512-square optimization | **NOT DONE.** P-ref 512 tests validate the reference/query path only. Run the actual positive-student optimization at 512 with target, topology, gradients, timing and memory before extending that claim. Route I's selected 512 cases do not fill this gap. |
| End-to-end CNN training | **NOT DONE, OPTIONAL FOR THIS PHASE.** Test a learned image encoder through the actual decoder, retain out-of-distribution/failure behavior, and measure full activations and optimization. |
| Higher-order differentiation | **UNSUPPORTED.** Custom backward is explicitly once-differentiable. A differentiable adjoint implementation and independent second-order checks are needed before claiming Hessian/meta-learning support. |
| Exact arbitrary prescribed Beltrami recovery | **NOT ESTABLISHED.** Teacher recovery is for realizable target-derived P1 data with exact boundary. The finite local cone obstruction rules out generic exact local tensor matching; global map expressivity needs a separate analysis. |
| Compatible flux implies better nodal conjugacy or bijection | **FAIL as a general inference.** The implemented Whitney primal operator is identical to P1; RT0 and a nodal-conjugate embedding theorem were not delivered. A different specified discretization and the corresponding theorem/measure are required. |
| Formal exact finite-precision homeomorphism for every accepted input | **NOT ESTABLISHED.** Existing target predicates are floating screens. Exact/adaptive target predicates or proved numerical error bounds would be needed for the stronger certification claim; universal solver availability is a separate issue. |
| Million-control-vertex, arbitrary-mesh neural scalability | **NOT DONE.** Dense-query resolution is not control DOF. Run actual large control/batch/optimizer workloads and include all memory scopes; do not extrapolate from N25/R256 or P-ref N129. |

The canonical NNLS correction is accepted in its disclosed scope. At a
`1e-8` support transition, permutation invariance does not imply
machine-precision secondary coefficients. This is an offline coverage-reference
limitation, and `git diff` confirms it does not change the formal frozen-MLP
instance runner, inherited symmetric solver or Whitney implementation from
their authority commits. Requiring a learned-P1 rerun solely because the
offline canonical fit changed would not address a real dependency.

## 5. Constraint on the T+18h synthesis

The final decision can now answer the five section-56 questions with bounded
recommendations. It must preserve the following conclusions:

1. The topology mechanism is the positive fixed-planar barycentric/Tutte
   decoder with its boundary hypotheses. CPU factor reuse and symmetric
   matrix-free GPU execution are demonstrated engineering options; a single
   universally fastest deployment choice is not proved.
2. MVC supplies a valid canonical section and local update geometry. O4 is
   promising on the map instance and fails on the image instance. O2 remains
   a relevant successful robustness baseline. A fair-tuned optimizer winner
   remains unresolved.
3. Full Whitney is a teacher/reference; positive-Hodge is an approximate
   hard-valid decoder candidate. Neither is an exact arbitrary-mu solution.
4. Conditioning, small-kernel/synchronization cost, additional decoder work
   and anisotropic approximation are evidenced bottlenecks. Network prediction
   is unmeasured. Dense warp is not automatically the dominant observed cost.
5. Choose only three next-round tasks, as required by section 56, focused on
   the unresolved evidence rather than claiming that passing this review
   solves it.

**No implementation, receipt or common-table correction is required by this
fresh review before a truthful bounded synthesis.** A statement that all
success ambitions, fair ranking or production readiness are achieved would
be rejected. The coordinator must separately pass the live completion gate
at or after 18:10:26 local time before publishing the final phase decision.
