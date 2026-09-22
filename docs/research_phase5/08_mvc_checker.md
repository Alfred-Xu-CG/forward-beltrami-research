# Route II independent checker: MVC canonical coordinates and retractions

Date: 2026-09-22 (Asia/Shanghai).  
Reviewed frozen HEAD: `16465701fb2c04ed119f26cfab1311a8d4247935`.  
Checker: independent of the Route II theorem, implementation, experiments, and
authorship of chapters 05--07; no subordinate reviewers were used.

## 1. Verdict and time boundary

**PASS for bounded Route II research closure under plan sections 31 and 49.**
This authorizes proceeding to Route III, not declaring Phase V or the final
fast, accurate, memory-efficient neural-layer objective complete.

The Route I closure time was `2026-09-21T22:04:59Z`; the five-hour Route II
minimum therefore ended at `2026-09-22T03:04:59Z`. The final independent clock
check returned `2026-09-22 03:06:41 UTC` (11:06:41 Beijing), after the fresh
regression below. HEAD and the clean working tree were independently checked
before these two checker-file edits. No verdict was issued before the time
boundary. The final author correction from `d681d27` to `1646570` changed only
chapters 05--07, not executable code or receipts.

This PASS means that the required questions have bounded, reproducible answers,
including negative answers. In particular, the formal O4 image failure is
**adjudicated as a retained negative result**, not repaired, deleted, or counted
as optimization success. The mandatory instance comparison requires evidence,
not an invented all-method winner. CNN training was optional and was not done.

The exact topology statement remains conditional: an oriented triangulated
disk, positive supported convex-combination rows, an orientation-preserving
convex boundary homeomorphism, and no dividing edge mapped into that boundary
give the applicable exact P1 homeomorphism theorem. Returned floating-point
maps additionally undergo solver/face/boundary/global screens. These screens
are not exact-arithmetic predicates, a universal numerical-availability
theorem, or proof of arbitrary prescribed-Beltrami reproduction.

## 2. Claim-by-claim review

| Requirement | Independent finding | Scope |
|---|---|---|
| Canonical subset and section | The encoder retains the boundary and selects the explicit MVC row and supported arithmetic-zero-mean logit gauge. Uniqueness of the linear system and MVC affine precision give `D(E(Y))=Y`; `E(D(p))=p` is false for general raw rows. | Exact theorem on the stated valid-map space; probability rows alone do not encode boundary scale/translation. |
| MVC formula and local stars | The cyclic predecessor/current half-angle numerator divided by edge length is correct. Link order is obtained from oriented mesh incidence, not floating-point angle sorting. A valid embedded star supplies positive angles below pi, winding one, and a center in the strict kernel; the neighbor polygon need not be convex. | Sufficient local hypotheses, not a claimed necessary characterization of every positive MVC configuration. |
| Global topology hypotheses | Linear uniqueness needs boundary reachability, not convexity. Global injectivity uses the separate Tutte--Floater theorem. Weakly convex subdivided rectangles require the dividing-edge condition and preserved side/corner incidence. | Improved direct/directed iterative paths are accepted for the prescribed structured rectangle; no unimplemented generic-boundary preflight is assumed. |
| Fibers and Jacobian kernel | For independent directed rows with affine rank three, the probability fiber has dimension `d_i-3`; raw logits add a row-shift direction, giving `d_i-2`. Fixed-boundary decoder-null directions satisfy the differentiated barycentric constraints. | Local dimension statements; do not transfer to tied/symmetric parameter families. |
| Covariance lift | With `q_i=sum_j p_ij Y_j`, `s_ij=Y_j-q_i`, and `C_i=sum_j p_ij s_ij s_ij^T`, the centered lift solves the row constraint and uniquely minimizes `sum_j p_ij z_ij^2` in the native weighted gauge. Rank two is required. | A decoder differential right inverse only at an equilibrium `Y=D(p,b)`; an off-equilibrium local algebra identity is insufficient. |
| MVC derivative versus lift | Both realize the same admissible vertex tangent at equilibrium, but their latent vectors generally differ by a decoder-kernel vector. Native weighted-zero-mean and arithmetic-zero-mean gauges must not be conflated. | Exact differential identity, independently checked numerically. |
| Moore--Penrose comparison | The tiny explicit Jacobian uses supported slots and a declared orthonormal arithmetic-gauge basis. Its Euclidean pseudoinverse differs from both MVC and covariance lifts. Weighted minimum norm is separately checked in the native gauge. | Diagnostic on a fixed-boundary variable-degree disk, not a large dense-Jacobian algorithm. |
| Implicit VJP and boundary | `A^T Lambda=g_I`, the positive logit-gradient sign, softmax centering, and `g_B+P_IB^T Lambda` are correct. The ordinary row sum of the logit gradient vanishes; its probability-weighted sum need not. | First derivatives only; legal moving-boundary parameters receive their own chain rule. |
| M1/M2 backward | M1 differentiates encoder after decoder. M2 differentiates final decode, latent addition, covariance lift, encoder, and initial decode, including direction/step inputs. Fixed-boundary directions are explicitly zero on the boundary. | No detached lift surrogate or Euler predictor is substituted for the returned map. |
| Retraction and round trip | The exact zero-step and first-variation identities follow on the admissible tangent space. Every accepted finite update is a fresh decoder output. | Local retraction property is not uniform finite-step numerical stability or objective descent. |
| Incremental correction and Woodbury | Signs in `(B'-B)-(A'-A)Y` and the row-local Woodbury system agree with direct algebra. Warm/correction solves use the same absolute residual contract as cold solves. | Nine bounded local Woodbury cases per host; large/global updates are explicitly not run, not called accelerated. |
| Prior art | Classical MVC, positive barycentric parameterization, injective tiling morphs, differentiable cage coordinates, and learnable positive Laplacians are acknowledged. | Targeted primary-source checking does not establish publication priority for the present combination. |
| O1--O4 fairness | Shared target, initialization, boundary, loss, decoder and public tolerances are enforced. Exact-83 completed solves and common-40 updates are separate slices. O3 retains Adam moments through projection; O4 uses vertex-coordinate Adam. | A shared numerical rate is a robustness comparison, not equal parameterization scale or equal tuning effort. |
| CPU/GPU 256-squared evidence | Clean CPU and eight clean CUDA method receipts contain the mandatory map/image tasks, residual/condition audit, topology, timings, and failures. Independent direct re-decode is distinct from GPU execution. | A 25-by-25 control mesh with 256-by-256 queries is not a 256-by-256 control mesh. Local CUDA tests were unavailable; remote receipts were reviewed rather than relabelled as fresh local executions. |
| Memory, timing, failures | CUDA timers are synchronized; stage accounting was checked. CPU HWM is process lifetime, allocator peaks are not process memory, and failed attempts consume work even when no completed solve is counted. | No universal speed, memory, or convergence claim follows. |
| O4 trust negative control | Baseline O4 is unchanged; trial re-decodes, rejected successful solves, area floors, and failures are retained. The formal trust variant still fails. | No vertex repair, weakened topology tolerance, untested larger-trial guarantee, or substituted exact-83 row. |

## 3. Fresh executable verification

On the frozen HEAD, the final combined selection returned **202 passed, 7
skipped in 51.87 seconds, exit code 0**. All seven skips explicitly report that
CUDA is unavailable. The selection was:

```text
test_phase5_mvc.py
test_mvc_retraction.py
test_phase5_mvc_lift_comparison.py
test_phase5_mvc_pseudoinverse_comparison.py
test_phase5_mvc_incremental.py
test_phase5_mvc_instance_optimization.py
test_phase5_mvc_o4_trust.py
test_phase5_route2_mvc_roundtrip.py
test_phase5_route2_mvc_layer_benchmark.py
test_phase5_route2_mvc_instance_benchmark.py
test_phase5_route2_incremental_benchmark.py
test_phase5_tutte_direct.py
test_phase5_tutte_iterative.py
test_phase5_tutte_boundary.py
test_tutte_directed_topology.py
test_injectivity.py
```

These are files under `tests/`, passed to `pytest.main([...,'-q'])` after
`sys.path.insert(0,'src')`. On this Windows environment an initial ordinary
invocation aborted on duplicate Intel OpenMP initialization. The successful
bootstrap imported NumPy and evaluated `np.linalg.cond(np.eye(1))` before
importing pytest/Torch. It did not set `KMP_DUPLICATE_LIB_OK`, alter source, or
ignore a failed test. An intermediate bootstrap without the source-path insert
selected an installed older package and failed collection; that invocation is
not counted as a successful test run.

A separate inline checker oracle used a six-vertex/six-face disk with degree
five and degree three interior rows, seed 94102, and a non-square diamond
boundary. It independently assembled the two-by-two interior NumPy system,
evaluated MVC half angles from face arcs, assembled the decoder Jacobian by
row forcing and a linear solve, and solved the weighted minimum-norm problem.
It did not obtain its expected Jacobian from the package VJP. Observed errors:

| Independent comparison | Error |
|---|---:|
| NumPy decoder versus package | `2.78e-17` maximum coordinate |
| Hand MVC versus package canonical logits | `3.33e-16` maximum |
| Analytic decoder Jacobian versus central differences | `1.19e-10` Frobenius |
| Covariance lift versus independent weighted pseudoinverse | `3.52e-16` norm |
| Decoder Jacobian applied to covariance lift versus requested tangent | `2.87e-17` norm |
| Full M1 directional VJP versus finite differences, logits/boundary | `1.53e-10` / `1.14e-10` absolute |
| Full M2 directional VJP versus finite differences, logits/boundary/direction/step | `1.16e-12` / `2.89e-12` / `3.37e-12` / `1.92e-11` absolute |

The raw supported Jacobian had rank four and kernel dimension four; removing
the two row shifts leaves the expected two-dimensional geometric kernel.
This oracle is supplementary numerical evidence, not the proof of the theorem.

## 4. Receipt-level reconstruction and fresh replays

The checker reparsed the receipts rather than inferring correctness from test
counts. All 12 local Markdown links in chapters 05--07 resolved. Key checks:

- The two [local](raw_results/route2_mvc_roundtrip_local.json) and
  [AI](raw_results/route2_mvc_roundtrip_ai.json) round-trip matrices each contain
  27 cases: sides 11/25/49, three seeds, three strengths. Maximum vertex errors
  are `7.02e-15` and `9.40e-15`; barycentric residuals and topology are recorded
  separately. These are 54 observations, not a uniform conditioning theorem.
- The [explicit pseudoinverse receipt](raw_results/route2_mvc_pseudoinverse_local.json)
  has a 12-by-6 common-gauge Jacobian of rank four at threshold `6.972e-14`.
  Euclidean norms are `0.2435816` (Moore--Penrose), `0.2601148` (MVC), and
  `0.2437569` (arithmetic-recentered covariance); the distinct pairwise vectors
  have Jacobian images below `3.1e-17`. Native weighted covariance norm and
  weighted-pseudoinverse agreement were checked separately.
- Both [local](raw_results/route2_incremental_local.json) and
  [Turing](raw_results/route2_incremental_turing.json) incremental receipts have
  24 cases. All table means, warm/correction tuple differences, and Woodbury
  timing ratios were independently recomputed. Matching warm/correction
  tuples are 4/24 and 5/24, not algebraic or numerical identity of the two
  Krylov trajectories. The nine eligible Woodbury cases per host agree with
  refactorization within `3.11e-15`; N49 five-percent changes 111 rows and
  exceeds the declared 64-row cap.
- From the clean CPU [map](raw_results/route2_mvc_instance_formal_map_clean_cpu_b7169ef.json)
  and [image](raw_results/route2_mvc_instance_formal_image_clean_cpu_b7169ef.json)
  receipts, the exact-83 map states have steps 41/41/27/81 and objectives
  `7.0055138e-5`, `4.9662127e-5`, `6.7622553e-5`, `4.8074105e-7` for O1--O4.
  Every ledger satisfies completed global = primal + adjoint. Threshold first
  hits, best values, accepted-state topology, and absent failed-state values
  were recomputed from the traces. Image O4 has no exact-83 state.
- Eight `route2_mvc_instance_formal_{map,image}_gpu_O{1,2,3,4}_ai_74b452c_clean.json`
  receipts report clean provenance. Cross-host full-trace objective differences
  remain below `1.1e-9` for O1--O3 and `2.2e-9` for O4; CPU/GPU target summaries
  differ by at most `2.84e-14`, not bitwise equality. The condition/residual
  forward-error bounds were checked against the independent direct authority.
  Dense SVD condition audits are offline bounded checks, not matrix-free
  layer memory or complexity evidence.
- Layer timing sums and accuracy fields were checked for all six rows on each
  of local CPU, AI A6000 and Turing A40. Layer float64 rtol is `1e-11` with
  2,000 primary iterations; formal-instance float64 rtol is `1e-10` with 500.
  Completed execution does not imply acceptable gradients: A6000 float32 N49
  M2 FD error is about `0.19987`; the tighter-tolerance sweep reaches only
  about `0.001338`, with no preregistered pass threshold. The fresh A40 M2
  allocator peak is `1,150,018,048` bytes, or 1.150 GB / 1.071 GiB; its ratio
  to the original A40 peak is about 0.993, so accumulated cache is not an
  adequate explanation. Runtime/build and hardware effects are confounded.

During this independent review, full-size N25/R256 CPU O4 map, image, and trust
runs were repeated from the current executable code. Their temporary receipts
were inspected, but are not substituted for the durable clean receipts:

| Replay | Independent result |
|---|---|
| Baseline map, rate `0.001`, 81 updates | 81 accepted, 83 completed solves, objective `4.807410494346388e-7`; full objective trace identical to the clean authority; exit 0. |
| Baseline image, same rate | Rejected retraction decode at attempted step 18; 17 accepted, 19 completed solves, objective `0.00433907382463981`; full objective trace/failure identical; intentional failure exit 2. |
| Formal image trust variant | 31 accepted; all 12 trials at step 32 rejected; final objective `0.0045855503993316666`; trace and solve-budget audit identical to the durable negative control. |

For [formal trust](raw_results/route2_o4_trust_formal_image_n25_lr0p001_cpu.json),
the final accepted state is at 97 completed solves; terminal accounting reaches
102 completed solves, 137 primal attempts and 135 trial attempts. The trace jumps
from 80 to 86 solves, so it has no exact-83 observation. At the last trial,
scale `1/2048` still produces area ratio `1.41885e-5`, below the required
`1.61224e-5`. This disproves practical success of the tested bounded rule on
this instance, not all possible trust methods. The rejected finite-precision
map is not a counterexample to the exact positive-Tutte theorem.

## 5. Corrections required by this review

The author, not this checker, corrected the following before the final frozen
HEAD was accepted:

1. Proposition 5.2 now explicitly inputs a logit tangent to the logit decoder
   derivative; it no longer mixes a probability base variable with a logit
   perturbation.
2. The VJP discussion now identifies the ordinary supported-row gradient sum,
   not an incorrectly implied probability-weighted gradient sum.
3. Equal completed-solve budget and equal outer-update count are distinguished;
   failed-attempt work remains visible in attempts and elapsed time.
4. The encoder reports barycentric residual diagnostically; it does not reject
   on an unimplemented residual threshold. Actual geometric rejection checks
   are stated accurately.
5. Layer and instance GPU tolerances/iteration limits are distinguished, and
   GB/GiB units are corrected.
6. An extrapolation about increasing the trust trial limit was replaced by the
   explicit statement that larger limits were not tested.

The earlier runner/provenance/residual-contract corrections and retained
failure history in chapter 07 were also reviewed. The bounded PASS does not
retroactively validate rejected receipts or erase the historical failures.

## 6. Prior-art and final scope adjudication

Primary-source checks support the distinctions in chapter 05: the MVC formula
is classical [Floater 2003](https://doi.org/10.1016/S0167-8396(03)00002-5);
the applicable injectivity hypotheses come from
[Floater's one-to-one P1 result](https://doi.org/10.1090/S0025-5718-02-01466-7).
Injective barycentric tiling morphs already appear in
[Floater--Gotsman](https://doi.org/10.1016/S0377-0427(98)00202-7).
[Neural Cages](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yifan_Neural_Cages_for_Detail-Preserving_3D_Deformations_CVPR_2020_paper.pdf)
uses differentiable cage coordinates, while
[Generative Escher Meshes](https://arxiv.org/abs/2309.14564) already differentiates
positive Laplacian parameterizations. Neither should be silently relabelled as
new here. [Variational Barycentric Coordinates](https://hdl.handle.net/1721.1/153282)
learns coordinate functions over cages, a different object from selecting a
section of a graph-equilibrium decoder. Targeted verification, informed by the
lightweight research-suite source discipline, supports these distinctions but
does not prove novelty by absence of search results.

All section-31 deliverables now have evidence: prior art, canonical theorem,
exact round-trip identity plus numerical tests, differentiable M1 and M2,
supervised map fitting, 256-squared image fitting, raw/MVC optimization
comparison, incremental benchmark, and this independent checker. Section-49
MVC items are individually addressed in the table above.

The remaining limitations are substantive: poor tested float32 GPU gradients,
unfavorable measured GPU time, a large runtime-specific allocator peak,
nonuniform finite-precision availability, O4 image instability, no fair-tuning
optimizer ranking, no CNN training, and no exact arbitrary prescribed-Beltrami
solver. Neither `|mu|<1` nor sampling a continuous diffeomorphism is used as a
P1 fold-free proof. **Route II is closed as bounded research; the overall
production-layer objective is not closed.**
