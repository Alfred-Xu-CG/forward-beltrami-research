# Phase V final decision: Tutte, MVC, and positive-Hodge neural layers

Decision time: 2026-09-22 18:10:51 Asia/Shanghai. The immutable start time is
2026-09-22 00:10:26 Asia/Shanghai. The live completion check reported 64,825
elapsed seconds and `PHASE5_COMPLETION_ALLOWED`; therefore this document was
created only after the real T+18h boundary. The independent review began after
T+17h15m and committed a **QUALIFIED PASS for bounded synthesis** and a
**FAIL for production-objective completion** in
[14_final_independent_review.md](14_final_independent_review.md).

## 1. Executive decision

Phase V has completed its research objective: all three planned routes were
implemented deeply enough, tested on the common 256-by-256 tasks, independently
checked, and compared without discarding negative results. This is **not** the
same claim as completing the intended general solver.

The strongest implemented end-to-end statement is the following conditional
one. Let the source be an oriented, embedded triangulated topological disk. Fix
its boundary vertices in cyclic order on an orientation-preserving convex
polygonal boundary. Give every supported barycentric coefficient strictly
positive value and solve the interior equilibrium equations. Subject to the
weak-convex/dividing-edge hypotheses recorded in the Route-I theorem audit and
to a successful fail-closed numerical solve, the returned nodal embedding and
its piecewise-affine extension are a planar homeomorphism. The layer has a
first-order implicit reverse pass and was exercised in map fitting and dense
image warping. This is the current hard topology mechanism.

The stronger original objective is **not achieved**. The repository does not
yet contain a solver that, for an arbitrary admissible Beltrami coefficient on
a general mesh, simultaneously gives exact Beltrami reproduction, a universal
finite-precision PL-homeomorphism guarantee, fast and reliable float32 GPU
execution at large control-mesh scale, low-memory end-to-end CNN training, and
unrestricted gradient backpropagation. In particular, no 512-by-512 P1
positive-Hodge optimization, million-control-vertex run, higher-order
differentiation, clinical/real-data study, or image-to-latent CNN experiment was
completed.

The practical architecture supported by this phase is therefore

\[
 \boxed{
 \text{QC/metric latent}
 \longrightarrow
 \text{strictly positive planar graph weights}
 \longrightarrow
 \text{Tutte/electrical implicit solve}
 \longrightarrow
 \text{PL homeomorphism}
 }
\]

with full Whitney/P1 used as a teacher or reference rather than as the source of
the topology theorem. MVC is useful as a canonical coordinate system and local
optimization geometry, but the present data do not establish it as the best
optimizer.

## 2. What each route actually established

### 2.1 Route I: positive Tutte/electrical decoder

For an interior vertex \(i\), the decoder solves

\[
 \sum_{j\sim i} w_{ij}(x_i-x_j)=0,\qquad w_{ij}>0,
\]

with fixed convex boundary coordinates. Directed row weights produce a
nonsymmetric reduced system; shared undirected conductances produce a symmetric
positive-definite reduced Laplacian. These are distinct operator families. For
a linear system \(A(\theta)x=b(\theta)\) and loss cotangent \(g\), the reverse
pass solves

\[
 A(\theta)^\mathsf{T}\lambda=g,
 \qquad
 \frac{d\mathcal L}{d\theta}
 =\lambda^\mathsf{T}
 \left(\frac{\partial b}{\partial\theta}
      -\frac{\partial A}{\partial\theta}x\right).
\]

The directed implementation therefore uses the true transpose scatter, whereas
the symmetric edge-conductance implementation has the edge derivative
\(- (B\lambda)_e\!\cdot(Bx)_e\). Finite differences and independent dense
systems checked these formulas. The most reliable GPU/batched topology core in
the bounded tests is the **symmetric positive-conductance, matrix-free CG
decoder**. A retained-factor CPU direct solve remains attractive for small
single meshes. Neither is a universal speed winner.

Route I also measured control sides 11--49, batch sizes 1--8, one to four
composed layers, 256/512 dense queries, selected 1024-squared coordinate
workloads, conditioning failures, timings, and memory. The hard statement is a
conditional mathematical embedding theorem plus numerical input/output
validation, not merely an observed zero flip count. See
[04_tutte_checker.md](04_tutte_checker.md).

### 2.2 Route II: MVC canonical coordinates and covariance updates

Let \(D(p)\) be the Tutte decoder from redundant positive weights \(p\), and
let \(E_{\rm MVC}(Y)\) encode a valid decoded embedding \(Y\) by mean-value
coordinates on every one-ring. On the audited strict domain,

\[
 D(E_{\rm MVC}(Y))=Y.
\]

This is a canonical section of the decoder fiber; it is not the false identity
\(E_{\rm MVC}(D(p))=p\) for arbitrary redundant weights. M1 differentiates the
complete \(E_{\rm MVC}\circ D\) canonicalization. M2 transports a desired
first-order decoded-coordinate update through a weighted minimum-norm
covariance lift and then re-decodes. The implemented lift is not claimed to be
the MVC derivative or a universal Euclidean pseudoinverse.

The experiments compared:

- O1: raw bounded positive weights;
- O2: directed row-softmax weights;
- O3: MVC canonicalization with Adam;
- O4: covariance retraction.

At the common numeric learning rate, O2 completed both tasks and had the lowest
exact-83 objective among O1/O2/O3. O4 was dramatically better on the map task
but failed during the image task at attempted update 18, before an exact-83
state. O3 did not improve on O2 in this cohort. Because the coordinates have
different units, a shared numeric rate is a robustness cohort, not an unbiased
equal-tuning experiment. The bounded recommendation is therefore: **retain O2
as the strongest task-complete observation, keep O1 as the simplest robust
baseline, and treat O4 as a promising but unstable research method**. There is
no established parameterization-fair universal winner. See
[08_mvc_checker.md](08_mvc_checker.md).

### 2.3 Route III: Whitney teacher and positive-Hodge student

For \(\mu=re^{i\phi}\), the repository convention uses the symmetric positive
Beltrami tensor

\[
 A(\mu)=\frac{1}{1-r^2}
 \begin{pmatrix}
  1-2r\cos\phi+r^2 & -2r\sin\phi\\
  -2r\sin\phi & 1+2r\cos\phi+r^2
 \end{pmatrix}.
\]

The compatible reference constructs a Whitney/P1 one-form Hodge \(H_A\), so
the scalar nodal operator is exactly

\[
 L_A=B_0^\mathsf{T}H_A B_0.
\]

Independent element quadrature confirms that this is the ordinary anisotropic
P1 stiffness matrix. The variational cochain \(q_{\rm var}=H_A B_0u\) satisfies
weak equilibrium; on a disk its interior dual stream can be represented by a
face potential. This is not a physical RT0 normal-continuous flux, does not
create a better same-mesh nodal conjugate, and supplies no spatial injectivity
theorem. Annular periods give an explicit obstruction to a global exact stream.

The positive-Hodge student caps \(|\mu|<1\), forms \(A(\mu)\), projects the
face tensor to strictly positive scalar direction coefficients, averages them
to shared edge conductances, and invokes the symmetric Tutte decoder. Thus the
student retains the topology mechanism but generally approximates the target
anisotropy. A finite fixed dictionary of positive rank-one directions cannot
cover every SPD tensor orientation at arbitrary anisotropy. Coverage results
also show that lower local tensor residual need not imply lower assembled
operator, map, or Beltrami error; the stellar graph is the decisive negative
example.

The full Whitney/P-ref path exactly recovered the realizable target-derived P1
maps and passed first-order VJP checks, but it is a CPU, target-informed
one-shot teacher and is not speed-comparable to optimization. The P1 student is
the **hard-valid approximate decoder**. It is not an exact direct decoder for
arbitrary prescribed \(\mu\). See
[11_primal_dual_neural.md](11_primal_dual_neural.md) and
[12_primal_dual_checker.md](12_primal_dual_checker.md).

## 3. Common-input numerical evidence

The primary optimized slice fixes a 25-by-25 control mesh, 256-by-256 dense
queries, seed 20260922, target strength 0.25, the same target and medical
phantom, exact target boundary, float64 NVIDIA A6000-class execution, and the
observation at 83 completed global solves. The table reports the primary
objective and cumulative method wall time through that observation. It is an
input/solve-budget comparison, not a causal hyperparameter ranking.

| Method | Task | Objective at 83 solves | map RMSE | Beltrami RMSE | cumulative time (s) | Result |
|---|---:|---:|---:|---:|---:|---|
| T1/O1 raw positive | map | 7.0055e-5 | 0.012081 | 0.144374 | 28.00 | complete, topology certified |
| T2/O3 MVC Adam | map | 6.7623e-5 | 0.011887 | 0.144522 | 30.60 | complete, topology certified |
| P1 positive-Hodge | map | 7.6587e-6 | 0.004605 | 0.099185 | 17.72 | complete, topology certified |
| T1/O1 raw positive | image | 1.8133e-3 | 0.016079 | 0.151093 | 30.47 | complete, topology certified |
| T2/O3 MVC Adam | image | 1.7501e-3 | 0.017221 | 0.152211 | 36.68 | complete, topology certified |
| P1 positive-Hodge | image | 6.2592e-5 | 0.015286 | 0.158852 | 17.74 | complete, topology certified |

Two essential qualifications accompany this compact table:

1. P1 used route-specific learning rate 0.03 selected on a separate seed,
   whereas O1/O2/O3/O4 used the shared robustness rate 0.001. Initial maps and
   solver tolerances also differ. P1's observed advantage is real for this
   cohort but does not prove universal speed or optimizer superiority.
2. O2 is not one of the selected compact cross-route rows, but its clean
   exact-83 map/image objectives are 4.9662e-5 and 1.2160e-3, better than O1 and
   O3 under the shared rate. O4 gives 4.8074e-7 on the map task but has no
   exact-83 image state. These results prevent a misleading O1-versus-O3-only
   conclusion.

P1's image objective is much lower while its Beltrami RMSE 0.158852 is worse
than O1's 0.151093; image similarity is therefore not evidence of Beltrami
fidelity. P-ref's near-machine-precision recovery is excluded from ranking
because it receives target-derived tensors, uses CPU one-shot inference, and
performs no latent optimization. The authoritative 10-row, 57-column data are
in [13_cross_route_results.csv](13_cross_route_results.csv).

The experiments used clean remote checkouts on the AI A6000, Turing A40, and
Element/AI CPU environments where appropriate. Actual CUDA execution, not a
device label alone, was tested. Remote observations remain hardware/runtime
specific: equal GPU model does not imply equal system load, and framework
memory is not identical across hosts.

## 4. Required A--E decision

### A. Fastest practical hard-bijective layer

**Decision:** use a positive fixed-planar Tutte decoder as the non-negotiable
topology layer. For GPU and fixed-topology batches, the present preferred core
is the symmetric positive-conductance matrix-free CG implementation; for small
single CPU problems, reuse one sparse direct factorization and its transpose
solve. Both have correct first-order implicit gradients.

P1 is the best observed QC-informed instance in the common cohort and inherits
this symmetric core, but the data do not prove it is universally fastest. The
claim is restricted by host synchronization, conditioning-dependent iteration
counts, different tuning, and the absence of large-control-mesh training.

### B. Best optimization parameterization

**Decision under the recorded shared-rate budget:** O2 row-softmax is the best
method that completes both map and image tasks. O4 is the best map-only result
but is rejected as the production default because its image trajectory leaves
the valid decoder domain. O3 MVC Adam supplies a mathematically meaningful
canonical section but does not win this cohort; O1 remains the simplest stable
baseline.

**Decision beyond this cohort:** unresolved. A parameterization-fair winner
requires equal separate-seed tuning budgets, common initial-map semantics where
possible, multiple held-out targets, and time-to-threshold rather than a single
shared numeric learning rate.

### C. Best QC-informed route

**Decision:** split the roles.

- Full Whitney/P-ref is a **teacher/reference** for realizable target-derived
  P1 tensors and for distillation or diagnostic supervision.
- Positive-Hodge/P1 is a **hard-valid approximate decoder** and the best current
  route for inserting QC/metric information into the topology-safe solver.
- Preconditioning and initialization are plausible next roles but are not yet
  experimentally established.
- Neither path is an exact direct decoder for arbitrary prescribed Beltrami
  data.

### D. Main bottleneck

**Decision:** the primary bottleneck is the structural tension between
anisotropic expressivity and the positive scalar graph structure used for the
hard topology theorem. Full tensor P1/Whitney represents the anisotropic
operator exactly but lacks a general embedding theorem; restricting the
operator to positive scalar directions restores Tutte topology but creates a
finite-direction/local-to-global approximation frontier.

Conditioning, finite-precision solver availability, small-kernel GPU
synchronization, and O4 update stability are secondary measured bottlenecks.
Dense warping is not consistently the dominant measured cost. Network
prediction cannot yet be called a bottleneck because end-to-end prediction was
not measured.

### E. Exactly three next-round tasks

1. **Adaptive topology-preserving QC projector.** Learn or optimize a
   multiscale planar positive direction graph against global operator, map, and
   Beltrami losses, distilling from P-ref while preserving strict conductance
   positivity and the audited boundary theorem. Test whether adaptive graph
   refinement improves the center/stellar counterexamples rather than only the
   local tensor residual.
2. **Scalable reliable symmetric solve.** Add and independently check a
   multigrid or strong preconditioner for matrix-free symmetric CG and its
   implicit VJP. Measure actual P1 optimization at 512-by-512 dense resolution
   and then large control meshes/batches, including fail-closed topology,
   float32/float64 forward and VJP accuracy, total activations, process HWM, and
   time-to-useful-solution.
3. **Fair end-to-end neural benchmark.** Train a small image-to-latent encoder
   through O1, O2, O3, and P1 using equal separate-seed tuning budgets and
   multiple held-out targets. Report topology, map/image/Beltrami errors,
   gradient checks, convergence, and full forward/backward memory. Preserve O4
   failures and include it only after a stable validity-preserving update is
   demonstrated.

## 5. Explicit nonclaims and closure status

The following remain false, unsupported, or untested and must not be inferred
from the Phase V pass:

- \(|\mu|<1\) alone does not guarantee that a sampled P1 interpolation is
  fold-free.
- A continuous quasiconformal diffeomorphism does not automatically make its
  fixed-mesh P1 sampling a homeomorphism.
- Conservation or the identity \(B_1B_0=0\) is not a bijection theorem.
- Positive face Jacobians without validated boundary/degree hypotheses are not
  the complete global theorem.
- Exact target-derived P-ref recovery is not arbitrary-\(\mu\) expressivity.
- A solver residual is not an induced Beltrami-error bound.
- First-order custom autograd does not imply second-order differentiation.
- A 256-by-256 dense image is not a 256-by-256 control mesh, and a 1024-squared
  query test is not a million-control-vertex test.
- Synthetic common targets do not establish real medical-registration or
  clinical validity.

Final status: **the 18-hour Phase V exploration is closed with a bounded,
independently reviewed technical decision. The full production solver objective
remains open.** No route was terminated merely because another route succeeded,
and every decisive failure is retained as evidence rather than repaired after
the fact.

## 6. Final verification evidence

After this document was drafted, the selected implementation and application
suites were run together with warnings treated as errors. The first combined
run exposed two unclosed CSV readers in a test file: all functional assertions
passed, but the two resource warnings were correctly treated as failures. The
two-test failure was reproduced, both readers were changed to explicit context
managers, the exact reproduction passed 2/2, and the full fresh selection then
reported **341 passed and 9 skipped**. A final identical selection was also run
with `TEMP` and pytest `--basetemp` explicitly located inside this D-drive
worktree and returned the same counts. Every skip states that CUDA is
unavailable on the local Windows node; the clean remote CUDA receipts remain
the GPU evidence.

The common-results extractor was rerun from the clean authority receipts. It
produced 10 rows and 57 columns and was byte-identical to the committed
[13_cross_route_results.csv](13_cross_route_results.csv). `git diff --check`,
the live completion gate, document-presence checks, and repository provenance
are part of the final handoff verification rather than substitutes for the
scientific limitations above.
