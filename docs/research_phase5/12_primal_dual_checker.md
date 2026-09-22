# Route III independent closure check

Verdict: **PASS for bounded Route III research closure**, after adjudicating the
corrections below. This is not a Phase V final review or completion decision.
Checker context: `/root/route3_final_checker`, independent of the builders.
Reviewed implementation and receipts at `1f62d7c`, canonical projection code at
`f0fa66d`, and the new builder synthesis/table metadata at `c2d5ad8` and
`e56443e`, plus the observation-count correction at `2a671b5`. These later
changes do not alter the Route III solver or experimental
data. Review took place after the Route III closure boundary
`2026-09-22T08:56:21Z`; a live status check during review reported UTC epoch
`1790070121`, elapsed `63095` seconds, with Phase V completion still disallowed.

The accepted outcome is a differentiable positive symmetric Tutte student with
quantified approximation error, together with a full-tensor Whitney/P1 teacher
and a precise local finite-direction obstruction. This satisfies the research
content of PLAN sections 44 and 57, principally Outcomes A/B. Outcome C is
**not** claimed: nodal conjugacy is not improved over the identical P1 problem,
and an RT0 mixed flux implementation was not delivered. No trained CNN,
arbitrary-prescribed-Beltrami reproduction, production readiness, or universal
speed advantage is established.

## 1. Scope and fresh verification

Read the Route III portions of PLAN/WORKLOG, `09_primal_dual_theory.md`,
`10_primal_dual_prototype.md`, and the added `11_primal_dual_neural.md`; inspected
`forward/whitney_hodge.py`, `neural_bijection/tutte/positive_hodge.py`, the
inherited symmetric solver, all four `experiments/phase5/route3_*` runners,
their focused tests, formal/supplement/pilot receipts, and the common CSV
extractor. Read the earlier failed canonical-NNLS checker report; its
`ac50898` verdict is not silently overwritten.

Fresh Windows CPU float64 verification used `PYTHONPATH=src`,
`MKL_THREADING_LAYER=SEQUENTIAL`, and `OMP_NUM_THREADS=1`:

| Check | Result |
|---|---|
| Whitney, positive-Hodge, and P-ref focused suites | 38 passed in 24.10 s |
| Common-table and symmetric-solver suites | 23 passed, 1 CUDA-unavailable skip in 6.26 s |
| Full default coverage replay: three graphs, six radii, 180 held-out angles, two projectors | 6,480 observations; status ok; zero student topology failures |
| Fresh canonical CPU versus clean Linux deterministic summaries | 486 scalars; largest absolute difference 4.5519144e-15 |
| Fresh manufactured Whitney sweep, N5/N9/N25, two tensors | all six runs completed |
| Fresh common-input P-ref N25/R256, both tasks | target identities matched; coordinate error at most 3.9968e-15 |
| Independently reassembled formal/supplement P1 final states | six states; positive face determinants; coordinate discrepancies at most 5.0856e-11 |
| Rebuilt ten-row common CSV | byte-identical to the committed table |

Final combined rerun of the same five test files after the document corrections:
**61 passed, 1 CUDA-unavailable skip in 26.40 seconds**, exit zero.
`git diff --check` also passed. The two formal receipts' frozen projector
state dictionaries were compared directly and are identical.

Temporary checker code/results are `tmp/check_route3_closure.py`,
`tmp/check_route3_receipts.py`, `tmp/route3_final_checker_coverage.json`,
`tmp/route3_final_checker_whitney.json`, and
`tmp/route3_final_checker_cross.csv`. They are diagnostics, not new formal
receipts or a replacement implementation. Initial checker-only script errors
were an empty-matrix SVD on the annulus and NumPy integer JSON serialization;
the empty kernel was handled analytically and the scalar converted, after
which the complete script exited zero. No production code or receipt was
changed by this checker.

CUDA performance was inspected from the clean remote receipts, not rerun on
this CPU-only checker host. New CPU timings are not substituted into those
receipts. These tests and small independent probes do not claim exhaustive
floating-point correctness.

## 2. Finite-direction theorem and mesh hypotheses

**PASS, with the local tensor scope retained.** Normalize nonzero directions
to unit length. Trace bounds the coefficient sum in any convergent sequence
of positive dyad combinations, establishing closedness of the finite cone.
An absent rank-one ray cannot be a positive combination of the listed rays:
testing on its perpendicular direction forces every coefficient to zero.
Closedness then implies that some strictly SPD matrices are missing too;
normalization by the square root of the determinant preserves cone
membership. Thus the result applies on the determinant-one tensor family,
not merely the singular PSD boundary.

The trace-section map `z=((a-d)/(a+d),2b/(a+d))` converts membership exactly
to the convex hull of doubled direction angles. With maximum projective gap
Delta at most pi/2, its origin-centered inradius is `cos(Delta)`. With the
repository's negative Beltrami sign, `|z|=2r/(1+r*r)`. Center and stellar
families consequently have the same closed zero-floor all-orientations
threshold `r=sqrt(2)-1`; extra stellar directions do not close the remaining
45-degree gaps. Strict positivity uses the open threshold, and a fixed
positive floor can introduce additional bias. Standard SW--NE has zero
all-orientations radius and requires a zero diagonal coefficient for exact
isotropy at zero floor.

The three graph builders use actual interior centers/centroids, retain
consistent CCW triangulations, and do not insert crossing diagonals without
a vertex. At two cells per side their V/E/F counts are respectively
9/16/8, 13/28/16, and 17/40/24. Boundary loops and dividing edges are retained;
the standard and stellar meshes each have two dividing edges. A rectangle's
collinear side vertices require the weak-convex/dividing-edge condition, not
a strictly-convex-boundary shortcut.

For the positive student, shared averaging preserves conductance positivity.
The reduced graph Laplacian is SPD with boundary reachability, and positive
normalized equilibrium weights give the applicable Tutte/Floater embedding
under the disk, cyclic convex-boundary, and dividing-edge hypotheses. The
symmetric solver explicitly checks the weakly convex boundary/dividing edges,
recomputes its represented residual, and rejects nonpositive returned faces.
The independent degree/PL inversion justification additionally uses the
injective oriented boundary. Neither SPD, conservation, a positive wide
stencil, nor a residual alone proves spatial injectivity.

The global topology label is conditional on these represented mesh/boundary
and successful-return checks. It does not promise convergence for every
latent, or positive margins uniformly away from degeneracy. Dense sampling
evaluates the control P1 map; arbitrary resampling on a different triangulation
would need a new guarantee.

## 3. Whitney identities, dual exactness, and exact P1 recovery

**PASS for the stated reference; reject stronger conjugacy/topology claims.**
Expanding `w_ij=lambda_i grad(lambda_j)-lambda_j grad(lambda_i)` gives
`sum_e (B0 u)_e w_e=grad(u_h)` pointwise. Midpoint triangle quadrature is exact
for its degree-two tensor-energy integrand. Therefore `B0^T H_A B0` is the
ordinary P1 stiffness, and the edge Hodge is SPD because vanishing local
Whitney fields have all zero edge DOFs. Changing edge signs gives a diagonal
congruence and leaves the potential operator invariant.

`B1 B0=0` is an incidence identity. Conservation of the variational cochain
`q=H_A B0u` is instead `(B0^T q)_I=0`, which follows only after solving the
homogeneous interior PDE. For interior edges `D=(B1^T)_Eint`, with dual
orientation right face to left face. On a connected triangular disk,
`B0_I^T D=0`, `rank(D)=F-1`, and `Eint-Vint=F-1`; boundary reachability gives
the remaining rank, so the conserved interior cochains equal `image(D)`.
This uses no holes and no interior sources.

A fresh independent rank calculation on a 3-by-3-cell disk gave kernel and
dual-image dimension 17. Removing the center cell gave dimensions 16 and 15;
an explicitly constructed unit conserved period had best dual-gradient
residual 1.0. Thus the topology restriction is substantive. The affine
`u=x,A=I` sign test produces the face-centroid y coordinate up to gauge.
The LSQR residual is numerical, not a proof of arbitrary supplied connectivity.

The stream variable lives on faces, not vertices. It is not an RT0
normal-continuous physical flux, nor a reconstructed nodal conjugate. Fresh
N5/N9/N25 runs had maximum operator difference `4.44e-16`, solution difference
`1.47e-14`, energy discrepancy `5.56e-16`, and dual residual `4.35e-11`.
They reproduce the nonzero physical flux jumps and identical P1/Whitney nodal
conjugacy. The N129 remote receipt remains a scaling observation with this
same limitation. The earlier LSQR assertion relaxation changes a numerical
test threshold, not the rank theorem or operator.

For exact P1 recovery, let an already realizable orientation-preserving
continuous P1 map have face Jacobian F. Independently,
`A=det(F) F^{-1} F^{-T}` is its determinant-one Beltrami tensor and satisfies
`J A grad(u)=grad(v)` for positive 90-degree rotation J. Continuity of the
edge tangential derivative of v gives normal-flux cancellation for
`A grad(u)=-J grad(v)`; the second coordinate follows likewise. Hence the
target satisfies the assembled homogeneous P1 equations. SPD uniqueness
with the exact target boundary proves recovery in exact arithmetic.

This was independently tested on a prescribed affine-plus-sinusoidal nodal
map, whose interior was **not** generated by a harmonic solver. Pullback
tensors came from direct edge Jacobian inverses; minimum determinant was
0.8796454 and Whitney coordinate recovery error was `2.22e-16`. This rules
out a merely circular target-solver check. It does not make arbitrary
facewise prescribed mu realizable or make the full-Hodge reference generally
injective. The P-ref formal experiment supplies both target-derived mu and
the target boundary and therefore is an informed one-shot reference.

## 4. Implicit derivatives and computational scope

**PASS for first derivatives in the tested fixed-geometry settings.** For
Dirichlet elimination the adjoint satisfies `L_II^T Lambda_I=g_I` with zero
boundary adjoint. Differentiating energy gives
`dLoss/dA_T=-|T| sym(sum_r grad(Lambda_r) grad(U_r)^T)` and
`dLoss/db=g_B-(L^T Lambda)_B`. The implemented signs, symmetrization, multiple
RHS contraction, boundary term, and reused transposed SuperLU solve agree.
Independent dense differentiation and centered perturbations are present
in the focused tests; fresh manufactured absolute FD error was at most
`6.53e-10`.

For the positive shared-edge graph the fixed-boundary formula is
`dLoss/dc_e=-(B0 Lambda)_e dot (B0 U)_e`. Both interior-interior and
interior-boundary edges are included. Autograd handles averaging, the MLP,
tensor transform, radial latent transform, and the global conductance
normalization outside this implicit solve. An independent probe loaded the
actual formal frozen MLP and traversed latent w through the entire decoder
and dense weighted loss. At N4/R15, centered step `1e-5` gave adjoint
`-3.7559654197`, FD `-3.7559653348`, relative discrepancy `2.26e-8`.
Fresh P-ref full map/image-path relative FD errors were `8.65e-8` and
`8.24e-9`; its nonzero diagnostic probe is excluded from the primary score.

The Whitney solve is CPU float64 SciPy sparse factorization, not GPU,
matrix-free solution, or a batch of distinct systems. Multiple scalar RHS
share one factor. Hodge/matrix-free assembly being O(F) does not make sparse
factorization linear. The student uses the inherited symmetric matrix-free
CG/implicit-adjoint implementation, with no retained Krylov trajectory;
iteration count and conditioning still affect runtime. No geometry VJP,
second derivative, universal CG convergence, or high-resolution float32
gradient accuracy follows from this evidence.

## 5. Canonical NNLS correction and remaining numerical boundary

The original SciPy coefficient choice was underdetermined for redundant
dictionaries and changed shared-edge maps across runtimes. The `ac50898`
checker correctly failed its caller-order lexicographic near-tie selection.
**The `f0fa66d` correction passes the original counterexample and bounded
portability checks; unconditional exact coefficient accuracy is rejected.**

The primary projected tensor is unique. The secondary Euclidean norm is
strictly convex on its closed convex primary-minimizer set and has a unique
minimizer. Uniform floor plus the unit-direction trace identity makes the
minimum-free-coefficient and minimum-actual-conductance objectives equivalent
on each primary fiber. All supports, not only supports of size three, must
be considered for the secondary problem. The code enforces these assumptions
and content-sorts the declared distinct columns before enumeration.

Fresh independent SciPy primary NNLS plus equality-constrained SLSQP secondary
checks covered 90 tensors. Largest primary-residual gap was `7.72e-15`,
secondary coefficient difference `7.50e-15`, and random permutation spread
zero. SLSQP emitted intermediate bound-clipping warnings; all optimizations
completed successfully. The analytic determinant-one center example at
delta `1e-7` gave error `2.22e-16` across all 24 permutations, superseding the
old `1e-7` failure.

At delta `1e-8`, all permutations still agree, but the finite norm tolerance
selects the floor support and incurs `1.00000004e-8` coefficient error;
delta `1e-9` similarly gives about `1e-9`. A tiny squared-norm objective gap
does not certify tiny coefficient error. The builder now explicitly records
this limitation in section 5.2 of `11_primal_dual_neural.md`. PASS means
permutation-equivariant, KKT-screened numerical projection for the declared
dictionary with this limitation, not uniform machine-precision recovery of
the exact secondary minimizer or universal bitwise cross-platform behavior.

`git diff 35a1878 e56443e` leaves the learned instance runner, symmetric
solver, and Whitney implementation unchanged. The local fitting correction
does not enter `_train_projector`, MLP inference, face aggregation, or the
optimized P1 chain. The old learned-P1 formal provenance therefore remains
`35a1878`; relabeling it `f0fa66d` or requiring reruns solely for the offline
NNLS correction would be incorrect.

## 6. Coverage versus assembled operator versus map

**PASS for the disclosed bounded frontier.** The formal coverage authority is
the clean `route3_positive_hodge_coverage_gpu2_f0fa66d.json`. The fresh full
CPU replay independently confirms its deterministic metrics. Learned GPU/CPU
outputs are not asserted bitwise equal, and symmetry-related worst-angle
labels need not be unique.

The local metric is a unit-dyad moment fit. Assembled error compares interior
rows of the actual shared-direction graph against full P1 after one best
positive global scale, which leaves the Dirichlet map unchanged. The separate
global-class NNLS is restricted to one coefficient per direction class; it
is not the best arbitrary per-edge network or a Schur-complement effective
tensor. Map/mu errors compare against the full-P1 teacher under a specified
monotone non-affine rectangle boundary, not against the input mu itself.
The constant-tensor tiny graph experiment is not a general variable-field
convergence theorem.

At radius .9, standard/center/stellar deterministic maximum tensor errors are
0.70504/0.16637/0.16637, while assembled errors are
0.23752/0.83934/0.89531 and map RMSEs 0.007848/0.005199/0.141029.
The stellar mu RMSE reaches 0.880483. These are separate maxima, potentially
at different orientations. They decisively reject treating a richer local
cone as an automatic better global decoder.

## 7. Formal instances, common target, solve/time/memory accounting

**PASS for common-input evidence; no blanket ranking.** All four
formal/supplement P1 receipts are clean at `35a1878`. The formal rows use
target-independent zero w on all faces; the frozen MLP gives at identity
approximately `(0.99192115,0.02012113,0.99088946)`. Thus “uniform” describes
the tensor latent across faces, not equal edge conductances or an identity
interior map. The table metadata has been corrected accordingly. Exact target
boundary values are supplied in every compared instance and must remain
disclosed in any unsupervised-registration interpretation.

The eight separate-seed pilots reproduce geometric-mean objective-ratio
scores .309105/.109344/.032292/.012335 for rates .001/.003/.01/.03,
justifying the predeclared .03 selection. This gives P1 route-specific tuning;
T1/O1 and T2/O3 remain a .001 robustness cohort. Equal inputs and global-solve
counts do not equalize hyperparameter tuning or mathematical work per solve.

Fresh target regeneration matches the declared N25/R256 control/dense numeric
identities; P-ref checks also include latent and boundary summaries. Numeric
summaries are not injective array identities. The deterministic builder and
actual final-state recomputation supply the additional reproduction evidence.
Map uses height .9 and image height 1, consistent across methods within each
task. The image convention is fixed-to-moving backward sampling; no inverse
is computed and no inverse-consistency measurement should be invented.

For each optimized P1 row, trace index k observes `2k+1` completed solves:
one new primal after k earlier primal/adjoint update pairs. Index 41 is the
unique exact83 state. Forward/dense/objective cumulative components include
rows 0 through 41; backward/update sums include only 0 through 40. The later
backward at row 41 cannot be charged to that observation. Final index 81
means 82 primal observations plus 81 adjoints = 163 global calls. Across the
two formal tasks there are **164 serialized observations**, not 163.
The projection-only supplements have one primal/no adjoint; target-initialized
optimization and projection-only rows are correctly rank-ineligible.

The fresh checker independently reassembled Laplacians from serialized final
edge conductances and boundary values, recomputed Jacobians through edge
coordinate inverses, and recomputed map/mu errors for six final states.
All agree with receipts. Intermediate exact83 coordinates are not serialized;
their scalar metrics can be checked against trace/slice consistency, but an
independent geometry recomputation of that exact intermediate state was not
performed. Missing maximum-map-error/minimum-signed-area cells correctly
remain blank instead of borrowing final-state values.

P1 exact83 map/image objectives are `7.65868e-6`/`6.25915e-5`, at method-only
times `17.716`/`17.743` seconds. The first thresholds occur at 5/19 global
calls, with method-only times 1.373/4.283 seconds. Charging frozen-projector
setup yields about 5.383/8.470 seconds. Exact83 and threshold times omit that
setup unless explicitly added; wrapper totals include it. They also cannot
be presented as isolated neural-kernel timings because topology/audit and
Python overhead enter method wall time. The supervised map objective is dense
coordinate MSE, whereas reported map RMSE is a control-vertex Euclidean norm.
Those different measures need not be simple square roots of one another.

P1 has lower image MSE but exact83 mu RMSE .158852 versus T1's .151093.
The image result therefore does not establish improved geometric fidelity.
P-ref has two primary global calls (one two-coordinate forward plus one
two-coordinate adjoint), with separate target construction, direct-P1 oracle,
and FD diagnostic calls. It is target-derived CPU reference, not optimized
GPU competition. Its `comparable=False` table status is correct.

Whole-benchmark CUDA peaks include setup/projector and optimization; process
HWM also includes runtime and final CPU audit. Explicit Whitney/P-ref array
counts omit factor/workspace/runtime allocations and are not peak memory.
The P-ref 512-square receipts demonstrate dense-query/reference differentiation
scaling, not 512-square positive-student optimization. The same A6000 model
on the same host does not establish isolated identical GPU load across runs.

## 8. Adjudicated document issues and final limits

The missing required chapter 11 was supplied at `c2d5ad8`. Independent review
then required three corrections accepted at `e56443e`: identify the stream as
a face-valued variational-cochain diagnostic; distinguish closed zero-floor
threshold from strict positive/fixed-floor behavior; and remove the inference
from local tensor-cone obstruction to impossibility of arbitrary map
expressivity. The initialization and near-transition NNLS limitations are now
explicit. The worklog observation-count typo was corrected at `2a671b5`.

Targeted source rechecks support the cautious prior-art scope:
[FEEC](https://arxiv.org/abs/0906.4325) requires a subcomplex and bounded
cochain projection for the stated stability theory;
[Huang](https://arxiv.org/abs/1008.0562) gives anisotropic Delaunay-type DMP
conditions, not spatial bijection; and
[Geo-NeW](https://arxiv.org/html/2602.02788v1) already learns compatible
metric/operator structure with a conditional implicit-adjoint construction.
No priority or absence-of-prior-art claim is accepted. This is a targeted
recheck, not an independent systematic review of every source in chapter 09.

There is no outstanding blocker to the **bounded Route III** conclusion.
Any stronger conclusion would require new evidence: arbitrary mu realizability,
operator/map-faithful projection beyond local moment fitting, unbiased
optimization/tuning comparisons, accurate larger-scale gradients and solver
reliability, RT0 physical conformity, or end-to-end learned image prediction.
Phase V final synthesis and its separate real-time guard remain the
coordinator/final review's responsibility.
