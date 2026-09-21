# Route II application experiments: differentiable MVC layers and geometric optimization

## 1. Questions, claim boundaries, and current status

This chapter evaluates the computational part of Route II.  It asks four
separate questions.

1. Can the mean-value-coordinate encoder be differentiated as an ordinary
   PyTorch module?
2. Does the canonicalization layer
   (P_{\rm MVC}=E_{\rm MVC}\circ D) preserve the decoded control map?
3. Does the covariance retraction realize a requested first-order vertex
   direction while accepting only a fresh positive-Tutte decoder output?
4. Under a genuinely equal experimental budget, do canonical MVC coordinates
   or covariance retraction improve single-instance map and image fitting over
   raw positive Tutte coordinates?

Questions 1--3 have local CPU and two-host GPU evidence.  Question 4 is **not
yet closed**: an independent review rejected the first O1--O4 runner as a fair
scientific protocol before any formal 256-by-256 result was accepted.  The
rejected smoke observations are not promoted into the result tables below.

Three distinctions are essential.

- `status="ok"` in a layer receipt means that the configured program completed
  and returned topology-certified maps.  It is not a gradient-accuracy pass.
- A positive floating signed area is a numerical property of the returned P1
  map.  It is not an exact-predicate certificate for all real inputs.
- MVC is a canonical encoder for a subset of directed Tutte maps.  The hard
  topology mechanism remains the positive-weight Tutte decoder; neither MVC
  nor the covariance lift is a new global inversion theorem.

The exact MVC domain, round-trip theorem, redundancy counterexamples, and
prior-art audit are in [the theory chapter](05_mvc_theory.md).  The covariance
right inverse, retraction, and incremental linear algebra are derived in
[the optimization chapter](06_mvc_optimization.md).  This chapter defines the
measured quantities and reports only observations present in durable receipts.

## 2. Discrete objects and executable layers

### 2.1 Control triangulation and directed decoder

For a control-side value (N\geq3), the source is the unit square with
(N^2) vertices and (2(N-1)^2) consistently oriented triangles.  Its
interior and boundary vertex sets are (I) and (B).  Each interior vertex
(i) has a fixed cyclic neighbor list (N(i)) of degree (d_i).  Supported
row logits (\ell_{ij}\) define

\[
 p_{ij}=\frac{\exp \ell_{ij}}
 {\sum_{k\in N(i)}\exp\ell_{ik}},
 \qquad j\in N(i).
\tag{2.1}
\]

Padding slots are structurally absent, not zero-weight neighbors.  Given
ordered boundary positions (b), the directed decoder (D) solves

\[
 Y_i=\sum_{j\in N(i)}p_{ij}Y_j,\quad i\in I,
 \qquad Y_B=b.
\tag{2.2}
\]

The CPU direct backend assembles (A=I-P_{II}), factors it once per sample,
and solves both coordinate columns.  Its first-order reverse pass solves with
(A^{\mathsf T}).  The GPU backend applies (A) and (A^{\mathsf T})
matrix-free and uses implicit BiCGStab differentiation.  In the measurements
below its relative residual tolerance is (10^{-5}) in float32 and
(10^{-11}) in float64, with zero absolute tolerance and at most 2,000 primary
iterations.  A completed iterative call has also passed the solver's freshly
recomputed represented-residual check.

### 2.2 MVC encoder

For an accepted control map (Y\), let

\[
 e_{ij}=Y_j-Y_i,
 \qquad r_{ij}=\lVert e_{ij}\rVert_2,
\]

and let (\alpha_{ij}\in(0,\pi)) be the consistently oriented angle from
(e_{ij}) to the next one-ring edge.  The unnormalized planar mean-value
weight is

\[
 w_{ij}=
 \frac{
 \tan(\alpha_{i,j-1}/2)+\tan(\alpha_{ij}/2)
 }{r_{ij}},
 \qquad
 p^{\rm MVC}_{ij}=\frac{w_{ij}}{\sum_k w_{ik}}.
\tag{2.3}
\]

The encoder returns the zero-supported-mean logit gauge

\[
 \ell^{\rm MVC}_{ij}=\log p^{\rm MVC}_{ij}
 -\frac1{d_i}\sum_{k\in N(i)}\log p^{\rm MVC}_{ik},
\tag{2.4}
\]

together with the unchanged boundary.  It rejects zero edges, inconsistent
winding, nonpositive consecutive angles, nonpositive weights, large
barycentric residual, or a rank-deficient local covariance.  All geometry in
(2.3)--(2.4) is computed with Torch operations; the combinatorial cyclic order
is precomputed once from the mesh.

### 2.3 M1 canonicalization layer

The complete M1 layer is

\[
 (\ell,b)\xmapsto{D}Y
 \xmapsto{E_{\rm MVC}}(\ell_c,Y_B).
\tag{2.5}
\]

Its output includes the decoded control map (Y), canonical logits
(\ell_c), probabilities, and geometry diagnostics.  When all hypotheses and
the linear solve hold exactly,

\[
 D(\ell_c,Y_B)=Y.
\tag{2.6}
\]

The numerical canonical round-trip error reported below is an independent
second decode of the M1 output; it is not the already available (Y) copied
from the first decode.

### 2.4 M2 covariance-retraction layer

Let (d\in\mathbb R^{|V|\times2}) be a requested vertex direction with zero
boundary component.  At the canonical state define

\[
 \bar Y_i=\sum_jp_{ij}Y_j,
 \qquad s_{ij}=Y_j-\bar Y_i,
 \qquad C_i=\sum_jp_{ij}s_{ij}s_{ij}^{\mathsf T},
\tag{2.7}
\]

and

\[
 b_i=d_i-\sum_jp_{ij}d_j,
 \qquad
 \delta\ell_{ij}=s_{ij}^{\mathsf T}C_i^{-1}b_i.
\tag{2.8}
\]

Equation (2.8) is the unique minimum-
(\sum_jp_{ij}\delta\ell_{ij}^2) row lift under the differentiated
barycentric constraint.  It is not generally the derivative of the MVC
formula.  The accepted finite update is

\[
 R_Y(\alpha d)=D(\ell_c+\alpha\delta\ell,Y_B).
\tag{2.9}
\]

The Euler predictor (Y+\alpha d) is never returned as the accepted map.
M2 performs an initial decode, MVC canonicalization, covariance lift, and a
fresh final decode.  Thus its timing must not be interpreted as one bare
linear solve.

## 3. Layer-benchmark protocol

### 3.1 Inputs and bounded scale matrix

The formal bounded profile is

\[
 (N,B)\in\{(11,8),(25,4),(49,1)\},
 \qquad {\tt dtype}\in\{\text{float32},\text{float64}\}.
\tag{3.1}
\]

The largest row therefore has 2,401 control vertices.  This is a realistic
control mesh, not a three-by-three correctness example.  The decreasing batch
size controls total work; the experiment is not the full Cartesian product of
all three batch sizes and all three meshes.

One CPU generator with seed 2718 draws supported raw logits from
(0.12\mathcal N(0,1)).  The boundary is the identity unit-square boundary.
For source vertex ((x,y)), the requested direction is

\[
 d(x,y)=0.02\left(
 \sin(\pi x)\sin(\pi y),
 0.7\sin(2\pi x)\sin(\pi y)
 \right),
\tag{3.2}
\]

scaled by (1+0.1k) for batch item (k), and set exactly to zero on the
boundary.  The finite M2 step is (\alpha=0.05).  Every formal row has one
warmup and two measured repeats.

### 3.2 Measured stages

The receipt measures three distinct autograd graphs.

- `encoder_only`: (E_{\rm MVC}(Y)) for a detached accepted (Y).
- `full_m1`: (E_{\rm MVC}(D(\ell,b))).
- `canonical_m2`: the complete operation in (2.9), starting from raw logits.

For each graph the input clone is excluded.  Forward, deterministic scalar
loss construction, first-order reverse-mode backward, and their end-to-end sum
are timed separately.  CUDA is synchronized at every boundary.  CPU uses one
Torch, OMP, and MKL thread.  The measured backward differentiates with respect
to every declared graph input; finite gradients are necessary for execution
completion but are not an accuracy test.

GPU memory is the absolute PyTorch allocator peak after resetting peak
statistics following warmup.  The receipt also stores allocated and reserved
baselines.  It is not process RSS, driver context memory, or a proof of memory
complexity.  In particular, an absolute peak from two different Torch/CUDA
builds is an implementation observation, not a hardware-only comparison.

### 3.3 Accuracy and topology metrics

Let (Y_c=D(E_{\rm MVC}(Y))).  The canonical round-trip metric is

\[
 e_{\rm round}=\max_i\lVert (Y_c)_i-Y_i\rVert_2.
\tag{3.3}
\]

For a central step (h), the finite-difference first-variation error is

\[
 e_{\rm FD}=
 \frac{
 \left\lVert [R_Y(hd)-R_Y(-hd)]/(2h)-d\right\rVert_2
 }{\lVert d\rVert_2}.
\tag{3.4}
\]

The default is (h=2\times10^{-2}) in float32 and
(2\times10^{-4}) in float64.  These values are disclosed because iterative
solve noise is amplified by division by (h).  The projected reverse-mode
identity chooses a deterministic cotangent (g) and reports

\[
 e_{\rm proj}=
 \frac{
 \left|\partial_\alpha\langle g,R_Y(\alpha d)\rangle|_{\alpha=0}
 -\langle g,d\rangle\right|
 }{\max(|\langle g,d\rangle|,{\tt tiny})}.
\tag{3.5}
\]

This is one dual projection, not a complete Jacobian comparison.  Independent
CPU tests additionally compare (dE_{\rm MVC}) and the covariance lift and
verify that their latent difference decodes to zero to first order.

The represented row residual is

\[
 r_i=Y_i-\sum_jp_{ij}Y_j,
\quad
 \rho=\max_i
 \frac{\lVert r_i\rVert_2}
 {\max(\lVert Y_i\rVert_2,
 \lVert\sum_jp_{ij}Y_j\rVert_2,{\tt tiny})}.
\tag{3.6}
\]

For every base, canonical re-decode, zero-step M2, and finite-step M2 map, an
independent P1 audit reports flip count, minimum signed target-face area,
minimum determinant ratio, boundary gap, and global certificate.  A row is not
allowed to complete if this independent audit fails.

## 4. Environments and reproducibility

The local formal receipt was produced from commit `b2972ea`; the two remote
receipts were produced from commit `eeaed32`.  The latter commit adds only the
already generated local JSON to `b2972ea`, so the executable benchmark and
layer source are identical between these two provenance points.  The raw files
are:

- [local CPU direct](raw_results/route2_mvc_layer_local_cpu_b2972ea.json);
- [AI A6000-0](raw_results/route2_mvc_layer_ai_a6000_gpu0_eeaed32.json);
- [Turing A40-0](raw_results/route2_mvc_layer_turing_a40_gpu0_eeaed32.json).

The embedded remote identity is
`eeaed327ca563cfc4b536e83994350bea354c5a8`.  AI used Torch 2.4.0 with CUDA
12.1; Turing used Torch 2.8.0 with CUDA 12.8.  Both selected physical GPU 0 through
`CUDA_VISIBLE_DEVICES=0`.  Before the formal matrix, the actual-CUDA focused
test suite passed 13/13 independently on both hosts.

The local direct matrix completed 6/6 rows in 84.13 seconds of harness time.
AI completed 6/6 in 102.16 seconds and Turing completed 6/6 in 111.31 seconds.
Here “completed” retains the narrow execution meaning from Section 1.

## 5. Forward, backward, and memory observations

### 5.1 Mean end-to-end time

The following values are the mean of two post-warmup repeats and include each
stage's declared forward, scalar loss, and first-order backward.  Units are
milliseconds.

| Device/backend | dtype | (N,B) | encoder only | full M1 | canonical M2 |
|---|---|---:|---:|---:|---:|
| local CPU/SuperLU | float32 | 11,8 | 5.19 | 237.14 | 489.51 |
| local CPU/SuperLU | float32 | 25,4 | 15.60 | 731.24 | 1452.46 |
| local CPU/SuperLU | float32 | 49,1 | 18.62 | 813.12 | 1518.77 |
| local CPU/SuperLU | float64 | 11,8 | 5.46 | 275.36 | 483.23 |
| local CPU/SuperLU | float64 | 25,4 | 12.33 | 788.32 | 1499.62 |
| local CPU/SuperLU | float64 | 49,1 | 13.66 | 755.93 | 1487.30 |
| AI A6000/matrix-free | float32 | 11,8 | 4.44 | 313.89 | 651.96 |
| AI A6000/matrix-free | float32 | 25,4 | 5.54 | 787.39 | 1809.97 |
| AI A6000/matrix-free | float32 | 49,1 | 3.88 | 953.00 | 2764.64 |
| AI A6000/matrix-free | float64 | 11,8 | 5.27 | 349.63 | 735.23 |
| AI A6000/matrix-free | float64 | 25,4 | 4.84 | 930.38 | 1987.15 |
| AI A6000/matrix-free | float64 | 49,1 | 5.88 | 1333.48 | 2874.72 |
| Turing A40/matrix-free | float32 | 11,8 | 5.47 | 341.64 | 709.74 |
| Turing A40/matrix-free | float32 | 25,4 | 5.86 | 871.14 | 1769.27 |
| Turing A40/matrix-free | float32 | 49,1 | 5.85 | 1077.22 | 3056.54 |
| Turing A40/matrix-free | float64 | 11,8 | 6.45 | 403.78 | 849.18 |
| Turing A40/matrix-free | float64 | 25,4 | 6.27 | 1058.64 | 2236.30 |
| Turing A40/matrix-free | float64 | 49,1 | 5.46 | 1573.67 | 3261.61 |

The GPU times are not faster than the one-thread CPU direct reference in this
matrix.  The encoder performs many small one-ring Torch operations, and the
matrix-free solver includes convergence decisions and returned-map checks.
These observations do not prove that a fused implementation cannot be fast;
they do show that the current prototype is not yet a fast GPU layer at these
batch/mesh sizes.  M2 is more expensive than M1 because it intentionally
contains an additional decode and covariance work.

### 5.2 GPU allocator peaks

| Host/build | dtype | (N,B) | M2 baseline allocated MiB | M2 peak allocated MiB |
|---|---|---:|---:|---:|
| AI, Torch 2.4/CUDA 12.1 | float32 | 11,8 | 16.33 | 19.55 |
| AI, Torch 2.4/CUDA 12.1 | float32 | 25,4 | 16.61 | 27.06 |
| AI, Torch 2.4/CUDA 12.1 | float32 | 49,1 | 17.42 | 28.44 |
| AI, Torch 2.4/CUDA 12.1 | float64 | 11,8 | 16.36 | 22.77 |
| AI, Torch 2.4/CUDA 12.1 | float64 | 25,4 | 16.71 | 37.55 |
| AI, Torch 2.4/CUDA 12.1 | float64 | 49,1 | 17.58 | 39.44 |
| Turing, Torch 2.8/CUDA 12.8 | float32 | 11,8 | 16.33 | 182.99 |
| Turing, Torch 2.8/CUDA 12.8 | float32 | 25,4 | 16.61 | 560.72 |
| Turing, Torch 2.8/CUDA 12.8 | float32 | 49,1 | 17.42 | 585.74 |
| Turing, Torch 2.8/CUDA 12.8 | float64 | 11,8 | 16.36 | 349.32 |
| Turing, Torch 2.8/CUDA 12.8 | float64 | 25,4 | 16.71 | 1104.87 |
| Turing, Torch 2.8/CUDA 12.8 | float64 | 49,1 | 17.58 | 1152.65 |

The identical baselines but sharply different peaks require investigation.
The current evidence does not isolate GPU architecture, Torch version, CUDA
version, scatter implementation, or allocator behavior.  It is therefore
incorrect to quote either host as a universal M2 memory requirement.  The
A6000 observation is encouraging for million-pixel applications because this
control-only experiment remains below 40 MiB, but it excludes the dense image
grid, images, optimizer parameters, and any CNN.  It does not answer the full
training-memory question by itself.

## 6. Numerical accuracy and topology

### 6.1 Maximum error over the three profiles

| Backend/device | dtype | max round-trip (e_{\rm round}) | max (e_{\rm FD}) | max (e_{\rm proj}) | max M2 residual (\rho) | min M2 area ratio |
|---|---|---:|---:|---:|---:|---:|
| local CPU direct | float32 | 1.481e-6 | 8.613e-4 | 6.057e-5 | 7.887e-8 | 0.706508 |
| local CPU direct | float64 | 2.758e-15 | 2.090e-10 | 1.470e-14 | 8.811e-16 | 0.706506 |
| AI A6000 matrix-free | float32 | 4.066e-4 | 1.999e-1 | 3.377e-1 | 5.377e-6 | 0.706562 |
| AI A6000 matrix-free | float64 | 1.329e-10 | 4.227e-5 | 9.289e-8 | 8.318e-12 | 0.706506 |
| Turing A40 matrix-free | float32 | 4.066e-4 | 1.999e-1 | 3.209e-1 | 5.377e-6 | 0.706562 |
| Turing A40 matrix-free | float64 | 1.329e-10 | 4.227e-5 | 1.138e-7 | 8.318e-12 | 0.706506 |

Every base, canonical, zero-step, and finite-step returned map in the 18 rows
had zero flips and passed the independent global P1 audit.  The minimum area
ratios in the table are comfortably positive for this mild test direction.
This establishes bounded returned-map legality, not a uniform margin for all
learned logits or step sizes.

### 6.2 Why float32 completion is not sufficient

For (N=49,B=1) on A6000, the individual float32 values are

\[
 e_{\rm round}=4.066\times10^{-4},\qquad
 e_{\rm FD}=1.999\times10^{-1},\qquad
 e_{\rm proj}=3.377\times10^{-1}.
\tag{6.1}
\]

The maximum represented M2 residual is only (2.731\times10^{-6}) in that
row, but the displacement used by the central difference is itself small:
(h\lVert d\rVert\) has the (h=0.02) factor from Section 3.3.  Solver error is
therefore divided by both (h) and the direction scale in (3.4).  A residual
acceptable for rendering a map can be inadequate for a derivative test.

The cross-host near agreement of round-trip and finite-difference errors shows
that this is reproducible under the two tested builds.  It does not prove that
the residual tolerance is the only cause.  The separately launched explicit
tolerance sweep must distinguish attainable float32 solver accuracy from a
formula or implementation defect.  Until that sweep closes, the supported
statement is:

\[
 \boxed{\text{The current float32 directed GPU path is topology-safe here,
 but not accurate enough for the tested N=49 M1/M2 gradients.}}
\tag{6.2}
\]

Float64 is much more accurate in the same implementation, but its M2 mean time
is 2.87 seconds on A6000 and 3.26 seconds on A40 for (N=49,B=1), before any
dense 256-by-256 image loss or CNN.  It is therefore a correctness option, not
yet the desired fast neural-layer solution.

## 7. Independent M1/M2 correctness evidence

The focused local suite, after adding a regression that forbids a nominally
successful receipt from containing nonfinite timing/audit values, reports 12
passes and one CUDA-unavailable skip.  The combined MVC, retraction,
round-trip, direct, and iterative regression reports 103 passes and five local
CUDA-unavailable skips before that extra regression, hence 104 corresponding
passes after it.

The independent latent-lift comparison uses a variable-degree disk.  It
computes the autograd JVP (dE_{\rm MVC}(Y)[d]) and the covariance lift
(L_Yd).  After removing the row-softmax shift gauge, the two probability
tangents differ, while finite-difference decoder evaluations of both realize
the same requested vertex tangent.  Their difference decodes to zero to first
order.  This directly supports

\[
 J_DdE_{\rm MVC}(Y)[d]=J_DL_Yd=d,
 \qquad
 J_D(dE_{\rm MVC}-L_Y)d=0,
\tag{7.1}
\]

without asserting equality of the two latent vectors.

## 8. O1--O4 instance protocol under correction

### 8.1 Methods to be compared

All methods will use the same positive-Tutte target, exact target boundary,
uniform supported-row initialization, decoder implementation, dense query
table, loss, dtype, device, and declared optimizer hyperparameters.

- O1 represents positive entries as (a_{ij}=\operatorname{sigmoid}(q_{ij}))
  and supplies (\log a_{ij}) to the row-softmax decoder.
- O2 directly optimizes supported row logits.
- O3 takes an ordinary logit-Adam step, decodes the candidate, re-encodes it
  with MVC, and decodes the canonical state.  Retaining Adam moments across
  this projection is an explicit identity-transport policy, not plain
  unconstrained Adam.
- O4 computes the dense-loss gradient with respect to control vertices,
  applies vertex-space Adam with zero boundary direction, lifts the step with
  (2.8), and accepts only the decoder output (2.9).  Local reverse-mode work is
  reported separately from transpose global solves.

### 8.2 Two noninterchangeable fairness slices

A common 40-update trajectory is secondary evidence about what each update
rule does per outer iteration.  It does **not** have equal solve cost.  With
initialization included and target setup excluded, its totals are

\[
 (G_{\rm O1},G_{\rm O2},G_{\rm O3},G_{\rm O4})
 =(81,81,122,42).
\tag{8.1}
\]

The primary solve-budget slice uses exactly 83 completed global solves:

\[
 \begin{array}{c|c|c}
 \text{method}&\text{accepted updates}&\text{global solves}\\\hline
 \text{O1}&41&1+2(41)=83\\
 \text{O2}&41&1+2(41)=83\\
 \text{O3}&27&2+3(27)=83\\
 \text{O4}&81&2+81=83.
 \end{array}
\tag{8.2}
\]

One multi-right-hand-side primal or transpose call counts once.  The shared
target-generation solve is recorded separately.  Failed calls must be
distinguished from completed calls; Krylov iterations and true residuals are
additional work/accuracy measures, not aliases for call count.

### 8.3 Why the first runner was rejected

The independent audit found that the preliminary implementation:

1. compared equal outer steps while calling it a fair solver-budget result;
2. accepted a finite but control/dense-inconsistent external target;
3. incremented some failure-path solve counters before a solve occurred;
4. used a hard-coded decoder-provenance flag rather than an independent
   re-decode check;
5. left one O4 encoder call outside its component timing;
6. omitted dense interpolation/image-warp/loss forward timing;
7. used generic failure phases and ambiguous covariance labels; and
8. did not retain iterative residual/iteration diagnostics.

The current independent verdict is therefore **FAIL pending correction**.
Successful tiny smoke losses are intentionally omitted because publishing them
before the protocol repair would reward a known unfair comparison.

### 8.4 Formal experiments still required

After checker re-approval, the fixed protocol must run at least:

- supervised map fitting with (N=25,R=256), seed 20260922, target strength
  0.25, and objective threshold (10^{-4});
- medical-phantom registration with the same (N,R,) seed, strength, target,
  backward-warp convention, and threshold (10^{-3});
- common-40-step and exact-83-solve slices;
- one shared-learning-rate cohort, explicitly interpreted as a robustness
  slice; and, only if budget permits, an equal-size predeclared learning-rate
  pilot per method reported separately;
- every accepted-state topology metric, map and Beltrami error, best/final
  objective, threshold solve count, full forward/backward/step timing,
  condition diagnostics, and all failures.

The 256-by-256 image has 65,536 query points; the (25\)-side control mesh has
625 vertices.  These numbers must remain distinct in every result table.

## 9. Current answer to the project objective

Route II currently supplies a mathematically defined, differentiable M1
canonicalization layer and M2 decoder retraction.  Both return piecewise-affine
maps screened under the positive-Tutte topology conditions, and their local
float64 derivatives have strong independent numerical agreement.  It does
**not** yet supply the requested final fast, accurate, memory-efficient neural
layer:

- current GPU execution is slower than the CPU direct reference for the tested
  bounded profiles;
- float32 (N=49) derivative accuracy is poor at the existing directed-solver
  tolerance;
- float64 corrects accuracy but raises time and, on one runtime, allocator
  memory substantially;
- no plan-compliant O1--O4 256-by-256 comparison has yet been accepted; and
- no CNN has been trained through this Route-II layer.

These are research results, not reasons to abandon the route.  The immediate
decisive tasks are the tighter-float32 solver sweep, the repaired equal-budget
instance benchmark, and an independent Route-II checker.  Only then can the
route be classified as a practical candidate, a correctness-only teacher, or
a negative result for the final neural-layer goal.
