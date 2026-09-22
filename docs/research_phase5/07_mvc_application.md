# Route II application experiments: differentiable MVC layers and geometric optimization

## 1. Questions, claim boundaries, and current status

This chapter evaluates the computational part of Route II.  It asks four
separate questions.

1. Can the mean-value-coordinate encoder be differentiated as an ordinary
   PyTorch module?
2. Does the canonicalization layer
   \(P_{\rm MVC}=E_\ell\circ D_\ell\) preserve the decoded control map?
3. Does the covariance retraction realize a requested first-order vertex
   direction while accepting only a fresh positive-Tutte decoder output?
4. Under a genuinely equal experimental budget, do canonical MVC coordinates
   or covariance retraction improve single-instance map and image fitting over
   raw positive Tutte coordinates?

Questions 1--3 have local CPU and two-host GPU evidence.  For Question 4, the
corrected clean-CPU \(256\times256\) protocol has completed: all four methods
complete the supervised-map cohort, while baseline O4 has a genuine step-18
failure in the image cohort.  This is an incomplete comparison, not a missing
datum to impute.  The final clean all-method remote CUDA rerun under the
corrected residual/condition contract is complete; the independent final
review remains pending.  No Route-II pass is asserted here.

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

For a control-side value \(N\geq3\), the source is the unit square with
\(N^2\) vertices and \(2(N-1)^2\) consistently oriented triangles.  Its
interior and boundary vertex sets are \(I\) and \(B\).  Each interior vertex
\(i\) has a fixed cyclic neighbor list \(N(i)\) of degree \(d_i\).  Supported
row logits \(\ell_{ij}\) define

\[
 p_{ij}=\frac{\exp \ell_{ij}}
 {\sum_{k\in N(i)}\exp\ell_{ik}},
 \qquad j\in N(i).
\tag{2.1}
\]

Padding slots are structurally absent, not zero-weight neighbors.  Given
ordered boundary positions \(b\), the directed logit decoder \(D_\ell\) solves

\[
 Y_i=\sum_{j\in N(i)}p_{ij}Y_j,\quad i\in I,
 \qquad Y_B=b.
\tag{2.2}
\]

The CPU direct backend assembles \(A=I-P_{II}\), factors it once per sample,
and solves both coordinate columns.  Its first-order reverse pass solves with
\(A^{\mathsf T}\).  The GPU backend applies \(A\) and \(A^{\mathsf T}\)
matrix-free and uses implicit BiCGStab differentiation.  In the layer matrix
of Sections 3--6 its relative residual tolerance is \(10^{-5}\) in float32
and \(10^{-11}\) in float64, with zero absolute tolerance and at most 2,000
primary iterations.  The formal instance cohort in Section 8 instead uses
\(10^{-5}\) and \(10^{-10}\), respectively, with at most 500 primary
iterations.  A completed iterative call has also passed the solver's freshly
recomputed represented-residual check.

### 2.2 MVC encoder

For an accepted control map \(Y\), let

\[
 e_{ij}=Y_j-Y_i,
 \qquad r_{ij}=\lVert e_{ij}\rVert_2,
\]

and let \(\alpha_{ij}\in(0,\pi)\) be the consistently oriented angle from
\(e_{ij}\) to the next one-ring edge.  The unnormalized planar mean-value
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

together with the unchanged boundary.  It rejects under-resolved one-ring
edges, inconsistent winding, consecutive rays that are not strictly
counterclockwise and separated from both zero and \(\pi\), nonpositive or
nonfinite weights/row denominators, and a rank-deficient local covariance.
The barycentric residual is recorded as a diagnostic but is not
threshold-rejected by the encoder.  All geometry in (2.3)--(2.4) is computed
with Torch operations; the combinatorial cyclic order is precomputed once
from the mesh.

### 2.3 M1 canonicalization layer

The complete M1 layer is

\[
 (\ell,b)\xmapsto{D_\ell}Y
 \xmapsto{E_\ell}(\ell_c,Y_B).
\tag{2.5}
\]

Its output includes the decoded control map \(Y\), canonical logits
\(\ell_c\), probabilities, and geometry diagnostics.  When all hypotheses and
the linear solve hold exactly,

\[
 D_\ell(\ell_c,Y_B)=Y.
\tag{2.6}
\]

The numerical canonical round-trip error reported below is an independent
second decode of the M1 output; it is not the already available \(Y\) copied
from the first decode.

### 2.4 M2 covariance-retraction layer

Let \(d\in\mathbb R^{|V|\times2}\) be a requested vertex direction with zero
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

At a decoded equilibrium, \(\bar Y_i=Y_i\).  Only at such an equilibrium does
the centered formula align with the MVC/decoder derivative.  If an arbitrary
off-equilibrium array is supplied, centering at \(\bar Y_i\) still solves a
local algebra problem but is not a right inverse at that arbitrary base point.

Equation (2.8) is the unique minimum-
\(\sum_jp_{ij}\delta\ell_{ij}^2\) row lift under the differentiated
barycentric constraint.  It is not generally the derivative of the MVC
formula.  The accepted finite update is

\[
 R_Y(\alpha d)=D_\ell(\ell_c+\alpha\delta\ell,Y_B).
\tag{2.9}
\]

The Euler predictor \(Y+\alpha d\) is never returned as the accepted map.
M2 performs an initial decode, MVC canonicalization, covariance lift, and a
fresh final decode.  Thus its timing must not be interpreted as one bare
linear solve.

### 2.5 Implicit reverse pass

Let \(g_I\) and \(g_B\) be loss cotangents for the decoder's interior and
boundary outputs.  With \(A=I-P_{II}\), the decoder VJP solves

\[
 A^{\mathsf T}\Lambda=g_I
\tag{2.10}
\]

and returns

\[
 \frac{\partial\mathcal L}{\partial\ell_{ij}}
 =p_{ij}\Lambda_i^{\mathsf T}(Y_j-Y_i),
 \qquad
 \frac{\partial\mathcal L}{\partial b}
 =g_B+P_{IB}^{\mathsf T}\Lambda.
\tag{2.11}
\]

For M1, reverse mode first applies the local MVC-encoder VJP and then (2.11)
at the initial decoder.  For M2 it applies (2.11) at the final decoder, then
the local VJPs of the logit addition, covariance lift, and MVC encoder, and
finally (2.11) at the initial decoder.  A moving boundary \(b(q)\) receives
\(J_b(q)^{\mathsf T}\partial\mathcal L/\partial b\); a fixed boundary drops
that input gradient.  Section 7.2 of the optimization chapter derives both the
positive logit sign and these chain rules.

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
\(0.12\mathcal N(0,1)\).  The boundary is the identity unit-square boundary.
For source vertex \((x,y)\), the requested direction is

\[
 d(x,y)=0.02\left(
 \sin(\pi x)\sin(\pi y),
 0.7\sin(2\pi x)\sin(\pi y)
 \right),
\tag{3.2}
\]

scaled by \(1+0.1k\) for batch item \(k\), and set exactly to zero on the
boundary.  The finite M2 step is \(\alpha=0.05\).  Every formal row has one
warmup and two measured repeats.

### 3.2 Measured stages

The receipt measures three distinct autograd graphs.

- `encoder_only`: \(E_\ell(Y)\) for a detached accepted \(Y\).
- `full_m1`: \(E_\ell(D_\ell(\ell,b))\).
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

Let \(Y_c=D_\ell(E_\ell(Y))\).  The canonical round-trip metric is

\[
 e_{\rm round}=\max_i\lVert (Y_c)_i-Y_i\rVert_2.
\tag{3.3}
\]

For a central step \(h\), the finite-difference first-variation error is

\[
 e_{\rm FD}=
 \frac{
 \left\lVert [R_Y(hd)-R_Y(-hd)]/(2h)-d\right\rVert_2
 }{\lVert d\rVert_2}.
\tag{3.4}
\]

The default is \(h=2\times10^{-2}\) in float32 and
\(2\times10^{-4}\) in float64.  These values are disclosed because iterative
solve noise is amplified by division by \(h\).  The projected reverse-mode
identity chooses a deterministic cotangent \(g\) and reports

\[
 e_{\rm proj}=
 \frac{
 \left|\partial_\alpha\langle g,R_Y(\alpha d)\rangle|_{\alpha=0}
 -\langle g,d\rangle\right|
 }{\max(|\langle g,d\rangle|,{\tt tiny})}.
\tag{3.5}
\]

This is one dual projection, not a complete Jacobian comparison.  Independent
CPU tests additionally compare \(dE_\ell\) and the covariance lift and
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
- [Turing A40-0](raw_results/route2_mvc_layer_turing_a40_gpu0_eeaed32.json); and
- [bounded tolerance/memory diagnostic](raw_results/route2_mvc_layer_rtol_sweep_ai_gpu1_turing_memory_20260922.json).

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

| Device/backend | dtype | \((N,B)\) | encoder only | full M1 | canonical M2 |
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

| Host/build | dtype | \((N,B)\) | M2 baseline allocated MiB | M2 peak allocated MiB |
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
A6000 control-only observation remains below 40 MiB, but it excludes the dense
image grid, images, optimizer parameters, activations, and any CNN.  It
therefore supports no inference about million-pixel training memory by itself.

A warmup-free, fresh-process Turing repetition at \(N=25,B=4\), float64
started with only 484,352 allocated bytes and 6,291,456 reserved bytes before
M2, but reached 1,150,018,048 allocated bytes.  This is 99.26% of the original
Turing peak and 29.206 times the corresponding formal AI peak.  The encoder and
full-M1 peaks in that fresh process remained below 4.6 MiB.  Thus accumulated
warmup cache and a cross-stage live-tensor baseline do not explain the large
M2 allocation.  Device architecture, Torch version, CUDA version, and runtime
kernel/workspace selection remain confounded, so the evidence still does not
support a single-factor attribution.

## 6. Numerical accuracy and topology

### 6.1 Maximum error over the three profiles

| Backend/device | dtype | max round-trip \(e_{\rm round}\) | max \(e_{\rm FD}\) | max \(e_{\rm proj}\) | max M2 residual \(\rho\) | min M2 area ratio |
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

For \(N=49,B=1\) on A6000, the individual float32 values are

\[
 e_{\rm round}=4.066\times10^{-4},\qquad
 e_{\rm FD}=1.999\times10^{-1},\qquad
 e_{\rm proj}=3.377\times10^{-1}.
\tag{6.1}
\]

The maximum represented M2 residual is only \(2.731\times10^{-6}\) in that
row, but the displacement used by the central difference is itself small:
\(h\lVert d\rVert\) has the \(h=0.02\) factor from Section 3.3.  Solver error is
therefore divided by both \(h\) and the direction scale in (3.4).  A residual
acceptable for rendering a map can be inadequate for a derivative test.

The cross-host near agreement of round-trip and finite-difference errors shows
that this is reproducible under the two tested builds.  It does not prove that
the residual tolerance is the only cause.  A bounded explicit-tolerance sweep
on idle physical AI GPU 1 used the same \(N=49,B=1\), float32 inputs with no
warmup and one repeat:

| solver rtol | round-trip | FD relative | projected-JVP relative | M2 represented residual | M2 end-to-end seconds |
|---:|---:|---:|---:|---:|---:|
| \(3\times10^{-6}\) | 1.941e-4 | 7.102e-2 | 8.925e-2 | 9.799e-7 | 11.886 |
| \(1\times10^{-6}\) | 5.825e-5 | 1.362e-3 | 3.958e-2 | 2.923e-7 | 23.999 |
| \(3\times10^{-7}\) | 1.378e-5 | 1.338e-3 | 7.894e-3 | 1.280e-7 | 19.292 |

All three rows completed with certified topology.  Some solves at
\(10^{-6}\) and \(3\times10^{-7}\) used the bounded stationary fallback; the
largest reported per-row iteration count was 4,078 and 4,771 respectively.
These are cold single observations, so their nonmonotone forward/backward
times are not a rate theorem or a warmed comparison with the formal baseline.
No accuracy threshold was registered, and the sweep stopped at its three
declared tolerances rather than tuning until a desired label appeared.

The improvement with tolerance supports solver error as an important cause,
but it also exposes a practical floor/cost: FD error changes little between
\(10^{-6}\) and \(3\times10^{-7}\), projected-JVP error remains
\(7.89\times10^{-3}\), and the cold M2 call takes 19.3 seconds.  The supported
statement is:

\[
 \boxed{\text{The current float32 directed GPU path is topology-safe here,
 but not accurate enough for the tested N=49 M1/M2 gradients.}}
\tag{6.2}
\]

Float64 is much more accurate in the same implementation, but its M2 mean time
is 2.87 seconds on A6000 and 3.26 seconds on A40 for \(N=49,B=1\), before any
dense 256-by-256 image loss or CNN.  It is therefore a correctness option, not
yet the desired fast neural-layer solution.

## 7. Independent M1/M2 correctness evidence

The current independently rerun Route-II regression selection reports 102
passes and two CUDA-unavailable skips.  This count belongs to that explicit
selection; it is not interchangeable with older aggregates produced from
different test sets.  Numerical claims below are tied to their named receipts
rather than inferred from a test-count total.

The independent latent-lift comparison uses a variable-degree disk.  It
computes the autograd JVP \(dE_\ell(Y)[d]\) and the covariance lift
\(L_Yd\).  After removing the row-softmax shift gauge, the two probability
tangents differ, while finite-difference decoder evaluations of both realize
the same requested vertex tangent.  Their difference decodes to zero to first
order.  This directly supports

\[
 J_{D_\ell}dE_\ell(Y)[d]=J_{D_\ell}L_Yd=d,
 \qquad
 J_{D_\ell}(dE_\ell-L_Y)d=0,
\tag{7.1}
\]

without asserting equality of the two latent vectors.

### 7.1 Explicit Moore--Penrose closure for plan section 22.3

The earlier two-lift test did not compute the Euclidean Moore--Penrose lift in
a declared common gauge.  The new diagnostic
`experiments/phase5/route2_mvc_pseudoinverse_comparison.py` closes that narrower
gap on the same variable-degree disk.  It deletes padded slots, constructs an
orthonormal Helmert basis for supported arithmetic-zero-row-mean logits, holds
the boundary fixed, and differentiates every one of the \(2|V|=12\) decoder
outputs.  The resulting Jacobian has shape \(12\times6\); all boundary rows are
exactly zero, its measured rank is four under the recorded
\(6.972\times10^{-14}\) SVD threshold, and its common-gauge kernel dimension is
two.  Independent central differences of all six columns give relative
Frobenius error \(6.925\times10^{-11}\).

| Lift | \(\|z\|_2\) | \((\sum p_{ij}z_{ij}^2)^{1/2}\) | analytic tangent residual | central-FD tangent relative residual |
|---|---:|---:|---:|---:|
| \(QJ_{\mathcal G}^{+}d\) | 0.2435816 | 0.1235130 | \(2.05\times10^{-17}\) | \(3.79\times10^{-10}\) |
| \(dE_\ell(Y)[d]\) | 0.2601148 | 0.1281226 | \(1.77\times10^{-17}\) | \(2.25\times10^{-10}\) |
| arithmetic-recentered \(L_Yd\) | 0.2437569 | 0.1234326 | \(1.90\times10^{-17}\) | \(4.36\times10^{-10}\) |

The three pairwise Euclidean differences are \(0.0912564\), \(0.00924179\),
and \(0.0827460\), so the agreement is not caused by nearly identical latent
vectors.  Nevertheless, their pairwise Jacobian-image norms are all below
\(3.1\times10^{-17}\).  This is the requested direct numerical demonstration
that the three vectors are different right inverses and that their differences
lie in the decoder kernel for this fixed-boundary case.

The native covariance vector is also retained separately.  It has
probability-weighted row mean below \(3.5\times10^{-18}\), unweighted row-sum
maximum \(0.0229488\), and weighted norm \(0.1231707\).  Arithmetic recentering
makes it comparable to the other two displayed vectors but raises its weighted
norm to \(0.1234326\).  An explicit weighted pseudoinverse
\(W^{-1/2}(J_sW^{-1/2})^+d\), assembled on all eight supported raw-logit
coordinates with \(W=\operatorname{diag}(p)\), agrees with the native covariance
formula to \(1.30\times10^{-16}\) in Euclidean norm and has decoded residual
\(2.64\times10^{-17}\).  Therefore the receipt does not conflate its native
weighted-minimum property with the Euclidean common-gauge pseudoinverse.

The compact machine-readable evidence is
`raw_results/route2_mvc_pseudoinverse_local.json`; the focused contract suite
is `tests/test_phase5_mvc_pseudoinverse_comparison.py`.  Its CLI test starts a
fresh Windows process after deleting the `MKL_THREADING_LAYER`, `OMP_NUM_THREADS`,
and `MKL_NUM_THREADS` overrides; the NumPy-LAPACK bootstrap prevents the known
duplicate-OpenMP abort without enabling `KMP_DUPLICATE_LIB_OK`.  This is a tiny
float64 differential diagnostic, not a realistic-resolution speed benchmark,
an optimization result, or a Route-II completion verdict.

### 7.2 Realistic-control-mesh MVC round trips

The separate float64 round-trip matrix crosses
\(N\in\{11,25,49\}\), seeds \(1701,1702,1703\), and raw-logit strengths
\(0.15,0.5,1.0\), for 27 cases per host.  Every case performed a fresh MVC
encode and canonical decode and passed the independent topology audit.  The
extrema below are computed over each complete receipt, not selected examples.

| Host receipt | certified / total | max vertex error | max \(\lvert\Delta\mu\rvert\) | max barycentric residual | min MVC probability | max logit spread | max covariance condition | min triangle quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local Windows | 27 / 27 | \(7.02\times10^{-15}\) | \(1.67\times10^{-13}\) | \(2.78\times10^{-17}\) | \(2.05\times10^{-4}\) | 8.04 | 140.35 | 0.0162 |
| remote AI | 27 / 27 | \(9.40\times10^{-15}\) | \(1.24\times10^{-13}\) | \(2.11\times10^{-17}\) | \(2.05\times10^{-4}\) | 8.04 | 140.35 | 0.0162 |

The durable files are
[`route2_mvc_roundtrip_local.json`](raw_results/route2_mvc_roundtrip_local.json)
and
[`route2_mvc_roundtrip_ai.json`](raw_results/route2_mvc_roundtrip_ai.json).
The small round-trip errors verify the stated encoder/decoder identity on this
bounded matrix.  The minimum probability and triangle-quality extrema also
show that the matrix contains difficult stars; they do not establish a uniform
conditioning theorem outside these 54 observations.

## 8. O1--O4 instance protocol and results

### 8.1 Compared methods

All methods use the same positive-Tutte target, exact target boundary, uniform
supported-row initialization, decoder implementation, dense query table, loss,
dtype, device, and declared optimizer hyperparameters.

- O1 represents positive entries as
  \(a_{ij}=\operatorname{sigmoid}(q_{ij})\) and supplies \(\log a_{ij}\) to
  the row-softmax decoder.
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

One multi-right-hand-side primal or transpose call counts once.  O1 and O2
therefore use one initialization solve plus one primal and one adjoint call per
update; O3 uses two initialization calls plus two primal and one adjoint call
per update; O4 uses two initialization calls plus one final primal call per
update.  Target generation and independent audit re-decodes are recorded but
excluded from the optimization budget.  Attempted, completed, failed, primal,
adjoint, and local-backward counts are distinct; iterative right-hand-side
iterations and recomputed residuals are additional work/accuracy measures.
`primal_solves`, `adjoint_solves`, and `global_solves` count completed,
successfully returned decoder/adjoint calls.  If the linear algebra finishes
but the returned-face screen then raises, the call increments its attempt count
but not its completed-solve count.  Thus the exact-83 slice is correctly
defined, but `global_solves` alone understates failed-attempt work; attempts and
wall time must be read with it.

The common learning rate is a **shared-rate robustness test**.  O1--O4 have
different coordinate units and update maps, so it is not a
parameterization-fair speed comparison and cannot establish which method is
best after equal tuning effort.

### 8.3 Correction audit of the rejected runner

The first runner was rejected because it called equal outer steps a fair
solver budget, admitted a control/dense-inconsistent external target, counted
some nonexistent failure-path solves, trusted a hard-coded provenance flag,
omitted an O4 encoder timing, omitted dense interpolation/warp/loss timing,
used ambiguous failure/covariance labels, and dropped iterative diagnostics.
The repaired runner uses the exact solve ledger above, internally generates
and reuses one target, independently re-decodes returned states, times every
declared component, and preserves attributable failures.  The rejected smoke
numbers remain excluded.

### 8.4 Formal instance, losses, metrics, and timing semantics

The clean CPU instance uses \(N=25\) and \(R=256\): 625 control vertices,
529 interior rows, 96 boundary vertices, 1,152 control faces, and 65,536 dense
query points.  It uses float64, one CPU thread, seed 20260922, target strength
0.25, fixed exact target boundary, uniform supported probabilities, 81 maximum
updates, and Adam with \((\beta_1,\beta_2,\epsilon)=(0.9,0.999,10^{-8})\).
With independent standard-normal arrays from that seed, target interior and
boundary logits are \(0.25\xi_I\) and \(0.25\xi_B\).  The target rectangle
height is 0.9 for the map task and 1.0 for the image task.  One CPU-float64
positive-Tutte decode generates the target, which is then reused by every
method; optimization starts from uniform supported interior rows while holding
the decoded target boundary fixed.  The selected shared learning rate is
\(10^{-3}\).

Let \(f_Y(x_r)\) be fixed-table P1 interpolation of control vertices \(Y\) at
dense grid point \(x_r\), and let \(f_*\) be the target map.  The supervised
objective is the mean squared component error

\[
 \mathcal L_{\rm map}(Y)
 =\frac{1}{2R^2}\sum_{r=1}^{R^2}
   \lVert f_Y(x_r)-f_*(x_r)\rVert_2^2.
\tag{8.3}
\]

For the image task, a deterministic medical phantom \(m\) is the moving image
and the fixed image is \(I_*(x_r)=m(f_*(x_r))\).  The predicted backward warp
is \(I_Y(x_r)=m(f_Y(x_r))\), evaluated by bilinear `grid_sample` after mapping
unit-square coordinates to \([-1,1]^2\), with border padding and
`align_corners=True`.  Its objective is

\[
 \mathcal L_{\rm image}(Y)
 =\frac1{R^2}\sum_{r=1}^{R^2}|I_Y(x_r)-I_*(x_r)|^2.
\tag{8.4}
\]

The thresholds are \(10^{-4}\) for (8.3) and \(10^{-3}\) for (8.4).  The
reported threshold solve count is the first completed global-solve count whose
accepted state meets the threshold; absence means it was not reached.

Map RMSE and maximum map error are, respectively,

\[
 e_{\rm map,rms}=\left(\frac1{|V|}\sum_i\lVert Y_i-Y_i^*\rVert_2^2\right)^{1/2},
 \qquad e_{\rm map,max}=\max_i\lVert Y_i-Y_i^*\rVert_2.
\tag{8.5}
\]

For source face \(t=(i,j,k)\), the P1 Jacobian is

\[
 J_t=[Y_j-Y_i,,Y_k-Y_i][X_j-X_i,,X_k-X_i]^{-1}.
\tag{8.6}
\]

The minimum area ratio is \(\min_t\det J_t\); the minimum signed target area is
the source signed area times \(\det J_t\).  Zero flips and a positive local
margin are not alone a global certificate, so the independent boundary and
global-injectivity audit is also required.  The boundary gap is the minimum
length of a target boundary edge.

Writing \(J_t=\bigl[\begin{smallmatrix}u_x&u_y\\v_x&v_y\end{smallmatrix}\bigr]\),

\[
 f_z=\tfrac12[(u_x+v_y)+\mathrm i(v_x-u_y)],\qquad
 f_{\bar z}=\tfrac12[(u_x-v_y)+\mathrm i(v_x+u_y)],\qquad
 \mu_t=f_{\bar z}/f_z.
\tag{8.7}
\]

The Beltrami RMSE and maximum error apply the same RMS/max definitions to
\(|\mu_t-\mu_t^*|\) over faces.  The receipt also records supported-logit
spread, MVC canonicality error, maximum canonical local-covariance condition,
and \(\kappa_2(A)\).  A full dense SVD is the condition authority only when
the interior-row count does not exceed the declared limit; it requires
\(O(|I|^2)\) storage and \(O(|I|^3)\) work.  Exceeding that limit fails closed
rather than silently relabelling an estimate as exact.

Per observation, `primal_seconds`, `adjoint_seconds`, `dense_loss_seconds`,
`canonicalization_seconds`, `lift_seconds`, `local_backward_seconds`,
`optimizer_seconds`, and independent `audit_seconds` have disjoint meanings;
`observation_wall_seconds` is the enclosing elapsed time.  CPU RSS is a
snapshot and process HWM is a process-lifetime peak, not a per-method
allocation.  Method-attributable memory would require fresh one-method
processes.

### 8.5 Frozen-rate pilot and bounded recovery

The pilot and recovery use the explicitly smaller
\(N=17,R=128\), 21-update screening instance.  In the original pilot, each of
the shared rates \(0.003,0.01,0.03\) produced
an O4 image failure (at attempted steps 15, 5, and 3); all other image methods
and all map runs completed.  A bounded recovery tested only
\(10^{-4},3\times10^{-4},10^{-3}\).  All methods completed both tasks at each
of those rates.  Under the preregistered exact-23-completed-solve minimax
criterion, the worst objective/initial-objective ratios across methods and
tasks were 0.9662170231, 0.8984545177, and 0.6572357675, respectively, so the
predeclared rule selected \(10^{-3}\).  End-of-pilot ratios are not substituted
for this selection statistic.  This selection demonstrates common-rate
robustness; it does not equalize parameterization scale or tuning opportunity.

### 8.6 Clean-CPU formal results

The supervised-map exact-83-solve slice is complete and every row is topology
certified:

| method | step | objective | map RMSE | \(\mu\) RMSE | min area ratio | \(\kappa_2(A)\) | max covariance condition | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| O1 | 41 | \(7.006\times10^{-5}\) | 0.012081 | 0.144374 | 0.454326 | 137.610 | 3.233 | 14.47 |
| O2 | 41 | \(4.966\times10^{-5}\) | 0.010337 | 0.139274 | 0.468556 | 135.655 | 3.053 | 13.08 |
| O3 | 27 | \(6.762\times10^{-5}\) | 0.011887 | 0.144522 | 0.461943 | 176.451 | 3.167 | 13.22 |
| O4 | 81 | \(4.807\times10^{-7}\) | 0.001182 | 0.025837 | 0.444284 | 176.918 | 4.073 | 25.29 |

At common outer step 40, the objectives and completed global solves are:

| method | objective | solves |
|---|---:|---:|
| O1 | \(7.143\times10^{-5}\) | 81 |
| O2 | \(5.053\times10^{-5}\) | 81 |
| O3 | \(5.297\times10^{-5}\) | 122 |
| O4 | \(1.859\times10^{-5}\) | 42 |

The first map-threshold observations were O1 step 21 / 43 solves / 8.08 s,
O2 step 11 / 23 solves / 3.67 s, O3 step 18 / 56 solves / 8.90 s, and O4
step 21 / 23 solves / 6.69 s.  Hence O4 gives the strongest result on this one
map instance under both recorded slices, but this is neither an optimizer
theorem nor evidence that the shared learning rate is parameterization-fair.

For the image task, O1--O3 have exact-83 rows; O4 does not because it fails at
attempted step 18 after 17 accepted states:

| method | exact-83 step | objective | map RMSE | \(\mu\) RMSE | min area ratio | wall s |
|---|---:|---:|---:|---:|---:|---:|
| O1 | 41 | 0.001813 | 0.016079 | 0.151093 | 0.498401 | 13.48 |
| O2 | 41 | 0.001216 | 0.015393 | 0.149266 | 0.506053 | 12.50 |
| O3 | 27 | 0.001750 | 0.017221 | 0.152211 | 0.507211 | 15.09 |
| O4 | -- | -- | -- | -- | -- | failed before budget |

At common step 40, O1/O2/O3 objectives are 0.001849, 0.001242, and 0.001239;
O4 has no such state.  O2 first reaches the \(10^{-3}\) image threshold at
step 50 / 101 solves / 15.49 s, and O3 at step 50 / 152 solves / 27.17 s; O1
does not reach it.  The O4 failure makes this a negative/incomplete image
cohort, so no all-method image winner is declared.

The clean map process took 119.99 s; HWM rose from 246,693,888 bytes before
setup to 264,499,200 after target creation and 297,058,304 after the cohort.
The image process took 102.76 s; the corresponding values were 243,728,384,
266,657,792, and 303,345,664 bytes.  These process-level peaks must not be
assigned to a single method.  The receipts are
[`route2_mvc_instance_formal_map_clean_cpu_b7169ef.json`](raw_results/route2_mvc_instance_formal_map_clean_cpu_b7169ef.json)
and
[`route2_mvc_instance_formal_image_clean_cpu_b7169ef.json`](raw_results/route2_mvc_instance_formal_image_clean_cpu_b7169ef.json).

### 8.7 Remote CUDA residual contract and clean rerun

The first remote CUDA audit contained a false negative for the otherwise
successful O4 map run: independent re-decode differed from the CPU-float64
authority by \(8.398\times10^{-10}\), while a fixed coordinate absolute
tolerance of \(9.09\times10^{-13}\) ignored the declared \(10^{-10}\) residual
contract and system conditioning.  The residual-contract rerun records
\(\kappa_2(A)=176.918\), recomputed primal residuals, and residual/condition
forward-error bounds.  The accepted GPU state and independent re-decode differ
from authority by \(1.337\times10^{-15}\) and \(8.398\times10^{-10}\), both
inside their respective declared bounds.

The final clean CUDA evidence consists of eight strict JSON receipts from clean
commit `74b452c` (`dirty=false`).  All four map methods complete; image O1--O3
complete; image O4 reproduces the true step-18 failure, and the independent
audit remains valid.  Thus the earlier O4-map label was a contract bug, whereas
the O4-image failure is a reproducible algorithmic observation.  The strict
condition audit computes a full dense SVD only for
\(|I|\leq\texttt{condition_dense_limit}\); above that limit it fails closed,
because the audit itself costs \(O(|I|^2)\) storage and \(O(|I|^3)\) work.
The CPU and CUDA target-summary fields agree to at most
\(2.84\times10^{-14}\), but they are not bitwise identical because reduction
orders differ across numerical libraries; target identity here means the same
seed, latent definition, and decoded problem within that recorded roundoff.

These strict forward/audit outcomes do not repair the separate gradient and
memory negatives from Sections 5--6: the float32 \(N=49\) M2 finite-difference
error is about 0.19987, a tighter-tolerance probe still bottoms out near
\(1.34\times10^{-3}\), and the fresh-process A40 M2 peak is
1,150,018,048 bytes, or about 1.150 GB (1.071 GiB).
Accordingly, the clean CUDA rerun is not evidence of universally accurate GPU
gradients or a low-memory implementation.

### 8.8 O4 failure chain and separate trust/sensitivity negative control

The formal image run exposes a distinction that a binary fold count would
hide.  Baseline O4 at learning rate \(10^{-3}\) keeps every *accepted* state
certified, but after accepted step 17 its objective is 0.00433907, its
minimum P1 area ratio is only 0.00251579, and its maximum canonical local
covariance condition is 181.682.  The full step-18 lifted increment then
causes the retraction decoder to reject a returned map with nonpositive face
areas.  The same
step-18 failure occurred with the float64 matrix-free decoder on an A6000, so
it is not evidence of a CPU-SuperLU accident.  In contrast, the formal map
task at the same rate completes all 81 steps: objective
\(4.8074105\times10^{-7}\), final minimum area ratio 0.444284, and final
canonical covariance condition 4.07306.

To test whether a minimal trust rule could isolate the numerical issue, a
separately named negative-control method was added without changing baseline
O4.  If \(z_t\) is the accepted canonical logit vector and
\(\Delta z_t=L_{Y_t}d_t\) is the covariance lift of the vertex-Adam proposal,
trial \(k\) decodes

\[
 z_t^{(k)}=z_t+\alpha_k\Delta z_t,
 \qquad \alpha_k=2^{-k},\qquad k=0,\ldots,11.
\tag{8.8}
\]

Let \(m(Y)\) denote the minimum exact facewise P1 area ratio.  A trial is
accepted only if the unchanged global-injectivity audit passes and

\[
 m(D(z_t^{(k)},b))\geq \tfrac12 m(Y_t).
\tag{8.9}
\]

There is no vertex repair and no weakened topology tolerance.  Each decoder
call increments the attempt count; every call that returns a decoded map
increments the completed primal/global-solve count even if (8.9) rejects it.
The trace records trial attempts, completed trial solves, extra completed
solves, the accepted \(\alpha_k\), and the required area floor.  Exhausting the
12 trials fails closed with the last accepted map.

| cohort | rate | accepted steps | status | completed extra trial solves | minimum accepted scale | final objective / initial | final minimum area ratio | maximum canonical covariance condition |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| trust, \(N=17/R=128\) image | \(3\times10^{-3}\) | 21/21 | completed | 45 | \(2^{-9}\) | 0.1976 | \(2.706\times10^{-4}\) | \(1.951\times10^3\) |
| trust, \(N=25/R=256\) image | \(10^{-3}\) | 31/81 | failed at step 32 | 69 | \(2^{-11}\) | 0.4158 | \(3.224\times10^{-5}\) | \(1.321\times10^4\) |
| baseline O4 sensitivity | \(3\times10^{-4}\) | 52/81 | failed at step 53 | 0 | -- | 0.4112 | \(3.825\times10^{-6}\) | \(1.506\times10^5\) |
| baseline O4 sensitivity | \(10^{-4}\) | 81/81 | completed | 0 | -- | 0.6693 | 0.3399 | 3.435 |
| baseline O4 sensitivity | \(3\times10^{-5}\) | 81/81 | completed | 0 | -- | 0.8886 | 0.5494 | 3.043 |

The bounded trust experiment is therefore a counterexample to the proposition
that local backtracking alone makes this O4 optimization practically stable.
It postpones a forbidden step, but repeated accepted updates still approach
the boundary of the discrete-homeomorphism set while scales vanish and solve
cost grows.  The observed shrinking scales give no evidence that a larger
trial limit would recover an efficient method; larger limits were not tested.
The per-method sensitivity instead shows that O4's vertex
coordinates require a substantially smaller step scale on this image task:
\(10^{-4}\) completes with a useful area margin, while \(3\times10^{-4}\)
still fails.  Because coordinate units differ, this is a separately labelled
sensitivity result and cannot replace the frozen shared-rate comparison.
It also has no exact-83-solve row: the formal trust trace jumps from 80
completed solves at accepted step 28 to 86 at accepted step 29, ends its last
accepted state at 97, and reaches 102 only through subsequently rejected or
failed trials.  Neither the first state above 83 nor any common outer step is
reported as an exact-83 substitute.  The N=17 trust trace ends at 68 solves
and likewise never reaches that budget.

Machine-readable receipts are
`raw_results/route2_o4_trust_pilot_image_n17_lr0p003_cpu.json`,
`raw_results/route2_o4_trust_formal_image_n25_lr0p001_cpu.json`, the three
`route2_o4_sensitivity_image_*.json` receipts, and
`route2_o4_baseline_map_regression_after_trust_cpu.json`.  The post-change map
regression is numerically identical to the original formal map receipt in
objective, final-state numeric identity, topology, and solve counts; this
directly checks that the new variant did not alter baseline O4.

## 9. Current answer to the project objective

Route II currently supplies a mathematically defined, differentiable M1
canonicalization layer and M2 decoder retraction.  Both return piecewise-affine
maps screened under the positive-Tutte topology conditions, and their local
float64 derivatives have strong independent numerical agreement.  It does
**not** yet supply the requested final fast, accurate, memory-efficient neural
layer:

- current GPU execution is slower than the CPU direct reference for the tested
  bounded profiles;
- float32 \(N=49\) derivative accuracy is poor at the existing directed-solver
  tolerance;
- float64 corrects accuracy but raises time and, on one runtime, allocator
  memory substantially;
- the formal supervised-map instance favors O4, but the formal image cohort is
  incomplete because O4 reproducibly fails at attempted step 18;
- the shared-rate cohort is a robustness test, not a parameterization-fair
  speed ranking; and
- no CNN has been trained through this Route-II layer.

Completed evidence now includes M1/M2 derivative checks, 54 realistic-control
round trips, the common-gauge pseudoinverse diagnostic, two-host incremental
and Woodbury receipts, clean CPU formal map/image runs, and a clean strict CUDA
rerun.  Pending work is adjudication of the O4 image failure in the baseline
comparison, any separately predeclared parameterization-fair tuning study, and
the independent final Route-II review.  These items await independent
adjudication; this chapter therefore makes no Route-II pass claim and does not
classify the route as the requested production neural layer.  The fixed-mesh
decoder nevertheless continues to supply the hard piecewise-affine
homeomorphism mechanism for every accepted state.
