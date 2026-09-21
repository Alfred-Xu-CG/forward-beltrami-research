# Route I independent checker: differentiable Tutte homeomorphism layer

Date: 2026-09-22 (Asia/Shanghai)  
Checker role: independent of the Route I implementation and of the authorship of
the three preceding Route I chapters.

## Executive verdict

**Route verdict: PASS, with a bounded scope.** Route I satisfies the mandatory
closure items in the Phase V plan: the official TutteNet mechanism was audited;
four backend identities were implemented (two CPU direct baselines and two
matrix-free backends spanning directed and symmetric parameter families); the
first-order implicit derivatives were independently checked;
fixed dense queries were precomputed; the prescribed normal, adversarial, and
single-instance matrices were executed; failed cases were retained; and the final
application chapter passed an independent receipt-level review.

This PASS has a precise meaning. For a certified triangulated disk, the prescribed
ordered convex boundary, and strictly positive supported weights, the exact
single-layer Tutte construction is a P1 homeomorphism. The software returns only
solutions that pass its finite-precision solver, boundary, and original-face
screens. It supports first-order reverse-mode differentiation and was exercised at
256-squared, 512-squared, and selected 1024-squared query resolutions.

This PASS does **not** mean that the user's complete research objective has already
been achieved. In particular:

- target-side predicates are numerical screens, not exact predicates over every
  floating-point input;
- numerical availability is not uniform over dtype, conditioning, seed, mesh, or
  hardware;
- a small solver residual is not a map, VJP, or Beltrami-accuracy bound;
- a composition of certified P1 layers is piecewise affine on an induced refined
  partition, not generally P1 on the original control triangulation;
- second derivatives are not supported;
- no image-to-latent CNN was trained; the application evidence is direct
  single-instance latent optimization;
- no exact prescribed-Beltrami reproduction or universal speed advantage was
  established.

Accordingly, this document closes **Route I only** and permits work on Route II. It
does not declare the overall 18-hour Phase V objective complete.

## 1. Review question, evidence, and decision rule

The question is not merely whether the code runs. It is whether the current Route I
evidence supports a fast, memory-aware, differentiable neural-layer primitive whose
accepted output has a defensible discrete homeomorphism claim.

The review uses four evidence levels:

1. exact mathematical statements under explicit hypotheses;
2. code-level statements checked against the implementation;
3. numerical statements reconstructed from preserved receipts;
4. conclusions restricted to the tested population.

The durable primary record is the [official-code audit](01_tuttenet_code_audit.md),
the [solver and topology specification](02_tutte_fast_solver.md), the final-PASS
[application chapter](03_tutte_application.md), the [research worklog](WORKLOG.md),
and the [Phase III/Phase V independent-claim ledger](../research_phase3/07_independent_checks.md).
The checker also used the independent working reviews of the official audit,
topology implementation, direct/iterative/symmetric solvers, dense interpolation,
composition, validity suite, formal-summary validator, directed float64 studies,
symmetric CUDA headroom correction, and final application chapter. Those reviews
found real defects during development; the verdict below is not a restatement of
builder-side green tests.

The decision rule is:

- **PASS** if every mandatory Route I question has direct evidence, every material
  negative result remains visible, and the remaining limitations bound rather than
  contradict the Route I construction;
- **CONDITIONAL** if a mandatory Route I item remains unverified or depends on an
  unresolved correctness defect;
- **FAIL** if the topology, derivative, or measurement claim is contradicted by the
  implementation or evidence.

### 1.1 Fresh closure verification

Immediately before issuing this verdict, the checker ran the complete focused
Route I collection covering the direct, directed-iterative, symmetric, boundary,
dense-warp, composition, reference, topology, legacy-audit, instance, shared-data,
validity, summary, sweep, engineering-harness, and metric tests with
`PYTHONPATH=src`, `MKL_THREADING_LAYER=SEQUENTIAL`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`. The result was **333 passed, 7 skipped in 64.95 s**. All seven
skips are explicitly CUDA-only tests on the local non-CUDA checker host; the CUDA
claims in this document instead depend on the independently parsed remote receipts
and dedicated remote checks described below.

A fresh CommonMark parse found **65 links and zero missing local targets** in the
final application chapter, and **8 links and zero missing local targets** in this
checker chapter. `git diff --check` reports no whitespace defect in this file.

## 2. Mathematical object and exact topology claim

### 2.1 Source disk and represented P1 map

Let \(\mathcal T=(V,E,F)\) be a finite, coherently oriented planar triangulation of
a closed topological disk. Its fixed source vertices are \(x_i\in\mathbb R^2\),
every source face \(t=(i,j,k)\) is counterclockwise, and faces intersect only in
shared simplices. Let \(I\) and \(B\) denote interior and boundary vertices, with
the boundary stored as one cyclic loop.

For target vertices \(Y=(y_i)_{i\in V}\), the represented map is the continuous
P1 interpolant

\[
f_Y(x)=\sum_{a=0}^2\beta_a(x)y_{i_a},
\qquad x\in t=(i_0,i_1,i_2),
\]

where \(\beta_a\) are source barycentric coordinates. On a face,

\[
J_t=[y_j-y_i\;y_k-y_i]
    [x_j-x_i\;x_k-x_i]^{-1}.
\]

Because every source face is positively oriented, positive target signed area is
equivalent to \(\det J_t>0\). This is a local nondegeneracy/orientation condition.
It is not by itself a global injectivity theorem.

### 2.2 Tutte--Floater hypotheses

At every interior vertex the directed family imposes

\[
y_i=\sum_{j\in N(i)}p_{ij}y_j,
\qquad p_{ij}>0,
\qquad \sum_{j\in N(i)}p_{ij}=1,                                      \tag{1}
\]

using all mesh neighbors. The boundary must map homeomorphically and with the
correct orientation onto the boundary of a convex polygonal region.

For a weakly convex target boundary, the additional Floater condition is essential:
no *dividing edge*--a nonboundary mesh edge whose endpoints are both boundary
vertices--may be mapped entirely into the target boundary. Under these hypotheses,
the exact P1 map is injective. A strictly convex target makes the dividing-edge
obstruction automatic; a rectangle with subdivided sides is only weakly convex, so
the condition cannot simply be omitted. The structured-rectangle construction
closes it by preserving side/corner incidence and strict side order. This is the
scope supported by [Floater's one-to-one PL theorem](https://doi.org/10.1090/S0025-5718-02-01466-7).

There is also an a posteriori PL global-inversion route. On a triangulated disk, if
every represented face is nondegenerate and positively oriented and the boundary
map is a simple orientation-preserving homeomorphism, degree one in the bounded
target region implies that the whole P1 map is a homeomorphism. The relevant
boundary-degree statement is described by
[Lipman](https://arxiv.org/abs/1310.0955). This explains why the implementation
checks every original face and the boundary, rather than a sample of dense pixels.

The theorem package therefore separates three facts:

- boundary reachability and positive weights give uniqueness of the linear solve;
- the Tutte--Floater hypotheses give injectivity of the exact harmonic map;
- an independently valid positive-face/simple-boundary P1 map is globally invertible
  even if its numerical coordinates are only an approximate harmonic solution.

The older Phase III note that retained graph three-connectivity as a generic caller
precondition is superseded for this current, explicitly certified triangulated-disk
path by the sharper disk/link/boundary/dividing-edge formulation. It is not a
license to apply the solver to an arbitrary graph.

### 2.3 What is exact in the implementation, and what is numerical

`DirectedTutteSystem.from_mesh` checks the combinatorial disk rather than assuming
it from an Euler count alone: unique and used simplices, coherent edge incidence,
connected face complex, path/cycle vertex links, one boundary cycle, and Euler
characteristic one. For stored binary floating-point source coordinates, the source
orientation and boundary-intersection signs are evaluated after exact integerization
of their dyadic representations. These are exact statements about the represented
source inputs.

Target checks are different. Boundary simplicity/convexity and target-face areas
are evaluated in floating arithmetic. Every current Route I adapter checks the
faces after conversion to the **returned dtype**. The shared screen promotes those
already rounded coordinates to CPU float64 and requires signed double area greater
than

\[
64\epsilon_{64}s^2,
\qquad
s=\max\{\operatorname{range}(Y_x),\operatorname{range}(Y_y),1\}.       \tag{2}
\]

Promotion cannot recover information lost in a float32 output. Equation (2) is a
positive numerical margin, not an exact orientation predicate, a minimum singular
value bound, or a float32 error theorem. The independent global audit likewise uses
stated floating tolerances and winding tests.

The implementation's hard-topology semantics are consequently:

1. invalid source topology is rejected before learning;
2. realized probabilities/conductances and boundary coordinates must be finite and
   strictly admissible in their actual dtype;
3. failed or unconverged linear systems raise rather than fabricate a map;
4. detected boundary or face invalidity prevents return;
5. no projection, untangling, or post-hoc fold repair is used.

“Fail closed” means that a detected failure is not returned. It does not mean that
every near-degenerate floating input is decided by an exact predicate or that every
finite latent is guaranteed to produce a result. The improved direct and directed
matrix-free paths also do not call the generic cached dividing-edge preflight;
their theorem claim is therefore restricted to the prescribed structured rectangle,
whose incidence excludes the obstruction by construction. The legacy/reference and
symmetric paths perform the explicit preflight.

### 2.4 Ordered rectangle boundary and learnable aspect ratio

For side \(s\), the implementation uses positive softmax segment fractions

\[
q_{sr}=\frac{e^{a_{sr}}}{\sum_v e^{a_{sv}}},
\qquad
\tau_{sk}=\sum_{r=1}^kq_{sr},                                       \tag{3}
\]

and maps the four sides counterclockwise to

\[
(\tau,0),\quad(1,H\tau),\quad(1-\tau,H),\quad(0,H(1-\tau)),           \tag{4}
\]

with exact shared corners and

\[
H=H_{\min}+\operatorname{softplus}(m)>0,
\qquad H_{\min}>0.                                                   \tag{5}
\]

Positive real softmax values do not ensure representably distinct cumulative
float coordinates. The boundary module therefore checks the realized side order,
finite height, corners, and boundary geometry. Independent review found and closed
four defects: cumulative-coordinate collapse, stale device-buffer tuples, unstable
inverse softplus at large height, and finite latent values producing nonfinite
realized height. Composition bypasses inverse-softplus round trips and realizes the
represented unit height directly.

Calling \(H\) a learnable rectangle modulus is an explicit parameter convention. It
is not an extremal-length recovery theorem or a Beltrami modulus identity.

## 3. Official TutteNet understanding

The [official audit](01_tuttenet_code_audit.md) inspected pinned revisions
`GitBoSun/TutteNet@cb9f919` and
`flaport/torch_sparse_solve@89c227c`. It is a source audit, not a reproduced
training run or reproduced paper benchmark.

The official mesh is center-split: an \(N\)-by-\(N\) vertex grid plus one center
vertex in every cell, with four triangles per cell. Thus

\[
|V|=N^2+(N-1)^2,\quad |B|=4(N-1),\quad |F|=4(N-1)^2.
\]

At official \(N=11\), this is 221 vertices, 181 interior unknowns, 40 boundary
vertices, 400 faces, 620 undirected edges, 1,240 directed stored edge slots, and
1,117 structural nonzeros in the reduced system. The local structured rectangle at
the same label \(N=11\) instead has 121 vertices, 81 interior rows, and 200 faces.
Equal resolution labels therefore do not define equal systems.

The inspected official equation is

\[
L_{II}X_I=W_{IB}C,
\qquad L=D-W,
\]

or after row scaling,

\[
(I-P_{II})X_I=P_{IB}C,
\qquad P=D^{-1}W.                                                     \tag{6}
\]

The learning and standalone fitting paths are materially different. The learning
encoder's midpoint features tie reverse-edge predictions before row scaling; the
fitting script stores independent directed slots. Symmetric \(W\) still does not
make \(D^{-1}W\) symmetric in general. Their sigmoid scales and boundary increment
transforms also differ.

The official backend is a sequential CPU float64 KLU path. Within one call the x/y
columns share a factorization, so it is wrong to count them as two independent
factorizations. The factor is then freed. Backward invokes the complete solver on
the transposed sparse matrix, repeating conversion, symbolic analysis, and numeric
factorization. The learning head moves quantities from CUDA to CPU double for the
solve and returns CUDA float32 coordinates; this is neither a GPU sparse solve nor
an end-to-end float64 layer.

The active learning model has 24 planar sublayers. The standalone fitting model has
eight triplane blocks with three sequential planar maps each, also 24. One ordinary
sample traversal therefore means 24 two-RHS forward systems; differentiating all
of them means 24 additional transpose systems. These are source-derived counts,
not measured timing proportions.

Dense point location is path-dependent. The learning path performs detached CPU
SciPy search each forward; the fitting forward uses structured cell arithmetic,
while its inverse uses SciPy. Later composition queries change after every layer,
so even fixed source geometry does not justify precomputing all-layer locations.
No official historical dependency build, extension compilation, checkpoint,
training run, or paper timing was reproduced. The local `reference` backend must
therefore be called a legacy correctness baseline, not official KLU replication.

## 4. Route I operator families and solver fairness

### 4.1 Directed row-softmax family

Each interior row has supported probabilities

\[
p_{ij}=\frac{e^{z_{ij}}}{\sum_{k\in N(i)}e^{z_{ik}}}.
\]

Let \(Q=P_{II}\), \(R=P_{IB}\), \(C\) be the ordered boundary coordinates, and
\(X\) the interior coordinates. Then

\[
A=I-Q,\qquad AX=RC.                                                   \tag{7}
\]

With positive full-neighbor support and boundary reachability, \(Q\) is
substochastic and \(\rho(Q)<1\); hence

\[
A^{-1}=\sum_{k=0}^{\infty}Q^k\geq0.                                  \tag{8}
\]

This proves existence and uniqueness in exact arithmetic. It provides no uniform
condition-number or iteration bound over all finite logits.

The directed family is comparatively unconstrained but generally nonsymmetric.
Its matrix-free implementation applies

\[
(Av)_i=v_i-\sum_{j\in I}p_{ij}v_j
\]

by gather/reduction and applies \(A^T\) by incoming scatter-add. Confusing those
orientations would give a wrong VJP. BiCGStab solves each batch/RHS column with
independent activity, explicit represented residual checks, and scale guards.
Only CUDA float32 may attempt the stationary fallback

\[
x^{k+1}=Qx^k+b.                                                       \tag{9}
\]

The spectral-radius argument justifies (9) in exact arithmetic, but not a uniform
float32 rate or the heuristic iteration budget.

### 4.2 Symmetric conductance family

The symmetric family assigns one scalar to every undirected edge incident to an
interior vertex,

\[
c_e=c_{\min}+\operatorname{softplus}(z_e)>0,
\]

and solves

\[
KX=WC.                                                               \tag{10}
\]

For an interior vector \(v\),

\[
v^TKv=
\sum_{\{i,j\}\in E_{II}}c_{ij}(v_i-v_j)^2+
\sum_{\{i,b\}\in E_{IB}}c_{ib}v_i^2>0,
\]

where strict positivity uses the certified connected disk property: every interior
component reaches the Dirichlet boundary. Hence \(K\) is SPD and the same operator
is used in primal and adjoint CG. A global positive conductance scale is a gauge;
the implementation normalizes by the sample maximum inside autograd.

Row normalization produces

\[
d_ip_{ij}=d_jp_{ji},                                                  \tag{11}
\]

so symmetric conductances form a reversible operator/parameter subfamily of
independent directed rows. Equation (11) does not prove a strict separation of
attainable vertex maps, because different operators can produce the same harmonic
coordinates. Conversely, a directed target is not assumed to be symmetrically
representable.

The final CUDA-float32 CG implementation uses a stricter internal threshold

\[
\tau_{\mathrm{int}}=0.99\tau_{\mathrm{req}}
\]

for recurrence activity and reliable-update candidates, while a fresh represented
residual must still satisfy the unchanged public \(\tau_{\mathrm{req}}\) before
return. CPU and float64 use equality. This is numerical headroom for an observed
threshold-boundary effect, not a relaxed contract or a convergence theorem.

### 4.3 Reference, direct, and matrix-free identities

| Backend | Actual algorithm and factor lifetime | Device/dtype scope | What it is evidence for |
|---|---|---|---|
| `reference` | Legacy Phase III path; factors \(A\) and \(A^T\) during forward and retains the transpose factor for backward | Sequential CPU; NumPy/SciPy float64 internals | Correctness/cost baseline, not official KLU |
| `direct` | One `splu(A)` per nonempty sample; x/y solved together; same numeric factor used with `trans='T'` in backward | Sequential CPU; float64 sparse solve, public float32/64 | Benefit of retaining one factor within one autograd graph |
| `directed_iterative` | Matrix-free gather/scatter BiCGStab; CUDA-float32-only stationary fallback | CPU or CUDA, float32/64 with explicit settings | Directed no-assembly implicit layer |
| `symmetric` | Matrix-free edge-scatter unpreconditioned CG; same SPD operator in backward | CPU or CUDA, float32/64 with explicit settings | Reversible SPD subfamily |

The reference/direct comparison holds the logical directed workload close, but even
there float32 probability realization differs: the legacy path normalizes in NumPy
float64 whereas direct uses the public Torch dtype before float64 assembly. Neither
backend reuses a factor across changed parameters or separate forwards.

The symmetric and directed paths are not solver-only substitutions for one matrix;
they parameterize different operator families. CPU and GPU results also use
different hosts, software, arithmetic, synchronization, and validation transfers.
Therefore the application tables are workload evidence, not hardware-only speed
rankings.

A logical global solve means one sample's two-coordinate system. Factorizations,
triangular solves, Krylov iterations, and fallback attempts are separate counters.
Broadcasting a solved singleton layer across a later batch does not create duplicate
linear systems, and a boundary-only map has no interior system.

## 5. Implicit differentiation: formulas, signs, and evidence

### 5.1 Directed VJP

Let a scalar loss supply interior and boundary cotangents \(G_I,G_B\). From (7),

\[
dA=-dQ,
\qquad
A\,dX=dQ\,X+dR\,C+R\,dC.                                            \tag{12}
\]

Solve the two-column adjoint

\[
A^T\Lambda=G_I.                                                       \tag{13}
\]

Then

\[
\boxed{
\overline C=G_B+R^T\Lambda,
\qquad
\overline p_{ij}=\Lambda_i\mathbin{\cdot}y_j.}                       \tag{14}
\]

The probability sign is positive because \(dA=-dQ\). Using a primal solve in
place of (13), omitting the explicit \(G_B\), or scattering outgoing rather than
incoming transpose contributions would be wrong.

The row-softmax chain is

\[
\boxed{
\overline z_{ij}=p_{ij}\left(
\overline p_{ij}-\sum_kp_{ik}\overline p_{ik}\right)
=p_{ij}\Lambda_i\mathbin{\cdot}
\left(y_j-\sum_kp_{ik}y_k\right).}                                   \tag{15}
\]

Only for an exact harmonic solution may the row mean be replaced by \(X_i\).
For an inexact primal, that replacement injects primal residual error into the
formula. The improved direct and iterative implementations retain the mean form.

### 5.2 Symmetric VJP

For (10), solve

\[
K\Lambda=G_I.
\]

For an interior--interior edge and an interior--boundary edge respectively,

\[
\boxed{
\overline c_{ij}=-(\Lambda_i-\Lambda_j)\mathbin{\cdot}(X_i-X_j),}     \tag{16}
\]

\[
\boxed{
\overline c_{ib}=\Lambda_i\mathbin{\cdot}(C_b-X_i),
\qquad
\overline C_b=(G_B)_b+\sum_i c_{ib}\Lambda_i.}                       \tag{17}
\]

There is no factor of two in (16), because each undirected edge is parameterized
once. The outer chain includes both `softplus` and the differentiable
\(\widehat c=c/\max c\) normalization; detaching that normalization would lose a
real derivative term.

### 5.3 Independent derivative evidence and limits

The checker evidence is not just self-consistency of one custom backward:

- the direct backend agrees with an independently assembled variable-degree dense
  solution within \(4.45\times10^{-16}\), passes nonsymmetric gradcheck, and passes
  joint logit/boundary directional finite differences;
- the directed matrix-free path agrees with a separate dense full-vertex solve on
  an irregular nonsymmetric disk, with coordinate/VJP discrepancies at roundoff and
  a joint directional finite-difference error \(1.67\times10^{-10}\);
- the symmetric path agrees with an independent dense incidence solve to at most
  \(4.45\times10^{-16}\) in the reviewed double-precision comparison, with joint
  directional finite-difference error \(7.22\times10^{-10}\);
- heterogeneous direct/symmetric composition agrees with an independent face-search
  oracle to \(2.23\times10^{-16}\), with joint directional finite-difference error
  \(5.50\times10^{-9}\);
- the N=49 float64 CPU-direct/CUDA-directed comparison has maximum dense-coordinate
  component difference \(3.45\times10^{-9}\), row-logit VJP relative difference
  \(4.54\times10^{-9}\), and boundary-logit VJP relative difference
  \(1.46\times10^{-9}\) over three seeds.

Independent review also found and closed failures that ordinary gradcheck did not
initially expose: saved-primal alias corruption after in-place output mutation,
tiny-residual norm underflow in BiCGStab, large-RHS norm overflow in CG, and an
overflowed fallback norm that returned an unconverged zero solution. Their
regressions now fail closed or compute the intended VJP.

These are first-order reverse-mode claims. The custom backward is not a supported
second-order differentiable operation. Dynamic triangle location is differentiable
almost everywhere but has branch-selected derivatives on mesh edges. The implicit
VJP is the derivative of the mathematical linear solution; with inexact primal and
adjoint solves it approximates that VJP, not the derivative of the Krylov stopping
branch itself.

## 6. Dense queries and composition

For each fixed source query \(q\), `StructuredDenseQueryTable` stores three vertex
indices and barycentric weights:

\[
F(q)=\sum_{a=0}^2\beta_{qa}Y_{v_{qa}}.                                \tag{18}
\]

After explicit device/dtype preparation, an interpolation call performs gather,
multiply, reduce, and reshape. It performs no point location, sparse solve,
factorization, host transfer, or dense \(Q\)-by-\(|V|\) matrix multiplication.
At 512-squared queries, the independent affine check has maximum error
\(6.66\times10^{-16}\); 104 independently located edge/corner queries agree to
\(1.30\times10^{-15}\); gradcheck passes.

For layers \(f_1,\ldots,f_L\), coordinate composition is

\[
q_0=q,
\qquad q_\ell=f_\ell(q_{\ell-1}),
\qquad F=f_L\circ\cdots\circ f_1.                                   \tag{19}
\]

Only the first layer can use the fixed query table. Later points \(q_{\ell-1}\)
depend on earlier learned maps and require dynamic structured cell/triangle
location. Within a selected triangle,

\[
dq_\ell=\sum_a\beta_a(q_{\ell-1})\,dY_{\ell,a}
          +J_{\ell,t}\,dq_{\ell-1}.                                 \tag{20}
\]

Composition acts on coordinates, not on repeatedly resampled images. If every
domain-wide layer is a square homeomorphism, their composition is a homeomorphism.
As a composition of PL maps it is PL on a suitable common refinement involving
preimages of later triangle edges. It is **not generally P1 on the original mesh**.
The implementation does not claim that dense samples, a bilinear image grid, or a
new original-mesh interpolation inherit the per-layer certificate.

This distinction resolves the user's output requirement:

- a returned single control layer is explicitly a P1 map on its declared source
  triangulation;
- a retained sequence of accepted layers defines a domain-wide PL homeomorphism;
- a tensor of composed dense samples alone is not a certified P1 mesh output;
- producing one explicit refined triangulation for the composition, or certifying
  an original-mesh resample, remains additional work.

## 7. Forward/backward cost and memory

### 7.1 Structural accounting

For bounded mesh degree, a directed matvec costs
\(O(Bn_IDr)\) and a symmetric edge matvec costs
\(O(B|E_{\mathrm{active}}|r)\), multiplied by the realized iteration count. The
matrix-free custom functions save probabilities or conductances, boundary values,
and one primal solution, not all Krylov iterates. Their solver tape is therefore
constant in the iteration budget, but not constant in mesh, batch, layers, RHS, or
queries.

Fixed dense interpolation costs \(O(BQ)\) and stores an immutable three-entry
table of size \(O(Q)\). A composition retains solver state plus ordinary query
activations for each layer, approximately \(O(LBn+LBQ)\) at fixed degree. Dense
coordinates can dominate memory even when the Krylov tape is iteration-independent.

The complete CUDA layer is not zero-transfer. Boundary and returned-face checks
copy geometry to CPU, scalar convergence decisions can synchronize the device, and
scatter-add may be nondeterministic. These costs remain in the measured workload.

### 7.2 Measured normal-scale cost

The 90 matched CPU configurations give median paired
\(T_{\mathrm{reference}}/T_{\mathrm{direct}}=1.1274\), range
\([0.9672,1.2350]\), with direct faster in 86/90. This supports a modest bounded
factor-lifetime benefit, not a twofold end-to-end speedup.

The clean committed symmetric A6000 matrix includes:

- \(N=49,B=4,L=4,R=256\): mean 8.077845 s;
- \(N=49,B=8,L=4,R=256\): mean 15.413787 s;
- \(N=49,B=8,L=4,R=512\): mean 16.088929 s, 780.903809 MiB peak allocated,
  1054 MiB peak reserved, and 1156.339844 MiB process HWM.

At 1024-squared queries, four selected symmetric \(B=1\) coordinate workloads all
complete. The largest listed layer case, \(N=49,L=4\), takes 2.784996 s with
420.614746 MiB allocated and 508 MiB reserved. These are coordinate surrogate
forward/loss/backward measurements, not image-optimization timings.

Explicit-float64 directed N=49 normal cases complete, but the B=8/L=4/R=256 seed
times span 18.284--20.762 s, and the three identical-input CPU-direct/GPU-double
comparisons are slower on GPU double. Float64 is a bounded reliability option, not
a demonstrated speed winner. One B=8/L=4/R=512 receipt takes 18.214490 s and
allocates 1212.616699 MiB.

Timing scopes are not interchangeable. Engineering time covers boundary creation,
control solves and their internal screens, coordinate composition, the surrogate
loss, and backward; it excludes setup, warmup, the later independent audit, and
dense-only replay. Formal optimization wall time includes actual repeated decoder
calls but is single-run and task-dependent. Torch allocated/reserved peaks are not
whole-device memory, and process HWM includes libraries and earlier allocations.

## 8. Experimental evidence and preserved negatives

### 8.1 Ground-truth legality is validity, not accuracy

The clean legality suite contains exactly 93 cases:

| Population | Accepted | Meaning |
|---|---:|---|
| Eight analytic maps at five control sizes | 40/40 | sampled P1 targets pass the numerical topology audit |
| Three strengths, three seeds, five sizes of positive directed targets | 45/45 | generated ground truth is legal in CPU float64 |
| Four image families at 256/512 | 8/8 | sampling convention and independent bilinear oracle pass tolerance |

This is not 93 fits, 93 production solver trials, or proof of a uniform margin.
The random targets include minimum signed area \(4.95\times10^{-13}\), minimum
area ratio \(8.31\times10^{-10}\), and maximum sampled \(|\mu|=0.999999939\).
The image oracle tolerance is \(8\epsilon_{32}R\), not exact pixel equality.

### 8.2 Full normal engineering matrix

The original normal product has

\[
N\in\{11,17,25,33,49\},\quad
B\in\{1,4,8\},\quad
L\in\{1,2,4\},\quad
R\in\{256,512\},
\]

or 90 configurations per backend.

| Backend | Original complete/requested | Material finding |
|---|---:|---|
| CPU `reference` | 90/90 | legacy two-factor baseline |
| CPU `direct` | 90/90 | one-factor graph-local reuse |
| CUDA float32 directed | 72/90 | all 18 N=49 cells fail in warmup adjoint work |
| CUDA float32 symmetric | 88/90 | two N=49/L4/R256 warmup failures at B=4,8 |

Every completed configuration has finite gradients and passes all final-repeat
per-layer/sample audits. Failed rows have no invented timing, topology, or accuracy.
The original two symmetric failures remain in their original receipt.

The symmetric threshold-boundary mechanism was then isolated. Repeated evaluation
of the same CUDA float32 scatter operator moved an already deactivated residual from
just below to just above \(10^{-5}\). After committing the internal 0.99-headroom
change as `e688abc`, a clean rerun of the **same 90-cell set** completes 90/90:

- 910/910 final layer/sample maps audited, zero flips, finite gradients;
- maximum native primal/adjoint residuals
  \(9.8963\times10^{-6}\)/\(9.9086\times10^{-6}\);
- maximum independent CPU-double primal replay
  \(9.9017\times10^{-6}\);
- a separate explicitly indexed repeated cohort completes 10/10 and audits
  240/240 maps;
- together, 1,150 maps pass the numerical audit.

This closes those exact normal-strength failures without erasing them. It does not
prove deterministic CUDA reduction, universal CG convergence, or VJP accuracy from
residual acceptance.

### 8.3 Full adversarial float32 matrix

For every strength \(\sigma\in\{0.5,1,3\}\), the adversarial product contains 72
CUDA requests across directed/symmetric backends, three mesh sizes, two batches,
two layer counts, and three seeds.

| Strength/backend | Complete/requested | Preserved first failures |
|---|---:|---|
| 0.5 directed | 25/36 | 11 stationary-budget failures |
| 0.5 symmetric | 36/36 | none |
| 1.0 directed | 26/36 | 10 stationary-budget failures |
| 1.0 symmetric | 32/36 | 4 CG nonconvergences |
| 3.0 directed | 3/36 | 22 positive-face screens, 6 boundary-order failures, 5 stationary-budget failures |
| 3.0 symmetric | 23/36 | 10 boundary-order failures, 3 CG nonconvergences |

The exact total is 145 completions and 71 failures out of 216. Every completed row
passes the layer audits. A positive-face screen rejection can mean an insufficient
margin rather than a proven negative-area fold, while a boundary-order failure
demonstrates that theoretically positive segments collapsed in represented
arithmetic. Accepted-only timing is selection biased.

### 8.4 Full single-instance matrix

The formal application set has 13 task configurations per device: supervised map
fitting and synthetic image fitting at control sizes 17/25, 256-squared queries,
selected 512-squared Adam cases, and Adam/LBFGS where prescribed. CPU uses direct,
directed iterative, and symmetric; GPU uses the two iterative families. Thus 26
receipts contain 65 backend rows.

| Backend/device | Complete/requested | Threshold hits/requested | Map hits/5 | Image hits/8 |
|---|---:|---:|---:|---:|
| direct / CPU | 13/13 | 12/13 | 5 | 7 |
| directed iterative / CPU | 4/13 | 4/13 | 1 | 3 |
| symmetric / CPU | 13/13 | 11/13 | 4 | 7 |
| directed iterative / GPU | 13/13 | 12/13 | 5 | 7 |
| symmetric / GPU | 13/13 | 11/13 | 4 | 7 |

There are 56 completed rows, 9 explicit CPU-directed failures, and 50 threshold
hits among all 65 requests. The nine failures are five shadow-residual breakdowns,
three alpha-denominator breakdowns, and one 500-iteration nonconvergence. Six
completed runs miss the predeclared threshold: four checkerboard rows and the two
symmetric N=25 map-LBFGS rows.

All 56 completed runs certify every observed control map; the minimum recorded area
ratio is 0.0590 and the maximum flip count is zero. Failed rows have no completed
trace or final accuracy. Hit-only time-to-threshold medians are censored and are not
a fair speed ranking.

These are real decoder/warp/backward optimization loops but not neural-network
training. Every Adam evaluation or LBFGS closure calls the Tutte solver and dense
warp. The latent and boundary/modulus values themselves are `nn.Parameter` objects,
initialized neutrally. The shared target is generated outside optimization and
passed unchanged between backends.

The longer 200-update checkerboard extension preserves the original misses. Its
symmetric run reaches the image threshold at evaluation 100; directed does not
reach it within 200 updates. This shows the 40-update symmetric miss was not an
expressivity lower bound, while the directed non-hit is still only a bounded-budget
observation.

### 8.5 Explicit-float64 directed boundary

At N=49, R=256, strength 0.15, the exact
\(B\in\{1,4,8\}\times L\in\{1,2,4\}\times\) two-seed matrix completes 18/18 in
explicit float64 on one A6000. All 182 final layer/sample maps pass the audit, no
fallback occurs, and native/external residual maxima are below \(10^{-10}\). The
three identical-input CPU-direct/CUDA-double comparisons give the map and VJP
differences stated in Section 5.3.

This success is bounded by the stronger-weight extension. Its exact 24-cell
Cartesian product gives:

| Strength | Complete/requested | Certified final maps | First failure |
|---:|---:|---:|---|
| 0.5 | 8/8 | 50/50 | none |
| 1.0 | 7/8 | 34/34 | one primal iteration-limit failure |
| 3.0 | 0/8 | not reached | four primal iteration-limit failures and four primal shadow-residual breakdowns |

All nine failures occur during warmup before a measured repeat; no topology,
gradient, timing, or GPU allocator peak is imputed. Minimum accepted area ratio
falls from 0.0863 at strength 0.5 to 0.00393 at strength 1. Float64 is therefore
not a universal repair for arbitrary positive-weight contrast.

### 8.6 Failure-preservation ledger

| Evidence population | Preserved negative result | What is not inferred |
|---|---|---|
| Original normal float32 | directed N=49: 18/18 warmup failures; symmetric: 2/90 failures | no timing/topology/accuracy for failed rows |
| Clean symmetric postfix | 90/90 plus 10/10 succeeds under committed `e688abc` | does not rewrite the original two failures |
| Float32 adversarial | 71/216 failures across solver, boundary, and face screens | no hidden repair and no population failure probability |
| Formal instance matrix | 9/65 solver failures and 6 completed threshold misses | no missing final accuracy for failures; no expressivity lower bound from a miss |
| Checkerboard 200-step extension | directed still misses; traces differ from the 40-step run | no universal convergence comparison |
| Float64 directed conditioning | 9/24 warmup primal failures at stronger weights | no claim that float64 universally solves positive systems |
| Development regressions | alias corruption, under/overflow false convergence, empty-edge and boundary-order defects | fixes are supported by regressions, not erased from the worklog |

The final application chapter was independently re-parsed after Sections 8.4/8.5
were added. Its Cartesian counts, solve arithmetic, timing/memory scopes,
old-versus-postfix distinction, and raw JSONL-to-CSV projections match. After the
durability cleanup, a fresh CommonMark scan finds 65 local links and zero missing
targets.

## 9. Residual, topology, map accuracy, and Beltrami accuracy are different

For an approximate solution \(\widetilde x\),

\[
x-\widetilde x=A^{-1}(b-A\widetilde x).                               \tag{21}
\]

Therefore residual controls solution error only through the conditioning factor
\(\|A^{-1}\|\). The implicit VJP uses both an approximate primal and approximate
adjoint, so residual acceptance alone is not a conditioning-independent derivative
certificate.

For a positive-orientation P1 face,

\[
\mu_t=\frac{f_{\bar z}}{f_z},
\qquad
\det J_t=|f_z|^2-|f_{\bar z}|^2>0,
\]

and hence \(|\mu_t|<1\). This is a consequence of that returned face, not exact
reproduction of a desired coefficient or a fixed margin \(|\mu|\leq k<1\).
Because \(J_t\) contains an inverse source-edge matrix, a small coordinate error can
still yield a larger facewise coefficient error. The application chapter correctly
reports topology, residual, coordinate error, and \(\mu\)-error separately.

Likewise, the 93/93 legality result answers whether the benchmark targets are legal.
It does not answer whether an optimizer recovers them. The 56/65 completed formal
runs answer whether raw parameters can be optimized under the fixed protocol. They
do not identify a deformation uniquely from a scalar image: the accepted CPU-direct
checkerboard result has control-map RMSE 0.29242 and \(\mu\)-RMSE 0.67002 despite
being a valid map.

## 10. Adjudication against the user's target

The target is a fast, differentiable neural layer whose output is a piecewise-affine
homeomorphism and whose gradient can be backpropagated.

| Target component | Checker finding | Status |
|---|---|---|
| Learnable latent to map | Directed logits plus boundary/modulus, or symmetric conductances plus boundary, produce control coordinates through actual solves | **Achieved in bounded form** |
| P1 homeomorphism output | A single accepted layer is P1 on the declared mesh under the exact theorem hypotheses; software applies fail-closed numerical screens | **Achieved for the certified disk/structured-boundary contract, not as a universal exact floating predicate** |
| Multilayer homeomorphism | Retained domain-wide composition of accepted square layers is a PL homeomorphism on an induced refinement | **Achieved mathematically; not exported as one certified original-mesh P1 field** |
| Differentiable backward | Directed and symmetric first-order VJPs, boundary logits, modulus, batching, and composition have independent dense/FD evidence | **Achieved for first-order reverse mode** |
| Higher-order differentiation | Custom implicit backward and locator branches have no second-order contract | **Not achieved** |
| Fast forward/backward | Factor reuse, matrix-free state, query precomputation, full matrices, and million-query probes establish practical bounded workloads | **Partially achieved; no universal speed win and float64 directed is not a speed winner** |
| Memory scalability | Solver tape is iteration-independent; 1024-squared B=1/L=4 runs complete within about 421 MiB allocated | **Supported for selected workloads; memory still scales with B, L, mesh, and Q** |
| Neural-network integration | The layer uses Torch tensors/parameters and valid custom first-order backward | **Interface-ready, but no CNN training evidence** |
| Beltrami fidelity | \(|\mu|<1\) follows from positive faces, but arbitrary prescribed \(\mu\) is not recovered exactly | **Not achieved** |

The scientifically correct answer is therefore: Route I has produced a usable
hard-topology differentiable decoder prototype, but it has not yet produced the
universal, exact-Beltrami, uniformly fast neural layer envisioned by the overall
project.

## 11. Why Route II is still required

Route II is not needed to rescue a failed Route I topology theorem. It is needed to
address the optimization geometry and canonicalization problems that Route I has
made visible.

First, raw Tutte latents are redundant. Each directed softmax row is invariant to
adding a constant to all its logits. Symmetric conductances have a global positive
scale gauge. More generally, multiple positive operators can yield the same harmonic
coordinates. Thus the decoder can be valid while the learning coordinates remain
nonunique and poorly conditioned.

Second, directed and symmetric results do not decide representation versus
optimization. The symmetric family is reversible, but a missed target under one
optimizer/budget is not a proof that the target lies outside its attainable map set.
The 200-step checkerboard result already demonstrates that one earlier miss was a
budget effect. Route II's MVC/canonical-coordinate program is needed to test a
canonical encoding \(E(Y)\), the valid direction \(D(E(Y))=Y\) where it is actually
proved, and fair raw-Tutte-versus-canonical optimization on identical targets. It
must not claim the generally false inverse \(E(D(p))=p\).

Third, a future CNN will emit latents outside the small tested distribution unless
its output geometry is controlled. Extreme logits already cause probability
underflow, collapsed boundary increments, near-degenerate maps, and Krylov failure.
A canonical subset, legal retraction, and incremental update rule could improve
conditioning and trainability while retaining the Route I decoder as the final
topology-preserving map generator.

Fourth, Route I gives hard topology but not exact quasiconformal content. Route II
can test whether canonical coordinates improve map recovery and optimization; it
must not pretend that this alone solves prescribed-Beltrami accuracy. Route III's
primal-dual program remains necessary for the stronger conductivity/conjugacy and
anisotropy questions.

## 12. Mandatory closure checklist

| Mandatory item | Exact checker evidence | Verdict |
|---|---|---|
| Official paper/code audit | Pinned paper/repositories; center-split counts; learning/fitting distinction; KLU factor lifetime; 24-layer and locator accounting; no reproduction overclaim | PASS |
| Precise topology theorem | Certified disk, ordered convex boundary, strict supported weights, dividing-edge condition, and a posteriori PL degree theorem are stated separately | PASS |
| Finite-precision hard semantics | Exact represented-source checks, returned-dtype face screen, floating target limitations, fail-closed behavior, and no repair are explicit | PASS, bounded |
| Boundary and aspect parameter | Four ordered side distributions, exact corners, positive finite height, boundary/modulus VJPs, and dtype/device regressions independently checked | PASS |
| Reference and direct solvers | Legacy two-factor baseline is not relabeled; direct uses one factor and transpose solve; alias/cast-collapse defects closed | PASS |
| Directed matrix-free backend | Primal/transpose orientations, BiCGStab, restricted fallback, residual guards, constant-in-iteration saved state, and failures independently checked | PASS, availability empirical |
| Symmetric matrix-free backend | SPD edge operator, CG, normalization VJP, empty system, large RHS, reliable update, and CUDA headroom checked | PASS, bounded normal-strength scope |
| Exact first-order VJP | Directed and symmetric signs, boundary terms, normalization chain, broadcasting, gradcheck, dense references, and directional finite differences agree | PASS |
| Dense-query decoupling | Fixed 256/512 tables do no point location in prepared forward; 1024 probes complete; later composition queries correctly remain dynamic | PASS |
| Composition semantics | 1/2/4-layer coordinate composition and VJPs checked; domain-wide PL refinement claim is distinguished from original-mesh P1 resampling | PASS |
| Normal engineering matrix | Original four-backend 90-cell products retained; clean symmetric `e688abc` 90/90 and repeated 10/10 recomputed | PASS |
| Adversarial/conditioning evidence | 216 float32 cells and 24 stronger float64 cells preserve every failure and near-degenerate margin | PASS as negative/positive evidence |
| Mandatory instance optimization | 65 backend rows cover supervised map and image objectives, N=17/25, 256 and selected 512, Adam/LBFGS, actual solves/warps, topology and solve counts | PASS |
| Timing and memory scope | Solver, dense, audit, allocator, and process-HWM scopes remain distinct; no CPU/GPU causal ranking is claimed | PASS |
| Independent application review | Counts, CSV projection, failure blanks, thresholds, topology, accuracy, timing/memory, float64, postfix evidence, and 65-links/0-missing check independently reverified | PASS |
| CNN training | Explicitly optional under the Route I plan and explicitly absent | NOT REQUIRED FOR ROUTE I |

Every mandatory Route I closure item is present. The qualifications in the table are
scope boundaries, not missing mandatory experiments.

## 13. Strongest counterargument and checker response

The strongest counterargument is that “hard-bijective” overstates what floating
software proves. The exact theorem applies to an exact positive harmonic solution
and an exact valid boundary, while the actual layer may use an approximate Krylov
solution, nondeterministic scatter reductions, and floating boundary/area tests.
Some directed paths do not execute a generic dividing-edge preflight. Moreover,
many solver failures show that a theorem about existence is not an availability
guarantee.

That counterargument is valid against an unqualified production claim, but it does
not invalidate the bounded Route I result. The current claim is narrower:

- the exact construction is a homeomorphism under explicit disk/boundary/weight
  hypotheses;
- the prescribed structured rectangle closes the relevant dividing-edge condition;
- approximate returned coordinates are screened on every original face and boundary;
- numerical failures are surfaced rather than repaired or counted as success;
- the evidence reports where arithmetic or conditioning prevents a return.

Thus Route I has a genuine hard construction and a conservative numerical return
contract, but not a formal exact-predicate proof for all machine inputs. The latter
would require exact target predicates or a proved forward-error margin tied to
conditioning, neither of which is present.

## 14. Final Route I decision

**PASS -- bounded Route I closure.**

The Route I deliverables establish:

1. an independently audited understanding of official TutteNet rather than a
   mislabeled reproduction;
2. a precise disk/convex-boundary/positive-weight homeomorphism theorem and a
   fail-closed implementation contract;
3. correct first-order implicit differentiation for directed and symmetric
   systems;
4. actual control/query decoupling and coordinate composition;
5. complete prescribed normal, adversarial, and single-instance evidence with
   failures preserved;
6. a clean committed symmetric CUDA normal-strength closure and a bounded float64
   directed reliability option;
7. explicit limitations that motivate Route II rather than hiding Route I's
   redundancy, conditioning, expressivity, or training gaps.

This decision authorizes proceeding to the Route II MVC/canonical-coordinate
investigation. It is not evidence that Route II will succeed, not a declaration
that the final neural solver is complete, and not completion of the overall Phase V
wall-clock objective.
