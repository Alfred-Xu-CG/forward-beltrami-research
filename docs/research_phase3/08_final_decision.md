# Phase III final decision: what is proved, what is falsified, and what remains

## 0. Decision in one paragraph

The Phase III objective was **not** to declare a successful prototype a finished
solver. The evidence does not establish a universal, fast, differentiable
forward Beltrami solver that accepts an arbitrary admissible coefficient and
always returns a piecewise-affine homeomorphism with a reliable neural-network
backward pass at million-vertex scale. The strongest closed results are instead
(i) a continuum mixed-Beltrami/conductivity rectangle theorem, (ii) a corrected
electrical primal--dual reference construction, (iii) a conditional
structure-preserving \(P_1\) conjugacy criterion, (iv) a precise local
weighted-cotangent sign identity, and (v) finite CPU differentiable prototypes
for implicit MBM and directed Tutte. The directed Tutte layer is the best
engineering starting point **under explicit graph and boundary certificates**;
the MBM layer has the stronger continuum interpretation but currently pays a
large sparse-factor memory cost and has a nonzero discrete conjugacy residual.
Neither can be called universal, production-ready, GPU-ready, or certified
hard-bijective for arbitrary \(\mu\).

## 1. Exact target and adjudication rule

The desired layer would map a mesh and a Beltrami coefficient
\(\mu=\mu_1+i\mu_2\), \(\|\mu\|_\infty<1\), to a map \(f_h\) such that:

1. \(f_h\) is piecewise affine on the input triangulation;
2. every face has positive signed Jacobian and the global map is injective;
3. the output approximates the Beltrami equation
   \[
   \partial_{\bar z}f=\mu\,\partial_z f;
   \]
4. the forward computation and reverse-mode derivative are fast and memory
   feasible on realistic meshes;
5. the claim remains valid for intended general-mesh and varying-\(\mu\) inputs,
   not only for a manufactured affine control.

The adjudication deliberately separates a local positive determinant check from
a global homeomorphism certificate. It also separates a continuum theorem from
the discrete \(P_1\) equations obtained after sampling \(\mu\). A finite VJP or
a low residual is evidence of differentiability or approximation; it is not by
itself a proof of global injectivity or exact discrete conjugacy.

## 2. Closed continuum theorem

Let \(\Omega\) be a bounded simply connected Lipschitz Jordan domain whose
boundary is divided into four arcs in counter-clockwise order. Let
\[
\mu\in L^\infty(\Omega;\mathbb C),\qquad \|\mu\|_{L^\infty}\le k<1,
\]
and define the real symmetric conductivity tensor
\[
A(\mu)=\frac{1}{1-|\mu|^2}
\begin{pmatrix}
|1-\mu|^2 & -2\,\operatorname{Im}\mu\\
-2\,\operatorname{Im}\mu & |1+\mu|^2
\end{pmatrix}.
\]
Then \(A\) is uniformly elliptic and \(\det A=1\) almost everywhere. Solve
\[
\operatorname{div}(A\nabla u)=0,\qquad
u=0\text{ on }\Gamma_L,\quad u=1\text{ on }\Gamma_R,\quad
(A\nabla u)\cdot n=0\text{ on }\Gamma_T\cup\Gamma_B.
\]
On a simply connected domain define the stream function \(v\), up to an
additive constant, by
\[
\nabla v=J A\nabla u,\qquad
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
\]
The divergence equation makes this one-form closed. The pair \(f=u+iv\), after
choosing the additive gauge and normalizing the right-side flux, satisfies the
Beltrami equation. The measurable Riemann mapping theorem, boundary extension
for a Jordan domain, and conformal rectangle uniformization give a
quasiconformal homeomorphism from \(\overline\Omega\) to a closed rectangle;
the interior image is the open rectangle. The side traces are therefore
ordered on the marked arcs. This is a continuum statement and does not say that
two independently assembled \(P_1\) solves have an exact common stream.

The modulus has the equivalent identities
\[
M=E_A(u)=\int_\Omega \nabla u^\mathsf T A\nabla u\,dx
=\int_{\Gamma_R}(A\nabla u)\cdot n\,ds.
\]
Solving the same-\(A\) complementary problem gives the reciprocal identity for
the complementary energy. Boundary equalities are weak trace/flux pairings;
they are not pointwise assertions for merely \(L^\infty\) coefficients.

### Conditional sampling lemma

If the continuum map is additionally \(C^2\) on each region sampled by a
conforming polygonal mesh, if a certified lower singular-value bound \(m_T>0\),
Hessian upper bound \(M_T\), and one fixed subordinate matrix/vector norm are
available, and if each triangle diameter \(h_T\) satisfies the stated Taylor
perturbation bound (in particular \(\eta_T<m_T\)), then the affine interpolant
has positive determinant on that face. A simple, noncollapsed boundary polygon
and a separate boundary-chord argument are also required for a global disk
homeomorphism. The measurable theorem \(\|\mu\|_\infty<1\) alone does not supply
those sampling hypotheses.

## 3. Exact discrete facts

### 3.1 Compatible \(P_1\) conjugacy

For a face \(T\), let \(p_T=\nabla u_h|_T\) and \(r_T=J A_Tp_T\). If \(r_T\)
is integrated as an edge one-form, then continuity of \(u_h\) and assembled
primary equilibrium imply normal flux cancellation across interior edges. A
stream coordinate can therefore be reconstructed only when the integrated edge
data are cycle-consistent. With \(B_0\) the vertex-to-edge incidence and \(B_1\)
the edge-to-face incidence, the distinct conditions are
\[
q=H_A B_0u,\qquad (B_0^\mathsf Tq)_I=0,
\]
for primal Kirchhoff conservation, \(B_1B_0u=0\) for gradient circulation, and
\(\widetilde B_1 Rq=0\) for dual exactness. Equivalently, if
\(\delta=Rq=B_0v\), then \(B_1\delta=0\). These are not interchangeable
notations. The checked edge-integrated prototype reproduces the compatible
layered manufactured map to numerical tolerance and rejects a randomly
incompatible potential; it is a compatibility diagnostic, not a general
solver.

### 3.2 Electrical rectangle

For a unit-conductance \(n_x\times n_y\) grid with left/right Dirichlet values
0 and 1 and natural top/bottom boundaries,
\[
u_{i,j}=\frac{i}{n_x-1},\qquad
M=E(u)=\text{right flux}=\frac{n_y}{n_x-1}.
\]
The corrected dual tiling has \(n_y\) strips, not \(n_y-1\). Independent energy,
flux, and area checks agree at \((2,2),(3,2),(3,3),(9,7)\), giving
\(2,1,1.5,0.875\). The weighted helper is restricted to row-path-independent
rectangular networks; this is not a general anisotropic Hodge theorem.

### 3.3 Anisotropic edge sign

For a facewise SPD tensor \(A_T\),
\[
K^T_{ij}=|T|(\nabla\phi_i)^\mathsf T A_T\nabla\phi_j.
\]
The assembled coefficient on an interior edge is exactly a weighted sum of
transformed cotangents. For general SPD tensors the weights include
\(\sqrt{\det A_T}\); for the Beltrami tensor \(\det A_T=1\), so the weights
cancel. The direct-gradient versus weighted-cotangent audit over 20,000 random
two-triangle samples had maximum absolute discrepancy \(3.73\times10^{-11}\)
and zero sign mismatches under the stated hypotheses. This is a local assembled
sign certificate, not a global \(M\)-matrix or homeomorphism theorem.

## 4. Closed counterexamples and negative results

1. **Fixed-\(P_1\) mixed-boundary monotonicity fails.** On the checked \(3\times3\)
   vertex mesh, facewise tensors induced by \(|\mu|=0.8\) are SPD, yet the
   left-side middle-row trace increment is \(-0.10057537\). Thus ellipticity
   alone does not guarantee ordered traces for an independently solved discrete
   complementary problem. This does not contradict the continuum canonical-map
   theorem.

2. **Local nonobtuse is not globally necessary.** A shared edge can have one
   positive local off-diagonal contribution and still have a negative assembled
   coefficient. The correct certificate is the assembled weighted edge sum plus
   the required \(Z\)-matrix/connectivity conditions; no universal necessity is
   claimed.

3. **Naive wide-stencil decoding is not topology preserving.** On an unstructured
   Delaunay mesh with 12,384 vertices and 24,382 faces, a one-ring support had no
   flips but a large coefficient-fit residual. A two-ring support reduced the
   residual mean to 0.002908684 but introduced 1,641 flipped original faces,
   with minimum original-face determinant \(-4.9312\times10^{-4}\). More support
   directions alone therefore do not supply a planar monotone decoder.

4. **Periodic Beurling FFT is not the free-space operator.** The Beurling
   transform is a whole-plane singular convolution. FFT diagonalization computes
   its periodized torus version. On the compact-disk test, periodic versus
   pad-2 differed by 5.19% in the central region and 27.5% in the outer annulus;
   periodic versus pad-8 differed by 5.53% and 29.5%. Pad-4 versus pad-8 agreed
   centrally to \(2.03\times10^{-4}\). Zero padding is an approximation, and a
   nonuniform mesh needs quadrature, NUFFT, treecode, or another non-FFT
   acceleration. No topology certificate was obtained.

## 5. Route-by-route decision

### Route A — continuum MBM and implicit \(P_1\) layer

**Decision: mathematically meaningful prototype; not the final solver.** The
implicit CPU implementation has finite forward and reverse values and a 7x7
directional finite-difference VJP relative error \(9.37199\times10^{-10}\).
For the smooth coefficient, direct implicit-output metrics and independent
reference metrics agree below \(1.2\times10^{-13}\), but the actual fixed-\(P_1\)
conjugacy residual is about 0.02416, 0.02366, 0.02352, 0.02355 at 65, 129,
257, 513 vertices. The induced face-\(\mu\) RMSE decreases from 0.0021288 to
0.0002682, while the maximum error remains about \(1.01\times10^{-2}\). At
513 vertices, forward/backward time was 69.093/48.669 s and process RSS delta
was about 1,230.8 MB on the recorded CPU. SuperLU fill ratios for one factor
reached 29.07--29.43 times the assembled matrix nonzeros. The route needs an
exact compatible stream construction, matrix-free or iterative solves,
checkpointed/recomputed VJPs, and a topology certificate before it can meet the
target.

### Route B — structure-preserving discrete conjugacy

**Decision: strongest theoretical repair, not yet a complete layer.** The
incidence equations and stream integration are now explicit and tested. The
remaining problem is constructive: for a general sampled \(\mu\), produce a
compatible positive-area \(P_1\) map while retaining sparse differentiability.
The fixed-\(P_1\) counterexample shows that solving primary and complementary
systems independently is insufficient.

### Route C — electrical primal--dual / anisotropic Delaunay / \(M\)-matrix

**Decision: reference oracle and conditional certificate.** The isotropic
rectangle bookkeeping is closed. The weighted edge identity is closed. A
general anisotropic, arbitrary-mesh, globally monotone, differentiable decoder
is not closed. Positive directional decompositions are not unique; a
nonnegative decomposition exists only on the documented interval and cannot be
assumed for every SPD face tensor.

### Route D — directed Tutte neural layer

**Decision: best near-term engineering candidate under caller certificates.**
The method solves an equilibrium system with positive directed weights and a
convex/weakly convex boundary. Its local-star argument is valid for an already
valid straight-line PL embedding with positive face areas and boundary-reachable
support. The implementation checks boundary orientation, repeated vertices,
turning, device, and empty-interior cases, but does not certify graph
3-connectivity-to-boundary. Thus the theorem is conditional. At 257x257 the
prototype had finite gradients and lower measured RSS/backward cost in the
unmatched comparison; at 513x513 it stayed positive in the tested stress runs,
but logit spread 3 reduced the minimum signed-area ratio to
\(2.5495\times10^{-12}\), a near-singular conditioning warning. Sine-bump fitting
was empirical only: map RMSE was \(1.27\times10^{-3}\) to \(4.12\times10^{-3}\),
face-\(\mu\) RMSE \(1.95\times10^{-2}\) to \(8.91\times10^{-2}\), and the
determinant margin approached \(3.75\times10^{-6}\).

### Route E — Beurling transform / Fourier acceleration

**Decision: useful analytical and regular-grid accelerator, not a certified
general-mesh layer.** The FFT computes a periodic convolution unless free-space
boundary effects are controlled by padding or a different discretization. The
route remains useful for whole-plane or padded regular-grid experiments, but it
does not solve the nonuniform-mesh and global-injectivity requirements.

### Route F — Beltrami holomorphic flow

**Decision: supporting direction only.** The recorded GPU-native 512x512-cell
single-step experiment assembled 524,288 faces in 1,044.19 s and completed in
1,066.53 s with zero flips and minimum face ratio 0.6343647; a separate 32x32
parity run had relative \(L^2\) error \(1.66\times10^{-7}\). The receipts lack
complete driver/software/seed/commit provenance and do not demonstrate a
multilevel method, arbitrary-mesh certificate, or production multistep cost.
Multigrid remains a plausible next optimization, but no Phase III theorem or
neural-layer result was established.

## 6. Fair benchmark interpretation

The constant real coefficient control \(a=0.3\), whose analytic map is
\[
f_\star(x,y)=\left(x,\frac{1-a}{1+a}y\right),\qquad
\frac{1-a}{1+a}=0.5384615385,
\]
was recovered to below \(4.3\times10^{-13}\) map RMSE with unit normalized
face-area ratio at both 129 and 257 vertices for both prototypes. This is a
matched analytic-target sanity check, not a common-input optimization benchmark:
MBM receives \(\mu\), while Tutte receives the exact boundary polygon and zero
logits. The fair rows now mark global flip_count as NA because no independent
global injectivity audit was run for that control. This is the correct scope,
not missing evidence to be silently inferred from positive face areas.

## 7. Bottlenecks that prevent the requested final solver

* **Topology:** positive local determinants do not replace a global injectivity
  certificate; arbitrary graph connectivity and boundary reachability remain
  external obligations for directed Tutte.
* **Compatibility:** independent primary/complementary \(P_1\) solves do not
  automatically produce an exact discrete stream.
* **Memory:** direct factors, not only sparse matrix assembly, dominate the MBM
  footprint. The 513-axis run already used roughly 1.23 GB RSS delta before
  considering batching or multi-sample training.
* **Differentiation:** finite CPU VJPs are demonstrated, but neither route has a
  validated CUDA kernel, batched sparse solve, or million-vertex memory plan.
* **Approximation:** the varying-\(\mu\) MBM output has a measurable residual and
  the Tutte fitting route has no arbitrary-\(\mu\) expressivity theorem.

## 8. Recommended next phase (three bounded tasks)

1. Construct a certified compatible discrete layer: parameterize a positive-area
   map or stream jointly, enforce the incidence equations as differentiable
   constraints, and prove the exact conditions under which the output is a
   planar homeomorphism.
2. Replace direct factor retention with a GPU-capable matrix-free/iterative
   solve plus checkpointed or recomputed adjoint, measuring peak allocator
   memory and batched scaling rather than process RSS deltas.
3. Build a fair arbitrary-mesh benchmark in which both methods receive the same
   target representation, run an independent global injectivity audit, and
   explicitly certify graph connectivity, boundary reachability, and
   conditioning.

Until those tasks are complete, the scientifically correct label is:

> **Phase III produced validated theorems, counterexamples, and differentiable
> CPU prototypes, but did not yet produce the requested universal fast
> hard-bijective forward Beltrami neural layer.**

## 9. Reproducibility and closure evidence

* Long-term documents: docs/research_phase3/00_correction_audit.md through
  08_final_decision.md, 06_results.csv, 07_independent_checks.md, and
  WORKLOG.md.
* Full regression after the final checker fixes: 318 passed, 1 warning in
  123.86s with PYTHONPATH=src, MKL_THREADING_LAYER=SEQUENTIAL, and
  OMP_NUM_THREADS=1.
* CSV audit: 30 data rows, 18 fields each; MBM v2 receipts finite and direct vs
  reference conjugacy/minimum-determinant agreement below 1.2e-13.
* All artifacts remain on the D: drive in D:\QC_optimization; the repository
  is synchronized with the GitHub remote after the final commit.
