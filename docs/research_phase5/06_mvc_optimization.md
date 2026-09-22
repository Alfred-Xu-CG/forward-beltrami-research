# Route II optimization geometry: covariance lifts, decoder retractions, and incremental solves

## 1. Scope and status

This chapter develops the optimization geometry used by Phase V Route II.  It
does not introduce a second hard-bijective decoder.  The hard-valid map is
still produced by the positive-weight Tutte decoder from Route I.  Mean value
coordinates (MVC) select one representation of an already valid map, while a
covariance lift converts a desired first-order vertex displacement into one of
the many latent displacements that realize it.

Unless a paragraph is explicitly labelled as an experiment, the statements
below are exact finite-dimensional algebra under the listed hypotheses.  An
exact identity does not imply that a floating-point iterative solve converges,
that a represented triangle retains a useful area margin, or that a sampled
map is accepted by the numerical topology screen.

## 2. Fixed-mesh directed Tutte decoder

Let \(K=(V,F)\) be a fixed oriented triangulation of a topological closed disk.
Write \(V=I\mathbin{\dot\cup}B\) for its interior and boundary vertices.  For
each \(i\in I\), let \(N(i)\) be its graph neighbors in a fixed slot order and
let

\[
 p_{ij}>0,\qquad \sum_{j\in N(i)}p_{ij}=1.
\]

The directed Tutte map \(Y=D_p(p,b)\in\mathbb R^{|V|\times2}\), with prescribed
boundary \(Y_B=b\), is the unique solution of

\[
 Y_i=\sum_{j\in N(i)}p_{ij}Y_j,\qquad i\in I. \tag{2.1}
\]

Split each row into interior and boundary neighbors.  In matrix form,

\[
 A(p)Y_I=B(p,b),\qquad
 A(p)=I-P_{II},\qquad B(p,b)=P_{IB}b. \tag{2.2}
\]

Under the certified disk, boundary-reachability, and strictly positive-weight
hypotheses used in Route I, \(A(p)\) is nonsingular.  For completeness, the
global topology theorem additionally requires an orientation-preserving
homeomorphism from the source boundary to a convex target polygon.  A
*dividing edge* is a nonboundary mesh edge whose endpoints are both boundary
vertices; for a weakly convex target, no such edge may map wholly into the
target boundary.  A strictly convex target makes this obstruction automatic,
while the side-subdivided rectangle satisfies it through fixed side/corner
incidence and strict side order.  These are sufficient, not claimed necessary,
conditions for arbitrary positive rows to give a P1 homeomorphism.  Equation
(2.2) by itself is only a linear-algebra statement.

We parameterize a row with logits

\[
 p_{ij}=\frac{\exp \ell_{ij}}
 {\sum_{k\in N(i)}\exp\ell_{ik}}. \tag{2.3}
\]

Adding a row constant \(c_i\) to every supported \(\ell_{ij}\) leaves \(p_i\)
unchanged.  This is the softmax row-shift gauge.

## 3. Two distinct sources of non-uniqueness

The row-shift gauge is not the only redundancy.  For fixed neighbor positions
and a fixed interior point \(Y_i\), a barycentric row satisfies

\[
 \sum_jp_{ij}=1,\qquad \sum_jp_{ij}(Y_j-Y_i)=0. \tag{3.1}
\]

For a generic degree-\(d_i\) one-ring spanning the plane, these are three
independent scalar constraints on \(d_i\) weights.  Consequently the affine
space of weight perturbations preserving \(Y_i\) has dimension \(d_i-3\)
before positivity is imposed.  A strictly positive point of this affine space
therefore ordinarily lies in a local positive fiber when \(d_i>3\).  This is a
local row statement; coupling between interior rows does not restore global
injectivity of \(D\).

For logits, the softmax differential also annihilates the all-ones direction.
Thus one must not conflate:

1. the one-dimensional logit row-shift gauge, which leaves \(p_i\) unchanged;
2. barycentric null directions, which change \(p_i\) but preserve the same
   barycentric point to first order; and
3. the global decoder kernel, which can couple changes across several rows.

MVC canonicalization chooses one member of the positive fiber; it does not
make \(D\) injective on the entire positive-weight space.

## 4. Differential of one barycentric row

Define the neighbor ray and a target vertex direction by

\[
 r_{ij}=Y_j-Y_i,\qquad d_i=\left.\frac{d}{dt}Y_i(t)\right|_{t=0}. \tag{4.1}
\]

Differentiating the softmax gives

\[
 \delta p_{ij}
 =p_{ij}\left(\delta\ell_{ij}
 -\sum_kp_{ik}\delta\ell_{ik}\right). \tag{4.2}
\]

Differentiate (2.1):

\[
 d_i-\sum_jp_{ij}d_j
 =\sum_j\delta p_{ij}Y_j. \tag{4.3}
\]

Because \(\sum_j\delta p_{ij}=0\), the right side is unchanged if \(Y_i\) is
subtracted from every neighbor.  Because \(\sum_jp_{ij}r_{ij}=0\), the
softmax row mean in (4.2) also disappears.  Hence, with

\[
 b_i=d_i-\sum_jp_{ij}d_j, \tag{4.4}
\]

the row constraint on a logit lift is exactly

\[
 \sum_jp_{ij}r_{ij}\,\delta\ell_{ij}=b_i. \tag{4.5}
\]

The symbol \(b_i\) in (4.4) is a tangent residual and is unrelated to the
boundary-coordinate matrix \(b=Y_B\) in Section 2.

## 5. Covariance lift and its variational characterization

Define the \(2\times2\) row covariance

\[
 C_i=\sum_jp_{ij}r_{ij}r_{ij}^{\mathsf T}. \tag{5.1}
\]

If the neighbor rays span \(\mathbb R^2\), then for every nonzero
\(q\in\mathbb R^2\),

\[
 q^{\mathsf T}C_iq
 =\sum_jp_{ij}(q^{\mathsf T}r_{ij})^2>0,
\]

so \(C_i\) is symmetric positive definite.  The covariance lift is

\[
 \boxed{\delta\ell_{ij}=r_{ij}^{\mathsf T}C_i^{-1}b_i.} \tag{5.2}
\]

Substitution into (4.5) gives

\[
 \sum_jp_{ij}r_{ij}r_{ij}^{\mathsf T}C_i^{-1}b_i=b_i, \tag{5.3}
\]

so (5.2) is a right inverse of the row differential.  It automatically has
zero probability-weighted row mean:

\[
 \sum_jp_{ij}\delta\ell_{ij}
 =\left(\sum_jp_{ij}r_{ij}\right)^{\mathsf T}C_i^{-1}b_i=0. \tag{5.4}
\]

This is a convenient gauge, but it is not the unweighted mean-zero gauge used
to print canonical MVC logits.  The two gauges differ only by a row constant.

Equation (5.2) is also the unique minimizer, modulo unsupported slots, of

\[
 \min_{z_i}\ \frac12\sum_jp_{ij}z_{ij}^2
 \quad\text{subject to}\quad
 \sum_jp_{ij}r_{ij}z_{ij}=b_i. \tag{5.5}
\]

Indeed, the stationarity equation for a Lagrange multiplier
\(\lambda_i\in\mathbb R^2\) is
\(z_{ij}=r_{ij}^{\mathsf T}\lambda_i\); the constraint then gives
\(C_i\lambda_i=b_i\).  Thus “minimum norm” here means the
specific probability-weighted logit norm in (5.5), not the Euclidean norm of
probability increments and not the Moore--Penrose solution under every metric.

## 6. From row constraints to the global decoder Jacobian

Let \(d_B\) be the derivative of the supplied boundary.  Assemble (5.2) at
every interior row, using (4.4) with both interior and boundary neighbor
directions.  Differentiating (2.2) yields a linear system for the decoder
tangent.  The prescribed \(d\) satisfies every differentiated row by (4.5) and
has the prescribed boundary \(d_B\).  Nonsingularity of \(A(p)\) makes that
solution unique.  Therefore

\[
 J_{D_\ell}(\ell,b)[\delta\ell,d_B]=d. \tag{6.1}
\]

This proof is global even though the lift is assembled row by row: uniqueness
of the coupled differentiated solve is the step that turns the local row
identities into (6.1).

## 7. MVC derivative versus covariance lift

Suppose the logit encoder \(E_\ell\) is differentiable at \(Y\) and the exact canonical
identity holds on a neighborhood:

\[
 D_\ell(E_\ell(Y))=Y. \tag{7.1}
\]

The chain rule gives

\[
 J_{D_\ell}\,dE_\ell(Y)[d]=d. \tag{7.2}
\]

Equation (6.1) gives the same decoded tangent for the covariance lift \(L_Yd\):

\[
 J_{D_\ell} L_Yd=d. \tag{7.3}
\]

Consequently

\[
 J_{D_\ell}\bigl(dE_\ell(Y)[d]-L_Yd\bigr)=0. \tag{7.4}
\]

There is no reason for the two latent vectors themselves to be equal.  The MVC
derivative differentiates a particular nonlinear canonical section; the
covariance formula solves the weighted minimum-norm problem (5.5).  Equality
would require additional structure and must not be assumed from (7.4).

### 7.1 Which pseudoinverse is being compared?

For the explicit plan-section-22.3 comparison, padded slots are deleted and
each supported row is restricted to the arithmetic-zero-mean gauge.  Let \(Q\)
be an orthonormal row-block Helmert basis of that gauge and define the complete
fixed-boundary Jacobian

\[
J_{\mathcal G}
=\left.\partial_\theta D(\ell+Q\theta,b)\right|_{\theta=0}.
\tag{7.5}
\]

It contains all \(2|V|\) output coordinates; its boundary rows are identically
zero.  The Moore--Penrose representative is

\[
z_{\rm MP}=QJ_{\mathcal G}^{+}\operatorname{vec}(d).
\tag{7.6}
\]

Thus \(z_{\rm MP}\) minimizes the ordinary Euclidean logit norm *within this
fixed gauge*.  It must not be confused with the probability-metric solution
of (5.5).  To compare the covariance lift in the same gauge, subtract its
unweighted supported-row mean.  This row shift leaves the softmax and decoder
differentials unchanged, but the resulting vector no longer carries the
native weighted-minimum norm value.

The deterministic two-interior-vertex diagnostic has eight supported slots,
six gauge coordinates, four fixed-boundary tangent coordinates, and a
two-dimensional geometric kernel.  With the predeclared threshold

\[
\tau=64\epsilon_{64}\max(12,6)\sigma_{\max}
=6.9720\times10^{-14},
\tag{7.7}
\]

the six singular values were

\[
(0.4088450,\,0.3689694,\,0.1942412,\,0.1501054,
\,1.8298\times10^{-17},\,1.6853\times10^{-17}),
\tag{7.8}
\]

so the measured rank was exactly four.  A central finite-difference assembly
of the entire \(12\)-by-\(6\) Jacobian differed from the autodiff Jacobian by
\(6.925\times10^{-11}\) in relative Frobenius norm.  The numerical comparison
was

| common-gauge lift | Euclidean norm | \(p\)-weighted norm | \(\|Jz-d\|_2\) |
|---|---:|---:|---:|
| \(z_{\rm MP}\) | 0.2435816 | 0.1235130 | \(2.05\times10^{-17}\) |
| \(dE_\ell(Y)[d]\) | 0.2601148 | 0.1281226 | \(1.77\times10^{-17}\) |
| recentered covariance \(\widehat z_C\) | 0.2437569 | 0.1234326 | \(1.90\times10^{-17}\) |

The Euclidean distances \(\|dE-z_{\rm MP}\|_2\),
\(\|\widehat z_C-z_{\rm MP}\|_2\), and
\(\|dE-\widehat z_C\|_2\) were respectively \(0.0912564\), \(0.00924179\),
and \(0.0827460\).  Their images under the same Jacobian had norms between
\(8.05\times10^{-18}\) and \(3.03\times10^{-17}\).  Hence the vectors are
genuinely distinct while their differences are numerical kernel vectors.

The native covariance representative had \(p\)-weighted row mean below
\(3.47\times10^{-18}\) and weighted norm \(0.1231707\), smaller than each of
the three displayed representatives.  Let \(J_s\) be the independently
assembled Jacobian on all eight supported raw-logit coordinates and let
\(W=\operatorname{diag}(p)\).  The explicit weighted solution
\(W^{-1/2}(J_s W^{-1/2})^+d\) had Euclidean difference from the native covariance
formula was \(1.30\times10^{-16}\), with decoded residual
\(2.64\times10^{-17}\).  The native vector's unweighted supported-row sum was
\(0.0229488\), so it was deliberately reported outside the common-gauge table.
This separates two conditional but noninterchangeable theoretical
characterizations: \(z_{\rm MP}\) is the Euclidean minimizer in the declared
arithmetic gauge, while the native covariance lift is the probability-weighted
minimizer over all supported representatives.  The diagnostic verifies their
hypotheses and outcomes at this one state; it does not turn either
characterization into a conditioning or accuracy guarantee on other meshes.

### 7.2 Implicit decoder VJP and layer chain rules

The decoder reverse pass can be written without differentiating through a
factorization or storing Krylov iterates.  Let a scalar loss \(\mathcal L\)
provide cotangents

\[
 g_I=\frac{\partial\mathcal L}{\partial Y_I},
 \qquad
 g_B=\frac{\partial\mathcal L}{\partial b},
\tag{7.9}
\]

where \(g_B\) includes any direct loss dependence on the returned boundary.
Solve the one two-right-hand-side adjoint system

\[
 \boxed{A(p)^{\mathsf T}\Lambda=g_I.}
\tag{7.10}
\]

To obtain the signs, differentiate \(A(p)Y_I=P_{IB}(p)b\).  Since
\(\delta A=-\delta P_{II}\),

\[
 A\,\delta Y_I
 =\delta P_{II}Y_I+\delta P_{IB}b+P_{IB}\,\delta b.
\tag{7.11}
\]

Taking the Frobenius inner product of (7.11) with \(\Lambda\), using
(7.10), and differentiating the row softmax gives

\[
\begin{aligned}
 \delta\mathcal L
 &=\sum_{i\in I}\sum_{j\in N(i)}
   p_{ij}\,\Lambda_i^{\mathsf T}
   \left(Y_j-\sum_kp_{ik}Y_k\right)\delta\ell_{ij}
   +\left\langle g_B+P_{IB}^{\mathsf T}\Lambda,\delta b\right\rangle.
\end{aligned}
\tag{7.12}
\]

At a decoder equilibrium, \(\sum_kp_{ik}Y_k=Y_i\).  Therefore the supported
raw-logit and boundary VJPs are

\[
 \boxed{
 \frac{\partial\mathcal L}{\partial\ell_{ij}}
 =p_{ij}\,\Lambda_i^{\mathsf T}(Y_j-Y_i),
 \qquad
 \frac{\partial\mathcal L}{\partial b}
 =g_B+P_{IB}^{\mathsf T}\Lambda.}
\tag{7.13}
\]

The positive sign in the logit formula follows from
\(\delta A=-\delta P_{II}\); reversing that sign is an implementation error.
Equation (7.13) is invariant to an additive constant in a logit row because
the probability-weighted row sum is zero at equilibrium.  If the legal
boundary is parameterized by \(b=b(q)\), its parameter VJP is
\(J_b(q)^{\mathsf T}(g_B+P_{IB}^{\mathsf T}\Lambda)\).  For a fixed boundary,
the boundary cotangent is reported for diagnostics but is not propagated to a
learnable input.

These formulas also specify the complete layer reverse passes.  Write
\(z=(\ell,b)\).  For M1, \(M_1=E_\ell\circ D_\ell\), so a cotangent
\(\bar c\) at the canonical output gives

\[
 \operatorname{VJP}_{M_1}(\bar c)
 =J_{D_\ell}(z)^{\mathsf T}
  J_{E_\ell}(Y)^{\mathsf T}\bar c.
\tag{7.14}
\]

If the API also exposes \(Y\) and receives a direct cotangent \(\bar Y\), the
argument of the decoder VJP is
\(\bar Y+J_{E_\ell}(Y)^{\mathsf T}\bar c\).  For M2, the forward graph is

\[
 z\xmapsto{D_\ell}Y
 \xmapsto{E_\ell}c
 \xmapsto{L_Yd}\delta\ell
 \xmapsto{D_\ell(c_\ell+\alpha\delta\ell,\,c_b)}Y^+.
\tag{7.15}
\]

Reverse mode applies (7.13) at the final decoder, then the ordinary local VJPs
of addition, \(L_Yd\), and \(E_\ell\), and finally (7.13) at the initial
decoder.  A learnable \(d\), \(\alpha\), or boundary parameter receives its
corresponding local chain-rule term.  Thus M1 contains one implicit decoder
VJP and M2 contains two; neither requires unrolling a global iterative solve.

Finally, the centered covariance implementation sometimes writes
\(q_i=\sum_jp_{ij}Y_j\) and \(s_{ij}=Y_j-q_i\).  This equals the theory's
\(Y_j-Y_i\), and hence aligns with the MVC/decoder derivative, only at a
decoded equilibrium \(q_i=Y_i\).  Off equilibrium it can still solve a
centered row algebra problem, but it is not a right inverse of the decoder
differential at the arbitrary supplied array.

## 8. Decoder retraction

Let \(\ell_c=E_{\ell}(Y)\) be canonical logits.  First consider a fixed
boundary and a direction with \(d_B=0\).  Define

\[
 R_Y(\alpha d)=D_\ell(\ell_c+\alpha L_Yd,\,Y_B). \tag{8.1}
\]

Then (7.1) gives \(R_Y(0)=Y\), while (6.1) gives

\[
 \left.\frac{d}{d\alpha}R_Y(\alpha d)\right|_{\alpha=0}=d. \tag{8.2}
\]

For a learned legal boundary parameter \(q\), let \(b(q)\) denote the Route-I
ordered rectangle layer and choose a parameter direction \(h\).  Put
\(d_B=J_b(q)h\) when forming (4.4), and define

\[
 R_{Y,q}(\alpha;d,h)
 =D_\ell(\ell_c+\alpha L_Yd,\,b(q+\alpha h)). \tag{8.3}
\]

The same proof yields the complete vertex tangent \(d\), provided its boundary
part equals \(J_b(q)h\).  An arbitrary boundary vertex vector is not generally
tangent to the ordered-rectangle parameter family.

Every accepted state in (8.1) or (8.3) is a fresh decoder output.  The Euler
predictor \(Y+\alpha d\) can guide a step, but it is never accepted as the map
and receives no Tutte topology guarantee.  In floating arithmetic, the
decoder's represented-weight, solve-residual, boundary, positive-face, and
independent topology checks still apply.

## 9. Conditioning and degeneracy

Let \(0<\lambda_{\min}(C_i)\le\lambda_{\max}(C_i)\).  Equation (5.2) implies
the bound

\[
 |\delta\ell_{ij}|
 \le \frac{\|r_{ij}\|\,\|b_i\|}{\lambda_{\min}(C_i)}. \tag{9.1}
\]

Thus a nearly collinear or extremely anisotropic one-ring can turn a modest
vertex direction into a very large latent step.  If all rays lie on a line,
\(C_i\) is singular and a generic two-dimensional \(b_i\) has no row lift.
The implementation must fail closed based on a declared rank/conditioning
criterion rather than silently regularize \(C_i^{-1}\) and call the result
exact.  A regularized lift could be studied separately, but it would solve a
different approximate problem.

## 10. Four optimization coordinates to compare

All comparisons use the same mesh, boundary family, target object, dense query
table, decoder backend, loss, initialization, number of accepted evaluations,
and public solve/topology tolerances.

### O1: bounded positive raw weights

For free variables \(a_{ij}\), set \(w_{ij}=\operatorname{sigmoid}(a_{ij})\)
and \(p_{ij}=w_{ij}/\sum_kw_{ik}\).  This is a positive-weight parameterization
with a nonlinear saturation geometry.  Calling \(a\) an unnormalized Laplacian
weight does not remove row-scale invariance after normalization.

### O2: row-softmax logits

Use (2.3) directly.  This removes raw row scale but retains the logit shift and
barycentric fibers described in Section 3.

### O3: MVC canonical re-encoding with ordinary gradient

Take an ordinary optimizer step through \(D_\ell\), decode the candidate, and
replace its latent row by \(E_\ell(Y)\) before the next iteration.  Because the
re-encoding preserves the decoded map under its hypotheses, it is a gauge
choice rather than a post-hoc map repair.  Failed MVC-domain or decoder checks
remain failed observations.

### O4: MVC canonical coordinates with covariance retraction

Compute a geometric direction \(d\), lift it with (5.2), and accept only the
decoded state (8.1) or (8.3).  The definition of the geometric direction,
boundary tangent, step-size rule, and number of trial solves must be reported;
otherwise solve counts are not comparable to O1--O3.

No variant is declared superior a priori.  The relevant observations are loss
versus global solves, loss versus wall time, map and Beltrami error, minimum
Jacobian/area margin, covariance and linear-system conditioning, logit spread,
and peak memory.

## 11. Exact incremental correction

Let an accepted old state satisfy

\[
 AY=B,
\]

and let the next represented state satisfy

\[
 A'Y'=B'.
\]

With \(\Delta=Y'-Y\), direct subtraction gives

\[
 \boxed{A'\Delta=(B'-B)-(A'-A)Y.} \tag{11.1}
\]

Equivalently, the right side is the new residual \(B'-A'Y\) of the old
solution.  Solving (11.1) from zero is algebraically the same shifted Krylov
problem as solving \(A'Y'=B'\) from initial guess \(Y\).  With the same absolute
true-residual stopping contract, the two should have the same Krylov iteration
history up to implementation rounding.  A correction solve is therefore not
automatically fewer solves than a warm start; it is a useful formulation for
predictors, preconditioners, and local updates.

## 12. Row-local Woodbury update

If only the row set \(R=\{r_1,\ldots,r_k\}\) changes, write

\[
 A'=A+UV^{\mathsf T},\qquad
 U=[e_{r_1},\ldots,e_{r_k}], \tag{12.1}
\]

where row \(m\) of \(V^{\mathsf T}\) is the changed row
\((A'-A)_{r_m,:}\).  Provided both \(A\) and
\(S=I_k+V^{\mathsf T}A^{-1}U\) are nonsingular,

\[
 A'^{-1}B'
 =A^{-1}B'
 -A^{-1}U S^{-1}V^{\mathsf T}A^{-1}B'. \tag{12.2}
\]

This can reuse an existing factorization of \(A\).  Its dense Schur matrix is
\(k\times k\), so the construction is attractive only when the changed-row
set is genuinely small.  A global CNN update ordinarily has \(k=|I|\); then
the Schur problem is not low rank and (12.2) supplies no generic acceleration.

## 13. Initial incremental experiment

The compact receipt
[`route2_incremental_local.json`](raw_results/route2_incremental_local.json)
contains 24 float64 cases: control sides \(N\in\{25,49\}\), three seeds, and
one-row, five-percent-row, global-small, and global-large updates.  Every
method in a row solves the identical new system.  A direct new factorization is
the accuracy reference; cold and warm BiCGStab use the same per-coordinate
absolute residual tolerance; the correction equation uses that same absolute
tolerance rather than a looser relative tolerance on its small right-hand
side.  Every reference map passes the independent P1 topology audit.

Observed maximum coordinate-iteration means across the three seeds were:

| control side | update | cold mean | warm mean | correction mean |
|---:|:---|---:|---:|---:|
| 25 | one row | 51.00 | 40.00 | 39.33 |
| 25 | five percent | 51.33 | 44.67 | 45.33 |
| 25 | global small | 51.33 | 41.33 | 40.67 |
| 25 | global large | 51.33 | 47.33 | 47.00 |
| 49 | one row | 105.67 | 74.00 | 76.00 |
| 49 | five percent | 106.33 | 88.33 | 86.67 |
| 49 | global small | 106.00 | 81.33 | 81.33 |
| 49 | global large | 104.33 | 91.67 | 92.33 |

Thus the old solution reduced iterations in this bounded sample, including the
global updates, but the advantage shrank for the large update.  The exact
shifted equation and warm start were close rather than bitwise identical: only
4 of 24 two-coordinate iteration tuples matched exactly, the largest
per-coordinate count difference was 7 iterations, and the mean absolute count
difference was 1.5.  The reason is numerical, not algebraic: the correction
code evaluates \((B'-B)-(A'-A)Y\), while SciPy's warm start forms
\(B'-A'Y\); these equal expressions round differently and then enter a
nonsymmetric Krylov recurrence.  Both paths met the same declared absolute
residual contract and agreed with refactorization to the recorded tolerance.
This is evidence for warm-start reuse, not proof of improvement for every
update or every nonsymmetric Krylov implementation.

Woodbury was executed only when the update was declared local and contained at
most 64 rows.  Global cases retain the projected dense-Schur storage and an
explicit `not_run` reason.  At \(N=49\), a five-percent update already changes
111 rows and therefore exceeds this deliberately bounded local experiment.

An independent Linux rerun is preserved in
[`route2_incremental_turing.json`](raw_results/route2_incremental_turing.json).
It used commit `be08f45`, Python 3.11.7, NumPy 1.26.4, SciPy 1.11.4, and one
OMP/MKL thread on host `turing`.  All 24 rows completed, all reference maps
were certified, and maximum coordinate errors relative to a fresh SuperLU
factorization were \(5.92\times10^{-9}\), \(5.79\times10^{-9}\), and
\(5.74\times10^{-9}\) for cold, warm, and correction solutions.  The old
SciPy `tol` interface initially exposed a genuine cross-version failure before
any solve; a regression now maps both old `tol` and new `rtol` APIs to the same
zero-relative, declared-absolute contract.

The independent iteration pattern agrees qualitatively but not bitwise with
Windows.  At \(N=49\), the mean maximum-coordinate counts for
`local_one/global_small/global_large` were respectively
\(106.33/107.33/103.67\) cold and \(76.33/79.00/93.67\) warm.  Correction
means were \(75.33/79.67/94.33\).  Across all rows, five warm/correction tuples
matched exactly, their largest per-coordinate count difference was eight, and
their mean absolute difference was 2.10 iterations.  Mean observed Turing
cold-versus-warm times ranged from 4.66 versus 3.69 ms at \(N=25\) one-row to
14.61 versus 10.20 ms at \(N=49\) one-row; the \(N=49\) global-large means were
14.31 versus 12.74 ms.  These sequential single-host observations support an
iteration/work reduction for the tested small updates, but they are not a
randomized performance theorem, and fresh SuperLU remained faster in these
CPU cases (roughly 1.4--1.5 ms at \(N=25\) and 6.6--7.1 ms at \(N=49\)).

The bounded Woodbury branch supplies a separate exactness and timing result.
Each receipt contains nine successful local cases: three seeds for the
\(N=25\) one-row and five-percent-row updates and for the \(N=49\) one-row
update.  The other updates were intentionally not run because the declared
64-row cap was exceeded.  Relative to fresh refactorization, the observed
maxima were:

| Receipt | successful / eligible | max coordinate error | max full-system relative residual | max Schur condition |
|---|---:|---:|---:|---:|
| Windows local | 9 / 9 | \(3.11\times10^{-15}\) | \(1.60\times10^{-15}\) | 1.02256 |
| Turing | 9 / 9 | \(2.78\times10^{-15}\) | \(1.58\times10^{-15}\) | 1.02256 |

The mean Woodbury-time divided by mean fresh-refactorization-time ratio was:

| Receipt | \(N=25\), five percent | \(N=25\), one row | \(N=49\), one row |
|---|---:|---:|---:|
| Windows local | 4.72 (slower) | 1.61 (slower) | 0.284 (faster) |
| Turing | 1.066 (slower) | 0.641 (faster) | 0.215 (faster) |

These are sequential, platform-dependent pilot timings.  They demonstrate
machine-precision algebraic agreement and show that a sufficiently small
update can amortize a stored factorization at \(N=49\); they do not imply that
Woodbury accelerates five-percent or globally dense neural updates.

## 14. Evidence ledger and remaining work

The following items are completed in durable receipts and are summarized in
the application chapter: finite-difference and explicit-Jacobian checks of the
covariance lift; differentiable M1 and M2 CPU/GPU layer runs; all 27 local and
27 AI float64 MVC round trips at control sides 11, 25, and 49; the declared
common-gauge Moore--Penrose comparison; and the two-host incremental and
bounded-Woodbury experiments above.  The corrected 256-by-256 CPU instance
protocol has also run, including its genuine O4 image-task failure, and a clean
strict CUDA rerun reproduces the successful map cohort and the same O4 image
failure.

The remaining items are deliberately not converted into a Route-II pass:

1. adjudication of the incomplete O4 image cohort rather than silently
   replacing it with a tuned or trust-region variant;
2. a parameterization-fair speed study if a speed ranking, rather than the
   existing shared-rate robustness observation, is claimed; and
3. an independent final checker that did not author these formulas,
   implementation paths, or result synthesis.
