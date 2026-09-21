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

Let (K=(V,F)) be a fixed oriented triangulation of a topological closed disk.
Write (V=I\mathbin{\dot\cup}B) for its interior and boundary vertices.  For
each (i\in I), let (N(i)) be its graph neighbors in a fixed slot order and
let

\[
 p_{ij}>0,\qquad \sum_{j\in N(i)}p_{ij}=1.
\]

The directed Tutte map (Y=D(p,b)\in\mathbb R^{|V|\times2}), with prescribed
boundary (Y_B=b), is the unique solution of

\[
 Y_i=\sum_{j\in N(i)}p_{ij}Y_j,\qquad i\in I. \tag{2.1}
\]

Split each row into interior and boundary neighbors.  In matrix form,

\[
 A(p)Y_I=B(p,b),\qquad
 A(p)=I-P_{II},\qquad B(p,b)=P_{IB}b. \tag{2.2}
\]

Under the certified disk, boundary-reachability, and strictly positive-weight
hypotheses used in Route I, (A(p)) is nonsingular.  The topology theorem also
requires the declared target-boundary and dividing-edge hypotheses.  Equation
(2.2) by itself is only a linear algebra statement.

We parameterize a row with logits

\[
 p_{ij}=\frac{\exp \ell_{ij}}
 {\sum_{k\in N(i)}\exp\ell_{ik}}. \tag{2.3}
\]

Adding a row constant (c_i) to every supported (ell_{ij}) leaves (p_i)
unchanged.  This is the softmax row-shift gauge.

## 3. Two distinct sources of non-uniqueness

The row-shift gauge is not the only redundancy.  For fixed neighbor positions
and a fixed interior point (Y_i), a barycentric row satisfies

\[
 \sum_jp_{ij}=1,\qquad \sum_jp_{ij}(Y_j-Y_i)=0. \tag{3.1}
\]

For a generic degree-(d_i) one-ring spanning the plane, these are three
independent scalar constraints on (d_i) weights.  Consequently the affine
space of weight perturbations preserving (Y_i) has dimension (d_i-3)
before positivity is imposed.  A strictly positive point of this affine space
therefore ordinarily lies in a local positive fiber when (d_i>3).  This is a
local row statement; coupling between interior rows does not restore global
injectivity of (D).

For logits, the softmax differential also annihilates the all-ones direction.
Thus one must not conflate:

1. the one-dimensional logit row-shift gauge, which leaves (p_i) unchanged;
2. barycentric null directions, which change (p_i) but preserve the same
   barycentric point to first order; and
3. the global decoder kernel, which can couple changes across several rows.

MVC canonicalization chooses one member of the positive fiber; it does not
make (D) injective on the entire positive-weight space.

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

Because (sum_j\delta p_{ij}=0), the right side is unchanged if (Y_i) is
subtracted from every neighbor.  Because (sum_jp_{ij}r_{ij}=0), the
softmax row mean in (4.2) also disappears.  Hence, with

\[
 b_i=d_i-\sum_jp_{ij}d_j, \tag{4.4}
\]

the row constraint on a logit lift is exactly

\[
 \sum_jp_{ij}r_{ij}\,\delta\ell_{ij}=b_i. \tag{4.5}
\]

The symbol (b_i) in (4.4) is a tangent residual and is unrelated to the
boundary-coordinate matrix (b=Y_B) in Section 2.

## 5. Covariance lift and its variational characterization

Define the (2\times2) row covariance

\[
 C_i=\sum_jp_{ij}r_{ij}r_{ij}^{\mathsf T}. \tag{5.1}
\]

If the neighbor rays span (mathbb R^2), then for every nonzero
(q\in\mathbb R^2),

\[
 q^{\mathsf T}C_iq
 =\sum_jp_{ij}(q^{\mathsf T}r_{ij})^2>0,
\]

so (C_i) is symmetric positive definite.  The covariance lift is

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
(\lambda_i\in\mathbb R^2) is (z_{ij}=r_{ij}^{\mathsf T}\lambda_i); the
constraint then gives (C_i\lambda_i=b_i).  Thus “minimum norm” here means the
specific probability-weighted logit norm in (5.5), not the Euclidean norm of
probability increments and not the Moore--Penrose solution under every metric.

## 6. From row constraints to the global decoder Jacobian

Let (d_B) be the derivative of the supplied boundary.  Assemble (5.2) at
every interior row, using (4.4) with both interior and boundary neighbor
directions.  Differentiating (2.2) yields a linear system for the decoder
tangent.  The prescribed (d) satisfies every differentiated row by (4.5) and
has the prescribed boundary (d_B).  Nonsingularity of (A(p)) makes that
solution unique.  Therefore

\[
 J_D(p,b)[\delta\ell,d_B]=d. \tag{6.1}
\]

This proof is global even though the lift is assembled row by row: uniqueness
of the coupled differentiated solve is the step that turns the local row
identities into (6.1).

## 7. MVC derivative versus covariance lift

Suppose (E_{\rm MVC}) is differentiable at (Y) and the exact canonical
identity holds on a neighborhood:

\[
 D(E_{\rm MVC}(Y))=Y. \tag{7.1}
\]

The chain rule gives

\[
 J_D\,dE_{\rm MVC}(Y)[d]=d. \tag{7.2}
\]

Equation (6.1) gives the same decoded tangent for the covariance lift (L_Yd):

\[
 J_D L_Yd=d. \tag{7.3}
\]

Consequently

\[
 J_D\bigl(dE_{\rm MVC}(Y)[d]-L_Yd\bigr)=0. \tag{7.4}
\]

There is no reason for the two latent vectors themselves to be equal.  The MVC
derivative differentiates a particular nonlinear canonical section; the
covariance formula solves the weighted minimum-norm problem (5.5).  Equality
would require additional structure and must not be assumed from (7.4).

## 8. Decoder retraction

Let (ell_c=E_{\rm MVC}(Y)) be canonical logits.  First consider a fixed
boundary and a direction with (d_B=0).  Define

\[
 R_Y(\alpha d)=D(\ell_c+\alpha L_Yd,\,Y_B). \tag{8.1}
\]

Then (7.1) gives (R_Y(0)=Y), while (6.1) gives

\[
 \left.\frac{d}{d\alpha}R_Y(\alpha d)\right|_{\alpha=0}=d. \tag{8.2}
\]

For a learned legal boundary parameter (q), let (b(q)) denote the Route-I
ordered rectangle layer and choose a parameter direction (h).  Put
(d_B=J_b(q)h) when forming (4.4), and define

\[
 R_{Y,q}(\alpha;d,h)
 =D(\ell_c+\alpha L_Yd,\,b(q+\alpha h)). \tag{8.3}
\]

The same proof yields the complete vertex tangent (d), provided its boundary
part equals (J_b(q)h).  An arbitrary boundary vertex vector is not generally
tangent to the ordered-rectangle parameter family.

Every accepted state in (8.1) or (8.3) is a fresh decoder output.  The Euler
predictor (Y+\alpha d) can guide a step, but it is never accepted as the map
and receives no Tutte topology guarantee.  In floating arithmetic, the
decoder's represented-weight, solve-residual, boundary, positive-face, and
independent topology checks still apply.

## 9. Conditioning and degeneracy

Let (0<\lambda_{\min}(C_i)\le\lambda_{\max}(C_i)).  Equation (5.2) implies
the bound

\[
 |\delta\ell_{ij}|
 \le \frac{\|r_{ij}\|\,\|b_i\|}{\lambda_{\min}(C_i)}. \tag{9.1}
\]

Thus a nearly collinear or extremely anisotropic one-ring can turn a modest
vertex direction into a very large latent step.  If all rays lie on a line,
(C_i) is singular and a generic two-dimensional (b_i) has no row lift.
The implementation must fail closed based on a declared rank/conditioning
criterion rather than silently regularize (C_i^{-1}) and call the result
exact.  A regularized lift could be studied separately, but it would solve a
different approximate problem.

## 10. Four optimization coordinates to compare

All comparisons use the same mesh, boundary family, target object, dense query
table, decoder backend, loss, initialization, number of accepted evaluations,
and public solve/topology tolerances.

### O1: bounded positive raw weights

For free variables (a_{ij}), set (w_{ij}=\operatorname{sigmoid}(a_{ij}))
and (p_{ij}=w_{ij}/\sum_kw_{ik}).  This is a positive-weight parameterization
with a nonlinear saturation geometry.  Calling (a) an unnormalized Laplacian
weight does not remove row-scale invariance after normalization.

### O2: row-softmax logits

Use (2.3) directly.  This removes raw row scale but retains the logit shift and
barycentric fibers described in Section 3.

### O3: MVC canonical re-encoding with ordinary gradient

Take an ordinary optimizer step through (D), decode the candidate, and replace
its latent row by (E_{\rm MVC}(Y)) before the next iteration.  Because the
re-encoding preserves the decoded map under its hypotheses, it is a gauge
choice rather than a post-hoc map repair.  Failed MVC-domain or decoder checks
remain failed observations.

### O4: MVC canonical coordinates with covariance retraction

Compute a geometric direction (d), lift it with (5.2), and accept only the
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

With (\Delta=Y'-Y), direct subtraction gives

\[
 \boxed{A'\Delta=(B'-B)-(A'-A)Y.} \tag{11.1}
\]

Equivalently, the right side is the new residual (B'-A'Y) of the old
solution.  Solving (11.1) from zero is algebraically the same shifted Krylov
problem as solving (A'Y'=B') from initial guess (Y).  With the same absolute
true-residual stopping contract, the two should have the same Krylov iteration
history up to implementation rounding.  A correction solve is therefore not
automatically fewer solves than a warm start; it is a useful formulation for
predictors, preconditioners, and local updates.

## 12. Row-local Woodbury update

If only the row set (R=\{r_1,\ldots,r_k\}) changes, write

\[
 A'=A+UV^{\mathsf T},\qquad
 U=[e_{r_1},\ldots,e_{r_k}], \tag{12.1}
\]

where row (m) of (V^{\mathsf T}) is the changed row
((A'-A)_{r_m,:}).  Provided both (A) and
(S=I_k+V^{\mathsf T}A^{-1}U) are nonsingular,

\[
 A'^{-1}B'
 =A^{-1}B'
 -A^{-1}U S^{-1}V^{\mathsf T}A^{-1}B'. \tag{12.2}
\]

This can reuse an existing factorization of (A).  Its dense Schur matrix is
(k\times k), so the construction is attractive only when the changed-row
set is genuinely small.  A global CNN update ordinarily has (k=|I|); then
the Schur problem is not low rank and (12.2) supplies no generic acceleration.

## 13. Initial incremental experiment

The compact receipt
[`route2_incremental_local.json`](raw_results/route2_incremental_local.json)
contains 24 float64 cases: control sides (N\in\{25,49\}), three seeds, and
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
code evaluates ((B'-B)-(A'-A)Y), while SciPy's warm start forms
(B'-A'Y); these equal expressions round differently and then enter a
nonsymmetric Krylov recurrence.  Both paths met the same declared absolute
residual contract and agreed with refactorization to the recorded tolerance.
This is evidence for warm-start reuse, not proof of improvement for every
update or every nonsymmetric Krylov implementation.

Woodbury was executed only when the update was declared local and contained at
most 64 rows.  Global cases retain the projected dense-Schur storage and an
explicit `not_run` reason.  At (N=49), a five-percent update already changes
111 rows and therefore exceeds this deliberately bounded local experiment.
Timing conclusions will be drawn only after an independent rerun on another
host; the current Windows observations primarily establish algebra and
iteration counts.

## 14. Required remaining evidence

Route II is not closed by this derivation or the initial incremental receipt.
The following remain mandatory:

1. finite-difference and explicit-Jacobian checks of (5.2)--(7.4);
2. differentiable M1 and M2 PyTorch layer checks on CPU and an idle remote GPU;
3. exact MVC encode--decode tests at control sides 11, 25, and 49;
4. fair O1--O4 supervised-map and 256-by-256 image instance comparisons;
5. forward/backward and peak-memory measurements;
6. an independent checker that did not author the formulas or implementation.
