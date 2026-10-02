# Does the initial similarity unnecessarily burden the nonrigid map?

## 1. A specific question, not a new deformation decoder

The original image-only initializers are stored as matrices A and translations
b and are called affine maps in the application. Inspection of the actual
fitting code shows that they are restricted to POSITIVE SIMILARITIES:

\[
A=\begin{pmatrix}a&-c\\c&a\end{pmatrix},\qquad a^2+c^2>0.
\]

These maps allow rotation, uniform scaling and translation, but not shear or
different scaling along different axes. The subsequent residual map has a fixed
identity boundary and a cumulative strain prior. Thus a missing global affine
component could burden the residual optimization. This is a hypothesis; a
six-parameter fit having lower TRAINING error does not demonstrate it.

This study asks whether the extra degrees of freedom improve a fixed label-free
out-of-fit check on the SAME frozen machine matches. It does not introduce a
new matcher, neural encoder or topology mechanism. No manual annotations or
label-oracle maps enter this diagnostic. These are known development images,
not independent unseen patients. Machine matches are not anatomical truth.

## 2. Inputs and the unchanged physical correspondence data

For each of the three original MIIT pairs, the saved matcher record contains
fixed-image coordinates q_j in [0,1]^2, affine-prepared moving coordinates p_j,
and confidence w_j in [0,1]. Let A_old,b_old be the ORIGINAL saved sampling
affine, faithfully promoted from its stored float32 values to float64. Define
the original-moving target

\[
y_j=A_{old}p_j+b_{old}.
\]

Freeze the original eligibility indicator 1_{y_j in [0,1]^2} and multiply it
into w_j. Both candidate models use exactly the same original eligible points,
coordinates and confidences. Do not rematch, change a confidence threshold,
delete residual outliers or redefine eligibility under a candidate initializer.
No manual-coordinate argument is accepted by the diagnostic tool.

The original machine loss uses an eight-canvas-pixel robust scale. Keeping the
512-pixel canvas and this scale, define the dimensionless error

\[
e_j(A,b)=64(Aq_j+b-y_j),\qquad
\rho(e)=\sqrt{1+\|e\|_2^2}-1.
\]

Use the numerically stable equivalent
rho(e)=||e||^2/(sqrt(1+||e||^2)+1). For a fitting set S,

\[
L_S(A,b)=\frac{\sum_{j\in S}w_j\rho(e_j(A,b))}
                   {\sum_{j\in S}w_j}.
\]

The similarity has four real parameters (a,c,b_x,b_y). The full affine has six
(A_11,A_12,A_21,A_22,b_x,b_y). Both use this SAME loss; the original RANSAC
similarity is only a historical reference. Comparing a new all-match affine
against only the old differently fitted similarity would confound model family
with fitting data/objective, so it is not the decisive comparison.

## 3. A small robust regression, not a large mesh inverse

Both model predictions are linear in a parameter vector theta of dimension
four or six. Write Aq_j+b=D_j theta, where D_j is its 2-by-d design matrix.
With normalized weights omega_j=w_j/sum_S w_j, the analytic gradient is

\[
\nabla_\theta L_S=64\sum_{j\in S}
\omega_jD_j^T\frac{e_j}{\sqrt{1+\|e_j\|^2}}.
\]

Its Hessian is

\[
64^2\sum_{j\in S}\omega_jD_j^T
\left(\frac{I}{s_j}-\frac{e_je_j^T}{s_j^3}\right)D_j,
\qquad s_j=\sqrt{1+\|e_j\|^2}.
\]

The middle matrix is positive definite for finite errors. Consequently a
full-column-rank weighted design gives a strictly convex fitting problem and a
coercive objective: its value tends to infinity as the coefficient norm tends
to infinity. Existence follows from coercivity and continuity; strict convexity
then gives a unique unconstrained minimizer. Rank deficiency is an explicit failure, not a
reason to regularize or silently replace the model.

Iteratively reweighted least squares is sufficient here. At theta_k, form
weights w_j/s_j(theta_k) and solve the corresponding weighted linear least-
squares problem for the next theta. This follows from the tangent upper bound
of the concave function sqrt(1+t) in t=||e||^2; the surrogate touches the actual
loss at the incoming parameters. Each solve contains ONLY four or six unknowns,
not 66,049 deformation vertices. Record the actual solve count, stationarity
norm and iteration cap. Numerical failure or lack of convergence is retained.

The unconstrained full-affine minimizer need not have positive determinant.
Check it in float64 and again after the production float32 storage conversion.
A nonpositive, degenerate or nonfinite fit FAILS; it is not projected or repaired
and is not replaced by the similarity while reported as a successful affine.
The positivity check is separate from convex regression's uniqueness claim.

## 4. One predeclared spatial two-fold check

Use the fixed source-coordinate rule

\[
k_x=\min(3,\lfloor4q_{jx}\rfloor),\quad
k_y=\min(3,\lfloor4q_{jy}\rfloor),\quad
c_j=(k_x+k_y)\bmod2.
\]

This is a 4-by-4 checkerboard of image regions, not a tunable random split.
Fit each model to parity0 and evaluate on ALL eligible parity1 points; reverse
once. For each fitting and held-out set report count, confidence mass, support,
robust loss, mean/p90/maximum Euclidean error in512-canvas pixels, fitted
singular values, determinants and convergence. Do not scan alternative folds,
spatial cell counts, penalties or robust scales.

Advance only if full affine has STRICTLY LOWER held-out robust loss in BOTH
directions of ALL three pairs, and all required fits are full-rank, converged
and positive after storage. Otherwise retire this EXACT application branch.
This is a project decision rule, not statistical evidence of clinical benefit.
The folds are correlated image regions and the points are machine predictions.

## 5. Conditional downstream experiment

Only if the diagnostic supports advancement, refit each model once to ALL the
same eligible points. Within each initializer arm, analytic, corrected F2 and
native DHR receive exactly that common initial transform and raw image pair.
Keep the original transport descriptor, masks, physical loss units, weights,
safe budgets and output interpretation. Preserve original results and failures.

If correspondences are represented in the new affine-prepared frame, convert
them by p_new=A_new^{-1}(y_j-b_new), retaining the same original-moving y_j and
confidence. This is a two-by-two coordinate conversion, NOT a new matching
operation or dense deformation solve. Check compatibility with the existing
loader's residual-target domain explicitly; do not silently drop points that
would become inconvenient under this conversion.

Score manual annotations only AFTER image-only inference; never select an
initializer, iteration or loss weight by them. Changing A changes the output
polygon and the physical interpretation of residual strain, so this is an
initialization/prior-interaction study, NOT an equal-functional geometry
ablation. A successful machine held-out test would authorize this bounded
experiment, not prove anatomical superiority.

## 6. Status

The question and decision rule were recorded before executing this diagnostic.
The actual one-shot local CPU probe and independent recomputation are complete.
ALL twelve fits are full rank, satisfy gradient-infinity-norm<=1e-8, have
positive exact stored float64/float32 determinants and finite offsets. Fits
take7--13 IRLS iterations. The advance condition nevertheless FAILS: only four
of six held-out directions improve. No all-point refits or new registrations
are executed, and no alternate split/model/loss is tried.

| Pair | Training parity | Held-out count | Similarity robust loss | Full-affine robust loss | Improve? |
|---|---:|---:|---:|---:|---|
|2 to3|0|99|.173494465|.188221722|No|
|2 to3|1|84|.156932992|.153249943|Yes|
|7 to8|0|77|.088803843|.080261813|Yes|
|7 to8|1|82|.118469733|.106114440|Yes|
|10 to11|0|144|.389868150|.385440288|Yes|
|10 to11|1|129|.390508540|.399405220|No|

Original raw/eligible counts are184/183,159/159,273/273; the one excluded
2-to3 point is outside the ORIGINAL moving world rectangle, not removed by
candidate error. All original eligible points are scored; no eligible point
has zero confidence here. The implementation retains zero-confidence rows
if present and explicitly fails a zero-total-confidence held-out fold.

Held-out MEAN/p90/max machine residuals are not manual TRE. For example, in
the failed2-to3 direction the mean changes4.768535 to5.017357 canvas pixels;
in the failed10-to11 direction it changes7.763569 to7.895493. Conversely7-to8
improves mean in both directions, but its p90 grows in both; lower robust
mean-type cost is not a universal tail improvement. Full-affine TRAINING losses
are lower throughout, as expected for a larger nested model, and do not overrule
the failed held-out rule. This closes the EXACT variant, not every possible
full-affine initializer or the mathematical usefulness of affine freedom.

Independent source-array/coordinate recomputation reproduces losses within
2.23e-16 and physical errors within1.26e-13 pixels. Separate direct-matrix
autodifferentiation agrees with gradients within4.11e-14. All actual Hessians
are positive definite; the smallest eigenvalue across these fits is81.0165.
Independent SciPy trust-region solves starting from their own weighted least
squares estimates agree within3.07e-11 in coefficients and4.45e-16 in objective.
Five SciPy runs report a numerical-progress limit near the optimum, not a
successful convergence flag; their independently recomputed gradient norms
still meet8.39e-9. The observed failure is not explained by an unconverged fit.

Before actual execution, twelve focused tests and independent off-optimum
finite differences, automatic derivatives, Hessian and majorizer calculations
check the implementation. Exact similarity/shear cases, rank deficiency,
negative/rounded-degenerate determinant, overflow translation, iteration-cap,
zero-confidence and retained-source-failure fixtures are included.

Code: `tools/coordinated_affine_capacity_probe.py`; actual complete report:
`outputs/coordinated_instance_registration/miit_initializer_capacity_t18.json`.
It accepts only an original prediction manifest and a fresh output path, not
annotations or teacher maps. Metadata explicitly discloses that the branch
was chosen after previously viewed errors/oracle analysis, despite opening
no annotation file itself. There is no untouched-validation or novelty claim.
