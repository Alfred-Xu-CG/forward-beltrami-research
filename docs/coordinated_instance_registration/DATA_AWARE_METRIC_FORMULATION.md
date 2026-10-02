# Data-aware direction metric for the unchanged registration objective

Approved implementation card, 2026-10-02 11:31 UTC. The terminal1024 paired
experiment is complete: both arms worsen all four specimen mean errors versus
the retained512 recipe, with more time/memory. Independent saved-result checking
continues. There is no reason from that result to increase raster resolution
again. This card returns to the EXACT frozen-pose512 fused objective, not a
combination of pose/resolution changes. Implementation and accuracy are unproven.

## Research question, claim and limitations

Can accounting for image-residual sensitivity produce better progress on the
existing objective, and does that progress improve anatomical registration?
The proposed positive-definite matrix is a SEARCH METRIC, not the Hessian of
the nonsmooth objective, not a new topology decoder, and not a neural network.
The output remains the same fixed257-by257 source P1-ac map followed by the
same frozen positive affine. The test compares optimizers of the SAME objective.
It compares a physical-fiber metric/PCG/Armijo PACKAGE against analytic-latent
Adam, not an isolated ablation in which only the matrix changes. A benefit
would not alone identify the image diagonal as its unique cause.

Prior stiffness-only directions often required small Armijo steps. Existing
secant probes assigned most nonlinear remainder to MIND, not ARAP or the small
corner-shape term. Crucially, along an x-only or y-only update, bilinear image
sampling is piecewise affine because the other image coordinate stays fixed.
Absolute descriptor error is then piecewise linear away from image/residual
knots: its classical Hessian there is zero. A large finite-step secant remainder
means crossings of these knots, NOT a large smooth image Hessian eigenvalue.
An iteratively reweighted least-squares direction metric may anticipate these
crossings better than stiffness alone. This is a falsifiable hypothesis.

The proof-level claims are restricted: in exact arithmetic the stated matrix
is symmetric positive definite, and a trial passing actual feasibility and
original-objective decrease checks has those properties. Approximate Krylov
solves, automatic-differentiation choices at kinks and mixed-precision sampling
do not supply global convergence, stationarity or anatomical guarantees.

Classical image alignment using image Jacobians and Gauss--Newton approximations
provides the prior-art framework; this construction is not claimed novel merely
because it uses that framework. See [Baker and Matthews, Lucas--Kanade 20 Years
On](https://publications.ri.cmu.edu/lucas-kanade-20-years-on-a-unifying-framework).
Our absolute descriptor objective and hard P1 corner constraints are not the
smooth least-squares assumptions of a standard convergence argument.

## 1. The variables and the unchanged objective

Let Y0 be a legal incoming257-square vertex table, e either(1,0) or(0,1),
and c the255-square interior scalar coefficients. Zero boundary coefficients
are prescribed. At this final level, P inserts the interior into the same fine
grid; the general existing operator first zero-pads then bilinearly prolongs.
The stage maps are

\[
Y(c)=Y_0+(Pc)e.
\]

Let S be the actual fixed SOURCE P1 interpolation at512-square pixel centers,
and T=SP. At coarser levels this composition is NOT interchangeable with a
coarse-P1 map; this first experiment uses only the final equal-sized grid.
The fixed affine A,b, moving/fixed rasters and independently normalized frozen
SuperGlue/MatchAnything points are unchanged. MIND denotes the implemented
eight-channel self-similarity descriptor, not a newly trained feature extractor.

The complete current objective is

\[
E(Y)=I(Y)+3R(Y)+10^{-4}C(Y)+P_{fusion}(Y)+O(Y).
\]

Here I is the fixed-mask-normalized mean absolute difference of the eight
descriptors, R the actual P1 ARAP energy, C the actual four-corner shape energy,
O the original-moving-frame squared out-of-bounds penalty, and
P_fusion=.1P_SG+.1P_MA. Full definitions and units are preserved from FORMULATION,
the fusion card in OPTIMIZER_REDESIGN and JOINT_POSE_FORMULATION. No term, mask,
point weight, robust scale, interpolation or boundary condition changes here.
Compute g=gradient_c^AD E(Y(c)) through the UNCHANGED production Evidence code.

## 2. A frozen positive-definite metric at one current iterate

The proposed direction approximately solves H d=-g, where

\[
H=3P^TKP+T^T\operatorname{diag}(D_I+D_O)T
  +L^T\operatorname{diag}(\beta)L.
\]

K is the already verified fine P1 frozen-rotation ARAP stiffness: on this
uniform batch-one unit-square grid its interior matrix is the unscaled
five-point Dirichlet matrix (diagonal4, cardinal neighbors-1). The factor3
is the actual objective coefficient, not an arbitrary regularization weight.
This block is positive definite on zero-boundary interior vectors. L samples
the same scalar update at the frozen machine-point queries. The fused point
table already carries the exact separate normalization, so use its existing
weights and total coefficient .2 rather than duplicating them.

For pixel p and descriptor channel a let r_pa be its CURRENT signed descriptor
residual, j_pa its scalar-axis image sampling AD sensitivity, m_p its fixed mask
weight, and Z=sum_p m_p. With exactly eight channels,

\[
(D_I)_p=\frac{m_p}{8Z}\sum_{a=1}^8
              \frac{j_{pa}^2}{\max(|r_{pa}|,10^{-3})}.
\]

The fixed1e-3 is in descriptor units and stabilizes ONLY this metric. It does
not smooth the original loss or its gradient. Do not sweep it. Moving features
are already in the shared affine-aligned frame, so j has NO extra factor A.
Obtain j through eight channel-sum grid_sample VJPs at the actual float64
query, including the production `(2*q64-1).to(float32)` operation, zero padding
and align_corners=False. Do not substitute the old phase7 alignTrue/border
gradient helper. These are production AD slopes at the rounded sampling grid,
not classical derivatives of the full floating-point quantization function.

For v_p=A f_Y(q_p)+b, the native-frame outside contribution is

\[
(D_O)_p=\frac{2m_p}{Z}\sum_{b=1}^2(Ae)_b^2
           \mathbf1\{v_{pb}<0\text{ or }v_{pb}>1\}.
\]

At exactly0or1 use the production branch convention (zero). For each robust
point term lambda*w_j*(sqrt(1+||z_j||^2)-1), put
z_j=(512/8)A(f_Y(q_j)-p_j) and a=(512/8)Ae. Its EXACT scalar curvature is

\[
\beta_j=\lambda w_j
\frac{\|a\|^2+(a_1z_{j2}-a_2z_{j1})^2}{(1+\|z_j\|^2)^{3/2}}.
\]

This stable form avoids subtracting close positive terms. Shape remains in g
and E but has no added curvature model. Each data block is positive semidefinite;
the Dirichlet stiffness makes their sum positive definite in exact arithmetic.

Refresh all current residuals, slopes and metric weights ONCE PER OUTER STEP.
Use one additional feature forward and eight VJPs beside the unchanged full
objective gradient, recording this work. Collapse channels to one raster
diagonal; do not materialize a dense image Jacobian. Gather/scatter operators,
weights, coefficient vectors and Krylov arithmetic use float64. No sampler
second derivative is needed.

## 3. Approximate linear solve and actual feasible acceptance

Precondition conjugate gradients with M=3P^TKP+gamma*Identity, where
gamma=trace(H_data)/n and n=255^2. At this final level, trace is the sum of each
data weight times the squared norm of its actual INTERIOR query row (boundary
columns are not unknowns). The shifted separable sine-transform stiffness
inverse applies M^{-1}. Spatially varying data weights prevent it being an
exact inverse of H; no such claim is made.

Start PCG from zero; relative Euclidean residual tolerance .1, maximum20steps.
There is no additional Levenberg--Marquardt damping. Record residual, iterations
and reason. A capped approximate solve can be used only if finite and g^T d<0;
otherwise stop the stage with its last accepted map. Do not silently substitute
another solver or report a capped solve as converged.

Let Q be each corner determinant normalized by the reference cell area. The
fixed stage floor is Q_floor=.001+.05*(Q(Y0)-.001). Scalar-direction motion
makes every Q affine in c. For the current actual slack and negative corner
increments along d, calculate the exact-real maximum allowable alpha, then
start alpha=min(1,.99alpha_max). Allow at most12halvings, requiring actual
rounded corner feasibility AND E(candidate)<=E(current)+1e-4*alpha*g^T d.
Require the incoming actual Q(Y0)>.001 strictly, alpha>0, and the additional
computed strict inequality E(candidate)<E(current). There are at most13trial
evaluations: the initial trial plus12halvings. This extra strict comparison
prevents accepting an unchanged rounded objective when the Armijo right-hand
side rounds back to E(current). Report numerical stagnation/line-search
exhaustion rather than calling such equality strict descent.
The original complete objective is the authority. If no step passes, stop and
retain the last accepted map. At kinks, a negative computed slope alone does
not prove an acceptable step exists. Boundary values remain unchanged.

## 4. One bounded real experiment

Use all25 existing directions; these are FOUR REPEATEDLY VIEWED DEVELOPMENT
specimens, not held-out subjects. First obtain exactly the unchanged240-gradient
prefix through levels17/33/65/129, retain its last accepted map AND its best
full512-objective prefix. Both arms start from the SAME saved numerical prefix.
Any extraction/replay cost must be reported, not treated as free. Only replace
the final257-square horizontal/vertical stages:

- The existing analytic latent Adam update, fixed2seconds per axis.
- The new physical-fiber data-aware metric, fixed2seconds per axis.

Preserve the original final Adam learning rate and best-trial selection.
Include axis-specific layer/query-row/metric setup, full gradient, descriptor linearization, PCG, trials and necessary
synchronization in the stage clock. Finish an in-progress bounded iteration;
record actual elapsed time/overrun and counts. Equal nominal seconds are not
identical operation counts. No fixed300-gradient claim applies to these arms.
Original-image loading and common Evidence descriptor construction occur once
per suffix call and are separately timed in its COMPLETE call cost, not hidden
or claimed to occur inside both two-second axis clocks.
The archived fusion300 remains a lower-cost reference. Final selection compares
the common best prefix (including the original identity map) and accepted suffix states using ONLY unchanged E512.
Always record the suffix-start map and objective so objective progress is visible.

Before medium execution independently check actual-sampler slopes, point/OOB
coefficients, T/T^T adjoints, explicit tiny H versus matrix-free H, symmetry/SPD,
preconditioner trace including boundary rows, PCG status, and accepted rounded
corner floors. Tiny time-budget tests may use deterministic injected clocks;
the real experiment uses actual synchronized wall time. Default optimizer paths
must remain unchanged. All50predictions finish before ordinary annotation scoring.

Report E/components, mean/p90/worst registration errors, failure/stop reasons,
actual gradients/feature forwards/eight-channel VJPs/PCG iterations/trials,
accepted steps, actual time/peak memory and minimum corner ratio. Retain all
failed/early-stopped cases; legal retained maps are not evidence of completed
time budgets or stationarity. Never use labels to select the better arm per case.

Interpretation: lower E plus better anatomy is useful optimizer evidence;
lower E without better anatomy strengthens objective--anatomy mismatch; no
better E at comparable cost limits this metric. No result alone proves global
optimality, insufficient map capacity, or impossibility of another optimizer.
