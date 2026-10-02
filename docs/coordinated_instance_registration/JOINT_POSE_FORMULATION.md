# Candidate: joint positive global pose and a hard-feasible P1 residual

Status, 2026-10-02 10:12 UTC: approved bounded implementation experiment after
independent mathematical review. No joint-pose experiment or accuracy claim
has been completed. The preceding all-50 SG2/fusion comparison is complete and
independently checked. This candidate was derived by the coordinator and an
independent Astra xhigh context; it is not a new published registration theorem.

## 1. The question this changes

The saved image-only initializer is a positive similarity, denoted by
`A z + b`, with a nonsingular 2-by-2 matrix A and a two-component offset b.
The current residual deformation fixes the rectangle boundary. Consequently,
remaining global motion may need to be represented by interior displacement
and a transition back to that fixed boundary. Our residual strain penalty then
penalizes that transition. The following candidate removes this particular
restriction by adding six global variables. It does not assume that this is
the cause of the observed anatomical errors.

The older point-only initializer experiment is not rerun: it compared affine
and similarity fitting to subsets of old machine matches. The candidate here
would update global pose jointly with image evidence and the evolving residual.
It changes both the output polygon and the prior, so it is an application
experiment, not an equal-functional comparison of two geometry solvers.

## 2. Variables and the exact output map

Let Omega=[0,1]^2. Its fixed 257-by-257 vertex grid is triangulated with the
same southwest-to-northeast diagonal in every cell. A vertex table Y defines
the continuous piecewise affine (P1) function f_Y by barycentric interpolation
on those SOURCE triangles. Boundary vertices equal their reference positions.
The existing coordinated updates keep all four oriented corner determinants
of each cell positive, not merely the two belonging to the chosen diagonal.

Put o=(1/2,1/2). The additional affine function is

\[
G(z)=o+t+B(z-o),\qquad
B=R(\theta)\exp S,\qquad
S=\begin{pmatrix}s+d&k\\k&s-d\end{pmatrix},
\]

where t=(t_x,t_y), theta, s, d and k are six real scalar parameters. R(theta)
is the standard counterclockwise rotation matrix and exp is the matrix
exponential, defined by its convergent power series. Because S is symmetric,
exp(S) is positive definite and

\[
\det B=\exp(\operatorname{tr}S)=\exp(2s)>0.
\]

The complete fixed-to-moving sampling map would be

\[
F(x)=A\,G(f_Y(x))+b.
\]

An affine postcomposition preserves each source triangle, so F is P1 on the
SAME connectivity. There is no resampling of a composed nonlinear map. If f_Y
is a homeomorphism of the rectangle and the stored combined affine is
orientation-preserving, F is a homeomorphism onto the resulting parallelogram.
This is not a promise that F(Omega) equals the original moving image square.

There is no exact affine gauge ambiguity: if two such complete maps agree,
their boundary values agree at three noncollinear corners. Since each residual
fixes those corners, their G functions agree, and then their residuals agree.
This uniqueness statement does not guarantee numerical conditioning.
It identifies G as an affine function, not a unique angle parameter: theta
and theta+2pi represent the same rotation.

## 3. What the digital floor means with a variable pose

For cell-corner determinant q_Y, define Q_Y=q_Y/h^2, where h=1/256 and the
identity has Q_Y=1. The old residual floor is eta=.001. Affine postcomposition
gives, in real arithmetic,

\[
q_F=\det(A)\det(B)q_Y.
\]

Keeping only det(B)>0 retains positivity but does NOT retain the original
A-normalized .001 floor. To retain that quantitative restriction as well,
require BOTH Q_Y>eta and det(B) Q_Y>eta at every corner.

During a residual block, B is fixed. The existing safe-update operator can
therefore use the constant residual floor

\[
\eta_Y=\max(\eta,\eta/\det B).
\]

During a pose block, Y is fixed. Let m=min Q_Y over every cell corner. The
second condition becomes s>a with a=(1/2)log(eta/m). One possible parameter is
s=a+softplus(r), where softplus(r)=log(1+exp(r)). At block entry r must be
initialized from the CURRENT physical s and the NEW a; rebasing the lower
bound must not silently move the current map. Exact arithmetic legality does
not excuse floating-point overflow or rounding: trials and the final stored
map still require finite values, the declared strict floor, and actual binary
orientation checks. No post-hoc fold repair is introduced.
During a residual block, freeze the PHYSICAL B and s, not just the softplus
coordinate r. Recomputing a(Y) while holding r fixed would silently make B
depend on Y and invalidate the stated constant-floor update and derivative.

## 4. Image and point evidence

All images, masks and initial A,b remain the existing inputs. At raster side
r in {32,64,128,256,512}, pixel samples are at ((i+.5)/r,(j+.5)/r). Let Phi be
the IMPLEMENTED eight-channel MIND-like descriptor: for offsets (+/-2,0),
(0,+/-2), and (+/-2,+/-2), average squared intensity differences over a 3-by-3
patch, subtract the channel minimum, divide by the channel mean plus 1e-4,
and exponentiate the negative result. The existing replicate/patch boundary
rules are retained. This is our descriptor, not a claim to reproduce every
published variant called MIND.

At each level, first use the same area-downsampled original moving raster
I_{m,r} as the frozen-pose control; do not exchange downsampling and warping.
For candidate pose G, bilinearly sample that UNWARPED raster at A G(z)+b on the
r-square aligned raster, using zeros and align_corners=False.
Call that raster J_{r,G}. During pose optimization this operation and Phi must
remain differentiable with respect to the six pose parameters. During residual
optimization G is frozen and Phi(J_{r,G}) may be cached. The dense term is

\[
I_r(Y,G)=\frac{\sum_x m_r(x)\,
 \|\Phi(I_{f,r})(x)-\Phi(J_{r,G})(f_Y(x))\|_1/8}
 {\sum_x m_r(x)}.
\]

Descriptor evaluation at f_Y(x) uses the unchanged image sampler; source
queries use the declared P1 map. The original fixed mask m_r and its denominator
stay fixed. At G=identity this is exactly the current shared-affine formulation.

It would be WRONG simply to sample the OLD aligned descriptor canvas at
G(f_Y(x)): this can leave that intermediate square even when the actual point
A G(f_Y(x))+b is inside the original image. It also leaves descriptors in the
old pose frame. The current frozen-affine preparation helper explicitly rejects
trainable affine inputs; a new differentiable implementation must state and
test the changed assumption rather than bypass that check.

An old machine correspondence consists of source q_j, target p_j in the
ORIGINAL A-aligned frame, and confidence c_j. Keep its original static eligibility
e_j=1 when A p_j+b is in [0,1]^2, otherwise zero. Put
w_j=c_j e_j/(sum_k c_k e_k). With the eight-CANVAS-pixel robust scale, use

\[
P(Y,G)=\sum_jw_j\left[
 \sqrt{1+\|512 A(G(f_Y(q_j))-p_j)/8\|^2}-1\right].
\]

No target is refitted, re-matched or discarded because the new map changes.
The affine translation b cancels from this residual but remains essential to
eligibility and image sampling. The out-of-bounds penalty uses the COMPLETE
coordinates F(x), with the unchanged original mask and denominator.

An unchanged denominator is not a valid-support certificate. For example, take
A=identity, b=0 and Y=identity, shrink B to .032 times identity, and translate
the mapped square completely outside the original raster. The normalized
full-map determinant is still .001024>.001. The prewarped zero intensity has
an all-ones descriptor, just as constant fixed tissue does. Its dense loss can
therefore vanish without informative overlap. The point and out-of-bounds
terms still remain; this example does NOT prove the complete objective is
lower. It does show why final actual outside fractions and nominal descriptor
footprint/texture diagnostics matter. No adaptive denominator, query deletion
or newly tuned support filter is authorized by this observation.

The approved objective is I_r + 3 ARAP(Y) + 1e-4 Shape(Y)
+ .1 P_SG(Y,G) + .1 P_MA(Y,G) + O_r.
ARAP and the existing corner-shape penalty act on the residual Y, NOT on G(Y).
Global affine strain is intentionally not penalized. SG and MA denote the
existing SuperGlue and official MatchAnything tables. Each has its own
confidence-normalized eligible mean, represented by the already archived merged
table with total match weight .2. The all-25 fusion result supports this ONE
global exploratory recipe; neither evidence source nor weight is selected per
case. No new matching or label-based selection is part of this experiment.

## 5. Approved alternating experiment

The joint arm allocates five levels times (10 pose gradients,
25 horizontal residual gradients,25 vertical residual gradients), totaling 300.
Each pose block holds Y fixed. Each residual block holds G fixed and uses the
adjusted floor above. At pose-block entry define the rebased physical chart
w=(t_x,t_y,theta,r,d,k), with its current value w_0. Optimize z starting at zero,
using w_j=w_0,j+z_j/lambda_j. Let x run over that stage's fixed pixel queries,
and let v_j(x)=512 A partial_j G(f_Y(x)) evaluated at w_0. The frozen scale is

\[
\lambda_j=\left(\frac{\sum_x m_r(x)\|v_j(x)\|^2}
{\sum_x m_r(x)}\right)^{1/2}.
\]

Each lambda must be finite and greater than 1e-12; a degenerate chart is a
reported numerical failure, not an unexplained clipping/fallback. These six
fields can be evaluated from the small affine Jacobian without constructing
a dense image-to-parameter Jacobian. Fresh Adam uses the existing default
betas/epsilon, with the same physical learning rate at each control side l:

\[
\ell_l=512(0.004)\frac{16}{l-1}
\quad\text{for }l=17,33,65,129,257,
\]

namely 2.048, 1.024, .512, .256 and .128 moving-canvas pixels. This is a fixed
scale-calibrated schedule, not a guarantee that an Adam step has exactly that
displacement or lowers the loss. Residual blocks retain their existing learning
rates. No schedule or hyperparameter is chosen from evaluation landmarks.

The fresh paired experiment runs all 25 previously used directions with:

- frozen pose, 25 horizontal plus 25 vertical steps per level: 250 gradients;
- joint pose, 10 pose plus 25 horizontal plus 25 vertical: 300 gradients.

The archived frozen-pose fusion 300-gradient arm remains a third comparison.
This separates a pose effect from simply using 50 fewer residual steps. All
50 fresh jobs must reach a recorded terminal outcome before manual landmark
scoring. Failures remain in the comparison. Tiny analytic/derivative tests and
a same-input smoke run precede the full experiment; neither may use landmarks.
Production scope, evaluation data, raster levels and initialization are unchanged.

Stage acceptance and final full-resolution selection must store/select PAIRS
(G,Y). A previously best Y evaluated under a later G is a different map. Every
gradient, feature rebuild, objective evaluation, serialization and topology
check must be charged to actual runtime.300 gradients is not equal compute
to300 gradients of the frozen-pose control.

Export Y plus

\[
A_{out}=AB,\qquad b_{out}=A(o+t-Bo)+b,
\]

and retain original A,b separately. Check the actual rounded combined affine
and residual corner determinants; do not reuse a validator that assumes the
exported affine equals the original initializer.

Minimal decisive checks are identity-pose objective equivalence, all six pose
derivatives, literal composition/corner scaling, original-point coordinate
equivalence, and an intermediate-canvas-outside/original-image-inside sample.
The descriptor, absolute-value loss and bilinear sampling are only piecewise
differentiable; derivative tests must identify/avoid kinks rather than promise
classical derivatives everywhere. A separate test must freeze the physical
pose during residual differentiation, including the rebased floor chart.
Then compare real TRE, tails, failures and cost on all four specimens without
manual labels in optimization. Lower new objective alone is not anatomical
progress. A negative result limits this specific pose/prior hypothesis; it
does not establish that all legal maps or all affine formulations are inadequate.
