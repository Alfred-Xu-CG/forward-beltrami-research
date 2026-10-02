# Optimizer diagnosis and controlled registration experiments

Question: does retaining coarse structural image evidence during fine optimization
and output selection improve the actual MIIT anatomical correspondence?

The current three-pair development recipe has 930 analytic trial scales equal to
one. Its stage anchors, full-objective prefix selection and physical coordinate
units match the stated algorithm; no implementation error was found in this
audit. More fine iterations, second cycles, joint coordinates, proposal filtering,
and physical-fiber L-BFGS have already been tested with mixed/negative anatomy.
Earlier bending diagnostics also do not support excess roughness as the cause.
These observations do not prove convergence or adequate correspondence evidence.

The concrete hypothesis is that the final single-resolution data term discards
large-neighborhood information. At a 32-square raster, descriptor offset 2 plus
patch radius 1 reaches approximately 48 canvas pixels at a 512-square canvas;
at the final raster the same construction reaches only 3 pixels. This is a
receptive-footprint statement, not a guaranteed registration capture radius.

Exact claim and units: for S={32,64,128,256,512}, construct the existing frozen
shared-affine features separately at each raster scale, using area-reduced raw
intensities and fixed masks. For the SAME 257-square P1-ac residual map Y, let
I_s(Y) be the current mask-normalized eight-channel mean absolute descriptor
error. Define

    I_MS(Y) = (I_32(Y)+I_64(Y)+I_128(Y)+I_256(Y)+I_512(Y))/5,
    E_MS(Y) = I_MS(Y)+3 ARAP(Y)+1e-4 Shape(Y)+.1 Matches(Y)+OOB_512(Y).

Each I_s has its own immutable fixed-mask denominator. Source queries are that
raster's pixel centers; the map is evaluated directly with its declared P1-ac
interpolation, not resampled as a new certified map. Affine preparation occurs
once per scale. Priors, machine points and full-raster OOB appear EXACTLY ONCE.
No per-candidate foreground dropping or confidence change occurs. All terms are
dimensionless; machine points retain their original eight-canvas-pixel robust
scale. Equal image weights follow their common normalized descriptor range,
not manual landmarks or oracle calibration.

At EVERY coefficient level and directional stage, optimize and accept by E_MS;
select among identity/accepted prefixes by that SAME E_MS. In contrast, the
comparison recipe uses image continuation 32,64,128,256,512 and selects by E_512.
This changes both the data functional and its continuation schedule. It is NOT
an isolated optimizer or geometry ablation. The gradients are the weighted sum
of image-term gradients plus one full-resolution prior/point/OOB gradient.

Assumptions: same frozen positive affine, identity residual boundary, exact four
corner constraints and .001 floor; shared-affine descriptor frame in BOTH arms;
same images/matches/confidences, rates, and 300-gradient budgets. Analytic uses
five x/y stages; F2 uses its existing two cycles. Neither reads manual labels.

Smallest decisive tests: independently summed values and full-map gradients on
nonaligned raster/control grids and fractional masks; fine-only weights reproduce
E_512; a counting fixture verifies priors/points are not duplicated; tiny complete
analytic/F2 runs preserve budgets, fixed boundaries and certificates. Then all
three existing MIIT pairs, both methods, baseline and MS, with scoring only after
all prediction attempts terminate. Report mean, p90, maximum, time, memory and
failures per pair plus equal-pair aggregates. The same known specimen remains
development-only. No neighboring scale/weight sweep is authorized by this card.

Falsifiers and tradeoffs: coarse descriptors may favor incorrect cross-stain
structures, and using fine features from the first stage can introduce local
minima. Worse aggregate anatomy, worse tails, or increased cost without useful
accuracy is retained as a negative outcome. Mixed per-case results are reported
as a Pareto tradeoff rather than discarded by an all-cases-strictly-better gate.
One step costs five raster terms (349184 pixel centers, 1.33203125 times the full
raster), plus the unchanged single priors. Across 300 gradients, this is exactly
FIVE times the raster/channel work of the original evenly-budgeted continuation,
because all five scales are now present at each stage. Geometry/prior counts
remain unchanged; neither pixel ratio is a measured runtime prediction.

Prior work: multiresolution registration and sums of normalized image losses are
conventional constructions. This experiment claims no new registration metric,
topology theorem, neural architecture, or global convergence result.

Implementation checks: 48 focused tests pass in 7.49 seconds using the existing
clean local environment. The new tests compare values and complete vertex VJPs
against separately summed terms on 7-by-9 controls with 11/19/31-square rasters
and fractional masks, verify exact fine-only equivalence and absence of coarse
prior calls, check finite differences away from bilinear/L1 knots, and run tiny
complete analytic/F2 optimizations with unchanged budgets and valid exports.
Original shared-affine and descriptor application tests also pass. A further 73
affected application/nested/ARAP tests passed. These checks establish code
consistency, not anatomical effectiveness.

## Actual result: this MS recipe is not retained

The primary MS outputs are `miit_multiscale_sum_direct_t19`, scored against
`miit_multiscale_control_t19`; both live under the existing outputs directory.
The earlier subtract/add arithmetic run is retained separately and is not the
primary comparison. Direct reconstruction of the complete scalar objective is
algebraically equivalent but avoids an extra float32 subtraction rounding.

|Method|Control mean / mean p90, canvas px|MS mean / mean p90, canvas px|
|---|---:|---:|
|Analytic|3.54875190 / 5.90795739|3.60818017 / 6.06018587|
|F2|3.55432749 / 5.94294291|3.61253747 / 6.06281617|

All six pair/method means worsen. Both methods slightly improve pair 7-to-8's
p90 and maximum, but these limited tail changes do not offset the aggregate
regression and increased cost. All six primary MS attempts complete 300
gradients with valid outputs. The tested recipe is a nonsurviving main candidate;
no neighboring image-weight or scale sweep follows. This does not show that all
coarse information or all multiscale objectives are ineffective.

## Approved bounded experiment: postwarp intensity NGF

The fixed released-tissue-mask experiment did not improve aggregate mean TRE,
so this experiment retains the original raw-grayscale fixed support. The
coordinator approved this exact card and one paired all-three-case analytic/F2
pilot. Targeted searches across current tools/source/tests, coordinated documents
and Phase VI/VII/digital-topology archives found no existing NGF implementation
before the isolated module and objective hook added for this experiment.

Choose one conventional squared-dot NGF data term over a new coupled discrete
matcher. The official [FAIR NGFdot implementation](https://raw.githubusercontent.com/C4IR/FAIR.m/master/kernel/distances/NGFdot.m)
forms spatial finite differences of the already-warped intensity vector, then
differentiates those difference operators. Thus transporting precomputed moving
gradient channels would implement another functional. FAIR's numerator has no
epsilon addition; choose that exact variant explicitly. The histology pipeline
of [Lotz, Weiss and Heldmann](https://arxiv.org/pdf/1903.12063) supports NGF as an
established histology cue, but uses a different epsilon-augmented formula,
curvature regularization and L-BFGS. Our bounded data-term substitution would NOT
reproduce their complete method or inherit its empirical claims.

At image side s and normalized fixed pixel centers q, compute

    w_s(Y) = bilinear_zero(I_m,s, A f_Y(q)+b),
    a = G_s I_f,s,             b_Y = G_s w_s(Y),
    A_i = ||a_i||^2 + eps_f,s^2,
    B_i = ||b_Y,i||^2 + eps_m,s^2,
    c_i = a_i dot b_Y,i,
    D_NGF,s(Y) = sum_i m_s,i [1-c_i^2/(A_i B_i)] / Z_s.

Here G_s is the central difference s/2 times the two-neighbor intensity
difference; use the first-order one-sided difference s times the adjacent
difference at array endpoints. Its units are intensity per normalized canvas
length. Images remain in [0,1]. The denominator Z_s=sum m_s is fixed. Sampling
uses the ORIGINAL moving intensity once through the complete map: no affine is
applied twice and no second interpolation through an affine-prewarped raster.
The postwarp gradient is automatically in the fixed image frame. It is a
finite-difference gradient of raster observations, not an exact P1 derivative
of image intensity. The map still has its original exact P1 representation.

One explicit, image-only edge rule, fixed before any trial:

    eps_f,s = max(s/255, .1 * sum_i m_s,i ||G_s I_f,s|| / Z_s),
    eps_m,s = max(s/255, .1 * sum_i m_s,i ||G_s w_s(identity)|| / Z_s).

The floor corresponds to one 8-bit grayscale step per raster pixel, converted
to the same normalized-coordinate derivative units. The .1 factor is ONE
declared engineering preset, not a value inferred from evaluation landmarks or
a claim to reproduce another paper's edge parameter. Compute both from frozen
inputs/initial affine and never differentiate or update them with the candidate.
Separate fixed/moving thresholds reduce sensitivity to stain contrast scale.
Publish their actual values; do not tune them after TRE is read.

The weighted pointwise derivative with respect to b_Y is

    d_i = -2 m_s,i/Z_s [c_i a_i/(A_i B_i)
                       - c_i^2 b_Y,i/(A_i B_i^2)].

Then dD/dw = G_s^T d. Continue through the bilinear sampler, the frozen affine
transpose A^T and the existing source-P1 interpolation transpose. No derivative
through the warped intensity or its spatial gradient is detached. This is an
ordinary first-order VJP; constructing a dense Hessian is unnecessary.

This cue is exactly invariant to reversing either gradient's sign. Its values
are in [0,1] in exact arithmetic, but finite epsilon means identical nonzero
gradients generally have nonzero loss. Flat regions have value 1 and zero
image gradient; a contrast-normalization floor does NOT prove true-map
optimality. Locally sharpening a warped edge can increase normalized gradient
magnitude, so the unchanged geometric prior remains important. NGF supplies
different orientation evidence; it does NOT by itself enlarge the capture range
or guarantee correction of the persistent 53-pixel outlier.

Smallest test: first check literal stencil/transpose, a generic full-map finite
difference, positive/negative contrast, constant images and an exact known
rotation/shear frame fixture. Then one paired all-three-case analytic/F2 run
with the original raw-grayscale fixed-mask support, frozen affine and points, ARAP3,
shape1e-4, match.1, existing stage OOB, rates, 32..512 continuation and 300
gradients. Replace ONLY the data functional; final prefix selection uses the
complete NGF512 objective of that arm. Keep native DHR as an archived comparator.
No manual labels enter prediction or epsilon calibration, no sweep, and no new
geometry operator. Per-case mean/p90/maximum plus cost decide whether this exact
recipe survives; a negative result remains a negative result.

Implementation: `tools/coordinated_ngf.py` contains the explicit stencils and
frozen edge thresholds; `Evidence(..., loss="ngf")` samples the original moving
intensity before differentiating the warped raster. The existing paired pilot
accepts `--data-term ngf --image-objective continuation`. Its default MIND arm is
unchanged. Relative to the archived shared-affine MIND control, both the data
functional and descriptor/sampling order differ; this is not a pure optimizer
or geometry ablation. No conclusion is drawn before the six-case pilot results.

Why not coherent matching next: the previous capture prefix used 81 INDEPENDENT
labels on a 128-square feature raster, only +/-16 full-canvas pixels. The
[ConvexAdam paper](https://arxiv.org/pdf/2112.03053) instead couples a dense cost
volume with repeated spatial smoothing/displacement penalties before refinement.
The earlier negative prefix does not refute that approach. However, a meaningful
2D implementation adds a coupling objective, search range and continuation
calibration, and its candidates still face our current full-objective acceptance
(which rejected every earlier independent-label prefix). It is a larger next
experiment with no present evidence that our remaining errors are chiefly search
range. A stronger common image-only affine is also legitimate in principle, but
the existing full-affine correspondence fit already gave mixed held-out support;
no DHR dense field should be imported as a shortcut.

### NGF outcome: retire this exact recipe

All six NGF predictions completed successfully. Equal-pair 512-canvas mean/p90
TRE is 4.060915/6.914149 for analytic and 3.982225/7.187874 for F2, versus the
fresh shared-affine MIND control's 3.548752/5.907957 and 3.554327/5.942943.
This is negative anatomical evidence for the tested recipe, not a theorem about
NGF. No epsilon sweep follows. Reports are `miit_ngf_t19/landmark_scores.json`
and `miit_multiscale_control_t19/landmark_scores.json` in the existing outputs.

## Approved card: ARAP-stiffness physical-fiber optimizer

Question: can an objective-aware, globally coupled descent direction resolve
the current corrected-MIND optimizer's finite-budget limitation? Fresh control
traces put the best trial at step30 in29/30 analytic directional stages; the
remaining stage chooses29. All six finest stages choose30, still reducing E512
by approximately .0028--.0033 per30steps. This does NOT establish that further
objective improvement will improve anatomy. Old extra-iteration/physical-rate/
L-BFGS/filter negatives used other configurations and do not settle this case.

Exact unchanged objective at stage raster s:

    E_s(Y) = shared_affine_MIND_s(Y) + 3 ARAP_P1ac(Y)
             + 1e-4 Shape(Y) + .1 Matches(Y) + OOB_s(Y).

Retain raw fixed support, frozen inputs/affine/matches, 257-square actual output,
17/33/65/129/257 coefficient levels, 32/64/128/256/512 raster continuation,
fixed identity boundary and full512 prefix selection. At a directional stage,
fix Y0 and optimize ACTUAL physical coefficients c in Y(c)=Y0+(P_l c)e, with
the same zero-boundary bilinear prolongation P_l, rather than Adam on a scaled
raw latent. There is no graph through optimization history. Let Q0 be all
normalized corner determinants and eta=.001. Preserve the original analytic
stage's contracted feasible set, Q(Y(c)) >= eta+.05*(Q0-eta). This is a linear
polytope in c; its strict interior is searched, not a larger deformation class.

Mechanism: g=dE_s(Y(c))/dc includes ALL original terms. Use

    H_l = 3 P_l^T K_f P_l,        d = - H_l^{-1} g.

K_f is the scalar Dirichlet five-point stiffness on the fine interior, diagonal4
and horizontal/vertical neighbor entries-1. For the actual equal-area P1-ac
energy .5*mean_faces||J-R*(J)||^2, freezing each current proper rotation gives an
upper quadratic touching ARAP in value and first derivative. Its scalar Hessian
is EXACTLY K_f; the coefficient Hessian is P_l^T K_f P_l. H_l is therefore the
existing weighted ARAP majorizer curvature, not the full image Hessian and not
an additional regularization penalty. Fixed boundaries make it SPD, hence
g^T d=-g^T H_l^{-1}g<0 for nonzero g in exact arithmetic. At nonsmooth L1 or
sampling knots an AD subgradient does not by itself guarantee directional
decrease; actual complete-objective acceptance remains mandatory.

There is a cheap EXACT separable inverse for these particular aligned grids.
Write n=l-1, r=256/n, and T=tridiag(-1,2,-1) on n-1 interior entries. The 1D
prolongation B satisfies S=B^T T_f B=T/r and M=B^T B with diagonal
(2r^2+1)/(3r) and off-diagonal (r^2-1)/(6r). Thus

    P_l^T K_f P_l = S tensor M + M tensor S.
    theta_k=pi*k/n, k=1,...,n-1,
    s_k=(2-2*cos(theta_k))/r,
    m_k=(2r^2+1)/(3r)+(r^2-1)*cos(theta_k)/(3r),
    eigenvalue_ij(H_l)=3*(s_i*m_j+m_i*s_j).

Orthonormal 2D DST-I diagonalizes H_l, implemented by odd-extension FFTs.
No sparse assembly, iterative linear tolerance, learned metric, screening
parameter or edge-learning-rate calibration is needed. It IS a global linear
preconditioner in the instance optimizer; do not call the whole pipeline
global-solve-free. The topology construction itself is unchanged. This differs
from four local averaging passes and from five-pair scalar-metric L-BFGS.

Step rule: from the current c, compute exact corner change v for P_l d and
slack b=Q(Y(c))-[eta+.05*(Q0-eta)]. Set amax=min_{v<0} b/(-v), infinity if empty,
and a0=min(1,.99*amax). Backtrack a=a0/2^j, j=0,...,12, until
E_s(Y(c+a*d)) <= E_s(Y(c))+1e-4*a*g^T d and all actual rounded contracted
corner/boundary checks pass. Form EVERY trial directly from fixed Y0 and c+a*d.
Update c only on acceptance. Exhaustion ends this stage with its best feasible
iterate and an explicit reason; it is not a stationarity certificate. Each
stage permits at most30 gradients; report actual gradients, accepted steps,
backtracks, scale bounds and full objective calls. No hidden numerical jitter,
post-hoc map repair or landmark-selected stopping is allowed.

Smallest decisive tests: independently assemble tiny fine P1 frozen-rotation
Hessians and prolongation matrices, compare against the spectral operator and
inverse for r=1,2,4 and odd/even interior sizes; verify value/gradient touching,
SPD/descent, boundary constraints, a deliberately active affine constraint and
line-search rejection. Then ALL THREE fixed MIIT cases: archived fresh Adam300
reference, fresh corrected-objective Adam900 (90 gradients per directional
stage), and this stiffness method with at most300 gradients. The latter two
produce six new predictions; score only after all terminate. All arms share
the objective/inputs, but gradient budgets and forward-only evaluations differ;
report complete-call cost and accuracy, not an unmeasured trajectory or equal
compute. Existing
F2 remains context, not a new geometry ablation at this decision point.

What would falsify usefulness: complete-E gains fail to exceed ordinary longer
Adam at comparable elapsed time, or lower E again yields worse aggregate/tail
anatomy. That outcome narrows the diagnosis toward this objective's anatomical
tradeoff, without proving it impossible. Repeated active-face exhaustion instead
exposes a different constrained-optimization limitation. Do not retune ARAP,
introduce a cost volume simultaneously or sweep preconditioner parameters.

Prior work: [Sorkine and Alexa's ARAP paper](https://igl.ethz.ch/projects/ARAP/)
establishes conventional rotation/quadratic global updates; our exact scalar
uniform-grid stiffness identity above must be checked for THIS implementation.
[Mang and Biros](https://arxiv.org/abs/1604.02153) study preconditioned registration
in a different velocity/PDE formulation. Neither source supplies a theorem of
global convergence or anatomical success for this piecewise-smooth finite-grid
optimizer. The coordinator approved this one mechanism, with independent
formula/unit checking before the three real stiffness runs. The separate
corrected Adam900 control can run while the new implementation is checked.

### Actual fixed test: stiffness does not accelerate this objective

All six new predictions completed before any manual scoring. Each stiffness run
used300gradients and300accepted steps without early stops; all exported maps
passed the actual topology checks. All three methods below use the same corrected
shared-affine MIND objective and the same previously viewed three-case specimen.

|Analytic optimizer|Equal-pair mean TRE (512px)|Mean pair-p90 (512px)|Complete objective calls per case|
|---|---:|---:|---:|
|Archived corrected Adam300|3.54875190|5.90795739|332|
|Fresh corrected Adam900|3.55935383|5.90498132|932|
|Stiffness300|3.58415413|5.93691899|1877 / 1967 / 1886|

In pair order2-to3,7-to8,10-to11, final stiffness E512 is
.2405317303/.2221633084/.2419423997, WORSE than Adam300's
.2397436816/.2214678987/.2415485892 and Adam900's
.2382588491/.2201690697/.2400392129. Stiffness complete calls take9.27--11.26s,
versus11.48--12.80s for Adam900 and4.1--6.1s for the archived Adam300 calls.
These are serial screening calls, not paired-repeat timing confidence intervals
or time-stamped convergence curves. Stiffness allocated peaks are about142MiB.
It achieves neither better complete energy nor better aggregate anatomy than
Adam300. Its final ARAP is lower, but BOTH MIND and machine-point losses are
higher on every pair. This is not an anatomical regression at a better optimum.

The trace identifies the numerical mechanism: all900initial alpha values are1;
NONE is limited by the geometry bound. Minimum contracted slacks remain at least
.618/.712/.642. Yet1265/1355/1274objective backtracks are needed. At level257,
median accepted alpha is1/128,1/128,1/64 respectively; median raw RMS proposals
are7.17/9.60/5.76canvas pixels, reduced to median accepted RMS steps of
.064/.078/.060pixels. Medians use the mean of the two middle values for each
even60-step sample. The geometry theorem or its normalization is not the
observed bottleneck.

More precisely, for each stored accepted step define the DIRECTIONAL SECANT
remainder ratio

    R(a)=2*[E(c+a*d)-E(c)-a*g^T*d] / [a^2*(-g^T*d)].

Here -g^T*d=d^T*H*d; the frozen-rotation weighted-ARAP upper quadratic contributes
at most1 to this ratio in exact arithmetic. The finest-level medians are
149.34/179.89/118.49, so the omitted non-ARAP terms dominate the finite-step restriction.
This is NOT a Hessian eigenvalue or a smoothness proof: L1 and interpolation
knots matter. An additional read-only x-direction probe at each saved final map
(not a replay of the stored path), using a=1/128, separates the same ratio into
MIND164.48/227.56/205.04, weighted matches4.11/5.72/5.96, weighted ARAP about.75,
and OOB/shape below.0003. Those three finite-step probes strongly implicate the
transported descriptor term's local variation; they do not prove a universal
curvature model or failure of the feature itself.

Decision: retire this exact stiffness recipe without an alpha/weight/screening
sweep. A data-Jacobian-aware Gauss-Newton/prox-linear method would be genuinely
different from both this fixed ARAP metric and the earlier scalar-initialized
five-pair L-BFGS; current results do NOT refute it. However, corrected Adam900
already lowers all three complete objectives with mixed/slightly worse mean
anatomy, so another local solver is not the highest-information NEXT mechanism.
Proceed to formulate ONE spatially coupled finite-displacement matching/search
experiment, explicitly distinct from the old81independent labels and their
immediate full-E rejection. No coupling implementation is yet authorized by
this result paragraph. Exact reports remain under `miit_adam900_t20` and
`miit_stiffness300_t20`; predictions and the subsequent scoring are separate.

## Approved card: coupled finite-displacement seed, then original refinement

Question: does a spatially coupled finite-displacement search provide a useful
alternative basin for the ORIGINAL corrected-MIND registration functional?
This is one image-only seed followed by the original analytic300 refinement,
not a multiseed search or another stiffness/step-size variant. Cost centers are
the IDENTITY residual after the SAME frozen positive affine, not the Adam300
output or any DHR field. No manual target affects labels, coupling or selection.

Exact data: use the existing frozen shared-affine MIND descriptors at128-square
resolution, still eight-channel L1 (NOT squared feature distance). On the63x63
INTERIOR vertices q_i=(j/64,k/64) of a65-square proposal grid, use nine offsets
xi in{-1,0,1}^2/128. The label set is L={-16,...,16}^2, zero first then fixed
lexicographic ordering. Label k denotes residual displacement k/128: four
512-canvas pixels per step, +/-64pixels per aligned-residual axis. Original
moving displacement is A*k/128, so this is not a native-moving-pixel radius.
Features are bilinear/zero sampled with align_corners=False. Let

    m_ij=m128(q_i+xi_j),        w_i=sum_j(m_ij)/9,       Z=sum_i(w_i),
    h_ij(k)=mean_channels|phi_f(q_i+xi_j)-phi_m,shared(q_i+xi_j+k/128)|
             + ||relu(-v_ij)+relu(v_ij-1)||^2,
    v_ij=A*(q_i+xi_j+k/128)+b,
    C_i(k)=sum_j m_ij*h_ij(k) / sum_j m_ij.

For an empty patch set C_i=0; require Z>0. All mask samples/denominators are
fixed before label enumeration. There is no label-dependent overlap dropping,
moving-mask gating or confidence selection. The OOB coefficient is the original1
in normalized original-moving coordinates. Boundary labels/displacements are
exactly zero and are not optimization variables. Empty interior patches still
participate in UNWEIGHTED spatial coupling; only their image weight is zero.

Coupling: z_i is a discrete two-component label; u_i is a continuous two-component
shift in the SAME128-raster pixel units. Let P65 be the same zero-boundary
bilinear RAW prolongation to257 vertices and G=P65^T*Kfine*P65 (unweighted).
For c in the fixed schedule(.003,.01,.03,.1,.3,1), use the surrogate

    J_c(z,u) = [sum_i w_i*C_i(z_i) + c*sum_i ||z_i-u_i||^2]/Z
                + 3/(2*128^2) * sum_components u_component^T G u_component.

The last term is the existing ARAP3 frozen-identity-rotation quadratic in UNIT
displacements u/128, not an extra fitted smoothing weight. Both components use
the same positive Galerkin stiffness. The proximity term is on ALL interior
nodes, WITHOUT w_i; weighting it would create a variable diagonal and invalidate
the following separable solve. Each alternating block is exact in real arithmetic:

    z_i = first_argmin_{k in L} [w_i*C_i(k)+c*||k-u_i||^2],
    [2*c*I + (3*Z/128^2)*G] u_component = 2*c*z_component.

Use the already independently checked Galerkin DST-I eigenbasis for this
screened linear solve; never call it solve-free. Initialize z by image-only
first_argmin(w_i*C_i); obtain u by the screened solve at c=.003. Then perform
EXACTLY TWO label/continuous alternations at each of the SIX coupling values,
in that order, and retain the final u only. At a fixed c the exact block steps
cannot increase J_c; different-c values are different objectives, not a common
monotone trace. Discrete labeling makes the joint problem NONCONVEX. This is
related to [ConvexAdam's coupled search](https://github.com/multimodallearning/convexAdam),
whose released schedule motivates these six constants, but replaces its local
averaging with a physically specified global quadratic solve. It is NOT a
ConvexAdam reproduction, globally convex solver or new correspondence theorem.

Always-legal construction: u is an IMAGE-DERIVED PROPOSAL, not a certified map.
Form the frozen fine desired displacement d*=P65(u/128), with exact zero
perimeter. Starting at identity, perform EIGHT fixed x/y construction cycles.
At each coordinate step, the raw amplitude is the corresponding component of
identity+d* minus the current legal map. Apply the existing global ANALYTIC
coordinated update with eta=.001 and theta=.95, refresh geometry after that
accepted construction step, and check every rounded intermediate corner/boundary.
There are16safe coordinate steps, no objective gradients, map resampling,
blending of accepted maps, DHR initialization or unsafe-map repair. The requested
target remains fixed; there is no landmark-dependent stopping or alternate seed.
Multiple cycles allow x/y geometry to readjust, but do NOT prove the target
reachable. A bad local target can still limit global progress; this must be
measured, not hidden by a topological-success claim.

Report raw target corner-fold count diagnostically, every construction scale,
raw displacement RMS/max, decoded displacement RMS/max, raw-to-decoded RMS
distance, seed MIND/ARAP/match/OOB/full-E, and actual final seed validity. Only
the constructed legal seed is used. Large contraction or a poor seed is retained
as the actual outcome, not grounds for a second parameter choice or silent
identity replacement. Construction failure is an explicit prediction failure.

Acceptance change is EXPLICIT: this alternative initialization may have HIGHER
original E512 than identity; do NOT repeat the old immediate full-E rejection of
the seed. The surrogate selects the one initialization; it is not advertised as
an E512-descent optimizer step. Starting there, run the unchanged300-gradient
analytic schedule, original shared-affine MIND/ARAP3/shape1e-4/matches.1/OOB,
and original complete-stage-E acceptance. Select the final output by ORIGINAL
E512 over identity, the legal seed, and all accepted stage prefixes. No cost
volume term replaces or augments the final objective. This changes finite-search
initialization/protocol, not the final data functional, and is not an isolated
geometry ablation. Selection does not include an extra baseline-Adam trajectory.

Budget and decisive test: cost volume has3969*1089=4,322,241 entries (~33MiB
in float64), with nine patch samples each and eight descriptor channels. Build
in fixed16-label batches; do not allocate all label/sample/channel tensors at
once. Coupling adds13screened solves and13label minimizations; construction
adds16safe steps before the original300gradients. Report cost construction,
coupling, legal construction, refinement, complete-call time, actual objective
counts and CUDA allocated peak separately. Each of the13screened solves handles
TWO components (26scalar systems total). No equal-compute claim follows.

First tiny independent checks: literal label cost/OOB/zero-tie and empty patches;
nonconstant fixed masks without denominator changes; dense versus spectral
screened solve and decrease of both blocks at fixed c; exact label units under
a rotated affine; deliberately folding raw proposals with all16 constructed
states still legal; unchanged original Evidence and final prefix selection,
including a seed with higher E than identity. Then run ALL THREE fixed MIIT
cases once, analytic only. Compare existing corrected Adam300 and Adam900,
all available labels, mean/p90/max per case and equal-pair aggregates plus cost.
Read labels only after all three new attempts terminate; no F2 rerun yet.

What redirects the work: absent useful final accuracy/cost or complete-E gains
retires this exact recipe without range/coupling sweeps. Lower E with worse
anatomy again identifies a remaining objective-anatomy tradeoff, not a topology
failure. Strong raw-to-decoded contraction isolates an initialization-construction
limitation and does not refute finite search. Only a measured useful result merits
an independent specimen/paired-cost confirmation. The coordinator approved
these exact details for one implementation and all-three-case test. The final
case archive additionally retains the legal initializer vertices and raw65-grid
pixel coefficients for inspection; only the ordinary final vertices are scored.

Implementation check, 2026-10-02: the new cost/coupling/construction module,
minimal optional initializer hook, and three-case runner passed16author tests;
the same run including affected shared-affine/stiffness suites passed69tests.
The independent checker used a NumPy bilinear cost and separately assembled
FE/Galerkin matrix: costs agreed to1.11e-16(double) and1.44e-8(mixed-float32),
the complete13solve/13label sequence to8.89e-16, and all fixed-c block energies
to4.45e-16. A folded raw target produced16legal steps; a more extreme fixture
triggered the declared rounded-construction failure instead of an unsafe export.
A higher-original-E injected legal seed was still the first refinement anchor,
with identity retained for final selection and19actual/reported calls including
one seed evaluation. These are correctness checks, NOT anatomical results.

### Actual three-case coupled result: legal, negligible accuracy benefit

All three frozen predictions completed on AI GPU6 after the08:18:41UTC launch,
before labels were scored by the coordinator. Each completed300gradients and
333original-objective calls, selected final stage9, and passed the saved-map
strict floor. Equal-pair mean TRE/p90 were3.54760908/5.91743723canvas pixels,
versus original shared-affine Adam300's3.54875190/5.90795739: mean differs by
only-.00114282pixels and the tail is worse. No useful accuracy gain is established.

| Direction | A300 final E | Coupled final E | Seed E vs identity E | Final corner floor | Seed / complete seconds |
|---|---:|---:|---:|---:|---:|
| 2-to-3 | .239743682 | .239860450 | .276476748 / .273036889 | .342235 | .399 / 7.204 |
| 7-to-8 | .221467899 | .221461896 | .255957573 / .254375348 | .345279 | .306 / 4.555 |
| 10-to-11 | .241548589 | .241559266 | .292362426 / .296359203 | .305014 | .301 / 4.484 |

All48construction scales were exactly1 and every raw target was already
four-corner positive. Raw/decoded RMS displacements were identical at
1.67227/1.59400/2.64247aligned512-pixels, with maxima5.64334/5.65480/7.70939.
Thus this experiment was NOT limited by safety contraction. The first two
higher-E seeds were genuinely refined, not replaced by identity. Final maps
returned close to ordinary Adam300 (residual-map RMS differences
.21655/.07960/.08267pixels). Adam900 has lower original E on all three cases;
this new seed does not materially improve the final functional either.

Decision: retire this exact coupled seed without radius/coupling/weight sweeps.
Its negative outcome does not refute all discrete matching, but neither the
global prior inverse nor this alternative seed improves the established
accuracy/cost tradeoff. Move the next bounded hypothesis toward a demonstrably
different anatomical evidence source; do not declare topology the bottleneck.
Reports and the three exported maps are in `miit_coupled_seed_t20`.

## Approved card: contrast-calibrated fixed hematoxylin-proxy evidence

Question: does separating a biologically shared brightfield stain proxy give
the EXISTING safe instance optimizer more useful anatomical evidence than its
inverted RGB grayscale, across the available cohorts? This is the ONE next
evidence mechanism proposed after the stiffness and coupled-seed failures; do
not combine it with a new optimizer, feature model, seed, mask or parameter sweep.

Adjudication: changing a descriptor's coordinate frame was helpful on the
large-rotation MIIT sample, but worsened mean accuracy on the other three known
specimens. Calling shared-affine MIND a universal correction is therefore false.
Nevertheless, fixed shared-affine controls now exist for all25directions, so
use that SAME frame for this matched evidence test; do not select a frame per
cohort using its labels. Native DHR is a distinct pipeline: the kidney native
initializer is already better than the common affine, while lung nonrigid
performance is substantially worse and Histo has nearly matched initialization.
Those facts do not isolate a universally superior native nonrigid mechanism.

Why this before the other candidate mechanisms: a fixed stain proxy directly
tests a recognizable common biological signal in HE/IHC and IHC/IHC images,
using existing RGB inputs and a tiny deterministic transform. By contrast,
[DINO-Reg](https://papers.miccai.org/miccai-2024/paper/2230_paper.pdf) establishes
a frozen-feature optimizer for3Dmedical modalities, not histology; its ViT patch
resolution, upscaling and feature reduction introduce substantial new choices.
The [pathology deep-matching paper](https://arxiv.org/abs/2208.07655) supports
image-derived correspondences with outlier handling, not an already validated
drop-in dense feature functional or hard-topology guarantee. Both remain
possible later routes, not grounds for assuming immediate transfer. Faithfully
transplanting the DHR NCC pipeline would jointly change normalization, pyramid,
similarity, deformation regularization and potentially initialization; it is
useful engineering but a less isolated next test of this specific evidence gap.

Exact fixed transform, applied to each existing512-square RGB canvas BEFORE
the ordinary scalar-image pyramid or any affine prewarp: let I=uint8_RGB/255
as stored, with channel order R,G,B. Define row-vector optical-density proxy
d=-log(max(I,1e-6)), and

    B = [[.65,.70,.29], [.07,.99,.11], [.27,.57,.78]],
    c = d @ inverse(B),
    c_H = max(c[...,0],0),
    H = -expm1(-c_H) = 1-exp(-c_H),
    s_raw = quantile_linear(H[original_gray_support],.99),
    s = max(s_raw,1e-6),
    H_feature = clamp(H/s,0,1).

B's ROWS are H/E/DAB absorbance directions; concentrations multiply B on the
LEFT, hence the displayed right-multiplication by its inverse. Compute the
fixed3x3 transform in float64 and cast the resulting normalized image once to
the original float32 image precision. Both H and H_feature are in[0,1], high
for dark H signal, white at0. The quantile uses linear interpolation between
sorted order statistics, EACH input's own ORIGINAL gray>.04 support, and its
512canvas only. Empty support gives s_raw=0; the old fixed-empty-support handling
still applies. The same formula governs every image: allzero H stays zero;
sparse positive H with q99=0 uses s=1e-6 without a new failure or raw-gray
fallback. Record numerical-floor activation. Freeze s before the pyramid and
affine prewarp; never renormalize at each level. Clamp negative concentrations
to0; do not use absolute values, stain-vector fitting, additional histogram or
min-max matching, gamma linearization, fitted white balance, adaptive recipe
selection or a learned model.
The RGB lower floor prevents infinite logs on saturated zeros; count such
samples diagnostically, never drop them. No alternative treatment is selected
from registration scores.

Prior work/source: [Ruifrok and Johnston2001](https://pubmed.ncbi.nlm.nih.gov/11531144/)
introduced color deconvolution from stain-specific absorption, with limitations
including saturation and stain interactions. The B values and orientation are
the published HED convention in the
[official scikit-image implementation](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/color/colorconv.py).
Its implementation scales logs by -log(1e-6); the formula above deliberately
uses unscaled natural OD, then a bounded darkness transform. Thus H equals
1-exp(-log(1e6)*rgb2hed(I)[...,0]), not raw rgb2hed output. This fixed darkness
choice keeps an interpretable0..1 raw darkness range, not exact physical OD.

Units adjudication: range0..1 by itself does NOT give comparable effective
contrast to inverted grayscale. The descriptor's1e-4 stabilizer has squared
intensity units; direct low-amplitude H would artificially increase its relative
effect. Therefore the ONE predeclared robust scaling above sets each input's
99th-percentile H signal to one before any registration, unless the numerical
floor activates. MIND epsilon remains1e-4 in these normalized-contrast units:
a contrast stabilizer, NOT a measured physical noise variance. Its effective
scale is approximately1%of robust full intensity range. Scaling is image-adaptive
through this SINGLE fixed formula, not a per-case rescue chosen after outcomes.
It can amplify weak stain or noise and clips the high tail; no universal
accuracy/noise improvement follows. The same q99/floor rule applies even to
the known weak-counterstain image.

Exact claim: under the IDEAL RGB absorption model I=exp(-c_true@B), nonnegative
concentrations and no active RGB floor, the recovered H channel depends only
on c_true,H and is invariant to E/DAB concentrations. The normalized feature
has the same invariance only when its calibration support is held fixed; a
gray-derived support can itself change when other stains change. That is an
algebraic property of this model, NOT a theorem about real slides. Actual encoded sRGB,
white illumination, stain variability, coarse pixel averaging, sequential
sections and stain-vector mismatch violate its hypotheses. Call the output
a hematoxylin PROXY, not measured concentration or guaranteed shared anatomy.
No improvement of TRE, contrast invariance beyond that model, registration
global optimum, or feature-resolution sufficiency is claimed in advance.

Unchanged registration: preserve the ORIGINAL inverted-gray>.04 fixed mask
and its per-level area averages/denominators; never threshold H to define
support. Keep original images for the frozen raw machine matches, original
positive affine, identity residual start, shared_affine transported8-channel
MIND, image weight1, ARAP3, shape1e-4, matches.1, OOB1, P1ac257, exact boundary,
eta.001, all original five levels/rates,300Adam gradients, original stage
acceptance and best-full selection. This is a different IMAGE functional;
its complete-E values are not comparable as same-objective convergence numbers
to raw-gray E. Do not retain coupled initialization or add a baseline trajectory.

Required diagnostics, all reporting-only: on each unwarped512input, record
pre-clamp negative-H fraction, RGB-floor fraction, H zero fraction, raw H range,
s_raw, s, numerical-floor activation, and H_feature saturation fraction within
that input's original gray support. At each ordinary image-pyramid scale record
the NORMALIZED-H MIND local variance median (ordinary linear median over
pixels with positive support weight) and the mask-WEIGHTED fraction at or below
the unchanged1e-4 epsilon, using the corresponding static gray-mask area
average. Moving-input diagnostics use its OWN original gray
support only for reporting; they never gate moving overlap or the fixed loss.
Label these as unwarped-channel diagnostics, not bounds on deformed descriptor
support. Constant/near-flat H remains in the experiment with its diagnostics;
there is no raw-gray fallback, mask removal or parameter rescue.

Adverse pre-implementation evidence: an image-only direct calculation on the
six MIIT canvases found H local variance medians .006625--.017541 on five, but
the strongly pink miit7moving canvas had H zero fraction .687592 and median
6.2185e-5, below the fixed1e-4 descriptor epsilon (its raw-gray median was
.00381798). Its H99was.127710. These were RAW-H diagnostics before the approved
contrast calibration. The universal normalization boosts this image by about
7.83 before upper clipping, but its .687592 zero fraction cannot acquire
information from scaling. This is a known risk of weak counterstain or matrix
mismatch, not evidence for discarding the case or selecting another recipe.

Smallest decisive tests before production: literal matrix orientation and
pure-H/E/DAB plus mixed Beer--Lambert fixtures, H invariance when only E/DAB
changes at fixed calibration support, white and saturated/zero RGB, negative
recovered H, exact linear-quantile scale, sparse-positive q99zero/floor and
allzero H, boundedness and finite float32 output; unchanged raw-mask values
and pyramids; bitwise default
raw-preprocessing Evidence/path; independent frozen-affine coordinate/order
check; the existing complete300budget and saved-map certificate. An independent
checker must use a separate scalar/matrix calculation, not call this transform
as its expected answer. The known weak-H case must not acquire a fallback.

Decisive real test: all25existing directions once with this ONE fixed recipe:
20correlated lung directions, one Histo CD4/CD68 direction, one kidney
HE/PanCytokeratin direction, and the three known MIIT directions. Compare each
against its already computed shared-affine raw-gray analytic300 control. Keep
per-cohort mean/p90/max and costs; twenty lung directions are ONE specimen,
and the three MIIT directions are ONE other specimen. A per-direction pooled
average is not a patient-level result. No new matcher, competitor field or
manual calibration is allowed. All predictions must terminate before labels
are scored; failures retain their original denominators.

What falsifies/redirection: no meaningful accuracy/cost gain or worsening
across cohorts retires this fixed proxy, without nearby matrix/scale/epsilon
sweeps. An isolated win is a specimen-specific lead, not universal improvement;
a weak/flat channel shows a proxy limitation, not a topology defect. If useful
across cohorts, confirm with a separate specimen before any superiority claim.
The coordinator approved this EXACT contrast-calibrated recipe before any
H-proxy registration was run. Implementation is handed to a separate
Astra-medium builder; this card's author does not implement it. An independent
high-effort checker reviews the resulting transform and application protocol.

### Actual H-proxy result: no useful general gain

The Astra-medium implementation is tools/coordinated_stain_proxy.py plus a
minimal preprocessing hook; the raw default is unchanged. Eight focused tests
and the affected author suite56 pass; root's overlapping suite61 passes.
Independent scalar/cofactor, linear-quantile, raw-mask and separate affine
sampler checks passed before production. All25 AI GPU1 predictions completed
in121.2556s after the08:46:46UTC launch and before either scorer read labels.
Each used300gradients/332objective calls and zero failed trials. Final exported
minimum corner ratio across cases is.0068614954, strictly above.001.

| Specimen | Shared-gray mean / p90 | H-proxy mean / p90 |
|---|---:|---:|
| MIIT,3 directions | 3.54875190 /5.90795739 | 3.56822147 /5.94232848 |
| Lung,20 directions | 4.56854112 /9.59367916 | 4.56755994 /9.63640989 |
| HistoReg | .85190347 /1.58600541 | .92517151 /1.66460296 |
| Kidney | 2.36486646 /4.75690671 | 2.35532726 /4.55294922 |

Units are512moving-canvas pixels; the first two rows average within-specimen
direction statistics. Kidney improves its tail and slightly its mean; lung
is nearly neutral in mean with a worse tail, while MIIT/Histo worsen. Retire
this exact proxy as a general improvement without a stain-matrix/scale/epsilon
sweep. This does not refute all stain-aware features or show raw gray optimal.
H and gray E totals are different functionals, not convergence comparisons.

The weak7moving raw H q99=.1277100953 and zero fraction=.687592053 are preserved.
The declared normalization gives512 variance median.00358927669 and weighted
epsilon-incidence.27303326. Thus this implementation did not simply leave the
weak image's global contrast below the unchanged descriptor stabilizer. It also
could not create missing counterstain information. These are unwarped diagnostic
statistics, not deformable-matching accuracy theorems.

Complete per-case calls4.2049--5.8584s include H preprocessing/diagnostics but not
frozen initial-affine/matcher preparation; allocated peaks208002560--213791744
bytes. Independent postrun checks recompute all2074 errors (maximum discrepancy
1.14e-13canvas pixels), all25 boundaries/corners/budgets/selection and all15unique
input H/mask calibrations. Outputs and paired comparison are in
stain_proxy_all25_t21. No failed attempt or label is excluded.

## Proposed next card: frozen cross-modality ELoFTR correspondences

Status: coordinator approved this exact card after bounded Astra-high selection.
The inference adapter and one real512smoke are complete as recorded below;
the full25production experiment still awaits independent integration review.

Question: can pair-conditioned, detector-free correspondences pretrained for
cross-modality matching improve the present safe registration, where changing
local scalar evidence and its optimizer repeatedly produced little useful gain?
Choose ONE mechanism: the released **MatchAnything ELoFTR** checkpoint replaces
the existing frozen SuperPoint/SuperGlue point table. Retain shared-gray MIND,
the frozen positive affine, ARAP3, shape1e-4, point weight.1, robust scale8pixels,
OOB1, identity residual start, P1-ac257, exact boundary, eta.001, original five
image/control levels, rates,300Adam gradients and best-full prefix selection.
Do not add Phikon/DINO features, a new affine fit, coupled seeding, point-only
optimization, competitor displacement supervision or network training.

Why this particular intervention: ELoFTR does not require repeatable detected
keypoints before matching; its descriptors are conditioned on BOTH images.
MatchAnything's additional cross-modality pretraining addresses appearance
transfer explicitly. This differs from another pointwise stain/intensity cue
and from exchanging one sparse assignment backend while keeping SuperPoint.
The primary [MatchAnything paper](https://arxiv.org/html/2501.07556v1), sections
2 and4.6.1, reports ANHIR cross-stain evaluation using predicted matches followed
by affine and B-spline fitting. That is relevant prior evidence, NOT a result
for our fixed affine,512rasters, P1map or specimen set. Its DeepHistReg baseline
is not the presently executed DeeperHistReg standard pipeline. We reproduce
neither its full registration pipeline nor its published accuracy claim.

Alternatives checked before this choice:

* Original-detail1024/257controls already gave only modest Histo/kidney changes:
  analytic means.793449->.756767 and2.351586->2.287044 in512-equivalent units,
  with Histo p90 worsening (PROGRESS, T+9.7h). This is not an untested next leap.
* Native normalization/CLAHE plus NCC7 was already tested; historical analytic
  means.794/3.936/2.945 versus native-preprocessed MIND.808/3.713/2.468 gave no
  consistent rescue. It did not reproduce the DHR pyramid/mask/boundary pipeline.
  Full native DHR now wins Histo/kidney but loses lung; thus transplanting all of
  its evidence/regularization is not an isolated, universally supported fix.
* Frozen DINOv2-S/14 weights ARE cached on AI,88283115bytes. The earlier
  `digital_dinov2_safe_optimize.py` used replicated grayscale448,32-square patch
  features and old17/33/65F1 with100steps. Archived kidney/lesion native-pixel
  means were9.477376/10.285254. These are not present-control comparisons or a
  decisive rejection of RGB DINO; neither are they evidence to repeat it blindly.
* [Phikon-v2's own model card](https://huggingface.co/owkin/phikon-v2) specifies
 20x H&E tiles and primarily tile-level downstream evidence. Our existing
  overview rasters do not restore that nuclear-scale information merely by
  being called histology. No demonstrated drop-in cross-stain registration
  advantage at this scale was found in the bounded check.

Exact input and coordinate contract: use each existing fixed/moving512 RGB
canvas, convert with PIL `convert("L")` and divide by255 to ordinary grayscale
in[0,1], white high. This follows the released ELoFTR input convention; our old
SG extractor instead used inverted luminance, so MATCHER preprocessing changes
as part of the model intervention and must be disclosed. Prewarp the moving
grayscale ONCE by the SAME saved affine using `warp_moving_to_fixed`: bilinear,
border padding, align_corners=False, fixed512pixel centers. No new affine fit,
grayscale normalization, CLAHE, stain transform, resizing or tissue segmentation.
The512square already satisfies the released multiple-of32 shape rule. Feed
`image0=fixed_gray`, `image1=aligned_moving_gray` in eval/no_grad FP32 mode.

Use the released `src/config/default.py` plus `configs/models/eloftr_model.py`,
including its coarse matching threshold.1 and actual MTD all-thresholded-pairs
policy after border removal; no additional confidence threshold. FORCE_NEAREST
is configured true but UNUSED by the pinned CoarseMatching class. Preserve that
released behavior, not an inferred mutual-nearest rule. Following the released inference adapter, set coarse NPE
to[832,832,512,512] for its megadepth position-encoding convention and actual
512input, while keeping FP16 disabled. Use MatchAnything's own LoFTR class and
checkpoint, NOT vanilla Kornia LoFTR with incompatible weights or default config.
Do not import or run its Gradio UI, training wrapper, ROMA or B-spline fitter.

Network output is `mkpts0_f`, `mkpts1_f`, `mconf`: floating pixel-index coordinates
in the fixed and aligned-moving rasters plus model confidence. Convert once by
q=(mkpts0_f+.5)/512, p=(mkpts1_f+.5)/512. Store original-moving target A*p+b only
through the existing affine convention; do not apply A twice. Do not confuse
the adapter's resize-ratio multiplication with a normalization to unit coordinates.
For512->512 its ratio is one. Matching itself returns point observations, not
a dense warp for the safe method to imitate.

Confidence/support/outlier policy: ANY nonfinite point/confidence output fails
the entire extraction before domain filtering. Retain finite network matches admitted
by the released threshold, with original confidence in[0,1]. Discard and count
only outputs whose q or p lies outside the declared[0,1]square; never clip them.
As in the existing point loader, assign zero eligibility to original-moving
targets A*p+b outside[0,1]^2. Use no new fixed/moving tissue filter, no global
RANSAC/homography, no distance-to-affine cutoff and no label-selected rejection.
The fixed image mask remains solely the existing dense MIND support. Report
point counts before/after eligibility, confidence mass, occupied4x4 source bins,
and fraction of sources on original fixed gray support as diagnostics, not
anatomical correctness estimates or selection criteria. Correlated semidense
matches are not independent observations. Normalize by total eligible confidence,
so a larger number of matches does not multiply point-loss strength.

The exact substituted point functional is the EXISTING ImageCorrespondences:

    P(Y) = sum_j w_j [sqrt(1 + ||(f_Y(q_j)-p_j) A^T *512/8||^2) - 1],
    w_j = c_j * eligible_j / sum_k(c_k * eligible_k),
    E(Y) = I_shared-gray-MIND(Y) +3 ARAP(Y) +1e-4 Shape(Y) +.1 P(Y) +OOB(Y).

Its bounded influence is the only extra outlier resistance in this first test;
it does not make false matches harmless. New and old E totals are different
functionals. If fewer than8positive eligible matches remain, declare extraction
failure and do not optimize that case; retain it in the25-attempt denominator.
There is no silent SG fallback or per-case model/threshold selection. A failure
can justify a later explicitly approved robustness change, not retroactive rescue.

Exact claim/assumptions: only the registration's point evidence changes; the
existing decoder's geometric guarantee remains conditional on its unchanged
implementation and saved-map checks. Better anatomical correspondence is a
testable hypothesis, not a theorem. It requires reliable cross-stain matches at
overview scale and tolerable serial-section differences. The new matcher may
still miss structures, hallucinate low-texture correspondences or overweight
repeated gland boundaries. The point weight.1 is a first matched control, not
a universal calibration across matcher confidence distributions. Ordinary
subsequent global tuning is possible only as a new declared decision after
examining actual evidence, never per-label/per-case optimization.

Availability and implementation scope checked2026-10-02: the
[official repository](https://github.com/zju3dv/MatchAnything) links to public
HF inference source at Space revision6a7bcb589ec8da3a9e861e799122beaa5eba2193.
The release README links to the author-hosted
[weights.zip](https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view).
A read-only HEAD request returned200, application/octet-stream,482746196bytes
and byte-range support; no archive contents were downloaded or verified here.
Its model includes `matchanything_eloftr.ckpt` according to the release adapter.
The adapter's named LittleFrog/MatchAnything_checkpoints endpoint currently
returns401, so do not assume it works or bypass authentication. A third-party
64.4MB ELoFTR mirror exists but was not established as the authoritative release;
prefer the author archive. Neither MatchAnything source nor weights were found
in the bounded AI cache checks. Cached LightGlue is not this model.

Use one isolated research dependency location. The direct ELoFTR source import
path was inspected: torch/numpy, einops, loguru, yacs/PyYAML and Kornia suffice
for the traced inference classes; the actual compatible versions are below.
Avoid the broad old requirements.txt, which includes unrelated training pins.
Construct LoFTR directly from the lowercased released config, load
`torch.load(..., map_location="cpu", weights_only=True)["state_dict"]`, and
require complete parameter matching after the class's explicit `matcher.` prefix
removal. A missing/unexpected model key or safe-loader failure is an integration
failure, not permission for random missing weights or unsafe unpickling. No
RepVGG rewrite is needed for the first inference. Preserve source notices: the
HF package carries Apache2; current top-level GitHub PRL explicitly permits
academic research/evaluation without registration but differs for project use.
Do not describe every downstream use as unrestricted.

Smallest decisive test and scope: after approval, one dependency/checkpoint-load
and512forward smoke test, then literal injected-match coordinate/confidence tests
(nontrivial affine, pixel centers, border prewarp, eligibility, insufficient
matches, point-loss value/VJP). Independent checker verifies input conventions
and actual configuration. No geometry-module edit is required: one extractor
adapts outputs to the existing point JSON and a thin batch runner substitutes
that path only in all25saved shared-gray controls. Extract all25point tables
before any optimization or label scoring, then run the25analytic300attempts and
score only after all attempts terminate. Report all four specimen cohorts and
failures, not25independent patients. Include model setup, extraction, optimizer,
serialization/certification and peak memory separately and end-to-end; do not
inherit published ELoFTR timings. Aim to spend at most about one hour to first
complete-cohort answer, with a bounded integration stop if dependencies block.

What falsifies it: insufficient/nonfinite matches, practically prohibitive
integration/runtime, or no useful cohort accuracy/cost gain. A harmful match
table must not be rescued using manual landmarks or another method's dense map.
A positive result would support this pretrained-evidence-plus-hard-P1 pipeline
on already examined specimens, not a new matcher, anatomy guarantee, blind
generalization claim or SOTA result. If it fails, record the failure mechanism
before selecting another intervention; do not automatically launch a model zoo.

### Approved adapter: strict real-weight load and512smoke complete

The author archive downloaded successfully into the AI research project's own
`matchanything_cache`; its ordinary ZIP listing contains the ELoFTR checkpoint
64366723bytes and the separate ROMA checkpoint. Only ELoFTR was extracted.
Required weights and pinned inference source are mirrored under
`D:/QC_optimization_data/digital_topology_wsi/matchanything_eloftr/`; no images
were uploaded to an external inference service. HF source and current upstream
license notices are retained. No shared Python environment or service was changed.

An isolated AI system-site-packages venv reuses installed torch2.5.1+cu124 and
adds only einops0.8.1, loguru0.7.3, yacs0.1.8, PyYAML6.0.2, Kornia0.7.3 and
kornia_rs0.1.9 with no-dependency installs. The first import exposed missing
PyYAML; installing it in this SAME isolated venv resolved the dependency.
The broad release training requirements were not installed. Source imports,
weights-only loading and strict447state-entry matching succeed, with
16025216model parameters and the declared FP32/threshold.1/NPE configuration.

`tools/coordinated_matchanything.py` exposes one FrozenMatchAnything constructor
per cohort, `.setup_report`, `.extract(fixed,moving,affine,output=...)`, and
`.close()`. Its pure point_record helper keeps pixel-center coordinates and
original confidences; static world eligibility and confidence normalization
remain the existing optimizer's responsibility. The fixed-gray support diagnostic
uses the containing pixel floor(512q); q=1 maps to pixel511 ONLY for that
diagnostic. There is no new point-support filter.

Before the actual smoke, GPU5was idle at11MiB. The real MIIT2-to-3 image-only
512forward returned2910network matches,2908positive staticeligible matches,
16occupied4x4bins, zero finite-domain discards and finite coordinates/confidences.
Unweighted fixed-gray-support fraction was.999312715. Cold setup took1.01391s,
with351160320bytes allocated peak; first extraction (including image load,
prewarp, inference and point adaptation) took1.07034s and peaked986538496bytes.
These are smoke measurements, not warmed throughput or anatomical evidence.
The point JSON is `matchanything_eloftr/smoke_miit_2_to_3.json` in the D data
cache and `matchanything_cache/smoke_miit_2_to_3.json` in the AI research tree.

Twenty-one focused adapter/legacy-matcher tests pass. They cover literal half-pixel
coordinates, endpoints, world eligibility, all-output nonfinite rejection,
confidence bounds, insufficient evidence, ordinary PIL grayscale, border
prewarp, existing-loader compatibility and output nonoverwrite/model closure.
Two test-fixture mistakes were corrected: uint8-ramp reversal is not exactly
255-minus-rounded-ramp, and load_image_matches returns a pair, not just the
module. Neither correction changed the production adapter. Source diff-check
passes. An independent checker now verifies injected coordinates and the actual
smoke separately; no25case production or manual-label scoring has begun here.

Independent source adjudication: CoarseMatching stores `mtd_spvs` in `self.mtd`
and, when true, selects ALL above-threshold pairs after the ordinary border
removal. FORCE_NEAREST appears in config but is not read by this actual class.
The original card/setup label incorrectly inferred mutual-nearest behavior from
that flag. Corrected metadata uses `configuration_force_nearest` and records
the actual coarse policy explicitly. Fine matching remains the released TOPK1
plus local regression; mconf remains coarse confidence, not a new fine-level
anatomical confidence. No model/filter/weights were altered, so the first smoke
point coordinates are unchanged. Exact fine-source/target unique counts and
maximum duplicate multiplicities are now reporting-only diagnostics, with no
deduplication or influence reweighting. The coordinator explicitly approved
preserving the released behavior rather than imposing the mistaken description.

### Proposed: preserve SG anchors and add independently normalized MA evidence

Question (2026-10-02, after the completed MA25 replacement): did replacement
lose a distinct useful SG constraint, rather than merely overweighting a denser
MA table? This is an image-evidence question, not a topology or optimizer theorem.
The coordinator approved writing this card; no complement production run has
been authorized or started by this diagnostic worker.

Saved-map CPU diagnosis reads only frozen image/match tables and predictions,
not manual annotations. Reproducible source is
`outputs/coordinated_instance_registration/check_sources/matchanything_cross_functional_diagnostic.py`;
the per-case record is `matchanything_all25_t22/cross_functional_diagnostic.json`
under the same outputs tree. At each SAME saved SG map, differentiate both
weighted point terms through an additive33x33 physical-residual perturbation,
bilinearly prolonged to257with fixed-zero boundary. This is not the production
Adam/control gradient or a projection onto active Jacobian inequalities.

| Cohort | norm(.1 SG gradient) | norm(.1 MA gradient) | SG/MA cosine | E_MA(new)-E_MA(old) | E_SG(new)-E_SG(old) |
|---|---:|---:|---:|---:|---:|
| MIIT3 | .078286 | .043393 | -.032395 | -.0021782 | +.0030469 |
| Lung20 | .099194 | .096536 | +.071172 | -.0094399 | +.0145617 |
| Histo | .055118 | .036262 | -.100110 | -.0011300 | +.0014665 |
| Kidney | .079522 | .070858 | -.013986 | -.0052408 | +.0030173 |

Full cross-functional evaluation includes the unchanged dense image, ARAP,
shape and OOB terms.22/25MA maps beat the old SG map on the MA functional;
all25lose on the SG functional. The three own-MA exceptions are cc10-to-ki67,
cc10-to-prospc and prospc-to-cc10. Thus incomplete optimization remains possible,
but it is not the main explanation for most replacement changes. MA point
gradients do not dominate by norm, and their direction is nearly orthogonal to
the SG gradients. Image/MA gradient cosines at the old SG map are weakly positive
(cohort means .1634/.1023/.0129/.0460), not strong destructive opposition.
Confidence normalization already makes each point table's total eligible mass1;
more matches do NOT automatically increase its global coefficient.

The MA evidence changes support and constraint directions, not just scalar
strength. Fine-source exact duplicates are few (5--38 per raw table of roughly
2000--3100); therefore an exact-duplicate fix cannot explain the whole result.
Many8px source bins do contain multiple nearby observations with differing
targets, but nearby sources need not represent alternative matches for the
same anatomical point. This diagnostic does not justify a probabilistic
mixture, deduplication, a confidence threshold or a local-consistency filter.
Spatial density/coherence alone is not anatomical truth. Numerical gradient
statistics are evidence for testing complementarity, not proof that either
matcher's constraints are correct.

Exact next claim/hypothesis: keeping the original SG term and adding MA as a
separate, equally weighted evidence source can retain sparse anchors while
testing whether semidense pretrained correspondences add useful information:

    E_plus(Y) = I_shared-gray-MIND(Y) +3 ARAP(Y) +1e-4 Shape(Y)
                +OOB(Y) +.1 P_SG(Y) +.1 P_MA(Y).

For matcher h, let D_h=sum_j c_hj eligible_hj. Concatenate the existing source
and target rows, giving each original confidence the value c_hj/D_h, and use
the existing loader with match_weight=.2. Each matcher's eligible mass is1,
the combined denominator is2, and therefore the existing loss is EXACTLY
.1P_SG+.1P_MA, up to floating-point summation. Preserve each original row,
including statically ineligible rows; report the two original counts and masses.
There is no rematching, target change, spatial selection, new filter or clipping.
Check finite positive D_h and combined confidence bounds explicitly; the actual
tables have D_h>1 so rescaling preserves confidence<=1. Do not silently clamp an
unexpected incompatible table. Coordinate/affine/raster metadata must agree.

Assumptions: the two sources may provide complementary anatomical observations,
but may also share systematic errors or impose incompatible constraints. Equal
coefficients are one transparent default, not confidence calibration. This
intervention increases total point influence from.1to.2; if it helps, that does
not by itself isolate complementarity from total-weight effects. The required
matched-strength control is therefore E_SG2=I+3ARAP+1e-4Shape+OOB+.2P_SG,
using the ORIGINAL SG table unchanged. Compare both against the already saved
.1SG control. This tied-strength two-arm experiment is not a coefficient sweep.
No universal matching or anatomical guarantee is claimed. The hard geometric
decoder is unchanged.

Smallest decisive test: first verify concatenated versus two separate point
terms on literal nontrivial-affine P1 maps, including eligibility, unequal
confidence masses, loss values and vertex VJPs. Independently check this algebra
and the25configuration changes. Then, only after coordinator authorization,
construct all25combined tables from already frozen inputs and run both fusion
and SG2arms,25cases each, using the same analytic300identity-start controls.
Keep affine, images, dense MIND, ARAP3,
shape1e-4, robust scale8px, grid257, continuation, optimizer settings and
best-full selection unchanged. All50attempts terminate before ANY new ordinary
scoring; retain failures and four-specimen cohort reporting. A fusion benefit
over SG2supports information beyond merely doubling the SG coefficient; it
still does not establish universal anatomical correctness. No beta sweep or
per-case selection. Report table composition/setup separately from optimizer cost; retain
the already measured matcher extraction cost in any end-to-end comparison.

What falsifies it: no useful cohort accuracy/cost trade-off, loss/VJP algebra
failure, or a new anatomical regression despite retaining SG. A negative result
ends this bounded matcher-complement branch rather than prompting weight/filter
search. Prior work for MA is the released model and paper cited in the preceding
card; the additive loss is an ordinary evidence-combination control, not a new
matching algorithm. Joint global-affine/residual objective redesign is a
different possible later mechanism, not part of this intervention.

### Executed: tied-strength fusion is retained as one exploratory recipe

The50prediction attempts have now terminated:25/25fusion and25/25SG2succeeded,
with zero composition failures. The saved prediction manifest records completion
and `annotations_read=false`; ordinary scoring followed completion. Source:
`outputs/coordinated_instance_registration/match_fusion_all50_t23/comparison.json`
and its referenced prediction manifests. The archived SG1arm also retains all25
successful directions. Every cohort below retains its full direction denominator;
there are no failed cases omitted from these aggregates.

The metric entries are mean of pair means / mean of pair p90s, in512moving-canvas
pixels; they are not pooled-label percentiles or independent-patient averages.

| Cohort/directions | Original SG1(.1SG) | SG2(.2SG) | Fusion(.1SG+.1MA) |
|---|---:|---:|---:|
| MIIT/3 |3.54875 /5.90796|3.56414 /5.95346|3.53472 /5.85817|
| Lung/20 |4.56854 /9.59368|4.51702 /9.67054|4.47101 /9.34254|
| Histo/1 |.85190 /1.58601|.94696 /1.71447|.84206 /1.58014|
| Kidney/1 |2.36487 /4.75691|2.38222 /4.70210|2.24361 /4.64355|

Fusion improves all four cohort mean/p90 aggregates against both controls.
Equal-specimen deltas are -.06067mean/-.10504p90 versus SG1, and
-.07974mean/-.15404p90 versus SG2. This supports complementary evidence in this
development experiment beyond simply doubling the SG coefficient; it is a
modest gain, not a large registration breakthrough or same-objective convergence
claim. The coordinator tentatively retains fusion as ONE globally applied
exploratory evidence recipe, subject to the independent post-run check. No case
chooses its better arm, no coefficient sweep follows, and historical controls
and all their results remain available.

Aggregate improvement does not erase paired regressions. Versus SG1, fusion
improves pair means on2/3MIIT,16/20lung,1/1Histo and1/1kidney; versus SG2the
counts are3/3,12/20,1/1,1/1. Pair p90 improves on2/3MIIT and13/20lung versus
SG1, and2/3MIIT and14/20lung versus SG2; Histo/kidney improve in both comparisons.
The largest mean regression versus SG1is cd31-to-ki67 +.43696px (versus SG2
+.52400px). The largest p90 regression versus SG1is ki67-to-he +1.01575px;
cd31-to-ki67 worsens p90 by.96232px versus SG1and1.16403px versus SG2.

Severe tails remain: fusion's cohort worst maxima are53.02103MIIT,
38.51869lung,4.14147Histo and8.97058kidney. The MIIT/lung maxima exceed SG2's
52.87364/38.44655, and the kidney maximum exceeds SG1's8.93328. Only8/20lung
pair maxima improve against SG1and7/20against SG2; he-to-cc10's maximum
regresses2.42248px versus SG1. These are retained anatomical errors, not solver
failures, and must not disappear behind the four improved mean/p90 rows.

Cost remains part of the result: current batch wall time222.466s, including
1.096s table composition; optimizer complete-call totals are114.162s fusion
and106.531s SG2 (archived SG1total113.718s, not a synchronized speed trial).
Maximum optimizer allocated memory is214255616bytes fusion and213284864bytes
SG2. Fusion reused cached MA tables, whose required historical setup1.053s
and25extraction calls totaling3.237s are additional end-to-end costs, not zero.
Earlier affine/SG extraction is outside these totals, and allocated peaks from
different stages must not be summed.

All25directions come from FOUR already examined specimens. This is development
selection, not held-out evidence,25patients, clinical validation or a general
anatomical guarantee. The next joint-pose candidate remains a separate change
of map/prior and awaits its fixed schedule, implementation authorization and
independent checks; the fusion result does not establish that candidate.
