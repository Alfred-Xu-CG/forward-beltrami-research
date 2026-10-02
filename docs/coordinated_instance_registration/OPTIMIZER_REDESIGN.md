# One simultaneous multiscale image-objective experiment

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
