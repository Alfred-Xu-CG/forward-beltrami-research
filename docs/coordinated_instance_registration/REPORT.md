# Coordinated registration: self-contained living research report

This is a working synthesis, NOT the final review. The authorized window
started2026-10-01 11:26:23UTC. On 2026-10-02 the user resumed the paused goal
and extended the deadline by five hours to 2026-10-02 16:26:23UTC. Conclusions below
describe evidence available around T+17h and will be revised after later checks.
The detailed derivations and experiment definitions are in FORMULATION.md;
chronology and negative interventions are in PROGRESS.md. This report introduces
the main objects without requiring knowledge of those earlier discussions.

### Restart correction: scope of the earlier DHR comparison

The tables below retain their original measured values, but the method previously
called "native DHR" is more accurately **DHR shared-affine reduced-512**. Its
objective and nonrigid implementation are DHR's; its full configuration is NOT
the released complete pipeline. `tools/coordinated_dhr_common.py` substitutes our
supplied affine and sets a 512 raster, five levels and 30 iterations per level.
The released fast preset uses 2048 and seven levels of 100 iterations; the
standard preset uses 4096 and eight levels (seven times 100 plus 200). The old
0.690-second timing excludes native image-based initialization. Thus the earlier
comparison is useful for a shared-initialization diagnostic, not evidence of
superiority over recommended full RegWSI/DeeperHistReg. The restart adds genuine
released-preset runs, with native preprocessing and all configuration differences
disclosed. The first complete runs and the corrected MS experiment are below.

### T+19h restart experiments: full baseline and one failed objective hypothesis

Both released presets were actually run on AI RTX A6000, original native TIFFs,
loading ratio1, the preset's own preprocessing/initialization/nonrigid schedule,
and no supplied affine. All three directions terminated successfully as software
calls. Fast nevertheless failed anatomically on2-to3 and10-to11; successful call
status is not registration success. Every available123/107/98 annotation pair
was scored after prediction. The same512canvas conversion as before is used
only for the comparison units, not to reduce the native baseline's input.

| Method / objective | Mean pair mean TRE | Mean pair p90 TRE | Actual complete registration call |
|---|---:|---:|---|
| Shared-affine coordinated A, previous continuation | 3.548752 | 5.907957 | 4.14–6.13s |
| Shared-affine F2, previous continuation | 3.554327 | 5.942943 | 12.48–14.20s |
| Coordinated A, simultaneous five-scale objective | 3.608180 | 6.060186 | 7.53–8.85s |
| F2, simultaneous five-scale objective | 3.612537 | 6.062816 | 14.60–16.74s |
| Released DHR fast, native full pipeline | 119.044162 | 173.029174 | 8.18–9.79s |
| Released DHR standard, native full pipeline | 3.672192 | 6.432444 | 30.56–32.95s |

TRE is Euclidean target-registration error in common512canvas pixels, not
micrometers. Means weight the three pairs equally. Safe-method call times above
exclude the already-computed shared initialization/matches; native DHR times
include its own initializer. Therefore this table is NOT an apples-to-apples
end-to-end speed ratio. Initial parallel control/MS calls used separate idle
GPUs5/6; corrected MS calls were serial onGPU5. These are single-call descriptive
timings, not repeated hardware benchmarks. Allocated peaks were approximately
201–202MiB A-control,211–212MiB A-MS,461–462MiB F2-control and488–490MiB F2-MS.

The MS hypothesis failed this fixed test: every one of six method/pair means
became worse; aggregate mean and p90 also worsened while cost increased. It
averages32/64/128/256/512 normalized image terms at EVERY stage, keeping priors,
points and512OOB once, so it changes objective and continuation, not geometry.
All twelve primary safe runs completed300gradients with no failed trials;
actual corner floors remain strictly positive. Independent mixed-precision
value/gradient recomputation agrees to5.55e-17/4.44e-16 on a small separate test.
The selected MS recipe is not promoted; this is not a proof that all multiscale
metrics fail. No neighboring weight sweep follows this negative result.

One first MS run used an algebraically equivalent subtraction/addition of image
scalars; after a mixed-precision arithmetic cleanup only MS was rerun, BEFORE
primary scoring. The first output remains in `miit_multiscale_sum_t19`; the
primary result is `miit_multiscale_sum_direct_t19`. Control continuation was
unchanged and independently reproduces the earlier scores. Formula and test
details are in OPTIMIZER_REDESIGN.md. Full332objective calls require4.5474times
the prior continuation's raster queries; the300gradient evaluations require
five times its image raster work. Neither count is a measured runtime ratio.

Native standard has mean TRE3.087571/3.280505/4.648501 on the three pairs
and still a roughly53.5pixel worst error on7-to8. Scoring only its stored native
initial affine gives mean/p90=4.887064/8.282913, compared with the original
common initializer4.904193/8.897884. Its native nonrigid stage therefore reduces
mean by1.214872pixels (about24.9%), whereas the mean difference between initializers
is only.017129pixels. This is an observed stage decomposition, not the unrun
counterfactual of giving every method exactly the native initializer. The native
7-to8 maximum actually worsens from50.448319 to53.497967. Its actual saved bilinear
field has nonpositive local corners in all three cases; no post-hoc repair was
applied. The scorer checks its own pixel-center field cells and does not claim
a global boundary theorem or an equivalent257P1 representation. Independent
rectangular/resampling/saver fixtures verify the declared field evaluator,
not equivalence to DHR's optional cubic landmark utility or rendered-image path.

The audit also found approximately23–25% of our MIIT intensity-threshold support
outside the provided tissue mask. That is a modeling issue worth a separate
fixed-support test, not a demonstrated violation of DHR's mask-free preset.
The completed mask arm uses images PLUS released semi-manual tissue masks and
does not remove any evaluation landmark. It also shows no material improvement:
A mean/p90=3.565586/5.933243; F2=3.556160/5.926603. All six runs complete300
gradients with valid outputs. Thus the background-support observation is real,
but it is not supported as the main cause of the registration plateau. Keep the
image-only threshold baseline; do not require extra annotations without benefit.

The next six-run pilot substitutes postwarp-intensity NGF for MIND, using the
same geometry, frozen affine/matches, priors, original support, continuation and
300 gradients. It also fails to improve registration: A mean/p90 becomes
4.060915/6.914149 and F2 becomes3.982225/7.187874, versus the shared-affine MIND
control3.548752/5.907957 and3.554327/5.942943. All six runs finish without failed
trials and all exported maps pass the strict stored-binary certificate. A takes
5.60–7.06s and179–180MiB allocated peak; F2 takes12.82–15.95s and438–441MiB.
These are descriptive complete optimizer calls excluding the frozen common
initializer, not clean repeated end-to-end speed ratios. Outputs and all328
posthoc point errors are in `miit_ngf_t19`. The NGF derivative is independently
verified, but correctness of a derivative is not evidence of anatomical benefit.
This exact recipe is retired without an epsilon sweep. Its finite epsilon can
favor larger warped gradient magnitudes, and flat moving regions have zero
image derivative; no global-capture or true-map optimality theorem was claimed.

After these three distinct negative interventions (simultaneous multiscale loss,
released tissue support, NGF), further small objective substitutions are not the
next priority. The next decision separates lost high-resolution image evidence
from the local optimizer's inability to establish coherent correspondences.
Native standard DHR's stage decomposition above makes a purely better affine
initializer an insufficient explanation for its nonrigid gain on these cases.

## 1. What is implemented, and what is not

Implemented: a differentiable local forward decoder which takes learnable
displacement coefficients and an already valid vertex table, and returns another
valid P1 vertex table on the SAME dense source triangulation. Its geometry uses
local determinants and parallel reductions, not inversion of a global mesh
matrix. A sequence of such updates is used for image-driven per-instance
optimization, with actual257-by257 control vertices in the main experiments.

Not established: a newly trained image-to-map neural network, inference-time
anatomical superiority over state-of-the-art registration, clinical validity,
or universal approximation of every digitally admissible homeomorphism by a
particular finite schedule. Local decoder gradient checks do not establish
that a new neural encoder has learned complex registration. Optimizing one
image pair's latent coefficients is a different task from training an encoder.

## 2. Domains, coordinates and the output function

Let Omega=[0,1]^2. Let N be the number of vertices along each axis, with source
vertices X_ij=(j/(N-1),i/(N-1)),0<=i,j<N. The N=257 production lattice has66049
control vertices,65536 quadrilateral cells and131072 source triangles. These
are map degrees of freedom, not merely dense queries of a25-square control map.

For each cell its source vertices are called a,b,c,d in top-left,top-right,
bottom-right,bottom-left order. Fix the a-c diagonal. A P1 map f_Y is the unique
continuous function that maps X_ij to Y_ij and is affine on each of these source
triangles. P1 means degree-one polynomial on each triangle. The connectivity
and source triangulation are fixed even when Y becomes irregular.

The complete image-sampling function is

    F(q)=A f_Y(q)+b0,

where A is a frozen orientation-preserving2-by2 image-derived similarity and b0
is a frozen translation. The residual boundary is Y_ij=X_ij on all four sides.
Hence f_Y maps the rectangle to itself; F maps it onto its affine image, not
necessarily the entire moving-image canvas. The saved archive retains the
residual table and original affine separately. Evaluators apply A exactly once.

An image of width W,height H defines intensities at ((j+.5)/W,(i+.5)/H), its
pixel centers. Map vertices instead lie at endpoints j/(N-1),i/(N-1). Map
sampling is declared P1; intensity/feature sampling is bilinear with explicit
zero padding outside the image. These two interpolation operations are distinct.

## 3. The exact discrete feasibility constraints

For vectors v,w in R^2 define det(v,w)=v_x w_y-v_y w_x. For a mapped cell protect

    q1=det(b-a,d-a),  q2=det(b-a,c-b),
    q3=det(c-d,c-b),  q4=det(c-d,d-a).

Each q is twice a signed triangle area. Normalize by the source value
q_ref=1/(N-1)^2, so the identity has q/q_ref=1. Production requires all four
ratios greater than eta=.001 and an exactly fixed, ordered rectangle boundary.

The four inequalities ensure strict orientation for the triangles of BOTH
diagonal choices. Together with continuous edge sharing and an injective outer
boundary, they imply global P1 homeomorphism for the declared a-c map. A
homeomorphism is a continuous bijection whose inverse is continuous. The same
table also defines a legal quadrilateral-bilinear map, but that is a different
function and is not silently substituted into scoring.

The eta floor excludes some legal maps with tiny positive determinants. In real
arithmetic the formula below preserves that floor; implementation additionally
checks actual rounded candidates and the exported table. Returning the previous
valid anchor after a numerical failure is an explicitly recorded fallback, not
post-hoc fold repair. Native DHR receives no such repair and no global guarantee.

## 4. Coordinated forward updates and their gradients

Fix a current accepted table Y, a direction e equal to(1,0) or(0,1), and scalar
amplitudes u_i at the vertices, with zero boundary amplitudes. Update all interior
vertices simultaneously by Y'_i=Y_i+u_i e. For a triangle(i,j,k),

    det(Y'_j-Y'_i,Y'_k-Y'_i)
    =det(Y_j-Y_i,Y_k-Y_i)
     +(u_j-u_i)det(e,Y_k-Y_i)
     +(u_k-u_i)det(Y_j-Y_i,e).

The quadratic term vanishes because det(e,e)=0. Thus every protected normalized
constraint is EXACTLY s_k+(C_Y u)_k>0, where s_k=q_k(Y)/q_ref-eta is its current
positive slack and C_Y is a sparse local linear operator assembled implicitly
from current edges. No finite-difference approximation of determinant change
is used. The matrix need not be materialized or inverted.

At scale l a learnable scalar coefficient table z_l is interpolated as a RAW
proposal r=P_l z_l on the fixed fine vertices. P_l denotes ordinary scalar
proposal interpolation plus boundary zeroing; it is not resampling an accepted
map or composing maps on an unrelated control grid. Levels are17,33,65,129,257.
The actual output table remains257-square throughout these production stages.

Two entry operators were tested. Define the nonnegative feasibility gauge

    g_Y(r)=max(0,max_k[-(C_Y r)_k/s_k]).

The radial decoder uses u=r/(1+g_Y(r)); then each new slack is strictly positive
for every finite proposal in real arithmetic. Its gauge is positively homogeneous.
For unrestricted scalar amplitudes, this radial transformation reaches every
feasible single-direction displacement u: its inverse is
r=u/(1-g_Y(u)), because feasible u has g_Y(u)<1. This is a statement about the
single-direction feasible set, not a theorem that a fixed finite alternating
network reaches every2D homeomorphism. Coarse P_l may restrict its proposal space.

The production analytic-step operator uses

    sigma=min(alpha_trial,theta/g_Y(r)),  u=sigma r,

with theta=.95 and theta/0 interpreted as infinity. Here alpha_trial=1 for the
ordinary proposal. It leaves at least(1-theta)s_k slack whenever the geometric
bound is active. Both operators are differentiable away from max/min ties;
autograd includes the derivative of the active scale. Detaching sigma would be
a different and generally incorrect gradient. Finite-precision checks and image
accept/reject decisions are separate discrete branches, not claimed globally
smooth operations. Local value/gradient and deliberate failure tests cover the
decoder; a gradient through300optimizer steps is not claimed or retained.

For completeness, with the anchor held fixed and a scalar loss whose derivative
with respect to u is v, the coefficient vector-Jacobian product is

    dL/dz_l = P_l^T [sigma v+(r^T v) grad_r sigma].

P_l^T is the transpose interpolation accumulation. For an active constraint row
c_k of C_Y, radial sigma has grad_r sigma=sigma^2 c_k/s_k. The geometry-limited
analytic scale has grad_r sigma=theta s_k c_k/(c_k r)^2; when the trial limit is
strictly active, that gradient is zero. These formulas apply away from ties;
anchor derivatives additionally differentiate C_Y and s, which ordinary AD
can retain in a neural-layer usage. In the instance optimizer accepted anchors
are intentionally detached between stages. A vector-Jacobian product means
multiplying a scalar-loss derivative backwards through a function's Jacobian,
without storing or forming that entire dense Jacobian.

This is coordinated motion: adjacent vertices may move a long common distance
while their DIFFERENCES, not original mesh spacing, determine admissibility.
A worst active cell can still limit the global scalar scale. This limitation is
measured rather than denied. F2 provides a local-patch comparator under identical
input evidence and objective; it is not automatically worse by construction.

Successive accepted tables are on the same source triangulation. Geometrically,
one can define an affine update h on each current deformed triangle taking its
old vertices Y to its new vertices Y'; then f_Y'=h composed with f_Y EXACTLY.
There is no generic continuous-flow integration followed by uncertified sampled
P1 interpolation. This statement relies on current nondegenerate triangles,
shared-edge consistency and the preserved connectivity.

## 5. What optimization actually consumes

Inputs for one case: fixed and moving images, one common positive image-only
affine, frozen machine-generated correspondence points/confidences, and the
fixed reference mesh. Manual evaluation landmarks are not inputs. The output is
the saved P1 map, its affine factor, explicit geometry certificate, runtime,
gradient/evaluation counts and diagnostics. There is no trained CNN in this phase.

The retained objective is

    E = I +3 R +.0001 S +.1 M + O.

Here are definitions of every term. Convert images to inverted grayscale
intensities G in[0,1]. At each raster scale, use the eight integer offsets
(2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2). For each offset a compute
D_a(x), the3-by3 local average of(G(x)-G(x+a))^2, with replicated edge values
for shifting and available-neighbor averaging at pooling edges. Set

    Phi_a(x)=exp[-(D_a(x)-min_b D_b(x))/(mean_b D_b(x)+.0001)].

These eight values are the local self-similarity descriptor (MIND-like here;
not a claim of reproducing every published MIND variant). Compute Phi_f and
Phi_m separately before warping. Let m(x) be the fixed foreground mask: threshold
G_f>.04 at full resolution, area-reduced to lower raster scales. Let Z=sum_x m(x)
on the fixed pixel centers; this denominator is unchanged by the candidate map.
With bilinear zero-padded feature sampling B, define

    I=sum_x m(x) mean_a |Phi_f,a(x)-B(Phi_m,a,F(x))| / Z.

For a source triangle t=(i,j,k), define edge matrices
B_t=[X_j-X_i,X_k-X_i],D_t=[Y_j-Y_i,Y_k-Y_i],J_t=D_t B_t^{-1}.
Only a2-by2 SOURCE edge inverse is involved. The uniform mesh gives equal source
areas, and R=.5 mean_t ||J_t-R_t||_F^2. R_t is the closest orientation-preserving
rotation. Explicitly put a_t=J00+J11,b_t=J10-J01,s_t=sqrt(a_t^2+b_t^2);
then R_t=[[a_t,-b_t],[b_t,a_t]]/s_t. Positive detJ ensures s_t>0. Identity has
zero R; a rigid rotation also has zero R, whereas spatially varying rotations
usually induce stretch. This is an elastic prior, not the topology safeguard.

Let K_l be each of the four2-by2 cell-corner derivative matrices, with first
column equal to the relevant horizontal mapped edge divided by source spacing
and second column the vertical mapped edge divided by spacing. Define

    S=mean_l [||K_l||_F^2+||K_l^{-1}||_F^2-4].

For a2-by2 K with positive determinant, ||K^{-1}||_F^2=||K||_F^2/det(K)^2;
the implementation uses that identity rather than a mesh-system solve. This is
four-corner quadrature, not an exact quadrilateral integral; identity has S=0.

Frozen machine points are(q_j,p_j,c_j): fixed-unit keypoint, affine-aligned
moving-unit keypoint and original confidence. Their original moving location
is A p_j+b0. Statically discard only machine targets outside the original moving
rectangle, never difficult MANUAL evaluation points. Normalize eligible c_j
to weights w_j with sum1. With rho(r)=sqrt(1+||r||_2^2)-1 and kappa=8pixels,

    M=sum_j w_j rho(512 A(f_Y(q_j)-p_j)/kappa).

The translation cancels; A remains in the metric. There is no extra kappa^2
multiplier. Machine confidence is not anatomical ground truth, and no derivative
through keypoint selection/matching is claimed. Finally define componentwise
outside excess t_d(v)=max(-v_d,0)+max(v_d-1,0) and

    O=sum_x m(x) sum_d t_d(F(x))^2 / Z.

Thus out-of-bounds image queries are both zero-padded and penalized, not dropped.
I and O use the current raster scale; M always uses the full512pixel metric;
R and S use the actual fixed fine control mesh. A stage compares its COMPLETE
objective at that stage's raster scale, with all terms and the coefficients
above; it does not compare image loss alone. Accepted stage maps are additionally
scored by the complete512objective for final selection. Consequently the512
objective need not decrease at every lower-resolution stage; best-full selection
retains the best full-resolution accepted output rather than asserting monotonicity.

Each stage freezes its accepted anchor Y while Adam optimizes that stage's z_l.
Trial candidates do NOT accidentally accumulate within the latent solve. Only
a geometry-valid candidate that improves the complete STAGE-objective comparison
becomes the next anchor. The final result is selected by the full512-resolution
objective among accepted maps, never by manual landmark error.

Production analytic: five scalar scales, x/y alternating stages,30Adam steps
per stage,300gradient steps total,310decoder trials,332complete objective calls.
Learning-rate calibration is.004*16/(level-1), in pre-affine normalized units.
F2 uses two five-scale cycles with30steps each for the same300gradients and
has the corrected.95strict-floor reserve. Both start from the same affine plus
identity residual. Native DHR has its own objective/preprocessing and is an
application comparator, not an equal-objective geometry ablation.

## 6. Evidence available aroundT+15h

The lung comparison covers20ordered stain directions from ONE previously viewed
physical specimen, with all80matched manual IDs per direction. Directions and
landmarks are correlated, not20patients or1600independent observations. Errors
below are Euclidean target-registration errors in512canvas pixels, computed
after prediction by an independent P1 evaluator. Mean-p90 means the arithmetic
mean of each direction's90th-percentile error, not the pooled90th percentile.

|Method|Mean direction mean TRE|Mean direction p90 TRE|Output topology|
|---|---:|---:|---|
|Common affine|6.661222|12.234678|Positive affine|
|Coordinated analytic|4.520228|9.555206|Certified P1-ac/all-four corners|
|Corrected full-budget F2|4.551344|9.657840|Certified P1-ac/all-four corners|
|Native DHR|5.133593|11.552831|No hard global guarantee|

Analytic improves mean TRE versus affine in all20directions, but native DHR
wins six direction means, including all fourproSPC-source directions. Analytic's
aggregate advantage over corrected F2 is only0.68%mean and1.06%tail: do not claim
strong anatomical superiority. Native DHR remains much faster in these small
canvas experiments. Clinical competitiveness and unseen-patient generalization
are not established.

Warm complete-call engineering comparisons, with repeatedABBA order, reduce
coordinated runtime by compiling existing ARAP/shape work (about4.05to2.95s on
one known pair) and then by precomputing fixed machine-point triangle indices
and barycentric weights (2.95to2.70s on a second known pair). Map/objective
differences stay at the scale of measured repeat variation, not bitwise identity.
Cold compiler setup is not a speedup. Typical allocated GPU peaks for the main
analytic optimizer are about205--229MB; F2 about495--500MB; native DHR about70--73MB.
These are measured allocated peaks, not whole-process RAM/driver/compiler peaks;
setup/load/export inclusion is specified for each timing experiment.

Strong adverse interventions are retained: more iterations/budget redistribution
can lower the objective while worsening landmarks; removing dense appearance
worsens mean TRE in all20directions; a finite independent-label displacement
prefix improves appearance but raises the complete objective and is rejected
in all six attempted axes. Neither lower proxy loss nor valid topology alone
proves anatomical benefit. Those recipes are retired rather than swept nearby.

Additional prostate transfer now completes after a DISCLOSED image-only
initializer amendment: four right-angle moving views, with original-frame
keypoints and fixed inlier/support/fit ranking, specified before reading manual
coordinates. The original raw-image initializer's two failures remain recorded.
The original assumption that all124labels are finite is also false: the release
uses(+inf,+inf) for absent annotations. Author metric code excludes absent labels.
An explicit source-availability correction, before computing errors, scores ALL
123,107,98both-source-available IDs respectively, identical across methods; all
missing labels and original strict-scoring failures are reported separately.
Finite-point bounds and invalid-prediction guards are not relaxed.

|Method|Prostate equal-pair mean TRE (512px)|Mean pair-p90 TRE (512px)|
|---|---:|---:|
|Common affine|4.904193|8.897884|
|Coordinated analytic|3.897540|6.289528|
|Corrected F2|3.964294|6.381702|
|Native DHR|4.050615|7.181205|

Independent native-frame/CSV/generic-triangle evaluation agrees within1.13e-12px;
actual six safe maps have1,572,864positive corners, minimum ratio.2015743,
exact boundaries/shared affines and all300gradient budgets complete. Native
fields have11506,5271,15158nonpositive corners, retained in scoring. Analytic
improves affine mean/p90 on all three pairs, but loses to F2 on7-to8mean/p90
and to DHR on7-to8p90. Its maximum error is worse than DHR on two pairs; the
7-to8worst error worsens versus affine and remains52.71px. It is not an outlier
cure. Single mean complete calls are4.218s analytic,13.212s F2,.690s DHR;
this is not a paired timing estimate or equal-objective application comparison.
This additional previously used specimen supports bounded development transfer,
not unseen-patient/clinical/general superiority. Details and actual files are in
FORMULATION47, miit_three_rotations_t153/predictions.json and its separately
saved available_landmark_scores.json.

As a bounded post-hoc diagnostic, all123/107/98available targets lie inside
the common affine image rectangle. Their distance-to-range necessary lower
bounds are zero. The worst7-to8target is38.98canvas pixels inside its closest
range boundary despite52.71pixels of registration error. Its error is therefore
not forced by a target outside the decoder's image. This does not establish
that boundary constraints have no other effects or that image evidence is adequate.

## 7. Current interpretation, not final closure

The strongest established result is a scalable, differentiable, no-global-solve
safe update operator plus a useful instance optimizer on the known lung cohort.
The geometry problem is not equivalent to solving arbitrary prescribed
facewise Beltrami coefficients. No Beltrami-recovery objective is required here.
The main unresolved tasks are robust image-derived initialization/evidence,
cross-specimen anatomical usefulness and eventual amortization into a learned
network. The24-hour goal remains active; this document does not mark it complete.

An additional engineering test consolidated detached trial diagnostics into one
CPU transfer while preserving guards and gradients. Despite CPU bitwise checks
and valid28actualGPUoutputs, its paired real timing was not robust: HECC warm
median2.906to2.738s, HEK2.731to2.757s (regression), with~.4swithin-arm ranges.
The predeclared two-pair condition fails; the default remains unchanged. This is
a retained negative result, not a speed benefit. See FORMULATION48.

### 7.1 Why more iterations are not an automatic solution

Six unchanged-recipe MIIT replays save ten accepted maps each. All60snapshots
pass strict geometry checks. Prefix outputs are selected by the complete512
objective E1, never landmarks. On2-to3 the analytic E1 falls.382334to.374758
from90to300gradients, but mean TRE increases3.34619to3.55292pixels. On10-to11,
F2 correctly chooses stage4 by E1 although terminal stage9 has better anatomy.
This is an observed objective-anatomy disagreement, not a prefix-selection bug.
It does not prove optimization globally adequate or image evidence impossible.
Snapshot times include copying overhead; no clean speed claim uses them.
Replays are not bitwise originals: the largest F2 vertex difference is7.72e-5
unit coordinates, although final landmark scores differ by only~4e-8pixels.

![Full E1 and anatomical error along unchanged-recipe development paths](../../outputs/coordinated_instance_registration/results/miit_trajectory_t16/trajectory_summary.png)

Each column is one correlated pair from the same sample. Horizontal axis is
cumulative gradient evaluations, NOT time. Top: complete512E1 of accepted maps.
Middle/bottom: mean/p90TRE; solid is the accepted state, dashed the E1-selected
prefix. All available123/107/98points remain. Different stages use different
coefficient/image resolutions, but the plotted objective is ALWAYS full512E1.
These curves do not authorize TRE-based stopping or early-checkpoint selection.

### 7.2 A feature-frame limitation with an exact controlled example

Our eight MIND-like channels measure directional patch self-similarity.
Under a known affine image transform, both offset directions and patch windows
transform. Merely moving a precomputed descriptor field leaves channel identities
unchanged and is generally not equivariant. For exact90-degree rotation the
necessary correction is a known channel permutation. On an existing512texture,
the uncorrected true-map mean channel error is.26229, while the corrected maximum
is2.22e-16; identity error is zero. A separate independent pixel-loop calculation
confirms the formulas. This isolates a mechanism, not real MIIT causality.
Generic affine transforms need more than channel permutation. Sparse landmarks
also do not define true neighborhood warps. Consequently no registration objective
has been changed on the strength of this control. See FORMULATION50.

A follow-up OFFLINE fixed-affine-frame comparison retains all328available
centers. Raw descriptors prefer the annotated center over the current mapped
center for160IDs; affine-prepared descriptors do so for174. Mean cost ranking
margin changes-.002027to+.008449, but7-to8's mean margin gets worse. All
nominal support flags are valid, and an independent implementation reproduces
every ranking sign. This is weak/mixed representation evidence, not a registration
improvement. It is not a true-neighborhood ground-truth experiment.
The one six-configuration pilot is now complete and independently checked.
Analytic aggregate mean/p90 improves from 3.897540/6.289529 to
3.548752/5.907957 canvas pixels; F2 changes from 3.964294/6.381702 to
3.554328/5.942943. However, BOTH methods worsen 7-to-8 mean, whose maximum
stays around 53 pixels; 120/328 analytic and 118/328 F2 individual errors worsen.
Under the predeclared mixed-result condition the EXACT variant is retired from
main-line adoption, while its positive aggregate evidence is retained. These
same labels influenced choosing it: no untouched confirmation is claimed.
All six 257-square outputs pass strict topology checks, minimum corner ratio
.228070. Serial optimizer calls are about 4.05-5.84 seconds analytic and
12.45-13.55 seconds F2, with allocated peaks about 219-223 and 479-480 MiB;
no paired speed claim follows. See FORMULATION52 for definitions, support
limitations, all pair scores and the archived metadata correction.

The next bounded diagnostic is explicitly a manual-label ORACLE, not registration:
fit ALL 107 available 7-to-8 centers with the same fixed-boundary/affine/257-square
P1 class and exact safe operator, starting from the archived original map.
It tests simultaneous sparse representational capacity, which individual
affine-range inclusion cannot establish. Its map may not be used as production
initialization or a training teacher. The 24-hour goal remains active.

### 7.3 A rigorous forward-expressivity statement and an actual capacity witness

The earlier theoretical claim assumed a legal vertex-motion path existed.
For strictly positive FOUR-corner tables with a fixed rectangular boundary,
that assumption can now be proved: represent a target by positive directed
four-neighbor mean-value weights, suppress the graph's four degree-two corners
to apply the appropriate weak-convex-boundary Tutte theorem, and restore those
corners with explicit area calculations. Interpolating positive weights gives
a continuous path in the same four-corner class. A sufficiently fine finite
partition can be realized exactly by alternating full-nodal x/y forward updates.
This is an EXISTENCE bridge to classical convex-drawing/morphing theory, not
an online linear solve. It does NOT give a bounded depth or prove connectivity
above the preset `.001` floor. Full definitions and proof are in
[THEORY_FORWARD_EXPRESSIVITY.md](THEORY_FORWARD_EXPRESSIVITY.md).

Separately, the one actual 7-to-8 label-oracle fit succeeds at the IMPLEMENTED
floor: all 107 centers have error <=.026773 canvas pixels, mean .015168,
p90 .021693, with minimum corner ratio .0130772 and the original boundary/affine.
It uses 300 additional gradients after the archived image registration, not a
new image-only inference result. The original E1 rises .379627 to .589959;
87.18% of this particular cost increase is weighted ARAP. That does not prove
every accurate map must be costly or justify a prior change. Independent
recomputation confirms the map, all point errors and objective decomposition.
See [ORACLE_CAPACITY.md](ORACLE_CAPACITY.md) for its complete inputs, equations,
units, budget, timing and interpretation. A separately predeclared matched F2
oracle control now tests motion efficiency; no oracle map enters production.

The declared warmup+ABBA oracle collection is now complete, including three
FAILED/incomplete F2 attempts. Analytic first certifies all107 errors<=1pixel
after150 gradients (measured1.0783/1.1226s including snapshot certification),
then completes300 gradients and reaches maximum.026773pixel. F2 retains legal
outputs but reaches only165/188 measured gradients before strict rounded-floor
rejections; maximum remains37.48/37.04pixels. This is an implemented-protocol
advantage on one sparse labelled task, NOT a completed equal-budget comparison,
finite speedup ratio, general F2 impossibility or automatic image registration.
Independent recomputation confirms all nine endpoint/threshold maps, including
exact binary-rational near-floor F2 corners. ORACLE_CAPACITY Section8 gives
the full method, counts, timing/memory scopes and failure interpretation.

### 7.4 One actual initializer-restriction check, with no downstream sweep

The original 'affine' initializer is in fact a four-parameter positive
similarity, not an unrestricted six-parameter affine. A one-shot label-free
probe fits both model families to the SAME frozen machine correspondences,
original confidences/world eligibility and eight-pixel pseudo-Huber loss.
The predeclared4-by4 spatial parity split holds each parity out once on all
three pairs. All12 fits converge and remain positive, but full affine improves
only4/6held-out losses. The failed directions are2-to3 train-parity0
(.173494to.188222) and10-to11 train-parity1(.390509to.399405).
The exact branch stops, with no all-point refits or downstream registrations.
Independent coordinates/losses/gradients and a separate optimizer confirm the
result. It is not explained by fitting failure, and it does not establish that
every full-affine initializer is unsuitable. See
[INITIALIZER_STUDY.md](INITIALIZER_STUDY.md) for the complete equations/data scope.
