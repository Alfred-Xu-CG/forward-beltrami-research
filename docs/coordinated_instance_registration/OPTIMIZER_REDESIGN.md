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
