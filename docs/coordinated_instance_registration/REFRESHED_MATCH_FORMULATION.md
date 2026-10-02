# One deformation-conditioned refresh of the existing MA observations

Approved bounded experiment, 2026-10-02 12:15 UTC. This is a NEW observation
step followed by an ordinary safe registration suffix, not another matcher,
point-weight sweep, image-resolution change, topology decoder or repair. The
retained global recipe is frozen-pose shared-gray512 SG+MA fusion300. Its
existing saved endpoints are the incumbents for BOTH new arms. The ongoing
data-metric experiment does not select their incoming maps or settings.
Implementation and anatomical benefit from this refresh are unproven.

## Research card: decision, hypothesis, scope and falsifier

Question: can registration-conditioned correspondence extraction reveal useful
corrections that the original INITIAL-AFFINE MA table missed? Frozen MA was
extracted before nonrigid registration. Removing some local displacement and
orientation differences may allow the SAME frozen network to identify better
correspondences. This is a capture/observation hypothesis, not a conclusion
that the previous optimizer reached a global minimum or that all remaining
error is caused by its point table.

The positive motivation is the completed SG+MA complement result: preserving
both independently normalized sources modestly improved all four development
specimen mean/p90 aggregates relative to SG1 and the matched-strength SG2
control. MA replacement alone did not improve all specimens. We therefore
preserve SG, replace only MA's observation table, and keep both coefficients.
The negative optimizer/resolution results motivate a different observation
step but do NOT prove this hypothesis or justify a new pretrained-model search.

Exact claims: the observation conversion below expresses a refreshed match in
the ORIGINAL affine-aligned coordinates, without an inverse map or a composed
output map. With its unchanged validity checks, the existing safe optimizer
still returns the declared fixed257 P1-ac output. Lower anatomical error is a
falsifiable empirical hypothesis. No convergence of the outer rematching step,
joint objective descent, independence of observations, calibrated confidence,
blind generalization or algorithmic novelty is claimed.

Assumptions: the actual incumbent Y0 is a legal257-square P1-ac table with the
original identity perimeter and strict actual four-corner ratios above .001;
its saved affine A,b is the same frozen positive affine used by the original
fusion run; the accepted512 input canvases, SG table, MA checkpoint and released
inference configuration are available. Check these concrete inputs, not a new
hash/schema framework. Never use annotations, an oracle map, another method's
dense field, or the new metric arm's score to select an incumbent.

Smallest decisive test: literal rendering/point-conversion checks and incoming
map/default-path regression tests, then all25 refresh attempts and the two
same-incumbent300-gradient suffix arms. Score only after all prediction attempts
terminate. Meaningful anatomy/cost improvement over the SAME-budget frozen-table
suffix, not merely improvement over the shorter original300 run, supports this
conditioning package. A lower refreshed point loss or a better new-objective
value alone does not.

Falsifier/stop: failed extraction/conversion, unusable integration cost, or no
useful accuracy/cost tradeoff versus the frozen-table suffix retires this ONE
refresh experiment. Report small/mixed gains and adverse tails honestly; do
not require every individual direction to improve, hide failures, or choose
the better arm per case. Do not rescue a negative result with new confidence,
matching-direction, model, iteration, strength or resolution sweeps. Prior work
is ordinary registration-guided rematching using the already pinned released
MatchAnything ELoFTR and the repository's existing safe instance optimizer;
this is not an EM convergence theorem or a new learned matcher.

## 1. Frozen domains and the incumbent

All source/affine-aligned coordinates lie in the unit square. Let f0=f_Y0 be
the ACTUAL source-triangulation P1-ac interpolation of the stored257-square
incumbent vertices, and define the ideal real-arithmetic complete map

\[
F_0(x)=A f_0(x)+b.
\]

The saved vertex table, its declared diagonal, identity boundary and effective
original affine values are authoritative. Do not interpret Y0 as Q1, replace
it with a sampled512 displacement field, prolong/downsample the accepted map,
or infer an inverse from its forward vertex samples. Fixed SOURCE queries can
be located directly in the fine P1 mesh. The normal17/33/65/129/257 RAW scalar
coefficient levels may still be prolonged to this same257 output grid; this is
not resampling an accepted deformation.

Use the existing accepted512 fixed and moving RGB canvases, read with the
same PIL L/255 ordinary white-high grayscale convention as the original MA
adapter. No native-image rerender, new decoder environment, resize, stain
transform, intensity renormalization, mask, CLAHE or descriptor substitution
is part of this experiment. The matcher does NOT receive inverted grayscale
or MIND features. The unchanged registration Evidence keeps its own existing
preprocessing.

## 2. One direct moving-image sample with the declared float32 order

At fixed512 pixel centers s=(j+.5,k+.5)/512, first evaluate f0(s) by fine P1
interpolation in float64. Rendering deliberately follows the old MA affine
prewarp's FLOAT32 affine arithmetic after that evaluation:

```text
s64 = fixed512_pixel_centers(dtype=float64)
r64 = fine_P1_ac(Y0_float64, s64)
r32 = r64.to(float32)
A32 = saved_A.to(float32)
b32 = saved_b.to(float32)
v32 = r32 @ A32.T + b32
W0  = grid_sample(original_moving_gray_float32, 2*v32-1,
                  mode="bilinear", padding_mode="border",
                  align_corners=False)
```

Keep the same operation order/shape convention as digital_affine_prewarp.py.
There is ONE image sample from the original moving512 grayscale raster. P1
evaluation of a coordinate map is not an extra image interpolation. In
particular, do NOT first affine-prewarp the moving image and then sample that
prewarped image through f0; that would introduce a second image interpolation.

This arithmetic makes the identity257 map at512 pixel centers reproduce the
old affine-prewarp rendering EXACTLY on the same runtime/backend: those P1
identity queries are binary-exact and become the same float32 pixel centers.
Test this directly, including torch.equal of the rendered outputs against the
existing helper. Do not silently replace the prescribed float32 affine with
float64 affine arithmetic followed by a cast. For a nonidentity map, the
rendered image is a finite-precision sample, not an exact real-arithmetic
composition. The later float64 point conversion is deliberately not claimed
to reproduce the rounded image-query arithmetic exactly.

Run the SAME frozen released MA model on image0=fixed_gray and image1=W0,
in eval/inference mode. Preserve checkpoint/source, FP32, threshold .1,
NPE=[832,832,512,512], the actual all-thresholded coarse-pair policy, and the
released fine matching. No pair swap, mutual/cycle filter, RANSAC, new local
search radius or source/target tissue filter is introduced.

## 3. Convert new observations back to the original point coordinates

Let u_j,v_j be the network's fixed and W0 pixel-index coordinates. Define

\[
q_j=(u_j+\tfrac12)/512,\qquad
s_j=(v_j+\tfrac12)/512,\qquad
p_j=f_0(s_j).
\]

Compute p_j DIRECTLY from the stored fine P1-ac Y0 in float64 at the float64
s_j. Do not bilinearly sample a rasterized f0/F0, use a coarse control mesh,
set p_j=s_j, transform q_j through f0, or apply A twice. The adapted table is
still fixed-unit q -> affine-aligned moving-unit p. Its physical target is
A p+b in original moving-unit coordinates. A candidate ABSOLUTE vertex table Y
then satisfies, in real arithmetic,

\[
F_Y(q_j)-F_0(s_j)
 = A\bigl(f_Y(q_j)-f_0(s_j)\bigr)
 = A\bigl(f_Y(q_j)-p_j\bigr).
\]

The affine offset cancels. Thus the existing point penalty uses the SAME
dimensionless error z_j=(512/8)A(f_Y(q_j)-p_j). The rendering cast above does
not redefine these point units or cause an additional affine offset in the
loss. Use the existing production affine values promoted to float64 for
point conversion metadata, world eligibility and the point objective.

Filter/failure order is explicit:

1. Validate ALL raw network coordinates/confidences for shape, finiteness and
   confidence in[0,1] BEFORE domain filtering. Any nonfinite output or invalid
   confidence fails the extraction, as in the original adapter.
2. Convert to q,s. Discard and count only raw finite pairs for which q or s is
   outside[0,1]^2, preserving existing behavior. Do not clip or extrapolate.
3. Evaluate p=f0(s) for retained rows. A nonfinite or out-of-unit p is an
   extraction/conversion FAILURE, not another discard/clip rule. Exact legal
   geometry should keep p in the square; an actual failure must remain visible.
4. Compute original-moving eligibility e_j=1{A p_j+b in[0,1]^2}. Retain the
   rows and their original confidences; existing-loader eligibility gives
   ineligible rows zero influence. Do not test eligibility using A s_j+b.
5. Require at least8 positive-confidence eligible matches and a finite positive
   eligible confidence mass. Otherwise report extraction failure, with no
   silent old-MA/SG fallback or alternate matcher invocation.

Store q,p and original confidence in the normal point table. Retain s, or an
equivalent simple diagnostic record, to make the new coordinate conversion
recomputable. Record original input/affine/incumbent paths and truthful
registration-conditioned preprocessing metadata. No new integrity framework
is needed. Point multiplicities, support fractions, confidence masses and
source-bin coverage are reporting-only, never anatomical correctness tests.

## 4. Exactly one changed evidence term

For h in{SG,old MA,refreshed MA}, let

\[
w_{hj}=c_{hj}e_{hj}/\sum_k c_{hk}e_{hk},\qquad
P_h(Y)=\sum_jw_{hj}\left(\sqrt{1+\|z_{hj}(Y)\|^2}-1\right).
\]

The two complete functionals are

\[
\begin{aligned}
E_{\rm frozen}(Y)&=I_{\rm old}(Y)+3R(Y)+10^{-4}C(Y)+O(Y)
 +.1P_{\rm SG}(Y)+.1P_{\rm MA,old}(Y),\\
E_{\rm refresh}(Y)&=I_{\rm old}(Y)+3R(Y)+10^{-4}C(Y)+O(Y)
 +.1P_{\rm SG}(Y)+.1P_{\rm MA,refresh}(Y).
\end{aligned}
\]

The dense MIND descriptors remain the ORIGINAL frozen affine-only shared-gray
features, with unchanged image pyramid, fixed support, channel convention and
OOB term. Do not compute new dense descriptors under F0. The P1 ARAP3,
four-corner shape1e-4, original-frame OOB1, point scale8pixels, geometry boundary
and .001 floor remain unchanged. SG is unchanged. Replace MA, do not add old
and refreshed MA together or double total point strength. Existing independent
per-source normalization and fusion weight .2 implement the stated two .1
terms; a denser new table does not increase the global MA coefficient.

These are DIFFERENT functionals. The network is not proven to minimize their
joint extension, and updating the table is not an E_frozen-descent step.
Do not infer improvement by comparing E_refresh at its output with E_frozen
at another output. Cross-evaluating both saved maps under both functionals is
allowed as diagnosis, never as an arm/point-table selector.

## 5. Same-incumbent, same300-gradient suffix control

Use ALL25 existing directions from FOUR repeatedly examined DEVELOPMENT
specimens. Obtain every Y0 from the already retained globally fixed fusion300
recipe, not a per-case best method. Both arms begin from exactly the SAME
saved numerical fine vertex table and original affine.

Run one suffix per arm with the existing analytic latent Adam mechanism:
257 actual fine control vertices per axis throughout; raw coefficient levels
17/33/65/129/257; image continuation32/64/128/256/512; x then y;30gradients per
axis/level; exactly300 NEW gradients in a complete attempt. Preserve each
case's original physical learning-rate/calibration, optimizer settings,
stage-anchor rule, .95 safe fraction, fixed boundary and production acceptance.
Do not substitute the new metric solver, a simultaneous multiscale objective,
joint pose, F2 or additional cycles. No gradients flow through Y0, extraction,
model inference or the preceding optimization trajectory.

The optional incoming-map hook must set BOTH current and initial best-full
candidate to Y0, evaluating it under that arm's functional. Identity remains
the geometric REFERENCE, not a secretly reset initial map. Final selection is
the minimum own full512objective over Y0 and the accepted suffix stage states.
Do not add the incumbent's old intermediate trajectory or another arm's output
as free candidates. At every stage, reconstruct trials from its fixed accepted
anchor and the current raw coefficients; do not compound unaccepted trials.
Export the absolute Y and original A,b in the existing P1-ac format. No
correction-map composition, inverse generation or post-hoc fold repair occurs.

Default/no-incoming-map behavior must remain unchanged. Report300 NEW gradients
separately from the historical300 that produced Y0. Early stops and invalid
exports are failed/incomplete attempts even if an incoming legal map can be
retained for an explicitly labelled failure record.

Finish all25 refresh EXTRACTION attempts first, then release the matcher and
run both suffix arms. Run the frozen control for every case even if that case's
refresh extraction fails. All prediction attempts must terminate before new
annotation scoring; failures remain in the declared denominator and are not
rescued using labels, alternate tables or unreported incumbent substitution.
Complete50predictions means50successful maps only if there were no failures;
otherwise report each failed attempt explicitly.

## 6. Expected self-confirmation and what limits it

Conditioning can make the matcher simply return q approximately s, producing
p approximately f0(q) and low new point residual without correcting anatomy.
The incumbent also consumed the old MA table, so refreshed measurements are
not statistically independent observations. Resampling changes the network
input and can remove detail or create confident incorrect matches. These are
expected risks, not errors that automatically justify changing the protocol.

Limit the feedback to ONE refresh. Preserve SG and the old dense evidence.
Do not reject observations for disagreeing with Y0, reward agreement with Y0,
apply a distance-to-incumbent cutoff, confidence boost, bidirectional filter,
or multiple self-training rounds. Report the new observations' residuals at
Y0 and source/target coverage as diagnostics; near-zero residuals or abundant
matches are not an anatomical success condition. The same-incumbent frozen
suffix controls the benefit of merely spending300 more gradients. Only the
post-prediction anatomical comparison evaluates usefulness; no claim of
eliminating self-confirmation follows from this design.

## 7. Required tiny checks, one compact result table and cost scope

Independent checking before medium execution should cover:

- Identity257 rendering against the actual old affine-prewarp helper, including
  EXACT equality on the same backend for the prescribed float32 affine order.
- A legal non-affine P1 fixture and queries where Q1 differs: independently
  evaluate f0(s), direct original-raster image samples and p=f0(s).
- A nontrivial positive affine/offset: literal native-coordinate residual equals
  A(f_Y(q)-p); no extra b, inverse or second A; existing point-loss value/VJP and
  unequal eligible-confidence normalization agree.
- Raw q/s endpoints, finite out-of-domain rejection counts, all-output NaN
  failure, invalid transformed-p failure, original-world eligibility after p,
  and fewer-than8eligible failure. No clipping or hidden fallback.
- Incoming-map first evaluation/current/best selection, both arm initial maps,
  actual fixed boundary/fine corner checks, a case where Y0 remains the selected
  own-objective winner, and unchanged default-path behavior.

Then one all25-by-two attempt collection answers the registration question.
Use the existing manual-label coordinate/evaluation conventions and all eligible
annotations, only AFTER prediction. Report per-case rows plus four specimen
mean-of-pair-mean/p90 aggregates and worst tails, paired improvements/regressions,
retained errors, failure reasons, own objective/components, actual counts,
time/memory and saved-map corner/boundary results. Also retain original
fusion300 as the lower-cost reference. Do not call25directions25patients,
held-out validation, or a universal improvement from one positive cohort.

Time the incumbent load/validation, raster load/render, model setup/forward,
point adaptation/serialization, Evidence setup, suffix optimization, export and
certification with their actual scopes. Reset/report allocated peaks by stage
without summing peaks from different lifetimes. Reuse one model load across the
25extractions, reporting its cold cost once rather than claiming it is zero.
Report both incremental research cost and required deployment cost:

\[
T_{\rm refresh,pipeline}=T_{\rm common\ init/SG/oldMA}
 +T_{\rm incumbent300}+T_{\rm refresh\ extraction}
 +T_{\rm suffix300}+T_{\rm required\ output\ work}.
\]

The frozen-table suffix also requires the incumbent and its preparation.
Historical costs that were not measured in a compatible scope remain explicitly
unmeasured; do not fabricate a full end-to-end speed ratio. Existing cached
matching, the incumbent, or an optional comparison baseline is not free at
deployment. A useful result must justify the extra work relative to both the
matched frozen suffix and the cheaper globally retained fusion300 recipe.
