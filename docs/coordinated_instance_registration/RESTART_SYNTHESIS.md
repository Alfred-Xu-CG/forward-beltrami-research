# Restart synthesis: a useful safe instance optimizer, not yet a registration breakthrough

Written 2026-10-02 by the Astra synthesis role using actual results and independent
checks; not goal completion or the final review. The authorized window ends at
2026-10-02 16:26:23 UTC. [REPORT](REPORT.md) contains full experiments;
[BASELINE_PROTOCOL_REVIEW](BASELINE_PROTOCOL_REVIEW.md) resolves comparison and
provenance details.

## 1. What question was actually answered?

The original research ambition is a fast, memory-efficient, differentiable neural
layer mapping learnable latent variables to guaranteed-bijective discrete maps.
The current approved phase deliberately asks a nearer question: can that geometry
support accurate, economical **per-case image registration**? The restart
prioritized faithful baseline comparison, then justified algorithm changes on
actual histology images.

The resulting evidence supports a practical, topology-preserving instance
optimizer and a modest correspondence-fusion improvement. It does **not** establish
that a trained neural encoder generalizes, that backpropagating through the entire
optimization trajectory is practical, or that the original neural-layer objective
has been achieved. Frozen pretrained matchers supply observations; they are not
a newly trained registration network.

All principal application results concern **25 directions from four repeatedly
viewed development specimens**: three MIIT serial-section directions from one
prostate sample, twenty ordered stain directions from one lung specimen, and one
HistoReg and one kidney direction. There are 2,074 available paired annotation
observations, not 2,074 independent subjects. No new held-out specimen result,
clinical validation, official challenge score, or state-of-the-art claim is established.

## 2. What the retained method does

The sampling map is fixed-to-moving, $F(x)=A f_Y(x)+b$: $x$ is a source unit-square
coordinate, $A,b$ are the saved image-derived affine with positive determinant,
and $f_Y$ interpolates deformed vertex positions $Y$. P1 means piecewise affine,
linear on each triangle. The fixed 257-by-257 grid uses AC diagonals connecting
row/column $(i,j)$ to $(i+1,j+1)$. Its boundary remains identity. Five
coefficient/image levels alternate horizontal and vertical updates, totaling
300 objective gradients.

Each scalar update is $Y_i+\alpha u_i e$, with vertex amplitude $u_i$, step length
$\alpha$, and one common axis $e$ for all vertices. Both edge changes are parallel
to $e$, so the quadratic determinant
term vanishes because $\det(e,e)=0$. Corner determinants are therefore affine in
step length. The algorithm restricts that length using every adverse corner
change, then checks the actual candidate and complete objective. Four-corner
positivity, consistent connectivity and the simple preserved boundary support
the declared discrete homeomorphism; exported coefficients receive separate sign
checks. Sampled positive Jacobians alone are insufficient. Arbitrary resampling
or changing the interpolant does not inherit the certificate.

SG means SuperPoint/SuperGlue matching; MA means frozen MatchAnything ELoFTR.
Fusion uses

\[
E=D+3R,\qquad D=I+O+10^{-4}C+.1P_{SG}+.1P_{MA}.
\]

Here $I$ compares affine-aligned MIND-like eight-channel neighborhood
self-similarity descriptors; $O$ penalizes out-of-bounds image queries; $C$
penalizes poor corner shape. $P_{SG},P_{MA}$ are separately confidence-normalized
robust correspondence errors. ARAP means as-rigid-as-possible: $R$ measures
triangle Jacobians' deviation from their closest proper rotations.

Manual evaluation landmarks never enter these optimizations or output selectors.
However, repeated post-run evaluation influenced development decisions, so the
result is not blind validation.

## 3. The improvement worth retaining

SG1 denotes the preceding shared-affine-frame recipe with only $0.1P_{SG}$.
Fusion preserves it and adds $0.1P_{MA}$.
The control SG2 doubles SG alone to $0.2P_{SG}$. Inputs, saved affine, geometry,
priors, and the 300-gradient schedule stay common.

TRE (target-registration error) is Euclidean distance from a mapped annotation
point to its paired annotation. Entries are mean TRE / mean per-direction
90th-percentile TRE, in 512-equivalent moving-canvas pixels. Directions are
averaged within specimens. DHR abbreviates DeeperHistReg.

| Specimen / directions | SG1 | Retained fusion | Native STANDARD DHR, own initialization | Native STANDARD DHR, shared saved affine |
|---|---:|---:|---:|---:|
| MIIT / 3 | 3.548752 / 5.907957 | 3.534718 / 5.858170 | 3.672192 / 6.432444 | 3.712720 / 6.260251 |
| Lung / 20 | 4.568541 / 9.593679 | 4.471007 / 9.342538 | 6.046615 / 13.914145 | 6.240578 / 14.626331 |
| HistoReg / 1 | .851903 / 1.586005 | .842064 / 1.580138 | .712328 / 1.618927 | .712976 / 1.523942 |
| Kidney / 1 | 2.364866 / 4.756907 | 2.243606 / 4.643549 | 1.908163 / 3.178382 | 2.464310 / 5.245805 |

Fusion improves all four specimen means and p90s against both SG1 and SG2; mean
reductions from SG1 are approximately 0.4%, 2.1%, 1.2%, and 5.1%. Simply doubling
SG does not reproduce this pattern. This supports useful complementary evidence
in the tested recipe, not universally superior matching. Five of 25 direction
means worsen, kidney's maximum rises slightly, and MIIT retains an approximately
53-pixel worst error. These are substantial unresolved failures despite small
aggregate gains. Fusion remains one global recipe, not a landmark-selected
collection of per-case winners.

SG1 and fusion have different objectives: their reported energies must not be
treated as convergence measurements of one functional. The native comparisons
also differ in image processing, resolution, loss, regularization and boundaries;
even shared initialization does not isolate the topology layer. Fusion wins some
comparisons but does not uniformly beat native DHR, especially its HistoReg mean
and own-initialization kidney result.

## 4. What additional interventions actually established

Several plausible changes executed successfully without yielding a general
accuracy/cost advance: simultaneous multiscale terms, tissue-support changes,
normalized-gradient-field (NGF) and stain-proxy evidence, longer Adam optimization, stiffness preconditioning,
coupled finite-displacement MIND seeding, MA replacement, and later refinements.
Earlier joint-coordinate, limited-memory quasi-Newton (L-BFGS), and two-component
patch-update (F2) results remain negative evidence for their tested configurations,
not impossibility theorems.

Joint positive global pose improved the lung mean by about 3.84%, but worsened
other means or tails; kidney's maximum rose to 11.497 pixels. Original-image
terminal 1024 detail and its enlarged-512 control both worsened all four specimen
means. This tested additional terminal image information with the same 257 grid,
not a 1024-control map or every possible high-resolution algorithm. One
deformation-conditioned MA refresh worsened all four means versus original fusion.

The same-prefix timed experiment provides cleaner convergence evidence: ordinary
Adam lowered the unchanged full fusion objective on 25/25 cases; a data-aware
metric lowered it on 24/25. Neither supplied a consistent anatomical improvement.
Mean energy decreases were .00170144 and .00107103 respectively. Both largely
reduced image/ARAP terms while often worsening the point term. Lower objective
is therefore not a reliable anatomical selection signal along these trajectories;
this does not prove convergence or characterize the global minimum.

The distortion-budget experiment instead minimized unchanged data $D$ subject
to $R\leq R(Y_0)$, using the fusion incumbent's actual ARAP as the fixed budget.
A frozen-rotation quadratic majorizer, exact sine-transform spectra and scalar
dual search produced cap-feasible directions, followed by actual geometry,
distortion and descent checks. All 25 outputs passed; 7,272 steps were accepted.
Data decreased without buying progress through larger total ARAP. Yet means
improved on only 14/25 directions: kidney improved to 2.179386, while its maximum
worsened to 9.405735. Required incumbent-plus-suffix calls averaged 13.575 seconds
before missing historical preparation, versus 4.566 seconds for the incumbent
call. This is a working constrained solver, not a practical breakthrough or a
stationarity certificate; 35/250 stages exhausted their line searches.

## 5. Corrections that materially change interpretation

The old “native DHR” was a **shared-affine reduced-512** configuration with five
30-iteration levels. It was not the released full pipeline. The restart executed
the actual fast preset (2048, seven 100-iteration levels) and STANDARD preset
(4096, eight levels totaling 900 iterations), retaining native preprocessing and
initialization. STANDARD subsequently covered all 25 directions; the separate
shared-affine replay changed its initializer only. Native saved fields have
nonpositive local corners, but this is not proof that every tissue region folds
or that the continuous method was globally certified.

The common initializer is **not uniformly direct SG**. MIIT uses disclosed
four-quarter-turn SG similarity selection; lung uses direct SG similarity.
HistoReg and kidney use recovered historical DHR **initial-only** outputs, with
nonrigid registration disabled. Exact archive reconstruction verifies their
saved affines. They are image-only initialization, not competing nonrigid teachers.
HistoReg's actual canvas matrix is not an exact similarity; the valid general
assumption is positive determinant. Later SG point tables are a separate input.
This corrects provenance labels without changing any map or score.

Coordinate checks retain x/y versus array ordering, half-pixel canvas conversion,
the fixed-to-moving map direction, and one application of the affine. PIL's
dimension-derived resize scales and DHR's nominal resampling ratio are different
conventions. Current canvas scores are not physical micrometers or official ANHIR
rTRE. Missing released annotations are distinguished from prediction failure;
all methods retain identical available IDs.

JPEG decoding also mattered: historical canvases used different decoder
environments. Lossless decoded RGB caches now reproduce accepted 512 canvases
exactly. No source image or accepted canvas was replaced. Interrupted cache setup
means a complete fresh-from-empty setup time remains unmeasured.

## 6. What the final information diagnostic does not prove

Unchanged MA exposes a 4096-by-4096 coarse confidence matrix on an 8-pixel lattice.
Its extractor already retains all thresholded coarse pairs, not coarse top-1.
All 25 diagnostic forwards reproduced original point tables bitwise. Comparing
anatomical targets with exact fusion predictions retained all 2,074 IDs; 2,054
were supported inside the coarse-center hull, with 20 explicitly unsupported.

Bilinear scores favored truth 707 times and prediction 1,347 times. A four-corner
maximum sensitivity favored truth 74 times, prediction 333 times, and tied 1,647
times. All four specimen mean rank advantages were nonpositive. The 1,682 ordering
differences include ties, not 1,682 reversals. This supplies no affirmative reason
to build another coarse-assignment optimizer now. Heavy coarse-grid ambiguity
prevents inferring “no information”: it says nothing definitive about finer
features, other matchers, annotation correctness, or achievable anatomical accuracy.

Generic safety choking is likewise unsupported: 22/25 original fusion runs never
contracted a recorded trial, and cap corner restrictions concentrated in two lung
directions. Capacity witnesses establish attainable labelled point fits, not an
image-only route to them. The defensible diagnosis is an unresolved interaction
of evidence, regularization, boundaries and optimization, not one proven universal bottleneck.

## 7. Cost status and three priorities

The independently checked conditional batch completed all 25 attempts in
149.517736 s: 5.980709 s/attempt, .167204 maps/s, and first saved map at
16.337309 s. It stages 25 SG setups/extractions, 25 MA extractions with one setup, then
fusion/optimization. Allocated/reserved peaks are 1.028263424/1.837105152 GB.
All 75 tables match archived values exactly. Maps are not bitwise identical
(maximum component difference 2.472e-5 unit), but maximum individual TRE drift
is 1.414e-7 canvas pixels. These are batch/amortized measurements, not per-case
cold latencies: historical affine generation and decoded-input preparation remain
excluded. Memory measures PyTorch allocations/reservations, not board capacity;
phase peaks are not summed. No equal-scope end-to-end speedup claim follows.

The additional three-pair MIIT reverse-direction check completed all nine
SG1/fusion/native STANDARD calls, sharing the inverse saved affine within each
task. Independent checking passed for all 984 method errors plus 328 affine
errors against the original CSVs. Mean / mean-p90 are 3.562529 / 5.672112 for
fusion, 3.565420 / 5.857085 for SG1, and 3.567001 / 5.983648 for native STANDARD
with that shared initializer. Fusion improves two of three direction means and
p90s versus SG1, but worsens 11-to-10 and two direction maxima. Its worst error,
52.045358 pixels, still exceeds the affine-only 51.142462. This is modest
directional robustness on the same prostate specimen, not independent-specimen
confirmation or inverse consistency. Forward and reverse residual domains and
fixed-boundary classes need not be inverses. The initial filename-related failed
attempt is retained separately; it is not part of the successful rerun timing.

1. Finish the pending same-functional engineering transfer check: actual Inductor
   priors plus frozen point sampling versus original dispatch on current fusion.
   Preserve first-call costs, require full-value/VJP agreement, and independently
   check the resulting maps and measurements before any speed claim. Final
   independent review remains pending.
2. Obtain genuinely new, independently grouped labelled specimens with verified
   access and provenance. Freeze fusion and native baselines before evaluation;
   more directions from the current samples cannot provide this confirmation.
3. Reopen algorithm design only around a discriminating new failure or useful
   finer anatomical evidence. Require both accuracy and cost benefit; do not
   substitute another lower development loss for progress toward the neural goal.
