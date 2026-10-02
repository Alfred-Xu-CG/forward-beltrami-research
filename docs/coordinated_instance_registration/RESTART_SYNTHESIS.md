# Restart synthesis: a useful safe instance optimizer, not yet a registration breakthrough

Written 2026-10-02 by the Astra synthesis role using actual results and the
focused independent final review linked below; not original-goal completion.
The user explicitly approved a pause after handoff; recorded at
2026-10-02 16:11:57 UTC (2026-10-03 00:11:57 Asia/Shanghai), before the extended
deadline of 2026-10-02 16:26:23 UTC. The research goal is paused, not achieved.
[REPORT](REPORT.md) contains full experiments;
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
row/column $(i,j)$ to $(i+1,j+1)$. Its boundary remains identity. The residual
map must exceed the normalized corner-determinant floor .001, an extra numerical
margin beyond positivity. Five
coefficient/image levels alternate horizontal and vertical updates, totaling
300 objective gradients.

Each scalar update is $Y_i+\alpha u_i e$, with vertex amplitude $u_i$, step length
$\alpha$, and one common axis $e$ for all vertices. Both edge changes are parallel
to $e$, so the quadratic determinant
term vanishes because $\det(e,e)=0$. Corner determinants are therefore affine in
step length. The algorithm restricts that length using every adverse corner
change, then checks the actual candidate and complete objective. Four-corner
positivity, consistent connectivity and the simple preserved boundary support
the declared discrete homeomorphism onto $A[0,1]^2+b$, generally a parallelogram.
Exported coefficients receive separate sign checks. Sampled positive Jacobians alone are insufficient. Arbitrary resampling
or changing the interpolant does not inherit the certificate.

SG means SuperPoint/SuperGlue matching; MA means frozen MatchAnything ELoFTR.
Fusion uses

\[
E=D+3R,\qquad D=I+O+10^{-4}C+.1P_{SG}+.1P_{MA}.
\]

Here $I$ compares affine-aligned MIND-like eight-channel neighborhood
self-similarity descriptors; $O$ penalizes out-of-bounds image queries. Both use
fixed foreground support and a fixed denominator; difficult nonoverlap is not
silently removed. $C$ penalizes poor corner shape. $P_{SG},P_{MA}$ are separately confidence-normalized
robust correspondence errors. ARAP means as-rigid-as-possible: $R$ measures
triangle Jacobians' deviation from their closest proper rotations. $R$ and $C$
act on residual $f_Y$, before the saved affine, not on the composed map $F$.

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

| Specimen / directions | SG1 | Retained analytic fusion | F2, same fusion evidence | Native STANDARD DHR, own initialization | Native STANDARD DHR, shared saved affine |
|---|---:|---:|---:|---:|---:|
| MIIT / 3 | 3.548752 / 5.907957 | 3.534718 / 5.858170 | 3.541998 / 5.869288 | 3.672192 / 6.432444 | 3.712720 / 6.260251 |
| Lung / 20 | 4.568541 / 9.593679 | 4.471007 / 9.342538 | 4.498976 / 9.438715 | 6.046615 / 13.914145 | 6.240578 / 14.626331 |
| HistoReg / 1 | .851903 / 1.586005 | .842064 / 1.580138 | .832973 / 1.644260 | .712328 / 1.618927 | .712976 / 1.523942 |
| Kidney / 1 | 2.364866 / 4.756907 | 2.243606 / 4.643549 | 2.231108 / 4.559938 | 1.908163 / 3.178382 | 2.464310 / 5.245805 |

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

The final current-fusion analytic/F2 comparison completes all 50 calls, with
independent saved-map, objective and 4,148 annotation-error checks. Both use
identical evidence, affine, full objective and 300 gradients, with eager priors
and original point dispatch. Analytic is faster in all 25 paired observations:
average application calls are 5.568625 / 15.616001 s, and median paired F2/analytic
ratio is 3.034313. Allocated peaks are 209.70–214.30 / 481.87–485.35 decimal MB.
Calls include load/features/optimization/export/validation, excluding matching
and initializer preparation. There is only one paired observation per case,
not repeated timing or an equal-error time curve.

Anatomy is mixed: analytic wins 12/25 direction means, 17 p90s and 16 maxima;
F2 has better HistoReg and kidney means. Analytic's largest adverse specimen-mean
difference is about 1.09% on HistoReg, not a demonstrated noninferiority margin.
Crucially, **F2 reaches lower values of the same full objective on 24/25 cases**.
This supports a useful observed endpoint accuracy/cost tradeoff, not superior
objective minimization or anatomical dominance. Analytic alternates scalar axes
within one five-level cycle; F2 uses two five-level vector-update cycles with its
established corrected floor reserve. Parameter dimensions, continuation order,
patch work and geometry backends differ, preventing pure-decoder attribution.
The separate compiler gain below is not part of this F2 comparison.

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

## 7. Cost status and remaining priorities

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

A separate, independently checked 56-call experiment transfers existing Inductor
prior compilation plus frozen machine-point sampling to the unchanged fusion
objective. Warm complete-application medians, original / accelerated, are
5.010 / 2.995 s (MIIT 2-to-3), 6.138 / 4.589 s (lung HE-to-Ki67),
5.937 / 4.332 s (HistoReg), and 5.408 / 4.300 s (kidney). All twelve paired
ABBA blocks favor acceleration; median-call ratios range from 1.26 to 1.67.
The first MIIT calls with fresh local compiler/Triton caches instead cost
5.191 / 15.147 s: a real first-call penalty, not a measurement of compiler time
alone. Later cases reuse caches. Calls include loading, features, optimization,
export and validation, but exclude matcher/initializer generation and input-cache
preparation. This is dispatch engineering, not a full-pipeline speedup or evidence
that the optimizer beats F1/F2.

Full objective-component values and full vertex gradients were compared at
identity and a saved deformation at every image level in these four cases.
Fixed value tolerances are rtol/atol $10^{-8}/10^{-10}$ and gradient tolerances
$10^{-5}/10^{-8}$; observed cross-backend maxima are $6.11\times10^{-16}$ and
$1.15\times10^{-14}$. These are sampled numerical checks, not exact equality or
a bound for every input. All 56 saved maps pass independent geometry checks;
their objectives and all 4,886 original-CSV errors were independently recomputed.
HistoReg's largest warm cross-backend map
difference, $6.8735\times10^{-5}$ unit, is similar to same-backend repeat variation
$6.8703\times10^{-5}$; corresponding maximum individual TRE differences are
$2.1737\times10^{-7}$ and $2.1477\times10^{-7}$ pixels. Outputs are not bitwise
identical. The independent checker recomputed saved outputs; it audited recorded
GPU timings/VJPs rather than rerunning the performance experiment.

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

The [final independent review](BASELINE_PROTOCOL_INDEPENDENT.md#final-focused-independent-review--plan-section19-2026-10-02)
answers all nine questions in [PLAN section 19](PLAN.md#19-最终验收必须直接回答的九个问题).
Its conclusion is bounded: declared fixed-mesh topology and a useful endpoint
cost tradeoff are supported; repeated matched-time comparisons across target
classes, independent-specimen confirmation and the original trained-neural-layer
objective remain unestablished. Technical closure is not a registration breakthrough.

1. Obtain genuinely new, independently grouped labelled specimens with verified
   access and provenance. Freeze fusion and native baselines before evaluation;
   more directions from the current samples cannot provide this confirmation.
   The bounded cache inventory established no additional ready labelled histology
   specimen. COph100 retinal images/labels are cached with patient-grouped splits,
   but are cross-domain and not automatically blind; historical label exposure
   remains unaudited. They are not a substitute for histology confirmation.
2. Test a concrete finer-evidence hypothesis before opening another optimizer
   branch. One next-phase option is to expose frozen MA fine feature fields before
   coarse-gated window selection and use a dense feature-distance loss. This
   exposure and its coordinate/gradient checks are not implemented here. Compare
   one fixed extra-optimization budget against both an unchanged-$E$ suffix and
   the same feature field spatially coarsened to the eight-pixel lattice, starting
   from identical fusion incumbents. Keep masks/denominators and geometry fixed;
   optimize and select outputs without labels, then score all available IDs and
   retain failures and extraction/optimization costs. Unlike the negative raw-1024
   MIND and coarse-$C$ tests, this asks whether learned fine spatial structure
   beyond sparse coarse-gated matches is useful. Predeclare meaningful accuracy,
   cost and tail tradeoffs, interpret them against measured variation, and retain
   informative mixed outcomes rather than imposing a tiny-single-case veto.
   This is an option requiring next-phase approval, not an authorized new run.

### What the finer-feature option actually assumes

A final source-only inspection by the Astra synthesis role, independently checked
by the protocol role, resolves one implementability question without claiming a
new result. In the installed MatchAnything ELoFTR configuration, the feature
pyramid constructs two dense 64-channel fields before gathering the selected
coarse-match windows. For the accepted 512-square input pair the source-implied
shape is `[1,64,512,512]` per field. Earlier cross-attention makes both fields
depend on the **entire image pair**; they are not reusable independent per-image
descriptors. Their values do not depend on which correspondences are selected,
but the stock execution returns early if no coarse match exists, before building
these fields. Exposing them in empty-match cases therefore requires an explicit
extraction-path change, not just reading an already returned array.

This is a more precise candidate than “use the matcher features.” It means
freezing one pair-conditioned field pair after the existing saved-affine prewarp,
then testing whether their spatially varying similarity supplies useful dense
registration evidence. It does not mean differentiating through matching or
recomputing the network after every deformation. A new dense-distance loss would
also NOT be the released matcher's original training/inference objective. Its
channel selection and normalization need explicit definition: the active fine
matcher separates 56 matching channels from 8 local-regression channels.

The coordinate convention is a material unresolved issue. For an unpadded
8-pixel fixed window, a token with local index $r\in\{0,\ldots,7\}$ and coarse
index $k$ comes from dense-array index $u=8k+r$. The released point decoder uses
$x=8k+r-3.5$ before the moving-point subpixel correction. Our sparse-point adapter
then uses $(x+.5)/512$, exactly as specified for released point coordinates.
Array indices therefore cannot simply be treated as those point coordinates.
This source relation is **not a demonstrated model bug**, does not establish the
physical receptive-field center, and does not retroactively invalidate the
current sparse-point experiment. A prospective extractor must reproduce existing
point outputs and test its dense sampling convention before any registration
claim. No feature extraction, new loss, VJP or registration run was performed
for this source-only question.

Source evidence is in the installed tree
`D:/QC_optimization_data/digital_topology_wsi/matchanything_eloftr/source/`:
`configs/models/eloftr_model.py` (active branch),
`src/loftr/loftr_module/fine_preprocess.py:136,220,225,247` (early return,
fields, unfolding, selection), `src/loftr/utils/coarse_matching.py:244` and
`src/loftr/utils/fine_matching.py:455` (point coordinates). The local adapter is
[`coordinated_matchanything.py`](../../tools/coordinated_matchanything.py).
This source tree/checkpoint is an external experimental dependency, not assumed
to be present in a data-free GitHub clone.
