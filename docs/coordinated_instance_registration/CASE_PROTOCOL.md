# Initial declared development protocol

This protocol precedes new confirmation selection. The first matrix contains three
already viewed specimens: HistoReg CD68→CD4, ANHIR/BIRL lung lesion proSPC→HE,
and rat kidney PanCytokeratin→HE. These are development evidence, not three human
patients or untouched tests. All pair metrics retain every shared landmark ID:
77 HistoReg, 78 lesion, 69 kidney. Kidney's two fixed-only IDs are separately disclosed.

All methods use the existing jointly scaled/padded 512-square canvases and EXACT
same stored positive image-only initial affine. Initializers were not fitted with
these labels. Fixed→moving maps query original moving evidence. The residual has
257² vertices (66,049), 65,536 cells and 262,144 digital corner constraints; batch1.
It has an identity outer boundary; the complete output is its affine postcomposition.
First image optimization/evaluation is Q1, not P1. Both P1 diagonal interpretations
are supported by the corner signs, but are not the same sampled image function.

Objective: mean MIND absolute descriptor difference OR one-minus-squared local NCC
(7×7 window), averaged on the fixed inverted-grayscale>.04 mask. The denominator
is fixed throughout optimization. NCC is insensitive to correlation sign. Padding
is zero; queries outside [0,1] are retained, and squared normalized-coordinate excess
has weight1. Cumulative residual edge strain has weight.05. No moving-mask selection,
landmarks or competing dense field are used by the new optimizers. MIND descriptor
orientation and noncorresponding tissue can still be genuine evidence limitations.

Initial rates are physical coefficient units .004 at17², scaled16/(level−1) for
33,65,129,257. Coefficient boundary is zero before interpolation, giving a coarse
boundary transition. Each stage fixes its accepted anchor, optimizes five Adam steps
from zero and evaluates the sixth/final candidate. Selection is by the COMPLETE
objective, not by labels or image-only score. Radial/analytic alternatex/y overtwo
cycles; joint-component F1/F2 usefourcycles, matching120 trials/100 gradient updates
when no failure. This does not match scalar-coordinate work or geometry-pass counts.
F1 uses current-edge basis, F2 fixed-h patches with staggered overlap calibration;
physical normalization is exact infinitesimally at identity, not at every deformed
anchor. F1/F2 safety_fraction=.75; analytic theta=.95; all share normalized floor.001.
Rejected rounded-margin candidates and incomplete gradient budgets remain reported.

Metrics: Euclidean fixed-landmark→predicted-moving-landmark error in moving canvas
pixels, using independent NumPy Q1 interpolation and saved half-pixel resize/padding
metadata. Also report original moving JPEG pixels (different scales across specimens,
so do not average native-pixel means as if they were comparable). Predeclare equal
specimen mean, per-pair p90 and max, worse-than-affine cases and failed/time-out cases.
Do not treat individual landmarks or several stain directions as independent patients.
These are NOT official ANHIR rTRE or ACROBAT micrometre leaderboard scores.

Timing: full staged optimizer includes setup/trials/Adam and geometry/objective/VJP.
Loading, evidence extraction, common affine initialization and saved exact-sign
certificate are separate components. Record allocated peak, not GPU capacity as
memory usage. Cold/warm runs and nondeterministic CUDA sampling-backward variation
must be distinguished; a repeated Histo radial run already differs in TRE (.798/.837).

Native DHR baseline has TWO variants: newly estimated native affine (application
pipeline only), and exact supplied common affine before native nonrigid optimization.
The common-affine conversion is independently verified over512² centers. Native
preprocessing,3×3 NCC/defaultregularizer andfive-level optimizer differ from the
shared seven-window/MIND objective; do not call this an isolated geometry comparison.
No post-hoc safe repair or DHR target distillation is included in the main methods.

Data availability checks: [BIRL](https://github.com/Borda/BIRL) exposes the reused lesion
and kidney samples. [Histology landmark repository](https://github.com/Borda/dataset-histology-landmarks)
documents annotation conventions and the previously viewed lung-lesion3 sample.
The ANHIR download page returned403 and its linked older CIMA dataset page404 during
this run. No authentication bypass was attempted. Independent new labelled-specimen
confirmation remains required; these failures do not block existing-data development.
