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
preprocessing,7×7 NCC/diffusion-relative regularizer andfive-level optimizer differ
from the shared masked MIND/NCC objective; do not call this an isolated geometry comparison.
Correction: direct inspection of the saved executable config shows win_size=7,
not3. The earlier descriptive three-window statement was wrong; the run itself
used7. CLAHE/normalization, masking, boundary class and regularizer still differ.
No post-hoc safe repair or DHR target distillation is included in the main methods.

Data availability checks: [BIRL](https://github.com/Borda/BIRL) exposes the reused lesion
and kidney samples. [Histology landmark repository](https://github.com/Borda/dataset-histology-landmarks)
documents annotation conventions and the previously viewed lung-lesion3 sample.
The ANHIR download page returned403 and its linked older CIMA dataset page404 during
this run. No authentication bypass was attempted. Independent new labelled-specimen
confirmation remains required; these failures do not block existing-data development.

## Precision, conditioning and continuation diagnostics

The mixed option uses float64 exported geometry and float32 original image evidence.
Coordinates are cast only at raster sampling, not in the stored map. Full-float64
F2 removes float32 floor contacts at gain.75 but uses about513MB allocated peak;
mixed F2 uses about464--465MB here. Neither result establishes an anatomical gain.
An optional shared symmetric-Dirichlet corner weight1e-4 conditions thin cells;
first12-case matrix completes all100 gradients, but anatomy does not consistently
improve. It is not a substitute for the exact feasible decoder or saved certificate.

Image continuation uses32/64/128/256/512 queries, associated with17/33/65/129/257
coefficient levels. All maps still have257² control vertices throughout. Each
original raster is independently area-downsampled; mask area weights are conserved
for the power-of-two divisible raster sizes used here, not arbitrary resize ratios.
Every stage accepts against the COMPLETE objective at that image resolution. Thus
full512 objective can increase when changing resolution; it is separately traced.
Pyramid and nonpyramid runs are different algorithms, not the same fixed objective.

Known-map texture control: fixed image is generated from the ORIGINAL histology
moving raster through an independently analytic, actually digital-valid257 Q1 target.
Identity initializer, three targets with peak37--61px movement; shared image-only
optimization never reads the target. The PNG intensity quantization floor is reported.
Off-raster query seed20261001/count4096 is used ONLY after the selected final output.
These are synthetic deformations of real texture, not biological registration or
independent patients. Target MIND objective is nonzero because descriptor transport
does not commute with image deformation, while the raster truth error is near the
PNG quantization floor. A true target lower than optimized full objective is useful
evidence of convergence/capture failure, not proof that MIND's optimum is anatomical.

Actual P1(ac) is now separately executed: query weights and offline scorer use the
source-cell a--c diagonal throughout all stages,131072triangles at257² vertices.
The exported archive declares its interpolation; old Q1 archives remain Q1 evidence.
Public P1 query APIs reject nonfinite/out-of-domain source coordinates rather than
silently extrapolating a certified interior function. A new OPTIONAL best_full output
rule selects the initial/accepted-stage map with minimum COMPLETE512 objective.
It changes no stage trajectory, never reads manual labels, and counts the existing
per-stage full-objective evaluation rather than inventing zero-cost validation.
The legacy last-stage rule remains explicitly selectable for paired comparison.
