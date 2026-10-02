# Terminal original-image detail with unchanged dense P1 geometry

Status, initially approved2026-10-02 10:46 UTC; independent prechecks and
actual-source smoke now pass, production started11:19UTC. No accuracy result yet. The preceding
joint-pose comparison gives a lung-specific gain but mixed cross-specimen tails
and higher cost; it is not a uniform replacement for frozen-pose fusion300.
The following uses ONE frozen-pose fusion300 recipe on all25 directions, not
the best pose choice separately on each specimen. Independent checking of the
joint result is complete and agrees with the mixed result.

## Question and falsifiable scope

Does information retained in the original image files improve terminal local
registration, beyond changing raster density and the physical descriptor
footprint? Compare direct-original1024 rendering with a deliberately enlarged
accepted512 raster at the SAME1024 query density. Both retain257-by257 control
vertices,131072 source triangles and262144 protected corner determinants.
The ordinary512 fusion300 result is a third, archived context.

This is not an attempt to reconstruct the large MIIT tail by using annotations.
The worst annotated displacement is opposed by local automatic correspondences,
and a similar error survives the released higher-resolution DHR pipeline. No
annotation-dependent region, correspondence deletion or parameter tuning is used.

## 1. Coordinate-preserving rendering

For each image, let W=(W_x,W_y) be its original dimensions, n=(n_x,n_y) the
accepted resized dimensions, and p=(p_x,p_y) the accepted integer canvas
padding. The accepted canvas side is512. A zero-based original pixel-center
coordinate x maps to unit coordinates

\[
u=\frac{(x+\tfrac12)\odot(n/W)+p}{512},
\]

where multiplication and division inside the numerator are componentwise.
Direct-original1024 uses resized dimensions2n, padding2p and canvas side1024.
Consequently the SAME original point has the SAME unit coordinate u. The
affine arrays, source/target machine points, control grid and evaluator remain
unchanged. In raster pixel indices the relation is c'=2c+1/2, not c'=2c.

For each role, construct two float32 inverted-grayscale rasters:

- **Direct-original1024:** read the original source file, convert to RGB,
  resize directly to2n using PIL BILINEAR, and paste into a white1024 canvas
  at2p. Apply the existing PIL convert('L') and1-gray/255 preprocessing once.
  Do not enlarge the old512 image for this arm.
- **512-information lift:** load the accepted complete512 image through the
  UNCHANGED raw grayscale-inversion loader, then bilinearly enlarge that
  float32 intensity to1024 with torch interpolation, align_corners=False.
  Do not introduce a second uint8 rounding or RGB/grayscale conversion. This
  adds no source information.

Neither arm is a mathematically pure frequency filter. Direct rendering can
also change low-frequency resampling, grayscale quantization and padding-edge
blending. Save prepared intensities as float32 arrays with explicit metadata;
do not round them back through a PNG. The first four
optimization rasters, however, are constructed from the UNCHANGED accepted512
images, not downsampled from either new1024 rendering.

Actual-source precheck amendment, 2026-10-02 11:10 UTC: the old lung JPEG canvases
were decoded with Anaconda Pillow10.3/JPEG9, whereas the current numerical
local numerical environment uses Pillow12.3/JPEG8. This changes decoded source RGB and breaks
pixel-exact512 reconstruction despite identical dimensions and PIL BILINEAR
resizing. On both checked lung roles, current resizing of the HISTORICALLY
decoded RGB reproduces the accepted512 canvas EXACTLY; this isolates decoding,
not the coordinate frame or resize kernel. Preserve the strict reconstruction
check. Resolve each original JPEG's historical decode explicitly in a PIL-only process,
save lossless full-resolution RGB PNGs with original source identity/dimensions
and decoder metadata, then use those RGB arrays in the ordinary current
renderer. No Torch/GPU optimization runs in Anaconda. TIFF sources remain
unchanged. Seven sources (lung/HistoReg) use the old10.3/JPEG9 decoder; the two
kidney sources use the local12.3/JPEG8 decoder that exactly reproduces their
accepted canvases. The remote10.4/JPEG9 environment cannot reproduce kidney
without this bridge, even though the local current environment can. Store the
decoder provenance PER SOURCE and cache all nine full-resolution RGB images,
not just the seven first identified. The old512 inputs/masks and all archived maps are never rewritten.
This explicit decoding bridge is part of preparation cost, not new anatomy,
fresh matching, a hidden fallback or a relaxed equality tolerance.

All25 directions remain. The five lung JPEGs are approximately892--895 by660--661
pixels, so parts of direct-original1024 rendering UPSAMPLE their available
pixels. HistoReg originals are about7k by10k; kidney about1.1k by.7k; MIIT about3k.
Record every source dimension, target resized dimension and per-axis resize
ratio. Call the arm direct-original rendering, not uniformly native-resolved
1024 detail. The old no-upsampling contract in coordinated_scale_canvas.py is
unchanged; this is an explicit different experiment, not a relaxed hidden check.

## 2. Fixed support and unchanged point evidence

Let m512 be the ORIGINAL fixed-image grayscale-inversion>.04 mask already used
by fusion300. At1024 repeat every entry into its2-by2 child pixels:

\[
m_{1024}[2i+a,2j+b]=m_{512}[i,j],\qquad a,b\in\{0,1\}.
\]

Use this SAME fixed support in both arms; its normalized integral is unchanged.
Lower-stage masks remain the original area-downsampled512 masks. No new tissue
segmentation, warped support, adaptive denominator or query filtering enters.

The frozen SuperGlue and MatchAnything tables were predicted at512. Preserve
their unit points, confidences, original-A eligibility and separate normalized
means. For residual P1 map f_Y and original positive affine Az+b, the unchanged
point term for either table is

\[
P(Y)=\sum_jw_j\left[
\sqrt{1+\left\|\frac{512A(f_Y(q_j)-p_j)}8\right\|^2}-1\right].
\]

The combined weight remains .1P_SG+.1P_MA. If any interface represents distances
in1024 pixels instead, its robust scale MUST be16, yielding the identical
ratio1024/16=512/8. No fresh matching or re-fitting of A is performed. Keep the
point-table metadata honest: new raster evidence does not mean new1024 matches.

## 3. Objective and exact stage schedule

Write Phi for the implemented eight-channel MIND-like descriptor defined in
JOINT_POSE_FORMULATION section4. For raster side r, first prewarp the moving
intensity ONCE by the frozen A,b on that raster's pixel centers, then compute
Phi. The fixed descriptor is computed on the corresponding fixed intensity.
Sampling both descriptors on the fixed-frame residual uses f_Y evaluated as
P1 on the fixed source triangles; moving raster samples are bilinear with
zeros and align_corners=False. The dense mean is

\[
I_r(Y)=\frac{\sum_xm_r(x)\|
\Phi(I_{f,r})(x)-\Phi(I_{m,r}\circ(A\cdot+b))(f_Y(x))
\|_1/8}{\sum_xm_r(x)}.
\]

The complete objective remains I_r+3ARAP(Y)+1e-4Shape(Y)+.1P_SG+.1P_MA+OOB_r,
with the same actual P1 ARAP, corner-shape penalty, complete-map out-of-bounds
coordinates and normalized-coordinate penalty. No global pose variables are
optimized in this experiment.

Coefficient sides are17,33,65,129,257. Their image sides are32,64,128,256,1024.
The first four are exactly the old accepted512-derived intensities, masks and
features. The last uses the respective1024 arm. Each level has30 horizontal
and30 vertical residual gradients, totaling300. Coefficient learning rates
remain .004*16/(level-1), not doubled. The same analytic common-direction safe
operator retains the strict residual corner floor .001 and fixed unit boundary.

Every stage accepts using its complete CURRENT-resolution objective. Select
the initial or accepted-stage map with minimum complete1024 objective within
that arm. This full-resolution selector is different from ordinary512; the two
1024 arms share that change. Store actual query_count=1024^2 and image levels,
never relabel the control grid1024. Evaluate f_Y directly at each new pixel
center ((j+.5)/1024,(i+.5)/1024); do not resample/reconstruct a new vertex map.
Query locations on a source edge share the same continuous P1 value. Source
queries are fixed, so there is no dynamic moving-triangle location problem.

## 4. Decisive checks, costs and interpretation

Implementation uses a narrow optional terminal-evidence argument to the existing
optimizer. Its absent/default branch stays unchanged. With the explicit override,
original image loading and correspondence validation retain source side512,
while the reported final image side, final stage and query count are1024. The
prepared override carries actual arrays and source metadata, not a monkeypatch
of image loading. The old strict native-render helper remains untouched.

Before the full run, check one nonsquare padded example's coordinate equality,
512/8 versus1024/16 point-loss and vertex-gradient equality, fixed-support
normalized mass, and exact preservation of lower-stage image tensors. A tiny
same-raster override must reproduce the old optimizer's objective/updates.
The actual saved output still needs the ordinary boundary/corner certificate.
No broad legacy suite or separate integrity framework is introduced.

Both arms attempt all25 directions before manual scoring. Retain failures and
the same required labels. Report four specimen mean/p90 rows, every paired
regression, maxima, complete-call time and allocated peak memory. Render/setup
time is separate and not treated as free. Metrics remain512-equivalent moving
canvas pixels plus separate native-image pixels, not a new1024-unit score.

If direct-original improves over BOTH controls, this supports useful available
source detail for this recipe, with cost and resampling limitations disclosed.
If both1024 arms improve similarly, the gain is not established to come from
additional source information. If neither improves, retain the negative result;
do not automatically proceed to2048 or a parameter sweep. Mixed specimen results
remain mixed; do not select a different arm per case using landmarks.
