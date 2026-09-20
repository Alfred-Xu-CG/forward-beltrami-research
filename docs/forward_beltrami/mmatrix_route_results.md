# Route E: Beltrami tensor to M-matrix diagnostic

For a facewise coefficient `mu`, the associated conductivity tensor is

\[
A(\mu)=\frac1{1-|\mu|^2}
\begin{bmatrix}
1-2\operatorname{Re}\mu+|\mu|^2 & -2\operatorname{Im}\mu\\
-2\operatorname{Im}\mu & 1+2\operatorname{Re}\mu+|\mu|^2
\end{bmatrix}.
\]

It is SPD whenever `|mu|<1`, but SPD is not the same as a monotone discrete
operator. For the simple axis-plus-diagonal directional split used here, the
remaining axis weights are `a-|b|` and `c-|b|` for
`A=[[a,b],[b,c]]`. A negative remainder means that this stencil cannot be an
M-matrix without changing the stencil, the mesh, or the discretization.

The tests confirm both regimes:

- `mu=0.2+0.1i`: SPD and nonnegative directional weights;
- `mu=0.9 exp(i*pi/4)`: still SPD, but one axis weight is negative, so the
  simple monotone split is impossible.

This is a useful feasibility boundary for Route E. A successful M-matrix QC
decoder needs an adaptive directional stencil or a mesh condition; merely
rewriting the continuous anisotropic PDE does not provide the Tutte hard-
injectivity theorem for arbitrary `mu`.

An eigen-direction decomposition was added as a control: every SPD tensor has
strictly positive spectral conductances and reconstructs exactly from its two
eigen-directions. Those directions are generally absent from a fixed mesh
stencil, which makes the obstruction precise: continuous SPD/ellipticity is
easy, while a mesh-native M-matrix requires adaptive edges, a richer stencil,
or a mesh-alignment condition.

On a `256x256` coefficient field (`|mu|` up to about `0.93`), a sampled
finite-direction fit quantified the tradeoff. With 4,096 samples, 8 directions
had mean/max tensor residual `8.31e-2`/`5.16e-1` in `8.07 s`; 32 directions
reduced these to `3.93e-12`/`5.53e-10` in `21.66 s`. A 128-direction fit took
`57.27 s` but numerical conditioning caused a larger maximum residual
(`5.27e-3`). Thus “more stencil directions” is not automatically better: the
nonnegative fit itself needs conditioning/active-set safeguards. Receipt:
`artifacts/mmatrix_dictionary_audit/mmatrix_dictionary_audit.json`.

The conditioning safeguard now grows a small angular dictionary and stops at
the first residual target instead of using an overcomplete 128-direction fit.
For the tested anisotropic tensors it selects at most 32 directions and keeps
the nonnegative fit in the well-conditioned regime. This addresses numerical
dictionary selection, but not the separate geometric issue that those
directions may not be edges of a fixed mesh.

## Integer wide-stencil control

To test whether adaptive directions can be mesh-native on a regular grid, an
integer wide-stencil dictionary was added. Each `(p,q)` is a primitive integer
edge (up to sign), so a positive coefficient corresponds to a centered second
difference between grid vertices separated by that edge. On a `256x256`
high-anisotropy field sampled at 1,024 points, the mean tensor-fit residual
decreased from `0.583` (max step 1, 4 directions) to `0.225`, `0.090`, `0.039`,
and `0.00985` for max steps 2, 3, 4, and 6 (48 directions); all fitted weights
remained nonnegative. The max-step-6 solve also emitted a SciPy conditioning
warning, so adding directions is not free. Receipt:
`artifacts/mmatrix_wide_stencil_audit/mmatrix_wide_stencil_audit.json`.

This is concrete evidence for an adaptive regular-grid M-matrix stencil, but it
does not yet give a general unstructured-triangulation decoder: boundary
closure, mesh connectivity, and a vector-map injectivity theorem remain open.

## Manufactured-solution consistency/order audit

The regular-grid stencil was tested against a periodic smooth manufactured
solution at `128²`, `256²`, and `512²` for two high-anisotropy coefficients.
For the first coefficient, max-step `6` represented the conductivity tensor to
`2.37e-14` with nonnegative weights and converged at empirical order
`1.991/1.998`; for the second, the fit residual was `2.68e-3` and the observed
orders were `1.986/1.998`. Lower dictionaries exposed the separate cone-fit
floor: max-step `1` residuals `3.46` and `5.88` produced resolution-independent
errors, while max-step `3` residuals `5.46e-2` and `1.09e-1` produced only
partial improvement before saturation. This distinguishes genuine second-order
wide-stencil consistency from a positive-but-inexact tensor fit. It is still a
periodic constant-coefficient control; variable tensors, boundary closure, and
general unstructured meshes remain open.
Receipt: `artifacts/mmatrix_consistency_order_audit_512/mmatrix_consistency_order_audit.json`.

## Unstructured Delaunay edge-cone audit

The tensor fit was then evaluated on a genuinely nonuniform Delaunay mesh with
4,096 vertices and 8,169 triangles. At each vertex, nonnegative conductances
were fitted over directions to the one-ring and two-ring graph neighbors. The
one-ring dictionary had mean support `5.99`, median residual `1.51e-15`, but a
95th-percentile residual `1.04e-1` and only `67.4%` exact fits (threshold
`1e-8`). Adding two-ring directions increased mean support to `19.63`, reduced
the mean residual to `8.65e-3`, the 95th percentile to `2.24e-11`, and raised
the exact-fit fraction to `96.3%`; all fitted weights stayed nonnegative. This
quantifies why an adaptive graph stencil can repair the fixed-grid obstruction,
while also showing that a local cone fit is not yet a globally assembled
M-matrix or injective QC decoder. Receipt:
`artifacts/mmatrix_unstructured_stencil_audit/mmatrix_unstructured_stencil_audit.json`.

A global unstructured control has now been assembled: 4,056 vertices, 7,854
oriented Delaunay triangles, and 11,909 graph edges. Positive one-ring cone
fits were symmetrized into edge weights, a sparse two-coordinate harmonic
system was solved with 256 convex-square boundary vertices fixed, and the
independent face audit found zero flips. With a `1e-4` conductance floor, the
minimum target face determinant was `1.10e-10`; the local tensor-fit residual
mean and p95 were `0.123` and `0.743`, respectively. Thus global positive
weights do provide a hard-injective unstructured decoder control, while the
large anisotropic residual quantifies why it is not yet an exact arbitrary-μ
solver. Receipt:
`artifacts/mmatrix_global_unstructured_audit/mmatrix_global_unstructured_audit.json`.

## Variable-tensor consistency audit

The fixed-direction issue was tested with a vectorized positive-cone fit on a
spatially varying periodic tensor at `128²`, `256²`, and `512²`. In the
cone-exact case, the tensor was constructed from three positive grid-native
directions `(0,1)`, `(1,0)`, and `(1,1)`. The fit residual stayed below
`2e-15`, and the divergence-form wide-stencil operator converged at empirical
orders `1.9990` and `1.9997` between the three resolutions. In contrast, a
rotating Beltrami tensor with the same three-direction cone had residual RMS
`0.34669` and max `0.84295`; operator errors were `22.0757`, `22.0752`, and
`22.0770`, with empirical orders approximately zero. Refinement therefore
cannot remove a positive-cone representation error: variable-tensor
consistency requires a direction dictionary whose cone contains the local
tensor field, or a different discretization. Receipt:
`artifacts/mmatrix_variable_consistency_audit_512/mmatrix_variable_consistency_audit.json`.

The batch active-set fit is implemented in
`qcopt.forward.mmatrix.batch_positive_directional_conductances`; it avoids one
nonnegative optimizer call per grid node while retaining the residual as an
explicit certificate. This is a variable-coefficient periodic control, not a
boundary or arbitrary unstructured-mesh theorem.

## 65k-vertex unstructured Delaunay stress

The globally assembled positive-edge decoder was raised from the 4k-vertex
control to a genuinely nonuniform Delaunay mesh with `66,024` vertices,
`131,022` triangles, `197,045` graph edges, and `65,000` interior unknowns.
Assembly, local positive cone fitting, sparse solve, and independent face audit
completed in `117.71 s`. The boundary was exact, all weights stayed positive
(floor `1e-4`), and every face remained oriented (`0` flips); the minimum
target face determinant was `6.59e-15`. The local anisotropic fit residual
remained large (mean `0.1246`, p95 `0.7716`), so this is strong realistic
evidence for scalable hard-injective unstructured decoding, but equally strong
negative evidence against treating one-ring positive weights as an accurate
arbitrary-μ discretization. Receipt:
`artifacts/mmatrix_global_unstructured_audit_65k/mmatrix_global_unstructured_audit.json`.

## Positive wide-stencil width stress on a rotating tensor

The follow-up audit `artifacts/mmatrix_stencil_width_audit_256_512/mmatrix_stencil_width_audit.json`
tested integer directions through radius 3 on 256² and 512² periodic grids.
Every tensor was represented to machine precision (`fit_residual_max` below
`1.1e-15`, exact-fit fraction 1.0), but the overcomplete nonnegative fit was
not spatially smooth: neighbour RMS conductance variation jumped from about
`0.0063/0.0031` at radius 1 to `0.447/0.446` at radius 2, with pointwise jumps
above `2.0`. The corresponding scalar divergence-form operator error stayed
near `4.45` for radius 1 but grew to `447.2`/`897.3` for radius 2 and remained
about `446.0`/`892.5` for radius 3 at 256²/512². Thus exact local SPD cone
representation is not enough: active-set nonuniqueness can inject grid-scale
coefficient noise and destroy consistency. Any M-matrix QC decoder needs a
canonical, regularized, or globally coupled conductance selection rule in
addition to positivity; the vector-map injectivity theorem remains open.

## Spatially continuous canonicalization of the cone fit

The diagnostic selector `qcopt.forward.mmatrix_smooth` enumerates the same
nonnegative one-/two-/three-direction exact fits, but selects the candidate
closest to the already selected left and upper neighbors. It does not relax
the tensor equation: on both 256² and 512², the fit residual stayed below
`1.4e-15` and every pixel remained exactly representable. The effect on the
operator was substantial:

| grid | local active-set error | smooth-selection error | local neighbor RMS | smooth neighbor RMS |
|---|---:|---:|---:|---:|
| 256² | `447.21` | `6.71e-2` | `0.4471` | `4.23e-3` |
| 512² | `897.30` | `1.68e-2` | `0.4458` | `2.12e-3` |

The 512² run used eight integer directions (radius 2) and took `500.7 s` for
the continuation pass. A 256² reverse-column scan reproduced the forward
result exactly (`6.71e-2`), reducing concern that the improvement is only a
one-sided raster artifact. Receipts:
`artifacts/mmatrix_stencil_width_audit_256_smooth/mmatrix_stencil_width_audit.json`,
`artifacts/mmatrix_stencil_width_audit_512_smooth/mmatrix_stencil_width_audit.json`,
and
`artifacts/mmatrix_smooth_scan_audit_256/mmatrix_smooth_scan_audit.json`.

This is a meaningful positive result for a mesh-native M-matrix route: local
cone nonuniqueness can be regularized without sacrificing positivity or local
tensor fidelity. It is not yet a theorem or a production decoder—the current
selection is a raster continuation heuristic, periodic scalar-conductivity
consistency has not been turned into a vector Beltrami injectivity theorem,
and boundary closure/general unstructured meshes remain open.

## Planar barycentric enrichment

To avoid the crossing problem of a two-ring graph, a planar barycentric
subdivision was tested: every original Delaunay face receives one face-center
vertex and is split into three triangles. On `8,384` original vertices
(`49,146` refined faces), positive edge weights reduced the original-vertex
tensor-fit mean/p95 residual to `0.0192/0.0699`, while the refined map retained
zero flipped faces and minimum determinant `2.80e-7`. At the larger
`12,384`-vertex / `73,146`-refined-face run, the residuals were
`0.0187/0.0652`, zero refined-face flips, and minimum determinant `7.45e-8`.
The boundary remained exact in both runs.

This is a positive direction for Route E: adding degrees of freedom through a
planar refinement improves local anisotropic representability without the
two-ring graph's obvious global folding failure. It is not yet an arbitrary-
`mu` solver: face-center rows are currently uniform, determinant margins become
small, and a full compatibility/convergence theorem for the enriched
anisotropic system is still missing. Receipts:
`artifacts/mmatrix_barycentric_enrichment_8000/mmatrix_barycentric_enrichment_audit.json`
and
`artifacts/mmatrix_barycentric_enrichment_12000/mmatrix_barycentric_enrichment_audit.json`.

## Anisotropic face-center ablation

The same `12,384`-vertex barycentric mesh was rerun with conductivity fits at
the face-center rows instead of uniform center weights. The original-vertex
fit stayed at mean/p95 `0.0187/0.0652`, but the three-direction center cones
had mean/p95 residual `0.608/2.297`, so the all-node residual rose to
`0.414/1.918`. The refined map still had zero flips and exact boundary values,
but its minimum determinant collapsed to `1.60e-14`. Receipt:
`artifacts/mmatrix_barycentric_enrichment_12000_anisotropic_centers/mmatrix_barycentric_enrichment_audit.json`.

This ablation is important: adding anisotropic equations at every new vertex
is not automatically better. The face-center geometry has only three rays and
often lies outside the positive cone of the local Beltrami tensor. Uniform
center rows currently give the better hard-decoder tradeoff; a useful next
step is to design center locations/weights from a compatible local metric
rather than forcing an infeasible positive cone fit.

## Realistic unstructured one-/two-ring decoder comparison

The new wide-ring audit used `12,000` random interior points plus a `96`-point
per-side square boundary (`12,384` vertices and `24,382` original Delaunay
triangles). One-ring positive fits reproduced the earlier obstruction: mean
tensor-fit residual `0.1290`, p95 `0.7951`, zero original-face flips, and
minimum target determinant `2.41e-12`. Expanding the decoder graph to two-ring
directions increased the mean support from `5.94` to `19.42` and reduced the
mean/p95 fit residual to `0.00291/7.44e-12`, but the resulting wide graph is
not planar: the original-face audit then found `1,641` flipped faces and a
minimum determinant `-4.93e-4`.

This is useful negative evidence. A larger positive direction cone can repair
local anisotropic representability without automatically preserving a global
Tutte-style injectivity guarantee. The route therefore needs a planar
positive-edge construction, a crossing-aware graph embedding, or a separate
global barrier/certification layer; simply adding two-ring edges is not a
valid hard-bijective decoder. Receipt:
`artifacts/mmatrix_global_unstructured_wide_12000/mmatrix_global_unstructured_wide_audit.json`.

## Beltrami magnitude/rotation cone phase diagram

The local cone barrier was quantified on a `256²` spatially varying field with
`4,096` regularly sampled sites per case. The scan varied `rho=|mu|` over
`0.2,0.4,0.6,0.8,0.9`, angle variation amplitudes `0,0.5,1.0,1.5`, and
integer stencil radii 1--4. For `rho=0.2` and `0.4`, every tested field was
represented exactly (residuals below `2e-15`) already by the radius-1
axis/diagonal cone. At `rho=0.6`, radius 1 covered only `16.7%--23.1%` of
sites for nonconstant angle fields, but radius 2 recovered 100% exact fits. At
`rho=0.8`, radius 4 covered `70.1%--88.1%` of sites with residual means
`0.0020--0.0049`; at `rho=0.9`, even radius 4 covered only `30.3%--41.1%`
with residual means `0.0457--0.0963`. All fitted conductances remained
nonnegative by construction. Thus the obstruction is controlled jointly by
Beltrami magnitude and principal-axis angle, not by spatial variation alone.
Receipt:
`artifacts/mmatrix_cone_phase_diagram_256/mmatrix_cone_phase_diagram_audit.json`.
