# Native multi-chart high-genus registration validation

Date: 2026-08-28

## Outcome

The prototype implements a genuine native PL-atlas refinement method for
closed same-genus surfaces.  It stores one global map state per sample as
`(target face id, barycentric coordinates)`.  A point crosses target triangles
by deterministic edge walking and intrinsic unfolding, so chart membership is
updated during optimization and there is no preassigned chart correspondence,
cut seam, Poincare disk, or universal cover.

The complete primary matrix is 4 cases x 3 seeds x 3 modes = 36 trials:

- generated, smoothed exact-genus-2 double torus;
- official genus-3 pair from `Fig11_genus/3`;
- official genus-5 pair from `Fig11_genus/5`;
- official pretzel genus-3 pair from `Fig12_texture/pretzel`;
- seeds 3, 11, and 29;
- landmark-only, curvature-only, and combined objectives.

Every run starts from a smooth diffeomorphic perturbation of a topology-valid
base map and uses 192 held-out surface samples, 20 exposed landmark pairs, 10
registration iterations, a maximum update of `0.05` median edges, at least 8
integration substeps, and a CFL cap of `0.15`.

## Independently recomputed results

| case | mode | audits | certificates | dense improvements | mean initial | mean final | mean improvement |
|---|---|---:|---:|---:|---:|---:|---:|
| official genus-3 | landmark | 3/3 | 3/3 | 3/3 | 0.2552 | 0.1955 | +23.4% |
| official genus-3 | curvature | 3/3 | 3/3 | 0/3 | 0.2552 | 0.2837 | -11.4% |
| official genus-3 | combined | 3/3 | 3/3 | 3/3 | 0.2552 | 0.2169 | +14.9% |
| official genus-5 | landmark | 3/3 | 3/3 | 3/3 | 0.2157 | 0.1949 | +9.6% |
| official genus-5 | curvature | 3/3 | 3/3 | 0/3 | 0.2157 | 0.2229 | -3.4% |
| official genus-5 | combined | 3/3 | 3/3 | 2/3 | 0.2157 | 0.2129 | +1.2% |
| official pretzel genus-3 | landmark | 3/3 | 3/3 | 3/3 | 0.2178 | 0.1992 | +8.6% |
| official pretzel genus-3 | curvature | 3/3 | 3/3 | 0/3 | 0.2178 | 0.2524 | -16.5% |
| official pretzel genus-3 | combined | 3/3 | 3/3 | 2/3 | 0.2178 | 0.2185 | -0.5% |
| generated genus-2 | landmark | 3/3 | 3/3 | 3/3 | 0.2708 | 0.2255 | +16.7% |
| generated genus-2 | curvature | 3/3 | 3/3 | 3/3 | 0.2708 | 0.2484 | +8.3% |
| generated genus-2 | combined | 3/3 | 3/3 | 3/3 | 0.2708 | 0.2355 | +13.1% |

Errors in the table are normalized by target median edge length.  Overall,
36/36 independently reloaded trials matched their saved metrics and flow
states, 36/36 passed the numerical homeomorphism audit, and 25/36 improved the
held-out dense correspondence.  Real published pairs account for 16/27 dense
improvements; the generated genus-2 case accounts for 9/9.

The global worst audited values were:

- minimum orientation ratio: `0.2459048776`;
- maximum forward/inverse round-trip error: `0.0036297933` minimum target
  edges, below the declared `0.005` tolerance;
- maximum representative Lipschitz-CFL policy ratio: `0.1499743816`, below
  `0.15`;
- maximum common-overlay projection error: `8.2200753e-7`;
- maximum reloaded-flow replay discrepancy: `9.1719474e-16` target coordinate
  units.

## What the method actually establishes

The initial common-refinement map supplies a continuous oriented base
homeomorphism.  Optimization composes it with small target-surface flows built
from edge-compatible quadratic tangent fields.  C0 consistency is automatic
because overlapping charts are only coordinate views of the same global
surface point; no duplicated seam variables exist.

The numerical certificate rechecks equal closed oriented topology, base-map
projection, local target-chart orientation, CFL/Lipschitz limits,
forward/inverse flow replay, and trajectory health.  `degree_one` is inferred
from composition of the base homeomorphism with these reversible
orientation-preserving flows.  It is not exact-predicate preimage counting and
must not be described as a formal proof.

The CFL value uses the same first-percentile representative face scale in the
integrator and independent auditor.  A direct P2+bubble Jacobian sampling check
on a genus-5 primary flow found a maximum local spectral norm `35.22` versus
the policy bound `63.84` (per-substep ratio `0.0827`), but this remains
floating-point numerical evidence rather than a formal all-triangle derivative
bound for arbitrary sliver meshes.

## Negative results and remaining research problem

Landmark refinement is robust on all three real pairs (9/9 improvements).
Curvature-only matching reduces its own smoothed Gaussian/mean-curvature
residual, yet worsens held-out correspondence in every real trial (0/9).  The
combined objective improves 7/9 real trials, but its pretzel mean is slightly
negative.  Curvature alone is therefore neither discriminative nor robust
enough for automatic correspondence.

The most important unsolved component is automatic `F0`: discovering a
topology-valid initial map and homotopy class for unrelated raw meshes without
using a published common refinement.  This implementation validates the
native multi-chart **refinement and topology-preservation mechanism**, not a
complete end-to-end raw-surface matcher.  Strong next steps are intrinsic
feature/functional-map initialization, cycle-consistent discrete homotopy
search, QC/Jacobian regularization of the flows, and exact or interval-based
certificate predicates.

## Reproducibility artifacts

After installing the official archive at the inventory path, reproduce the
fixed primary budget and independently audit the resulting 36 directories:

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:PYTHONPATH='src'
& 'C:\Users\xuzhehao\anaconda3\python.exe' -m qcopt.experiments.high_genus_registration `
  --preset primary-2026-08-28 `
  --data-root external_data/s2020-intersurfacemaps-data `
  --output artifacts/high_genus_registration_primary

$primaryTrialDirs = Get-ChildItem artifacts/high_genus_registration_primary `
  -Recurse -Filter metrics.json | ForEach-Object { $_.Directory.FullName }
& 'C:\Users\xuzhehao\anaconda3\python.exe' -m qcopt.experiments.audit_high_genus_registration `
  @primaryTrialDirs --output artifacts/high_genus_registration_final
```

The audit command refuses missing, extra, or duplicate `(case, seed, mode)`
entries unless `--allow-incomplete` is explicitly supplied.

- [Aggregate audit report](../artifacts/high_genus_registration_final/research_report.md)
- [Machine-readable 36-trial audit](../artifacts/high_genus_registration_final/independent_audit.json)
- [Per-trial CSV](../artifacts/high_genus_registration_final/independent_trials.csv)
- [Aggregate improvement plot](../artifacts/high_genus_registration_final/dense_improvement.png)
- [Official data inventory](high_genus_dataset_inventory.md)

Each trial directory referenced by `independent_audit.json` contains
`metrics.json`, `surface_points.npz`, `flow_history.npz`, `correspondence.png`,
and `objective.png`, with SHA-256 values recomputed by the independent auditor.

## Dense texture-map visualization

The original correspondence figures contain only 192 colored samples and do
not make the full registration map visually legible.  The texture renderer
now evaluates every target **face centroid**.  For a target point `y` and saved
map `F = Phi o F0`, it computes `Phi^-1(y)`, locates that point on the target
side of the published common refinement, transfers the identical overlay
face/barycentric coordinates to the source side, and samples the source
texture there.  The target geometry is held fixed in the last three panels;
only the pulled-back texture changes.

Face centroids are essential.  The intrinsic P2 velocity field vanishes at PL
cone vertices to guarantee edge compatibility, so a target-vertex texture
sample would falsely show every nontrivial flow as stationary.  A regression
test now covers observable face-interior motion.

The four panels mean:

1. source surface with its source texture;
2. target surface textured by the published/base map `F0` (reference, not an
   optimizer prediction);
3. the same target surface textured through the perturbed initialization;
4. the same target surface textured through the refined registration.

Official cases use the per-face-corner UV coordinates published in each
source `A.obj`, including their genuine UV seams.  The generated genus-2 case
has no supplied UV map and uses discrete seam-free 3D coordinate cells.  The
HTML files contain four rotatable Plotly `mesh3d` surfaces; rotating any target
panel synchronizes all three target cameras.

| representative trial | target faces | initial texture RMSE | final texture RMSE | improvement |
|---|---:|---:|---:|---:|
| synthetic genus-2, combined, seed 3 | 2,904 | 0.2859 | 0.2642 | 7.6% |
| official genus-3, landmark, seed 3 | 1,310 | 0.3887 | 0.2963 | 23.8% |
| official genus-5, landmark, seed 29 | 3,818 | 0.3383 | 0.3157 | 6.7% |
| official pretzel genus-3, landmark, seed 3 | 9,124 | 0.2130 | 0.2032 | 4.6% |

Texture RMSE is a separate dense diagnostic: source-preimage displacement at
all target face centroids, normalized by the **source** median edge length.  It
must not be numerically conflated with the primary 192-sample error normalized
by target median edge length.  Both diagnostics improve for these four
representative trials, but the genus-5 and pretzel improvements are visibly
and numerically modest.

- [Synthetic genus-2 interactive texture map](../artifacts/high_genus_registration_stage3_smooth_g2_seed3/synthetic_genus2/seed_003/combined/texture_registration.html)
- [Official genus-3 interactive texture map](../artifacts/high_genus_registration_stage1/official_genus3/seed_003/landmark/texture_registration.html)
- [Official genus-5 interactive texture map](../artifacts/high_genus_registration_stage2_genus5/official_genus5/seed_029/landmark/texture_registration.html)
- [Official pretzel genus-3 interactive texture map](../artifacts/high_genus_registration_stage1/official_pretzel_genus3/seed_003/landmark/texture_registration.html)

Reproduce one visualization from its saved histories without rerunning the
registration optimizer:

```powershell
$env:PYTHONPATH='src'
python -m qcopt.experiments.surface_registration_texture `
  --case official_genus3 `
  --data-root external_data/s2020-intersurfacemaps-data `
  --run-directory artifacts/high_genus_registration_stage1/official_genus3/seed_003/landmark
```

The official paper's first-page figure uses the same diagnostic principle: a
recognizable source texture is transported to the target so continuity and
distortion are visible.  Our colors and interactivity are new visualization
assets; they do not reproduce the paper's copyrighted texture image.
