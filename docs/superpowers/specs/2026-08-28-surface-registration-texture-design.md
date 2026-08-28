# Surface Registration Texture Visualization Design

## Purpose

Make the computed surface map visible. Aggregate RMSE plots and colored point samples do not reveal how a dense registration transports neighborhoods across handles. The visualization must therefore render a dense source-defined texture on the fixed target geometry.

## Representation

For every target vertex `y`, evaluate the preimage under the displayed map. The optimized map is `F = Phi o F0`, where `F0` is the published common-refinement map and `Phi` is the saved target-surface flow. The renderer computes `Phi^{-1}(y)`, locates that point on the target side of the common refinement, transfers its face/barycentric coordinates to the source side, and evaluates a source procedural texture there.

A closed high-genus surface has no single seam-free global UV chart. The texture is therefore an intrinsic display signal built from normalized source PCA coordinates: colored coordinate cells with dark isolines. It is continuous as a function of the embedded source position and does not introduce an artificial chart seam. It is a visualization of correspondence, not a claim that PCA coordinates are a conformal parameterization.

## Outputs

Each representative trial produces:

- `texture_registration.png`: source texture, published `F0`, perturbed initialization, and refined map. The three target panels use identical target geometry and camera.
- `inverse_map_error.png`: initial and final source-preimage error on the fixed target, normalized by source median edge length with a shared color scale.
- `texture_metrics.json`: exact vertexwise pullback RMSE/max error, flow-step counts, locator residuals, and provenance.

A method comparison for one official genus-3 case places truth, common initialization, landmark-only, curvature-only, and combined refinements on the same fixed target.

## Correctness and failure handling

The renderer rejects missing/inconsistent histories, a correction history longer than the full history, non-finite texture coordinates, or common-refinement localization residuals larger than a scale-aware tolerance. Tests cover identity pullback, nontrivial common-refinement barycentric transfer, finite bounded colors, and history splitting.

## Validation scope

Generate and visually inspect representative synthetic genus-2, official genus-3, official genus-5, and official pretzel genus-3 trials. Run focused tests and the complete project test suite. The pictures show the registration relative to the published common map; they do not turn the published map into independent biological ground truth.
