# Surface Registration Texture Visualization Design

## Purpose

Make the computed surface map visible. Aggregate RMSE plots and colored point samples do not reveal how a dense registration transports neighborhoods across handles. The visualization must therefore render a dense source-defined texture on the fixed target geometry.

## Representation

For every target **face centroid** `y`, evaluate the preimage under the displayed map. The optimized map is `F = Phi o F0`, where `F0` is the published common-refinement map and `Phi` is the saved target-surface flow. The renderer computes `Phi^{-1}(y)`, locates that point on the target side of the common refinement, transfers its face/barycentric coordinates to the source side, and evaluates a source procedural texture there. Face interiors are required: the intrinsic P2 flow deliberately vanishes at PL cone vertices, so target-vertex samples would falsely make a nontrivial map look stationary.

A closed high-genus surface has no single seam-free global UV chart. The texture is therefore a display signal built from normalized source PCA coordinates: colored coordinate cells with dark isolines, following the labeled color-grid visual language of Figure 1 in *Inter-Surface Maps via Constant-Curvature Metrics*. It is continuous as a function of the embedded source position and does not introduce an artificial cut seam. It is a visualization of correspondence, not a claim that PCA coordinates are a conformal parameterization.

## Outputs

Each representative trial produces:

- `texture_registration.html`: four rotatable Plotly surfaces (source texture, published `F0`, perturbed initialization, refined map), with linked target cameras and hoverable face/preimage information.
- `texture_registration.png`: static source/`F0`/initial/refined overview using the same face colors as the HTML.
- `inverse_map_error.png`: initial and final face-centroid source-preimage error on the fixed target, normalized by source median edge length with a shared color scale.
- `texture_metrics.json`: exact face-centroid pullback RMSE/max error, flow-step counts, locator residuals, and provenance.

## Correctness and failure handling

The renderer rejects missing/inconsistent histories, a correction history longer than the full history, non-finite texture coordinates, or common-refinement localization residuals larger than a scale-aware tolerance. Tests cover identity pullback, observable face-interior motion, nontrivial common-refinement barycentric transfer, finite bounded colors, history splitting, and an independently parsed Plotly HTML payload.

## Validation scope

Generate and visually inspect representative synthetic genus-2, official genus-3, official genus-5, and official pretzel genus-3 trials. Run focused tests and the complete project test suite. The pictures show the registration relative to the published common map; they do not turn the published map into independent biological ground truth.
