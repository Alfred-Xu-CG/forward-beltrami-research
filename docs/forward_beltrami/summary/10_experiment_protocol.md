# Experiment protocol and reproducibility notes

## Meshes and resolutions

Experiments use structured square grids, jittered Delaunay meshes, periodic
torus grids, and charted sphere meshes. “Realistic resolution” means examples
such as 256² or 512² cells, 66,049 or 262,144 vertices, and 131,072 or
524,288 faces; small 32² tests are used only for debugging and multilevel
controls.

## Independent audits

The solver does not certify itself. After a map is produced, an independent
routine recomputes each mapped triangle's signed area, counts flips, checks the
minimum area ratio, and checks boundary orientation where applicable. For
gradient claims, finite differences or an independently assembled directional
derivative are used.

## Timing and memory

Forward and backward timing starts after data preparation and includes the
stated blocked quadrature or iterative solve. CUDA timing synchronizes the
device. Peak device memory is read from the CUDA allocator; block sizes and
mesh resolution are recorded with the receipt. A missing JSON receipt after a
resource-bounded run is reported as a non-result, never as a numerical result.

## Environment

The local checks use one-thread MKL/OpenMP settings to avoid Intel OpenMP
runtime collisions. The dedicated remote forwards use ports 18183, 18184, and
18185; the long-used port 18082 is not modified. The latest health probe found
all three dedicated forwards healthy.

## Verification status

The focused regression set after the multigrid addition passed: 17 BHF/M-matrix
tests. The new multigrid experiment also passes Python compilation. Existing
route receipts and their limitations are listed in
`docs/forward_beltrami/route_status.md`, `completion_audit.md`, and
`validation_current.md`.

## Final acceptance checklist

Before claiming the final solver, require all of these at once:

- arbitrary admissible test coefficients on structured and unstructured meshes;
- piecewise-affine output with an independently checked homeomorphism theorem;
- finite and accurate reverse gradients through successful and near-boundary
  cases;
- forward/backward cost and peak memory at million-vertex scale;
- comparison against Beurling and linear-system baselines at matched accuracy;
- explicit treatment of incompatible facewise data and boundary conditions.
