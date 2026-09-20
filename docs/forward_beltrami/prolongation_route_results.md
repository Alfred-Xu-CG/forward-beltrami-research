# Route I: coarse-to-fine regular-grid prolongation

`prolongate_regular_grid` performs nested-grid bilinear interpolation of a
coarse vertex map. On an affine manufactured map, the fine map is reproduced to
below `1e-12`. On a smooth shear (`16x12` to `64x48`), the fine P1 map passed
the independent global injectivity audit and its target/source area ratio stayed
above `0.5`.

This is positive evidence for a coarse-to-fine implementation primitive, not a
hard guarantee for arbitrary coarse maps. Bilinear interpolation can create
new extrema or fold fine triangles even when the coarse samples look valid; a
production decoder must run the same face/boundary/degree checks or use a
prolongation theorem with stronger hypotheses. The route is promising because
it reduces latent resolution, but it cannot replace a hard decoder by itself.

A fixed counterexample makes the limitation concrete: both coarse diagonal
faces can have positive determinants (`1.4191` and `0.6077`), while bilinear
prolongation to `16x16` creates a fine face with determinant
`-2.37e-3`. Thus coarse validity alone is not a fine-grid topology certificate.

`prolongate_positive_increments` adds a parameter-space alternative: refine
strictly positive 1D interval increments in log-space and renormalize their
sum. A 4-to-32 refinement stayed strictly positive and preserved total length,
so the associated separable monotone decoder cannot fold along that axis. This
does not repair a general 2D bilinear map; it demonstrates why prolonging
positive latent parameters is safer than prolonging unconstrained coordinates.

## High-resolution audit

The smooth positive-map case was extended to realistic resolutions. A `64²`
coarse map prolonged to `512²` (`524,288` fine faces) in `2.52 s`; a `128²`
coarse map prolonged to `1024²` (`2,097,152` fine faces) in `10.64 s`.
Independent fine-grid determinant checks found zero flipped faces in both
cases. These results establish scalability for a benign realizable map, while
the fixed folding counterexample above remains the reason that the same result
cannot be generalized to arbitrary coarse coordinates. Receipt:
`artifacts/prolongation_highres_audit/prolongation_highres_audit.json`.

## Coupled hard-decoder prolongation

The positive increments were then prolonged to a fine grid before applying the
determinant-one affine shear coupling. At `512x512` (524,288 faces) from a
32-interval latent, and at `1024x1024` (2,097,152 faces) from a 64-interval
latent, independent fine-grid audits reported zero flipped faces and minimum
determinants `2.006e-6` and `5.003e-7`. The analytic inverse returned maximum
round-trip error `3.51e-16` in both cases. Receipt:
`artifacts/coupled_prolongation_audit/coupled_prolongation_audit.json`.

This closes the coupled injective-prolongation control for the implemented
monotone-plus-affine family, but not for arbitrary spatially varying QC maps;
the affine shear remains the expressivity bottleneck.
