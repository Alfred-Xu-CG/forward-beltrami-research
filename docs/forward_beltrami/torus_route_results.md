# Route L: positive-weight periodic Tutte embedding

The torus route now has a mesh-native periodic harmonic baseline. A regular
`nx`-by-`ny` grid is kept on the fundamental cell, while seam neighbors carry
the learned period vectors as explicit quasi-periodic shifts. Positive
horizontal and vertical edge weights enter each row through normalized
barycentric coefficients; one vertex fixes the translation gauge.

Uniform weights recover the identity quasi-periods to `1e-12` on a 12x10 test
grid. A `512x512` periodic grid with spatially varying positive weights
(`262,144` vertices and `524,288` lifted triangular faces) solved in `23.89 s`.
The lifted face determinant range was `2.059e-6` to `7.534e-6`, with zero
flipped faces and finite output. Receipt:
`artifacts/torus_tutte_audit/torus_tutte_audit.json`.

The spatial-`mu` decoder boundary is now quantified at both `256x256` and
`512x512`. At the realistic `512x512` resolution, a
separable axis-aligned target (positive periodic x/y derivatives), reciprocal
finite-edge weights recover the target map with relative map error
`1.30e-12` and face-`mu` relative error `4.57e-12`, with zero flips. An
off-diagonal shear control has the same positive-weight solve but leaves a
face-`mu` relative error of `1.00` (map error `4.14e-2`), exposing the missing
cross-direction conductivity/coupling rather than suggesting arbitrary-
`mu` expressivity. Receipt:
`artifacts/torus_spatial_mu_decoder_audit_512/torus_spatial_mu_decoder_audit.json`
(with the 256² companion at
`artifacts/torus_spatial_mu_decoder_audit_256/torus_spatial_mu_decoder_audit.json`).

This closes a useful positive-weight periodic embedding baseline: seam
connectivity and target periods are handled without cutting the torus into a
disk. It does not yet provide the full theorem for arbitrary triangulated torus
meshes, nor does it invert a spatially varying Beltrami coefficient. In
particular, the embedding is a hard periodic decoder/control, not a replacement
for the torus Beurling solver.

## Cross-direction positive graph decoder

To address the explicit off-diagonal failure, the periodic graph was extended
with positive edges in directions `(1,1)` and `(1,-1)`, while retaining the
quasi-periodic seam shifts. Edge weights were obtained by nonnegative fitting
of the local Beltrami conductivity tensor over the four-direction dictionary.
At `512x512` (`262,144` vertices and `524,288` lifted faces), the axis-only
control had map/face-`mu` relative errors `0.04483/1.02535`; the
axis-plus-diagonal graph reduced them to `0.01679/0.36217`. Both maps remained
finite, had zero flipped faces, and had minimum lifted determinants
`3.56e-6` and `2.77e-6`, respectively. The same improvement was reproduced at
`256x256` (`0.04490/1.02534` to `0.01684/0.36247`). This is concrete evidence
that cross-direction graph coupling repairs a large part of the axis-only
expressivity gap, but the residual is still substantial: the four-direction
positive cone and regular-grid graph are not an arbitrary spatial Beltrami
decoder.
Receipt: `artifacts/torus_cross_direction_decoder_audit_512/torus_cross_direction_decoder_audit.json`.

### Two-step primitive-direction extension

The same `512x512` target was then decoded with an eight-direction positive
periodic graph, adding primitive lattice directions `(2,1)`, `(1,2)`,
`(2,-1)`, and `(1,-2)` to the axis-plus-diagonal dictionary. The map relative
error fell from `0.01679` to `0.00681`, and the facewise-μ relative error fell
from `0.36217` to `0.15281` (absolute L-infinity error `0.03141`). The local
nonnegative tensor fits were machine-small (`mean residual 3.16e-16`, p95
`5.69e-16`), while the periodic decoder retained zero flipped lifted faces
and minimum lifted determinant `2.93e-6`. The solve took `228.17 s` for
`262,144` vertices and `524,288` lifted faces. Receipt:
`artifacts/torus_extended_graph_decoder_512/torus_extended_graph_decoder_audit.json`.

This is strong evidence that enriching the positive periodic graph cone repairs
the cross-direction expressivity gap without sacrificing hard orientation. The
remaining residual is still substantial, and the regular-grid graph,
triangulated-torus theorem, and arbitrary-μ inverse remain open.

The Beurling branch now also retains the spatially varying solver's zero mode
at the map level. For `h=f_bar`, `b=mean(h)` and
`f=z+b*conj(z)+C(h-b)` recover the correct quasi-periods. On a `256x256`
coefficient with constant and two oscillatory components, the composed map
objective/implicit VJP matched a fresh-solve finite difference to `1.06e-7`
and had adjoint residual `4.60e-10`. This closes the map-objective zero-mode
channel, while arbitrary triangulated-torus hard decoding and full spatial-μ
coupling remain open. Receipt:
`artifacts/torus_spatial_zero_mode_map_audit/torus_spatial_zero_mode_map_audit.json`.
