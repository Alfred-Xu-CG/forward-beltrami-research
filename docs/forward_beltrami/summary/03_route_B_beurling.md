# Route B: Beurling transform, FFT, and nonuniform evaluation

## Continuous and discrete picture

The Beurling transform is a singular convolution on the whole complex plane.
On a uniform periodic grid, the Fourier transform changes convolution into
pointwise multiplication by a known frequency multiplier. The fast Fourier
transform (FFT) makes this operation roughly `O(N log N)` for `N` grid points.

This does not mean that a bounded-domain problem is automatically periodic.
Embedding a bounded coefficient in a box and applying an FFT periodically
repeats that box. Zero-padding pushes image interactions farther away, but it
does not remove the periodic model, discretization, target interpolation, or
principal-value singularity.

## Experiments and numbers

On a 512² disjoint source/target control, zero-padded FFT relative errors were
`7.07e-2`, `4.41e-3`, and `2.76e-4` for padding factors 2, 4, and 8. Thus
padding helps substantially, but finite padding is not exact free-space
evaluation.

For a 65,536-source smooth cloud, FINUFFT box-tail tests kept the physical
cutoff fixed while changing `(L,modes)` from `(8,2048)` to `(12,3072)` and
`(16,4096)`. Relative errors changed only from `0.0141276` to `0.0140020` to
`0.0139502`; the finite box tail was not the dominant error in that control.

For scattered points, a nonperiodic treecode agreed with independent GPU direct
sums at relative errors around `2.5e-7` to `4.4e-7` when sources were well
separated. Paired source/target offsets near `1e-4` worsened the error to about
`4.53e-4`, exposing the singular/PV quadrature bottleneck.

## Decision

FFT/NUFFT is attractive for uniform periodic or carefully padded grids, and
tree/FMM-style methods show that general meshes need not be periodic. However,
the project has not yet produced a production singular-quadrature/FMM backend
with bounded-domain boundary closure, arbitrary unstructured meshes, and a
hard homeomorphism guarantee. Route B is therefore a fast operator component,
not the completed solver.
