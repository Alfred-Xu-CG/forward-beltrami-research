# Route C: finite-element Beltrami holomorphic flow

## Method

Beltrami holomorphic flow (BHF) evolves a map by repeatedly applying a velocity
computed from the current map and the prescribed Beltrami variation. The
implementation uses real pairs `(real,imag)` rather than complex autograd,
piecewise-affine triangle quadrature, near-field Duffy quadrature for the
singular kernel, and blocked far-field evaluation.

An adaptive step can be clipped before the first triangle determinant reaches a
chosen positive margin. A checkpoint/recompute layer stores the initial state
and static mesh data, then replays the trajectory during backward instead of
retaining every dense interaction graph.

## Mesh-native evidence

On jittered rectangular Delaunay meshes, 128² cells (16,641 vertices and
32,768 faces) took `6.10 s` and `15.76 s` for near orders 8 and 16; the
relative velocity difference was `1.92e-7`. At 256² cells (66,049 vertices and
131,072 faces), the times were `63.75 s` and `109.56 s`, with relative
difference `1.14e-7`. Candidate steps `0.25`, `0.5`, and `1.0` had zero
flips; the 256² minimum area ratios were about `0.2860`, `0.2854`, and
`0.2840`.

## Coarse-to-fine prototype

A differentiable `32x32 -> 128x128` layer performed two BHF steps on the coarse
grid, bilinearly prolonged the map with PyTorch, then performed two fine-grid
correction steps. Against four fine-grid steps, the relative map error was
`1.80e-7`; both maps had zero flips and minimum area ratio about `0.85958`.
The rerun measured fine-only versus coarse-to-fine forward/backward times of
`19.94/97.95 s` versus `13.81/51.33 s`. Peak CUDA allocation was
`4,913,521,152` bytes (about 4.91 GB), and all tested gradients were finite.

## Limits

The attempted `64x64 -> 256x256` extension used roughly 24 GB GPU memory and
was stopped after 27 minutes without a receipt. Direct quadrature remains the
dominant cost. The global principal-value theorem, a scalable nonlinear
integrator, a general unstructured prolongation theorem, and a global
homeomorphism guarantee are still open. Exact determinant-root clipping also
has active-set switches; the smooth conservative controller is a surrogate,
not an established globally equivalent adjoint.

## Decision

BHF is the best current forward/differentiable research direction, especially
with multilevel scheduling, but it is not yet the final solver under the strict
speed, memory, and universal-homeomorphism requirements.
