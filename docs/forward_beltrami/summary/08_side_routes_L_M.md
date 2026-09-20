# Side routes L and M: torus and sphere/orbifold experiments

These routes were explored for mathematical understanding but are not the main
candidate under the user's final solver criterion.

## Route L: flat torus

Periodic connectivity and an affine-period lift make a clean controlled test.
At 512², adding diagonal directions reduced cross-direction face-coefficient
error from `1.025` to `0.362` with zero flips. An extended eight-direction
graph reduced map error to `0.00681` and coefficient error to `0.15281`, with
minimum lifted determinant `2.93e-6`. The results are periodic and regular-grid
specific; they do not prove a general bounded-domain decoder.

## Route M: spherical/orbifold charts

Two-chart cone constructions had seam mismatches near machine precision,
inverse error below `1.56e-14`, and zero flips at realistic chart resolution.
They validate chart orientation, cone-coordinate transitions, and atlas
bookkeeping. They do not close an arbitrary cone-angle theorem or couple to a
global BHF solver.

## Decision

Keep these as side validation and geometry controls. They should not be used as
evidence that the planar general-mesh neural-network layer is complete.
