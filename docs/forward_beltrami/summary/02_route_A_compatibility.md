# Route A: compatibility and projection of facewise Beltrami data

## Idea

The Beltrami coefficient can be prescribed independently on each triangle.
The unknown vertex coordinates must nevertheless agree on shared edges. Route A
forms the sparse compatibility equations that express this shared-edge
agreement and solves or projects the facewise data onto the realizable set.

## Evidence

Manufactured compatible fields reconstruct with errors around machine
precision (`1.22e-15` in the 512² holonomy control). A 256² nonlinear projection
with an analytic sparse Jacobian reached relative residual `3.01e-9` for a
fixed-R2 target while retaining zero flips and minimum area ratio `0.65460`.
The high-resolution compatible projection assembled a 512² Jacobian with about
6.79 million nonzeros and converged in a small number of evaluations, but took
hundreds of seconds.

The random incompatible 256² field is the important negative result: after 11
function evaluations under a 20-evaluation budget, the relative residual was
`0.7081017153`, despite zero flips and minimum area ratio `0.63326`. Projection
therefore preserves topology in this run but does not make arbitrary facewise
data exactly realizable.

## Interpretation

This route cleanly separates a mathematical obstruction from an implementation
problem. A general facewise coefficient does not automatically correspond to a
single continuous piecewise-affine map. Projection can produce a valid nearby
map, but the projection itself is an optimization/root solve, its uniqueness
is not established, and its exact second-order reverse derivative is not yet
closed.

## Decision

Route A is a useful compatibility preprocessor or constrained latent layer. It
is not currently the required universal fast forward solver.
