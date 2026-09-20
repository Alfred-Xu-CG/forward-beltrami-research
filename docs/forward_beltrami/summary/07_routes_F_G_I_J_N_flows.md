# Routes F, G, I, J, and N: flows, prolongation, latent decoders, and preconditioners

## Route F: mesh-native triangular flows

Positive-increment parameterizations and triangular compositions guarantee
positivity of selected one-dimensional derivatives. Their composition can be
piecewise-affine and differentiable, and high-resolution 512²/1024² controls
had zero flips. The limitation is expressivity: a finite triangular latent
parameterization does not cover arbitrary two-dimensional Beltrami fields.

## Route G: certified piecewise-linear residual flow

An exact determinant-root step computes the first positive step at which any
face reaches a determinant margin. This gives a strong local certificate and
worked in rotation/shear waypoint experiments, including a direct-path stall
that was repaired by two homotopy segments. The root selection is nonsmooth at
active-face changes, so its ordinary reverse derivative is only a fixed-active
local derivative unless a smooth conservative surrogate is used.

## Route I: coarse-to-fine prolongation

Bilinear prolongation alone can fold a map; positive-increment and coupled
triangular prolongation controls avoid that failure on tested grids. The missing
piece is a mesh-independent theorem and arbitrary-QC latent expressivity.

## Route J: learned flow/decoder

Positive-increment latent training and conditional flow matching produced finite
512² maps with zero flips; a 256-interval stress had minimum determinant
`1.57e-6`. These are hard-decoder and distribution-learning controls, not a
solver for a supplied arbitrary Beltrami coefficient.

## Route N: learned or Neumann preconditioners

Warm starts and truncated-Neumann preconditioners reduce iterations in some
matrix solves. A 512² rough-band stress had cost-inclusive speedups about
`1.19x`, while several smooth/random cases had only about `1.02x` median speedup
or slowdowns. These methods accelerate an existing solve; they do not replace
the topology or expressivity guarantee.

## Decision

These routes are valuable components for a future neural parameterization or
multilevel preconditioner. They are not independent universal solvers.
