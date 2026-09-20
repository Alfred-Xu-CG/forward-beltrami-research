# Executive answer

## Short conclusion

The final target has **not** been fully achieved. We have several working
pieces of a forward, piecewise-affine and differentiable map pipeline, but no
single solver that simultaneously provides all of the following for an
arbitrary admissible Beltrami coefficient on a general mesh:

1. fast forward evaluation at realistic resolution;
2. a piecewise-affine output map;
3. a global homeomorphism (no triangle inversion and no self-overlap);
4. a reliable gradient for neural-network backpropagation; and
5. practical memory scaling to roughly million-vertex meshes.

The strongest current candidate is a real-pair finite-element BHF layer with
determinant-safe steps and checkpoint/recompute differentiation. On a jittered
Delaunay mesh it has finite gradients and zero flipped triangles for tested
steps. A differentiable coarse-to-fine prototype reduced cost on a `32x32` to
`128x128` experiment, with relative map error `1.80e-7`. However, direct
near/far/Duffy quadrature is still too expensive at fine resolution, the
global principal-value theorem is not closed, and the safe-step controller is
not a globally smooth implicit adjoint.

The Beurling route is fast on periodic uniform grids and can be differentiated
through fixed-point or implicit linear solves. It is not yet a general-mesh,
bounded-domain, singular-quadrature solver with a hard homeomorphism guarantee.
Zero-padding reduces periodic wrap-around but does not remove all boundary,
nonuniform sampling, or principal-value errors.

Positive Tutte/Floater decoders and barrier/KKT layers give the strongest hard
topology safeguards. They either restrict the representable Beltrami fields or
reintroduce a nonlinear solve and active-set/conditioning issues. Therefore
they are useful components and baselines, not the completed universal solver.

The honest research conclusion is: **the project has established a validated
design space and several differentiable/topology-safe building blocks, but the
universal fast differentiable hard-bijective solver remains open.**

## What is already credible

- Piecewise-affine maps are represented by vertex coordinates and linear
  interpolation inside each triangle.
- Independent orientation audits certify zero flipped triangles for many
  controlled experiments.
- Real-pair BHF kernels, determinant-safe flow steps, checkpoint/recompute
  backward passes, sparse Tutte adjoints, and matrix-free KKT adjoints have
  finite gradients in their tested regimes.
- Coarse-to-fine BHF can reduce the measured cost without disconnecting the
  gradient path.

## What is still missing

- A theorem and implementation covering arbitrary compatible or projected
  Beltrami data on a general triangulation.
- A global injectivity/homeomorphism certificate for every differentiable
  forward evaluation, not only for sampled test cases.
- A production singular-integral backend whose error and cost remain controlled
  near coincident source/target elements.
- A million-vertex memory/throughput demonstration.
- A smooth, globally valid backward rule through determinant safety, active
  topology constraints, or a nonlinear equilibrium solve.
