# Route G: Certified PL residual updates

For a current face Jacobian `J` and residual Jacobian `D`, each face determinant
along an explicit update is exactly

\[
\det(J+\alpha D)=c+b\alpha+a\alpha^2.
\]

`maximum_safe_step` finds the first positive root of this quadratic relative to
a prescribed determinant margin, and `certified_residual_step` takes a strict
fraction of the minimum root. Tests confirm that a deliberately aggressive
vertex displacement remains above the margin and passes the independent
injectivity audit, while an already-degenerate input is rejected.

This gives a usable topology-preserving update primitive for BHF, Beurling,
optimization, or learned residual proposals. It is not a complete solver: the
step is conservative, boundary self-intersection still needs an independent
audit, and the active face/root selection makes the exact map-to-step function
piecewise smooth rather than globally differentiable. A neural layer can use it
as a safeguarded forward pass and differentiate only within a fixed active set,
or use an implicit/KKT treatment at switches.

`maximum_safe_step_with_active_face` now returns the face attaining the first
determinant root along with the step value. A structured-grid test verifies
agreement with the scalar API. This exposes the active set needed for a custom
piecewise-smooth derivative, while explicitly retaining the nondifferentiable
tie boundary instead of pretending the min-root operation is smooth.

The fixed-active-face directional derivative is now implemented analytically
from the determinant quadratic coefficients. A randomized structured-grid test
matches finite differences to below `1e-6`; tied active faces remain explicitly
treated as nonsmooth events.

## Large-deformation continuation counterexample

`certified_step_toward_target` caps the residual scalar at one and exposes the
large-deformation behavior of straight continuation. On a `64x64` mesh, the
identity map and a 180-degree rotation are both valid orientation-preserving
maps, but their linear homotopy collapses at the midpoint. With determinant
margin `0.05`, the direct path reduced its step from `0.3843` to below
`1e-12`, stalled at a relative target error `0.6118`, and never crossed the
barrier. A staged path through a 90-degree rotation reached the same 180-degree
target in one certified second-stage step with relative error `2.94e-17`.
This is concrete evidence that safe residual steps need path planning or a
different velocity parameterization for large deformations; determinant safety
alone cannot make an arbitrary endpoint reachable. Receipt:
`artifacts/safe_continuation_audit/safe_continuation_audit.json`.

The same counterexample was repeated at `256x256` cells (`131,072` faces).
Straight identity-to-180° continuation again stalled at relative error
`0.6118034`, with the certified scalar shrinking below `1e-12`; the staged
identity-to-90°-to-180° path reached the endpoint with relative error
`2.95e-17` in one certified second-stage step. This confirms that the barrier
is geometric rather than a coarse-mesh artifact. Receipt:
`artifacts/safe_continuation_audit_256/safe_continuation_audit.json`.

## Automatic waypoint planner control

The determinant-root step was then wrapped in a simple waypoint planner rather
than hard-coding a single 90-degree split. On a `256x256` mesh (`131072` faces),
the planner used `60° -> 120° -> 180°`; every segment completed in one
certified step, with minimum face determinants between `0.9999999999999284`
and `0.9999999999999716`, zero flips, and final target error `4.01e-15`.
This demonstrates an executable path-planning control for a family of large
rigid deformations. It does not establish a general waypoint-selection theorem:
arbitrary targets, nonsmooth active-face switches, and coupling to a nonlinear
H/KKT solve remain open.
Receipt: `artifacts/safe_rotation_waypoint_planner_audit_256/safe_rotation_waypoint_planner_audit.json`.

## Adaptive arbitrary-target path control

The planner was generalized to a smooth injective target path consisting of a
rotation composed with an area-preserving sinusoidal shear, rather than a
preselected sequence of rigid rotations. On a `256x256` mesh (`131,072` faces),
the one-segment identity-to-final-target homotopy stalled after 9 safe steps at
determinant margin `0.05`, with final map error `129.57`. An adaptive path
parameter subdivision automatically selected two `0.5` segments; each segment
completed in one certified step, with minimum determinants
`0.9999999999999452` and `0.9999999999999432`, zero flipped faces, and final
target error `6.47e-15`. The 128² control gives the same behavior (direct error
`65.28`, two successful half-parameter segments, final error `3.26e-15`).
This is realistic-resolution evidence that waypoint selection can be made
target-path adaptive, while arbitrary homotopy construction, nonsmooth active
set handling, and a solver-level continuation theorem remain open. Receipt:
`artifacts/general_waypoint_planner_256/general_waypoint_planner_audit.json`.

## 256² active-face switching audit

The determinant-root step map was stress-tested on a `256x256` mesh
(`131,072` faces) using random residual and perturbation directions. In 47
trials, 39 retained the same active face on both `+eps` and `-eps` sides; the
fixed-active analytic derivative matched centered finite differences to about
`1e-7`--`1e-6`. Eight trials crossed an active-face boundary. In those cases
the centered derivative differed from the derivative of the base active face
by `2.6e-3` to `2.9e-2`, despite all individual safe steps remaining finite.
This is direct realistic-resolution evidence that the current custom VJP is
piecewise valid but not differentiable at active-set switches; a generalized
Jacobian, semismooth/KKT rule, or explicit switch handling is required for a
fully differentiable continuation layer. Receipt:
`artifacts/safe_step_active_set_audit_256/safe_step_active_set_audit.json`.
