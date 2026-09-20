# Route A: Discrete realizable Beltrami coefficients

For a facewise P1 map write `a_T=f_z` and `f_bar=mu_T*a_T` on face `T`. On a
shared source edge with complex tangent `dz`, continuity gives

\[
 (a_T-a_U)dz+(\mu_Ta_T-\mu_Ua_U)\overline{dz}=0.
\]

`qcopt.forward.compatibility` assembles these equations as a sparse complex
matrix. A manufactured P1 map supplies a nonzero vector `a_T` in its numerical
nullspace; a generic random facewise coefficient is generically full column
rank on a small mesh, so an arbitrary bounded `mu_T` is not automatically
realizable by a continuous P1 map.

## Evidence

- Manufactured smooth shear: compatibility residual below `1e-12`.
- Random `|mu|<0.7` on a `4x4` grid: smallest dense SVD singular value above
  `1e-4`, i.e. no nonzero compatible `f_z` detected at that discretization.
- Realistic `256x256` grid: `131072` faces, `196096` interior-edge equations,
  `392192` nonzeros, assembly `2.38 s`, manufactured residual
  `1.94e-18`.

This is the first concrete obstruction to an unconstrained facewise-BC neural
latent: the latent must either satisfy these compatibility constraints, be
projected onto their realizable manifold, or be decoded through a different
hard-bijective representation such as Tutte weights. The edge test is local
compatibility; rectangle boundary homeomorphism and global degree remain
separate gates.

## Kernel dimension audit

The dense SVD diagnostic was repeated on regular `4x4`, `5x5`, and `6x6`
cell grids (32, 50, and 72 faces). For the manufactured deformation the
complex compatibility nullity was exactly one in all three cases, matching
the expected global complex scale. Independent random facewise coefficients
with `|mu|<0.6` had nullity zero in all cases. The JSON receipt is
`artifacts/compatibility_dimension_audit/compatibility_dimension_audit.json`.

This supports the holonomy/manifold dimension picture on simply connected
small meshes, but is not a proof: the diagnostic uses a dense SVD and does not
yet handle boundary target constraints, singular strata, or a scalable
projection onto the nonlinear realizable set.

## 256² dual-holonomy reconstruction

The compatibility picture was tested without assembling a dense nullspace. On
a `256x256` grid (`66,049` vertices and `131,072` faces), face scales were
propagated across `261,121` dual cycles and image edges were integrated across
`328,192` primal cycles. For a manufactured P1 homeomorphism with
`max|mu|=0.331`, the maximum relative dual-holonomy residual was
`1.05e-14`, maximum primal edge closure was `9.19e-15`, and reconstruction
after the optimal complex similarity alignment had relative error
`1.22e-15`. A bounded random field with `max|mu|=0.62` had dual-holonomy
maximum `7.64e25`, p95 `7.51e6`, primal closure maximum `6.50e17`, and aligned
reconstruction error `0.9996`. This is strong realistic-resolution evidence
that `|mu|<1` is not sufficient for discrete realizability, and that the
holonomy-zero condition is numerically recoverable; it remains a numerical
characterization rather than a formal theorem or rectangle-boundary chart.
Receipt: `artifacts/holonomy_reconstruction_audit_256/holonomy_reconstruction_audit.json`.

## Nonlinear projection prototype

`project_facewise_mu` fixes the boundary map and optimizes interior P1 vertex
coordinates so that the induced facewise Beltrami coefficient matches a target
field. The projection now supplies an analytic sparse quotient-rule Jacobian
instead of a dense finite-difference Jacobian. Starting from the identity
interior and the manufactured boundary, it recovered smooth fields on `4x4`,
`8x8`, `16x16`, `32x32`, and `64x64` meshes (32 through 8,192 faces) in
`0.012`, `0.024`, `0.082`, `0.407`, and `1.687 s`, respectively, with
residuals below `1.3e-9`. For a random `5x5`
face field with `max|mu|=0.65`, the optimizer converged but retained residual
norm `3.1773`, making the incompatibility visible rather than silently
returning a “realizable” map. Receipt:
`artifacts/compatibility_projection_audit/compatibility_projection_audit.json`.
An independent PL injectivity audit certified all manufactured projections
through `64x64` and also certified the random-field projection while reporting
its nonzero Beltrami mismatch; this is a useful separation between topology of
the returned map and exact realizability of the requested field.

The sparse Jacobian removes one major implementation bottleneck, but this is
still a projection mechanism and a diagnostic, not a hard decoder: it has no
global injectivity guarantee or uniqueness theorem. The 256² run converges in
the tested budget, while the 512² stress run reaches a no-folding state but
exhausts five evaluations without convergence; neither is a production
differentiable projection layer.

An independent three-point finite-difference check on a `3x3` mesh found a
maximum Jacobian discrepancy below `1e-6` (the observed error was about
`6e-10`). The check is part of `tests/test_forward_compatibility_projection.py`
and is included in the full regression receipt.

## 128x128 projection probe

To test medium-mesh scaling beyond the original 64x64 sweep, the same
fixed-boundary manufactured field was projected on a `128x128` grid (`32,768`
faces). The analytic sparse Jacobian converged in 5 function evaluations and
`9.07 s`, with residual norm `3.18e-9`, zero flipped faces, and an independent
injectivity certificate. This is a realistic medium-resolution control, not a
global guarantee; a scalable 256² projection and uniqueness/boundary theorem
remain open. Receipt:
`artifacts/compatibility_projection_128_probe/compatibility_projection_128_probe.json`.

The same analytic sparse-Jacobian projection was then attempted at `256x256`
(`131,072` faces) with a smaller manufactured amplitude. It converged in 5
function evaluations and `49.47 s`, with residual norm `8.06e-9`, zero flipped
faces, and an independent injectivity certificate. This is the first realistic
256² nonlinear projection control; it demonstrates feasible assembly/solve
scaling but still does not provide uniqueness, a hard global homeomorphism
theorem, or a production differentiable projection layer. Receipt:
`artifacts/compatibility_projection_256_probe/compatibility_projection_256_probe.json`.

## 512x512 realistic-resolution projection stress test

The fixed-boundary manufactured projection was pushed to `512x512` (`524,288`
faces and `522,242` free coordinates) with the same sparse analytic Jacobian
and a five-evaluation budget. Sparse assembly and the trust-region iterations
completed in `304.57 s`; the Jacobian had `6,789,146` nonzeros (about `4.32`
per residual row). The residual decreased to `2.887e-6`, and the independent
PL audit found minimum face determinant `0.9999996154`, zero flipped faces, and
an injectivity certificate. The optimizer did **not** report convergence within
five evaluations, so this is a scalability/no-folding stress result rather
than a successful 512² projection solve. It shows that the sparse formulation
can be assembled at realistic resolution, while its nonlinear iteration count,
conditioning, and lack of a global projection/homeomorphism theorem remain
open. Receipt:
`artifacts/compatibility_projection_highres_512/compatibility_projection_highres_audit.json`.

## Implicit projection VJP control

The sparse projection Jacobian is now exported as a reusable operator, and an
implicit normal-equation VJP was checked at `128x128` (`32,768` faces,
`32,258` free coordinates). The Jacobian had `419,354` nonzeros and the normal
matrix `447,406` nonzeros. The manufactured projection was injectivity-certified
with minimum face determinant `0.999999999813`. For a tangent target direction
generated by the linearized realizable manifold, the fresh-projection
directional error was `8.47e-7` and the perturbed residuals stayed near
`1.5e-9`. An arbitrary target direction produced a larger `5.56e-5` error and
residuals around `7.10e-5`; this is evidence that a Gauss--Newton normal
equation is not the full second-order KKT derivative for an off-manifold target
perturbation. Thus the exported operator closes the sparse linearization
plumbing gate but not a general exact differentiable projection manifold chart.
Receipt: `artifacts/compatibility_projection_vjp_audit_128_directional/compatibility_projection_vjp_audit.json`.

## R2 rectangle-side boundary audit

The earlier projection experiments fixed every boundary vertex and therefore
did not test the more useful rectangle model in which each side remains on its
corresponding side but its tangential parameterization may slide. The new
`compatibility_boundary_theorem_audit.py` constructs such a manufactured map:
the four corners stay fixed, bottom/top points move only tangentially, and
left/right points likewise move only tangentially. The interior coupling
vanishes on the boundary.

On a `512x512` cell mesh (`263,169` vertices, `524,288` faces, `2,048`
boundary vertices), the manufactured field had `max|mu|=0.26134`. Dual-scale
propagation gave maximum relative holonomy `1.64e-14`; primal image-edge
closure was `7.95e-15`; and reconstruction after the unavoidable complex
similarity gauge had relative L2 error `1.44e-15`. An independent audit found
zero flipped faces, minimum signed area ratio `0.65459`, no rectangle-side
violations, and positive boundary orientation. Four interior sample points had
polygon winding exactly one. Receipt:
`artifacts/compatibility_boundary_theorem_audit_512/compatibility_boundary_theorem_audit.json`.

This is realistic-resolution evidence for the discrete degree argument's
hypotheses: positive P1 faces plus an ordered rectangle boundary behaves as a
global homeomorphism in floating point. It does not prove the exact theorem,
does not show that arbitrary bounded `mu` admits an R2 projection, and does
not establish uniqueness or a differentiable projection chart.

## Fixed-R2-boundary nonlinear projection

The R2 construction was then used as an actual boundary condition for the
sparse nonlinear projection. A second manufactured map adds an interior-only
perturbation, so its facewise `mu` is nontrivial while its four boundary sides
are exactly the same side-sliding rectangle boundary. At `256x256` (`131,072`
faces), five trust-region evaluations gave projection residual L2
`1.20e-7` and relative face-`mu` error `3.01e-9`. The prescribed boundary was
reproduced to `0`, the independent rectangle injectivity audit certified the
map, there were zero flipped faces, and the minimum signed-area ratio was
`0.65460`. Receipt:
`artifacts/compatibility_rectangle_projection_audit_256/compatibility_rectangle_projection_audit.json`.

This closes a realistic fixed-R2-boundary projection control, not the full
arbitrary-boundary problem: boundary tangential coordinates are supplied, not
optimized, and no uniqueness or global projection theorem has been proved.
