# Route M — spherical charts and orbifold cone maps

## 1. Chart formulation

Let `phi_alpha` map a source chart to planar coordinates and `psi_alpha` map
back to the surface. A chartwise map is

`F_alpha=psi_alpha^{-1} o G_alpha o phi_alpha`.

For an overlap between charts `alpha,beta`, consistency requires

`F_alpha = F_beta`

on the overlap seam after applying the transition maps. Numerically, the seam
residual is the maximum discrepancy of the two chart representations.

## 2. Cone/orbifold coordinates

For a cone exponent `alpha`, radial coordinates use a power law such as

`r -> r^alpha`.

The derivative is singular or vanishing at the cone point depending on `alpha`,
so the vertex is treated through local chart rules rather than an ordinary
smooth Euclidean neighbourhood.

## 3. What was audited

The experiments checked chart orientation, seam agreement, inverse chart error,
and zero face flips at realistic chart resolutions. These are atlas and
coordinate-transition controls.

## 4. Scope

The sphere/orbifold route does not address the main planar arbitrary-mesh
forward Beltrami layer. In particular, it does not supply a global planar PV
solver or a neural-network-compatible general homeomorphism theorem.
