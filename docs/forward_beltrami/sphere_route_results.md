# Routes C/M: sphere BHF and orbifold-Tutte baseline

The new sphere utility provides a common truth case rather than claiming a new
BHF theorem. A UV sphere with single poles is mapped by the exact positive-real
stereographic Möbius transformation `z -> 1.7 z`; the map fixes the north pole,
stays on the unit sphere, and preserves outward face orientation.

At `512` longitude samples and `256` latitude bands (`130562` vertices,
`261120` faces), generation took `0.99 s`, Möbius evaluation `0.023 s`, and the
orientation audit `0.047 s`. Minimum orientation proxy was
`1.9184e-6`, sphere norm error `4.44e-16`, and the north pole remained fixed.

This establishes a realistic sphere benchmark and pole-stability gate for a
future sphere-native BHF or orbifold-Tutte decoder. It does **not** establish
that a stereographic chart plus Möbius maps reconstructs arbitrary sphere
Beltrami data. A general BHF implementation still needs two-chart/atlas
compatibility, normalization, and a method for computing the holomorphic-flow
velocity; those are separate from this analytic truth map.

The north/south chart utilities now provide an atlas consistency gate. The
south chart uses the orientation-compatible coordinate
`w=(x-iy)/(1+z)`, so the overlap transition is the holomorphic inversion
`w=1/z`. On a 64-by-32 sphere all non-polar overlap vertices round-tripped
through both charts below `1e-12` and satisfied the inversion identity. This
does not by itself compute an arbitrary BHF velocity, but it fixes the atlas
chain rule: `wdot=-zdot/z^2`. The transition and velocity operators were tested
on a realistic `512x256` sphere (`130562` vertices, `117248` overlap points).
Coordinate, analytic-velocity, and velocity-round-trip errors were respectively
`3.95e-14`, `3.93e-14`, and `6.60e-14` in `1.01 s`. This closes the explicit
chart-velocity plumbing gate for the analytic flow; evaluating the nonlinear
BHF variation operator consistently across the overlap and applying the
determinant/orientation audit to each update remain separate gates. Receipt:
`artifacts/sphere_chart_velocity_gluing_audit/sphere_chart_velocity_gluing_audit.json`.

## Orbifold/Tutte cap baseline

As a first orbifold-Tutte experiment, the northern spherical cap was cut along
the equator, projected through the south chart, and decoded with a positive
Tutte solve. At `256` longitude samples and `128` latitude bands this gave
`16,385` vertices and `32,512` faces; the planar Tutte audit was certified in
`0.732 s`. The chart orientation convention required conjugating the mapped
south-chart coordinate before lifting back to the sphere; after that explicit
fix, the minimum outward spherical orientation proxy was `7.66e-5` and the
unit-sphere error was `3.33e-16`.

This is meaningful evidence for a hard spherical-cap decoder, but not a closed
sphere/orbifold solver: the equator seam still needs a second cap and a
globally compatible cone-point/transition construction. The chart orientation
issue is precisely why a naive “run planar Tutte in one stereographic chart”
cannot yet be claimed as Route M completion. Receipt:
`artifacts/orbifold_tutte_cap_audit/orbifold_tutte_cap_audit.json`.

## Closed-sphere two-cap assembly

The seam issue was then resolved rather than left as a cap-only result. North
and south hemispheres were decoded independently in their stable stereographic
charts, the chart orientation was corrected, and shared equator vertices were
matched by source vertex ID (not by the two loops' opposite traversal order).
At `256x128`, both caps were planar-certified; each had `32,512` faces and
minimum spherical orientation `7.656e-5`. The 256-vertex equator seam mismatch
was `6.75e-16`, with sphere norm error `3.33e-16`, in `3.86 s` total.

This is now a closed-sphere hard decoder baseline. It still does not implement
arbitrary orbifold cone angles or BHF coupling, but the previous “seam remains
open” gate is no longer a blocker for the two-cap sphere construction. Receipt:
`artifacts/orbifold_tutte_closed_audit/orbifold_tutte_closed_audit.json`.

## Global two-chart cone-angle atlas control

The local radial cone map was then assembled globally on the closed sphere by
using independent north/south power laws in colatitude, with both charts
anchored to the same equator. At `512` longitude samples and `256` latitude
bands (`130,562` vertices, `261,120` faces), the mixed cases
`(alpha_n, alpha_s)=(0.65,1.4)` and `(1.4,0.65)` had equator displacement
`2.48e-16`, unit-sphere error `2.22e-16`, inverse round-trip error below
`1.56e-14`, minimum outward spherical orientation `1.14e-7`, and zero flipped
faces. The symmetric `(0.65,0.65)` case had round-trip error `3.56e-15`,
minimum orientation `9.41e-5`, and zero flips. This closes the explicit
two-chart seam/normalization/topology control at realistic resolution. It is
still a prescribed analytic cone map: arbitrary cone-point transition laws,
orbifold normalization, and nonlinear BHF coupling remain open.
Receipt: `artifacts/orbifold_global_cone_atlas_audit_512/orbifold_global_cone_atlas_audit.json`.

To probe the exponent-transition boundary rather than one selected pair, a
five-case sweep was run on the same `512x256` sphere (`130,562` vertices,
`261,120` faces), covering `(0.35,1.8)`, `(0.5,2.0)`, `(0.8,1.2)`,
`(1.2,0.8)`, and `(1.8,0.35)`. Every case preserved all `512` equator seam
vertices to `2.48e-16`, sphere norm error below `3.4e-16`, and zero flipped
faces. Inverse round-trip errors ranged from `8.8e-15` to `1.4e-11`; the
smallest orientation proxy was `3.38e-10` for the most concentrated exponent
pair. This strengthens the analytic atlas control across a wider cone-angle
range, while still not establishing arbitrary transition laws, orbifold
normalization, or nonlinear BHF coupling. Receipt:
`artifacts/orbifold_cone_exponent_sweep_512/orbifold_cone_exponent_sweep_audit.json`.

## Local arbitrary cone-angle chart

As a local Route-M control, a polar disk mesh was paired with the explicit
radial cone map `r -> r^alpha`, whose inverse is `r -> r^(1/alpha)`. At
`256` radial rings and `512` angular samples (`131,073` vertices and `261,632`
faces), both `alpha=0.65` and `alpha=1.4` had zero flipped faces, exact
boundary-circle preservation, and inverse errors below `3.4e-16`. The minimum
face determinants were `5.17e-6` and `2.22e-9`, respectively. This validates a
hard local cone-point chart at realistic resolution. The global control above
now supplies the two-cap seam check, but arbitrary transition data, global
orbifold normalization, and BHF comparison remain open.
Receipt: `artifacts/orbifold_cone_chart_audit/orbifold_cone_chart_audit.json`.

For the analytic Möbius truth path, `mobius_scale_velocity_sphere` gives the
exact velocity with respect to log-scale. A `64x32` finite-difference check was
below `1e-9`. This validates velocity/integration/orientation plumbing for a
known holomorphic flow, but it is not an arbitrary-`mu` BHF solve: the remaining
research problem is still the global variation operator and its two-chart
discretization.

## Arbitrary-base BHF variation reference

The published BHF formula was implemented explicitly for a non-conformal base:

The source is the BHF equation and triangle-discretization discussion in
[Wong--Gu--Chan, *Parallelizable Inpainting and Refinement of Diffeomorphisms
using Beltrami Holomorphic Flow*](https://archive.ymsc.tsinghua.edu.cn/pacm_download/73/642-33._Parallelizable_Inpainting_and_Refinement_of_Diffeomorphisms_using_Beltrami_Holomorphic_Flow.pdf).

\[
 \dot f(w)=-\pi^{-1}\int \nu(z)R(f(z),f(w))f_z(z)^2\,dxdy,
 \quad R(u,v)=\frac1{u-v}-\frac{v}{u-1}+\frac{v-1}{u}.
\]

The implementation takes current source/evaluation images and facewise `f_z`,
so it is not restricted to the identity base. It passes three-point
normalization, linearity, and an identity-base consistency test against the
normalized operator. A fixed seven-point degree-five triangle rule was added
to replace single point samples by a piecewise-affine mesh quadrature. On a
`128x128` regular grid (`32,768` triangles, `229,376` quadrature points and
`2,048` evaluations), the direct operator took `48.20 s` and returned finite
velocities; the `64x64` case took `5.82 s`. Receipt:
`artifacts/bhf_triangle_mesh_audit/bhf_triangle_mesh_audit.json`.

This closes the “no arbitrary-base formula” gap at the reference-operator
level, but not the production solver gate. The diagonal is currently handled
by fixed quadrature (and explicit omission for point collisions), so exact
triangle integrals or a principal-value rule, nonlinear BHF time integration,
and consistent north/south overlap gluing are still required before claiming a
fast differentiable sphere BHF layer. The direct arbitrary-base cost audit is
in `artifacts/bhf_general_variation_audit/bhf_general_variation_audit.json`.

A one-step flow smoke test then evaluated the same arbitrary-base velocity at
all `4,225` vertices of a `64x64`-cell (`8,192`-face) mesh. For explicit
steps `dt=0.01, 0.05, 0.10`, every face stayed positive; the minimum
determinants were `0.63885, 0.63428, 0.62857`, respectively. This is a useful
topology-safety signal for small explicit BHF updates, not a convergence proof
for the nonlinear flow. Receipt:
`artifacts/bhf_one_step_mesh_audit/bhf_one_step_mesh_audit.json`.

## Local Duffy treatment of the BHF pole

To address the remaining vertex-singularity issue, a Duffy coordinate rule was
added for the simple pole in `1/(f(z)-f(w))`. On a `64x64`-cell mesh with
`8,192` triangles and an interior vertex target, order-8 and order-16 local
quadrature produced velocities differing by only `1.76e-7`; the two passes
took `11.55 s` and `37.55 s`. This demonstrates a convergent near-field
primitive at a mesh vertex, but not a global fast assembly: near/far splitting,
all target positions, nonlinear time integration, and atlas gluing remain open.
Receipt: `artifacts/bhf_duffy_singularity_audit/bhf_duffy_singularity_audit.json`.

## Near/far BHF assembly control

The local Duffy primitive was then integrated into a mesh-level near/far
assembly: all faces use the degree-five rule, while faces incident to the
target vertex are replaced by Duffy quadrature. On 16 interior targets of a
`64x64` mesh (8,192 faces), order-16 assembly took `0.724 s`, versus `37.25 s`
for applying Duffy quadrature to every face for one target; the near/far result
differed from the all-Duffy reference by `5.42e-6`. On 16 targets of a
`128x128` mesh (32,768 faces), order-16 assembly took `1.53 s`, and the order-8
versus order-16 discrepancy was `1.67e-10`. All outputs were finite. Receipt:
`artifacts/bhf_near_far_assembly_audit/bhf_near_far_assembly_audit.json`.

This is the first concrete global assembly speedup for the BHF reference
operator, but it currently covers vertex targets only. Arbitrary target
near-field rules, nonlinear time integration, and two-chart gluing remain open.

The same assembly was extended to a target strictly inside one affine face by
splitting that face into three target-centered subtriangles and applying Duffy
quadrature there. At `64x64` and `128x128` (8,192 and 32,768 faces), order-16
versus order-8 differences were `9.75e-6` and `4.80e-6`; order-24 versus
order-16 differences fell to `3.05e-7` and `1.50e-7`. The individual order-24
runs took `0.044 s` and `0.098 s`. This closes the interior-target near-field
case, but edge/vertex multi-face PV cancellation is still separate. Receipt:
`artifacts/bhf_interior_target_audit/bhf_interior_target_audit.json`.

## Interior-edge near/far control

The near/far assembly was extended to a target in an interior shared edge by
splitting both incident triangles into target-centered subtriangles. On 64²
and 128² meshes (8,192 and 32,768 faces), order-16 versus order-8 differences
were `2.02e-7` and `1.02e-7`; order-24 versus order-16 differences fell to
`8.28e-11` and `4.17e-11`. The order-24 evaluations took `0.049 s` and
`0.110 s`. This closes the local edge quadrature primitive, but not a global
multi-edge principal-value cancellation theorem or scalable nonlinear BHF.
Receipt: `artifacts/bhf_edge_target_audit/bhf_edge_target_audit.json`.

A global incident-face audit now evaluates the same far-field assembly with
Duffy replacement on all six faces incident to an interior vertex and on both
sides of an interior edge. At 256² (131,072 faces), the order-16 to order-24
changes were `1.29e-15` (vertex) and `1.41e-10` (edge); the order-8 to order-16
changes were `4.46e-11` and `1.30e-7`. The corresponding 64² and 128²
records show the same monotone order convergence. This is evidence for a
stable incident-face PV control, not a proof of the full nonlinear BHF
principal-value theorem. Receipt:
`artifacts/bhf_global_pv_cancellation_audit/bhf_global_pv_cancellation_audit.json`.

The same incident-face PV cancellation audit was raised to a `512x512` grid
(`524,288` faces). For the vertex target, order-8 to order-16 and order-16 to
order-24 changes were `1.12e-11` and `3.20e-16`; for the interior-edge target
they were `6.51e-8` and `7.09e-11`. All six vertex incident faces and both edge
sides remained finite. This extends the local/global cancellation control to
the realistic half-million-face scale, but it is still a quadrature convergence
receipt rather than a proof of the nonlinear BHF PV theorem. Receipt:
`artifacts/bhf_global_pv_cancellation_audit_512/bhf_global_pv_cancellation_audit.json`.

## Adaptive nonlinear flow control

The near/far assembly was used inside an adaptive explicit flow on a `32x32`
mesh (2,048 faces). Two successive steps accepted the requested `dt=0.2`
under the exact determinant-root controller; safe bounds were `13.889` and
`13.729`, minimum determinants remained `0.6342` and `0.6284`, and the
independent injectivity audit certified both states with zero flips. The full
two-step run took `22.63 s`, dominated by vertex velocity assembly. Receipt:
`artifacts/bhf_near_far_flow_audit/bhf_near_far_flow_audit.json`.

This is a genuine nonlinear flow-control result, but not yet convergence to a
prescribed target: the planar regular-grid experiment still needs sphere atlas
coupling of the *BHF operator*, edge/vertex PV cancellation, and a scalable
velocity backend.

The same near/far explicit flow was raised to a `64x64` mesh (`8,192` faces)
for one determinant-safe step. Velocity assembly and the topology audit took
`63.43 s`; the exact safe bound was `13.89`, the accepted step was `0.2`, the
minimum determinant remained `0.6343`, and zero faces flipped. This is realistic
resolution evidence for the nonlinear flow cost and safety margin, while also
showing why a production fast velocity backend is still required. Receipt:
`artifacts/bhf_near_far_flow_audit_64/bhf_near_far_flow_audit.json`.

The same near/far flow was raised to `96x96` cells (`18,432` faces) for one
determinant-safe step. Velocity assembly and the topology audit took
`216.16 s`; the exact safe bound was `13.8892`, the accepted step was `0.2`,
the minimum determinant remained `0.63431`, and zero faces flipped. This is
realistic-resolution evidence that the Duffy-corrected explicit flow remains
topology-safe, while also quantifying the current reference backend's poor
scaling. Receipt: `artifacts/bhf_near_far_flow_audit_96/bhf_near_far_flow_audit.json`.

## Normalized BHF variation operator (conformal-base reference)

The normalized first-variation formula with fixed `0,1,infinity` was implemented
as a blocked Cauchy quadrature. It exactly honors the three normalization
points and is linear in the Beltrami variation. A direct cost audit took
`0.371 s` for 4,096 source / 2,048 evaluation points and `2.942 s` for 16,384
source / 4,096 evaluation points. This is valuable evidence for the BHF
velocity kernel and its normalization, but it is deliberately not promoted to
an arbitrary-base nonlinear BHF solver: the principal-value treatment,
base-μ variation formula, and two-chart gluing remain open. Receipt:
`artifacts/bhf_variation_cost_audit/bhf_variation_cost_audit.json`. Its kernel
orientation is checked against the arbitrary-base formula at the identity base.
