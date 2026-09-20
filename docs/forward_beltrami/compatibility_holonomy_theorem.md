# Route A: discrete-realizability holonomy theorem (P1 disk)

This note separates a derivation from a numerical observation. It addresses the
remaining Route-A question: when does a facewise constant Beltrami field come
from one continuous, orientation-preserving P1 map on a fixed triangulation?

The literature uses the same P1 fact that a map is affine on every triangle
and that its Beltrami coefficient is constant per triangle; see [Fargion et
al., *Fast Injective Mesh Parameterization via Beltrami Coefficient
Prolongation*](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70341) and the
edge/1-form viewpoint in [Weber and collaborators, *Computing Extremal
Quasiconformal Maps*](https://cims.nyu.edu/gcl/papers/weber2012ceq.pdf). The
theorem below is the fixed-mesh compatibility statement needed by this project;
it is not claimed as a new continuous measurable-Riemann-mapping theorem.

## Setup

Let `K` be a connected, oriented triangulation of a planar domain. On face
`T`, write the complex affine derivative as

\[
 f_T(z)=a_T z+b_T\bar z+c_T,
 \qquad b_T=\mu_T a_T,
 \qquad |\mu_T|<1.
\]

For an oriented source edge `e=z_j-z_i` of `T`, define

\[
 q_T(e)=e+\mu_T\bar e.
\]

The image increment along this edge is `a_T q_T(e)`. Because `|mu_T|<1`,
`q_T(e) != 0` for every nonzero source edge: otherwise
`|mu_T|=|e/\bar e|=1`.

## Theorem (simply-connected P1 disk)

Assume the underlying triangulated domain is a topological disk. A facewise
field `mu_T`, with `|mu_T|<1`, is the Beltrami field of a continuous P1 map
with nonzero `f_z` on every face if and only if the following dual holonomy
conditions hold:

1. For every interior edge shared by faces `T` and `S`, choose one common edge
   orientation and require
   \[
   a_T q_T(e)=a_S q_S(e).
   \]
2. Equivalently, define the nonzero transition ratio
   \[
   r_{TS}(e)=q_T(e)/q_S(e).
   \]
   The product of `r` around every closed walk in the dual graph must be one.

When these conditions hold, the collection `a_T` is unique up to one global
complex multiplier. The resulting map is unique up to one global complex
similarity (complex scale plus translation).

### Proof, direction one: a map implies holonomy

Continuity on a shared edge makes the two affine restrictions agree for every
point on that edge. Differentiating in its tangent direction gives
`a_T q_T(e)=a_S q_S(e)`, hence `a_S/a_T=r_{TS}`. Multiplying ratios around a
dual closed walk telescopes to one.

### Proof, direction two: holonomy implies a map

Choose one root face and a nonzero `a_root`. Propagate `a` through a spanning
tree using `a_S=a_T r_{TS}`. Trivial dual holonomy makes this value independent
of the chosen path, so every face receives a nonzero `a_T`.

For each oriented primal edge define the complex increment
`p_e=a_T q_T(e)` using either incident face. The shared-edge equations make
this independent of the incident-face choice. On each triangle,

\[
\sum_{e\in\partial T}p_e
 =a_T\left(\sum_{e\in\partial T}e
 +\mu_T\sum_{e\in\partial T}\bar e\right)=0.
\]

Thus `p` is a closed primal 1-form. A disk has no nontrivial first homology,
so a vertex potential `f_i` exists with `f_j-f_i=p_{ij}`. Affine interpolation
on every face gives a continuous P1 map and recovers exactly the prescribed
`a_T` and `mu_T`. Since `a_T != 0` and `|mu_T|<1`,

\[
 \det Df_T=|a_T|^2(1-|\mu_T|^2)>0.
\]

The only free choice in propagation is `a_root`; integrating the edge 1-form
adds one translation, which proves the stated similarity gauge.

## Dimension consequence

The dual graph has `F` face vertices and `E_int` interior-edge links. Its cycle
rank is `E_int-F+1`. Holonomy gives one complex scalar condition per independent
dual cycle, while the compatible scale field has one global complex degree of
freedom. Therefore a generic compatible disk field has complex nullity one;
an arbitrary bounded facewise field usually violates at least one cycle and
has no nonzero compatible scale field. This explains the observed contrast
between the manufactured and random fields without invoking a dense nullspace
calculation.

The count is a generic rank statement, not a claim that every cycle equation
is numerically independent on a degenerate mesh. Degenerate geometry and
`|mu|` near one can make the compatibility matrix badly conditioned.

## Rectangle boundary characterization

The theorem above only constructs a disk map. To obtain a rectangle-to-
rectangle homeomorphism, add a boundary model:

1. designate four ordered boundary chains and four corner vertices;
2. map the corners to four distinct target corners in cyclic order;
3. constrain each chain to its corresponding target side; and
4. require strictly monotone tangential increments along every chain.

These conditions make the boundary restriction an orientation-preserving
homeomorphism of the boundary circle. If all face determinants are positive,
the planar degree/winding argument then gives global injectivity: every regular
target point has positive local degree at each preimage, while the boundary
degree is one, so there can be exactly one preimage. A fully formal treatment
must state the handling of vertices and target points on edges separately; the
present paragraph is the proof contract used by the numerical audits, not a
replacement for that topological lemma. The mesh-degree literature provides a
closely related framework for bijective PL maps with adjustable boundaries:
[Bijective Mappings of Meshes With Boundary and the Degree](https://arxiv.org/abs/1310.0955).

## Multiply-connected warning

For a domain with holes, dual holonomy still constructs the face scales, but a
closed primal 1-form need not be exact. One must additionally impose zero
periods around a basis of primal homology cycles (or prescribe the periods,
which become torus/annulus moduli). This is the discrete reason that a local
Beltrami compatibility test is insufficient for torus and annulus decoders.

## Numerical evidence in this repository

The derivation is checked against the existing realistic-resolution receipts:

- `artifacts/holonomy_reconstruction_audit_256/holonomy_reconstruction_audit.json`:
  `66,049` vertices, `131,072` faces, manufactured dual-holonomy residual
  `1.05e-14`, primal closure `9.19e-15`, and aligned reconstruction error
  `1.22e-15`; the bounded random field has holonomy up to `7.64e25`.
- `artifacts/compatibility_boundary_theorem_audit_512/compatibility_boundary_theorem_audit.json`:
  `524,288` faces, zero flips, minimum signed-area ratio `0.65459`, ordered
  rectangle-side boundary, and winding-one samples.
- `artifacts/compatibility_rectangle_projection_audit_256/compatibility_rectangle_projection_audit.json`:
  fixed side-sliding rectangle boundary, relative face-`mu` error `3.01e-9`,
  zero flips, and minimum signed-area ratio `0.65460`.

These receipts support the theorem's computational implications but do not by
themselves prove the projection uniqueness, arbitrary-`mu` solvability, or an
exact second-order differentiable projection chart. Those remain open Route-A
gates.
