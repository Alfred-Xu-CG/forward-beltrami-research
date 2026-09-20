# Phase II final decision

Status: converged provisional architecture decision (Phase II reset).

## 1. First candidate: directed positive Tutte map-primary layer

The current first candidate is

\[
z\longrightarrow
\bigl(\text{positive directed row logits},\text{convex boundary latent}\bigr)
\longrightarrow
Y_h,
\]

where the interior coordinates solve

\[
(I-P_{II})Y_I=P_{IB}Y_B,
\qquad p_{ij}=\operatorname{softmax}_j(\ell_{ij}).
\]

Why it is first:

- Under the explicit planar graph, boundary-cycle, convex-boundary, and
  strictly-positive-row hypotheses, the directed Tutte theorem supplies the
  strongest available hard-topology statement: a non-overlapping straight-line
  embedding with positive triangle areas.
- The forward and backward passes use one sparse solve and one transpose solve;
  the measured implicit VJP check passed, and existing realistic-resolution
  audits had zero flips.
- The new boundary latent uses four side-wise softmax vectors. It produces a
  closed, strictly ordered rectangle for every finite logit, and its composition
  with the directed solve had 128 positive faces and finite gradients on an
  8-by-8-cell mesh.

What it does **not** yet prove is arbitrary Beltrami expressivity. It is a
hard-bijective map decoder whose weights may be learned to approximate a QC
objective; it is not an exact solver for every prescribed \(\mu\). A general
convex-polygon latent also remains open.

## 2. Second candidate: MBM-LBS implicit layer

The MBM-LBS route remains the strongest QC-fidelity candidate:

\[
\mu\longrightarrow A(\mu)\longrightarrow K(A)
\longrightarrow (u,w)\longrightarrow (u,Mw).
\]

The custom implicit layer has a double-precision directional VJP relative error
of \(9.4\times10^{-10}\). On a smooth field, forward plus backward took 1.56 s,
6.13 s, and 26.21 s at 64, 128, and 256 grid resolutions, with about 485 MB
RSS at 256. The reference maps tested at 128 and 256 had positive triangle
determinants and monotone side traces.

It is not first because the independent complementary solve is not an exact
fixed-P1 conjugate for arbitrary facewise \(\mu\), and the mixed boundary
conditions do not currently provide a theorem that the recovered side traces
are globally ordered. The decreasing minimum determinant at 256 is a warning
about conditioning, not a fold-free theorem. A topology-safe boundary closure
or a compatible primal-dual discretization is required before this can be a
hard-bijective production layer.

## 3. Supporting primal-dual/electrical route

For a quadrilateral planar network with positive scalar conductances, the mixed
Dirichlet--Neumann graph problem has a published embedded rectangle-tiling
theorem and an energy--area identity. Our 9-by-7 \(\mu=0\) reference reproduced
the linear potential to \(4.44\times10^{-16}\), with modulus and total tiling
area both 0.75.

The anisotropic gate is not yet viable on a fixed stencil. The exact four-
direction decomposition of \(A=\left[\begin{smallmatrix}a&b\\b&c\end{smallmatrix}\right]\)
requires \(c_x=a-|b|\) and \(c_y=c-|b|\). The angular audit showed positive
conductances for only 18.46% of phases at \(|\mu|=0.6\), 3.22% at 0.8, and
0.68% at 0.9, despite tensor reconstruction error below
\(3.6\times10^{-15}\). Thus the route is a rigorous isotropic supporting
decoder, not yet an anisotropic QC layer.

## 4. Claims that were falsified or restricted

1. **False:** (A\succ0) alone implies that the P1 stiffness matrix is an
   M-matrix. Counterexamples occur when transformed element angles are obtuse;
   the phase diagram contains negative off-diagonal entries.
2. **False as a generic fixed-mesh claim:** solving an independent complementary
   mixed problem automatically produces the exact conjugate for every bounded
   facewise \(\mu\). Fixed-P1 holonomy/compatibility conditions can fail, and the
   measured conjugacy residual is nonzero.
3. **False:** a fixed positive axis/diagonal Hodge stencil represents all
   \(|\mu|<1\). The angular sign audit above gives a direct counterexample.
4. **Not established:** natural mixed FEM boundary conditions alone imply a
   globally ordered rectangle boundary. Random positive-conductance searches
   found no small counterexample, but that is evidence, not a theorem.
5. **Qualified theorem:** every valid straight-line embedding of a fixed disk
   triangulation with a strictly convex boundary has a strictly-positive
   directed Tutte representation, by the angular-star and substochastic
   uniqueness argument in `04_tutte_latent.md`. This does not establish a
   direct representation for every prescribed Beltrami field, non-convex
   target, or degenerate face margin.

## 5. Frozen routes for this phase

The following remain archived supporting evidence, not active production
branches: planar BHF quadrature/FMM/GPU engineering, generic periodic/free-space
Beurling acceleration, sphere/orbifold and torus specializations, high-genus
constructions, generic barrier/SLIM/AMIPS optimization, learned preconditioners,
optimal transport, circle packing, full diffusion training, and 1024-squared
stress tests. Their existing artifacts are preserved under `artifacts/` and
`docs/forward_beltrami/`.

## 6. Highest-value next tasks (at most three)

1. Prove or refute the precise directed-positive Tutte universality statement
   for a fixed triangulation, and replace the rectangle-only boundary latent by
   a closed convex-polygon parameterization if the theorem permits it.
2. Implement one adaptive/rotated anisotropic Hodge construction (or a
   compatible DEC/orthodiagonal variant) and measure both positive-conductance
   coverage and Beltrami error before investing in a primal-dual neural layer.
3. Add a fair 256-resolution benchmark with identical mesh, latent, topology,
   gradient, and memory metrics for directed Tutte versus MBM-LBS, including a
   boundary-closure variant for MBM.

The first candidate is therefore a hard-bijective geometric decoder with
learnable QC loss, while the second is a differentiable QC solver prototype
whose topology theorem is still incomplete. Neither is honestly claimed to
already satisfy every item of the ultimate objective (exact arbitrary-μ
fidelity, guaranteed PL homeomorphism, fast backward, and bounded high-resolution
memory) simultaneously.
