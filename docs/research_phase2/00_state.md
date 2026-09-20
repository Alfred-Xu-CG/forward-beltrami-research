# Phase II research state

## Status

Phase II is initialized from the approved master plan. The repository is now in
research-convergence mode. The legacy 'docs/forward_beltrami/' tree is preserved
as evidence and reusable code; it is not the current agenda.

Authoritative plan: [PLAN.md](PLAN.md)

## Current hypotheses

1. **MBM-LBS:** an anisotropic mixed-boundary conductivity solve may provide a
   fast implicit layer, but a global PL-homeomorphism guarantee requires a
   separate boundary-order theorem or closure.
2. **Primal-dual QC:** a positive conductance / discrete-Hodge construction may
   produce boundary order intrinsically, possibly on a refined primal-dual mesh.
3. **Directed Tutte:** positive directed weights give a hard-safe map decoder,
   and every valid straight-line embedding with a strictly convex fixed
   boundary admits such weights; direct arbitrary-\(\mu\) expressivity remains
   separate.

## Completed in this reset

- PLAN.md and the root AGENTS.md were created from the supplied planning files.
- A real implementation bug was found in
  src/qcopt/forward/rectangle_conductivity.py: interior entries of
  boundary_values were used before the conductivity solve. The solver now
  uses boundary entries only, solves u, and then reconstructs the flux from
  the solved u.
- Three focused assertions pass for affine recovery, a smooth manufactured
  map, and arbitrary corrupted interior inputs. Ordinary pytest is currently
  affected by the Anaconda environment's duplicate Intel OpenMP runtime;
  running the same focused test with the isolated test environment gives
  '3 passed'.
- The new MBM-LBS reference module passes constant-real-\(\mu\) exact tests.
  On a smooth variable field, 16x16, 32x32, and 64x64 grids all had positive
  face determinants and monotone left/right side traces. The conjugacy residual
  decreased from 0.1676 to 0.0536, so the independent complementary solve is
  convergent evidence, not an exact fixed-mesh conjugacy theorem.
- The \(\mu=0\) electrical-grid unit test recovered the exact linear potential
  and positive rectangle cells; a 9x7 grid gave modulus and total tiling area
  0.75. **Phase III correction:** the old 0.75 value omitted a boundary strip;
  the independently derived 9x7 value is 0.875. Phase II remains
  reconnaissance only; see `docs/research_phase3/03_electrical_primal_dual.md`.
- A concrete four-direction diamond Hodge stencil exactly reconstructs every
  sampled Beltrami conductivity tensor (maximum reconstruction error
  \(3.6\times10^{-15}\), but its conductances stay nonnegative for only 18.46%
  of phases at \(|\mu|=0.6\), 3.22% at 0.8, and 0.68% at 0.9. This is a
  decisive sign/positivity limitation, not a tensor approximation error.
- The existing directed Tutte implicit layer passed its focused VJP and
  unstructured topology audits. Its fixed-convex-boundary valid-PL
  representation theorem is now recorded; direct Beltrami-field expressivity
  is still open.
- Legacy fixed-P1 holonomy evidence is now incorporated into the Phase II
  formulation: an arbitrary bounded facewise \(\mu_T\) is generally not
  realizable by a continuous P1 map. This is why a small conjugacy residual in
  the independent MBM complementary solve cannot be interpreted as only a
  sparse-solver error.
- The projected-Beurling formulation is retained as supporting theory only:
  its range projector explains the fixed-P1 compatibility manifold, while the
  existing realistic random-field control retains relative Beltrami error about
  0.7081. It is not being promoted to a generic production backend.
- A 16x16 smooth boundary-fixed PL deformation with positive face determinants
  admitted strictly positive local directed-Tutte rows at all 225 interior
  vertices; the smallest maximin row weight was 0.139. This is initial evidence
  that the hard decoder is not trivially too restrictive.
- A differentiable rectangle boundary latent was added for directed Tutte:
  four side-wise softmax vectors produce strictly positive segment lengths and
  an exactly closed convex rectangle. On an 8x8-cell mesh, composing this
  boundary with the directed implicit solve gave 128 positive face
  determinants and finite gradients for both boundary and interior logits.
- A theorem/proof note now establishes strict directed-weight universality for
  any valid straight-line embedding of a fixed disk triangulation with a
  strictly convex boundary: positive star angles imply each interior vertex is
  in the strict neighbor convex hull, and boundary reachability makes the
  substochastic solve unique. This does not cover arbitrary prescribed
  Beltrami fields or non-convex targets.
- At 128x128 and 256x256, the MBM reference output retained positive face
  determinants and monotone side traces, but the minimum determinant decreased
  to \(6.08\times10^{-6}\) at 256x256. This is evidence for the tested field,
  not a global hard-bijection certificate.

## Provisional candidate snapshot

| Candidate | differentiable evidence | topology evidence | 256-scale cost | main unresolved gate |
|---|---|---|---|---|
| MBM-LBS implicit | directional VJP relative error \(9.4\times10^{-10}\) | positive tested field, but no mixed-boundary theorem | 26.21 s forward+backward; RSS about 485 MB | side-order/global homeomorphism |
| Directed Tutte | implicit VJP, boundary-logit composition, and unstructured tests pass | fixed-convex-boundary valid-PL universality theorem; zero flips in legacy 256 audit | about 6.4 s for 65,536 vertices | direct \(\mu\)-to-weights map and general convex boundary latent |
| Primal-dual electrical | exact \(\mu=0\) bookkeeping; mixed-D/N rectangle theorem supports graph decoder | embedded rectangle tiling is guaranteed for the quadrilateral isotropic theorem; fixed diamond anisotropic stencil has a narrow positive cone | not yet benchmarked | adaptive/rotated positive Hodge star and PL realization |
- The structured MBM implicit layer now passes a double-precision directional
  finite-difference VJP check with relative error \(9.4\times10^{-10}\).
  Forward-plus-backward wall time was 1.56 s at 64x64, 6.13 s at 128x128,
  and 26.21 s at 256x256; process RSS increased to about 485 MB at 256x256.
  This is a real differentiable prototype, but its current Python assembly,
  sparse factorization cost, and mixed-boundary topology gap prevent a
  production-layer claim.

## Next decisive questions

- Does the continuous mixed problem plus the conjugate coordinate actually
  imply ordered side traces, or is a boundary closure necessary?
- Under which mesh geometry and facewise A_T conditions is the P1 stiffness
  matrix an M-matrix?
- Can a mu=0 primal-dual electrical construction produce a non-overlapping
  rectangle tiling before anisotropy is attempted? (The quadrilateral mixed
  Dirichlet--Neumann theorem now answers this at the graph-to-rectangle level.)
- Can the rectangle boundary latent be extended to a useful convex-polygon
  latent without losing cyclic closure or creating ill-conditioned rows?

## Resource policy

Tiny experiments are capped at approximately two minutes, medium experiments at
approximately ten minutes, and large experiments at approximately thirty
minutes unless a written high-information reason justifies an exception.
