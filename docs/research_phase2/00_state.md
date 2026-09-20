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
   but its expressivity as a latent parameterization must be quantified.

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

## Next decisive questions

- Does the continuous mixed problem plus the conjugate coordinate actually
  imply ordered side traces, or is a boundary closure necessary?
- Under which mesh geometry and facewise A_T conditions is the P1 stiffness
  matrix an M-matrix?
- Can a mu=0 primal-dual electrical construction produce a non-overlapping
  rectangle tiling before anisotropy is attempted?

## Resource policy

Tiny experiments are capped at approximately two minutes, medium experiments at
approximately ten minutes, and large experiments at approximately thirty
minutes unless a written high-information reason justifies an exception.
