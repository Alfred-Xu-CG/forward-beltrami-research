# Phase III correction audit

Status: independently checked for C1, C2, C3, C5, and C7; C4 and C6 require
scope wording changes before closure. The continuum canonical-map question in
C7 remains separate from the fixed-(P_1) counterexample.

This document audits the seven concrete failure modes identified by the Phase
III master plan. It separates code defects, mathematical overclaims, and
missing validation paths.

| ID | Previous issue | Independent finding | Fix/status |
|---|---|---|---|
| C1 | Electrical rectangle modulus/energy and boundary indexing may be off by one. | **PASS:** old 9x7 value 0.75 omitted one horizontal strip; corrected reference returns (n_y/(n_x-1)). Independent 2x2, 3x2, 3x3 graph calculations agree for energy, right flux, and area. | Keep the weighted extension explicitly restricted to row-path-independent rectangular networks. |
| C2 | Four-direction tensor decomposition was called unique. | **PASS with API wording fix:** rank is 3 with null vector ((-1,-1,1,1)). Exact signed decompositions allow any finite mass (t); the returned interval is only the nonnegative-conductance interval (|b|\le t\le\min(a,c)). | Source now accepts finite signed (t), reports `feasible`, and names `positive_feasible_interval`; compatibility alias remains. |
| C3 | Local anisotropic nonobtuse condition was written as a global M-matrix necessity. | **PASS:** local transformed nonobtuse is sufficient only. A 120°/30° pair has one positive local entry but negative assembled edge sum. | Use assembled weighted cotangent sums and reserve “(M)-matrix” for all sign/connectivity conditions. |
| C4 | Directed Tutte strict-convex-boundary wording did not match collinear rectangle subdivisions. | **REVISE/PARTIAL FIX:** strict convexity applies to the corner polygon; the subdivided loop is weakly convex. The implicit decoder now rejects repeated, negatively turning, clockwise, or degenerate boundaries, but still does not certify 3-connected-to-boundary. | Keep the graph certificate as a caller obligation before topology claims. |
| C5 | Rectangle latent advertised learnable aspect ratio while width/height were Python floats. | **PASS:** tensor-valued width/height and (H=\varepsilon+\operatorname{softplus}(m)) pass an independent finite-difference gradient check (absolute error (1.6\times10^{-10})). | Document that learnability requires tensor arguments or the modulus-logit helper. |
| C6 | Tutte universality was simultaneously asserted and listed as open. | **REVISE:** local-star positivity is valid only for an already valid PL embedding with positive areas, cyclic stars, and boundary-reachable support. It is not arbitrary-(\mu) universality. | Rename scope to surjectivity onto valid embeddings with fixed boundary; keep graph hypotheses explicit. |
| C7 | Mixed-boundary monotonicity remained open but random tests were treated as progress toward closure. | **COUNTEREXAMPLE for fixed facewise-(P_1) complementary solve:** a 3x3 vertex mesh with (|\mu|=0.8) yields left middle-row increment (-0.10057537) despite every face tensor SPD. This does not settle continuum canonical-map monotonicity. | Remove universal discrete monotonicity claim; retain continuum, isotropic graph, and compatible-(P_1) cases as separate questions. |

## Evidence and adjudication

The independent checker report is `tmp/phase3_correction_checker.md`. The
corrected electrical and decomposition tests are green under
`PYTHONPATH=src`. C4 and C6 remain conditional until the directed Tutte graph
and boundary preflight are implemented or explicitly made caller obligations.
No Phase III claim should be called universally closed merely because the
builder's own tests pass.
