# Route D: Tutte/Floater hard-bijective baseline

## Decoder

`qcopt.forward.tutte.tutte_embedding` fixes a counter-clockwise convex boundary
and solves positive uniform barycentric equations for all interior vertices.
The resulting map is audited independently with the existing face-orientation,
boundary, and branch-index checks. The implementation rejects non-convex target
boundaries instead of returning an unqualified map.

## Evidence

- Unit tests pass for a convex rectangle deformation and for rejection of a
  non-convex boundary.
- Local `256x256` structured mesh (`131072` faces): sparse solve in `10.54 s`,
  minimum determinant `2.2499999999973355`, zero flipped faces, and an independent
  injectivity audit marked the map certified.
- Remote ai `512x512` structured mesh (`524288` faces, `261121` interior
  unknowns): sparse solve in `12.08 s`, peak RSS about `507 MiB`, minimum
  determinant `2.25`, zero flipped faces, and maximum interior error against the
  known affine solution `2.63e-13`.

## What this proves and does not prove

This is a genuine hard-bijective differentiable-layer candidate when the target
boundary is convex and the mesh graph satisfies the usual Tutte hypotheses. It
is not an arbitrary Beltrami coefficient solver: the interior distortion is
selected by the harmonic/convex-combination equations, so expressivity and
`mu` fidelity must be measured against the realizable-BC benchmark. The next
tasks are positive nonuniform weights, implicit differentiation, and an
expressivity comparison against manufactured QC maps.

## Differentiation

For small/medium meshes, `tutte_weights` exposes the harmonic barycentric
matrix and `tutte_embedding_torch` applies it as a differentiable linear layer.
Two PyTorch tests verify finite gradients and exact agreement with the explicit
transpose-Jacobian formula. The dense matrix is intentionally not used for the
512x512 benchmark: its boundary-by-vertex storage would be excessive. A
production high-resolution layer should use the same sparse factorization with
an adjoint solve or matrix-free JVP/VJP.

That sparse implicit layer is now implemented as `TutteImplicitSystem` /
`tutte_embedding_torch_implicit`. On a local `256x256` mesh (`65025` interior
unknowns, `131072` faces), factorization took `3.65 s`, forward evaluation
`0.033 s`, and backward adjoint `0.064 s`; all gradients were finite. The
boundary coupling remains sparse, so this measurement does not pay the dense
boundary-weight memory cost of the small-mesh explicit layer.

## Expressivity audit

On a `256x256` mesh with the identity square boundary, a manufactured
non-harmonic interior map was constructed by adding smooth sine displacements
that vanish on the boundary. Uniform Tutte necessarily returns the identity
map for that boundary, while the truth remains orientation preserving. The
maximum interior vertex error was `0.141707` and RMS error `0.062142`; the
decoded identity still passed the independent injectivity certificate. Thus
positive harmonic decoding gives a strong hard-bijective guarantee, but it
cannot represent arbitrary interior Beltrami fields without richer positive
weights, additional interior constraints, or a composition of decoders.

## Positive nonuniform weights

`tutte_embedding_weighted` now accepts a strictly positive sparse edge-weight
field rather than uniform weights. At `512x512` it assembled/solved a graph with
`263,169` vertices, `524,288` faces, and `787,456` edges in `22.35 s`. The
minimum face determinant was `5.04e-6`, there were zero flipped faces, and the
independent injectivity audit was certified. Thus nonuniform learned weights
are compatible with the hard Tutte guarantee as long as strict positivity and
convex boundary conditions are maintained. This expands expressivity, although
it still does not make the decoder an arbitrary Beltrami solver.
Receipt: `artifacts/tutte_weighted_audit/tutte_weighted_audit.json`.

## Learned-weight implicit VJP

The weighted decoder now also exposes a sparse transpose-solve VJP with
respect to the undirected edge weights. A finite-difference test agrees with
the analytic edge derivative on a `5x5` mesh. On a `256x256` mesh (`66,049`
vertices, `131,072` faces, `197,120` edges), the weighted solve plus VJP took
`7.54 s`; all output and edge gradients were finite. Receipt:
`artifacts/tutte_weighted_vjp_audit/tutte_weighted_vjp_audit.json`.

The VJP is local to the strictly-positive-weight region. It does not license
signed weights, and it does not remove the interior expressivity limitation of
Tutte/Floater decoding.

## Edge-weight expressivity optimization

The edge VJP was used in a positive-log-weight Adam toy to fit the manufactured
non-harmonic, identity-boundary map from the expressivity audit. On a `32x32`
mesh (`2,048` faces and `3,136` edges), 60 VJP updates reduced map MSE from
`9.01e-4` to `6.19e-7`, with minimum determinant `5.59e-4` and zero flips. A
realistic `64x64` run (`8,192` faces and `12,416` edges) reduced MSE from
`9.29e-4` to `4.41e-5` in 20 updates, with minimum determinant `1.33e-4` and
zero flips. This shows that learned positive weights can materially expand
Tutte expressivity while retaining the hard positivity regime; it is not yet a
universal arbitrary-μ representation or a converged training algorithm.
Receipts: `artifacts/tutte_weight_learning_audit/tutte_weight_learning_audit.json`
and `artifacts/tutte_weight_learning_audit_64/tutte_weight_learning_audit.json`.

The same positive-log-weight optimization was pushed to `128x128` cells
(`32,768` faces and `49,408` edges) for 12 updates. Map MSE fell from
`9.43e-4` to `9.97e-5` (ratio `0.106`), the minimum determinant remained
`3.37e-5`, and zero faces flipped. The final loss oscillation and finite
weight range (`0.474` to `2.39`) show that this is still an optimization toy,
but the hard positivity certificate survives a realistic larger mesh.
Receipt: `artifacts/tutte_weight_learning_audit_128/tutte_weight_learning_audit.json`.

The same positive-log-weight optimization was then raised to `256x256` cells
(`131,072` faces and `197,120` edges) for eight updates. Final map MSE fell
from `9.50e-4` to `8.58e-5` (ratio `0.0903`), the minimum face determinant was
`8.34e-6`, and zero faces flipped. The trajectory oscillated before recovering,
so this is realistic-resolution evidence for a stable hard decoder plus
nontrivial expressivity—not evidence of arbitrary-μ fitting. Receipt:
`artifacts/tutte_weight_learning_audit_256/tutte_weight_learning_audit.json`.

The same positive-log-weight optimizer was finally exercised at `512x512`
(`524,288` faces and `787,456` edges) for three updates. The map MSE fell from
`9.541e-4` to `5.168e-5` (ratio `0.0542`); the minimum face determinant was
`2.704e-6`, with zero flipped faces. The short run is intentionally a stress
test rather than a convergence claim: it demonstrates that the sparse
weighted-Tutte/VJP pipeline remains finite and hard-injective at realistic
resolution, while the very small determinant and limited iteration budget
show why positive weights alone do not establish a robust arbitrary-QC
decoder. Receipt: `artifacts/tutte_weight_learning_audit_512/tutte_weight_learning_audit.json`.

## Determinant-aware positive-weight optimization

The positive-log-weight optimizer was upgraded with a trust-region/backtracking
step: every candidate is accepted only if the map MSE decreases and the
minimum face determinant remains above a prescribed margin. On a `256x256`
mesh (`131,072` faces and `197,120` edges), all 12 updates were accepted with
monotone loss decrease. MSE fell from `9.5043e-4` to `3.0077e-6` (ratio
`0.00316`), the final minimum determinant was `9.31e-6`, and zero faces flipped.
The accepted log-step shrank from `0.5` to `0.03125` as the optimizer
approached the hard map boundary. Receipt:
`artifacts/tutte_weight_learning_safe_256/tutte_weight_learning_safe_audit.json`.

This is a stronger neural-decoder control than the unconstrained Adam stress:
the hard positivity regime is retained while the optimizer explicitly respects
a determinant margin. It remains an optimization result rather than a proof
of arbitrary-Beltrami expressivity, and each backtracking trial requires a
sparse solve, so its training cost must be addressed separately.

## Symmetric versus nonsymmetric positive weights

The roadmap asks whether hard injectivity is tied to symmetric edge-energy
weights or only to positivity. A `256x256` mesh (`66,049` vertices,
`131,072` faces) was tested with the same identity convex boundary:

| row model | min signed area | flips | independent certificate |
|---|---:|---:|---|
| symmetric positive edge weights | `9.32e-3` | 0 | yes |
| directed positive row weights, log-spread 1 | `1.97e-3` | 0 | yes |
| directed positive row weights, log-spread 3 | `4.94e-10` | 0 | yes |

All three solves were finite, had positive boundary orientation, no boundary
intersections, and no bad interior branch vertices. This is evidence that a
positive directed decoder can remain injective in this planar stress without
an SPD energy. The heterogeneous directed case approaches the determinant
margin by four orders of magnitude, so this is not a nonsymmetric Tutte
theorem or a differentiable directed-weight layer. Receipt:
`artifacts/tutte_symmetric_nonsymmetric_256/tutte_symmetric_nonsymmetric_audit.json`.

## Direct induced-μ fitting through the positive implicit layer

The earlier edge-weight experiments optimized vertex-map MSE. A separate
control now differentiates the induced facewise Beltrami coefficient itself
through the directed positive-row implicit layer. On a `128x128` mesh, 20
Adam steps at learning rate `0.02` reduced induced-μ MSE from `0.0228557` to
`0.00978713` (ratio `0.428`), with finite gradients, zero flipped faces,
minimum signed-area ratio `0.6743`, and a passing independent injectivity
audit. At `256x256`, 10 steps at learning rate `0.005` reduced MSE from
`0.0228584` to `0.00871626` (ratio `0.381`), again with finite gradients,
zero flips, minimum signed-area ratio `0.4427`, and a passing injectivity
audit. Receipts: `artifacts/tutte_mu_fit_128_lr002/tutte_mu_fit_audit.json`
and `artifacts/tutte_mu_fit_256_lr0005/tutte_mu_fit_audit.json`.

The same `256x256` case at learning rate `0.02` retained zero flips but
worsened induced-μ MSE to `0.107407` and reduced the minimum area ratio to
`0.1828`, exposing a resolution/conditioning dependence rather than a
topology failure. A deliberately aggressive `128x128`, learning-rate-`0.25`
stress drove the optimizer into numerical near-degeneracy (3,194 reported
negative-area faces and minimum signed area about `-1.38e-13`); this is
recorded as an optimizer failure mode, not as a counterexample to the
positive-row Tutte theorem. Receipts:
`artifacts/tutte_mu_fit_256_lr002/tutte_mu_fit_audit.json` and
`artifacts/tutte_mu_fit_128/tutte_mu_fit_audit.json`.

These runs establish that hard topology and direct induced-μ fidelity can be
optimized jointly in a stable regime. They do not establish arbitrary-μ
expressivity, a global convergence theorem, or a production-ready optimizer;
the learning-rate sensitivity and per-step sparse implicit solve remain open
gates.
