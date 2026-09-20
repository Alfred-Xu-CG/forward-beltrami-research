# Phase III worklog

## T+0 — reset and correction audit

Question: Which Phase II claims or implementations are too strong or incorrect?
Exact claim: The six corrections named by the Phase III plan can be independently checked before new theory is accepted.
Assumptions: The Phase II repository state and artifacts are the evidence under audit; no final ranking is valid before T+22h.
What would falsify it: A listed correction is not reproducible, or a supposedly independent check agrees only because it reuses the same code path.
Smallest decisive test: Re-derive the electrical bookkeeping, four-direction degrees of freedom, theorem hypotheses, and learnable-modulus gradient with independent calculations.
Prior work: docs/research_phase2/ and the Phase III master plan.

Next: create the correction-audit draft, dispatch an independent checker, then close only confirmed issues before moving to continuum MBM.

## T+0.5h — C1 electrical bookkeeping

Question: Does the isotropic electrical reference include all boundary rows in the modulus and rectangle area?
Exact claim: For an \(n_x\times n_y\) unit-conductance vertex grid with left/right potentials 0/1, the effective conductance is \(n_y/(n_x-1)\), and the Hersonsky-style edge-rectangle area equals graph energy and right flux.
Assumptions: Top and bottom boundaries are natural; horizontal edge rectangles include the top and bottom vertex rows.
What would falsify it: An independent 2x2, 3x2, or 3x3 circuit calculation disagrees, or a boundary row is not part of the theorem's edge complex.
Smallest decisive test: Compare the closed-form parallel-path conductance against graph energy, right flux, and the reconstructed dual strip area.
Prior work: Phase II `electrical_rectangle.py`, Hersonsky mixed Dirichlet--Neumann rectangle theorem.

Finding: The old 9x7 value 0.75 omitted one top/bottom strip family; tests were written red first, then the reference was corrected to 0.875 for 9x7 and to \(n_y/(n_x-1)\) for 2x2, 3x2, and 3x3.

## T+1h — C5 learnable modulus latent

Question: Can the rectangle boundary latent make its modulus a genuine neural variable?
Exact claim: \(H=\varepsilon+\operatorname{softplus}(m)\) gives positive height, exact side closure, and a valid gradient with respect to \(m\).
Assumptions: Width and height are scalar tensors on the logits' device/dtype; finite logits and finite modulus logit are required.
What would falsify it: The top edge does not equal \(H\), closure drifts with \(m\), or the analytic/autograd gradient is zero or inconsistent.
Smallest decisive test: Two segments per side, direct top-right height check, implicit closure check, and finite-difference gradient check for \(m\).
Prior work: Phase II rectangle boundary helper and Phase III CQ11.

Finding: Added tensor-valued width/height support and
\(H=\varepsilon+\operatorname{softplus}(m)\); tests now verify top-edge height,
implicit closure, and finite nonzero gradient with respect to \(m\). Independent
checker still pending.

## T+1.5h — C2 decomposition degrees of freedom

Question: Is the fixed four-direction tensor decomposition unique?
Exact claim: For \(A=\left[\begin{smallmatrix}a&b\\b&c\end{smallmatrix}\right]\), the exact family has one free parameter \(t=c_++c_-\) and coefficients \(c_x=a-t\), \(c_y=c-t\), \(c_+=(t+b)/2\), \(c_-=(t-b)/2\).
Assumptions: Directions are \(e_x,e_y,e_+=(1,1),e_-=(1,-1)\); \(A\) is symmetric positive definite.
What would falsify it: A rank calculation gives four independent equations, or two different \(t\) values fail to reconstruct the same tensor.
Smallest decisive test: \(A=[[2,-0.5],[-0.5,3]]\), \(t=0.5\) and \(t=1\).
Prior work: Phase II anisotropic_hodge helper and its incorrect uniqueness wording.

Finding: The implementation now exposes the exact interval
\(|b|\le t\le\min(a,c)\); four regression tests pass. Independent checker still pending.

## T+2h — CQ1/CQ2 continuum MBM theorem

Question: Does the mixed anisotropic conductivity solve plus its stream function equal the normalized QC rectangle map, and what is the exact modulus identity?
Exact claim: Under a bounded simply connected Jordan quadrilateral, nontrivial boundary arcs, \(\mu\in L^\infty\) with \(\|\mu\|_\infty<1\), and the weak mixed problem, \(f=u+iv\) is the unique normalized QC homeomorphism onto a rectangle; \(M=E_A(u)=\int_{\Gamma_R}A\nabla u\cdot n\), and the normalized complementary coordinate \(w=(v-v_B)/M\) solves the same tensor \(A\) with reciprocal energy \(1/M\).
Assumptions: Boundary traces are interpreted weakly; piecewise \(C^1\) or Lipschitz boundary is used for flux traces, and stronger classical regularity is stated separately rather than silently assumed.
What would falsify it: The stream form is not globally exact, conformal postcomposition fails to preserve the conductivity equation/boundary data, or the complementary tensor is \(A^{-1}\) rather than \(A\).
Smallest decisive test: \(\mu=0\), constant complex \(\mu\) on an affine preimage quadrilateral, and a smooth variable coefficient with independent flux/energy checks.
Prior work: MRMT, quadrilateral conformal modulus mixed BVP, conductivity–Beltrami correspondence, Phase II MBM notes.

## T+3h — C3/C4/C6/C7 independent adjudication

Question: Which correction claims survive an adversarial theorem audit?
Exact claim: local nonobtuse is sufficient only; directed Tutte and mixed-boundary claims need explicit scope.
Assumptions: fixed facewise P1 conductivity, triangulated disk, positive directed rows.
What would falsify it: a checker finding a sign/algebra error or a universal discrete monotonicity proof.
Smallest decisive test: 120/30 degree edge sum, directed rectangle boundary hypotheses, and 3x3 anisotropic trace search.
Prior work: checker report tmp/phase3_correction_checker.md.

Finding: C3 passes as sufficient-only; C4/C6 are revised to conditional; C7 has a fixed-P1 counterexample but continuum monotonicity remains separate.

## T+4h — exact compatible P1 conjugacy

Question: Does an already compatible piecewise-affine map solve both mixed finite-element systems exactly?
Exact claim: facewise grad(v)=J A grad(u) plus continuity implies H(div) flux cancellation and exact primary/complementary weak equations.
Assumptions: conforming positively oriented triangulation, SPD determinant-one face tensors, side-compatible traces.
What would falsify it: nonzero face conjugacy residual or a nonzero assembled free-row residual on a layered manufactured map.
Smallest decisive test: u=x, v=g(y) with slopes 0.4 and 1.6 on a 4x2-cell mesh.
Prior work: CQ3/CQ4 and src/qcopt/forward/discrete_conjugacy.py.

Finding: exact residuals and positive face determinants pass; independent arbitrary-mu solves are still not proven compatible.

## T+5h — C8 monotonicity escalation

Question: Can positive anisotropic face tensors force ordered complementary side traces?
Exact claim: universal fixed-P1 monotonicity would survive an exhaustive 2x2-cell face-coefficient search.
Assumptions: eight face coefficients independently chosen from {-0.8i, 0, +0.8i}; bottom/top Dirichlet, left/right free.
What would falsify it: one negative left or right boundary increment with every face tensor SPD.
Smallest decisive test: enumerate all 3^8 assignments and solve the three middle-row unknowns independently.
Prior work: planar resistor response matrices, discrete maximum principle, Phase II random search.

Finding: 6561 cases contain 87 counterexamples; minimum left increment -0.2749922961. The continuum canonical-map question remains separate.

Follow-up: the same 6561-case enumeration with scalar conductances {1,2,10} on the right-triangle stencil had no negative increment (minimum 0.0909090909), so the positive result is restricted to that M-matrix graph family.

## T+7h — independent code audit and repair

Question: Are the new solver primitives actually enforcing their mathematical preconditions?
Exact claim: SPD/unit-determinant tensors, simple convex boundaries, empty-interior meshes, and device-consistent VJPs are handled explicitly.
Assumptions: finite arrays, conforming meshes, same-device Torch inputs.
What would falsify it: negative-definite acceptance, star-boundary acceptance, empty reduction, or a device mismatch reaching NumPy.
Smallest decisive test: -I and det != 1 faces, regular pentagram, 1x1-cell boundary-only mesh, meta-device logits.
Prior work: `tmp/phase3_code_checker.md`.

Finding: all four issues were reproduced and fixed; focused tests now pass. Electrical target-area field was removed because it was circular; CSV determinant units/provenance were made explicit.

## T+8h — anisotropic route escalation beyond the fixed stencil

Question: Did the negative four-direction audit disprove anisotropic primal-dual methods in general?
Exact claim: eigenvector directions, A-metric Delaunay meshes, or block Hodge stars can represent SPD tensors, but each changes the global discrete complex.
Assumptions: facewise SPD tensors; conforming primal/dual incidence is the limiting constraint.
What would falsify it: an algebraic obstruction for every rotated representation, or a fixed graph that realizes all principal directions without new edges.
Smallest decisive test: spectral rank-one decomposition and comparison with the four-direction positive-feasibility interval.
Prior work: C2/C3/C6 and anisotropic Delaunay literature.

Next: have the independent code checker audit the scope, then keep this as a route redesign question rather than declaring the whole anisotropic route false.

## T+9h — supporting Beurling periodicity audit

Question: Does FFT acceleration of the whole-plane Beurling transform silently impose a harmful periodicity assumption?
Exact claim: periodic FFT computes a torus convolution; zero-padding only approximates free-space by separating periodic images.
Assumptions: uniform 128x128 grid, smooth coefficient supported in a central disk, padding factors 2/4/8.
What would falsify it: periodic and converged zero-padded operators agreeing to discretization error at both interior and boundary regions.
Smallest decisive test: relative periodic-vs-padding difference on central and outer masks, plus padding convergence.
Prior work: Daripa/Gaidashev route and periodic/zero-padded Beurling modules.

Finding: periodic differs 5.19% centrally and 27.5% near the outer annulus; padding 4/8 converges centrally. Uniform FFT is therefore not a general-mesh or exact bounded-domain solver.

## T+10h — MBM implicit VJP benchmark

Question: Does the MBM reference actually have a usable differentiable backward path at realistic resolution?
Exact claim: the CPU implicit prototype can backpropagate through two sparse P1 solves, but this does not establish exact discrete conjugacy or topology.
Assumptions: smooth coefficient (0.25e^{-r^2/.18}e^{0.7i}), 65/129/257 structured vertices, one CPU process.
What would falsify it: nonfinite gradients, finite-difference disagreement, or unbounded memory at the tested sizes.
Smallest decisive test: directional FD on 7x7 plus RSS/timing at all three resolutions.
Prior work: CQ1-CQ4, `mbm_lbs_torch_implicit`, `phase3_mbm_vjp_benchmark.py`, independent `solve_mbm_lbs` residual evaluator.

Finding: VJP is finite and FD-consistent (relative error 9.37e-10), but isolated 257x257 costs 14.128 s forward + 12.125 s backward and 247.25 MB RSS; conjugacy residual remains 0.02352. This upgrades MBM from “no VJP prototype” to “differentiable CPU reference, not production candidate.”

## T+11h — batching and accelerator boundary audit

Question: Are the two candidates already usable as batched GPU neural layers?
Exact claim: no; both public prototypes are unbatched CPU/SciPy layers with host-device copies in the VJP path.
Assumptions: current public APIs, local Windows environment, read-only remote GPU availability probe.
What would falsify it: accepted leading batch dimensions and a measured GPU solve without host copies.
Smallest decisive test: pass a `(batch,ny,nx)` MBM tensor and a batched boundary to Tutte; inspect CUDA availability.
Prior work: CQ11-CQ12 implementation audit.

Finding: both reject batch dimensions; local CUDA is unavailable, and the remote GPUs were occupied. GPU and batched-solve work remain open engineering tasks.

## T+13h — unstructured Tutte stress

Question: Does the directed layer retain finite VJPs and fold-free output on a realistic nonuniform mesh?
Exact claim: finite stress evidence may support the conditional theorem, but cannot replace a graph/topology certificate or a uniform conditioning bound.
Assumptions: Delaunay mesh with 4,056 vertices/7,854 faces, 256 boundary vertices, fixed affine boundary, logit spreads 1 and 3.
What would falsify it: flipped faces, boundary intersections, or nonfinite VJP.
Smallest decisive test: independent injectivity audit after both forward/backward passes.
Prior work: CQ9-CQ12 and `tutte_directed_unstructured_audit.py`.

Finding: both runs had zero flips/intersections and finite gradients, but the minimum signed-area ratio dropped to (5.547e{-11}) at spread 3. This is a conditioning warning and confirms the need for a quantitative margin certificate.

## T+12h — mandatory midpoint review

This is a checkpoint, not a final decision.

1. **Theorem overclaim audit.** The continuum MBM result is now stated for a
   bounded simply connected Jordan quadrilateral with an open rectangular image
   and a separate closure homeomorphism. Its proof uses the measurable Riemann
   mapping theorem, conformal rectangle uniformization, conductivity--Beltrami
   correspondence, and mixed-problem uniqueness. It is not a theorem about a
   fixed P1 mesh. The exact P1 result is conditional on an already compatible
   piecewise-affine homeomorphism. Directed Tutte is conditional on a simple
   weakly convex boundary, positive weights, and a boundary-reachable
   3-connected graph; the implementation checks only the boundary part. No
   document may call either decoder universal for arbitrary μ.

2. **Numerical self-consistency audit.** The electrical rectangle tests now use
   independent graph energy, boundary flux, and dual tiling-area paths, including
   2x2, 3x2, 3x3, 9x7, and a nonuniform grid. The MBM smooth experiment reports
   the raw right-flux mismatch (2.0163e-3 at 129x129) instead of treating the
   same stiffness quadratic form as an independent flux certificate. The new
   MBM VJP benchmark uses a separate directional finite-difference test and a
   separate P1 residual evaluator. The full regression run after the code audit
   is `312 passed, 1 warning` in 126.49 s; the warning is an external Paramiko
   deprecation.

3. **Negative-route audit.** The anisotropic counterexample falsifies only a
   fixed facewise P1 complementary solve and the fixed four-direction positive
   stencil. It does not falsify adaptive rotated directions, anisotropic
   Delaunay meshes, or a fully specified block Hodge/de Rham construction. The
   Beurling experiment establishes a periodized-FFT versus free-space boundary
   discrepancy, not impossibility of every NUFFT/treecode method. BHF remains a
   frozen supporting branch under the authoritative plan rather than silently
   being declared mathematically impossible.

4. **Open question not bypassed.** Continuum canonical-map monotonicity of the
   free-side trace is still unresolved; the fixed-P1 3x3 negative increment is
   explicitly not used as a continuum counterexample. Also open are a scalable
   structure-preserving primal--dual complex, an enforced graph certificate for
   Tutte, arbitrary-μ exactness, batched/GPU solves, and peak allocator memory.

5. **Remaining budget.** The next block should use an independent checker to
   adjudicate the continuum monotonicity statement and the new MBM benchmark,
   then stress the strongest surviving candidates at the existing realistic
   resolutions. No final ranking or `08_final_decision.md` is permitted before
   the T+22h review window.

## T+6h — C10/C12 expressivity and medium benchmark

Question: Does the directed Tutte layer remain accurate, fold-free, and differentiable at realistic resolution?
Exact claim: a valid target can be fitted while the implicit VJP remains finite and sparse.
Assumptions: structured 16x16-cell target for expressivity; 129x129 and 257x257 vertices for same-resolution timing.
What would falsify it: finite-det/gradient failure or an MBM VJP that is already implemented and comparable.
Smallest decisive test: sine-bump targets alpha 0.05/0.15/0.30, then 129/257 sparse forward-backward runs.
Prior work: CQ9-CQ12, existing directed implicit solver and MBM-LBS reference.

Finding: Tutte fits but mu error and determinant margin worsen with alpha; 257x257 Tutte takes 4.084 s forward + 1.801 s backward, while MBM takes 17.257 s forward and has no VJP.

## T+14h — Wide-stencil anisotropic stress

Question: Does enlarging the positive edge-direction stencil simultaneously
improve anisotropic tensor coverage and preserve a hard embedding certificate?

Exact claim: one-ring and two-ring positive-conductance decoders should be
evaluated on a realistic unstructured mesh, with fit residual, solve cost, and
original-face signed determinants reported independently.

Assumptions: 12,384-vertex Delaunay mesh, 24,382 original faces, 384 boundary
vertices, seed 20260919, edge-weight floor (10^{-4}). The determinant is the
raw signed double area on the original mesh; it is not a normalized Jacobian.

What would falsify it: a wider stencil that improves tensor fit while retaining
positive signed area on every original face would support the route; any flips
or a near-singular margin would refute the naive universal claim.

Finding: one-ring support (36,765 decoder edges) had residual mean/p95
0.129011/0.795089, total fit-plus-solve time 21.83114 s, zero flips, but minimum
determinant (2.4053\times10^{-12}). Two-ring support (120,275 edges) reduced
residual mean/p95 to 0.00290868/(7.44\times10^{-12}) but took 34.98491 s and
flipped 1,641 original faces, with minimum determinant
(-4.9312\times10^{-4}). The result is numerical counterevidence to the naive
wide-stencil decoder, not a theorem against adaptive planar constructions. The
checked-in artifact reports Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1,
`OMP_NUM_THREADS=1`, and the local Intel i7-4770 CPU. Residuals include boundary
vertices with at least three supports and are absolute pre-clipping coefficient
norms, not final-network residuals.

## T+15h — Larger same-machine neural-layer stress

Question: Do the surviving directed-Tutte and MBM CPU prototypes remain finite
at 513-by-513 vertices, and how do time/memory scale beyond 257-by-257?

Exact claim: isolated-process measurements at the next realistic resolution
can expose a practical memory or solve-time wall even when smaller runs pass.

Assumptions: same smooth coefficient family, zero/affine convex boundary,
float64 CPU execution, one thread, separate fresh process per measurement.

What would falsify it: non-finite output/gradient, topology failure, or a
resource wall before the requested resolution.

Prior work: CQ12 129/257 rows and the independent checker audit. Next step is
to run both prototypes and record only independently observable quantities.

## T+16h — 513-vertex-axis stress result

Finding: MBM at 513x513 vertices was finite with 70.961 s forward, 46.875 s
backward, +1,290.8 MB process RSS, normalized minimum determinant 0.50538, and
conjugacy residual 0.02355. Directed Tutte at 512x512 cells (263,169 vertices)
was finite and flip-free at logit spreads 1 and 3, taking 117.990/118.219 s for
combined forward-plus-backward passes; its minimum signed-area ratio dropped to
2.5495e-12 at spread 3. This is a conditional numerical pass with a clear
memory/conditioning warning, not a production-layer conclusion.

## T+16.5h — Regression environment audit

The first full-suite invocation aborted in `tests/test_beltrami.py` with Intel
OpenMP error #15 because Anaconda MKL and PyTorch loaded duplicate OpenMP
runtimes. This reproduced in the focused test and disappeared when the
documented `MKL_THREADING_LAYER=SEQUENTIAL` and `OMP_NUM_THREADS=1` variables
were set before Python startup. With that explicit environment the complete
suite passed: 312 tests, one unrelated Paramiko Blowfish deprecation warning,
129.15 s. No source fix was needed.
