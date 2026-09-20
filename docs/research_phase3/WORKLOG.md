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

## T+6h — C10/C12 expressivity and medium benchmark

Question: Does the directed Tutte layer remain accurate, fold-free, and differentiable at realistic resolution?
Exact claim: a valid target can be fitted while the implicit VJP remains finite and sparse.
Assumptions: structured 16x16-cell target for expressivity; 129x129 and 257x257 vertices for same-resolution timing.
What would falsify it: finite-det/gradient failure or an MBM VJP that is already implemented and comparable.
Smallest decisive test: sine-bump targets alpha 0.05/0.15/0.30, then 129/257 sparse forward-backward runs.
Prior work: CQ9-CQ12, existing directed implicit solver and MBM-LBS reference.

Finding: Tutte fits but mu error and determinant margin worsen with alpha; 257x257 Tutte takes 4.084 s forward + 1.801 s backward, while MBM takes 17.257 s forward and has no VJP.
