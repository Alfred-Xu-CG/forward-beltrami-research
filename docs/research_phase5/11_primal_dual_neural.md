# Route III: structure-preserving primal–dual and positive-Hodge neural layers

## Abstract

This document gives the self-contained mathematical specification, implementation map, and experimental result of Phase V Route III. The route asked whether Beltrami data can be converted into a fast differentiable deformation without a generic backward linear Beltrami solve, while guaranteeing that the returned discrete map is a piecewise-affine homeomorphism.

The investigation produced two distinct objects.

1. A full anisotropic Whitney finite-element reference exactly represents the ordinary P1 anisotropic energy. It is a useful differentiable teacher and a compatible conservation reference, but it is not a new solver and has no automatic homeomorphism theorem.
2. A positive-conductance student maps a learnable face latent to a symmetric Tutte system. Strictly positive conductances and a convex boundary give the implemented graph family a hard discrete topology guarantee. The price is approximation: a fixed finite direction cone cannot reproduce all Beltrami tensors exactly.

The implemented student is therefore a genuine first-order differentiable hard-bijective P1 layer for the stated planar graph and boundary hypotheses. It is not an exact arbitrary prescribed-Beltrami solver, not a trained convolutional network, and not yet evidence of million-vertex training scalability.

## 1. Discrete domain and the meaning of the output

Let K = (V,E,F) be an oriented triangulation of a topological disk. Each reference vertex i has position p_i in the plane. A discrete deformation is specified by mapped vertex positions y_i in the plane and extended affinely on each triangle T = (i,j,k).

On T, define the reference and mapped edge matrices

    P_T = [p_j - p_i, p_k - p_i],
    Y_T = [y_j - y_i, y_k - y_i].

When det(P_T) is nonzero, the constant P1 Jacobian on T is

    J_T = Y_T P_T^{-1}.

The map is locally orientation preserving on T exactly when det(J_T) > 0. Local positivity on every face is necessary but is not, by itself, a general global injectivity theorem. In this work global topology comes from the positive symmetric Tutte construction and its hypotheses; face-area checks are an independent numerical audit.

A piecewise-affine homeomorphism means that the continuous P1 extension is one-to-one, onto its image, continuous, has a continuous inverse, and is affine on every input triangle. It is stronger than a fold count of zero and stronger than a small linear-solver residual.

## 2. Beltrami coefficient and conductivity tensor

Write one affine map as f = u + i v, with

    f_z     = 0.5 (u_x + v_y) + 0.5 i (v_x - u_y),
    f_bar_z = 0.5 (u_x - v_y) + 0.5 i (v_x + u_y).

If f_z is nonzero, its Beltrami coefficient is

    μ = f_bar_z / f_z.

The condition |μ| < 1 is equivalent to an orientation-preserving nonsingular affine Jacobian. For a facewise coefficient μ = a + i b with a²+b² < 1, the implementation uses the determinant-one symmetric positive-definite tensor

                  1
    A(μ) = ----------------- [
             1 - a² - b²

             1 - 2a + a² + b²      -2b
                   -2b          1 + 2a + a² + b² ].

Its determinant is one. Its eigenvalue ratio grows as |μ| approaches one, so large Beltrami modulus is an anisotropic and increasingly ill-conditioned regime.

The latent variable in the optimized student is a two-vector w_T per face. It is mapped smoothly into a disk of radius ρ = 0.95 by

    r_T = ||w_T||,
    μ_T = ρ tanh(r_T) w_T / r_T,

with the continuous limiting value μ_T = 0 at r_T = 0. This cap makes every produced A(μ_T) positive definite, but it is not the topology proof.

## 3. Full anisotropic Whitney reference

### 3.1 Incidence and energy

Choose an orientation for every mesh edge and let B0 be the vertex-to-edge incidence matrix. For a scalar vertex vector q, the edge differences are B0 q.

On one triangle T, the gradients of the three barycentric P1 basis functions are constant. If G_T contains those gradients, the ordinary anisotropic P1 stiffness contribution is

    L_T = |T| G_T^T A_T G_T.

The assembled scalar operator is the sum of these contributions. The Whitney one-form construction stores the same bilinear form on oriented edge degrees of freedom in a matrix H_A. On exact discrete gradients the global operator is

    L_A = B0^T H_A B0.

For every vertex vector q,

    q^T L_A q
      = (B0 q)^T H_A (B0 q)
      = Σ_T |T| (grad q_T)^T A_T (grad q_T).

Thus the implemented Whitney primal operator is algebraically the ordinary anisotropic P1 operator. This exact identity is a consistency result, not evidence that Whitney is more accurate than P1 for the same mesh and tensor.

### 3.2 Dirichlet solve and differentiability

Partition vertices into interior I and boundary B. After imposing boundary values g, each coordinate solves

    L_II x_I = -L_IB g.

For uniformly positive-definite face tensors on a connected disk whose interior is connected to the boundary, L_II is symmetric positive definite. The reference implementation uses a CPU float64 sparse solve.

If a scalar loss ℒ depends on x_I, the adjoint λ satisfies

    L_II^T λ = ∂ℒ/∂x_I.

Since L_II is symmetric, the same operator is reused. Tensor and boundary vector–Jacobian products follow by differentiating the bilinear form. A directional finite-difference check exercises the complete tensor-to-solve-to-dense-map path rather than only the sparse solve.

### 3.3 Flux and dual-stream scope

For a solved scalar field u, the physical face flux is q_T = A_T grad u_T. Interior weak balance follows from the finite-element equations. A rotated flux can be locally integrated to a stream function on a simply connected disk when the corresponding discrete one-form is exact.

Three distinctions are essential.

- Weak vertex balance is not pointwise normal-flux continuity.
- The reconstructed physical flux is not automatically an RT0 mixed finite-element flux.
- On an annulus, a closed one-form can have nonzero period and therefore fail to be globally exact.

The disk stream diagnostic is useful for sign and compatibility checks, but neither conservation nor conjugacy proves that the two-coordinate map is injective.

### 3.4 Reference result

At N = 129 the clean reference contains 16,641 vertices, 32,768 faces, and 49,408 edges. The relative difference between the Whitney and direct P1 operators was 4.33e-16. Interior balance was 1.24e-14, the dual residual was 5.61e-10, and the stated conjugacy L2 diagnostic was 0.024. Forward and backward times were 0.128 and 0.016 seconds on the recorded CPU run; explicitly listed arrays occupied 29.7 MB.

The common-input P-ref experiment derives exact face μ from the target P1 map and imposes the target boundary. It recovers the map target to 2.22e-15 maximum coordinate error and the image target to 2.55e-15. These are target-derived teacher results. They do not show recovery from a generic learned latent or a general topology guarantee.

## 4. Why finitely many positive directions cannot be exact

For unit directions d_m in the plane and nonnegative scalar coefficients c_m, a local positive-direction tensor has the form

    A_hat(c) = Σ_m c_m d_m d_m^T,     c_m ≥ 0.

The cone generated by finitely many matrices d_m d_m^T is polyhedral. In contrast, the cone of two-dimensional positive-semidefinite matrices has a continuum of extreme rays {d d^T : d on the projective circle}. A finite polyhedral cone therefore cannot equal the full positive-semidefinite cone. Its interior cannot contain every positive-definite orientation with arbitrary anisotropy.

This theorem applies to fixed scalar directions and fixed connectivity. It does not rule out adaptive directions, changing connectivity, block-valued Hodge stars, or a full anisotropic finite-element tensor.

For the explicit center-split and stellar direction sets, the worst-orientation uniform exact positive radius in Beltrami modulus is

    r_star = sqrt(2) - 1.

The standard southwest-to-northeast triangulation has zero uniform radius because its missing projective direction makes some infinitesimal signed off-diagonal perturbations leave the positive cone. These are local tensor-cone statements, not global operator or map statements.

## 5. Canonical positive tensor projection

### 5.1 Frobenius-isometric vectorization

For a symmetric matrix A, define

    s(A) = (A_11, sqrt(2) A_12, A_22).

Then ||s(A)-s(B)||_2 equals the Frobenius norm ||A-B||_F. The design column for direction d_m is s(d_m d_m^T). Stacking columns gives a 3 by M matrix D.

With strict floor ℓ > 0, the primary projection is

    minimize over c ≥ ℓ  ||D c - s(A)||_2².

Redundant directions can make c nonunique even when the projected tensor D c is unique. The formal implementation therefore defines a lexicographic problem:

1. minimize the tensor residual;
2. among all primary minimizers, minimize ||c||_2.

### 5.2 Exact small-dictionary algorithm

Write c = ℓ 1 + x with x ≥ 0. All active supports are enumerated; each support is solved by least squares and screened by primal feasibility and nonnegative-least-squares Karush–Kuhn–Tucker conditions.

The minimum-actual-conductance interpretation uses two checked facts. First, ℓ is uniform. Second, every unit-direction column satisfies

    D_1m + D_3m = 1.

Hence D z = 0 implies Σ_m z_m = 0, so every null-space displacement is orthogonal to the uniform floor. Minimum ||x|| and minimum ||ℓ1+x|| therefore select the same support solution. The code rejects a design without this trace identity.

Columns are sorted by their numerical content before enumeration and restored afterward. A determinant-one counterexample near a support transition is tested over all 24 permutations of the four center-split directions. This test was necessary: the first implementation accepted a 1e-7 order-dependent secondary error even though tensor residual and primary KKT conditions were correct.

The unique secondary minimizer is the exact-arithmetic specification. The implemented floating-point solver uses scale-aware primary and secondary tolerances. Content sorting makes its result permutation equivariant for the fixed distinct declared columns, but coefficient accuracy is not uniform arbitrarily close to a support transition. An independent determinant-one probe with a 1e-8 transition perturbation remained permutation invariant but selected the floor support and differed from the analytic minimum-norm coefficients by about 1e-8. The projected tensor and primary optimum were still correct. Consequently, the numerical claim is deterministic, KKT-screened projection with a quantified near-transition secondary limitation, not exact recovery of the unique secondary coefficient vector at every floating input.

The exhaustive algorithm is an offline reference for at most six declared directions, not the fast neural path.

## 6. Learned strictly positive local projector

The deployed student uses a small deterministic multilayer perceptron rather than running NNLS inside each neural forward pass. Its tensor features are

    h(A) = ( log tr(A),
             (A_11-A_22)/tr(A),
             2 A_12/tr(A) ).

Two hidden SiLU layers map h(A) to M logits. Conductances are

    c_m(A) = ε + softplus(logit_m(A)),

with ε = 1e-6. They are therefore finite and strictly positive for every finite network output.

The projector is trained once on radii 0, 0.2, 0.4, 0.6, 0.8, and 0.9 and 180 angles, for 1500 steps with fixed seed 20260922. The state is frozen during instance optimization and is identical for the map and image tasks. This is a target-independent deterministic preprocessing model, not a convolutional network trained end to end.

## 7. From face predictions to one global planar graph

Each graph edge has a declared geometric direction class. A boundary edge or one-face edge receives the incident face prediction. A shared interior edge receives the arithmetic mean of its incident face predictions:

    c_e = (1 / number of incident faces) Σ_{T incident to e} c_{T,k(e,T)}.

An average of strictly positive values remains strictly positive. This aggregation is a modeling choice; it does not preserve an independently fitted tensor face by face after global sharing.

Three planar disk families were audited.

| graph | projective directions in degrees | V/E/F at two cells per side | dividing edges |
|---|---|---:|---:|
| standard southwest–northeast | 0, 45, 90 | 9 / 16 / 8 | 2 |
| center split | 0, 45, 90, 135 | 13 / 28 / 16 | 0 |
| stellar triangle-centroid | 0, atan(1/2), 45, atan(2), 90, 135 | 17 / 40 / 24 | 2 |

The audit checks disk Euler characteristic, one boundary loop, incidence consistency, absence of nonincident crossings or collinear overlaps, direction reconstruction, and the dividing-edge hypothesis required by the selected Tutte–Floater statement.

## 8. Positive symmetric Tutte layer

For mapped vertices y and undirected positive edge conductances c_e, define

    E_c(y) = 0.5 Σ_{e=(i,j)} c_e ||y_i-y_j||².

Every interior vertex satisfies

    Σ_{j adjacent to i} c_ij (y_i-y_j) = 0,

or equivalently

    y_i = Σ_j α_ij y_j,
    α_ij = c_ij / Σ_k c_ik > 0,
    Σ_j α_ij = 1.

Thus each interior point is a strict convex combination of its neighbors. For the verified planar disk graph, a boundary mapped in cyclic order to a convex rectangle, positive symmetric weights, and the applicable dividing-edge/nondegeneracy hypotheses, the straight-line solution is an embedding. Its P1 extension is the required discrete homeomorphism.

This is the topology mechanism. Neither |μ| < 1, closeness to the teacher tensor, nor conjugate-gradient convergence is substituted for it. In floating point the implementation also audits all face orientations, boundary order, and an independent CPU-float64 direct reassembly of the final state.

## 9. Implicit backward pass

Let B be the oriented incidence restricted to active edges and let C = diag(c). The full graph Laplacian is

    L(c) = B^T C B.

After Dirichlet elimination, the interior coordinate solve is A(c)x = b(c,g). For fixed boundary and a scalar loss ℒ, solve

    A(c)^T λ = ∂ℒ/∂x.

For one edge e, with full adjoint equal to zero on the boundary,

    ∂ℒ/∂c_e = - (B λ)_e dot (B y)_e.

The dot product sums the two output coordinates. Automatic differentiation then propagates this edge gradient through face averaging, softplus network, A(μ), and the radial map from w to μ.

This backward is implicit: it stores the accepted state and solves one adjoint system rather than retaining every conjugate-gradient iterate. The asymptotic state is linear in graph and dense-query size, not in Krylov iteration count. First-order differentiation is implemented and finite-difference tested. Higher-order derivatives are not established.

## 10. Complete neural-layer chain

For each forward call the implemented map is

    face latent w
      -> bounded face μ
      -> determinant-one SPD A(μ)
      -> frozen learned positive direction conductances
      -> positive shared-edge averaging
      -> symmetric positive Tutte solve
      -> P1 control homeomorphism
      -> dense P1 sampling
      -> map or backward-image loss.

The returned deformation is the control P1 map, not a post-hoc fold-repaired field. If the linear solver cannot meet its residual contract or an audit fails, the run fails closed.

The formal method name P1_positive_uniform refers specifically to initialization with w_T = 0, and hence μ_T = 0, on every face. The frozen multilayer projector is then evaluated on the same identity tensor at every face before face predictions are assigned to direction classes and averaged onto graph edges. On the standard graph the recorded identity-tensor output is approximately (0.99192115, 0.02012113, 0.99088946), so it is visibly direction dependent. The initialization is spatially uniform and target independent in tensor latent space. It does not assert that all direction coefficients or all assembled edge conductances are numerically equal, and it does not assume that the initial decoded interior is the identity map.

## 11. Metrics

For n control vertices and target y star, map root-mean-square error is

    map_RMSE = sqrt( (1/n) Σ_i ||y_i-y_i star||² ).

Maximum map error is max_i ||y_i-y_i star||. For N_F faces,

    μ_RMSE = sqrt( (1/N_F) Σ_T |μ_T(y)-μ_T(y star)|² ).

Maximum μ error is the corresponding maximum absolute complex difference.

Image mean-square error is the pixel average of the squared difference between the fixed image and the moving image sampled at the predicted backward coordinates.

The area ratio on face T is mapped signed area divided by the positive reference area. The minimum area ratio is the minimum over faces. A topology certificate additionally requires zero flips, positive minimum signed area, correct boundary cyclic order, and the graph-theoretic theorem hypotheses.

Local tensor error is ||A_hat-A||_F / ||A||_F. Local operator-symbol error is a spectral-norm comparison of the two 2 by 2 symbols. Assembled-operator error compares the interior rows after one best positive global scale. These quantities are deliberately separate from decoded map and Beltrami errors.

## 12. Coverage experiment

The fixed evaluation uses six radii, 180 held-out half-step angles per radius, three graph families, and both canonical NNLS and learned positive projectors. The formal receipt was generated on a clean Linux checkout at f0fa66d on AI A6000 GPU 2. A Windows CPU replication agreed over 486 deterministic scalar summaries to maximum absolute difference 4.56e-15.

At |μ| = 0.9:

| graph and projector | max tensor error | max assembled operator error | max map RMSE | max μ RMSE |
|---|---:|---:|---:|---:|
| standard, canonical | 0.70504 | 0.23752 | 0.007848 | 0.032910 |
| standard, learned | 0.70508 | 0.23587 | 0.007996 | 0.033469 |
| center, canonical | 0.16637 | 0.83934 | 0.005199 | 0.029078 |
| center, learned | 0.16662 | 0.81300 | 0.005403 | 0.028119 |
| stellar, canonical | 0.16637 | 0.89531 | 0.14103 | 0.88048 |
| stellar, learned | 0.16774 | 0.81719 | 0.14022 | 0.87809 |

All sampled student maps were topology certified. The stellar result is decisive negative evidence: a richer local direction set and low local tensor error can coexist with a poor globally assembled map. Local representability is not a sufficient surrogate for global deformation accuracy.

## 13. Common-input instance results

All optimized rows below use the same N25 control target, 256 by 256 dense grid, seed 20260922, strength 0.25, float64 A6000 class hardware, and the exact observation at 83 completed global solves. P1 uses a separately preregistered learning rate selected on seed 20260921; T1 and T2 come from the shared-rate robustness cohort. Therefore input and solve-budget comparisons are valid, while optimizer-hyperparameter superiority is not proved.

| task | method | objective at solve 83 | map RMSE | μ RMSE | wall to observation |
|---|---|---:|---:|---:|---:|
| map | T1/O1 positive Tutte | 7.0055e-5 | 0.012081 | 0.144374 | 28.00 s |
| map | T2/O3 MVC Adam | 6.7623e-5 | 0.011887 | 0.144522 | 30.60 s |
| map | P1 positive Hodge | 7.6587e-6 | 0.004605 | 0.099185 | 17.72 s |
| image | T1/O1 positive Tutte | 1.8133e-3 | 0.016079 | 0.151093 | 30.47 s |
| image | T2/O3 MVC Adam | 1.7501e-3 | 0.017221 | 0.152211 | 36.68 s |
| image | P1 positive Hodge | 6.2592e-5 | 0.015286 | 0.158852 | 17.74 s |

Every listed row is topology certified. The P1 map task improves both map and Beltrami error. The image task greatly reduces the image objective but slightly worsens Beltrami RMSE relative to T1. This confirms that image agreement is not geometric fidelity.

P1 exact83 cumulative solver/backward times were 11.108/5.468 seconds for map and 10.836/5.770 seconds for image. Whole-run CUDA allocated peaks were 37.83 and 38.88 MB. Process high-water values, 1.295 and 1.327 GB, include runtime and independent dense direct audits and are not interchangeable with CUDA allocator peaks.

The P-ref rows use one forward and one adjoint on a single-thread CPU and exactly recover this target. They are not ranked against the optimized GPU methods because the teacher is target derived, the solve budget differs, and it has no general topology guarantee.

## 14. Scaling and current answer

The P-ref dense-query test was also run at 512 by 512. Its differentiable path took 0.196 and 0.192 seconds for the two tasks; explicitly listed state was 27.31 MB and process HWM about 598 and 593 MB. This tests dense interpolation and teacher differentiation at the preferred image resolution, but not 512-square P1 optimization.

For a mesh with one million vertices, a two-component face latent alone is on the order of four million scalars. In float32 this is roughly 16 MB and in float64 roughly 32 MB, before optimizer state, conductances, graph buffers, dense queries, and solver workspaces. The implicit adjoint avoids memory proportional to hundreds of solver iterations, which is the correct architecture for scale. However no million-vertex end-to-end GPU training receipt exists in this phase, so practical throughput and memory headroom at that scale remain open.

The precise answer to the project question is:

- Yes: the repository contains a differentiable first-order layer whose output is a P1 homeomorphism for the verified fixed planar graph, positive symmetric conductances, convex cyclic boundary, successful solve, and stated Tutte–Floater hypotheses.
- Yes: the backward pass propagates through the complete latent-to-map-to-loss chain without unrolling the iterative solver, and independent finite differences test the edge and full-chain gradients.
- No: it is not an exact solver for an arbitrary prescribed Beltrami field. A finite fixed positive direction cone makes that impossible, and the measured approximation frontier is nonzero.
- No: the full Whitney reference is not itself hard bijective; its observed target topology cannot be generalized.
- Not yet: there is no trained convolutional network, batched arbitrary-mesh implementation, higher-order derivative guarantee, or million-vertex end-to-end experiment.

## 15. Reproducibility map

The core implementation is in src/qcopt/neural_bijection/tutte/positive_hodge.py. The coverage and instance programs are experiments/phase5/route3_positive_hodge_coverage.py and route3_positive_hodge_instance_benchmark.py. Focused tests are tests/test_phase5_positive_hodge.py. The exact common table is rebuilt by experiments/phase5/cross_route_results.py and tested by tests/test_phase5_cross_route_results.py.

Formal machine-readable evidence is:

- route3_positive_hodge_coverage_gpu2_f0fa66d.json;
- route3_p1_formal_map_gpu3_35a1878.json;
- route3_p1_formal_image_gpu3_35a1878.json;
- the two explicitly rank-ineligible supplement receipts;
- route3_pref_common_map_cpu_0da8ffa.json and image counterpart;
- 13_cross_route_results.csv.

Superseded noncanonical coverage files are retained only in tmp as negative process evidence and are not formal authorities.
