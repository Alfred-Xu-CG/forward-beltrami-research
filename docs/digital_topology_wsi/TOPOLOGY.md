# Q1 topology and F1-D / F2-D derivations

Geometry Builder draft, 2026-09-29. This document supplies mathematical derivations and legacy-code deltas; it is **not independent approval**, an IEEE arithmetic proof, or a G1–G4 verdict. Statements marked PROVED below are exact-arithmetic derivations subject to the independent geometry review. No production code was modified for this document.

## 1. Representation, units, and hypotheses

Let the source be the closed rectangle with tensor-product axes `x[0] < ... < x[n]`, `y[0] < ... < y[m]`, and no hanging nodes. Coordinates are mathematical `(x,y)`; arrays have layout `[batch,row,column,component]`, with increasing row corresponding to increasing mathematical y. A scanner's downward image axis or reflected orientation must be explicitly converted. The deployed backward sampling map is the **full map** `Phi`, not just its displacement: `I_warped(x) = I_moving(Phi(x))`.

One shared array defines every cell and every common edge. For a source cell of widths `hx,hy > 0`, denote its mapped SW, SE, NE, NW vertices by `a,b,c,d`. Source-local coordinates `s,t` range over `[0,1]`. Q1 means exactly

\[
 F(s,t)=(1-s)(1-t)a+s(1-t)b+stc+(1-s)td
       =a+se+tf+stg,
\]

where `e=b-a`, `f=d-a`, `g=a-b+c-d`. The determinant in physical source coordinates is the local-coordinate determinant divided by `hx*hy`. All determinants below are signed **double triangle areas**, not triangle areas; an absolute floor has these same units.

The global theorem additionally assumes that the boundary's piecewise-linear map is an orientation-preserving bijection onto the entire boundary of a nondegenerate convex polygon `K`. Boundary subdivision points may be collinear, but consecutive points cannot coincide or backtrack, and nonadjacent boundary edges cannot intersect. Pointwise identity boundary on a rectangle satisfies this hypothesis. Fixing only four corners does not establish it.

## 2. The four-corner theorem — PROVED in exact arithmetic

Write `[u,v]=u_x v_y-u_y v_x` and `T(u,v,w)=[v-u,w-u]`. The four quantities are

\[
\begin{aligned}
q_{00}&=[b-a,d-a]=T(a,b,d),\\
q_{10}&=[b-a,c-b]=T(a,b,c),\\
q_{11}&=[c-d,c-b]=T(b,c,d),\\
q_{01}&=[c-d,d-a]=T(a,c,d).
\end{aligned}
\]

**Claim.** All four `q > 0` are equivalent to `det D_(s,t) F > 0` on the closed square. They imply that `F` is a homeomorphism of the closed square onto the closed strictly convex quadrilateral with cyclic vertices `a,b,c,d`. Both diagonal P1 interpolants are also homeomorphisms onto this same quadrilateral. These are different functions inside the quadrilateral in general.

**Determinant proof.** Since `F_s=e+tg`, `F_t=f+sg`, bilinearity and `[g,g]=0` give

\[
J(s,t)=[e,f]+s[e,g]+t[g,f].
\]

Thus J is affine (not quadratic) in `(s,t)`. Its corner values are the four q's, and

\[
J(s,t)=(1-s)(1-t)q_{00}+s(1-t)q_{10}+stq_{11}+(1-s)tq_{01},
\qquad q_{00}+q_{11}=q_{10}+q_{01}.
\]

The coefficients are nonnegative and sum to one, proving the claimed equivalence and `min J=min q`. Four values remain necessary to check a minimum; the affine relation does not allow dropping an arbitrary corner.

**Convexity proof.** Each cyclic edge has both remaining vertices strictly to its left: for edge `a -> b` these tests are `q00,q10`; for `b -> c` they are `q10,q11`; for `c -> d` they are `q11,q01`; for `d -> a` they are `q01,q00`. These four supporting-edge conditions define a simple strictly convex CCW quadrilateral.

**Injectivity proof without a sampled-Jacobian argument.** For any `p=(s,t)` and `p'=(s',t')`, direct expansion of the bilinear term gives the exact identity

\[
 F(p)-F(p')=DF\!\left(\frac{p+p'}2\right)(p-p').
\]

The midpoint belongs to the square and its derivative is nonsingular. Consequently `F(p)=F(p')` forces `p=p'`. A continuous injection from a compact square to the plane is a homeomorphism onto its image. The image is contained in the convex quadrilateral by the nonnegative Q1 weights. The boundary maps once around its four edges. By the Jordan separation property / boundary winding number, every point inside that polygon belongs to the image; hence the image is exactly the closed quadrilateral. Equivalently, invariance of domain makes the interior image open, and compactness together with the boundary image makes it closed relative to the connected polygon interior.

**Scope.** Strict corner positivity is a sufficient condition for a homeomorphism and is necessary for a positive determinant on the *closed* cell. It is not claimed necessary for every topological Q1 homeomorphism: zero derivatives at boundary points can coexist with topological injectivity. Our contract deliberately excludes such degeneracy. Across source cell boundaries the assembled map is generally only C0, with one-sided derivatives, not a global C1 diffeomorphism.

## 3. Global homeomorphism — PROVED with the stated boundary hypothesis

Triangulate every source rectangle along SW–NE and use the same vertex values to form a continuous P1 map `G`. The two determinants on each cell are `q10,q01`, so every triangle is nondegenerate and orientation preserving. The boundary of `G` equals that of `F`.

The needed planar P1 fact is: an orientation-preserving, nondegenerate simplicial map of a finite triangulated disk with a bijective boundary map onto a simple polygon boundary is a homeomorphism onto that polygon. This is the two-dimensional specialization of [Lipman, Theorem 1](https://arxiv.org/html/1310.0955v2#S2.Thm1), whose hypothesis explicitly includes boundary bijectivity. The theorem statement and definitions in §2 were checked in the primary text; no claim of a new global inversion theorem is intended.

For completeness, the degree mechanism is as follows. For a target point away from mapped edges, its winding number around the mapped outer boundary equals the sum of the oriented indicators of all mapped triangles: interior edges cancel pairwise. Every nonzero contribution is +1, so points inside have exactly one preimage and points outside none. Positivity also supplies positive local degree at any interior source point: triangle interiors contribute one; the two sides of an interior edge map to opposite sides; an interior vertex fan has positive oriented wedges and winds a positive integer number of times. Thus the map is open at every interior point, including vertices. Two distinct interior preimages would give overlapping open images and contradict the preceding generic-point count. An interior preimage of a boundary point would give an open image containing exterior points, also impossible. Generic interior points exhaust a dense subset of the target; compactness supplies surjectivity on its closure. Boundary bijectivity then completes injectivity. A continuous bijection from the compact disk to its Hausdorff target has a continuous inverse.

Each source cell under `G` therefore occupies one closed convex quadrilateral, and distinct such quadrilaterals intersect only along their common mapped source edges or vertices. Section 2 proves that the Q1 restriction maps each source cell bijectively onto **that same** quadrilateral, with the same linear edge values. A target point in a cell interior has a unique Q1 preimage in that cell. On an edge or vertex, neighboring inverses agree because the shared edge map is the same injection. This proves global bijectivity of `F`; compactness again gives a continuous inverse.

Convexity of the outer target polygon is a convenient deployment hypothesis, not asserted to be logically necessary for the P1 boundary theorem. The proof above does not assume global injectivity merely from positive local Jacobians. It establishes it using the boundary hypothesis and shared topology. No conclusion applies to separately produced tiles with inconsistent common vertices.

## 4. Relation to forward/backward differences — PROVED

At a grid vertex define positive-direction derivatives using actual physical spacings, e.g. `D_x^- Phi=(Phi_ij-Phi_i,j-1)/hx_left`, `D_x^+ Phi=(Phi_i,j+1-Phi_ij)/hx_right`, and analogously in y. At a particular cell's SW, SE, NE, NW corners, the matching combinations `(++, -+, --, +-)` are respectively `q00,q10,q11,q01`, divided by that cell's positive area. Thus evaluating every cell corner is equivalent to all valid forward/backward combinations, including the one-sided combinations available at the domain boundary. Computing only at strict interior vertices omits constraints anchored on the outer boundary.

On a uniform grid, `D_x^0=(D_x^-+D_x^+)/2`, so determinant bilinearity yields

\[
J^{00}=\tfrac14(J^{--}+J^{-+}+J^{+-}+J^{++}).
\]

All four positive imply a positive central result; the converse fails. With nonuniform spacings, the two-neighbor secant is `D_x^0=(hx_left D_x^-+hx_right D_x^+)/(hx_left+hx_right)`, so use the products of positive x and y weights instead of `1/4`. This statement concerns that secant definition, not every possible higher-order nonuniform difference stencil.

Primary prior art: [Liu et al., §2.2 Definition 3 and §2.3](https://arxiv.org/html/2212.06060#S2.SS2) gives the digital four-difference / alternative-triangle criterion and explains central-difference failure. The Q1 interpolation and boundary extension above are explicit derivations for this repository, not quotations of a stronger theorem from that source.

## 5. F1-D incident constraints — PROVED formula, implementation pending review

Fix an initially valid map, a single active interior vertex `i`, and its proposed displacement `r_i`. A triangle's determinant is affine in any one of its vertices. The relevant list is all four corner triangles of all four incident cells **which contain i**, not just the four differences centered at i.

| i's role in an incident cell | Constraints containing i | Unchanged constraint |
|---|---|---|
| a / SW | q00, q10, q01 | q11 |
| b / SE | q00, q10, q11 | q01 |
| c / NE | q10, q11, q01 | q00 |
| d / NW | q00, q11, q01 | q10 |

An interior vertex therefore has 12 candidate constraints. Boundary vertices have fewer incident cells, but require additional boundary-order preservation if allowed to move. The first implementation should keep the entire boundary fixed.

For an oriented triangle `T(u,v,w)`, the exact linear coefficient under a displacement r of just one vertex is

\[
\begin{array}{c|c}
\text{moving vertex}&L(r)\\\hline
u&[r,v-w]\\
v&[r,w-u]\\
w&[v-u,r].
\end{array}
\]

One can cyclically rotate each oriented triangle so the active point is last and store its oriented opposite edge; then `L=[edge,r]`, matching the legacy implementation's convention. Do not reverse that edge when generating the extra six entries.

For every relevant triangle k let `q_k > 0`, `0 < gamma < 1`, and `0 <= beta_k < q_k`. Set

\[
\ell_k=(-L_k)_+,\qquad m_k=\min(\gamma q_k,q_k-\beta_k),\qquad
s_i=\min\left(1,\min_{\ell_k>0}\frac{m_k}{\ell_k}\right),
\]

where an empty inner minimum is ignored. For every `0 <= t <= s_i`,

\[
q_k(t)=q_k+tL_k\ge q_k-s_i\ell_k\ge q_k-m_k
\ge\max((1-\gamma)q_k,\beta_k)>0.
\]

This proves the entire straight update path safe. An initial equality `q_k=beta_k>0` can also be handled: `m_k=0`, an adverse constraint forces scale zero, and `ell=0` must be ignored. If the state is below a requested floor, clipping its remaining budget to zero can prevent further deterioration, but cannot retrospectively prove that floor was attained. Floors for normalized Jacobian `j_min` are `beta_k=j_min*hx*hy` for that triangle's cell.

Use four vertex colors `(row mod 2,column mod 2)`. Each cell has exactly one vertex of each color, so a color pass has at most one moving vertex per cell. Scales computed from a common pre-pass snapshot therefore do not interact. Later colors must use the updated map. Two-color checkerboard updates do not have this property. Features or raw proposals can be blended before computing these scales; independently accepted coordinate maps cannot inherit safety under arbitrary averaging.

The proposal `r_i=alpha*diag(hx,hy)*tanh(z_i)` is a practical bounded parameterization, not a condition needed for the algebra. For nonuniform grids a vertex has several neighboring widths: choose a documented physical scale, e.g. the minimum adjacent x and y widths, and keep every constraint's own area normalization.

## 6. F2-D affected-cell quadratic budget — PROVED formula

Let a patch proposal assign displacement `d_v` to each active vertex and zero to all other vertices. Collect every cell with at least one active vertex; for each collect the four oriented triples from §2. For `T(u,v,w)`, define

\[
e=v-u,\quad f=w-u,\quad \delta e=d_v-d_u,\quad \delta f=d_w-d_u.
\]

Then the exact polynomial under common patch scale t is

\[
q_k(t)=[e+t\delta e,f+t\delta f]
=q_k+tL_k+t^2Q_k,
\quad L_k=[\delta e,f]+[e,\delta f],\quad Q_k=[\delta e,\delta f].
\]

For `0 <= t <= 1`, let `C_k=(-L_k)_+ + (-Q_k)_+`. Because `t² <= t`,

\[
q_k(t)\ge q_k-t(-L_k)_+-t^2(-Q_k)_+\ge q_k-tC_k.
\]

With the same budgets m as F1-D, choose

\[
s_P=\min\left(1,\min_{C_k>0}\frac{m_k}{C_k}\right).
\]

Every cell corner stays above `max((1-gamma)q_k,beta_k)` for every `0 <= t <= s_P`. There is no convexity assumption on the joint patch feasible set. A pure common translation of all three triangle vertices has `delta e=delta f=0`; its constraint has `C=0` and must not restrict scale, including when its current area lies at a floor.

**Parallelism hypothesis.** In one pass, no affected cell may contain active vertices from two independently scaled patches. Disjoint active vertex IDs alone do not prove this. Fixed-perimeter, nonoverlapping rectangular patches meet the hypothesis because every affected cell lies within its own patch; sharing only fixed perimeter vertices/edges is harmless. A changed perimeter, overlap, or halo policy must rebuild the affected-cell proof. Incomplete border patches may be explicitly fixed, but their unchanged status and active-vertex coverage must be reported. Sequential staggered offsets are safe because each pass starts from the preceding accepted state.

**Less conservative proposal, OPEN implementation/numerics.** One can solve for the first positive root of `q_k(t)=q_k-m_k`, then use the first admissible interval common to all constraints. A second positive endpoint is not sufficient: a quadratic can cross a forbidden interval before returning positive. Degenerate linear/quadratic cases, tangencies, cancellation in roots, and derivatives at root switches need their own tests. The conservative adverse-budget formula already supplies a proof and should remain a reference.

## 7. Differentiability and refinement scope

The finite-dimensional hard-min/max layer is locally Lipschitz and differentiable almost everywhere on the open set of strictly feasible finite inputs with positive budgets, subject to the usual branch conventions. At tied minima, zero adverse terms, safety/filter changes, or zero budgets, claim only the derivative selected by the implementation or an explicitly tested one-sided derivative. A derivative through the current areas, budgets, and selected safety scale is required; detaching that scale is a surrogate gradient.

For a fixed initial geometry and a strictly active F1 constraint, a radial branch has `d(r)=m*r/(a^T r)` where m is independent of r, hence

\[
Dd(r)=\frac{m}{a^Tr}I-\frac{m}{(a^Tr)^2}ra^T,
\qquad Dd(r)r=0.
\]

Thus a finite/nonzero VJP alone does not establish a well-conditioned trainable representation. Measure active fractions and directional singular values. A candidate interior scale `s=1/(1+U)` is safe if `U >= max_k C_k/m_k >= 0` with all `m_k>0`, since `s*C_k<m_k`; construction of a useful smooth U, zero-budget behavior, and full-rank claims remain OPEN. A plain sum of nonnegative ratios is an upper bound but is generally conservative and still nonsmooth if C uses ReLU.

Q1 dyadic refinement uses edge midpoints and the **four-corner mean** at the cell center. Restricting a bilinear polynomial to an axis-aligned subrectangle is bilinear, so nodal reinterpolation reproduces the identical function. Local-coordinate child determinants are one quarter of the parent determinant evaluated at the corresponding location; physical-coordinate determinants are unchanged. More general nested tensor-product subdivisions have the analogous positive area factor. The finite-precision rounded version still needs checking.

The old P1 center `(a+c)/2` preserves the SW–NE P1 function, but generally changes the Q1 function. Agreement at coarse vertices does not resolve this difference. Arbitrary grids crossing old cell boundaries, coarsening, output casts, and sampled composition require fresh validation. An output-side affine composition `A(F)=MF+b` with `det M>0` preserves Q1 and multiplies all q by `det M`; generic input-side affine changes need not preserve the declared rectangular-cell partition. The inverse exists under §3 but is not generally a Q1 function on a regular target grid.

## 8. Exact rational failure fixtures

Named question: can fixed-boundary maps pass old P1 or central checks while violating the deployed four-corner standard? Both fixtures use the 3×3 source grid with axes `[0,1,2]`, keep every boundary vertex fixed, and change only `(1,1)`. All values below were recomputed with Python `fractions.Fraction`, independently of production geometry helpers, on 2026-09-29. This is the builder's arithmetic check, not the required independent checker.

**Fixture A — all old SW–NE triangles positive, Q1 folded.** Move the center to `(1/4,1/4)`. In cell order SW, SE, NW, NE, the four q vectors `(q00,q10,q11,q01)` are

\[
(1,1/4,-1/2,1/4),\quad
(1/4,1,7/4,1),\quad
(1/4,1,7/4,1),\quad
(5/2,7/4,1,7/4).
\]

Every old P1 determinant (`q10,q01`) is at least `1/4`; hence the old P1 map really is a fixed-boundary homeomorphism. Every cell-center Q1 determinant is positive: `(1/4,1,1,7/4)`. The unique interior-vertex central derivative is exactly the identity because its four axis neighbors have not moved. Nevertheless the SW cell has `J(s,t)=1-(3/4)(s+t)`, negative whenever `s+t>4/3`. A pipeline changing from the valid old P1 interpolation to Q1 therefore introduces an actual open folded region.

**Fixture B — central checks miss even a P1 inversion.** Move the center to `(5/2,1)`. The q vectors in the same cell order are

\[
(1,1,5/2,5/2),\quad
(1,1,-1/2,-1/2),\quad
(5/2,5/2,1,1),\quad
(-1/2,-1/2,1,1).
\]

All four cell-center values remain positive: `(7/4,1/4,7/4,1/4)`; the interior-vertex central derivative again equals the identity. Both Q1 and the old P1 map now fail positivity despite the unchanged simple boundary. The test must reject it. Scaling both source and target coordinates by `1/2` puts these fixtures on `[0,1]²`; q values scale by `1/4` and normalized Jacobians retain the listed values.

These examples distinguish center tests, the old diagonal-specific theorem, and the new Q1 theorem without relying on random floating-point failures. They do not show that the old P1 theorem was false; Fixture A satisfies that theorem and fails the stronger deployment representation.

## 9. Inspected legacy paths and exact deltas

References below describe the inspected P1 code, not any concurrent new Q1 implementation. Preserve legacy names/semantics or add a distinct Q1 path so old checkpoints are not silently reinterpreted.

| Inspected path | Existing behavior | Required Q1 delta |
|---|---|---|
| `src/qcopt/neural_bijection/dense/colored_vertex_relaxation.py:46` | Six cyclic opposite edges; buffered shape asserts `(N,6,2)` at line 74 | Generate all 12 oriented incident triples from §5; update buffered/generated consistency assertions and tests |
| Same file, `compute_area_floor`, line 81 | Computes only lower `(a,b,c)` and upper `(a,c,d)` minima | Include `(a,b,d)` and `(b,c,d)`; for nonuniform grids retain per-cell physical normalization |
| Same file, `_update_color`, line 95 | Four parity colors, radial adverse-area budget, guarded division | Retain coloring and differentiable scale; feed all 12 constraints; document extra conservatism of denominator guard |
| `src/qcopt/neural_bijection/dense/patch_field.py:111` | General triangle coefficient expansion already computes exact L and Q | Formula can be reused; four oriented triples must be supplied |
| Same file, lines 125–128 | Stacks only `(a,b,c)` and `(a,c,d)` per patch cell | Stack `(a,b,d)`, `(a,b,c)`, `(b,c,d)`, `(a,c,d)`; reduce one patch scale over the complete set |
| Same file, lines 61–77 | Complete square patch tiling; only disjoint active-ID assertion | Current fixed-perimeter regular tiling is compatible; explicitly test affected-cell nonconflict before supporting generalized/partial patches |
| Same file, lines 129–145 | Area floor uses `1/(side-1)^2`, adverse-only scale and finite guard | Preserve C=0 bypass; general rectangular/nonuniform cells need own `hx*hy` floors; report below-floor inputs |
| `src/qcopt/neural_bijection/dense/forward_p1_pyramid.py:20` | Exact P1 prolongation with diagonal center mean at line 34 | A separate exact Q1 refinement uses the four-corner center mean |
| `src/qcopt/neural_bijection/dense/multilevel_forward_p1.py:16` | `certify_p1_or_identity` checks only SW–NE triangles at lines 60–61, fixed reference boundary, bounded coordinates | New final-coordinate check must cover all four corners and declared coordinate range; a whole-image identity fallback must be counted as a registration failure, not silent success |
| `src/qcopt/neural_bijection/dense/alternating.py:47` | Named `evaluate_structured_p1_with_jacobian` | A Q1 deployment must use a Q1 query/Jacobian path; do not reuse triangle barycentric semantics merely because nodal shapes match |

The denominator guards in F1/F2 use `max(adverse_or_C, allowance, guard)`. This makes an adverse scale no greater than `min(1,allowance/adverse_or_C)` in exact arithmetic, so it is conservative. It is **not** by itself an outward-rounded certificate for the final coordinate additions, nor does it establish the gradient's numerical accuracy.

## 10. Remaining numerical and review questions

1. Independently check the four-corner ordering, 12-entry generation, rectangular/odd-size coverage, same-pass snapshot semantics, and F2 affected-cell overlap. The two rational fixtures above must be rejected by a checker that does not call production determinant helpers.
2. Validate the actual returned and saved dtype/coordinates. A theoretical q margin may disappear after addition to large absolute coordinates or an export cast. Float64 or a heuristic positive margin is not a universal finite-precision proof. Nonfinite proposals must not silently become certified states.
3. Audit any claimed rounding error bound against the actual subtraction/product order, underflow assumptions, compiler fusion, and fast-math settings. Exact predicates on represented coordinates or rigorously bounded filtered predicates can establish signs; unclassified signs cannot be accepted as positive. Strict zero is failure even if a non-diffeomorphic-area metric reports zero.
4. Validate Q1 values and boundary agreement before/after nested refinement, physical-coordinate conversion, affine output composition, writing and rereading, and across tile seams. Image interpolation and map interpolation are distinct operations.
5. Test true directional derivatives across a range of finite-difference step sizes away from active switches, and report one-sided behavior at switches; account for rejection/fallback branches. Check compiled and eager arithmetic independently when used.

G1 numerical acceptance, learning benefit, competitive registration, and any SOTA claim remain unestablished by this document. A homeomorphism of coordinate domains also does not guarantee preservation of pixel-label connectivity after rasterization, thresholding, or downsampling.

## 11. Candidate: repeated updates with a current-geometry proposal metric

**Research question.** A single coarse F2 schedule followed by one new-vertex-only F1 schedule may freeze an incorrect coarse vertex permanently. Can repeated full-vertex passes add expressivity without reverting to composition-and-resampling or making the latent proposal scale depend on the *undeformed* cell width alone?

**Exact candidate claim (derivation, not yet experimental result).** Fix one F1 color at an already valid Q1 map. For an active interior vertex (y_i), enumerate its twelve incident oriented corner areas as (q_k(d)=q_k(0)+a_k^\top d), where (din\mathbb R^2) is that vertex's proposed displacement and other vertices of this color do not share an affected cell. Let (0<\gamma<1), (0\le\beta_k<q_k(0)), and (m_k=\min\{\gamma q_k(0),q_k(0)-\beta_k\}>0). Set (v_k=a_k/m_k), (C_i=\sum_{k=1}^{12}v_kv_k^\top), (C_{i,\lambda}=C_i+\lambda_i I) with \(\lambda_i>0\), and

\[
d_i(z_i)=\eta C_{i,\lambda}^{-1/2}\tanh(z_i),\qquad
0<\eta<1/\sqrt 2,
\]

where `tanh` is componentwise. For any finite (z_i), (q_k(d_i)>q_k(0)-m_k\ge\beta_k\) for every (k). Indeed (C_{i,\lambda}\succeq v_kv_k^\top) implies \(\|C_{i,\lambda}^{-1/2}v_k\|_2\le1\), while \(\|\tanh z_i\|_2<\sqrt2\); hence \(|a_k^\top d_i|/m_k<1\). Recompute (C_i) **after each color and each repeated round**. The fixed rectangular source connectivity remains regular; only its image vertices are geometrically deformed. Since each round acts directly on the one stored vertex table and rechecks current constraints, no functional-composition resampling is involved. With the same ordered fixed boundary, exact-arithmetic global Q1 homeomorphism follows inductively from the earlier theorem.

The matrix inverse square root is a *proposal metric*, not a linear-system solve for the map. Large eigenvalues of (C_i) shrink motion in tightly constrained directions; small eigenvalues permit larger proposals. It is not claimed affine-invariant. At finite (z_i), positive \(\lambda_i\) and (m_k>0), the proposal has full-rank derivative with respect to (z_i), unlike a fully clipped radial ray; the dependence on the current map and the `min` budgets is only almost-everywhere smooth. If (q_k(0)=\beta_k\), a prototype can freeze that vertex and retain its already-satisfied floor; if (q_k(0)<\beta_k\), freezing **does not repair** the violated floor. Floating-point returned maps still require independent exact-sign certification. F2 multi-vertex motion has quadratic area changes, so applying these F1 ellipsoids to each patch vertex does **not** by itself guarantee a safe F2 pass; its existing shared quadratic path bound remains necessary.

**Independent numerical attack and repair.** A naive binary32 Gram matrix overflows on a perfectly regular (257^2) unit-square grid if the specified floor is one float32 representable value below the initial area; its VJP can become nonfinite. A naive eigenvector-based inverse square root can also have undefined autodiff at repeated eigenvalues, including a symmetric identity one-ring. The prototype therefore evaluates the areas/budgets in float64, divides all (v_k) at a vertex by their largest absolute component before forming the 2×2 Gram matrix, uses the closed-form analytic 2×2 inverse square root, and restores that scale in the proposal. The implementable relative ridge is restricted to `[1e-6,1]`, default `.001`. Independent tests now cover the (257^2) near-floor input, a (3^2) rectangle scaled to (10^{-10}), both base/latent VJPs, and equal-eigenvalue identity states; they do **not** establish behavior for every finite floating-point coordinate scale (a float64 (10^{-160})-scale fixture still breaks the base VJP). The tested neural canvases are normalized near `[0,1]^2`; a general-purpose production primitive would need a stated supported scale domain or a further filtered arithmetic path.

**Falsifier and smallest test.** At (17^2), compare this formula against a separate double-precision implementation of all twelve affine area updates for many distorted valid maps and extreme finite latents, including equal-eigenvalue identity states, near-floor cells, gradient checks and four repeated colors. Then compare fixed-αh radial proposals, dynamic scalar inradius (\min_k m_k/\|a_k\|_2), and the ellipsoid on matched (257^2) workloads: accepted displacement, active safety saturation, forward/VJP time, memory and fit to a target that moves previously frozen coarse vertices. A fit or speed gain is **not** implied by the safety derivation. Related primitives include vertex coloring/local feasible domains and radial feasible-set parameterization; their prior-art relationship needs targeted verification before any novelty claim.

**Image-conditioned recurrence and implemented pilot.** At mesh level \(\ell\), let \(Y^{\ell,0}\) be the vertex table obtained from the previous level by exact nested Q1 subdivision, followed by the already trained one-pass new-vertex F1 update. For update round \(t=0,\ldots,K_\ell-1\), form an image-conditioned latent \(z^{\ell,t}=E_{\theta,\ell,t}(I_F, I_M\circ Y^{\ell,t},Y^{\ell,t})\), then set \(Y^{\ell,t+1}=T^{\ell,t}(Y^{\ell,t},z^{\ell,t})\), where \(T\) is one complete current-area-safe four-color F1 schedule. The moving image is already prewarped by an *external initial-only* affine; that affine is not repeated inside this recurrence. The next level starts from exact subdivision of \(Y^{\ell,K_\ell}\). Intermediate *image intensity* sampling is allowed to interpolate brightness; the *map* remains one shared vertex table and is not reinterpolated after composing separate continuous maps. This permits coarse rounds to correct large-scale drift before the mesh is fine and updates inherited as well as new vertices.

In the implemented first pilot, the trained 102-case base image encoder and seed/refinement heads are frozen. Its image feature tensor is computed **once**. At sides 17, 33 and 65, the added head receives interpolated static features, fixed-image intensity at source-grid nodes, moving-image intensity sampled at **current deformed nodes**, the two current coordinate residuals (Y-X), and a scalar round index. A shared two-convolution head at each side emits new two-channel logits for each round; the last convolution starts at zero. The schedules are (K_{17}=4,K_{33}=2,K_{65}=1); sides 129 and 257 retain only their base one-pass new-vertex updates. Each repeated step uses the current-edge soft radial formula below with **fresh** areas, margins and one-ring vectors after every color. The output remains a 257² Q1 residual followed by the same initial affine. This is a specific pilot, **not** an approximation theorem or evidence that its feature head improves registration. Its DHR-label evaluation, if any, is exploratory because those eight confirmation labels had already been opened before the recurrent architecture test.

**Second candidate if the ellipsoid is too conservative: current-edge soft radial update.** Let (E_{x,i}=(Y_{i+e_x}-Y_{i-e_x})/2) and (E_{y,i}=(Y_{i+e_y}-Y_{i-e_y})/2), current deformed one-ring vectors, and propose (r_i=\alpha[E_{x,i},E_{y,i}]\tanh z_i). With the same (m_k>0), define (M_i(r)=\max\{0,\max_k[-a_k^\top r/m_k]\}\) and

\[
d_i=\frac{r_i}{1+M_i(r_i)}.
\]

For finite (r_i), each adverse area loss is at most (m_k M_i/(1+M_i)<m_k). Thus every affected corner remains strictly above (q_k(0)-m_k\ge\beta_k), again recomputing the current vectors and margins after every color. Unlike a hard minimum (\min(1,1/M_i)r_i), the radial derivative is nonzero for finite (r_i) along a fixed active ray, although it can become small as (M_i\to\infty). More precisely, freeze the current geometry, put (M=M_i(r_i)), and scale the raw ray by (t\ge0). Then (d_i(tr_i)=tr_i/(1+tM)) and its derivative along the ray is (r_i/(1+tM)^2), nonzero if (r_i\ne0). On a unique active branch, (\det D_{r_i}d_i=(1+M)^{-3}>0). This is an exact-arithmetic statement for finite logits, nonzero (\alpha), and positive finite margins; float32 `tanh(10)` can already have zero computed derivative. The denominator's `max` creates only an almost-everywhere derivative.

The centered edge matrix is the current-image analogue of (hI) and is nonsingular in exact arithmetic for a strictly corner-positive interior one-ring. Write (e=R-P,n=U-P,w=L-P,s=D-P) for right/up/left/down neighbor vectors, respectively. Then

\[
\det[E_x,E_y]=\tfrac14\bigl([e,n]+[n,w]+[w,s]+[s,e]\bigr)>0,
\]

where ([a,b]=a_xb_y-a_yb_x). The four bracket terms are the four incident positive cell corners at (P). Positive determinant does not imply good conditioning. If a floor budget vanishes, this proposal is undefined; freezing a vertex maintains an *already satisfied* floor but cannot repair a preexisting violation. For F2, one may use analogous current-edge raw proposals but **must** retain a patch-level bound on both linear and quadratic area terms; the F1 formula cannot simply be applied independently to simultaneous neighbors. An independent checker exhibited a 4×4 grid with coordinate axes `[-4,0,1,5]`: two diagonal neighbors each individually pass all twelve F1 tests and move by (6/5,0) and (0,6/5), yet moving them simultaneously makes the shared cell corner (-11/25). Four-color F1 scheduling prevents this simultaneous interaction; two-color scheduling does not. This construction resembles known radial feasible-set parameterizations and is not claimed novel. The 257² random-latent implementation below is still not an image-conditioned recurrent registration decoder.
