# Mathematical formulation of the implemented operators

Status: implementation/theory note, not a claim of registration superiority.
The first real experiments optimize each image pair's own coefficients. They do
not yet deliver a trained image-to-map network. Operator gradients are tested
locally; stages intentionally detach accepted anchors in instance optimization.

Current reading guide (around T+14h; the research window is still active): the main real
comparison outputs a257x257 CONTROL-vertex P1-ac map, evaluated on512x512 image
pixel centers. It is a per-pair optimizer, not a newly trained image encoder.
Sections1--2 introduce Q1 to derive the four digital corner constraints; they
do not change the current P1 declaration. Read sections3--5 for latent updates,
7 for the actual P1 interpolation,9 and25 for the frozen machine-point/ARAP
objective,36 for the20-direction protocol and limitations,37 and40 for the F2
strict-floor correction/full-budget comparison,38--39 and41/43 for measured
runtime diagnostics, and42 for the negative fixed-budget accuracy experiment. Earlier
objective variants are documented as experiments, not all simultaneously enabled.

A homeomorphism is a continuous bijection with a continuous inverse. A P1 map
is continuous and affine on each triangle of a declared source triangulation.
A Q1 map is continuous and bilinear on each declared source quadrilateral.
These definitions concern the actual interpolated map, not only its vertex table.

## 1. Coordinates and the actual map

Let the reference domain be [0,1]². For R rows and C columns, node (i,j) is
X_ij=(j/(C−1),i/(R−1)), including four corners and boundary nodes. Y_ij is its
learnable mapped position. The current experiments use R=C=257, not a257² query
grid masquerading as a dense control grid. A cell has a=Y_ij,b=Y_i,j+1,
c=Y_i+1,j+1,d=Y_i+1,j. Its Q1 map, for local coordinates ξ,ζ in[0,1], is

    Q_Y(ξ,ζ)=(1−ξ)(1−ζ)a+ξ(1−ζ)b+ξζc+(1−ξ)ζd.

Adjacent cells share their edge values. Complete fixed-to-moving map is
F_Y(x)=A Q_Y(x)+b0, where A is a frozen2×2 matrix with positive determinant and
b0 a frozen2-vector. Symbols a,b,c,d denote cell vertices, not the affine matrix.
Both factors are retained in the saved archive. A new rounded materialization
is not silently covered by the original factorized certificate.

Pixel intensities are defined at centers ((j+.5)/W,(i+.5)/H). Map vertices are
defined at endpoints j/(C−1),i/(R−1). Hence map interpolation uses endpoint
conventions, while intensity interpolation uses pixel-center conventions.
These are different grids even when their dimensions happen to be equal.

## 2. Four corner constraints, and what they guarantee

Write det(v,w)=v_x w_y−v_y w_x. Protect

    q1=det(b−a,d−a), q2=det(b−a,c−b),
    q3=det(c−d,c−b), q4=det(c−d,d−a).

Each q is twice a signed corner-triangle area, not its area. The reference
determinants qref are positive; on the regular grid qref=1/((R−1)(C−1)).
The dimensionless corner Jacobian is J=q/qref. Identity has J=1.

The derivative determinant of Q_Y is affine in ξ,ζ: with B=b−a,D=d−a,
M=a−b+c−d it is det(B,D)+ξ det(B,M)+ζ det(M,D). Strictly positive corner values
therefore imply positivity throughout the cell. With continuous shared edges
and a simple injective outer boundary, the orientation-preserving cell maps
give a global homeomorphism. Boundary is a hypothesis, not a consequence of
merely inspecting interior determinants. The implemented default boundary is
exactly the identity rectangle, followed by the positive affine.

The SAME vertex table also defines P1 maps using either cell diagonal; both
have positive triangle orientations. Q1 and these P1 functions are generally
different. First image objective/evaluator is explicitly Q1. Sampling another
grid and interpolating its samples does not inherit this theorem.

The optimization additionally imposes J>η, initiallyη=.001. This is stronger
than necessary J>0. A legal target belowη is excluded; numerical experiments
explicitly demonstrate this whenη=.05. Floating point requires checking actual
rounded output coordinates, separately from the real-arithmetic derivation.

## 3. Exact single-direction geometry

Fix one unit vector e and scalar amplitudes u_i for every node. All nodes can
move simultaneously: Y'_i=Y_i+u_i e. For triangle (i,j,k),

    det(Y'_j−Y'_i,Y'_k−Y'_i)
    =det(Y_j−Y_i,Y_k−Y_i)
     +(u_j−u_i)det(e,Y_k−Y_i)
     +(u_k−u_i)det(Y_j−Y_i,e).

The quadratic term vanishes because det(e,e)=0. Thus, after normalization,
every protected constraint is s_k+(C_Y u)_k>0, where
s_k=q_k(Y)/qref_k−η and C_Y is the linear operator specified by this formula.
C_Y is applied through differences and determinants, not assembled as a dense
matrix and not inverted. Fixed-boundary amplitudes are exactly zero.

Boundary sliding is an optional operator mode, not used in the first real matrix.
Horizontal e allows top/bottom tangential movement, vertical e allows left/right
movement; corners are fixed. Every ordered boundary gap is an additional linear
constraint, normalized by its positive reference gap. Normals are constrained,
and ordering is explicitly protected rather than assumed.

## 4. Two different entries into this feasible set

For a proposal z, define g_Y(z)=max(0,max_k[−(C_Y z)_k/s_k]). The maximum is over
all corner and, if enabled, boundary-gap constraints. The radial entry is

    u=z/(1+g_Y(z)).

It gives s+C_Yu>0 in exact arithmetic. It is not a geometry line search. Let
K_Y={u:s+C_Yu>0}; then g_Y(u)<1 and the inverse is z=u/(1−g_Y(u)). This proves
coverage of THIS single-direction feasible slice, not all two-dimensional
homeomorphisms in one step and not good numerical conditioning.

Where a unique active constraint makes g differentiable,

    D_z u=I/(1+g)−z(Dg)^T/(1+g)².

Along the proposal ray its sensitivity is1/(1+g)². Largeg can therefore make
optimization very slow even though the entry covers the entire feasible slice.
Maxima are nonsmooth at ties; the operator is differentiable almost everywhere.

The alternative analytic entry defines

    αmax=min_{k:(C_Yz)_k<0} s_k/(−(C_Yz)_k),
    α=min(αtrial,θ αmax), u=αz, with0<θ<1.

An empty minimum is+infinity. It preserves all constraints without an inner
geometry search. A saturated bound makes displacement independent of multiplying
z by a positive scalar, so its radial derivative is zero. Image acceptance is a
separate decision. A branch-guarded equivalent computesα fromg without
differentiating a huge inactive reciprocal; this fixed an actual near-zero VJP
overflow. The safety scale remains in autograd; it is not detached.

For decoder output D(c) and an upstream array v, the vector-Jacobian product
(VJP) is (D_c D)^T v: reverse-mode differentiation without storing a dense
Jacobian. The gradient includes the safety scale. If the anchor is itself a
learnable input, C_Y and the slack s also depend on it and must be differentiated.
The current instance optimizer intentionally detaches accepted stage history;
that is different from end-to-end training through an entire decoder cascade.

## 5. Multiscale variables and accepted stages

At coefficient levelℓ, learn interior c∈R^(ℓ−2)×(ℓ−2) (two components for F1/F2).
Pad its boundary with zeros, then bilinearly interpolate its RAW proposals to
the actual257² output grid: z=P_ℓc. No accepted-map resampling is performed.
Coefficients are physical displacement units, not units of the finest mesh edge.
Initial Adam rate is .004·16/(ℓ−1). This calibrates edge-scale proposals but does
not make parameter-coordinate work or conditioning identical across methods.

Each stage fixes anchorYbar, starts coefficients at zero, evaluates decoder
D(Ybar,P_ℓc), and minimizes the declared complete objective. A declared number
n of gradient updates plus evaluation of the last update are included; current
mainline uniform stages use n=30. The best legal full-objective
candidate is accepted, detached, and becomes the NEXT stage's anchor. Trials
inside a stage are not cumulatively compounded. Directions alternatex/y for
coordinated operators. Fixed image evidence is always queried through the full
map, not through recursively warped rasters.

Optional image continuation associates coefficient level ell with image side W_ell.
Each reduced fixed/moving image is an area average of its ORIGINAL full raster.
The reduced fixed mask is the area average of the original fixed mask, not a mask
computed from current overlap; fractional tissue weights are retained. With powers
of two, mask sum times pixel area is conserved exactly by this averaging.
Each stage accepts against its own complete E_ell, including the same physical
strain, optional shape and OOB weights. Changing W_ell can increase the full-resolution
objective. The latter is recorded after each stage and reported finally; no global
monotonicity claim is made for continuation. The current budget includes initial,
stage-anchor, decoder-trial, accepted full-resolution and final objective calls;
decoder trials alone are not all objective evaluations.

## 6. Objective and metrics

Let m(x) be fixed foreground mask and N=Σ_x m(x)>0. Descriptor or local-NCC
error isℓ(x;F_Y). The implemented objective is

    E(Y)=Σ_x m(x)ℓ(x;F_Y)/N +λ S(Y)+β O(Y),
    S(Y)=.5[mean||(C−1)Δ_x(Y−X)||²+mean||(R−1)Δ_y(Y−X)||²],
    O(Y)=Σ_x m(x)||relu(−F_Y(x))+relu(F_Y(x)−1)||²/N.

S is the cumulative residual strain, not strain of only the latest update.
λ=.05,β=1 in the first matrix. Queries outside the rectangle are retained;
zero padding andO apply. N never shrinks when overlap worsens. MIND uses eight
fixed offset descriptors; NCC squares normalized cross-correlation, so positive
and negative correlation are treated equally. The actual preprocessing and
window size are documented in CASE_PROTOCOL.md. LowerE does not imply lower
anatomical error or correct matching of missing tissue.

More precisely, the eight-channel MIND-like descriptor uses offsets
{(2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2)} pixels.
For offset r, D_r is the3-by3 local mean of (I(x)-I(x+r))², with replicated
padding for shifted images. Let v be the channel mean of D_r and m their minimum.
The descriptor is exp(-(D_r-m)/(v+1e-4)); mean absolute channel difference gives
the image error after sampling ORIGINAL moving descriptors. This implementation
is called MIND-like, not asserted identical to every published MIND variant.

An optional corner-shape regularizer is

    R(Y)=mean over cells/corners [||J||_F²(1+det(J)^(-2))-4],

where J has the normalized Q1 derivative columns at that corner. For a positive
2-by2 J the expression equals ||J||_F²+||J^(-1)||_F²-4. It is zero on rotations,
diverges at singular compression, and is a four-corner QUADRATURE, not the exact
Q1 integral. When enabled, E includes the same declared weight of R for every
method. This conventional distortion regularizer is not the topology guarantee.

Geometry and evidence precisions are separately selectable. With float64 geometry
and float32 evidence, Q1 and affine coordinates are evaluated in float64, then
cast to float32 only for raster grid sampling. The original double vertex table
is retained and certified. Casting sampling queries neither changes the exported
map nor extends its certificate to a newly rounded vertex table. Gradients pass
through the cast; finite-precision numerical errors are still measured separately.

For independent evaluation landmark pairs (p_f,p_m), convert native pixel centers
to saved canvas coordinates and compute TRE=||512 F_Y(p_f)−512 p_m||_2. Report
all shared IDs, pairmean,p90,max and native moving-pixel units separately. Do not
use landmarks to select iterates. Cases are development specimens, not independent
clinical validation. Time-to-accuracy is a joint result; no-folding alone is not
successful registration. Forward/objective,VJP and complete optimizer time, memory,
initialization/evidence/certification cost are distinguished.

Image continuation is a sequence of different objectives E_l. Its stage acceptance
does not imply decrease of the final-resolution E_full. The optional best_full output
rule evaluates E_full on the initial and each accepted-stage table, retains the one
with smallest value, and returns it after the trajectory. Those evaluations were
already recorded for diagnostics; the rule adds a stored vertex table and conditional
copying, not an unreported second optimization. Intermediate anchors still follow
E_l acceptance and are not reset to the retained best map. This is an objective-only
selection rule, not selection by map truth, anatomical landmarks or held-out errors.

## 7. The actual P1 function and compatible current-mesh composition

P1 mode declares one fixed source diagonal in every cell. At local query(ξ,ζ),
diagonal ac uses

    (1-ξ)a+(ξ-ζ)b+ζc, if ζ<=ξ;
    (1-ζ)a+ξc+(ζ-ξ)d, otherwise.

Diagonal bd uses

    (1-ξ-ζ)a+ξb+ζd, if ξ+ζ<=1;
    (1-ζ)b+(ξ+ζ-1)c+(1-ξ)d, otherwise.

These are triangle barycentric weights (nonnegative and summing to1). Their
values agree on shared edges and at vertices. For a nonaffine quad they generally
DIFFER from Q1 and from one another. Archive metadata selects q1/p1_ac/p1_bd, and
the independent scorer solves source-triangle barycentric equations rather than
reusing the production Torch formulas. The public sampler enforces finite queries
inside the reference rectangle. Pixel centers are constructed inside it directly.

For FIXED source queries, P1 evaluation is linear in the mapped vertices. Even at
a source diagonal, the two incident formulas give the SAME vertex derivative,
because the opposite-vertex weights vanish. The spatial derivative with respect
to a MOVING query can be nonunique on triangle edges. These are different gradient
questions. Raster intensity interpolation introduces its own piecewise derivative.

Let f_Y be the P1 map before a safe vertex-table update, with one fixed diagonal.
It maps each source triangle to its current deformed triangle. Define G on that
CURRENT triangle as the affine map sending each Y_i to the updated Y'_i. Then

    f_Y' = G composed with f_Y

EXACTLY on every original triangle: composition of these two compatible affine
pieces sends each original vertex X_i to Y'_i. Adjacent pieces share edge values.
This is not composition of unrelated regular-grid warps followed by resampling.
There is no expanding overlay or triangle search when directly updating vertex
tables. The current geometry used to bound the step is Y, not the original h alone.
The final image map is A composed with f_Y'; A remains the frozen positive affine.
This compatible P1 composition interpretation does not make a general composition
of two Q1 maps bilinear, nor permit changing source diagonals silently mid-cascade.

Why the local checks are global here: on either source triangulation, every
triangle has positive orientation, and the boundary is the simple fixed rectangle.
For a generic point away from mapped edges, an oriented triangle contributes1
when the point is inside it and0 otherwise. Summing these contributions cancels
every interior oriented edge, leaving the outer boundary's winding count:1 inside
the rectangle,0 outside. Thus generic points have exactly one triangle-interior
preimage. Strict nondegeneracy and shared-edge/fan continuity extend this tiling
to edges/vertices. The map is therefore a continuous bijection; compactness of
the closed rectangle gives continuity of its inverse. Boundary injectivity and
strict orientation are essential hypotheses, not numerical residual claims.

## 8. Coverage is not an optimization guarantee

The radial formula covers the entire open SINGLE-DIRECTION feasible amplitude
set (or its specified coarse coefficient subspace). At dense coefficient level,
every interior amplitude can be represented. If two vertex tables are joined
by a continuous fixed-boundary path whose four corners stay strictly positive,
compactness gives a positive minimum slack. A sufficiently fine path partition
can be implemented by alternating horizontal and vertical updates: the intermediate
horizontal-only vertex table stays within the open feasible neighborhood, followed
by the vertical increment. Each feasible increment has the stated inverse radial
encoding. This is conditional finite-stage reachability in that path component.
It does NOT prove that every digital embedding has such a path, that a fixed small
depth is universal, or that image optimization finds the path quickly. No removal
of those assumptions is claimed without a separate theorem with correct boundary
and graph hypotheses.

## 9. Optional frozen image-correspondence evidence

This option uses an EXISTING frozen SuperPoint/SuperGlue model, not a new image-to-
deformation network. For fixed keypoint(pixel center)x_j and affine-prewarped moving
keypoint p_j^pix, store q_j=(x_j+.5)/(W,H) and p_j=(p_j^pix+.5)/(W,H). The latter
is in the affine-ALIGNED coordinate frame: the corresponding original moving query
is A p_j+b, not p_j. Original pixel numbering starts at0. For our square512canvas,

    e_j(Y)=512[A f_Y(q_j)+b-(A p_j+b)]=512 A(f_Y(q_j)-p_j).

Thus the offset cancels but the matrix must remain in the physical metric. Define

    r_j=e_j/κ, κ=8canvas pixels;
    ρ(r_j)=sqrt(1+||r_j||²)-1;
    E_match(Y)=sum_j w_j ρ(r_j), sum_j w_j=1.

This is DIMENSIONLESS pseudohuber, without the usual optionalκ² multiplier. Weights
are the original fixed matcher confidences, normalized after static validity:
targets A p_j+b outside the original moving rectangle receive weight0, because the
affine prewarp's border extension does not supply genuine outside image evidence.
Counts and excluded targets are reported. This selection does NOT depend on current
Y, overlap or objective values. Fixed image/manual evaluation denominators are not
altered; no manual correspondence or competitor's dense map is read by optimization.
The point term uses the DECLARED Q1/P1 interpolator and SAME512pixel metric at every
image-pyramid level. Complete acceptance/selection uses

    E_image,l + .05 E_strain + E_OOB + 1e-4 E_shape + λ_match E_match.

The first predeclared test usesλ_match=.1;λ=0is the exact paired control. Machine
matching is frozen and discrete, so we do NOT claim differentiation through its
keypoint detection/matching. Gradients of this fixed point evidence to decoder/map
vertices are tested independently. Topology comes ONLY from the feasible decoder
and saved map checks, not the matcher or this robust penalty. Accurate sparse fits
can leave unobserved dense regions wrong; confidence is not anatomical correctness.

Global-similarity RANSAC is not used in the new raw probe: measured known-target
matches are accurate even where their motion disagrees with one global similarity.
The prior filtered probe remains a distinct comparison, not silently overwritten.
Original affine metadata, raster names/size and static-domain counts are recorded;
basename comparisons permit relocation but are not a proof of image-content identity.

## 10. Physically calibrated residual smoothness, not total-field smoothness

Write the residual map as g(q)=q+r(q). The complete sampling map is A g(q)+b.
Our square nodal grid has endpoint spacing h=1/(N-1). Define

    E_strain(r)=.5[mean_edges ||(r_{i,j+1}-r_{i,j})/h||^2
                      + mean_edges ||(r_{i+1,j}-r_{i,j})/h||^2].

This expression penalizes the DISPLACEMENT gradient, not the map gradient.
Translations have zero energy, rotations generally do not. Native DHR's
executed optimizer first bicubic-prewarps the source with the frozen affine,
optimizes a zero-initialized residual field, regularizes that residual at every
image level, and only afterward composes it with the affine. Its normalized
sampling coordinates are [-1,1], so its displacement is d=2r. At pixel-center
spacing1/S its diffusion expression is

    D_S(d)=.25[mean ||S(d_{i,j+1}-d_{i,j})||^2
                    +mean ||S(d_{i+1,j}-d_{i,j})||^2].

For the SAME physical affine residual r(q)=Cq+t on their respective grids,
E_strain=.5||C||_F^2 and D_S(2r)=||C||_F^2, exactly and independently of
translation or grid size. Thus its coefficient1.5 motivates ONE test of our
coefficient3, rather than the previous.05. Nonaffine fields have different
finite-difference quadratures, image-level grids and boundary families; this
is a units-calibration test, NOT equality of the full optimizers/objectives.
Reinterpreting the same array on an endpoint grid versus a pixel-center grid
would introduce an S/(S-1) factor, but that compares DIFFERENT physical fields.
Neither A nor b enters this residual regularizer. Replacing r by A r or
(A-I)q+A r+b changes the energy and would not reproduce the native code path.
Large coefficients can bias away from a known legal target; numerical image
loss improvement cannot override that observed correspondence degradation.

## 11. Exact nested P1 refinement

Fix one global source diagonal, ac or bd, on every rectangular cell. Uniformly
divide each old horizontal and vertical interval into m equal pieces, with
positive integer m, and keep the SAME diagonal choice in each new cell. Set
each new mapped vertex to the OLD P1 value at its source location, keeping old
vertices unchanged. The fine triangulation refines every old triangle, so its
P1 map is EXACTLY the old function in real arithmetic, not a resampled
approximation. This statement is false for general Q1 interpolation or an
unrelated changed diagonal. For ac, the midpoint of a cell is (a+c)/2; for bd
it is (b+d)/2, generally NOT (a+b+c+d)/4.

The stronger four-corner digital condition is also inherited in real arithmetic.
Use local source coordinates in[0,1]^2. A fine cell wholly inside an old triangle
is mapped by its affine Jacobian, so all four normalized determinants equal that
parent face's positive determinant. A cell straddling ac starts at(s,s), has
width1/m, and its mapped vertices in order are

    T+(a,b,c,d)/m,  T=(1-s-1/m)a+s c.

A cell straddling bd starts at(u,v), u+v=1-1/m, and has the same form with
T=u b+v d. Such a cell is a translated homothetic copy of the old mapped quad:
its four determinants are old determinants/m^2; its reference determinants
are also divided by m^2. Consequently the minimum normalized four-corner
determinant is preserved (each old cell has a straddling fine cell).
Global homeomorphism additionally needs the old map's valid boundary/global
hypotheses, which refinement preserves by representing the same function.

Rounded coordinates are a separate issue. A positive one-ULP-height quad near
ordinate.5 has an unrepresentable midpoint and can become degenerate after
float64 refinement. Therefore every exported fine map still receives fresh
actual-corner/boundary checks; the real-arithmetic theorem is not a floating-
point certificate. General257maps cannot be exactly coarsened to129nodes;
our same-function scaling benchmark starts at257 and refines to513/1025.

## 12. Frozen source-query P1 evaluation and its transpose

Let N be the number of mapped vertices and Q the number of FIXED reference
queries. For each query q_k, locate its reference triangle ONCE, with vertex
indices i_k1,i_k2,i_k3 and barycentric weights w_k1,w_k2,w_k3. Define the
Q-by-N matrix B implicitly by B[k,i_kj]=w_kj (summing coincident indices if
needed) and all other entries zero. Then the evaluated map is simply

    V_k=sum_{j=1}^3 w_kj Y_{i_kj}, or V=B Y.

Given the upstream derivative G=dL/dV, the vertex derivative is B^T G:
each query scatters w_kj G_k into vertex i_kj. No dense matrix is assembled;
three indices/weights per query suffice. Deforming Y does not change this
ORIGINAL source-triangle assignment. Compatible vertex-table composition
and exact P1 refinement do not permit unrelated moving-query composition.
At fixed diagonal/edge queries, zero opposite weights preserve the same
vertex derivative. This cache supports arbitrary vertex batches sharing one
query table, but DOES NOT support query gradients: requires_grad queries
are explicitly rejected. Module dtype casts round stored weights rather
than reconstructing differently rounded query coordinates.

The optional application cache is prepared once for EVERY image-pyramid
resolution, before trial timing. Its setup time is charged to feature setup,
and its resident buffers count in allocated peak memory. It leaves the original
foreground mask, affine, moving-image sampling, point term and actual topology
checks unchanged. The same differentiable graph reaches map vertices and
latent coefficients. Raster float32 casts use ordinary floating-point autodiff
conventions, not a claim of differentiating the bitwise rounding operation.

## 13. Native baseline field units and localized topology comparison

Native DHR's internal displacement tensor uses[-1,1]sampling coordinates.
Its saver multiplies the x/y components by width/2,height/2. The executed
MHA files therefore contain PIXEL displacements d, not normalized values.
For the zero-padding/unit-resampling512 cases, the actual center-vertex map is

    Y_ij=((j+.5+d^x_ij)/512,(i+.5+d^y_ij)/512).

This defines an interpolant on the TRIMMED square[1/1024,1023/1024]^2.
It is not an endpoint grid on[0,1]^2; no padding/extrapolation is invented.
Our audit promotes savedfloat32 displacements tofloat64 before adding the
exact dyadic center reference, then tests those constructed stored coordinates.
Normalize four corner determinants by(1/512)^2. BothAC andBD P1 faces
are reported separately; four-corner failures also rule out everywhere-positive
Q1 Jacobians on a failing cell. Global boundary tests use exact dyadic integer
segment predicates after exact-comparison bounding-box pruning. A negative
minimum is separately recomputed using Fraction homogeneous triangle areas.

Localization uses ONLY the static original fixed grayscale mask, not current
overlap. An AND tissue cell has allfour mask-positive nodes; an OR tissue cell
has at leastone. Counts are separately reported for source boundary exclusion
bands0/1/4/16/32 and bothdiagonals. Failure of these native interpolants does
not imply inaccurate anatomical landmarks, and good TRE does not imply global
topology. Native timing/accuracy remain useful practical baselines, with these
geometric limitations stated; no post-hoc repair is used as our topology claim.

## 14. Candidate-only first-order adjoint without a constraint trajectory

Research card (2026-10-01, approximately T+5h):
Question: can the SAME coordinated candidate retain derivatives with respect to
both the current mapped vertices and the proposed scalar amplitudes, without
retaining an autograd graph for every corner constraint?
Exact claim: the formulas below reproduce first-order autodiff for the existing
fixed-boundary decoder, including its selected maximum subgradient and analytic
branch. This is an engineering adjoint, not a new deformation family.
Assumptions: reference vertices, direction and trial scale are constants; the
reference is valid; input/output checks still inspect actual rounded coordinates.
Only the candidate is the differentiable output. Auxiliary margin derivatives,
sliding boundaries and higher derivatives are NOT claimed by this new API.
Falsifiers: disagreement in current-vertex or latent gradients, finite differences
away from nondifferentiable ties, different forward rounding or tie selection,
or absence of a meaningful memory/performance benefit at the application scale.
Smallest decisive tests: a non-square batch, unique active constraints, positive
maximum ties, zero proposals, inactive analytic bounds and explicit invalid-input
rejection. Follow with GPU dense-grid and complete-registration measurements.
Prior work: ordinary reverse-mode differentiation of a scalar maximum and the
local determinant differential; Sections 2--4 specify the decoder itself. An
independent checker verified local formulas and current-Y derivatives before
implementation; this does not replace numerical checks of the implementation.

Let M zero the fixed boundary scalar entries, r=M z, and let e be the fixed unit
direction. Write the candidate as T=Y+sigma(g) r e. For each of the four corner
constraints k, use precisely the existing normalized forward quantities

    s_k=q_k(Y)/qref_k-eta,
    delta_k=Dq_k(Y)[r e]/qref_k,
    a_k=max(-delta_k,0),
    g=max_k a_k/s_k.

All s_k are strictly positive. For radial decoding sigma=1/(1+g).
For analytic decoding sigma=theta/g only when g*alpha_trial>theta;
otherwise sigma=alpha_trial. Equality selects the constant-trial branch, as in
the existing implementation, rather than an averaged minimum subgradient.
The radial derivative is sigma'=-sigma^2. The active analytic derivative is
-theta/g^2; on its constant branch it is zero.

Given upstream vertex derivatives G=dL/dT, define W_i=G_i dot e and
beta=sum_i W_i r_i. The direct derivatives are G with respect to Y and
sigma W with respect to r. The remaining scalar derivative is c=beta sigma'.
If I is the set of EXACT forward maximum ties and K=|I|, the implemented
amax convention distributes c/K to each tied row. For k in I its local
constraint derivatives are

    dL/ddelta_k=-(c/K)/s_k * 1_{delta_k<=0},
    dL/ds_k=-(c/K)*a_k/s_k^2.

The equality in the indicator matches torch.clamp_min's derivative at zero.
At g=0 all maximum rows are tied; one may not silently introduce an additional
zero row or divide by only a subset. If beta=0 the entire gauge adjoint is zero
and can be skipped, avoiding an unnecessary all-zero-proposal tie table.
An inactive analytic branch likewise needs no gauge adjoint.

Each q_k and delta_k uses only its corner's local mapped vertices and scalar
amplitudes. Their local derivatives are accumulated into Y and r, then the
boundary mask applies to the latter:

    dL/dY=G+sum_k [(dL/ds_k) Ds_k+(dL/ddelta_k) D_Y delta_k],
    dL/dz=M [sigma W+sum_k (dL/ddelta_k) D_r delta_k].

For interpretation, an oriented triangle with E=Y_b-Y_a, F=Y_c-Y_a,
p=r_b-r_a and v=r_c-r_a has q=det(E,F) and unnormalized change
delta=p det(e,F)+v det(E,e). With R(x,y)=(-y,x),

    D_Y q=(R(F-E),-R F,R E),
    D_Y delta=((v-p)R e,-v R e,p R e),
    D_r delta=(-det(e,F)-det(E,e),det(e,F),det(E,e)).

These expressions explain locality, but implementation must preserve the actual
four-corner arithmetic and normalization ORDER before selecting ties. Cancelling
qref algebraically or replacing a corner by an algebraically equivalent triangle
can change floating-point ties. A compact active-row table with bounded chunked
local recomputation is permitted; dropping tied rows is not. Input tensors need
not be copied, but their saved storage still counts toward resident memory.
Default rounded-output checks remain independent of this derivative computation.

Selected-stencil manual-backward research card (T+5.7h): the first explicit VJP
greatly reduces retained storage but is SLOWER for active257/513 trials. Question:
replace only tiny local reverse-mode graphs with direct local derivatives. Forward
normalization, tie indices, scale, candidate and rounded checks remain unchanged.
For the ACTUAL corner expression q=det(E,F), E=Y_j-Y_i, F=Y_l-Y_k,
the derivatives scatter (+R F,-R F,-R E,+R E) to (i,j,k,l), respectively.
Repeated indices sum their contributions. In row-major cell ordering(a,b,d,c),
the corner edge tuples are (0,1,0,2), (0,1,1,3), (2,3,1,3), (2,3,0,2).
The directional-change derivatives still use the original triangle triples
(0,1,2), (0,1,3), (2,1,3), (0,3,2) and the formulas above. This preserves
which edges are differentiated rather than rewriting corner values algebraically.
All tied rows remain in the subgradient; bounded processing is not row selection.
Different accumulation grouping can yield tiny floating-point gradient differences,
so agreement tolerances are reported, not bitwise gradient equivalence. Falsifiers:
full-Y/latent derivative or tie/branch/cascade/finite-difference tests fail, or
measured active performance still offers no benefit. No application integration
or optional compiler dependency is justified by algebra alone. This candidate
engineering refinement is independently checked by the coordinator before coding.

## 15. Known-image recovery with a declared P1 estimate and frozen Q1 truth

Research card (T+5.3h): the prior known-texture experiment used Q1 both for
generating images and evaluating fitted maps. Question: does its recovery carry
over when the actual optimizer and estimated function are P1 on a declared
diagonal? Keep ALL prepared Q1 images/matches and ground-truth vertex tables
unchanged. Estimate interpolation is an explicit option; no old result is relabeled.
Let f_true be the Q1 function used to generate fixed intensities from the original
moving raster, and f_est the fitted P1 function. Independent held-out queries
q_k use the same seed/count as before, NEVER optimization or output selection:

    RMSE_px=S sqrt[(1/Q) sum_k ||f_est(q_k)-f_true(q_k)||^2].

S is image width in canvas pixels. P1 estimates must NOT be scored using their
Q1 reinterpretation. The raster residual likewise evaluates the actual declared
estimated function; the PNG truth floor still uses the ORIGINAL Q1 function.
The truth-objective diagnostic evaluates that Q1 function and is labeled as such,
not as an exactly representable P1 optimum. Report separately the discrepancy
between P1 and Q1 interpretations of the SAME target vertex table: this measures
an interpolation discrepancy, NOT the best possible P1 approximation lower bound.
Smallest decisive tests: a nonaffine quad with distinct Q1/AC/BD values, exact
affine values/endpoints, independent triangle search, and a mocked image-only
optimizer receiving no truth or evaluation queries. Default Q1 behavior remains.
Failure at the new P1 setting is evidence about this objective/budget/representation,
not proof that no P1 homeomorphism can register those images.

## 16. Constant-anchor stage caching (instance optimization only)

Research card (T+5.4h): current per-instance optimization freezes the accepted
anchor Y during each thirty-step latent stage. Question: can reference areas,
current slacks and determinant slope coefficients be computed ONCE per stage,
without changing any trial or scalar-latent derivative? Claim: the same normalized
delta and slack enter the existing radial/analytic layer, with identical corner
arithmetic. Assumptions: Y, reference and direction are constants for the ENTIRE
stage; all are cloned when the cache is built. A changed anchor requires a new
cache. Trainable anchors/references are explicitly rejected, not silently detached.
This is NOT a full-Y neural-cascade adjoint; Section14 provides that separate API.
Falsifiers: candidate/diagnostic/proposal-gradient disagreement, mutable caller
inputs invalidating cached geometry, or no end-to-end improvement after counting
construction and resident storage. Test both modes, non-square batches, explicit
references, tiny/inactive proposals, caller mutation and rounded output rejection.
Prior work: reuse of fixed coefficients in a matrix-free linear operator; the
directional determinant identity of Section2. No new deformation family is claimed.

For each oriented corner triangle (p,q,r), precompute the constants

    k1=det(e,Y_r-Y_p), k2=det(Y_q-Y_p,e).

For every new scalar proposal u, the unnormalized determinant change remains
(u_q-u_p)k1+(u_r-u_p)k2. Use the same corner triples and operation order as the
existing implementation, then divide by stored qref. The normalized slack is
q(Y)/qref-eta and is constant throughout the stage. Reuse the existing gauge,
scale and all public diagnostic derivatives with respect to u. Output corners
are STILL freshly computed from the actual rounded candidate on EVERY trial;
constant-anchor caching does not replace this check with a predicted margin.
Stage construction time and extra constant arrays count in application cost.

## 17. Multilevel multi-stage latent decoder benchmark (not encoder training)

Research card (T+6h): one-layer VJP measurements do not establish memory or
gradient behavior for a complex decoder. Question: can ten compatible vertex-table
updates propagate derivatives from multilevel latents through their changing
geometry at257 and1025 control resolutions? This is a benchmark of the same
mechanism, not a new registration objective or a trained image encoder.
At nested coefficient levels17,33,65,129,257 (continuing to513,1025 for a
1025-control benchmark), let z_l contain two scalar interior
coefficient tables. Pad their boundaries with zero and prolong RAW proposals,
then apply horizontal and vertical analytic coordinated candidates sequentially
on the SAME material grid. Every successful intermediate output undergoes its
actual four-corner checks; Section6 gives the compatible P1 interpretation.
The final output is the final vertex table with the original fixed P1 diagonal,
not a resampled unrelated composition. Changing mapped geometry is an INPUT to
each subsequent stage, so its full derivative must remain connected.

Probe proposals use bounded deterministic synthetic latents, not anatomy or
evaluation targets. Their amplitudes and coefficient sizes are recorded. The
benchmark gradient is that of a declared synthetic scalar output probe with
respect to ALL latent tables and the initial map, not an image-training claim.
Compare ordinary autodiff, explicit manual adjoints and checkpointed explicit
adjoints. Checkpointing means recomputing selected forward blocks during backward,
not truncating their derivatives or freezing intermediate mapped vertices.
Compilation/cache/setup cost, warmup, successful/rejected stages, minimum actual
corner values, full forward/backward time and total/resident allocated memory
are separate measurements. Falsifiers: mismatched output or full-chain derivative,
rounded-output rejection, or no practical memory benefit after recomputation cost.
Start with tiny finite differences and two-stage equality before large grids.
No fixed-depth universality, image-to-latent training, or independent anatomical
registration conclusion follows from this benchmark.

Control resolution is NOT latent freedom: if every raw proposal is interpolated
on the FIXED source grid and each stage uses ONE spatially uniform scale, the
final displacement is a sum of those source-basis fields. For nested bilinear
coefficient grids all coarse spaces lie in the finest coefficient space.
Repeating such stages does not itself introduce new spatial basis functions.
Therefore the main dense-cascade comparison includes a coefficient level equal
to control resolution; a257-latent/1025-control auxiliary must be labeled limited
latent freedom, not generic1025 expressivity. Full-grid finite-depth universality
still does not follow, even when the finest coefficients have full dimension.

Checkpoint recomputation uses explicit use_reentrant=False and fixed, nonrandom
forward blocks, passing changing tensors as inputs rather than mutable loop/global
state. It trades computation for retained activations; see the runtime-version
[PyTorch2.5 checkpoint documentation](https://docs.pytorch.org/docs/2.5/checkpoint.html).

## 18. Actual nested control meshes with one fixed fine-grid objective

Research card (T+6.4h). Question: can actual coarse-control stages reduce complete
registration time without replacing the declared fine-grid objective? The present
baseline maintains the final control mesh at every coefficient level. The new
variant instead optimizes an actual n_l by n_l vertex table, then exactly refines
its fixed-diagonal P1 function before the next level. All levels are nested:
(n_next-1) must be divisible by (n_current-1), ending at the declared final N.
Only one increasing cycle and global radial/analytic stages are supported first.

Let P_l be exact diagonal-aware P1 prolongation from level l to the final mesh.
The objective is J_l(Y_l)=J_final(P_l Y_l), INCLUDING image sampling, frozen
point matches, strain, corner-shape and outside-image penalties. Hence the
gradient is P_l^T grad J_final, not a coarse-grid surrogate or a detached fine
map. Image-resolution continuation remains the existing declared schedule;
within each such resolution the fine material-grid functional is unchanged.
The geometry operator uses the current actual mesh. Actual rounded fine corners
are also checked on every trial; accepted boundaries and exported binary maps
retain their existing checks. Any global best iterate is stored at FINAL size,
even when it arose at a coarse stage. Refinement never coarsens or changes the
diagonal. This is a different feasible-subspace trajectory, not the old bilinear
proposal family evaluated more quickly and not a classic multigrid convergence
theorem. Section10 gives the exact-refinement argument; an independent checker
confirmed nesting, all-four-corner inheritance, adjoint and output-selection scope.

Falsifiers: function/gradient/individual objective discrepancies, rounded fine
violations, early best-output export at the wrong size, or no complete-call
benefit. Tiny tests precede real-data comparisons. Cache setup, all fixed buffers
and refinement/check time count. Fine evidence and its regularizers remain
O(N^2); reduced geometry/control work does not imply a cheaper whole algorithm.
Prior work: nested P1 finite-element subspaces and subspace correction, with the
additional digital four-corner feasibility and actual-rounded-output checks here.

## 19. Stage refresh versus more inner optimization at equal gradient budget

Research card (T+6.7h). Question: does the remaining real-registration error
reflect the single-cycle/30-inner-step budget, rather than conservative geometry
scaling? In the calibrated analytic development runs every proposal had scale1.
This observation excludes that particular guard as the active bottleneck along
those trajectories, not on arbitrary latent inputs. Compare (a) one coordinate
cycle with60 inner steps per stage and (b) two coordinate cycles with30 steps.
Both have600 gradient evaluations across levels17,33,65,129,257 and x/y stages;
the established one-cycle30-step baseline has300. Repeating the cycle returns
to coarse proposal spaces after the fine correction, without coarsening the
actual257 control mesh. Image continuation repeats its declared32..512 schedule.

For F1/F2 joint-vector controls, use twice as many cycles to keep gradient counts
equal (one joint stage instead of two scalar directional stages per level).
Pass counts, accepted-anchor refresh counts, total objective evaluations and
runtime are NOT equal merely because gradients are equal; report actual costs.
All evidence, initialization, fine-grid priors, boundary, physical learning rates,
best-full selection and manual evaluation denominator stay fixed. The stage cache
is used only for its supported radial/analytic cases; it changes engineering
cost, not the function family. No labels enter the optimization or stopping.

Decision-changing outcomes: more inner steps helping but extra cycles not helping
indicates different behavior from improved anchor refresh; neither helping means
do not keep increasing the same budget blindly. Proxy improvement with worse
anatomy remains an adverse outcome, not success. This is a development-only
convergence/optimization diagnostic, not an independent-cohort experiment, and
does not establish statistical superiority from repeated specimens.

Follow-up research card: the first two-cycle experiment does NOT improve the
best-full radial/analytic output on any development case: selection remains
stage9 of the first cycle. Recorded first second-cycle analytic coarse stages
increase full objective from.168639 to.176216, .308407 to.310256, and.217678
to.222312, while optimizing their declared32-pixel continuation objective.
All scales remain1. This motivates ONE targeted schedule intervention: keep
the original32..512 image continuation during cycle0, but use the same FULL512
evidence in every subsequent coefficient-level stage. Geometry remains fine257,
and the total budget remains600. This tests coarse coefficient correction on
the actual final objective rather than repeatedly restarting the coarse image
surrogate. Default repeated continuation remains available and unchanged.
Predicted falsifier: full-objective stages also fail to improve or increase
anatomical error despite proxy gains; do not call this guaranteed convergence.

## 20. Candidate exact coarse quadrature for the unchanged fine-grid priors

Research card (T+7h proposal; independently checked and dyadic implementation
benchmarked at T+7.4h, see PROGRESS and nested_priors_dyadic_gpu7.json).
Question: can the fine-grid prior of an EXACT uniformly refined P1 function be
evaluated and differentiated from its coarse vertices, instead of retaining the
entire fine-grid autograd graph? This could address the negative timing result
of Section18 without silently substituting a different regularization functional.
Scope: square coarse size n+1, unchanged global AC or BD diagonal, integer factor
m, fine size N+1 with N=mn, unit rectangle, exact arithmetic, shared fixed boundary.
Finite-precision materialization still needs fresh actual-output checks; algebraic
quadrature equivalence is NOT bitwise equivalence of rounded fine vertices.

For a mapped coarse cell a,b,c,d in row-major geometry, define four Jacobians
J1=[n(b-a),n(d-a)], J2=[n(b-a),n(c-b)],
J3=[n(c-d),n(c-b)], J4=[n(c-d),n(d-a)].
Let phi(J)=||J||_F^2+||J^-1||_F^2-4. For AC refinement there are m diagonal
fine cells, each a translated/scaled copy of the coarse mapped quad, and
m(m-1)/2 cells lying in EACH of its two affine triangles. The proposed exact
mean over all four fine corners WITHIN this old cell is therefore

    ((m-1)/(2m)) [phi(J2)+phi(J4)]
      + (1/(4m)) sum_{k=1}^4 phi(Jk).

For BD, replace J2,J4 by J1,J3. Averaging this quantity over coarse cells gives
the proposed same four-corner fine-grid shape prior. For m=1 it reduces to the
original coarse four-corner average; for growing m it approaches the equal-area
P1 two-face average. It is NOT the coarse Q1 integral or a newly chosen SLIM loss.

For the fine forward-edge strain prior, let R_ij=Y_ij-X_ij on the coarse mesh.
Define DxR=n(R_i,j+1-R_i,j) and DyR=n(R_i+1,j-R_i,j). Fine horizontal edges on
coarse horizontal boundary lines have slope DxR; in each coarse cell, its m-1
interior fine rows contribute m(m-1)/2 copies of each of the lower/upper slopes.
Each coarse horizontal edge thus receives multiplicity w_i=m(m+1)/2 at the two
outer coarse rows, and w_i=m^2 at interior coarse rows. Vertical edges use the
same weights w_j at outer/interior columns. Proposed fine-grid strain is

    [ sum_{batch,i,j} w_i ||DxR_ij||^2
      + sum_{batch,i,j} w_j ||DyR_ij||^2 ] / [2 B N(N+1)].

This formula is independent of AC versus BD after the symmetric edge counts.
Constant affine residuals recover the same strain since the total weight in
each orientation is N(N+1). No sparse inverse is involved. Identity/reference
rounding must be controlled (main dyadic meshes have exactly represented source
coordinates); non-dyadic/floating inputs do not automatically have exact bitwise
agreement. Image/match functions can also query the ORIGINAL coarse P1 map
directly, since exact P1 refinement preserves the function; their finite-arithmetic
query values still need measured comparison, not an equivalence assertion.

Falsifiers: independent cell counts fail, random legal AC/BD values or full
directional derivatives disagree with explicit refined priors beyond rounding,
or the reduced graph gives no complete-instance benefit after fresh fine checks.
Small decisive tests: affine residual, one nonaffine convex quad, random legal
coarse grids, factors1/2/3/4, both diagonals, gradients/FD and near-small-margin
fixtures. No implementation is authorized by the formula alone: obtain independent
derivation/check first, then profile bottlenecks and integrate only if justified.
Prior work: exact integration/assembly in nested finite-element spaces; the
particular four-corner and edge-multiplicity expressions here are proposed
derivations for the existing discrete objective, not claims of novel general FEM.

## 21. Reduced evaluation without changing the declared fine objective

Let P map coarse mapped vertices Y to their exact nested P1 fine vertex table;
let S_c and S_f evaluate the corresponding coarse/fine P1 functions at the SAME
fixed pixel-center or frozen image-match queries. In real arithmetic,

    S_f P Y = S_c Y.

This requires a nested triangulation with one unchanged AC/BD diagonal. It is
not true for replacing P1 by bilinear Q1 or changing the source triangulation.
The frozen affine, original moving features, interpolation of image brightness,
foreground denominator, OOB penalty, match points/confidences/robust canvas-pixel
scale and loss weights are unchanged. Image matches continue to use
pixel_scale*A(mapped(q)-p)/robust_scale, with NO added affine offset and NO change
of pixel_scale to the control-side count. The Section20 weighted quadrature then
provides the SAME fine strain/shape priors, not ordinary coarse-grid priors.

Write E_f(PY)=E_reduced(Y). Its full coarse-Y gradient is, where differentiable,
P^T grad E_f(PY), equivalently direct AD through the reduced graph. No field
gradient is detached; no sparse solve or inverse is introduced. The image
sampler is only piecewise differentiable. At exact bilinear image knots central
finite differences can average different branches, while AD selects a branch;
the retained knot fixture checks equal selected gradients, NOT a nonexistent
unique classical derivative. Knot-free fixtures separately check directional FD.

Floating arithmetic introduces different query/scatter/reduction rounding, so
we do NOT assert bit-identical E_f and E_reduced. Every trial materializes PY
without retaining its gradient graph and checks ALL four actual rounded fine
corner margins, with the SAME minimum eta. Before accepting a reduced winner,
evaluate the ORIGINAL complete fine functional at the current STAGE image
resolution on both anchor and candidate; retain the anchor if it worsens or is
nonfinite. Full-resolution best_full selection remains original E_f as before.
This costs two extra full objective calls per stage and is counted in runtime;
it protects acceptance, not equivalence of entire optimizer trajectories.

Optional --nested-evaluation coarse_exact is restricted to the existing
single-cycle nested_p1 path and supported dyadic references. full_fine remains
the unchanged default. Feature/mask/match objects are shared immutable inputs;
coarse source-query/count buffers are separately constructed and counted. An
already identical source-query cache is reused instead of duplicated. The
ordinary full Evidence remains authoritative for accepted/final reporting.
Actual application time/peak memory must be measured: fewer prior graph entries
alone do not prove a faster registration, especially at fine257.

## 22. Joint optimization of two sequential feasible coordinate updates

This tests the OPTIMIZATION organization, not a newly asserted representation
class. Let U_e(Y,p) denote the earlier radial/analytic safe single-direction
update with fixed residual boundary. During one stage the accepted anchor Y0
is fixed, and the jointly trainable variables are two interior coefficient
tables z_x,z_y on one source coefficient level. Let P_l pad their boundaries
with zero and interpolate their RAW scalar amplitudes to the current material
mesh (it does not interpolate a finished deformation). Define

    Y1 = U_x(Y0,P_l z_x),
    Y2 = U_y(Y1,P_l z_y),
    minimize_{z_x,z_y} E(Y2).

The second operator's constraints use Y1, NOT the old anchor Y0. Its geometry
cannot be cached as a constant or detached from z_x. Each actual rounded
substep margin and the final exported grid are checked. In exact arithmetic,
strict feasibility of both substeps yields a valid final vertex table on the
SAME original material triangulation. No unrelated regular-grid map is composed
and then resampled. This remains a per-instance optimization experiment with
frozen accepted stage anchors, NOT backpropagation through the optimizer history.

If g=grad_{Y2} E, the complete first-order derivatives are

    grad_{z_y} E = P_l^T (D_p U_y)^T g,
    grad_{z_x} E = P_l^T (D_p U_x)^T (D_Y U_y)^T g.

The full intermediate-Y adjoint is essential. Ordinary AD through both layers
is the oracle. Optional cached_manual caches ONLY the first constant geometry
and uses the existing explicit full-Y adjoint for the second layer. That API
supports first derivatives only; diagnostics are detached and computed by an
EXTRA no-grad second pass, whose time/pass count is explicitly reported.
No speedup of that backend is assumed from smaller graph storage.

The bounded application comparison uses joint5levels*60inner steps=300gradient
evaluations versus alternating5levels*2axes*30inner steps=300. These are NOT
equal layer-pass counts or equal compute: joint evaluates TWO feasible layers
per trial and updates TWO fields per gradient. Both include the final post-Adam
trial; joint has305trial objective evaluations versus alternating310. Report
actual wall time/memory and substep counts, not an unsupported fair-time claim.
Adam, physical edge-calibrated rates, original evidence/initialization, objective
weights, accepted-map refresh POLICY and output selection are unchanged. The
joint schedule refreshes five anchors rather than ten, an explicit part of this
optimization organization rather than an otherwise identical iteration sequence. Improvement
of image proxy alone cannot establish anatomical improvement.

## 23. The coordinated safety bound is about gradients, not absolute grid spacing

Here t ranges over ALL FOUR constrained corner triangles per quadrilateral,
not just the two faces of the declared P1 diagonal. For t=(i,j,k), define source
edge matrix B_t=[X_j-X_i,X_k-X_i], mapped edge matrix D_t=[Y_j-Y_i,Y_k-Y_i],
J_t=D_t B_t^{-1}, and scalar source gradient
w_t=B_t^{-T}[u_j-u_i,u_k-u_i]^T. Thus q_t=det D_t, qref_t=det B_t and
Q_t=q_t/qref_t=det J_t. All are positive for the current admissible map.
The common-unit-direction update gives exactly

    J_t^+ = J_t + e w_t^T,
    q_t^+ = q_t (1 + w_t^T J_t^{-1} e).

This is the rank-one determinant identity, not inversion of a global system.
Define v_t=J_t^{-T}w_t, the gradient of the scalar amplitude on the CURRENT
mapped auxiliary triangle. It is a per-triangle quantity; we do not pretend
the overlapping four auxiliary triangles define a single additional global mesh.
With eta>=0 and Q_t>eta, the earlier gauge can be written exactly

    g(u)=max_t [Q_t/(Q_t-eta)] (-v_t dot e)_+.

If Q_t>=rho>eta for every constrained triangle and ||e||=1, then

    g(u) <= [rho/(rho-eta)] max_t ||v_t||.

No explicit h occurs. Large amplitudes coordinated over a broad current-space
region can have small gradients; incoherent neighboring amplitudes, an O(h)-wide
boundary transition, or compressed J_t can instead make this bound grow like
1/h. Therefore a coherent proposal is NOT automatically safe, and fine-mesh
resolution-independent expressivity/depth is NOT established by the inequality.
Finite reachability along an ASSUMED strictly feasible path is a separate
conditional assertion, not connectivity of the whole digital feasible set.

The present optimizer uses Adam step .004*16/(level-1) in normalized residual
coordinate units under lr_calibration=edge. This is a chosen optimization scale,
not required by the above decoder. The existing physical option keeps .004
across coefficient levels. At257/1025 these differ by16x/64x, respectively.
Adam approximately cancels CONSTANT gradient-magnitude rescaling; it does not
cancel its explicit learning rate, nor do momentum/epsilon/curvature/support
variations disappear. A bounded calibration ablation can diagnose this choice.
Inactive observed analytic scales only mean their OBSERVED proposals were not
scaled; they do not show that larger proposals would be unrestricted. A257
ablation alone cannot explain a1025 change that also changes the parameter space
and image identifiability. All original geometry checks remain unchanged.

## 24. Explicit conditioning of raw proposals (bounded experiment)

For an M-by-N interior coefficient array z, with zero values outside that
interior rectangle, define

    (Kz)_ij = .5 z_ij + .125(z_(i-1,j)+z_(i+1,j)+z_(i,j-1)+z_(i,j+1)).

The physical proposal is p=P_l K^r z, where P_l first pads a zero coefficient
boundary then interpolates RAW amplitudes to the material vertices. The existing
safe operator computes U_e(Y,p). Neither Y nor its accepted deformation is
filtered. The boundary and topology argument are therefore unchanged.
This costs r local stencil passes and requires no global solve or inverse.
For upstream derivative v at p, the coefficient derivative is

    grad_z L = K^r P_l^T v.

Here K=I-L_D/8 for the unscaled Dirichlet five-point Laplacian L_D. Its sine
eigenvalues are .5+.25*cos(k*pi/(M+1))+.25*cos(l*pi/(N+1)), k=1..M,l=1..N,
strictly in(0,1). Thus K is symmetric SPD, and K^r is algebraically full rank.
This does NOT imply practical range/conditioning equality: for M=N=255,
lambda_min(K)~3.76e-5 and lambda_min(K^4)~2e-18. Bounded coefficients, floating
arithmetic and finite optimization budgets effectively restrict high frequencies.
With plain gradient descent, u=K^r z evolves through K^(2r)-preconditioned
gradients; Adam further changes this relation. This is an optimization
parameterization, not mere visualization or a map-repair operation.

Away from boundaries, r=4 has approximately one coefficient-cell per-axis
standard deviation, not scale-independent physical width. Zero ghosts attenuate
constant fields near boundaries. It need not reduce the distorted-mesh gauge
g(p), because J_t, slack and boundary transitions remain. The predefined test
uses r=4 only at levels129/257, against r=0 with physical Adam rate .004,
unchanged three development cases,300 gradients and all original objectives.
Report raw/filtered amplitudes and accepted displacement as well as gauge,
accuracy, time and memory; no anatomy improvement follows from this derivation.

## 25. A conventional rotation-aware elastic-prior ablation

This is an objective change, not a new homeomorphism construction. The source
rectangle and its fixed declared P1 triangles remain unchanged. For each actual
face t=(i,j,k), source edge matrix B_t and mapped edge matrix D_t define

\[
J_t=D_tB_t^{-1},\qquad
\phi(J)=\min_{R\in SO(2)}\|J-R\|_F^2,\qquad
S_{\rm ARAP}(Y)=\frac{1}{2A}\sum_t |t|\phi(J_t),
\quad A=\sum_t|t|.
\]

Here SO(2) is the set of two-dimensional orientation-preserving rotation
matrices, |t| is the SOURCE triangle area, and batch outputs are averaged over
maps. Uniform rectangular cells give equal source triangle areas, so this is
one half the mean over the TWO actual P1 faces per cell. Do not average all four
alternative corner Jacobians or weight by their deformed areas.
On a uniform grid with spacings hx=1/(columns-1), hy=1/(rows-1), cyclic cell
vertices a,b,c,d give AC Jacobians [(b-a)/hx,(c-b)/hy] and
[(c-d)/hx,(d-a)/hy]; BD gives [(b-a)/hx,(d-a)/hy] and
[(c-d)/hx,(c-b)/hy]. These formulas apply to the unit-rectangle material grid,
not arbitrary curved/general reference meshes.

For J=[[J00,J01],[J10,J11]], let a=J00+J11, b=J10-J01 and s=sqrt(a^2+b^2).
A rotation of angle theta satisfies tr(R^T J)=a*cos(theta)+b*sin(theta), whose
unique maximizer when s>0 gives

\[
R_*(J)=\frac1s\begin{pmatrix}a&-b\\b&a\end{pmatrix},\qquad
\phi(J)=\|J\|_F^2+2-2s.
\]

Since s^2=||J||_F^2+2detJ>0 for detJ>0, this formula is smooth on the valid
domain, including identity and repeated singular values. Implementation uses
||J-R_*||_F^2 directly to avoid cancellation near identity, with ordinary AD
through R_*; no global solve or SVD derivative is required. The scalar energy
has matrix gradient 2(J-R_*) (the minimizing rotation's variation cancels by
stationarity). Full-Y gradients also apply the actual source-face edge adjoint.

The old affine displacement-gradient prior is .5||J-I||_F^2. Near identity,
.5phi(I+H)=.5||symH||_F^2+O(||H||^3); the new prior deliberately removes the
infinitesimal rotational cost. Keeping coefficient3 does NOT make the priors
equivalent. A spatially varying rotation and its transition generally cost
stretch energy. For J=epsilon*I, .5phi(J)=(1-epsilon)^2 stays finite at collapse:
ARAP is NOT a flip barrier or a topology certificate. All four-corner safe
updates, eta, reciprocal shape penalty, boundary and binary checks stay intact.

Nor is ARAP convex or uniformly well-conditioned on positive determinants.
For J(t)=sI+t*e1*e2^T with s>0 (e1,e2 are Cartesian unit vectors), detJ=s^2
for all t, yet .5phi(J(t))=s^2+.5t^2+1-sqrt(4s^2+t^2), with second derivative
at t=0 equal to1-1/(2s), negative for s<.5. Thus even rank-one convexity can
fail under compression. The retained shape term is a safeguard, not a proved
convexification at its chosen weight. Rotation invariance alone cannot establish
convergence or good image correspondence.

Bounded application: choose displacement_gradient (default) or p1_arap as the
strain model, keeping weight3, MIND, original frozen points, shape1e-4, affine,
edge-calibrated Adam,300gradients and best_full policy. Use ordinary fine P1
evaluation, not the previously derived exact coarse membrane quadrature: its
formula is NOT an ARAP quadrature. Known Q1-generating truth remains unchanged;
report its raw-raster/query error separately from the same vertex table evaluated
by the declared P1 functional. Ground truth is never an optimization input.

Prior art: this is the established ARAP distortion measure, not a novel energy
or the local/global SLIM solver; see [SLIM2017, equations1--2](https://igl.ethz.ch/projects/slim/SLIM2017.pdf).
Its use here is motivated by a measured registration-objective conflict and
does not itself establish better anatomical correspondence or convergence.

## 26. Descriptor transport versus recomputation after image warping

Let If,Im be the immutable fixed/moving raster arrays at the current pyramid
resolution. Pixel centers in an H-by-W raster are x_ij=((j+.5)/W,(i+.5)/H).
For the actual declared P1 map and frozen affine, F_Y(x)=A I_hY(x)+b. Define
W_Y I as bilinear sampling of the ORIGINAL array I at F_Y(x_ij), with zero
padding outside its raster and align_corners=False. No image from a previous
stage is recursively warped. The geometry remains f64 when image arithmetic
is f32; query conversion changes raster precision, not the stored nodal map.

The EXISTING eight-channel self-similarity operator Phi is defined here to
avoid confusion with a faithful implementation of published MIND. For pixel p
and offset r in {(2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2)}, let

\[
D_r(I,p)=\operatorname{mean}_{s\in\{-1,0,1\}^2\,:,p+s\text{ in raster}}
 [I(p+s)-I_{\rm replicate}(p+s+r)]^2,
\quad V(I,p)=\tfrac18\sum_r D_r(I,p),
\]
\[
\Phi_r(I,p)=\exp\!\left[-\frac{D_r(I,p)-\min_vD_v(I,p)}{V(I,p)+10^{-4}}\right].
\]

Replicate extension applies to shifted intensities. The patch average excludes
out-of-raster patch locations, matching count_include_pad=False. The stabilizer
is fixed, not an adaptive function of the map. For fixed nonnegative tissue
mask m with denominator sum_p m_p (unchanged across iterates), compare

\[
E_{\rm transport}(Y)=\frac{\sum_p m_p\,\tfrac18\sum_r
 |\Phi_r(If,p)-(W_Y\Phi_r(Im))_p|}{\sum_p m_p},
\]
\[
E_{\rm after}(Y)=\frac{\sum_p m_p\,\tfrac18\sum_r
 |\Phi_r(If,p)-\Phi_r(W_YIm,p)|}{\sum_p m_p}.
\]

These are different objectives; their numerical values or iteration decreases
must not be compared as if they were the same functional. Initialization,
frozen machine matches, mask, explicit OOB penalty, ARAP3, shape1e-4 and geometry
are otherwise fixed in the bounded ablation. after_warp caches only Phi(If)
and the original moving raster, NEVER the candidate's Phi(W_YIm).

For pixel offset r define normalized offset r_n=(r_x/W,r_y/H).
Noncommutation follows from the neighborhoods: F_Y(x+r_n) generally differs
from F_Y(x)+r_n, so sampling a descriptor built in moving-image coordinates is
not the same as building one in fixed-image coordinates after deformation.
Even a rotation changes the ordered offset directions. If, in a noiseless
same-texture discrete model, If=W_Ytrue Im exactly, E_after(Ytrue)=0 by the
operator definition. This is a consistency property, not uniqueness of the
optimizer, anatomical validity, multimodal invariance or a general accuracy
theorem. Quantization, interpolation, different stains and tissue differences
break that equality. Resampling/texture loss can also create spurious minima.

For an image-term upstream derivative v at the candidate descriptor, the
chain is (D_YF)^T (D_FW_YIm)^T (D_Phi(W_YIm))^T v. Ordinary AD includes
descriptor normalization/minimum, pooling, exponential and sampling; no factor
is detached. Absolute values, channel minima and bilinear knots make the
objective piecewise smooth, so central finite differences are compared only
away from those knots. Safety remains the separate exact coordinated operator.
Recomputation costs additional local raster operations and their graph memory;
forward/VJP time and peak memory must be measured rather than inferred.

Prior art for neighborhood self-similarity:
[Heinrich et al., MIND2012](https://pubmed.ncbi.nlm.nih.gov/22722056/).
This experiment uses our existing MIND-like descriptor and makes no novelty
claim for feature recomputation or image registration by itself.

## 27. Exact ARAP reduction on nested P1 grids

Let the material domain be the unit square, with n-by-n equal cells and a
single declared global AC or BD diagonal. Let Y be the (n+1)^2 coarse mapped
vertices, and let P_mY denote the same continuous P1 function evaluated at the
uniformly refined (mn+1)^2 vertices, for a positive integer subdivision factor m.
Our implementation restricts n and mn to dyadic cell counts. P_m is a fixed
linear operator determined by material coordinates, not by mapped geometry.
It is exact P1 refinement, not four-corner bilinear averaging.

For a material triangle t with source vertices x0,x1,x2 and mapped vertices
y0,y1,y2, its constant deformation Jacobian is

    J_t=[y1-y0, y2-y0] [x1-x0, x2-x0]^{-1}.

Every coarse triangle has m^2 fine child triangles. Each child lies inside its
parent triangle and inherits exactly J_t. This includes the two triangles in
each fine square intersected by a coarse diagonal; treating those squares as
one affine quadrilateral would be wrong. Therefore, with B the batch size and
T=2n^2 triangles per batch element, our actual-face ARAP energy satisfies

    S_n(Y) = (1/(2BT)) sum_{b,t} min_{R in SO(2)} ||J_{bt}-R||_F^2,
    S_mn(P_mY) = S_n(Y).

Each child has 1/m^2 parent material area and all coarse source areas are equal,
so this is also the material-area average. There is NO m-dependent ARAP weight.
The closest-rotation definition and positive-determinant assumptions are those
of Section25. ARAP itself is not a fold barrier or a convex functional.

Differentiating on the positive-determinant domain gives the ALL-vertex identity

    P_m^T grad S_mn(P_mY) = grad S_n(Y).

No changing map, nearest rotation or safety scale is detached. This statement
includes boundary derivatives as a mathematical operator test, even when the
instance optimization fixes boundary values. It does not differentiate through
the full sequence of instance-optimizer iterations.

The shape penalty uses four corner Jacobians per quadrilateral, not just the
two actual P1 faces. Write phi(J)=||J||_F^2+||J^{-1}||_F^2-4, and let J1--J4 be
the four corner Jacobians defined earlier. In a coarse cell, m refined cells
cross its diagonal and are homothetic coarse quadrilaterals. The other m(m-1)
fine cells are affine, split equally between the two parent triangles. Thus
the EXACT fine-cell averaged shape contribution is

    ((m-1)/(2m)) [phi(Ji)+phi(Jj)] + (1/(4m)) sum_{k=1}^4 phi(Jk),

with (i,j)=(2,4) for AC and (1,3) for BD. An ordinary coarse four-corner mean
does not equal this expression unless m=1 or special affine geometry holds.
The existing count quadrature remains in use; only its strain branch changes
when strain_model="p1_arap" is selected explicitly.

Coarse image queries are exact evaluations of the SAME nested P1 function at
the SAME fixed image/query coordinates. Frozen matches, foreground weights,
normalization, descriptor and out-of-bounds penalties are shared unchanged.
Consequently the complete real-arithmetic reduced objective equals the full
fine objective restricted to the nested coarse P1 subspace. The next finer
level expands that subspace; it does not introduce extra image observations.

Floating-point refinement/subtraction and mixed raster-coordinate casts need
not give bitwise-equal values or optimizer trajectories. Every trial still
materializes and checks the actual fine-grid corners. Before accepting a stage,
the original full fine objective is recomputed and a spurious reduced winner
is rejected. Final map selection and saved topology certification likewise use
the actual fine representation. The reduction is an objective/VJP acceleration,
not a replacement topology certificate or an accuracy theorem.

## 28. Native-detail image scaling and comparable coordinate units

Original image coordinates x=(x1,x2) locate pixel centers at integer indices.
For a stored two-axis resize scale s, padding p and square canvas side S, the
normalized canvas coordinate is u=((x+1/2)*s+p)/S, componentwise. Rendering an
original JPEG at integer-multiplied resized dimensions, padding and canvas side
gives s'=ms, p'=mp, S'=mS, hence u'=u. In canvas pixel-center indices c=Su-1/2,
however, c'=mc+(m-1)/2; simply multiplying c by m would shift the frame.

The frozen normalized positive affine F=A f_Y+b and machine source/target points
are unchanged. The matcher is NOT rerun; records retain prediction_side=512
and explicitly describe transport to the new raster frame. Original RGB pixels
are directly resized using PIL BILINEAR; old512 PNGs are not enlarged. Native
source dimensions must be at least the requested resized dimensions on both
axes. The available lesions JPEGs fail this condition at1024 and are excluded
from native-detail validation, not silently upsampled.

For a machine-point residual delta in normalized aligned coordinates, our
robust penalty depends on r=(S/kappa) A delta. Maintaining its physical scale
requires kappa'=m kappa:1024/16=512/8. Foreground masking and all required manual
evaluation landmarks are retained. Manual labels enter only separate scoring.

For normalized moving-space prediction p and target v, canvas TRE is
S||p-v||_2. Report512-equivalentTRE=512||p-v||_2 alongside the actual canvas
number. Native-moving-pixel TRE inverts the stored moving-image two-axis scale:
|| (S(p-v))/s_m ||_2. With exact layout scaling this native TRE is unchanged for
the same prediction; it must not be pooled across unrelated native image scales.

The1024 experiment changes original raster detail, query count, physical size of
the fixed-pixel descriptor stencil and the32--512 versus64--1024 continuation
rasters. It is NOT an isolated proof that detail alone caused an improvement.
The first matrix has257^2 CONTROLS and1024^2 image QUERIES; neither number may be
substituted for the other. Analytic/radial updates and F1/F2 share these inputs,
affine, point evidence, ARAP3, shape1e-4 and300-gradient budget. Their geometry
pass counts differ and optimizer-only timing excludes image/matcher setup.

## 29. Four-pass fine-level coordinated patch support

This variant changes only the support of the final scalar-coordinate update;
global coarse stages and the complete image/prior/point objective remain fixed.
It is not a projection onto a repaired map and does not solve a linear system.

Let P>=2 be an even integer number of material grid cells. In one pass, complete
P-by-P cell patches start at (offset_row+kP,offset_column+lP). Each patch has
(P+1)^2 vertices. Only its (P-1)^2 strictly interior vertices move. For local
vertex indices r,c in{0,...,P}, define

    w(r,c)=sin^2(pi*r/P) sin^2(pi*c/P),

with perimeter entries set EXACTLY to zero in floating point. A raw physical
scalar proposal p_i is supplied for every vertex, with one common direction e.
The patch applies the existing exact common-direction operator to w_i p_i.
For its CURRENT mapped geometry Y, unchanged material reference determinants
q_ref, and normalized slack s=q(Y)/q_ref-eta, define

    g_patch=max(0, max_t [-(C_Y(w*p))_t/q_ref,t]/s_t).
    sigma_radial=1/(1+g_patch),
    sigma_analytic=min(1,theta/g_patch), with sigma=1 when g_patch=0.

Here t includes all four corner constraints of every patch cell, and theta=.95
in the current instance experiment. C_Y is the exact area derivative along e,
not a linearization that drops quadratic terms: they vanish because every
vertex in the substep moves parallel to e. Different patches use different
sigma. Patch boundaries and all vertices outside complete patches stay fixed.
Within a pass, affected cells and moved interiors are disjoint, so the separate
patch constraints are sufficient without an omitted cross-patch triangle.

Set Y^(0)=Y. The four sequential passes use offsets

    (0,0), (P/2,0), (0,P/2), (P/2,P/2).

Each pass uses the SAME raw p and direction e, but its gauge and scale are
recomputed from Y^(k-1). The output is Y^(4). There is no interpolation between
accepted vertex tables. Equivalently, each intermediate change is affine on
the CURRENT mapped triangles and its composition with the old map remains P1
on the SAME original material triangles. This is not arbitrary composition on
unrelated control grids followed by vertex resampling.

Starting from a fixed-boundary homeomorphism, unchanged patch perimeters plus
positive triangle orientations preserve the global P1 homeomorphism at each
pass. All four cell corners preserve either declared diagonal and the separate
Q1 interpretation, but these functions are not identical. In code, every
rounded intermediate margin is recorded as pass_margin_min[b,k]. The instance
application rejects a trial if ANY of these is nonpositive or nonfinite,
even when the final map would be positive. Actual fine-map checking, complete
fine-objective acceptance and saved binary-sign certification remain separate.

Complete-patch coverage is not universal. For N cells along an axis, the union
of interior vertex indices reached by offsets0,P/2 is1,...,E-1, where
E=(P/2)*floor(N/(P/2)). Full global interior coverage holds exactly when N is
divisible byP/2; otherwise trailing indices E,...,N-1 stay frozen. The1025
experiment has N=1024,P=32, so all interior vertices are reached. At least2P
cells along both axes are required by this implementation.

Complementary half-shifted sin-squared windows add to one where both offset
families exist. Thus the sum of all four 2D window weights is one in the
untruncated interior and at most one near boundaries. For the default trial1,
every sigma<=1, and the total displacement at i is

    Y_i^(4)-Y_i = p_i e sum_k sigma_patch(i,k) w_i^(k).

Its magnitude is at most |p_i|; no divide-by-four amplitude correction is
needed. The MINIMUM patch sigma is not a uniform multiplier for the whole
map. Distant patches may move freely while the worst patch barely moves.
Near boundaries, missing windows attenuate the proposal. Window gradients
scale like |p|/(P*h); keeping P fixed shrinks physical support on refined
grids and can reintroduce local amplitude limitations.

Ordinary AD differentiates through all intermediate geometries and through
each direct reuse of p. For a terminal loss, the proposal VJP includes the
sum of four direct-use contributions plus their subsequent geometry chains;
detaching Y^(k) would omit terms. Maxima/ties and analytic clipping retain
piecewise differentiability/subgradient limitations, as in the base operator.
The full-Y and proposal gradients are separately finite-differenced at unique
active constraints. This is a local decoder derivative, not AD through every
historical Adam iteration of instance optimization.

With five coefficient levels and30steps per coordinate, the original run uses
310 candidate evaluations/300 objective gradients/310 coordinated substeps.
Replacing only the two final-level stages by this four-pass operator retains
310 candidate evaluations/300gradients but uses496 coordinated substeps.
Increased gathered patch storage, setup and connected-geometry graphs must be
measured. The thin-region example establishes reduced distant coupling only;
it does not prove better anatomy, universal expressivity or a speed advantage.

## 30. Candidate-only manual adjoint for the four-pass update

This is an implementation of the SAME map in Section29, not a new geometric
family or an approximate inverse. For a patch, write its windowed proposal as
v=W p, where W is the fixed diagonal window, and let q_j(Y) denote each actual
corner determinant. Its constant material reference determinant is q_ref,j.
Put s_j=q_j(Y)/q_ref,j-eta>0, delta_j=C_Y(v)_j/q_ref,j,
a_j=max(-delta_j,0), and g=max_j a_j/s_j. The forward map is

    T(Y,p)=Y+sigma(g) v e,
    sigma(g)=1/(1+g)                         [radial],
    sigma(g)=min(alpha_trial,theta/g)        [analytic; g=0 uses alpha_trial].

Here delta_j is exactly affine in the scalar displacement, not a first-order
approximation of a general two-coordinate deformation. At a unique active row
j, dg=da_j/s_j-a_j ds_j/s_j^2. For an exact maximum tie, the implementation
uses the same equally shared subgradient as torch.amax. The adjoint of s_j
uses the actual local edge determinant differential and the adjoint of delta_j
uses its actual three-vertex common-direction stencil. Rows with inactive
analytic clipping have zero scale derivative. Radial dsigma/dg=-sigma^2;
active analytic dsigma/dg=-sigma/g. Clamp and tie conventions match the ordinary
operator; the layer is piecewise differentiable, not everywhere smooth.

For an incoming coordinate covector ell_i in R^2, define beta=sum_i
ell_i dot (v_i e). Its gauge adjoint is beta dsigma/dg. Direct coordinate
adjoints start with ell_i for Y_i and sigma(ell_i dot e) for v_i, followed by
the active-row contributions above. The raw proposal receives W times the
v-adjoint. Both Y AND p derivatives are required: a latent-only derivative
would be insufficient for composing four changing geometries.

Let T_k be one complete gathered/scattered patch pass, with Jacobians A_k
=partial T_k/partial Y and B_k=partial T_k/partial p, and Y_k=T_k(Y_(k-1),p).
For a terminal loss L(Y_4), the exact first-order reverse recurrence is

    lambda_4=grad L(Y_4),
    grad_p L=sum_(k=1)^4 B_k^T lambda_k,
    lambda_(k-1)=A_k^T lambda_k,             k=4,3,2,1.

Ordinary torch gather/index_copy operations propagate the identity derivatives
at untouched vertices and accumulate contributions at shared perimeter copies.
No intermediate Y_k or direct reuse of p is detached. Each patch retains its
actual GLOBAL material-reference subarray, not a rescaled local unit square.

The optional public diagnostics API returns candidate, scale, gauge and actual
rounded patch margin from ONE decoder invocation. Only candidate supports the
custom first-order VJP. Scale, gauge, amplitude, alpha_max and reported margins
are numerical diagnostics, explicitly non-differentiable in manual mode.
Reference and alpha_trial must be constant; trainable versions are rejected.
Higher derivatives are unsupported. A loss that differentiates any auxiliary
diagnostic must use the ordinary backend. The original Tensor-only seven-input
API and the ordinary default remain available and unchanged.

This eliminates the ordinary graph through all corner rows, saving active-row
state in bounded local-stencil chunks for reverse evaluation. It does NOT imply
constant memory: exact ties may activate many rows, and patch coordinate gathers
and the four changing coordinate tables remain resident. Numerical whole-grid
margin diagnostics include uncovered tails and every intermediate pass; the
instance caller still rejects any nonfinite/nonpositive intermediate margin,
checks actual final fine geometry, and evaluates its unchanged full objective.
No safety condition is relaxed to obtain a lower memory measurement.

Independent focused checking covered139 cases; the root affected application
suite covered95. The preserved float32 seed297, radial amplitude.3 fixture has
a negative rounded fourth-pass margin even though preceding passes are valid:
ordinary and manual candidates agree and BOTH reject. Successful float32 VJP
comparisons use a separately stated amplitude.1, not a relaxed area tolerance.
Tests establish local derivative consistency, not real-data acceleration.

## 31. Same-map-class physical-fiber optimization diagnostic

This experiment is traditional per-instance optimization. It does not replace
the differentiable decoder by a newly claimed neural architecture, and does
not claim gradients through an L-BFGS solve. Its purpose is to distinguish a
combined parameterization/optimizer limitation from lack of dense resolution.

Fix one accepted incoming vertex table Ybar, one common coordinate direction
e=(1,0) or (0,1), the material reference, and the complete image objective E.
The independent variables are actual scalar interior nodal displacements u,
zero at every boundary vertex. The candidate is Y(u)=Ybar+u e on the SAME
material grid with the SAME P1 diagonal. Because all displacements are parallel,
every corner determinant is exactly affine in u. For each of the four rows j
per cell, define s0_j=q_j(Ybar)/q_ref,j-eta>0. There is a fixed linear operator A
given by the existing corner-change stencil, such that

    q_j(Y(u))/q_ref,j-eta = s0_j+(A u)_j.

This equality is in real arithmetic; actual rounded coordinates are checked
separately. A is applied by local stencils, not assembled or inverted.

The original analytic decoder at this anchor uses

    g(p)=max(0,max_j[-(A p)_j/s0_j]),
    u(p)=min(1,theta/g(p)) p,               theta=.95,

with scale1 when g=0. Positive homogeneity gives g(u(p))<=theta. Conversely,
any u with g(u)<=theta is obtained by taking p=u, because its scale is1.
Thus the decoder's reachable set is EXACTLY the closed convex set

    K_theta={u: u_boundary=0, theta*s0+A u>=0}.

At the final coefficient level, the raw interior coefficient grid equals the
control grid, so this statement is not restricted by coarse interpolation.
K_theta is smaller than the entire topology-feasible domain. Every point in it
has ordinary corner slack at least (1-theta)s0>0; optimizing a larger domain
would invalidate a claim that only the parameterization/optimizer changed.

When g(p)>theta, positive scalar t remaining on that active branch satisfies
u(tp)=theta p/g(p), independent of t. Its radial directional derivative is0.
L-BFGS on raw p would therefore inherit an exact null mode, in addition to
max-row switching. Physical-fiber optimization instead minimizes

    F(u)=E(Ybar+u e) over K_theta.

Its unconstrained gradient at any valid iterate is
grad_u F_i=e dot grad_(Y_i) E, with boundary entries0. No derivative through
a safety retraction is included in this objective gradient. This is correct
for the constrained instance problem, not a substitute for the decoder VJP.

Given a current strict-interior u and a proposed descent direction d, let
c=theta*s0+A u>0 and h=A d. The exact feasibility bound is

    alpha_max=min_(j:h_j<0) c_j/(-h_j),

or infinity if no row decreases. A trial alpha<=rho*alpha_max, rho=.99,
retains strict contracted feasibility. The bound is algebraic; there is no
geometric line search or projection. Objective-only Armijo trials test the
UNCHANGED complete F, with finite trial budgets. Every rounded candidate must
also pass its actual contracted and ordinary corner checks. A rounded geometry
failure is reported, not hidden by relaxing eta or repairing the map.

The initial inverse-Hessian scale is calibrated ONCE so the first physical
descent vector has the declared interior RMS. In the matched edge-rate setup,
that value is learning_rate*(levels[0]-1)/(final_control_side-1), exactly the
existing Adam nominal physical rate, NOT learning_rate/(final_control_side-1).
After a reliable positive-curvature secant pair, the usual scalar initial
metric for the two-loop L-BFGS recursion is (s dot y)/(y dot y), with
s=u_new-u_old and y=gradF_new-gradF_old. Store at most5pairs, skip unreliable
curvature, and fall back to a descent gradient direction when necessary.
Objective, coordinate, anchor or mesh changes clear history.

The controlled application comparison shares a saved 129-control prefix within
each final grid, then compares only the final x/y suffix: original analytic
latent Adam versus physical-fiber L-BFGS. Both retain the same native1024
evidence, frozen affine/machine matches, ARAP3, shape1e-4, boundaries and
best-full-objective output selection. Evaluation landmarks are loaded ONLY
after outputs are saved. Prefix preparation, suffix optimizer time, all
forward-only trials, actual gradient counts, memory and failures are reported.
The primary limit is30gradient evaluations per coordinate, not a promise that
each optimizer completes30successful moves. Six backtracks permit up to seven
objective trials per search including the initial trial; counts are explicit.

This two-arm experiment changes BOTH chart and optimizer. A gain does not
uniquely identify which caused it. It supplies no nonconvex global-convergence
theorem, and repeated tiny feasible steps do not prove constrained stationarity:
an unconstrained quasi-Newton direction can point out of an active face.
Moreover rho=.99 retains the STRICT INTERIOR of K_theta: a boundary optimum
is approached but not attained in finite exact-arithmetic steps. A negative
comparison cannot exclude such an optimum; this numerical interior restriction
must not be silently called identical to every finite decoder output.
Cross-grid prefixes need not be bitwise identical, and the new nested257 run
is not identical to the earlier fixed257-control, interpolated-coefficient run.

Prior-art attribution: limited-memory quasi-Newton optimization is classical;
see [Nocedal's author-hosted L-BFGS description and original references](https://users.iems.northwestern.edu/~nocedal/lbfgs.html).
Objective sufficient-decrease backtracking is classical;
see [Armijo1966, publisher-hosted original paper](https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-p.pdf).
Those unconstrained/smooth convergence results are NOT asserted for this
piecewise-smooth, geometry-restricted pathology objective with finite searches.
The experimental contribution under examination is practical convergence and
cost on the already-defined exact scalar fiber, not inventing L-BFGS or Armijo.

## 32. A bounded tangent-direction diagnostic at a contracted face

This section states the bounded experiment precisely; its actual negative
application outcome is recorded at the end, not hidden by a successful toy.
It keeps the scalar-fiber problem in Section31 unchanged. In particular, it
does not enlarge the feasible map class or silently reset its fixed anchor.

### 32.1 Why an unconstrained descent direction can stop prematurely

Let n be the number of interior control vertices and m the number of corner
constraints. Boundary amplitudes are identically zero and are not independent
variables. Write u in R^n for actual scalar displacement, g=grad F(u), and
c=theta*s0+A*u>0 for contracted slack. A maps amplitudes to ALL normalized
corner changes; each row has at most three nonzero nodal coefficients before
boundary elimination. For the triangle (i,j,k), those coefficients follow from

    (A*u)_t = [(u_j-u_i)det(e,Ybar_k-Ybar_i)
              +(u_k-u_i)det(Ybar_j-Ybar_i,e)] / q_ref,t.

The same formula is applied to each of the four declared cell triangles with
its correct orientation. Neither the material spacing nor an assumed identity
geometry may replace the actual frozen Ybar edges in this expression.

An unconstrained direction p=-H*g can satisfy g^T*p<0 while (A*p)_j<0 at a
row with tiny c_j. The exact alpha_max then becomes tiny. This is not evidence
that F is minimized over feasible directions. Other directions may move along
that face without decreasing its slack. In the actual failed1025 suffixes,
independent reconstruction found exactly one row per coordinate stage with
c_j/(theta*s0_j)<1e-6. The ordinary, uncontracted triangle slacks were still
positive. That observation motivates a deliberately small active-row test.

### 32.2 Constrained local quadratic model, not a map projection

H denotes the symmetric positive-definite inverse-Hessian APPROXIMATION from
the current reliable L-BFGS pairs. It is an operator: applying it to a vector
uses the two-loop recursion, not an n-by-n stored inverse. The positive scalar
initial metric and retained positive-curvature secants give a positive-definite
H in exact arithmetic. Actual finite arithmetic is checked separately.

Collect near-face rows of A into B, of size k-by-n. The intended direction is
the solution of the small-equality-constrained quadratic model

    minimize_d  g^T*d + (1/2)d^T*H^{-1}*d
    subject to B*d=0.

This equation defines a SEARCH DIRECTION. It does not project an illegal
output map into the feasible set and does not differentiate an optimization
layer. H^{-1} is used only in the mathematical definition, not computed.
Introducing an equality multiplier lambda gives the first-order equations

    H^{-1}*d + g + B^T*lambda = 0,
    B*d=0.

With p=-H*g, define the k-by-k Gram matrix G=B*H*B^T. Then

    G*lambda=B*p,
    d=p-H*B^T*lambda.

For independent rows, G is positive definite. For dependent rows, use its
Moore-Penrose pseudoinverse G^+: the unique symmetric matrix obtained by
inverting positive eigenvalues and retaining zero on its nullspace. In exact
arithmetic B*p is in range(G), so d=p-H*B^T*G^+*B*p satisfies B*d=0.
An eigenvalue cutoff in floating point is a numerical rank decision, not a
proof that an omitted physical constraint is irrelevant. Every original
corner constraint remains in the subsequent FULL feasibility check.

Because B*d=0, multiply the first-order equation by d^T to obtain

    g^T*d = -d^T*H^{-1}*d <= 0,

with strict inequality for a nonzero d. Thus the correction preserves descent
in exact arithmetic. For ONE active row a, it reduces to

    d=p-(H*a^T)*(a*p)/(a*H*a^T).

This needs one additional H action. With k rows it needs k such actions and
a k-by-k eigensolve. It still touches n-vectors; its work is not independent
of control-grid size. If the usual direction guard instead falls back to
p=-g, this derivation must use H=I consistently. Projecting -g using an old,
unrelated H would not justify the same identity. An arbitrary Euclidean
projection of an L-BFGS direction does not supply this proof either.

### 32.3 The bounded experimental rule and its limitations

The implementation under review uses an optional default-OFF tangent rescue.
Near faces satisfy c_j/(theta*s0_j)<=1e-6; this single threshold is declared
before runs and is not tuned using labels. If any near-face row decreases
along p, correction includes ALL near-face rows, not just the outward ones.
Otherwise correction is unnecessary. More than eight near-face rows stops
this bounded diagnostic. It does not automatically expand into a large QP.

The code must report nonfinite rows or H actions, rank decisions, tangent
residuals, finite descent failures and the eight-row limit. Its residual
threshold is only a check on the computed direction: it is NOT permission
to violate a triangle constraint. The corrected direction is passed through
the unchanged ALL-row algebraic alpha_max computation, fraction-to-boundary
.99, finite-budget Armijo test and strict ACTUAL rounded-map checks from
Section31. There is no inward repair or relaxed area threshold.

Equality tangents deliberately examine only a subset of the feasible cone.
They may freeze a row that should move inward, or discard a useful direction
when several rows are nearly active. Failure therefore is not constrained
stationarity, nor an impossibility theorem for dense registration. No smooth
global-convergence result is asserted for the piecewise-smooth image objective.

The decisive application uses ONLY the two failed H/K1025 cases. It reloads
their already saved129-control prefixes; no upstream stages are rerun. The
native1024 rasters, frozen affine, machine matches, regularization, fixed
boundary, P1(ac) interpolation and30-gradient-per-coordinate budgets remain
unchanged. Selection is by the SAME complete objective over initial/prefix
and accepted suffix candidates. Evaluation labels are loaded only after map
export. Counts include extra H actions, Gram solves, all objective/gradient
evaluations and rejected trials. Inherited baseline timing is historical,
not a newly measured warmed comparison; early stops are not speed wins.

The direction construction belongs to classical constrained quasi-Newton
methods. Relevant modern primary prior art includes
[Brust, Marcia, Petra and Saunders2022, linear-equality constrained limited-memory optimization](https://epubs.siam.org/doi/10.1137/21M1393819).
Their reduced compact representations and trust-region algorithms are not
implemented or benchmarked here; the citation prevents attributing the general
idea of equality-constrained limited-memory directions as our invention.

Actual two-case result: H/K1025 mean512-equivalent landmark errors .793427/
2.299255 remain worse than original analytic-Adam .781702/2.287898. The rescue
does reduce the previous early stopping: K finishes both30-gradient stages;
H completes26/30 and still exhausts objective backtracking in x. No geometry
rejections occur. Complete E also remains worse for both. This intervention
is retained as a diagnosis, not the recommended performance candidate.

## 33. Pixel-center observations can weakly see fine AC nodal modes

This is an observation about the LINEAR SOURCE MAP EVALUATOR, not a theorem
about the complete registration objective or anatomical accuracy. The same
P1 map still has to satisfy every digital corner and boundary condition.

Let the source grid have N+1 vertices per axis and let the raster have N
pixels per axis. Source vertices are at (j/N,i/N), while pixel centers are
at ((j+.5)/N,(i+.5)/N). In every source cell the pixel center has local
coordinates (xi,zeta)=(.5,.5). For the a--c diagonal, it lies on that shared
triangle edge, so BOTH one-sided P1 formulas yield exactly

    f_Y(pixel_center)=(Y_a+Y_c)/2.

This is not an interpolation ambiguity: the query is FIXED in the source
domain, and its derivative with respect to the vertex values is the same
linear average. Only the derivative with respect to a moving source query
could differ across the triangle edge; that is not the present derivative.

Consider one scalar boundary-zero nodal perturbation. The source sampling
operator splits into independent a--c diagonal chains. A chain with L
interior unknown amplitudes x_1,...,x_L has fixed endpoints x_0=x_(L+1)=0
and observations (x_i+x_(i+1))/2 for i=0,...,L. Its matrix S has two adjacent
1/2 entries per observation, with endpoint boundary columns eliminated.
Consequently S^T*S has diagonal1/2 and adjacent off-diagonal1/4. Its normalized
sine eigenvectors have eigenvalues

    lambda_k=1/2+(1/2)cos(k*pi/(L+1)),       k=1,...,L.

Every eigenvalue is positive, so there is NO exact unobserved mode under
these boundary conditions. The smallest is sin²(pi/[2(L+1)]), and the
largest is cos²(pi/[2(L+1)]). Their ratio is cot²(pi/[2(L+1)]), growing
quadratically with L. The weakest mode alternates in sign along the chain
with a slowly varying sine envelope; adjacent amplitudes nearly cancel.
The condition number of S itself is the SQUARE ROOT of this Gram ratio.
At1025 controls/1024 pixels the longest chain has L1023: Gram condition
424971.1792, sampling condition651.8981. These are dimensionless operator
quantities, not image errors or measured iteration counts.

For comparison only, evaluate FOUR source queries per cell at quarter and
three-quarter positions in each axis. In vertex order(a,b,c,d), their local
P1 evaluation matrix is

    M = [.75  0   .25  0 ;
         .25 .50  .25  0 ;
         .25  0   .75  0 ;
         .25  0   .25 .50].

M is invertible. The eigenvalues of M^T*M are .1909830056,.25,.25,1.3090169944.
Assemble all such cell observations into S2. Each INTERIOR node belongs to
four cells, boundary amplitudes remain zero, and every cell is included with
the same weight. If mu_min and mu_max are those local eigenvalue bounds,

    4*mu_min*||u||² <= ||S2*u||² <= 4*mu_max*||u||².

Thus this UNWEIGHTED source Gram has condition at most6.854101967, uniformly
in source-grid size. Averaging four samples per cell rescales both bounds
equally and does not change the ratio. This does not invent extra image
information: intensities/features at these queries would still be interpolated
from the original raster. It is an evaluation/quadrature change, not refinement
of observed tissue detail and not a topology construction.

Actual evaluator probes at17/65/257/1025 controls agree with the midpoint
formula exactly in the stated float64 dyadic fixtures. Independently assembled
small boundary-eliminated matrices agree with the chain spectrum. Quarter
sampling gives small-grid Gram ratios2.5494/3.8413 at5/9controls, versus
5.8284/25.2741 for midpoint observations. No giant matrix is assembled at1025.

Limitations are essential: the actual image objective multiplies source map
derivatives by moving feature gradients, includes a nonuniform foreground mask,
uses absolute descriptor differences, and adds ARAP, shape and machine points.
Flat image regions can have zero gradient; masks can remove observations;
regularizers can eliminate weak modes. None of the bounds above automatically
applies to that full Hessian. This observation alone does not explain real TRE
or justify replacing the main optimizer. A separate controlled image experiment
would be needed before claiming any practical benefit.

Numerical quadrature for finite-element image correspondence is established
prior art, not a new registration principle; see
[Pierre et al.2016, finite-element digital image correlation and quadrature](https://doi.org/10.1016/j.optlaseng.2015.07.008).
Our current diagnostic concerns a specific source-grid/pixel-center alignment,
not a reproduction or performance comparison with that paper.

## 34. Controlled image-term quadrature experiment

This experiment is proposed after the negative physical-fiber result. It does
not combine that optimizer with a new loss. It uses the original analytic
scalar latent decoder and Adam, changing only how its IMAGE discrepancy is
sampled at the final dense stage. Its purpose is to test practical relevance
of Section33, not assume that source-operator conditioning determines anatomy.

### 34.1 Original observations and proposed observations

Let the original fixed and moving rasters have W columns and H rows. Their
descriptor tensors Phi_f and Phi_m are computed ONCE by the same eight-offset
MIND-like procedure on those rasters. Each tensor entry is a vector in R^8
defined at its original pixel center. Write tilde-Phi for the bilinear
interpolant of that tensor with zero padding. This is not a descriptor
recomputed from an enlarged image, nor a learned replacement feature.

Pixel p=(i,j) has its original fixed tissue weight m_p in[0,1]. That weight is
constant throughout optimization. D=sum_p m_p>0. The original source query is
x_p=((j+.5)/W,(i+.5)/H). The full fixed-to-moving map is
F_Y(x)=A*f_Y(x)+b, with the same frozen positive affine and P1(ac) residual.
If ell(v,w)=(1/8)sum_(k=1)^8 |v_k-w_k|, the original image term is

    E_image,1(Y)=sum_p m_p*ell(Phi_f[p],tilde-Phi_m(F_Y(x_p))) / D.

The proposed source queries are x_(p,a,b)=((j+a)/W,(i+b)/H), where
a,b are each1/4 or3/4. The image term becomes

    E_image,4(Y)=sum_p m_p*sum_(a,b) ell(tilde-Phi_f(x_(p,a,b)),
                    tilde-Phi_m(F_Y(x_(p,a,b)))) / (4*D).

Each of a pixel's four observations gets the SAME original m_p; masks are not
interpolated or changed according to overlap. Zero padding is used for BOTH
descriptor interpolants, so identical feature tensors queried at identity
give zero discrepancy, including canvas edges. PyTorch sampling uses
align_corners=False and grid coordinate2*x-1: its stored entries are pixel
centers, not mesh endpoint samples. Vertex-to-source-query interpolation is
the separate FrozenP1Evaluator on the declared material triangles.

This is a composite SUBCELL MIDPOINT rule, not an asserted exact integral or
a Gauss quadrature theorem. It introduces no new observed tissue information.
Its descriptor interpolation changes the objective's discretization, and can
also change bias or smoothness. More queries alone are not better evidence.

### 34.2 Exactly what stays unchanged

Let P(Y) denote the unchanged ARAP, shape and machine-point regularization.
The two complete functionals in the comparison are

    E1(Y)=E_image,1(Y)+E_oob,center(Y)+P(Y),
    E4(Y)=E_image,4(Y)+E_oob,center(Y)+P(Y).

In particular, the ORIGINAL CENTER-based out-of-bounds penalty stays identical.
It is not moved to quarter queries in this ablation. Quarter-query outside
fractions can be diagnostic only. The regularity weights remain ARAP3 and
corner symmetric-Dirichlet1e-4; machine-match weight.1 uses the same frozen
points/confidences and pseudo-Huber16native pixels. Boundary, affine, four
digital corner constraints, margin.001 and analytic theta.95 are unchanged.
No latent Hessian preconditioner or new inverse solve is introduced.

Each arm starts from the identical saved129-control prefix and optimizes
only final1025-control x/y stages. Each stage starts zero scalar interior
coefficients, fixes its accepted anchor and uses original Adam30updates plus
the last evaluated candidate. Nominal physical coefficient rate is
.004*16/1024=6.25e-5. The two loss arms therefore change neither the map
decoder nor the optimizer/nominal gradient budget. Four image samples versus
one still cost different forward/backward work; this is reported explicitly.

The common pre-suffix candidate family is identity, the stored E1-best prefix
and the incoming129 prefix. Missing older prefix states are not invented as
E4-optimal candidates. Each arm adds its own accepted x/y candidates and
selects by its OWN complete functional over that declared family, never by
evaluation landmarks. After export, evaluate BOTH final maps under BOTH E1
and E4. Only within-functional differences are interpretable as loss gains;
an E4 value must not be compared numerically to an E1 value as one objective.

### 34.3 Evidence required before conclusions

Tiny checks must independently reproduce the four-point sum, verify full-Y
derivatives away from interpolation/absolute-value knots, and show unchanged
center-based OOB, priors and machine-point terms. Same-feature identity,
fixed denominators, invalid configurations and actual output failures remain
explicit. The first real comparison is ONLY H/K1025 from the saved prefixes,
with all77/69 manual IDs evaluated after outputs exist. No upstream rerun,
new network, GT deformation, evaluation-point fit or enlarged input image.

Record setup and resident caches, image interpolation counts, actual objective
and gradient counts, topology checks, cross-objective values, elapsed suffix
time and allocated peak. Cold single-run timings cannot establish a precise
speed ratio. If accuracy, tails or time-to-accuracy are adverse, preserve the
negative result and stop this variant; the source-Gram theorem is not a reason
to keep tuning it. An improvement on these viewed specimens would remain
development evidence, not independent patient generalization or official SOTA.

### 34.4 Actual paired dense result and decision

Both original1024raster/1025control cases completed. Each arm made60gradient
updates and73complete-objective evaluations; two additional cross evaluations
per arm give150completeE evaluations per case. There were no failed trials.
Both selected the accepted y-stage map, never a landmark-selected iterate.
All saved prefix/center/quarter maps have exactfixedboundaries and positive
four-corner certificates; independent actual-array review is recorded separately.

| Case | E1 map mean512eqTRE | E4 map mean512eqTRE | E1 map p90 | E4 map p90 | E1 suffix s | E4 suffix s |
|---|---:|---:|---:|---:|---:|---:|
| H, all77IDs | .7817015492 | .7831290582 | 1.5943500880 | 1.5796892244 | 3.3747 | 4.2300 |
| K, all69IDs | 2.2878981413 | 2.2926002741 | 4.2550082692 | 4.2702678071 | 3.4154 | 4.2860 |

TRE is Euclidean landmark error, measured first on1024canvas then multiplied
by512/1024. These are not officialmicrometre/rTRE scores. K fixed-only IDs70/71
remain disclosed; neither shared-ID denominator is reduced. Means worsen in
both cases; H p90 improves and K p90 worsens. Tail and mean are not interchangeable.

The 2x2 cross-functional comparisons are essential:

| Case | E1(center map) | E1(quarter map) | E4(center map) | E4(quarter map) |
|---|---:|---:|---:|---:|
| H | .1757646105 | .1763116963 | .1675794921 | .1672776439 |
| K | .2281917408 | .2293928716 | .2162816450 | .2158488187 |

Thus E4 improves its OWN functional and worsens E1. One cannot compare the
smaller E4 numerical scale with E1 as if the losses were the same. The new maps
have less corner symmetric-Dirichlet distortion (.0298623->.0213853 H,
.0733454->.0506143 K); minimum trial scaling increases (.1131->.1724 H,
.0806->.1229 K). Even these better optimization/geometry diagnostics do not
establish improved anatomy.

Independent actual-map minimum q/q_ref is.006902->.016084 for H and
.003750->.009489 for K. Bothcenter outputs reproduce the previous center
baseline BITWISE; bothsourceprefixes preserve all129oldnodes exactly. Literal
independent AC/frame calculations reproduce every landmark error within
1.14e-13pixel512. More positive area clearance still does not imply anatomy.

Additional quarter-cache storage is671088640bytes (640MiB), shared by neither
trial nor selection: two intentionally separate wrapper roles each own four
immutable quarter caches. Both are resident before BOTH arms for this paired
experiment. Absolute CUDA allocated peaks are2752487424bytes for E1 and
3020957696/3020924416bytes for E4 H/K; incremental peaks over reported resident
memory are1850296832bytes versus2101727744/2101694464bytes. These are NOT peaks
of an isolated old center-only application, nor total physical GPU consumption.
Setup/cross/export are excluded from suffix times and reported separately;
the center arm runs first, so no exact warmed speed ratio is claimed.

Each arm has75evaluations including its two cross calls. Moving descriptor
interpolation equivalents are75for E1 and300for E4; quarterfixed setup uses
eight interpolations. CenterOOB map queries do not sample moving descriptors.
The extra queries add arithmetic, not observed tissue information.

Decision: retain the original center functional as the mainline. End this
quadrature variant without a parameter/feature/preconditioner sweep. Section33
remains a valid SOURCE-OPERATOR result, but is not a practical anatomy theorem.

## 35. CPU full-current-map and proposal adjoint portability

This is a deployment measurement of the existing operator, not a new map class
or a new registration optimizer. It uses the same saved real257P1ac anchor as
the earlier GPU operator test, exactly refined to1025 without new image detail.
Geometry, proposal and upstream vectors are float64. No image loss, landmark,
optimizer history or neural encoder is involved in the timed call.

For the candidate operator R(Y,z), both Y and scalar proposal z require gradients.
The upstream vector w has the shape of R and is generated once with seed20261001,
then divided by the number of vertex-coordinate entries. The measured reverse
operation returns BOTH (D_Y R)^T w and (D_z R)^T w, not just the proposal part.
This full-current-map derivative is relevant to chaining decoder layers; it does
not demonstrate backpropagation through the entire per-case optimizer or a
trained registration network. Only first-order derivatives are supported by
the compact custom adjoint; auxiliary diagnostics are not differentiated.

Each method uses validate=True, so its forward includes actual rounded-margin
checks. Fresh independent corner recomputation follows timing. Test proposals
are z(x,y)=a*sin²(pi*x)*sin²(pi*y), with exactzero boundary. Cases include radial
a=.02, analytic active a=.02 and analytic inactive a=.0001. Reference, input,
proposal, random upstream and all method configurations are shared per case.

Element supplied128logicalCPUs; pre-job load was~2.1 with~918GiBavailable.
The job used ONLY2CPUthreads, CUDA_VISIBLE_DEVICES empty and existingTorch2.5.1.
Threewarmups and10measured repetitions followed correctness checks. Existing
ordinaryAD runs before compactAD; these are repeated operator medians, not an
ABBA hardware-comparison study or measured image-registration time-to-accuracy.

| Controls/batch/branch | Ordinary forward ms | Compact forward ms | Ordinary VJP ms | Compact VJP ms |
|---|---:|---:|---:|---:|
|257²/B1/radial|9.55|6.56|10.41|1.39|
|257²/B1/analytic active|9.50|6.47|11.42|1.33|
|257²/B1/analytic inactive|10.07|6.70|12.02|.98|
|257²/B4/radial|55.88|46.61|55.68|4.52|
|257²/B4/analytic active|50.26|47.28|56.42|4.74|
|257²/B4/analytic inactive|60.01|46.39|54.89|4.25|
|1025²/B1/radial|338.10|238.63|372.40|18.26|
|1025²/B1/analytic active|266.62|263.31|370.79|19.60|
|1025²/B1/analytic inactive|290.76|212.10|349.91|16.45|

All candidate values match exactly in these fixtures. Maximum relative-L2
fullY derivative discrepancy is3.65e-16 and proposal discrepancy1.76e-16.
Actual minimumcorner ratios at1025 are.009545(radial),.002590(activeanalytic)
and.009873(inactiveanalytic), all above the required.001. These are measured
nontrivial legal current maps, not identity-only latent tests.

Untimed saved-tensor hooks count unique retained candidate-graph storage,
including saved inputY/z; they do NOT count the process's total/RSS/peak memory.
At1025/B1 this storage is453017672/453017673bytes ordinary versus
25215058/25215025bytes compact, approximately432.031MiB versus24.047MiB.
At257/B1 it is~27.008MiB versus~1.512MiB. Active-row/tie-dependent storage
must not be advertised as the same constant for every possible input. No CPU
allocatedpeak was measured, and GPUmemory values cannot fill that missing metric.

The observed CPU reverse improvement is useful engineering evidence. Forward
still visits the complete geometry, and activeanalytic1025 forward changes
little. Nothing here establishes competitive anatomy, an end-to-end learned
network or a hardware-independent speed ratio. The existing benchmark's
historical caveat about concurrentGPU6/applicationintegration was inherited;
this actual job was CPU-only and current integration status is documented in
the application sections rather than inferred from that old caveat string.

## 36. Predeclared20-direction known-specimen instance comparison

### 36.1 Inputs, outputs and variables

The stain set is S={HE,CC10,CD31,Ki67,proSPC}. The case set is exactly
{(s,t):s,t in S,s!=t}, so there are20ORDEREDcases from ONE physical specimen.
All these annotations were viewed in prior research. The experiment examines
direction/stain failure coverage, not blind generalization or20patients.

For each orderedcase, fixed image I_s and moving image I_t are original512square
canvases. A_s,t and b_s,t are the already saved image-only positive affine,
not a fit to human landmarks. Every method starts from that same affine.
The safe methods output one residual table Y in R^(1x257x257x2), including
allboundary/corner vertices, plus frozenA/b and interpolation tag P1ac.
The full fixed-to-moving map is F(x)=A*f_Y(x)+b. Its material grid has66049
vertices and131072triangles; all262144digitalcorner signs are protected.
The residual boundary is EXACTidentity, not a trained free boundary.
The full map is a homeomorphism onto the affine image of the unitrectangle,
not necessarily onto the entire movingunitcanvas. Out-of-canvas imagequeries
are retained with the declared zero-padding/penalty; no canvas clipping is
claimed to preserve the geometric homeomorphism.

This is an INSTANCEOPTIMIZER experiment. No newCNN is trained. Inputs to the
local differentiable decoder are the accepted current map and per-case scalar
or vector coefficient fields. Image gradients optimize those coefficients;
human landmark coordinates are not provided to that decoder/optimizer.
The nativeDHR output instead is its existing saved512pixel-center displacement
field, with its own interpolation/frame declaration and no hardglobal certificate.

### 36.2 Frozen image evidence and identical safe-method functional

First perform20calls of the EXISTING frozen raw-confidence image matcher.
No geometricRANSAC filter is newly applied, no human targets are used, and no
confidence is invented for older selected-inlier archives. All matcher attempts
finish before any registrationmethod call. Model setup/inference is counted.
These learned pretrained features provide IMAGE evidence, not groundtruth.

For eachcase, the two safe methods share the complete functional

    E(Y)=E_MIND_transport(Y)+3*E_P1_ARAP(Y)
         +1e-4*E_corner_symmetric_Dirichlet(Y)+E_center_OOB(Y)
         +.1*E_frozen_machine_matches(Y).

Every term/coordinate convention is defined in the preceding sections. The
original fixed foreground mask and its constant denominator are retained.
The machine-point robustscale is8canvaspixels. No region is excluded because
its current candidate maps badly. NativeDHR uses its standard preprocessing,
NCC7/diffusion-like regularization and boundary freedom; it is therefore an
application baseline, not a same-functional geometry ablation.

### 36.3 Optimizer budgets and selection

Coefficientlevels are17,33,65,129,257; corresponding imagelevels are
32,64,128,256,512. The outputcontrol table is always257², not32²--512².
Eachstage fixes its accepted map as anchor, zeros its coefficients and makes
30Adamupdates plus the last evaluated trial. Coefficient rate atleveln is
.004*16/(n-1). Geometry isfloat64 and frozen rasterfeaturesfloat32.

Analytic performs onecycle of x/y stages, giving1*5*2*30=300gradients.
F2 performs twocycles of vectorstages, giving2*5*1*30=300gradients. Its existing
patchsize8, rawspan.5, safetyfraction.75 and accepted_gain1 are retained from
the actual512ARAP baseline configuration; gain1 is not confused with earlier
gain.75 diagnostic experiments. Geometricpass counts, channel counts and
wallclockcosts are different despite equalnominal gradient counts.

The label-free best_full rule selects among initial/acceptedstates by the
COMPLETE512functional. Imagecontinuation stage objectives can differ, so
acceptedstage values atdifferentimagelevels are not one monotone sequence.
Report actualgradient/failedtrial counts, validmap and completedbudget separately.
NativeDHR retains its existing5levels/30iterations each; its150updates are NOT
called300matched scalar/vector gradients. No output is post-hoc repaired.

### 36.4 Evaluation chronology and denominator

The scorer runs only after all20prediction attempts are terminal. It reads
the SAME80IDs for each of the five stain annotations. Nominal50pc-CSV to5pc-JPEG
conversion is (p+.5)/10-.5; that center-preserving convention carries subpixel
uncertainty and is not advertised as measured physicalmicrometres.
OriginalJPEG→canvas conversion uses the stored peraxis resizeandpadding frame.

For eachfixed landmark p and corresponding movinglandmark p', evaluate the
declared FULLmap at the sourceunit query q. Error is Euclidean distance in
512canvaspixels, and separately in originalmoving5pcJPEGpixels. P1evaluation
uses independent NumPy barycentric matrix solves. Nativepixel displacement
is converted with its exported unit-resample/unpaddedframe; its stored affine
is already in the field and is NOT applied a secondtime.

Everydirectionreports meanTRE,90thpercentileTRE,maxTRE and all80values.
Equal-direction aggregates include meanof20directionmeans, meanof20direction
p90values, p90ofdirectionmeans and worstlandmarkerror. The second and third
quantities are different statistics; neither pools1600correlated observations
as independentpatients or claims an official challenge score.

Any failed/skipped/invalidexport retains its direction in denominator20.
If a method lacks anydirection score, its all20numeric aggregate is unavailable;
conditional successful-direction diagnostics are explicitly marked. Failed
methods are not silently replaced with the affine. Nativefoldedcases keep
their scores, with their negativecorner counts reported separately. Native
localcorner signs do not prove a simpleboundary/globalhomeomorphism.

The scorer verifies actualstoredmap geometry and commonaffines, not every
optimizerintermediate or the historical absence of priorlabelaccess. Chronology
is supported by the separate predictor and its completion/nolabel declarations;
known-specimen status remains explicit. The current experiment does not reuse
the three-case lesion78ID evaluation denominator as if it were this80ID source.

### 36.5 Actual complete cohort results and an important budget confound

All20 directions produced all three requested outputs; every score includes all80
IDs. The primary denominator remains20, including the native folded maps and
the two F2 runs whose gradient budgets did not finish. These are distinct events:
successful file export, validity of the exported geometry, and completion of
the requested optimizer budget must not be conflated.

|Method|Mean of direction mean TRE|Mean of direction p90 TRE|Worst direction mean TRE|Mean complete call (s)|CUDA allocated peak range (MiB)|
|---|---:|---:|---:|---:|---:|
|Common frozen affine|6.66122|12.23468|11.76354|Not timed here|Not measured|
|Analytic coordinated|4.52023|9.55521|7.76521|4.79099|214.442--218.436|
|F2|4.59946|9.70151|8.57700|15.07797|472.522--479.319|
|Native DHR|5.13359|11.55283|11.98744|0.72903|69.184|

TRE units in this table are512canvaspixels, not micrometres. Each direction has
equal weight, and these directions share one specimen. The actual sequential
cohort is not an ABBA hardware experiment. Timings exclude the separate raw
matcher calls (sum7.49196s, mean0.37460s); matcher model setup is included in
those calls. Peaks are reported CUDA allocated peaks, not total physical memory,
CPU resident memory or a simultaneously deployed learned network's footprint.
Analytic mean inner optimization is4.68476s,97.8%of its complete call.
Analytic is therefore about6.57times SLOWER than native DHR in this execution;
the current result is not a native-baseline speed win.

Analytic improves meanTRE over affine on20/20directions and p90 on19/20.
Against native DHR it improves mean on14/20 and p90 on16/20, with adverse cases
retained. Native DHR has196916nonpositive local digital corners summed over
all20outputs; every direction has at least one. These local counts do not
establish a global boundary certificate and do not remove native scores.

Analytic completed300gradients with0failedtrials on every direction. F2
completed300on18directions. HE-to-Ki67 completed242with2rejected trials;
Ki67-to-HE completed158with5. The rejected losses/coordinates were finite:
their extra normalized margin above eta=.001 was approximately-5.54e-16
to-1.49e-14. This is a roundoff-scale violation of the requested extra floor,
not a negative-orientation exported map. The existing guard was retained;
neither case was silently restarted with a relaxed floor or a new gain.

To identify the confound, the descriptive common complete-budget18subset has
meanTRE4.1795866 analytic versus4.1850395 F2, an advantage of only0.005453px.
Its mean-p90 is9.0243651 analytic versus9.0171700 F2: analytic is0.007195px
worse. The two incomplete F2 cases explain93.8%of the primary mean advantage
and104.4%of the primary tail advantage. This subset is supplementary, NOT a
replacement denominator. Current evidence supports analytic's budget reliability
and observed speed relative to F2, not substantial intrinsic anatomical
superiority on cases where both finished.

An independent checker recalculated every reported per-direction statistic and
cohort aggregate from stored per-ID errors, agreeing within1.25e-14. Literal
raw-map/frame checks are separately recorded in PROGRESS when complete.
No blind independent-specimen generalization, full image-to-latent training,
clinical validity or official challenge competitiveness follows from this cohort.

## 37. F2 strict-floor reserve: what the correction does and does not prove

### 37.1 A reproduced mismatch, not a negative-orientation claim

Consider one patch update on a uniform grid of side n and cell width h=1/(n-1).
Its outer vertices remain fixed; its interior vertices receive proposed vectors
v_i. For every protected digital corner, its unnormalized determinant along
the simultaneous motion is the quadratic polynomial

    q(t)=q0+t*L+t²*Q,  0<=t<=1.

Here q0 is the determinant of the current corner, L is the sum of the two
cross-products containing one current edge and one proposed edge difference,
and Q is the cross-product of the two proposed edge differences. The configured
floor is f=eta*h². The input assumption is q0>f for every affected corner.
Define negative part x_minus=max(-x,0), and B=L_minus+Q_minus. Because t²<=t,

    q(t)>=q0-t*B.

The historical F2 allowance is A_old=min(s*q0,q0-f), with total-area safety
fraction s=.75. Its scale can use ALL q0-f when the second branch controls.
Thus its exact-arithmetic floor conclusion is q(t)>=f, not strictly q(t)>f.
This differs from the instance optimizer's actual strict extra-floor test.

A concrete boundary-fixed3x3 example uses center(.001,.5), rawspan.5,
centerlogit(-1,0), eta=.001 and accepted_gain1. The current minimum normalized
determinant is.002; the historical pass returns center(.0005,.5), whose minimum
is exactly.001. No face has negative area. The input/output contrast exposes
the strictness mismatch without relying on a complex optimizer trajectory.

### 37.2 Optional strict reserve, with historical default preserved

Introduce a separately named floor_safety_fraction theta_f in(0,1]. Use

    A_new=min(s*q0,theta_f*(q0-f)).

The existing per-corner quotient divides A_new by max(B,A_new,guard), where
guard is the existing positive numerical denominator safeguard. For B>0 this
quotient is <=1 and its product with B is <=A_new. For B=0 take quotient1,
because neither polynomial coefficient is adverse. The patch scale sigma is
the minimum of all its corner quotients. The accepted gain gamma is in(0,1],
so the actual endpoint parameter t=gamma*sigma also satisfies these inequalities.
For theta_f<1,

    q(t)>=q0-t*B>=q0-theta_f*(q0-f)
          =f+(1-theta_f)*(q0-f)>f.

The same conclusion holds along the full accepted path. Fixed patch perimeters
and nonconflicting simultaneous patches retain the shared mesh/boundary; each
of the four sequential staggered passes uses its newly updated current map.
In exact arithmetic, four passes retain at least (1-theta_f)^4 of the input's
minimum normalized slack above eta. This lower bound is not a promised measure
of typical distortion or optimizer quality.

Historical default theta_f=1 preserves the original arithmetic/weak-floor
meaning. The predeclared new research variant uses theta_f=.95, the already
used analytic reserve, not a fraction fitted to landmark performance. Reports
must say which version ran. The original all20 cohort is not overwritten.

This is an exact-arithmetic strictness statement. A sub-ulp input slack can
still be lost to rounded coordinate addition or determinant evaluation. Keep
the actual rounded-output strict guard, boundary checks and exported-map
certificate unchanged. The denominator guard alone is NOT an endpoint rounding
certificate. No tolerance relaxation, new geometry line search or post-hoc
repair follows from this derivation. Local gradients are tested separately;
the minimum/max branches make the decoder piecewise differentiable, not smooth
at every tied active constraint.

## 38. Unchanged application profile: diagnosis, not a speedup result

Use the first predeclared all20 direction HE-to-CC10, its saved image-only affine
and its203 frozen raw machine correspondences. The input images, functional,
257² control mesh, coefficient/image levels,300gradient budget and actual-map
guards are unchanged. A process-local wrapper calls each original function
once inside a profiler annotation and restores it afterward, including inherited
methods. No human annotation, matcher rerun or optimizer replacement is used.

Three fresh registrations run in order: unprofiled before, CPU/CUDA profiled,
unprofiled after. Complete-call times are5.44376,9.39967,7.47894s; respective
inner optimization times are5.09,9.24,7.35s approximately. Allcomplete300gradients
with0failedtrials and identical evaluation counters. Boundary/affine/interpolation
archives match. Normal repeated maps differ at most4.91e-9 in unitcoordinates;
before-to-profile difference is7.52e-10. Final objective differences are order
1e-12; CUDA sampling backward is not claimed bitwise deterministic.

The profiler records190271 CUDA device events, including168105 kernel launches.
The sum of device event durations is939.648ms. This includes computation,
memcpy and memset, not compute-only time or floating-point operation count.
Separate CPU-linked operator self-device attribution sums940.034ms; it is NOT
added to the device-event sum. Kernel sums are not automatically GPU wall time
when execution overlaps. Summed CPU self-events are13266.1ms, larger than the
complete-call wall time because threads/annotation scopes can overlap.

|Profiler annotation|Calls|Inclusive CPU interval sum (ms)|
|---|---:|---:|
|Backward|300|4713.76|
|Complete evidence|332|2150.68|
|Decoder|310|1224.69|
|Machine-point prior|332|623.77|
|ARAP prior|332|415.41|
|Corner shape prior|333|374.52|
|Adam step|300|193.93|
|Frozen P1 map evaluation|332|121.78|
|Raster sampling|332|37.82|

These rows are NESTED and NONADDITIVE. A host annotation interval is not an
exclusive GPU phase duration. In particular, backward CPU self-time includes
dispatch/wait around autograd activity on other threads. CUDA launch host
self-events sum1539.61ms over168105calls. Scalar extraction has7161calls,
114.11ms inclusive; stream synchronization5971calls/28.48ms hostself, and
device synchronization926calls/17.27ms hostself. Waiting can represent genuine
device work, so not every synchronization duration is removable overhead.

The profiled complete call is1.455times the mean unprofiled bracket, and the
whole profiler session including startup/finalization is16.3213s. Before/after
normal times themselves differ by37.4%; this is NOT a precise isolated hardware
speed comparison. PyTorch warns that profiling adds operation overhead; the
deployed2.5.1 source distinguishes CPU-linked events from actual device events:
[profiler source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/autograd/profiler.py),
[event aggregation source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/autograd/profiler_util.py).

The diagnosis is a testable launch/autograd-granularity hypothesis, not proof
of a particular compiler's speedup. A bounded tensor-subgraph intervention must
retain the same mathematical functional and actual-map checks, test values and
VJPs, and measure cold compilation separately from cached execution. It cannot
use the0.94s event sum as a promised achievable complete-call time.

## 39. Joint-prior fusion operator and restricted application integration

The two unchanged functions are E_ARAP(Y), ONE HALF the P1 face-average squared
distance of its Jacobian to a proper rotation, and E_shape(Y), the existing four-corner
symmetric-Dirichlet average. Their definitions are given above. The isolated
closure returns BOTH scalar values. Its reverse test applies scalar seeds
(3,1e-4), so the compared full vertex derivative is

    G(Y)=d/dY [3*E_ARAP(Y)+1e-4*E_shape(Y)].

Inputs are actual saved float64 tables in R^(1x257x257x2), not image features
or a neural prediction. The first is predeclared HE-to-CC10; the second is the
smallest positive four-corner minimum among the19 other supplied analytic
outputs, HE-to-Ki67. Selection uses geometry only, not human landmarks. Their
minimum normalized corners are.294420 and.00655735. Three warmup pairs and ten
alternating eager/compiled timed pairs follow one observed cold pair per map.

The compiled closure uses existing PyTorch2.5.1 Inductor with default mode,
fullgraph=True and dynamic=False. It calls the ORIGINAL functions, with no new
epsilon, determinant clamp, rotation detachment or rewritten energy. Fullgraph
capture failure is an error, not silent eager fallback. Float64 remains unchanged.
No autotuning or CUDA-graph mode sweep was performed.

|Actual map|Eager forward (ms)|Compiled forward (ms)|Eager weighted VJP (ms)|Compiled weighted VJP (ms)|Eager F+VJP (ms)|Compiled F+VJP (ms)|
|---|---:|---:|---:|---:|---:|---:|
|HE-to-CC10|1.25777|0.44786|3.76298|0.78723|5.01571|1.22706|
|Geometry-selected HE-to-Ki67|1.26010|0.44991|3.70136|0.64621|4.96715|1.08166|

Columns are independently recomputed medians of ten stored synchronized samples.
Median F+VJP is computed from each paired sum, not necessarily the sum of the
two separate medians. All14 value/gradient comparisons per map pass: maximum
value difference6.11e-16 and full vertex-gradient difference9.77e-15.
This supports throughput of THIS operator, not a4times faster registration.
Fullfloat64 vertex derivatives are tested; higher-order derivatives are not.

The first compiled factory costs.6332s, first observed forward6.5341s and
first weighted reverse2.9703s. These costs are separate and additive observed
intervals, not isolated pure compiler time. The second map reuses code/disk
caches and is NOT an independent cold replicate. Input cloning, validation,
CPU output transfers and numerical comparisons are excluded from the operator
timing. The forward is gradient-enabled, not a no-gradient inference-only test.

Warm observed CUDA allocated peaks: forward30427648→14715904bytes and reverse
38816768→27282432bytes, eager→compiled. Phase baselines and increments are
recorded separately; the forward graph is already live at the reverse baseline.
These are neither CPU compiler memory nor whole-application memory claims.
Actual GPU7/A6000 resources were idle before the job; two CPU/compiler threads
were used, with research-local compiler caches and no package installation.

The restricted production optimizer now has an optional joint_prior_backend:
historical eager remains default; Inductor changes ONLY the joint ARAP/shape
dispatch. The complete original image/OOB/machine-point functional, coefficients,
weights, optimization and actual topology guards remain. Production rejects
non-AC interpolation, float32 geometry, absent shape weight, non-ARAP strain or
nested control hierarchy for this experiment. The generic Evidence class is
broader; these are restrictions of the production route, not a new theorem
about every possible class caller.

One prior-only callable is constructed and shared across the five image-level
Evidence objects, whose CONTROL shape is identical. It never caches an energy,
current vertex value or gradient. The initial constant-input path and the first
gradient-enabled trial may specialize differently. A complete first-call clock
starts before factory creation and counts both, followed by separately reported
warm ABBA applications. CPU eager-backend compiler-capture fixtures test dispatch
correctness only; they are not evidence for CPU or GPU Inductor speed.

### 39.1 Actual complete registration, not just an isolated prior benchmark

The input is the same frozen HE-to-CC10 image pair, positive affine and203
machine correspondences from section36. Each attempt starts with the identity
residual vertex table. The output is a fresh float64 table with257x257 control
vertices, declared P1-ac interpolation, and its unchanged affine. No saved map,
human annotation, matcher rerun or neural warm start enters any attempt.
The optimization uses the same300 coefficient-gradient steps,310 evaluated
trial maps and332 complete-objective evaluations. Only evaluation of the two
regularizers changes from separate eager operations to the joint compiled
closure. Every other image, point, boundary and feasibility operation is retained.

The measured order is one observed-cold eager call, one observed-cold compiled
call, then three groups E/C/C/E, where E means eager and C means compiled.
Thus each backend has six warm calls, and all14 attempts have distinct output
archives. GPU context initialization is separately recorded and excluded.
The synchronized complete-call clock begins BEFORE the compiler factory and
includes input loading, feature creation, initial no-gradient evaluation,
gradient-enabled trial evaluation/backward, optimization, saving and actual
saved-map certification. No application warmup is hidden before the cold pair.
Research-local cache paths were new for this application test; this does not
prove every system/global compiler cache was pristine.

|Measurement|Eager|Joint-prior Inductor|
|---|---:|---:|
|Observed cold complete call (s)|6.494887|13.935558|
|Median of six warm complete calls (s)|4.048639|2.950935|
|Warm reported peak CUDA allocation (MiB)|217.854|194.393|
|Cold reported peak CUDA allocation (MiB)|217.004|364.564|

The warm reduction is1-2.950935/4.048639=27.1129%, not the4x isolated-prior
speedup. The three ABBA groups' ratios of eager/compiled mean complete time
are1.3614,1.3828,1.4043. Every compiled warm call is faster than every eager
warm call in this bounded test. The allocator counter is reset inside the
optimizer after feature setup; it excludes complete compiler/host memory and
is not a process-RSS or physical-GPU-capacity measurement.

Independent recomputation reads all14 actual saved tables: all3,670,016
four-corner determinants exceed the same eta=.001 floor; the boundary is
exactly identity, the original affine remains positive and unchanged, and
the interpolation declaration is P1-ac. The normalized corner minimum spans
.2944199667463492 to.2944199667468752. All14 runs complete the full budget with
zero failed trials; no rejected/incomplete run was removed from timing.

Warm cross-backend coordinate differences have maximum1.21476e-8 in unit
coordinates and RMS8.13321e-11; same-backend repeat maxima are8.25076e-9 and
5.48869e-11. Cross differences are of the same order, but are NOT strictly
bounded by repeat variation. Final-objective differences are at most5.27e-12
cross-backend versus7.19e-12 within-backend; initial parts match and image terms
match. This supports negligible observed numerical change for this experiment,
not bitwise equivalence, equality of every future trajectory, higher-order
derivatives, or a clinical/anatomical benefit.

The cold compiled call is7.44067s slower. If all future calls cost the measured
warm medians, recovering this penalty would require ceil(7.44067/(4.04864-
2.95093))=7 additional warm calls, approximately8 calls total. This is a simple
amortization estimate, NOT a measured crossover or pure compilation cost.
Different control shapes, devices, guards or cache states may recompile.
The experiment demonstrates useful warm execution on ONE known development
pair/A6000, not all20 directions, unseen specimens or a trained neural encoder.

## 40. Completed corrected F2 comparison: equal gradient counts, not equal cost

The new cohort recomputes ONLY F2 with the section37 reserve theta_f=.95.
All20 directions use exactly the original images, affine, raw machine-point
records and300-gradient recipe. Analytic and native-DHR outputs are archive
references to section36, not rerun results. Original files/results are retained.
The new predictor reads no annotation; the unchanged production scorer reads
the same80 manual IDs per direction AFTER every prediction attempt is terminal.

Every new F2 run completes300 gradients with zero failed trials. All20 actual
float64257² P1-ac tables pass independent four-corner and boundary checks:
5,242,880 corner determinants exceed the unchanged eta=.001 floor, with minimum
normalized determinant .0077423580. Thus the historical two short-budget
directions no longer contaminate the nominal300-gradient baseline comparison.

|Method|Mean of20 direction mean TRE (px)|Mean of20 direction p90 TRE (px)|Mean recorded complete-call time (s)|
|---|---:|---:|---:|
|Common positive affine|6.661222|12.234678|not timed here|
|Analytic coordinated, archived|4.520228|9.555206|4.790985|
|F2 with strict reserve .95, newly computed|4.551344|9.657840|14.533357|
|Native DHR, archived|5.133593|11.552831|.729027|

TRE is the Euclidean distance between a mapped fixed landmark and the matching
moving landmark in512-canvas PIXELS, not micrometers. For each direction, mean
and p90 are computed over all80 IDs; displayed aggregates give equal weight to
the20 directions. The directions are correlated pairs of the SAME previously
viewed specimen, not20 independent subjects. Formal clinical or held-out
generalization conclusions are unavailable.

Analytic is better than corrected F2 in14/20 mean scores and12/20 p90 scores,
but its aggregate advantages are only .68% and1.06%. This does not establish
strong anatomical/optimizer superiority. Corrected F2 attains a lower FINAL
image-based objective in15/20 directions despite the slightly worse aggregate
anatomy scores. Lower registration objective is not equivalent to lower TRE.
Both safe methods improve mean TRE over affine in all20 directions, but both
worsen CC10-to-CD31 p90: analytic9.4164, F2 9.2982 versus affine8.6637.
F2 wins notably on Ki67-to-CD31 (mean5.0678 versus5.1906; p909.3569 versus
10.1032). Native DHR wins mean TRE in six directions, including all four
proSPC-source directions. These adverse cases are retained, not filtered.

The new F2 predictor's total elapsed time is290.810s; only new input checks,
F2 calls and manifest saving contribute. Historical matcher/A/DHR timing
fields are explicitly excluded. Archived analytic/new F2 mean times suggest
approximately3.03x throughput difference, but they are separate executions,
NOT contemporaneous paired timing. The section39.1 compiled-single-pair result
cannot be extrapolated into this all20 table. Native DHR remains faster, with
different evidence/objective/boundary and without the safe-output guarantee.

Independent literal CSV/frame/P1 interpolation of all20x80 IDs reproduces the
safe-map and affine scores within2.51e-13 pixels. Archived A/DHR score dictionaries
are unchanged. This checks numerical evaluation consistency, not historical
blindness, registration correctness at unlabelled tissue, or SOTA.

## 41. Frozen machine-point evaluation: sparse interpolation, not a frozen map

Each machine correspondence has a FIXED source coordinate q_j, an affine-aligned
moving target p_j and an unchanged confidence c_j. No quantity in this subsection
is a human evaluation landmark. For the fixed source P1 triangulation, locate
q_j once and let its triangle vertex indices be i_j1,i_j2,i_j3 with barycentric
weights w_j1,w_j2,w_j3. These weights sum to1 and depend on q_j/source grid,
NOT on the current deformed vertex table Y. Hence

    m_j(Y)=sum_(k=1)^3 w_jk Y_(i_jk).

The sparse linear interpolation operator can be called P, so m=PY. There are
only three nonzero interpolation weights per point. We store vertex indices
and weights, not a dense P matrix. Its derivative is exactly the same linear
operator; for cotangent g at the evaluated points the vertex VJP is P^T g,
implemented by weighted accumulation at the selected vertex indices.

Let S=512 be the full moving-canvas side, kappa=8 its robust scale in pixels,
and A the unchanged2x2 positive affine matrix. Define r_j=(S/kappa) A(m_j-p_j),
and normalized eligible confidence beta_j=c_j/sum_l c_l. The unchanged loss is

    E_match(Y,A)=sum_j beta_j (sqrt(1+||r_j||²)-1).

The implementation's cancellation-resistant penalty ||r||²/(sqrt(1+||r||²)+1)
is algebraically the same expression. A vertex cotangent is the weighted
scatter of beta_j*(S/kappa)*A^T r_j/sqrt(1+||r_j||²). The affine derivative
also remains connected. Targets, confidence, pixel units, static original-domain
eligibility and robust scale do NOT change. No deformed point location, energy,
gradient, map value or optimization history is cached.

The optional prepared sampler rejects a different control-grid shape, diagonal
or Q1 declaration; explicitly prepare again before such a change. Source buffers
are contractually immutable after preparation. This application enables the
cache only for fixed P1 controls with positive explicit match evidence, not the
changing nested-control route. Module.to may cast existing weights, which is
ordinary rounded conversion rather than recomputing query geometry at the new
precision. Source-coordinate gradients are intentionally unsupported because
the correspondence data are frozen.

Isolated actual-GPU benchmark input: the existing HE-to-CC10 float64257² map,
its original positive affine and203 raw-confidence points. Output: the original
scalar match loss and its FULL vertex/affine-matrix VJPs, not a new registration
or encoder. One first pair and three warmup pairs precede ten alternating timed
pairs. Input clones, output CPU copies and comparisons are excluded; separate
setup and loading costs remain recorded. The match outer weight .1 is not
applied in this isolated operator benchmark.

|Isolated operator|Existing point location|Frozen point indices/weights|
|---|---:|---:|
|Median gradient-enabled forward (ms)|.831952|.410646|
|Median full vertex and affine VJP (ms)|1.564415|1.020541|
|Median paired forward+VJP (ms)|2.394741|1.422012|

Setup costs .042511s and stores9744 bytes for203 points. All14 value/full-vertex/
affine-gradient comparisons are equal for this specific dyadic case; nonsquare
and edge fixtures allow normal rounding differences. The measured1.684x
operator ratio is NOT an application-speed claim. Optional application dispatch
preserves the historical existing sampler by default and shares ONE prepared
point object across all image resolutions; complete-call benefit is still to be
measured. This is standard P1 evaluation reuse, not a new geometry theorem.

## 42. One fixed-budget fine-stage allocation experiment

The optimization variable remains each stage's interior scalar coefficient
table. The same safe decoder and exact fine-grid constraints produce the trial
map; this section changes only the integer number of Adam updates per stage.
Coefficient sides are17,33,65,129,257 and image sides32,64,128,256,512.

For each level l there are two sequential x/y directional stages, each with
n_l gradients and n_l+1 trial evaluations. There is one cycle and a fresh fixed
accepted anchor per stage. Uniform n=(30,30,30,30,30) is compared against the
single predeclared allocation n=(30,20,20,30,50). Both satisfy

    gradients=2*sum_l n_l=300,
    stages=2*5=10,
    trials=2*sum_l(n_l+1)=310,
    complete-objective calls=trials+2*stages+2=332.

The last term counts initial/final plus the anchor/accepted evaluation of each
stage. Equal counts do not mean equal wall time: fine-resolution image/VJP
calls cost more. Physical learning rates, initialization, evidence, weights,
fixed257² output, extra eta floor and actual export checks remain identical.

The diagnosis motivating the test uses objective traces, not annotations:
16/20 directions never use scale<1; only257/6200 trials across four directions
are capped. All40 final fine stages still lower the full objective in their
last five trials, and all20 full-objective-selected maps are terminal stage9.
Thus a cohort-wide topological stall is not established. This does NOT prove
more fine iterations will improve anatomy or exclude other optimization issues.

Both allocation arms will be recomputed for all20 directions under the SAME
compiled-prior backend; cold costs and execution order are recorded. No human
landmark selects allocation, step count or stopping. Scoring occurs after both
complete prediction cohorts, with the same80-ID denominator. A lower objective
without improved TRE/tails is a negative accuracy result, not a successful
registration advance. No nearby allocation sweep or convergence theorem is
implied. The optional per-level budget defaults to the historical uniform
inner_steps setting; explicit uniform fixtures reproduce its map/objective.

### 42.1 Actual allocation result: proxy improves, anatomy does not

Both20-direction arms complete300 gradients,310 trials and332 objective calls
without failures. Independent reading of all40 maps verifies10,485,760 strict
four-corner constraints, exact fixed boundaries, common positive affines and
P1-ac interpolation. Literal scoring of every80-ID set agrees within2.54e-13
pixels. F2/DHR archive metrics stay unchanged.

|Allocation|Mean direction mean TRE (px)|Mean direction p90 TRE (px)|Median complete-call time (s)|
|---|---:|---:|---:|
|Uniform30/30/30/30/30|4.520228|9.555206|2.94133|
|Redistributed30/20/20/30/50|4.527272|9.578047|2.94025|

The redistributed full objective is LOWER in20/20 directions, with mean change
-.0008730, yet mean TRE improves in only5/20 directions and p90 in11/20.
This is a NEGATIVE anatomical result. Notable regressions: proSPC-to-HE mean
+ .07949px; HE-to-Ki67 p90+ .22767px. Ki67-to-HE mean improves by.06687px but
its p90 worsens by.16335px. Uniform output is retained; no neighbouring
allocation sweep follows. Lower objective alone does not validate registration.

First uniform call costs7.5554s and includes process/compiler initialization;
total mean times3.21456 versus2.94484s are therefore confounded. Excluding the
first pair gives paired19 means2.98610 versus2.94535s; near-equal medians and
no repeated allocation ABBA prevent a robust speed claim. Timing excludes
historical matcher/F2/DHR execution but includes current validation/output.

## 43. Actual complete-application frozen-point dispatch

Both arms use the SAME compiled joint priors from section39. Only the
machine-point evaluator changes, existing versus frozen indices/weights.
Each cold-plus-three-ABBA attempt starts a fresh identity residual, uses the
unchanged300-gradient HE-to-CC10 recipe and writes a separate P1-ac output.
Preparation happens exactly once per frozen run, zero times per existing run,
and its cost is INSIDE the synchronized complete-call clock.

Warm complete-call medians are2.944472 versus2.695455s: an8.4571% reduction.
Three ABBA ratios of mean times are1.09336,1.09127,1.08644, and every frozen
warm call is faster than every existing warm call. Independent14-table reading
checks3,670,016 strict corners, exact boundaries/affines and unchanged300/310/
332/0 counts. Cross-map maximum1.61410e-8 is below within-backend repeat
maximum1.78035e-8; full-objective differences4.68e-12 versus5.73e-12. Image
terms match. These are negligible observed changes, not universal bit equality.
Warm allocated peaks203853312 versus202764288bytes are whole-optimizer counters
with the previously stated feature/compiler-host exclusions, not graph-only or
RSS measurements.

Observed cold times7.52208 versus2.69539s do NOT measure a cold point-cache
advantage: the FIRST existing run performs initial prior compiler/runtime
setup, which the subsequent frozen run can reuse. Both share research compiler
caches, with no hidden warmup. Only warm ABBA supports point-dispatch speed.
The earlier prior27.1% gain and this8.46% gain were separate experiments;
their ratios are not a simultaneously measured combined all20 speedup.

## 44. Dense-image knockout: precisely what was changed and tested

This is an evidence ablation, not a new deformation representation. Let
Y contain the 257-by-257 residual vertices, and let f_Y be the fixed-source-grid
P1-ac interpolant. The complete fixed-to-moving map remains F_Y(x)=A f_Y(x)+b.
The frozen positive affine A,b, the exact identity residual boundary, and all
four corner constraints q_k(Y)/q_ref>eta with eta=.001 are unchanged.

Write the production full-resolution objective as

    E_beta(Y) = beta I(Y) + 3 S_ARAP(Y) + 1e-4 R_corner(Y)
                + .1 P_match(Y) + O(Y),   beta in {1,0}.

Here beta means ONLY the dense-image weight; the earlier section6 used beta
for the historical OOB weight, which here is fixed to1. The terms are:

- I: fixed-foreground-weighted mean absolute difference of the eight-channel
  original-raster MIND-like descriptors, transported through F_Y. Descriptor
  construction, pixel centers, zero padding, and fixed denominator are specified
  in sections6 and26. It is still computed and reported when beta=0.
- S_ARAP: one HALF of the material-area mean squared distance of each actual
  P1 triangle Jacobian from its closest proper rotation, as defined in section25.
  On this uniform mesh it is .5 times the mean over its131072 triangles.
- R_corner: the mean over262144 digital corners of
  ||J||_F^2+||J^{-1}||_F^2-4. This is the existing corner shape penalty, not
  an additional barrier or the primary topology guarantee.
- P_match: the original normalized-confidence machine-correspondence penalty
  from section9. With512canvas pixels and kappa=8pixels, its residual is
  (512/8) A(f_Y(q_j)-p_j), and its penalty is sqrt(1+||residual||^2)-1.
  Machine points are image-derived, frozen, imperfect, and NOT manual truth.
- O: the original fixed-mask mean squared out-of-bounds coordinate excess;
  no difficult pixel or manual evaluation point is discarded after deformation.

All terms and their gradients remain live except that beta=0 multiplies the
dense-image contribution by zero. For the SAME Y and evidence, mathematically
E_0(Y)=E_1(Y)-I(Y) and grad E_0=grad E_1-grad I. Focused tests check values
and full vertex VJPs, including nonzero retained-term gradients. The beta=1
branch preserves the historical arithmetic rather than introducing an extra
multiply-by-one. Tests do not establish anatomical correctness.

Both arms use the original uniform30updates at each of17/33/65/129/257
coefficient levels, scalar x/y stages, image sizes32/64/128/256/512, onecycle,
and fixed257control topology throughout:300gradient steps,310decoder trials,
332complete-objective evaluations and10stages. The beta0 arm uses the SAME
compiled prior backend and existing machine-point sampler as uniform_t14.
Each starts from the same affine plus identity residual, not the beta1 output.
Every arm selects among accepted maps using its own complete512objective E_beta.
Therefore final E0 and final E1 totals are NOT values of a common objective and
must not be ranked against one another. Anatomical TRE is scored afterwards.

### 44.1 Actual all20 result and decision

The beta0 run completes20/20 directions, each with300/310/332/0 counts
(the last number is failed trials). No matcher is rerun and no new manual point
enters prediction. The scorer retains all80shared IDs in every direction.
As before these20directions belong to ONE previously viewed physical specimen,
not20 independent patients, a blind test, or an official leaderboard submission.

|Dense weight|Mean direction mean TRE (512canvas px)|Mean direction p90 TRE (px)|
|---|---:|---:|
|beta=1, retained hybrid|4.520228|9.555206|
|beta=0, knockout|5.024223|10.020200|

Removing the dense term worsens mean TRE in ALL20directions; p90 improves
in only5/20. Largest mean regressions include proSPC-to-HE (+1.041997px),
proSPC-to-CC10 (+.841707px), andCC10-to-proSPC (+.767191px). Thus the
predeclared falsifier is met: reject beta0, retain the original hybrid, and do
NOT start an intermediate-image-weight sweep. This supports a useful dense
contribution in THIS finite-budget pipeline and prior balance; it does not
prove that MIND's global optimum is anatomical, diagnose every remaining error,
or establish that all other image objectives would be inferior.

Total new prediction time is63.7652s. The first complete call is7.45846s,
including first-process/compiler setup; complete-call median is2.95394s.
This is NOT a new speedup experiment: I remains computed, the compared beta1
cohort is historical, and no repeated paired timing design was used here.
Reported optimizer allocated peaks are202001408--206005248bytes, with existing
feature/setup/compiler-host exclusions. Exported minimum corner ratio is
.0073506453, strictly above eta. Independent generic triangle-affine scoring
of BOTH arms'40 maps and3200 query points agrees with all stored per-ID errors
within1.1413e-13px. Direct recomputation checks10,485,760 strict corners,
exact boundaries/common affines, all budgets, and unchanged F2/DHR artifacts.
Independent ARAP and corner-shape values agree within1.73e-18 and9.71e-16,
and each correctly weighted E_beta reconstruction within6.94e-18.

Files: tools/coordinated_dense_knockout_all20.py and
outputs/coordinated_instance_registration/lung_all20_dense_knockout_t14/
(predictions.json, landmark_scores.json, actual map/report pairs). Optional
nonfinite-return handling keeps the failed direction, records invalid diagnostic
paths, writes null instead of nonstandard NaN/Inf JSON, and continues the cohort;
it never converts a failed numerical result into a successful map.
