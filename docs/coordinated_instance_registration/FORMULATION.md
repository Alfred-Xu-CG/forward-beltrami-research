# Mathematical formulation of the implemented operators

Status: implementation/theory note, not a claim of registration superiority.
The first real experiments optimize each image pair's own coefficients. They do
not yet deliver a trained image-to-map network. Operator gradients are tested
locally; stages intentionally detach accepted anchors in instance optimization.

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
D(Ybar,P_ℓc), and minimizes the declared complete objective. Five updates plus
evaluation of the last update are included. The best legal full-objective
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
