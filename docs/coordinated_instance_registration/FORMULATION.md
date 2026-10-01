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
