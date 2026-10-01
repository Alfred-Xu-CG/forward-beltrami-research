# Mathematical formulation of the implemented operators

Status: implementation/theory note, not a claim of registration superiority.
The first real experiments optimize each image pair's own coefficients. They do
not yet deliver a trained image-to-map network. Operator gradients are tested
locally; stages intentionally detach accepted anchors in instance optimization.

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

For independent evaluation landmark pairs (p_f,p_m), convert native pixel centers
to saved canvas coordinates and compute TRE=||512 F_Y(p_f)−512 p_m||_2. Report
all shared IDs, pairmean,p90,max and native moving-pixel units separately. Do not
use landmarks to select iterates. Cases are development specimens, not independent
clinical validation. Time-to-accuracy is a joint result; no-folding alone is not
successful registration. Forward/objective,VJP and complete optimizer time, memory,
initialization/evidence/certification cost are distinguished.
