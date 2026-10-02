# Forward expressivity on a fixed rectangular digital mesh

This note is self-contained. It establishes an existence result for the
implemented coordinated forward updates; it does not claim convergence of
image registration, a fixed network depth, or a new classical embedding theorem.
The corner reduction and proof have been checked in a separate agent context.
All statements below are in exact real arithmetic unless explicitly qualified.

## 1. Mesh, boundary, and admissible outputs

Take positive integers m,n and source vertices

\[
X_{ij}=(j/m,i/n),\qquad 0\le j\le m,\quad 0\le i\le n.
\]

There are m by n **cells**, and (m+1)(n+1) vertices. The production 257-by-257
vertex grid has m=n=256; its four corners are included. Join horizontal and
vertical neighbors. For interpolation, also split each cell along the same
top-left to bottom-right diagonal. Connectivity never changes.

A vertex table Y assigns a vector Y_ij in R^2 to every vertex. Every boundary
vertex is fixed: Y_ij=X_ij when i is 0 or n, or j is 0 or m. Define
det(v,w)=v_x w_y-v_y w_x. In a cell, call the mapped vertices a,b,c,d in the
source order (top-left, top-right, bottom-right, bottom-left). Require

\[
q_1=\det(b-a,d-a)>0,\quad q_2=\det(b-a,c-b)>0,
\]
\[
q_3=\det(c-d,c-b)>0,\quad q_4=\det(c-d,d-a)>0.
\]

These are twice signed areas, not triangle areas. Define H_0 to be the set of
all such fixed-boundary tables. These inequalities mean each cell is strictly
convex with the declared cyclic orientation. They protect both diagonal choices,
not merely the two triangles of one P1 split. With the fixed ordered boundary,
the continuous, cellwise triangulated piecewise-affine map f_Y is a homeomorphism.
Here P1 means affine on each source triangle; a homeomorphism is a continuous
bijection with continuous inverse.

The numerical implementation uses the smaller set

\[
H_\eta=\{Y:q_k(Y)/q_{\rm ref}>\eta\ \hbox{for every cell and corner}\},
\quad q_{\rm ref}=1/(mn),\quad \eta=.001.
\]

Do not confuse H_0 and H_eta. The theorem proved here concerns H_0. A separate
qualification in Section 6 describes when the chosen positive floor is preserved.

## 2. A positive local barycentric representation of every table

Fix an interior vertex i, now using i as a single vertex label. Enumerate its
four horizontal/vertical neighbors j=1,2,3,4 cyclically, with j+4 identified
with j. Set

\[
v_j=Y_j-Y_i,\quad r_j=\|v_j\|,\quad e_j=v_j/r_j.
\]

Let alpha_j be the positive angle from e_j to e_{j+1}. Four-corner positivity
implies 0<alpha_j<pi. The angles sum to 2*pi: their sum is a positive integer
multiple of 2*pi but strictly less than 4*pi. Every r_j is positive.

Define directed, positive mean-value coefficients and their row normalization:

\[
w_{ij}=\frac{\tan(\alpha_{j-1}/2)+\tan(\alpha_j/2)}{r_j}>0,
\qquad \lambda_{ij}=\frac{w_{ij}}{\sum_{k=1}^4w_{ik}}.
\]

They sum to one and satisfy Y_i=sum_j lambda_ij Y_j. For a direct verification,
let J(x,y)=(-y,x), the counterclockwise quarter-turn. Then

\[
\tan(\alpha_j/2)(e_j+e_{j+1})=J(e_j-e_{j+1}).
\]

Summing over j telescopes to zero. The left side is precisely sum_j w_ij v_j.
This proves the barycentric identity. No symmetry lambda_ij=lambda_ji is assumed
or generally true. These coordinates are classical prior art, not a new latent
representation: [Floater, Mean value coordinates](https://doi.org/10.1016/S0167-8396(03)00002-5).

For any positive row-normalized coefficients, the fixed-boundary equations

\[
Y_i=\sum_{j\sim i}\lambda_{ij}Y_j\qquad (i\ \hbox{interior})
\]

have a unique solution. To prove uniqueness, subtract two solutions and consider
one scalar coordinate. An interior positive maximum must equal all its neighbors'
values because every weight is positive. Propagating along a path to the zero
boundary gives a contradiction. Apply the same argument to a negative minimum.
The finite square interior coefficient matrix is therefore invertible.

## 3. Why the ordinary Tutte theorem cannot be applied directly

A graph is three-connected if deleting any set of at most two vertices leaves
it connected. The original grid is NOT three-connected: its four rectangle
corners have degree two. Also, its densely sampled boundary is only weakly
convex, since adjacent boundary edges are collinear. Both issues must be addressed.

Assume m,n>=2. Delete only the four corner vertices and replace each two-edge
boundary corner path by one edge joining its former neighboring boundary
vertices. Call the resulting graph G'. Its boundary is a rectangle with four
small corner triangles chopped off. It is a simple convex polygon, allowing
collinear boundary vertices. No interior vertex has lost a neighbor, so no
interior barycentric equation changes.

Here is a direct proof of three-connectivity. Let B be the new perimeter cycle,
I the interior (m-1)-by-(n-1) rectangular grid, and S any set of at most two
removed vertices.

- If at most one removed vertex belongs to B, B minus S is connected. From every
  surviving interior vertex, the four axial grid paths reach four distinct
  noncorner boundary vertices. These paths meet only at their starting vertex.
  At most two deletions cannot destroy all four; every surviving interior vertex
  is connected to the surviving perimeter.
- If both removed vertices belong to B, I is untouched and connected, including
  the single-row or single-vertex cases. Each surviving boundary vertex has an
  inward neighbor in I. Thus all surviving vertices are connected through I.

This establishes the exact graph hypothesis, without substituting a triangulation
theorem or assuming that degree-two rectangle corners are harmless.

The applicable classical result is Gortler-Gotsman-Thurston Theorem 4.1: a
three-connected plane graph with a weakly convex embedded outer boundary and
strictly positive directed barycentric weights has an embedding with strictly
convex interior faces. Its Appendix B rules out degenerate interior angles and
areas. See the primary paper
[Discrete One-Forms on Meshes and Applications to 3D Mesh Parameterization](https://www.cs.harvard.edu/~sjg/papers/tutte.pdf).

Applying this result to G' therefore gives strict convexity of every unchanged
cell and of each chopped corner triangle, for EVERY positive weight choice.

## 4. Restoring corners: explicitly check all four areas

At one reference corner, translate/rotate coordinates so the original corner
is c=(0,0), its boundary neighbors are r=(a,0) and u=(0,b), with a,b>0.
Let the adjacent interior vertex be v=(s,t). The restored corner cell has cyclic
order c,r,v,u. Its four determinants are exactly

\[
q_1=ab,\qquad q_2=at,\qquad q_3=bs+at-ab,\qquad q_4=bs.
\]

The interior vertex lies strictly inside the chopped boundary polygon. In
particular s>0, t>0, and bs+at>ab, the strict inside half-plane of the new
diagonal boundary edge. All four quantities are therefore positive.
The other corners follow by rigid rotation and translation, which preserve
these oriented determinants. Reinserting the four fixed corners creates a
table in H_0 without changing any interior coordinate or weight equation.

If m=1 or n=1, every vertex belongs to the boundary; there are no learnable
interior coordinates and the fixed table is trivial. These strips do not
require the corner-suppression proof.

## 5. A path inside the actual four-corner space

Choose positive normalized weights lambda^0 representing the identity table X
and lambda^1 representing any target Y in H_0, using Section 2. Define

\[
\lambda(t)=(1-t)\lambda^0+t\lambda^1,\qquad 0\le t\le1.
\]

Every coefficient stays positive and every row sum stays one. Solve the
fixed-boundary equations for these weights and call the table Y(t). Uniqueness
and continuity of the inverse of a finite invertible matrix imply that Y(t)
depends continuously on t, with Y(0)=X and Y(1)=Y. Sections 3-4 ensure Y(t)
belongs to H_0 for EVERY t, with exactly the same boundary positions.

Thus H_0 is path-connected. Interpolating barycentric representations to obtain
injective morphs is classical; see
[Floater-Gotsman, How to morph tilings injectively](https://doi.org/10.1016/S0377-0427(98)00202-7).
The useful point here is checking that this argument applies to our stricter
four-corner rectangular-grid class, not merely to some triangulated P1 class.

This is an existence proof. The registration code does NOT compute these weights,
invert this matrix, or evaluate this path online. A linear solve used to prove
existence is not a hidden solve in the implemented forward decoder.

## 6. Finite reachability by the forward decoder

The interval [0,1] is compact. Continuity and strict positivity give a
target/path-dependent positive minimum

\[
\delta=\min_{t,\,\text{cell},\,k}q_k(Y(t))/q_{\rm ref}>0.
\]

Fix ANY eta with 0<=eta<delta. For an accepted table Z, set its normalized
slacks s_k=q_k(Z)/q_ref-eta>0. For a single common direction e, simultaneous
nodal motion Z'_i=Z_i+u_i e has EXACTLY affine determinant changes:

\[
q_k(Z')/q_{\rm ref}-\eta=s_k+(C_Zu)_k.
\]

C_Z is the local linear operator obtained by expanding the determinant;
quadratic terms vanish since det(e,e)=0. Boundary amplitudes are zero.
Define its feasibility gauge

\[
g_Z(u)=\max\bigl(0,\max_k[-(C_Zu)_k/s_k]\bigr).
\]

The existing analytic decoder takes a raw nodal proposal r and returns

\[
u=\sigma r,\qquad \sigma=\min(1,\theta/g_Z(r)),\quad 0<\theta<1,
\]

where theta/0 is infinity. A proposal with g_Z(r)<=theta is passed unchanged.

Uniform continuity of Y(t) permits a sufficiently fine FINITE partition of
[0,1]. Between consecutive tables Y^a,Y^b, change all interior x coordinates
to those of Y^b, then all interior y coordinates. For sufficiently close
tables, the intermediate mixed table remains in a small neighborhood of the
path, above the eta floor. The local coefficient operators C_Z are bounded
on a compact neighborhood and the slacks have a uniform positive lower bound.
Consequently the partition can be chosen so each leg satisfies g_Z(r)<=theta.
Every leg is then passed unchanged by the analytic decoder. Each leg's entire
straight segment is also feasible because its determinants depend affinely on
that single-direction amplitude.

The unrestricted finest-level nodal coefficient field can represent each raw
leg exactly. After finitely many alternating horizontal/vertical updates, the
table equals Y EXACTLY, not only approximately. An exact single-direction leg
can also be realized by the radial decoder u=r/(1+g_Z(r)): for a feasible u,
g_Z(u)<1 and its inverse raw proposal is r=u/(1-g_Z(u)).

**What has been proved:** any table in H_0 is reachable from identity by some
finite alternating full-nodal forward sequence while preserving a sufficiently
small positive floor depending on the constructed path. No online global
linear solve is required by that forward sequence.

**What has NOT been proved:** two endpoints in H_.001 have a path inside
H_.001; a particular 10-stage schedule reaches all targets; coarse coefficient
fields alone are sufficient; the required depth is uniformly small; gradients
are well-conditioned; Adam converges; a neural encoder learns the sequence;
or floating-point output is guaranteed without actual numerical checks.
The chosen path's delta may be below .001. This limitation is essential.

## 7. Interpretation for the research objective

This result replaces the earlier conditional phrase 'if a legal vertex-motion
path exists' for the H_0 rectangular-grid class with an explicit classical
construction proving that such a path exists. It gives a rigorous framework
for coordinated forward maps and clarifies which universality claim is valid.
It does not require quasiconformal geometry or prescribed Beltrami coefficients.

It is not a practical accuracy result. Real registration still needs informative
image correspondence and a useful optimizer. The separate 7-to-8 label-oracle
experiment tests sparse capacity at the IMPLEMENTED eta=.001 and actual budget;
this theorem neither predicts its outcome nor converts its manual-label fit
into image-registration success. Classical convex-combination and morphing
results are prior art; novelty claims must concern a precisely compared decoder,
its numerical implementation or measured application benefit, not this theorem
without acknowledging that lineage.
