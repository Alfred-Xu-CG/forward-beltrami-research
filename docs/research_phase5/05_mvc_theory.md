# Route II theory: mean-value canonical coordinates for directed Tutte maps

## 1. Scope and result status

This document answers the theoretical part of Route II.  It does **not** by
itself close Route II.  The companion chapters now contain differentiable
implementation, finite-difference, round-trip, optimization, and incremental
solve evidence, while adjudication of the O4 image failure and the independent
final checker remain separate obligations.

The main exact conclusion is the following.  On a fixed oriented triangulated
disk, every nondegenerate piecewise-affine homeomorphism with an admissible
convex boundary has a uniquely specified set of local mean-value-coordinate
(MVC) rows.  If those rows and the map's boundary vertices are fed to the same
directed Tutte decoder, the decoder returns the original vertex map exactly in
exact arithmetic.  Thus MVC defines a section of the many-to-one Tutte decoder,
not a new decoder and not a new topology theorem.

The claims in this document have three labels:

- **Theorem** means that a proof is given here, with prior work cited for the
  classical ingredient.
- **Prior-art fact** means that the cited primary source states or implements
  the claim.
- **Numerical expectation** means a falsifiable prediction for the Route II
  experiments, not an observed result.

No claim below says that MVC reproduces a prescribed Beltrami coefficient, that
`|mu| < 1` alone makes a sampled P1 map injective, or that a continuous map has
a fold-free P1 interpolant on an arbitrary mesh.

## 2. Fixed combinatorics and the directed Tutte decoder

Let

\[
T=(V,F),\qquad V=I\mathbin{\dot\cup}B,
\]

be a finite oriented simplicial complex homeomorphic to a closed disk.  The
boundary vertices `B` form one cyclically ordered boundary loop and `I` denotes
the interior vertices.  For an interior vertex \(i\), let

\[
N(i)=(j_1,\ldots,j_{d_i})
\]

be the cyclic order induced by the oriented link of \(i\); indices in a one-ring
are understood modulo \(d_i\).  A directed probability row is

\[
p_i=(p_{ij})_{j\in N(i)}\in\operatorname{int}\Delta^{d_i-1},
\qquad
p_{ij}>0,
\qquad
\sum_{j\in N(i)}p_{ij}=1.
\tag{2.1}
\]

Write \(p=(p_i)_{i\in I}\) and prescribe ordered boundary coordinates
\(b\in\mathbb R^{|B|\times 2}\).  The directed Tutte decoder \(D\) is the
unique solution, when it exists, of

\[
Y_i=\sum_{j\in N(i)}p_{ij}Y_j,\quad i\in I,
\qquad
Y_B=b.
\tag{2.2}
\]

After separating interior and boundary columns of the row-stochastic matrix,

\[
Q=P_{II},\qquad R=P_{IB},
\]

the two coordinate columns solve

\[
A(p)Y_I=R(p)b,
\qquad
A(p)=I-Q(p).
\tag{2.3}
\]

### 2.1 Existence and uniqueness of the linear solve

**Lemma 2.1 (absorbing-row uniqueness).**  Suppose every interior graph vertex
has a path to the boundary and every supported row entry is strictly positive.
Then \(\rho(Q)<1\), \(A=I-Q\) is nonsingular, and (2.3) has a unique solution
for every boundary array \(b\).

**Proof.**  The nonnegative matrix \(Q\) is substochastic.  A row adjacent to
the boundary has row sum strictly below one.  From every other interior state,
strictly positive supported transitions give a positive-probability path to
such a deficient row and then to the boundary.  Because the graph is finite,
there is an integer \(m\) for which every row of \(Q^m\) has sum strictly below
one.  Hence

\[
\rho(Q)^m\leq \lVert Q^m\rVert_\infty<1.
\]

The Neumann series \((I-Q)^{-1}=\sum_{k\geq0}Q^k\) therefore exists. \(\square\)

This algebraic lemma does not use convexity of \(b\).  Convexity belongs to a
separate sufficient theorem for global topology.  The exact theorem package
used throughout Route II is the following.

**Tutte--Floater sufficient hypotheses.**  The source is a finite, coherently
oriented triangulation of a closed disk; every interior equation uses all graph
neighbors with strictly positive coefficients summing to one; and the ordered
boundary is mapped homeomorphically, with the correct orientation, onto the
boundary of a convex polygonal region.  A *dividing edge* is a mesh edge that is
not a boundary-loop edge but whose two endpoints are boundary vertices.  If the
target polygon is only weakly convex, no dividing edge may be mapped entirely
into its boundary.  Under these hypotheses, the exact P1 solution of (2.2) is
one-to-one.  A strictly convex target makes the dividing-edge obstruction
automatic.  A rectangle with subdivided sides is weakly convex; the structured
rectangle used here satisfies the condition by preserving corner/side
incidence and strict order within each side.  This is the scope of
[Floater's one-to-one P1 theorem](https://doi.org/10.1090/S0025-5718-02-01466-7).

These are **sufficient**, not asserted necessary, conditions for arbitrary
positive rows to decode to a P1 homeomorphism.  They are stronger than the
conditions needed for Lemma 2.1 and stronger than the local conditions needed
to evaluate one MVC row.  Conversely, exact reconstruction of one already
valid map does not prove that arbitrary perturbations remain globally valid.
The implementation therefore also performs finite-precision face, boundary,
and global P1 audits; those numerical screens are evidence about the returned
array, not a replacement for the exact theorem.

### 2.2 Probability and logit spaces

Finite row logits \(\ell_i\in\mathbb R^{d_i}\) produce (2.1) by

\[
p_{ij}=\operatorname{softmax}(\ell_i)_j
=\frac{e^{\ell_{ij}}}{\sum_k e^{\ell_{ik}}}.
\tag{2.4}
\]

For every scalar \(c_i\),

\[
\operatorname{softmax}(\ell_i+c_i\mathbf 1)
=\operatorname{softmax}(\ell_i).
\tag{2.5}
\]

Equation (2.5) is the row-shift gauge.  It exists before any geometric
nonuniqueness is considered.

## 3. The valid-map space and the local star hypotheses

Fix source vertex coordinates \(X\) whose oriented faces agree with the
orientation of \(T\).  For target vertices \(Y\), let \(f_Y\) be the P1 map
obtained by affine interpolation of the vertex values over every source face.
Define \(\mathcal H\) to be the set of arrays \(Y\) satisfying all of the
following:

1. **Global validity.** \(f_Y\) is an orientation-preserving P1 homeomorphism
   from the source disk onto its image.
2. **Strict face nondegeneracy.** For every oriented face \((i,j,k)\),
   \[
   \det(Y_j-Y_i,Y_k-Y_i)>0.
   \tag{3.1}
   \]
3. **Admissible boundary.** \(Y_B\) is a simple counter-clockwise convex cycle
   in the boundary class accepted by the decoder.  Strict convexity suffices.
   A weakly convex, side-subdivided rectangle is included only with the
   side/corner incidence, strict side order, and no-mapped-dividing-edge
   condition defined in Section 2.1.
4. **Fixed combinatorics.** The cyclic link order comes from \(T\); it is not
   recomputed by sorting floating-point angles.

These assumptions deliberately separate the local MVC construction from the
global hard-bijectivity contract.  MVC itself needs only the following local
facts at each interior vertex, but membership in \(\mathcal H\) supplies them
without an extra geometric guess.

**Lemma 3.1 (strict star geometry).**  For \(Y\in\mathcal H\) and an interior
vertex \(i\), the neighbor polygon

\[
\Psi_i=(Y_{j_1},\ldots,Y_{j_{d_i}})
\]

is a simple polygon strictly star-shaped with respect to \(Y_i\).  If

\[
r_m=Y_{j_m}-Y_i,\qquad \rho_m=\lVert r_m\rVert,
\]

and \(\alpha_m\) is the counter-clockwise angle from \(r_m\) to
\(r_{m+1}\), then

\[
\rho_m>0,
\qquad
0<\alpha_m<\pi,
\qquad
\sum_{m=1}^{d_i}\alpha_m=2\pi.
\tag{3.2}
\]

**Proof.**  The restriction of a P1 homeomorphism to the closed star of \(i\)
is an embedding.  Its link is therefore a simple closed polygon, and each
radial triangle fills the sector between two consecutive link rays.  Strict
positive orientation gives \(0<\alpha_m<\pi\).  The sectors cover a punctured
neighborhood of the interior vertex once, so their angles sum to \(2\pi\).
Every segment from \(Y_i\) to a point of the link edge
\([Y_{j_m},Y_{j_{m+1}}]\) lies in the corresponding radial triangle; hence
\(Y_i\) lies in the strict kernel of \(\Psi_i\). \(\square\)

The one-ring polygon need **not** be convex.  Strict star-shapedness with the
center in the kernel is the sufficient local condition supplied by a valid P1
star and used in the proof below.  It is not asserted to be a necessary
characterization of every isolated configuration whose normalized MVC weights
happen to be positive.  Requiring convex one-rings would incorrectly exclude
valid P1 maps.

## 4. Mean value coordinates on an interior one-ring

Floater introduced planar MVC precisely to express the center of a star-shaped
one-ring as a convex combination of its neighboring vertices
([Floater 2003](https://doi.org/10.1016/S0167-8396%2803%2900002-5)).  Hormann and
Floater later treated arbitrary simple polygons and proved, among other
properties, affine precision and positivity inside the kernel of a star-shaped
polygon ([Hormann--Floater 2006, Equation (11) and Corollary
4.7](https://www.inf.usi.ch/hormann/papers/Hormann.2006.MVC.pdf),
[DOI](https://doi.org/10.1145/1183287.1183295)).

For the local star in (3.2), define

\[
t_m=\tan\frac{\alpha_m}{2}>0,
\tag{4.1}
\]

\[
w_m=\frac{t_{m-1}+t_m}{\rho_m}>0,
\qquad
p^{\mathrm{MVC}}_{i j_m}=\frac{w_m}{\sum_{k=1}^{d_i}w_k}.
\tag{4.2}
\]

The irrelevant factor two used by some MVC conventions cancels in the
normalization.  A trigonometry-free half-angle evaluation is

\[
t_m=
\frac{\det(r_m,r_{m+1})}
{\rho_m\rho_{m+1}+r_m^\top r_{m+1}}.
\tag{4.3}
\]

Under (3.2), both numerator and denominator in (4.3) are positive.  Equation
(4.3) avoids `acos`, but it does not remove the true singular limit
\(\alpha_m\uparrow\pi\).

### 4.1 A self-contained linear-precision proof

Let \(u_m=r_m/\rho_m\) be the unit ray and let \(J(x,y)=(-y,x)\) denote
counter-clockwise rotation by \(\pi/2\).  For two consecutive unit rays with
counter-clockwise angle \(\alpha_m\), direct half-angle algebra gives

\[
t_m(u_m+u_{m+1})=-J(u_{m+1}-u_m).
\tag{4.4}
\]

Consequently,

\[
\begin{aligned}
\sum_m w_m r_m
&=\sum_m(t_{m-1}+t_m)u_m\\
&=\sum_m t_m(u_m+u_{m+1})\\
&=-J\sum_m(u_{m+1}-u_m)=0.
\end{aligned}
\tag{4.5}
\]

Normalizing the positive \(w_j\) therefore yields

\[
\sum_{j\in N(i)}p^{\mathrm{MVC}}_{ij}=1,
\qquad
\sum_{j\in N(i)}p^{\mathrm{MVC}}_{ij}(Y_j-Y_i)=0,
\tag{4.6}
\]

or equivalently

\[
Y_i=\sum_{j\in N(i)}p^{\mathrm{MVC}}_{ij}Y_j.
\tag{4.7}
\]

This proves the exact property needed by the decoder; it is not an approximate
fit.

### 4.2 Canonical logits

Define

\[
\ell^{\mathrm{MVC}}_{i j_m}
=\log w_m-\frac1{d_i}\sum_{k=1}^{d_i}\log w_k.
\tag{4.8}
\]

Then

\[
\sum_m\ell^{\mathrm{MVC}}_{i j_m}=0,
\qquad
\operatorname{softmax}(\ell_i^{\mathrm{MVC}})
=p_i^{\mathrm{MVC}}.
\tag{4.9}
\]

The zero arithmetic mean in (4.8) fixes only the softmax row-shift gauge.  The
geometric formula (4.2) also selects one barycentric row from the larger family
that exists whenever \(d_i>3\).

MVC interior angles, TutteNet boundary angles, and Tutte row logits are three
different objects:

- \(\alpha_j\) in (4.1) is the angle between two neighboring rays based at an
  *interior* target vertex.
- TutteNet's boundary angles place boundary samples by rays from a global
  center to a square; they do not enter (4.1).
- \(\ell^{\mathrm{MVC}}_{ij}\) is the logarithm of an angle-and-distance
  expression.  It is not an angle.

### 4.3 Invariance and why the boundary must be encoded

The normalized row \(p_i^{\mathrm{MVC}}\) and centered logits (4.8) are
unchanged by a common translation, rotation, reflection with consistently
reversed cyclic order, or positive uniform scale of the local star.  In
particular, weights alone cannot distinguish globally similar maps.  Including
the ordered boundary coordinates in the encoder is therefore essential for the
injectivity theorem below.

## 5. Exact canonical-subset theorem

Probability coordinates and gauge-fixed logit coordinates are related but are
not the same space.  Define the two encoders

\[
E_p(Y)=\bigl(p^{\mathrm{MVC}}(Y),Y_B\bigr),
\qquad
E_\ell(Y)=\bigl(\ell^{\mathrm{MVC}}(Y),Y_B\bigr),
\tag{5.1}
\]

with codomains

\[
\mathcal P=
\left(\prod_{i\in I}\operatorname{int}\Delta^{d_i-1}\right)
\times\mathbb R^{|B|\times2},
\qquad
\mathcal L_0=
\left(\prod_{i\in I}\{\ell_i:\mathbf1^{\mathsf T}\ell_i=0\}\right)
\times\mathbb R^{|B|\times2}.
\tag{5.2}
\]

Let \(S(\ell,b)=(\operatorname{softmax}(\ell),b)\) row by row.  Then
\(S\circ E_\ell=E_p\).  Write \(D_p\) for the probability decoder (2.2) and
\(D_\ell=D_p\circ S\) for the logit decoder.  Their canonical images are

\[
\mathcal C_p=E_p(\mathcal H),
\qquad
\mathcal C_\ell=E_\ell(\mathcal H).
\tag{5.3}
\]

The adjective *canonical* always refers to these explicit MVC selections for
the fixed mesh link order and, in \(\mathcal C_\ell\), the arithmetic-zero-mean
row gauge.  It does not assert that either image is linear, convex, or a
globally complete Riemannian manifold.

**Theorem 5.1 (exact MVC encode--decode).**  Under the assumptions defining
\(\mathcal H\),

\[
\boxed{
D_p(E_p(Y))=D_\ell(E_\ell(Y))=Y
\quad\text{for every }Y\in\mathcal H.}
\tag{5.4}
\]

Moreover, \(E_p:\mathcal H\to\mathcal C_p\) and
\(E_\ell:\mathcal H\to\mathcal C_\ell\) are injective, and the corresponding
restricted decoders are their two-sided inverses.

**Proof.**  Lemma 3.1 makes every row (4.2) finite and strictly positive.
Equation (4.7) says that the given \(Y_I\), together with boundary
\(Y_B\), satisfies every row of the probability decoder.  Lemma 2.1 says that
this system has only one solution, so \(D_p(E_p(Y))=Y\).  Equation (4.9) and
\(D_\ell=D_p\circ S\) give the logit identity.  Equality of either encoded pair
for two maps, followed by the corresponding decoder, implies equality of the
maps.  For \(c_p=E_p(Y)\in\mathcal C_p\), for example,

\[
(E_p\circ D_p)(c_p)=E_p(D_p(E_p(Y)))=E_p(Y)=c_p,
\]

and the logit case is identical. \(\square\)

### 5.1 The canonicalization projector

On the two safe latent subsets

\[
\mathcal Z_{p,\mathcal H}
=\{(p,b):D_p(p,b)\in\mathcal H\},
\qquad
\mathcal Z_{\ell,\mathcal H}
=\{(\ell,b):D_\ell(\ell,b)\in\mathcal H\},
\tag{5.5}
\]

define

\[
P_{\mathrm{MVC}}^p=E_p\circ D_p,
\qquad
P_{\mathrm{MVC}}^\ell=E_\ell\circ D_\ell.
\tag{5.6}
\]

Each preserves its decoded map and is idempotent; explicitly,

\[
D_p(P_{\mathrm{MVC}}^p(p,b))=D_p(p,b),
\qquad
(P_{\mathrm{MVC}}^p)^2=P_{\mathrm{MVC}}^p,
\\
D_\ell(P_{\mathrm{MVC}}^\ell(\ell,b))=D_\ell(\ell,b),
\qquad
(P_{\mathrm{MVC}}^\ell)^2=P_{\mathrm{MVC}}^\ell.
\tag{5.7}
\]

Thus an M1 layer can re-encode a raw positive Tutte representation without
changing its decoded map.  This statement is exact in real arithmetic and only
on (5.5).  It does not define a global projector for folded, degenerate, or
boundary-invalid decoder outputs.

### 5.2 Differential consequence and latent lifts

Away from degenerate stars, all operations in (4.2)--(4.8) are smooth.  The
probability and gauge-fixed-logit versions give equivalent tangent identities.
Using \(D_\ell\circ E_\ell=\operatorname{id}_{\mathcal H}\), differentiation
gives

\[
dD_{\ell,E_\ell(Y)}\circ dE_{\ell,Y}
=\operatorname{id}_{T_Y\mathcal H}.
\tag{5.8}
\]

Therefore \(dE_{\ell,Y}\) is one right inverse, or latent lift, of the decoder
differential.  If a covariance construction \(L_Y\) is separately proved to
satisfy

\[
dD_{\ell,E_\ell(Y)}\circ L_Y=\operatorname{id},
\tag{5.9}
\]

then only the following follows automatically:

\[
(L_Y-dE_{\ell,Y})\dot Y
\in\ker dD_{\ell,E_\ell(Y)}.
\tag{5.10}
\]

It does **not** follow that the covariance lift equals the MVC derivative.
Boundary semantics cannot be omitted: for a fixed boundary,
\(\dot Y_B=0\); for a moving boundary, the lift must include
\(\dot b=\dot Y_B\).  At a canonical point, differentiating (5.6) also gives
the idempotent linear projection

\[
dP_{\mathrm{MVC}}^\ell=dE_\ell\,dD_\ell,
\qquad
(dP_{\mathrm{MVC}}^\ell)^2=dP_{\mathrm{MVC}}^\ell.
\tag{5.10a}
\]

These identities provide decisive finite-difference tests for M1/M2.  They do
not choose a metric.  A claim that a lift is “minimum norm” is incomplete until
the norm on probability or logit perturbations is stated explicitly.

### 5.3 Exact covariance lift and its metric

The alternative covariance lift can also be stated without ambiguity.  Let
\((p,b)\in\mathcal Z_{p,\mathcal H}\), let \(Y=D_p(p,b)\), and prescribe a full
vertex tangent \(v\in\mathbb R^{|V|\times2}\).  For each interior row set

\[
s_{ij}=Y_j-Y_i,
\qquad
g_i=v_i-\sum_jp_{ij}v_j,
\qquad
C_i=\sum_jp_{ij}s_{ij}s_{ij}^{\mathsf T}.
\tag{5.11}
\]

The local star spans the plane, so \(C_i\) is symmetric positive definite.
Define

\[
(L_Yv)_{ij}=s_{ij}^{\mathsf T}C_i^{-1}g_i.
\tag{5.12}
\]

**Proposition 5.2 (weighted-minimum-norm decoder right inverse).**  If the
boundary is fixed, require \(v_B=0\).  If the boundary moves, include the
parameter tangent \(\dot b=v_B\).  In either case, (5.12) satisfies

\[
dD_{(p,b)}(L_Yv,v_B)=v.
\tag{5.13}
\]

Among all supported logit perturbations satisfying (5.13), it is the unique
minimizer of the explicitly weighted objective

\[
\frac12\sum_{i\in I}\sum_{j\in N(i)}
p_{ij}(\delta\ell_{ij})^2.
\tag{5.14}
\]

It is generally **not** the minimum Euclidean-norm probability perturbation,
the minimum unweighted-logit perturbation, or the MVC logit differential
\(dE_{\ell,Y}v\).

**Proof.**  Differentiating the \(i\)-th equilibrium row gives

\[
v_i-\sum_jp_{ij}v_j
=\sum_j\delta p_{ij}Y_j.
\tag{5.15}
\]

For row softmax,

\[
\delta p_{ij}=p_{ij}
\left(\delta\ell_{ij}-\sum_kp_{ik}\delta\ell_{ik}\right).
\tag{5.16}
\]

Because \(\bar s_i:=\sum_jp_{ij}s_{ij}=0\), equations (5.15)--(5.16)
reduce to the independent row constraints

\[
\sum_jp_{ij}s_{ij}\delta\ell_{ij}=g_i.
\tag{5.17}
\]

Substitution of (5.12) into (5.17) gives \(C_iC_i^{-1}g_i=g_i\).
Together with the prescribed boundary tangent and uniqueness of the
differentiated linear solve, this proves (5.13).  For the optimization claim,
the Lagrangian of one row is

\[
\frac12\sum_jp_{ij}(\delta\ell_{ij})^2
-\lambda_i^{\mathsf T}
\left(\sum_jp_{ij}s_{ij}\delta\ell_{ij}-g_i\right).
\]

Stationarity gives
\(\delta\ell_{ij}=s_{ij}^{\mathsf T}\lambda_i\); the constraint then gives
\(C_i\lambda_i=g_i\).  Strict convexity of (5.14) on all supported slots makes
this row minimizer unique, including its softmax gauge.  Summing the independent
row objectives proves the global statement. \(\square\)

The selected gauge is automatically probability-mean-zero:

\[
\sum_jp_{ij}(L_Yv)_{ij}
=\left(\sum_jp_{ij}s_{ij}\right)^{\mathsf T}C_i^{-1}g_i=0.
\tag{5.18}
\]

This exposes two implementation traps.  First, the sign in (5.12) is positive
when \(g_i=v_i-\sum_jp_{ij}v_j\); reversing either one, but not both, fails the
right-inverse identity.  Second, the apparently row-local formula is globally
coupled through the prescribed neighbor tangents in \(g_i\).  A translation has
\(g_i=0\), so moving-boundary translation requires no logit change.

For an arbitrary array that is **not** a decoded equilibrium, one may replace
\(Y_i\) by \(q_i=\sum_jp_{ij}Y_j\) in the centered vectors and obtain an exact
algebraic solution of (5.17).  That construction is not literally a right
inverse of the decoder differential *at that arbitrary array*, because the
array is not the decoder's base point.  The right-inverse and minimum-norm
claims above apply when \(Y=D(p,b)\); this domain condition should appear in
code documentation and tests.

The two lifts are genuinely different even in the smallest nontrivial case.
Take one interior point \(Y_i=(1/4,1/4)\) in the boundary triangle
\((0,0),(1,0),(0,1)\).  Its unique probability row, hence its MVC row, is
\(p=(1/2,1/4,1/4)\).  Under the fixed-boundary tangent \(v_i=(h,0)\),
affine barycentric coordinates give

\[
\delta p=(-h,h,0),
\qquad
\frac{\delta p}{p}=(-2h,4h,0).
\tag{5.19}
\]

The covariance lift is the second vector in (5.19), whose \(p\)-weighted mean
is zero.  Differentiating the arithmetic-mean-centered canonical logits (4.8)
instead gives

\[
dE_{\ell,Y}v=(-8h/3,10h/3,-2h/3).
\tag{5.20}
\]

Their difference is \((-2h/3)\mathbf1\), a nonzero softmax row-shift direction
when \(h\neq0\), and therefore lies in \(\ker dD\) exactly as (5.9) predicts.
For degrees above three, the difference may additionally contain geometric
barycentric-fiber directions.

There is no uniform covariance bound on \(\mathcal H\).  For the four-neighbor
star

\[
(1,0),\quad(0,\varepsilon),\quad(-1,0),\quad(0,-\varepsilon)
\]

about the origin with uniform weights, every \(\varepsilon>0\) gives a strict
convex star, but

\[
C=\operatorname{diag}(1/2,\varepsilon^2/2),
\qquad
\kappa_2(C)=\varepsilon^{-2}.
\tag{5.21}
\]

At \(\varepsilon=0\), \(C\) has rank one and (5.12) cannot lift a general
two-dimensional tangent.  This is an exact degeneracy counterexample and a
quantitative reason to report covariance condition numbers rather than merely
checking finite weights.

### 5.4 A common-gauge Moore--Penrose comparison

The phrase “the pseudoinverse of the decoder Jacobian” is ambiguous until both
padding and the row-shift gauge have been removed.  Let \(d_i=|N(i)|\), and
define the supported arithmetic-zero-row-mean space

\[
\mathcal G_i=\{z_i\in\mathbb R^{d_i}:\mathbf 1^{\mathsf T}z_i=0\},
\qquad
\mathcal G=\bigoplus_{i\in I}\mathcal G_i.
\tag{5.22}
\]

Unsupported padded slots are fixed to zero and are not coordinates.  Choose a
block-diagonal orthonormal injection

\[
Q:\mathbb R^m\longrightarrow\mathcal G,
\qquad
Q^{\mathsf T}Q=I_m,
\qquad
m=\sum_{i\in I}(d_i-1).
\tag{5.23}
\]

In the experiment \(Q_i\) is the deterministic Helmert basis in sorted
supported-slot order.  Hold the boundary argument fixed and retain *all*
decoder output coordinates, including the constant boundary coordinates:

\[
J_{\mathcal G}
=\left.\frac{\partial}{\partial\theta}
D(\ell+Q\theta,b)\right|_{\theta=0}
\in\mathbb R^{2|V|\times m}.
\tag{5.24}
\]

The \(2|B|\) boundary rows of \(J_{\mathcal G}\) are zero.  Under the full-rank
covariance hypotheses of Proposition 5.2, the covariance construction lifts
every fixed-boundary interior tangent, so

\[
\operatorname{rank}J_{\mathcal G}=2|I|.
\tag{5.25}
\]

The rank equality in (5.25) is necessary and sufficient for a linear right
inverse on *every* fixed-boundary interior tangent.  Positive-definite row
covariances are a convenient sufficient condition through (5.12)--(5.13);
they are not claimed necessary for representability of one particular tangent,
nor is the local condition asserted to be the only possible proof of global
surjectivity.

For a tangent \(v\) with \(v_B=0\), define

\[
z_{\rm MP}=QJ_{\mathcal G}^{+}\operatorname{vec}(v).
\tag{5.26}
\]

Because \(Q\) is orthonormal, (5.26) is the unique solution in
\(\mathcal G\cap(\ker dD)^{\perp}\), and it minimizes the *unweighted
Euclidean* supported-logit norm over all \(z\in\mathcal G\) satisfying
\(dD[z,0]=v\).  This is a different variational problem from (5.14).

The native covariance lift \(z_C=L_Yv\) has
\(\sum_jp_{ij}z_{C,ij}=0\), not generally
\(\sum_jz_{C,ij}=0\).  Its common-gauge representative is therefore

\[
\widehat z_{C,ij}
=z_{C,ij}-\frac1{d_i}\sum_{k\in N(i)}z_{C,ik}.
\tag{5.27}
\]

Equation (5.27) changes no softmax probability tangent and hence no decoder
tangent.  It does, however, generally increase
\(\sum_{ij}p_{ij}z_{ij}^2\): the weighted-minimum statement belongs to the
native probability-mean-zero representative, whereas (5.27) exists only to
make an unambiguous same-gauge comparison with \(z_{\rm MP}\) and
\(dE_{\ell,Y}v\).
All three common-gauge lifts obey

\[
dD_\ell\,z_{\rm MP}=dD_\ell\,dE_{\ell,Y}v
=dD_\ell\,\widehat z_C=v,
\tag{5.28}
\]

and therefore every pairwise difference is in
\(\ker dD|_{\mathcal G}\).  Equality of the three latent vectors is neither
implied nor expected when a row has degree greater than three.

## 6. Why the unrestricted decoder is not injective

Theorem 5.1 is a bijection only between \(\mathcal H\) and its selected image
\(\mathcal C\).  The full positive-weight decoder remains many-to-one.

For a fixed valid target map \(Y\), define at each interior vertex

\[
M_i(Y)=
\begin{bmatrix}
1&\cdots&1\\
Y_{j_1,x}&\cdots&Y_{j_{d_i},x}\\
Y_{j_1,y}&\cdots&Y_{j_{d_i},y}
\end{bmatrix},
\qquad
c_i(Y)=
\begin{bmatrix}1\\Y_{i,x}\\Y_{i,y}\end{bmatrix}.
\tag{6.1}
\]

The complete set of strictly positive directed rows that reproduce \(Y_i\) is

\[
\mathcal F_i(Y)
=\{p_i\in\mathbb R^{d_i}_{>0}:M_i(Y)p_i=c_i(Y)\}.
\tag{6.2}
\]

For a valid two-dimensional star, the neighbor coordinates affinely span
\(\mathbb R^2\), so

\[
\operatorname{rank}M_i(Y)=3.
\tag{6.3}
\]

MVC supplies a point in the relative interior of (6.2).  Hence

\[
\dim\mathcal F_i(Y)=d_i-3.
\tag{6.4}
\]

With the boundary fixed, the exact directed-probability fiber is the row-wise
product

\[
D^{-1}(Y;Y_B)
=\prod_{i\in I}\mathcal F_i(Y).
\tag{6.5}
\]

Indeed, every member of the right-hand side makes \(Y\) satisfy all equilibrium
rows, and Lemma 2.1 makes it the unique decoded solution; the converse follows
directly from (2.2).  Thus a degree-three row is geometrically unique, while a
degree greater than three row generically has infinitely many strictly positive
representations.  This is a statement for independent directed rows; symmetric
edge tying couples rows and has a different fiber.

If \(q_i\in\ker M_i(Y)\setminus\{0\}\), then for sufficiently small positive
and negative \(\varepsilon\),

\[
p_i+\varepsilon q_i>0
\quad\text{and}\quad
D(p_1,\ldots,p_i+\varepsilon q_i,\ldots;Y_B)=Y.
\tag{6.6}
\]

The corresponding first-order logit null direction can be chosen as

\[
\delta\ell_{ij}=q_{ij}/p_{ij},
\tag{6.7}
\]

because \(\sum_jq_{ij}=0\) and the softmax differential maps (6.7) to
\(q_i\).  Adding an arbitrary multiple of \(\mathbf1\) gives the independent
row-shift gauge.  Locally, a degree-\(d_i\) raw-logit fiber therefore has
dimension \(d_i-2\): \(d_i-3\) geometric barycentric directions plus one row
shift.

### 6.1 Explicit degree-four counterexample

Take one interior vertex at the origin and the counter-clockwise diamond
boundary

\[
Y_1=(1,0),\quad Y_2=(0,1),\quad
Y_3=(-1,0),\quad Y_4=(0,-1).
\]

For every \(0<a<1/2\),

\[
p(a)=\left(a,\frac12-a,a,\frac12-a\right)
\tag{6.8}
\]

is strictly positive, sums to one, and has weighted average zero.  All these
different rows decode the same four-triangle P1 homeomorphism.  MVC selects
only \(p(1/4)=(1/4,1/4,1/4,1/4)\).  Equation (6.8) is a concrete disproof of
any global “Tutte latent and map are one-to-one” statement.

## 7. Assumption boundary and counterexamples

| Missing condition | What fails | Consequence for Route II |
|---|---|---|
| \(Y_j\neq Y_i\) | \(\rho_j=0\) in (4.2) | Encoder undefined; reject rather than clamp and claim exactness. |
| Strict face area | Adjacent rays can have \(\alpha_j=0\) | A triangle is collapsed; no valid P1 homeomorphism. |
| Every wedge below \(\pi\) | \(t_j\) is singular at \(\pi\) and changes sign beyond it | The stated positivity guarantee fails; some configurations may still have positive combined weights, so this is not an iff characterization. |
| Correct cyclic link | Telescoping proof (4.5) uses true consecutive rays | Arbitrary geometric sorting or an off-by-one link can silently destroy linear precision. |
| Closed interior link | A boundary vertex has an open fan | Do not apply the closed-ring formula to boundary vertices; pass the boundary through. |
| Center in polygon kernel | MVC for a general nonconvex polygon need not be positive | Kernel membership is a sufficient positivity condition used here, not a claimed necessary condition for every individual positive row. |
| Boundary included in \(E_p,E_\ell\) | MVC is similarity-invariant | Weights or logits alone do not identify the map. |
| Boundary in the hard-safe class | Linear solve can still exist for a concave boundary | Exact reconstruction of one known map does not extend to a topology guarantee for arbitrary perturbed rows. |
| Boundary reachability | \(I-Q\) can contain a closed stochastic class | Decoder may be singular or nonunique. |
| Floating-point margins | Positive exact quantities may underflow, overflow, or round to degeneracy | The implementation needs fail-closed finite, residual, boundary, and face checks. |

Hormann and Floater proved that MVC remains well-defined for arbitrary simple
polygons, but they also explicitly note that the coordinates are not everywhere
positive for nonconvex polygons and that the corresponding warp is not generally
guaranteed one-to-one (their Section 7).  Their broader well-definedness theorem
must not be misquoted as a positivity or bijectivity theorem.

## 8. Smoothness and conditioning

On any subset with quantitative margins

\[
\rho_j\geq\rho_{\min}>0,
\qquad
\alpha_j\in[\alpha_{\min},\pi-\alpha_{\min}],
\tag{8.1}
\]

the half-angle formula, normalized rows, centered logs, decoder solve, and their
first derivatives are smooth.  Without such margins, smoothness on the open
valid set remains true pointwise, but no uniform derivative or condition-number
bound follows.

Two distinct numerical mechanisms must be reported separately:

1. **Encoder conditioning.**  As \(\alpha_j\uparrow\pi\), the denominator in
   (4.3) approaches zero and \(t_j\) becomes large.  As edge lengths collapse,
   \(1/\rho_j\) becomes large.  Normalization can hide the magnitude in
   probabilities while the logit spread and derivatives still grow.
2. **Decoder conditioning.**  Even exact positive rows only prove that
   \(A=I-Q\) is nonsingular.  They do not bound \(\lVert A^{-1}\rVert\).  Weak
   absorption or extreme row contrast can make the solve and implicit VJP
   ill-conditioned.

**Numerical expectation.**  For a well-conditioned float64 direct solve, the
round trips \(Y\to E_p(Y)\to D_p(E_p(Y))\) and
\(Y\to E_\ell(Y)\to D_\ell(E_\ell(Y))\) should be close to machine precision.  This
is not an unconditional tolerance theorem.  If the assembled right-hand side
or solve has residual perturbation \(e\), then

\[
\widehat Y_I-Y_I=A^{-1}e,
\tag{8.2}
\]

so a near-degenerate star or poorly conditioned decoder can amplify rounding.
The benchmark must therefore report both barycentric residual and reconstruction
error, together with encoder and decoder conditioning diagnostics.

## 9. Targeted prior-art audit

The audit below used primary papers or official publication pages.  Searches
were targeted rather than systematic and were run on 2026-09-22 with variants
of `mean value coordinates`, `MVC neural deformation`, `differentiable
barycentric parameterization`, `canonical Tutte coordinates`, and
`natural/Riemannian gradient embedding weights`.  A negative search result is
not a priority claim.

| Source | Verified contribution relevant here | What it does **not** establish for Route II |
|---|---|---|
| [Floater, *Parametrization and smooth approximation of surface triangulations*, 1997](https://doi.org/10.1016/S0167-8396%2896%2900031-3) | Shape-preserving convex-combination parameterization is an early foundation for positive barycentric mesh maps. | It does not define the present MVC re-encoding projector on learnable directed Tutte fibers. |
| [Floater & Gotsman, *How to morph tilings injectively*, 1999](https://doi.org/10.1016/S0377-0427%2898%2900202-7) | Interpolates convex-combination systems to obtain valid morphs under its common-boundary hypotheses. | A choice of barycentric rows for morphing is not a proof that raw weights are unique or a canonical neural latent. |
| [Floater, *Mean value coordinates*, 2003](https://doi.org/10.1016/S0167-8396%2803%2900002-5) | Introduces the local formula used in (4.2) for a center in a star-shaped one-ring and motivates parameterization/morphing. | It predates differentiable-network layers and does not study optimization on the raw Tutte decoder fiber. |
| [Hormann & Floater, *Mean value coordinates for arbitrary planar polygons*, 2006](https://doi.org/10.1145/1183287.1183295) | Proves affine precision, partition of unity, smoothness, and positivity in the kernel; extends well-defined MVC beyond convex polygons. | Well-defined coordinates on arbitrary simple polygons are not necessarily positive and do not by themselves give a one-to-one warp. |
| [Wang et al., *Neural Cages for Detail-Preserving 3D Deformations*, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yifan_Neural_Cages_for_Detail-Preserving_3D_Deformations_CVPR_2020_paper.html) ([DOI](https://doi.org/10.1109/CVPR42600.2020.00015)) | Uses a differentiable 3D cage/MVC module inside an end-to-end network; the released project is named `deep_cage`. | Its coordinates express shape points with respect to a 3D cage and it penalizes negative MVC values.  This is not a planar one-ring Tutte canonicalization or a hard P1 disk-homeomorphism theorem.  “Neural Cages” and “Deep Cage” should not be counted as two independent papers. |
| [Dodik et al., *Variational Barycentric Coordinates*, TOG 2023](https://doi.org/10.1145/3618403) ([MIT record](https://hdl.handle.net/1721.1/153282)) | Parameterizes valid generalized barycentric-coordinate *functions* by a neural field and explicitly identifies nonuniqueness for nonsimplicial cages. | The learned object is a cage-coordinate function over a polytope, not a canonical section of a sparse graph-equilibrium decoder. |
| [Aigerman & Groueix, *Generative Escher Meshes*, SIGGRAPH 2024](https://doi.org/10.1145/3641519.3657452) ([arXiv](https://arxiv.org/abs/2309.14564)) | Optimizes positive, possibly nonsymmetric Laplacian entries through a differentiable linear solve and proves coverage of valid tile configurations under its orbifold boundary conditions.  Its proof uses existence of positive barycentric rows. | This already covers the broad claim “positive Laplacian parameters give an end-to-end differentiable valid-map family.”  It does not select a unique MVC row for each decoded tile or remove the raw-weight fiber. |
| [Sun et al., *TutteNet*, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_TutteNet_Injective_3D_Deformations_by_Composition_of_2D_Mesh_Deformations_CVPR_2024_paper.html) ([DOI](https://doi.org/10.1109/CVPR52733.2024.02020)) | Composes differentiable positive-weight 2D Tutte layers and trains/predicts their parameters in a 3D deformation architecture. | It optimizes raw Laplacian/boundary parameters and does not present \(E_{\rm MVC}\circ D\) as a canonical retraction. |

The literature therefore rules out several novelty overclaims:

- MVC, its positivity-in-the-kernel proof, and its differentiability are not
  new.
- Putting MVC in a neural computation graph is not new.
- Optimizing through a positive Tutte/Laplacian solve is not new.
- Surjectivity of positive directed barycentric rows onto a valid embedding
  family is not new in the broad form needed here.

The scoped candidate contribution of Route II is narrower: use the local planar
MVC formula as an explicit, differentiable section of the redundant directed
Tutte map; implement the exact logit-space map-preserving projector
\(E_\ell\circ D_\ell\);
compare its derivative to a separately derived covariance right inverse; and
measure whether that gauge choice improves solve count, conditioning, memory,
or optimization at realistic resolution.  This remains a research hypothesis
until the implementation and fair comparisons close.

## 10. Concrete obligations for the Route II implementation and checker

The following tests follow directly from the theorem and should be treated as
decisive rather than cosmetic.

1. **Combinatorial link test.**  Every interior link is one closed cycle and
   each supported decoder neighbor occurs exactly once.  Boundary links must
   never be passed to the closed-ring MVC formula.
2. **Local identity test.**  In float64, record
   \[
   \max_i\left\|Y_i-\sum_jp^{\rm MVC}_{ij}Y_j\right\|.
   \]
   This isolates the encoder from the global solve.
3. **Global round trip.**  Record max vertex error and RMSE for
   \(Y\to E_\ell(Y)\to D_\ell(E_\ell(Y))\), along with the linear residual and an estimate of
   decoder conditioning.  Do not repair a failed round trip with an optimizer.
4. **Canonical logit test.**  Verify finite logits, zero row mean, strict
   realized probabilities, and `softmax(ell_mvc) == p_mvc` within dtype-aware
   tolerance.
5. **Fiber test.**  On the diamond example (6.8), demonstrate two different
   positive rows with identical decoded vertices and show that MVC selects the
   uniform row.  Separately test row-shift invariance.
6. **Projector test.**  For a safe raw latent, verify
   \(D_\ell(P_{\rm MVC}^\ell(z))\approx D_\ell(z)\) and
   \(P_{\rm MVC}^\ell(P_{\rm MVC}^\ell(z))\approx P_{\rm MVC}^\ell(z)\).
7. **Derivative test.**  Apply gradcheck/directional finite differences to the
   encoder and projector.  Check (5.7) for fixed-boundary and moving-boundary
   tangents separately.
8. **Degeneracy negatives.**  Include a zero radius, wrong cyclic order,
   \(\alpha\) near zero, \(\alpha\) near \(\pi\), folded star, and invalid
   boundary.  Fail closed with an attributable diagnostic.
9. **Scale test.**  Run Route I valid maps at control resolutions 11, 25, and
   49.  Report minimum MVC weight, maximum logit spread, local covariance
   condition, triangle quality, barycentric residual, and global round-trip
   error.  These are numerical observations, not theorem substitutes.

The local and remote 27-case matrices, M1/M2 differential tests, projector
tests, and degeneracy negatives have now been executed and are summarized in
the application chapter.  This checklist remains here to define what those
receipts mean; it is not a declaration that every Route-II closure question
has passed.

## 11. Bottom line

Under explicit disk, valid-P1-star, positive-margin, boundary, and reachability
hypotheses, MVC provides an exact canonical *representation* of every map in
\(\mathcal H\):

\[
\boxed{D_p\circ E_p=D_\ell\circ E_\ell=\operatorname{id}_{\mathcal H},
\qquad
E_p:\mathcal H\leftrightarrow\mathcal C_p:D_p|_{\mathcal C_p},
\qquad
E_\ell:\mathcal H\leftrightarrow\mathcal C_\ell:D_\ell|_{\mathcal C_\ell}.}
\]

The unrestricted positive Tutte decoder is nevertheless noninjective, with a
degree-\(d_i\) directed probability row carrying \(d_i-3\) local geometric
fiber dimensions and raw logits carrying one additional row-shift gauge.  MVC
chooses one point in each such fiber.  The bounded experiments do not establish
that this canonical choice is uniformly faster, better conditioned, or more
trainable: the map instance is favorable to O4, the image instance contains a
reproducible O4 failure, and a parameterization-fair speed conclusion remains
open.  No Route-II pass follows from the theorem.
