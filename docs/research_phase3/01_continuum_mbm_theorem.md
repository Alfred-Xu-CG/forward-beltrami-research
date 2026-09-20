# Continuum MBM theorem and modulus identities

Status: independently checked; wording revisions below incorporate the
checker report `tmp/phase3_continuum_checker.md`.

## 1. Geometric and analytic setting

Let \(\Omega\subset\mathbb R^2\simeq\mathbb C\) be a bounded simply connected
Lipschitz quadrilateral. Its boundary is the cyclic union
\[
\partial\Omega=\Gamma_B\cup\Gamma_R\cup\Gamma_T\cup\Gamma_L,
\]
where each arc has positive length and consecutive arcs meet only at their
corners. The labels are chosen so that the target rectangle has bottom,
right, top, and left sides in counter-clockwise order.

Let
\[
\mu=a+ib\in L^\infty(\Omega),\qquad
\|\mu\|_{L^\infty}\le k<1,
\]
and define
\[
A(\mu)=\frac{1}{1-|\mu|^2}
\begin{pmatrix}
|1-\mu|^2&-2b\\
-2b&|1+\mu|^2
\end{pmatrix}.
\tag{MB3-1}
\]
Then \(A=A^\top\), \(A\) is uniformly elliptic, and
\(\det A=1\) almost everywhere. We use
\[
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
\]

The fixed-P1 and graph statements below are **not** consequences of this
continuum theorem. They require separate compatibility and topology proofs.

## 2. Weak mixed problem

Define
\[
V=\{\phi\in H^1(\Omega):\operatorname{Tr}\phi=0
\text{ on }\Gamma_L\cup\Gamma_R\}.
\]
Choose any \(u_0\in H^1(\Omega)\) with trace zero on \(\Gamma_L\) and one on
\(\Gamma_R\), and set \(u=u_0+\widetilde u\), \(\widetilde u\in V\), where
\[
\int_\Omega (A\nabla u)\cdot\nabla\phi\,dx=0
\qquad\forall\phi\in V.
\tag{MB3-2}
\]
Uniform ellipticity and the positive-measure Dirichlet portion give existence
and uniqueness by Lax--Milgram. In strong notation this is
\[
\nabla\cdot(A\nabla u)=0,\quad
u|_{\Gamma_L}=0,\quad u|_{\Gamma_R}=1,\quad
n^\top A\nabla u=0\text{ on }\Gamma_B\cup\Gamma_T.
\]
The last condition is a weak conormal trace; it is not an assumption that
\(\nabla u\) is classically continuous at corners.

## 3. Stream function and Beltrami equation

Let \(q=A\nabla u\). Equation (MB3-2) says \(\nabla\cdot q=0\) in the
distributional sense and gives zero conormal flux on the two Neumann arcs.
Because \(\Omega\) is simply connected, the rotated one-form \(Jq\) is exact:
there exists \(v\in H^1(\Omega)\), unique up to an additive constant, such that
\[
\nabla v=J A\nabla u.
\tag{MB3-3}
\]
The existence statement is the weak stream-function theorem for a
divergence-free \(L^2\) field on a simply connected Lipschitz domain. It is
not a claim that a discrete P1 flux is automatically exact.

For the tensor convention (MB3-1), direct substitution of (MB3-3) into the
complex derivatives of \(f=u+iv\) gives
\[
f_{\bar z}=\mu f_z\quad\text{a.e. in }\Omega.
\tag{MB3-4}
\]
Conversely, the real part of any \(W^{1,2}\) solution of (MB3-4) satisfies
\(\nabla\cdot(A\nabla u)=0\), and its imaginary part satisfies (MB3-3).
This is the standard planar conductivity--Beltrami correspondence.

On \(\Gamma_B\cup\Gamma_T\), (MB3-2) and (MB3-3) imply, as a distributional
arc identity in the appropriate \(H^{-1/2}\) trace space, that the tangential
derivative of \(v\) is zero. Thus \(v\) has a constant trace on each connected
Neumann arc. Write these constants as \(v_B\) and \(v_T\). On
\(\Gamma_L\cup\Gamma_R\), \(u\) is constant. Since a symmetric
determinant-one matrix obeys
\[
AJA=J,
\tag{MB3-5}
\]
we have \(A\nabla v=J\nabla u\); hence the conormal derivative of \(v\) is zero
on the left and right arcs. This already identifies the correct complementary
tensor: it is the same \(A\), not \(A^{-1}\), for this determinant-one
Beltrami convention.

## 4. Continuum rectangle theorem

**Theorem (weak mixed MBM closure).** Under the setting of Sections 1--3, the
stream map \(f=u+iv\), after adding a constant to \(v\), is the unique
orientation-preserving quasiconformal homeomorphism
\[
f:\Omega\longrightarrow (0,1)\times(0,M)
\]
onto the *open* rectangle interior. It has a continuous homeomorphic closure
extension \(\bar f:\bar\Omega\to[0,1]\times[0,M]\) that maps
\(\Gamma_L,\Gamma_R,\Gamma_B,\Gamma_T\) to the corresponding closed
rectangle sides and satisfies (MB3-4) in \(\Omega\). Here \(M=v_T-v_B>0\).

**Proof structure.**

1. Extend \(\mu\) by zero outside \(\Omega\). The measurable Riemann mapping
   theorem supplies a normalized global quasiconformal homeomorphism \(F\)
   solving the extended Beltrami equation.
2. Since \(\Omega\) is a Jordan domain, \(F(\Omega)\) is a Jordan domain and
   the boundary correspondence is continuous. Map the image quadrilateral
   conformally to a Euclidean rectangle, sending the four marked boundary
   arcs to its four sides. Scale the horizontal coordinate so the left/right
   values are 0 and 1. The conformal postcomposition does not change
   \(\mu\), so the resulting map \(g=U+iV\) still satisfies (MB3-4).
3. The conductivity--Beltrami correspondence gives
   \(\nabla\cdot(A\nabla U)=0\) and \(\nabla V=JA\nabla U\). Because \(U\)
   is constant on the vertical sides and \(V\) is constant on the horizontal
   sides, \(U\) satisfies exactly the mixed weak problem (MB3-2).
4. Mixed-problem uniqueness gives \(U=u\). The two stream functions \(V\) and
   \(v\) have the same gradient, so they differ by a constant. Hence
   \(f=g+\mathrm{i}\,c\), and \(f\) is the same quasiconformal homeomorphism
   onto the rectangle.

The theorem uses a Jordan/Lipschitz quadrilateral and weak traces. If one wants
pointwise derivatives at corners or a classical Neumann condition, stronger
boundary and coefficient regularity is required; that is a regularity
corollary, not part of the topological statement.

### 4.1 Boundary-trace monotonicity (continuum only)

Because the closure extension is an orientation-preserving homeomorphism and
maps each marked boundary arc to one rectangle side, each side restriction is
a homeomorphism of compact intervals. Therefore its nonconstant coordinate is
strictly monotone on the open arc. With positive counter-clockwise boundary
orientation, the order is: bottom side \(u\) increases from 0 to 1; right side
\(v\) increases from \(v_B\) to \(v_T\); top side \(u\) decreases from 1 to 0;
and left side \(v\) decreases from \(v_T\) to \(v_B\). In particular,
\(M=v_T-v_B>0\). This is a topological consequence of the canonical QC map
and does not assert a pointwise sign for \(q\cdot n\). It also does not imply
that an independently solved fixed-P1 complementary problem has ordered side
traces.

## 5. Modulus, energy, and flux

The energy is
\[
E_A(u)=\int_\Omega \nabla u^\top A\nabla u\,dx.
\]
Testing the weak equation by \(u-u_0\), or using Green's identity with the
mixed traces, gives
\[
E_A(u)=\int_{\Gamma_R} (A\nabla u)\cdot n\,ds
=\int_{\Gamma_R}q\cdot n\,ds.
\tag{MB3-6}
\]
The integrated right-hand side is positive because \(f\) is orientation
preserving and the right side has a positive total stream-coordinate change;
no pointwise inequality \(q\cdot n\ge0\) is asserted. Since the target
rectangle has width one and height \(M\), the stream-coordinate identity gives
\[
M=E_A(u)=\int_{\Gamma_R}q\cdot n\,ds.
\tag{MB3-7}
\]
This is a continuum identity. A fixed P1 code must validate its flux, energy,
and geometric area through independent calculations.

## 6. Complementary mixed problem and reciprocal energy

Set
\[
w=\frac{v-v_B}{M}.
\]
Then \(w=0\) on \(\Gamma_B\), \(w=1\) on \(\Gamma_T\), and
\(n^\top A\nabla w=0\) on \(\Gamma_L\cup\Gamma_R\). In particular,
\[
\operatorname{div}(A\nabla v)=\operatorname{div}(J\nabla u)=0,
\]
so \(w\) solves the
same tensor \(A\) with the complementary pair of Dirichlet arcs. Moreover,
\[
E_A(v)=E_A(u)=M,\qquad
E_A(w)=\frac{1}{M}.
\tag{MB3-8}
\]
The first equality follows from \(\nabla v=JA\nabla u\), \(AJA=J\), and
\(|J\xi|=|\xi|\) in the corresponding quadratic identity; the second follows
from \(v=v_B+Mw\). Therefore the reciprocal relation uses the same \(A\),
not \(A^{-1}\). It is still possible for an independent fixed-mesh
complementary solve to fail to reproduce \(v/M\), because its discrete
compatibility space may not contain the stream coordinate.

## 7. Sanity cases and failure boundaries

### 7.1 Constant real coefficient

For \(\mu=a\in(-1,1)\),
\[
A=\operatorname{diag}\!\left(\frac{1-a}{1+a},\frac{1+a}{1-a}\right).
\]
On the unit square, \(u=x\), \(v=((1-a)/(1+a))y\), and
\[
M=\frac{1-a}{1+a}=E_A(u).
\]
This is the exact affine check.

### 7.2 Constant complex coefficient

Let \(f(z)=z+\mu\bar z\) and let \(\Omega=f^{-1}(R^\circ)\) for the open
rectangle \(R^\circ=(0,1)\times(0,H)\). Then \(\Omega\) is an affine parallelogram,
\(f\) is an exact solution of (MB3-4), and the mixed data are inherited from
the four sides of \(R\). This is the appropriate constant-complex sanity case;
using the original axis-aligned square without changing its boundary data is
not an exact affine test when \(\operatorname{Im}\mu\ne0\).

### 7.3 What the theorem does not imply

- It does not prove that a sampled P1 interpolation is fold-free.
- It does not make arbitrary facewise \(\mu_T\) fixed-P1 realizable.
- It proves continuum side-trace monotonicity only through the closure
  homeomorphism; monotonicity of independently solved discrete side traces is
  not implied and is false for the fixed-P1 counterexample recorded in
  `04_mmatrix_delaunay.md` and `00_correction_audit.md`.
- It does not provide a fast factorization when \(A(\mu)\) changes with every
  neural sample.

## 8. Evidence status

| statement | level |
|---|---|
| Mixed weak problem has a unique \(u\) | continuum theorem |
| Stream function exists on a simply connected domain | continuum theorem |
| \(f_{\bar z}=\mu f_z\) | conductivity--Beltrami equivalence |
| Rectangle homeomorphism after QC existence and conformal uniformization | continuum theorem under stated Jordan assumptions |
| \(M=E_A(u)=\) right flux | continuum identity |
| Same-\(A\) complementary problem and reciprocal energy | continuum identity |
| Marked-side trace ordering | continuum consequence of the closure homeomorphism |
| Independent fixed-P1 complementary solve equals the stream coordinate | open/false in general |

### Continuum manufactured monotonicity check

For an explicit sanity case on the unit square, let

\[
f_\varepsilon(x,y)=x+\varepsilon\sin(\pi x)\sin(\pi y)+\mathrm{i}y,
\qquad 0<\varepsilon<1/\pi.
\]

Its Jacobian determinant is
\(1+\varepsilon\pi\cos(\pi x)\sin(\pi y)>0\), and its exact coefficient
\(\mu=f_{\bar z}/f_z\) satisfies \(\lVert\mu\rVert_\infty<1\). At
\(\varepsilon=0.2\), a dense-grid check gives minimum determinant
approximately 0.37168 and maximum coefficient magnitude approximately 0.45806.
Here \(u=x+\varepsilon\sin(\pi x)\sin(\pi y)\) and \(v=y\), so the right
side trace increases and the left-side trace decreases exactly as in the
continuum ordering above. This is a manufactured check of the canonical-map
statement, not a validation of an independent fixed-P1 complementary solve.

### Fixed-mesh numerical stress check

On the structured square with a smooth nodal coefficient
\[
\mu(x,y)=0.25e^{0.7i}\exp(-((x-0.5)^2+(y-0.5)^2)/0.18),
\]
the existing face-averaged \(P_1\) reference was run at \(65^2\) and \(129^2\)
vertices. At (129^2), the primary energy was (0.8365161234202931), the
complementary energy was (1.1954394999959734), and the reciprocal identity
held to (2\times10^{-15}) because both quantities use the same assembled
matrix. An independently integrated raw face-flux on the right boundary was
(0.8385324350104797), differing from the energy by (2.0163\times10^{-3}).
The discrepancy is expected for a conforming primal \(P_1\) solution whose
raw face flux is not an equilibrated \(H(\mathrm{div})\) field; it is not a
contradiction of the continuum theorem. The compatible manufactured map in
`02_discrete_conjugacy.md` supplies the missing exact flux-conjugacy condition.

Useful primary/authoritative references include the [mixed-boundary quadrilateral
modulus formulation](https://epubs.siam.org/doi/10.1137/24M1656840), the
[Hersonsky discrete mixed Dirichlet--Neumann rectangle theorem](https://arxiv.org/abs/1006.0026),
and [Leonetti--Nesi's planar conductivity/Beltrami analysis](https://doi.org/10.1016/S0021-7824(97)89947-3).
