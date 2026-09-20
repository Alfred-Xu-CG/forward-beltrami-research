# Mixed-Boundary Modulus Linear Beltrami Solver (MBM-LBS)

## 1. Problem and notation

Let \(\Omega\subset\mathbb R^2\) be a simply connected quadrilateral domain whose
boundary is decomposed, in counter-clockwise order, into four non-overlapping
arcs

\[
\partial\Omega=\Gamma_L\cup\Gamma_T\cup\Gamma_R\cup\Gamma_B.
\]

The intended target rectangle is \([0,1]\times[0,M]\), with the left and right
arcs mapped to the vertical sides and the bottom and top arcs mapped to the
horizontal sides. The input is a measurable Beltrami coefficient

\[
\mu:\Omega\to\mathbb C,\qquad \|\mu\|_{L^\infty(\Omega)}\le k<1.
\]

Write \(\mu=a+ib\). The associated conductivity tensor is

\[
A(\mu)=\frac1{1-a^2-b^2}
\begin{pmatrix}
(1-a)^2+b^2 & -2b\\
-2b & (1+a)^2+b^2
\end{pmatrix}.
\]

It is symmetric positive definite and satisfies \(\det A=1\). Its ellipticity
constants are bounded in terms of \(k\). We use

\[
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}
\]

for counter-clockwise rotation by \(90^\circ\).

The central question is not merely whether the PDE can be solved. It is whether
the complete map

\[
\mu\longmapsto u\longmapsto v\longmapsto f=u+iv
\]

is a fast differentiable decoder whose fixed-mesh output is a PL homeomorphism.

## 2. Continuum mixed-boundary problem

The primary scalar field \(u\) is defined by

\[
\nabla\!\cdot(A\nabla u)=0\quad\text{in }\Omega,
\]

with mixed boundary conditions

\[
u=0\quad\text{on }\Gamma_L,\qquad
u=1\quad\text{on }\Gamma_R,
\]

\[
n^\top A\nabla u=0\quad\text{on }\Gamma_B\cup\Gamma_T.
\]

The weak formulation is: find \(u\in H^1(\Omega)\) with the prescribed traces
on \(\Gamma_L\cup\Gamma_R\) such that

\[
\int_\Omega \nabla\varphi^\top A\nabla u\,dx=0
\]

for every \(\varphi\in H^1(\Omega)\) whose trace vanishes on
\(\Gamma_L\cup\Gamma_R\). Uniform ellipticity and a nonempty Dirichlet part
give existence and uniqueness by Lax--Milgram. This is a scalar PDE fact; it is
not yet a theorem that the associated two-component map is injective.

Define the flux

\[
q=A\nabla u.
\]

The PDE is \(\nabla\cdot q=0\).

## 3. Conjugate coordinate and the Beltrami equation

Define the one-form

\[
\omega=(Jq)\cdot dx=-q_2\,dx+q_1\,dy.
\]

Its exterior derivative is

\[
d\omega=(\partial_xq_1+\partial_yq_2)\,dx\wedge dy
       =(\nabla\cdot q)\,dx\wedge dy=0.
\]

Because \(\Omega\) is simply connected, there is a scalar potential \(v\), unique
up to an additive constant, such that

\[
\nabla v=J A\nabla u.
\tag{MBM-1}
\]

This implication is proved here in the weak/distributional sense; regularity of
v follows from the elliptic regularity available for the chosen class of
coefficients.

The identity \(\det A=1\) implies

\[
JAJ=-A^{-1}.
\tag{MBM-2}
\]

For \(f=u+iv\), the real Jacobian is

\[
J_f=\det Df
=\det(\nabla u,J A\nabla u)
=\nabla u^\top A\nabla u\ge0.
\tag{MBM-3}
\]

Where \(\nabla u\ne0\), it is strictly positive. Substitution of the displayed
formula for \(A(\mu)\) into (MBM-1) gives

\[
f_{\bar z}=\mu f_z.
\tag{MBM-4}
\]

Thus the conductivity reduction is algebraically equivalent to the Beltrami
equation. The converse is also valid for a sufficiently regular solution with
positive Jacobian.

## 4. What the boundary conditions imply

On \(\Gamma_B\cup\Gamma_T\), \(q\cdot n=0\). If \(t=Jn\) is the positively or
negatively oriented unit tangent, then

\[
\partial_t v=(JA\nabla u)\cdot t=q\cdot n=0.
\]

Therefore \(v\) is constant on each horizontal Neumann arc. On
\(\Gamma_L\cup\Gamma_R\), \(u\) is constant by construction. Consequently the
boundary is mapped into four candidate sides of a rectangle.

This statement is only a side-membership statement. It does not prove that the
free traces of \(v\) on \(\Gamma_L,\Gamma_R\) are strictly ordered, and it does
not prove that \(f\) is globally one-to-one.

## 5. Modulus, energy, and flux

If the side traces are ordered and \(f\) is a homeomorphism onto
\([0,1]\times[0,M]\), then the area formula and (MBM-3) give

\[
M=|f(\Omega)|=\int_\Omega \det Df\,dx
  =\int_\Omega \nabla u^\top A\nabla u\,dx
  =:E_A(u).
\tag{MBM-5}
\]

Integration by parts gives the equivalent right-side flux formula

\[
E_A(u)=\int_{\Gamma_R} (A\nabla u)\cdot n\,ds,
\tag{MBM-6}
\]

because \(u=1\) on \(\Gamma_R\), \(u=0\) on \(\Gamma_L\), and the flux vanishes
on the Neumann arcs. Formula (MBM-5) is therefore a theorem conditional on
global injectivity and correct boundary order; the scalar energy identity
itself does not require that condition.

## 6. Complementary mixed solve

Normalize the conjugate coordinate by

\[
w=\frac{v-v_B}{M},
\]

where \(v_B\) is the constant value on \(\Gamma_B\). Then \(w=0\) on
\(\Gamma_B\) and \(w=1\) on \(\Gamma_T\). Using (MBM-1), (MBM-2), and
\(\nabla u=-MJA\nabla w\), the curl-free condition for \(\nabla u\) becomes

\[
\nabla\!\cdot(A\nabla w)=0.
\tag{MBM-7}
\]

On \(\Gamma_L\cup\Gamma_R\), the tangential derivative of \(u\) vanishes; this
is equivalent to \(n^\top A\nabla w=0\). Therefore the complementary problem
uses the same tensor \(A\):

\[
w=0\ \Gamma_B,\qquad w=1\ \Gamma_T,\qquad
n^\top A\nabla w=0\ \Gamma_L\cup\Gamma_R.
\]

The reciprocal energy identity is

\[
E_A(u)=M^2E_A(w),\qquad E_A(w)=M^{-1}.
\tag{MBM-8}
\]

This corrects a common but incorrect shortcut that replaces \(A\) by \(A^{-1}\)
in the complementary scalar solve.

## 7. Constant-coefficient sanity checks

For real constant \(\mu=a\in(-1,1)\),

\[
A=\operatorname{diag}\left(\frac{1-a}{1+a},\frac{1+a}{1-a}\right).
\]

The mixed problem has \(u(x,y)=x\), \(v(x,y)=\frac{1-a}{1+a}y\), and hence

\[
M=\frac{1-a}{1+a}.
\]

The map \(f(x+iy)=x+iMy\) has Beltrami coefficient \(a\), positive Jacobian
\(M\), and exactly satisfies the rectangular boundary conditions. This is a
complete proved-here sanity check. For a constant coefficient with
\(\operatorname{Im}\mu\ne0\), a single affine function generally cannot satisfy
both the left/right Dirichlet and top/bottom zero-flux conditions; that is a
property of the mixed boundary problem, not a contradiction of the conductivity
reduction.

## 8. P1 finite-element discretization

Let \(\mathcal T_h\) be an oriented triangular mesh and let \(\phi_i\) be the
piecewise-linear nodal basis. With a constant tensor \(A_T\) on each triangle,
the stiffness entries are

\[
K_{ij}=\sum_{T\in\mathcal T_h}|T|
(\nabla\phi_i|_T)^\top A_T(\nabla\phi_j|_T).
\tag{MBM-9}
\]

The discrete primary solve fixes \(u_i=0\) on the left boundary,
\(u_i=1\) on the right boundary, and leaves top/bottom boundary nodes natural.
After solving for all \(u_i\), one can construct a facewise flux

\[
q_T=A_T\nabla u_h|_T,
\]

and reconstruct \(v_h\) in three ways:

- A3-a: solve a second scalar complementary FEM problem;
- A3-b: integrate/project \(Jq_T\) into a nodal potential;
- A3-c: introduce a block system only if an exact mixed formulation is
  derived, rather than forcing a one-solve formulation.

The ordinary nodal projection of \(Jq_T\) is not automatically an exact
discrete conjugate. Therefore a small residual

\[
\|\nabla v_h-JA\nabla u_h\|
\]

must be reported separately from the PDE residual.

## 9. Implementation audit and correction

The previous implementation in
src/qcopt/forward/rectangle_conductivity.py computed the flux right-hand side
from u_boundary before the interior conductivity solve. Since that array held
the caller's entire grid, the result depended on arbitrary interior values.

The correction is:

1. assemble the conductivity matrix;
2. keep only the supplied left/right boundary entries;
3. solve the interior values of u;
4. recompute each face gradient from solved u;
5. construct JA grad u and recover v.

The new regression test corrupts every interior entry of boundary_values while
keeping the boundary fixed. The map remains equal to the manufactured affine
solution to the test tolerance. This is a code-level correctness result, not a
global homeomorphism theorem.

The structured reference implementation also solves the complementary problem
independently. For the smooth field

\[
\mu(x,y)=0.35\sin(2\pi x)\cos(2\pi y)
+0.2i\cos(2\pi x)\sin(2\pi y),
\]

the 16x16, 32x32, and 64x64 vertex grids produced respectively

\[
\begin{array}{c|c|c|c|c}
N & \text{time (s)} & \min\det Df_h & \text{conjugacy residual}
& \text{side traces}\\ \hline
16 & 0.0675 & 0.3441 & 0.1676 & \text{both monotone}\\
32 & 0.2417 & 0.2896 & 0.1009 & \text{both monotone}\\
64 & 1.0098 & 0.2699 & 0.0536 & \text{both monotone}
\end{array}
\]

The decreasing conjugacy residual is consistent with mesh convergence, while
the nonzero residual confirms that an independent P1 complementary solve is not
an exact fixed-mesh discrete conjugate. The positive determinants and ordered
traces are numerical observations for this field, not a general theorem.

## 10. Theorem status

| Claim | Status |
|---|---|
| Mixed scalar BVP has a unique weak solution | proved here under standard uniform ellipticity and nonempty Dirichlet assumptions |
| Divergence-free flux has a stream potential on a simply connected domain | proved here |
| Conductivity relation gives \(f_{\bar z}=\mu f_z\) | proved here algebraically, subject to regularity |
| \(v\) is constant on top/bottom Neumann arcs | proved here |
| \(M=E_A(u)\) | proved here conditional on a global rectangle homeomorphism; energy identity itself unconditional |
| Complementary problem uses the same \(A\) and has reciprocal energy | proved here under the normalized conjugate construction |
| Mixed boundary automatically gives ordered side traces | open; requires theorem or counterexample |
| Positive local Jacobian implies global PL homeomorphism | false without additional global/topological hypotheses |
| P1 projected conjugate is exact for arbitrary facewise \(\mu_T\) | open; exact only on a compatible subspace |

## 11. Targeted references

- [TEMPO: Feature-Endowed Teichmüller Extremal Mappings of Point Clouds](https://www.researchgate.net/publication/284476371_TEMPO_Feature-Endowed_Teichmuller_Extremal_Mappings_of_Point_Clouds) — point-cloud Beltrami/MLS formulation; this is background, not a proof of the mixed-boundary neural layer.
- [A quadrilateral mixed Dirichlet--Neumann construction of a rectangle map](https://math.aalto.fi/~vquach/dippa/dippa_FINAL.pdf) — explicit isotropic mixed problem and rectangle modulus construction.
- [Leonetti--Nesi, Quasiconformal solutions to certain first-order systems](https://www.sciencedirect.com/science/article/pii/S0021782497899473) — conductivity/first-order-system relation and hypotheses needed for quasiconformality.

## 11.1. Scope of sigma-harmonic global theorems

The sigma-harmonic literature is relevant but does not immediately close the
MBM-LBS topology gap. Alessandrini and Nesi formulate a two-component mapping
\(U=(u_1,u_2)\) whose components solve divergence-form elliptic equations with
**full Dirichlet boundary data**. Their global-diffeomorphism criteria are
expressed through the boundary parametrization, including convex-target or
unimodality hypotheses.

The MBM construction instead prescribes Dirichlet data for \(u\) only on
\(\Gamma_L\cup\Gamma_R\), prescribes Neumann data on
\(\Gamma_B\cup\Gamma_T\), and obtains \(v\) as a stream potential. Therefore
one cannot quote a full-Dirichlet sigma-harmonic theorem without proving that
the induced four-arc boundary trace is an admissible unimodal/convex
parametrization. The correct status is:

- sigma-harmonic theory supports the principle that boundary structure, not
  ellipticity alone, controls global injectivity;
- it does not prove mixed-boundary side monotonicity for this decoder;
- a future theorem may be obtained by converting the mixed problem into a
  full-boundary statement with an explicit ordered trace, but that conversion
  is still open.

Reference: [Alessandrini--Nesi, Globally diffeomorphic sigma-harmonic mappings](https://arxiv.org/abs/1906.00902).

## 12. Supporting theory: exact fixed-P1 realizability

The mixed conductivity decoder and a general facewise Beltrami field are not
automatically compatible on a fixed triangulation. Let \(T\) be a source
triangle and write the complex affine restriction of a P1 map as

\[
f_T(z)=a_T z+b_T\overline z+c_T,\qquad b_T=\mu_Ta_T.
\tag{P1-1}
\]

For an oriented source edge \(e=z_j-z_i\), its image increment is

\[
p_T(e)=a_Tq_T(e),\qquad q_T(e)=e+\mu_T\overline e.
\tag{P1-2}
\]

Since \(|\mu_T|<1\), \(q_T(e)\ne0\) for every nonzero source edge. If two
triangles \(T,S\) share an edge, continuity requires

\[
a_Tq_T(e)=a_Sq_S(e).
\tag{P1-3}
\]

Equivalently, the face scales satisfy a multiplicative transition relation
\(a_S/a_T=q_T(e)/q_S(e)\). On a simply connected triangulated disk, a
nonzero scale field exists exactly when the product of these ratios around
every closed dual walk equals one. Once the scales exist, the edge increments
\(p_e\) form a closed primal 1-form, so a vertex map exists up to one complex
translation. The remaining global complex scale is a similarity gauge.

This is the fixed-mesh compatibility theorem used here. It proves exact
realizability for compatible facewise fields, not for arbitrary bounded fields.
For a compatible field, every face has

\[
\det Df_T=|a_T|^2(1-|\mu_T|^2)>0.
\tag{P1-4}
\]

The theorem still needs a separately ordered rectangle boundary to imply a
global PL homeomorphism. Positive face determinants alone are local.

The implication for MBM-LBS is important: an independently solved P1
conjugate can have a nonzero conjugacy residual even when both scalar systems
are solved to machine precision, because arbitrary facewise \(A_T\) need not
belong to the fixed-P1 compatible subspace. A residual is therefore a
discretization/compatibility diagnostic, not merely a linear-solver tolerance.

Legacy realistic-resolution evidence is consistent with this obstruction. A
manufactured 256-squared field had dual-holonomy residual \(1.05\times10^{-14}\)
and reconstruction error \(1.22\times10^{-15}\), while a bounded random field
had holonomy as large as \(7.64\times10^{25}\). The nonlinear projection
prototype recovered compatible manufactured fields through 256-squared meshes,
but arbitrary random face fields retained a nonzero Beltrami mismatch. These
are numerical validations of the compatibility mechanism, not a theorem that a
projection exists for every \(\mu\).

## 13. Consequence for the neural-layer choice

There are now three distinct decoder contracts:

1. MBM-LBS gives a fast scalar elliptic solve and a continuum Beltrami relation,
   but fixed-P1 conjugacy and boundary order need additional hypotheses.
2. Primal-dual electrical mapping builds conjugacy and tiling structure into
   the representation, but anisotropic positive Hodge-star conditions remain
   open.
3. Directed Tutte gives the cleanest hard-bijection certificate, but it is not
   an exact arbitrary-\(\mu\) solver.

The final architecture may therefore be hybrid, but a hybrid is only justified
after its interface states exactly which quantity is preserved: Beltrami
coefficient, metric energy, boundary order, or topology.
