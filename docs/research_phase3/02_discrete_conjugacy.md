# Structure-preserving discrete conjugacy

Status: the exact compatible-\(P_1\) theorem and a manufactured test are
implemented in `src/qcopt/forward/discrete_conjugacy.py`; the general neural
layer and large-mesh memory study remain open.

## 1. Finite-element setting

Let \(\mathcal T_h\) be a conforming, positively oriented triangulation of a
simply connected polygon. Let \(u_h,v_h\) be continuous functions that are
affine on every triangle. On a face \(T\), write
\[
p_T=\nabla u_h|_T,qquad r_T=\nabla v_h|_T,qquad
A_T=A_T^T\succ0,\qquad \det A_T=1.
\]
The discrete conductivity and conjugacy equations are
\[
r_T=J A_Tp_T,\qquad J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
\tag{DC-1}
\]
The facewise Beltrami coefficient is then the unique coefficient satisfying
\(f_{\bar z}=\mu_T f_z\) for \(f_h=u_h+\mathrm{i}v_h\), and positivity of
\(\det Df_h\) is checked face by face.

## 2. Why compatibility implies the primary weak equation

Because \(p_T\) and \(A_T\) are constant, \(q_T=A_Tp_T\) has zero divergence
inside every face. Let \(e\) be an interior edge with unit tangent \(t\) and
one chosen normal \(n=Jt\). Continuity of \(v_h\) gives equality of its
tangential derivatives from the two adjacent faces. Using (DC-1),
\[
t\cdot r_T=t\cdot Jq_T=-n\cdot q_T.
\]
Therefore the normal flux \(q_T\cdot n\) is continuous across every interior
edge. The assembled field \(q_h\) is in the discrete \(H(\mathrm{div})\) space,
and elementwise integration by parts yields
\[
\sum_T\int_T A_T\nabla u_h\cdot\nabla\phi_h=0
\]
for every \(P_1\) test function whose trace vanishes on the Dirichlet arcs,
provided the boundary conormal condition holds on the Neumann arcs. Thus a
compatible \(f_h\) makes \(u_h\) an *exact* finite-element solution of the
facewise conductivity problem.

## 3. Why the same tensor solves the complementary problem

From (DC-1) and \(A_T J A_T=J\) for determinant-one symmetric tensors,
\[
A_T r_T=A_TJA_Tp_T=Jp_T.
\]
Hence \(\operatorname{div}(A_T\nabla v_h)=0\) on each face. The same interior
edge argument uses continuity of \(u_h\) to give the complementary conormal
continuity. If \(u_h\) is constant on the left/right arcs and \(v_h\) is
constant on the bottom/top arcs, the two mixed boundary problems are exactly
the two coordinates of one compatible \(P_1\) map.

This is a sufficient construction theorem, not an assertion that two
independently solved mixed systems are automatically compatible. In general,
the independently computed \(v_h\) need not lie in the stream-coordinate
subspace determined by the primary flux.

## 4. Manufactured layered homeomorphism

On the unit rectangle, choose a strictly increasing piecewise-linear function
\[
g(0)=0,\qquad g(1/2)=0.2,\qquad g(1)=1,
\]
with slopes \(s_1=0.4\) and \(s_2=1.6\). Set
\[
u_h(x,y)=x,qquad v_h(x,y)=g(y),qquad
A_T=\operatorname{diag}(s_j,s_j^{-1})
\]
on the horizontal strip with slope \(s_j\). Then
\[
J A_T\nabla u_h=J(s_j,0)^T=(0,s_j)^T=\nabla v_h,
\]
so (DC-1) is exact. Every face determinant is \(s_j>0\), and the map is a
global piecewise-affine homeomorphism because it preserves \(x\) and \(g\) is
strictly increasing. The assembled stiffness residual vanishes at every
non-left/right vertex, while the complementary residual vanishes at every
non-bottom/top vertex.

The regression test uses a (4\times2)-cell rectangle and checks the
facewise conjugacy residual, both mixed weak residuals, and positive triangle
determinants. It is deliberately not a generic solver benchmark: it verifies
the exact compatibility theorem on a nonconstant coefficient.

## 5. Structure-preserving implementation requirements

A general layer based on this theorem must carry, or reconstruct, all of:

1. a conforming primal triangulation and its oriented edge incidence;
2. face tensors with \(\det A_T=1\) (or an explicitly generalized Hodge star);
3. a flux space in which interior normal traces cancel exactly;
4. a dual stream integration with a gauge and cycle-consistency check;
5. face determinant and boundary-order certificates.

Dropping item 3 or 4 gives the familiar sparse linear solve but loses the
discrete conjugacy invariant. Enforcing all five with a million-vertex neural
batch requires matrix-free incidence operations, checkpointed solves, and
careful storage of only edge fluxes rather than two full coordinate histories.

### Incidence/Hodge formulation (feasibility derivation)

For completeness, let \(B_0\in\mathbb R^{n_e\times n_v}\) be the oriented
vertex-to-edge incidence matrix and \(B_1\in\mathbb R^{n_f\times n_e}\) the
oriented edge-to-face incidence matrix. The chain-complex identity is
\[
B_1B_0=0.
\]
Let \(H_A\) be a positive (possibly block) discrete Hodge star on primal edge
one-forms. The primal potential and flux equations are
\[
q=H_A B_0u,qquad B_1q=0.
\tag{DC-2}
\]
On a simply connected dual complex, let \(\widetilde B_0\) be the dual
vertex-to-edge incidence. A stream coordinate is obtained by solving
\[
\widetilde B_0v=R q,
\tag{DC-3}
\]
where \(R\) rotates the oriented primal flux into the dual orientation. A
solution exists exactly when the right side has zero sum around every dual
cycle; this is the discrete exactness condition. One additive constant is fixed
as the gauge, and boundary dual edges encode the Neumann/Dirichlet side data.

The present code does **not** claim to implement (DC-2)--(DC-3) on an
arbitrary mesh. It implements the compatible \(P_1\) route above, where
continuity of \(u_h,v_h\) supplies the normal-flux cancellation analytically.
The incidence/Hodge equations identify the missing data structures for a
future matrix-free layer and explain why an independent complementary solve is
not automatically a stream integration.

### Edge-integrated stream prototype

The repository now contains a deliberately small diagnostic implementation,
`integrate_stream_from_face_flux`. For every oriented primal edge
\(e=(x_i,x_j)\) on a face \(T\), it forms the predicted stream increment

\[
\delta_e^T=(J A_T\nabla u_h|_T)\cdot(x_j-x_i).
\tag{DC-4}
\]

The two incident faces must give the same signed value; their maximum spread is
the **edge-inconsistency** statistic. Boundary edges have a single incident
face and therefore contribute zero to this particular mismatch statistic. The
mean edge value is then integrated on a spanning tree after fixing one gauge
vertex \(v_{i_0}=0\). Every non-tree edge supplies a cycle equation

\[
v_j-v_i=\bar\delta_{ij}.
\tag{DC-5}
\]

and the maximum violation is the **cycle residual**; tree-edge residuals vanish
by construction, so the non-tree edges carry the cycle information. The returned boolean is
true only when both quantities are below the requested tolerance. On the
layered manufactured homeomorphism, both are below \(10^{-12}\) and the
reconstructed stream agrees with \(v_h\) up to its additive gauge. For a
random incompatible \(P_1\) potential with \(A_T=I\), the diagnostic rejects
the field. This is evidence that a structure-preserving decoder can integrate
a compatible flux without solving an independent complementary Dirichlet
problem; it is not yet a general dual-complex Hodge star, boundary treatment,
or million-vertex neural layer.

## 6. Current conclusion

The compatible \(P_1\) route can produce an exact piecewise-affine
homeomorphism with exact forward conjugacy and a well-defined adjoint for the
assembled sparse system. It is not yet a fast universal decoder from arbitrary
facewise \(\mu\); realizability, sparse factorization cost, and global boundary
certification are still the decisive open questions.
