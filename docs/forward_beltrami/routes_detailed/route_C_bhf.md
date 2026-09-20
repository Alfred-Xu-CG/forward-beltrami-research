# Route C — finite-element Beltrami holomorphic flow

## 1. Continuous motivation

Beltrami holomorphic flow evolves a map rather than solving one global sparse
linear system. The state is a map `F_t`, and a coefficient variation `nu`
induces a holomorphic velocity obtained from a Beurling-type singular integral.
In a schematic continuous form,

`partial_t F_t(w)=V(F_t,nu)(w)`,

`V(w)=-(1/pi) integral K(F_t(zeta),F_t(w)) nu(zeta) (F_t)_z(zeta)^2 dA(zeta)`.

The exact kernel and normalization depend on the domain and boundary model; the
implementation uses the image-plane kernel plus the regularization terms in
`bhf_torch.py`.

## 2. P1 discretization

For each face `T`, compute the constant `fz_T=(F_U)_z|_T` and store a face
variation `nu_T`. The face contribution at a target vertex is approximated by
quadrature over `T`:

`V_T(w) approx -(nu_T fz_T^2/pi) sum_q omega_q K(F(q),F(w)) |det DF_T|`.

Incident target faces contain the singular point and are integrated with a
Duffy transformation. Nonincident faces use a fixed piecewise-affine quadrature
rule. Far interactions are evaluated in target blocks to avoid materializing
the complete dense matrix.

## 3. Explicit flow layer

Let `V(U^n)` denote the assembled vertex velocity. The explicit Euler layer is

`U^{n+1}=U^n+tau_n V(U^n)`.                       (C.1)

The multi-step output is the composition of these maps. The computational graph
contains dependence through `fz_T`, image positions, quadrature Jacobians, and
the singular kernel.

## 4. Determinant-safe controller

For a face with current edges `e_1,e_2` and velocity edges `de_1,de_2`,

`det(e_1+t de_1,e_2+t de_2)=c+b t+a t^2`.

Given margin `m>0`, compute the first positive root `r_T` of

`c+b t+a t^2-m=0`.

The exact controller selects

`tau=min(tau_requested, safety * min_T r_T)`.       (C.2)

If all current determinants exceed `m`, this preserves the margin up to
floating-point effects. The active face is discrete, so its exact derivative is
piecewise-defined. A smooth conservative controller replaces hard minimum and
absolute values by smooth lower envelopes; this buys a differentiable surrogate
at the cost of extra conservatism.

## 5. Backward implementation

The memory-bounded layer stores initial state and static arrays, then replays all
flow steps during backward. If `Phi_n(U)=U+tau_nV(U)`, the reverse recurrence is

`g_n=(D Phi_n(U^n))^T g_{n+1}`.

Replaying avoids storing every dense pair graph but repeats the quadrature cost.
The method is checkpoint/recompute differentiation, not an implicit adjoint of
the continuous BHF equation.

## 6. Coarse-to-fine formulation

For a coarse mesh map `U_c`, differentiable prolongation `P`, and fine flow,

`U_out=Phi_f^(s_f)(P Phi_c^(s_c)(U_0))`.

The VJP is

`g_0=(D Phi_c)^T P^T (D Phi_f)^T g_out`.

The 32²-to-128² experiment used two coarse and two fine steps. Relative map
error against four fine steps was `1.80e-7`; forward/backward times were
`13.81/51.33 s` versus `19.94/97.95 s`; peak CUDA allocation was about 4.91 GB.

The attempted 64²-to-256² run used about 24 GB and exceeded a practical budget,
so the multilevel idea is validated locally but not yet production-scalable.

## 7. Topology, accuracy, and decision

The independent audits found zero flips on tested structured and jittered
Delaunay meshes. This is strong numerical evidence for those inputs. It is not
a theorem for arbitrary `nu`, step count, mesh quality, or boundary coupling.
The unresolved mathematical gates are global principal-value consistency,
mesh-independent homeomorphism, smooth active-set differentiation, and an
efficient fine-grid quadrature or FMM.
