# Route G — determinant-root certified piecewise-linear continuation

## 1. Directional update

Given a current P1 map `U` and a proposed P1 velocity `Delta U`, define

`U(t)=U+t Delta U`.

For each triangle, the two mapped edge vectors are affine in `t`, so its signed
area is exactly

`d_T(t)=c_T+b_Tt+a_Tt^2`.                         (G.1)

The coefficients are obtained from two-dimensional cross products.

## 2. Safe step

Let `m>0` be a determinant margin. For each triangle find the smallest positive
root of

`a_Tt^2+b_Tt+(c_T-m)=0`.

Define

`t_safe=min(t_requested, eta min_T t_T^*)`, `0<eta<1`. (G.2)

If all `c_T>m`, then every triangle remains above the margin for
`0<=t<=t_safe`, up to floating-point error.

## 3. Why it is only piecewise differentiable

The forward map contains a discrete selection

`T_active=argmin_T t_T^*`.

Within a region where `T_active` is unchanged, one can differentiate the
selected quadratic root by the implicit equation

`(2a t+b)dt=-(t^2 da+t db+dc)`.

When the active face changes, the derivative can jump. The implemented fixed-
active VJP is therefore a local derivative, not a globally smooth derivative.

## 4. Homotopy use

For a large target deformation, a direct step may be too large or encounter a
root immediately. Route G constructs a sequence of intermediate targets

`theta_0 -> theta_1 -> ... -> theta_K`

and applies (G.2) on every segment. Experiments repaired a direct rotation-plus-
shear stall using two successful half-parameter segments with zero flips.

## 5. Decision

Route G is a certification and continuation mechanism. It can be attached to
BHF, a learned flow, or a decoder, but it does not prescribe the velocity or
solve the Beltrami equation itself.
