# Route F — mesh-native triangular and monotone flows

## 1. Parameterization

The route represents a map as a composition of simple orientation-safe layers.
A triangular layer has the form

`T(x,y)=(T_1(x),T_2(x,y))`.

Its Jacobian is triangular:

`DT=[[partial_xT_1,0],[partial_xT_2,partial_yT_2]]`,

so

`det(DT)=partial_xT_1 * partial_yT_2`.

Discrete positive-increment coordinates use

`Delta_i=exp(s_i)>0`

or a smooth positive transform, followed by cumulative summation. This makes
the one-dimensional map strictly increasing.

## 2. Composition

For layers `T_1,...,T_L`,

`F=T_L o ... o T_1`,

and

`det DF(x)=prod_{ell=1}^L det DT_ell(T_{ell-1}o...oT_1(x))`.

If every factor has positive determinant, the composition is locally
orientation preserving. Under suitable boundary behaviour and global
monotonicity assumptions, this can produce a homeomorphism.

## 3. Differentiation

The layer is an ordinary computational graph. For parameter `theta` and input
point `x`, reverse mode applies

`dL/dtheta=(D_theta F_theta(x))^T dL/dF`.

No sparse matrix adjoint is needed. This is the main attraction for neural
networks.

## 4. Expressivity boundary

The positivity theorem concerns the chosen triangular family. It does not imply
that for every `mu` there exists parameters `theta` satisfying

`mu_T(F_theta)=mu_T^target`.

High-resolution experiments showed zero flips for the tested family, but no
mesh-independent arbitrary-QC density theorem has been established.

## 5. Role in the final architecture

Route F is best used as a topology-safe latent decoder or as the prolongation
operator inside a multilevel BHF method. It is not, by itself, an inverse
Beltrami solver.
