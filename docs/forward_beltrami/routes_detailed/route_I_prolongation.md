# Route I — coarse-to-fine prolongation and injective refinement

## 1. Linear prolongation

Let `U_c` contain map values on a coarse nested grid and `U_f` values on a fine
grid. Standard bilinear prolongation is a sparse linear map

`U_f=P U_c`.

For a fine vertex at normalized coordinates `(s,t)`,

`U_f=(1-s)(1-t)U_{00}+s(1-t)U_{10}+(1-s)tU_{01}+stU_{11}`.

The derivative is exactly `dU_f=P dU_c`, so the operation is suitable for
automatic differentiation.

## 2. Folding counterexample

Even if the four coarse corner positions form a valid quadrilateral, bilinear
interpolation can have a Jacobian whose determinant changes sign inside the
cell. Consequently, ordinary interpolation does not preserve a homeomorphism
without additional shape constraints.

## 3. Positive and triangular alternatives

The experiments therefore also prolongate positive one-dimensional increments
in log-space and use coupled triangular corrections. If an increment is

`delta_i=exp(s_i)`,

its refined values remain positive after interpolation of `s_i`. A triangular
coupling controls cross-coordinate shear while retaining a positive diagonal
Jacobian.

## 4. Relation to BHF

For BHF, prolongation is inserted between coarse and fine flow operators:

`U_out=Phi_f(P Phi_c(U_0))`.

This reduces expensive fine steps and preserves a formal gradient chain. The
topology of `P Phi_c(U_0)` still requires an independent determinant audit or a
certified constrained prolongation theorem.

## 5. Decision

Route I is a multilevel transport mechanism, not an independent Beltrami
solver. Its value is reducing fine-grid work and memory when paired with BHF,
barriers, or a topology-safe decoder.
