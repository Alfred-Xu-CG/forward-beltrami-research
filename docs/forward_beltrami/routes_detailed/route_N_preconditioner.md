# Route N — learned and Neumann preconditioners

## 1. Linear-solver setting

Suppose a forward layer requires solving

`A(theta)x=b(theta)`.

The solution is unchanged by a preconditioner; only the iterative path changes.
For a split `A=I-K`, a truncated Neumann approximation is

`x_0^(p)=sum_{j=0}^p K^j b`.

It is used as a warm start for GMRES/CG, or as a left/right preconditioner when
the implementation supports it.

## 2. Differentiation

If the final solve is differentiated implicitly, the exact solution derivative
still satisfies

`A dx=-A_theta x+b_theta`.

The preconditioner affects iteration count and numerical residual, not this
equation. If one differentiates through a truncated initializer instead, the
gradient is the derivative of the approximation and must be reported as such.

## 3. Experimental interpretation

The controls reduced iteration counts for some 512² fields, with cost-inclusive
speedups up to about `1.19x`, while other smooth/random fields had only about
`1.02x` or slowdowns. This variability is expected because Neumann truncations
depend on the spectrum and nonnormality of `K`.

## 4. Decision

Route N is an acceleration layer for an existing sparse or implicit solver. It
cannot supply a piecewise-affine map, a topology certificate, or arbitrary
Beltrami expressivity by itself.
