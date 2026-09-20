# Definitions and evaluation metrics

## Beltrami equation

Identify a planar point `(x,y)` with the complex number `z=x+i y`. A map
`f=u+i v` satisfies the Beltrami equation

`f_bar = mu f_z`,

where `f_z` and `f_bar` are the complex derivatives and `mu` is the Beltrami
coefficient. The admissibility condition is `|mu|<1`; it bounds local
anisotropic stretching and is the continuous quasiconformal condition.

## Piecewise-affine map and homeomorphism

A triangular mesh has vertices and oriented triangular faces. A
piecewise-affine map stores a two-dimensional image coordinate at every vertex
and linearly interpolates it inside each face. It is locally orientation
preserving when every mapped face has positive signed area. A global
homeomorphism additionally requires one-to-one global behaviour and compatible
boundary handling; positive face areas alone are a strong local certificate but
are not, by themselves, a proof of global one-to-one behaviour on every domain.

For a face with source edge vectors `e1,e2` and mapped edge vectors `d1,d2`,
the signed-area ratio is

`r = cross(d1,d2) / cross(e1,e2)`.

The audit reports the minimum `r` and the number of faces with `r<=0`.

## Differentiable layer

A layer is differentiable if a small change in its input produces a defined
derivative used by reverse-mode automatic differentiation. A finite gradient
check only proves the tested computational path is numerically differentiable;
it does not prove differentiability at active-set switches, failed solves, or
all possible inputs.

## Core metrics

- **Relative map error:** `||f_test-f_ref||_2 / ||f_ref||_2`.
- **Relative velocity error:** the same ratio for two BHF velocity fields.
- **Residual:** the Euclidean norm of a discrete equation or tensor fit,
  divided by the norm of the reference quantity when a relative residual is
  reported.
- **Directional derivative error:** compare a finite difference
  `[L(theta+h d)-L(theta-h d)]/(2h)` with the reported reverse-mode derivative.
- **Stationarity:** norm of the nonlinear KKT residual at the computed map.
- **Forward/backward time:** wall-clock time for one forward output and one
  reverse pass, including the stated quadrature or iterative solve.
- **Peak memory:** maximum device allocation reported by the runtime. It is
  not the same as total system memory and must be interpreted with the mesh
  size and block sizes.

## Why “zero flips” is not enough

Zero flips is a local orientation result. Boundary crossings, disconnected
components, non-convex domains, and an incorrectly assembled topology can still
produce a globally invalid map. Every claim in this project therefore keeps
the independent topology audit separate from the optimization or solver code.
