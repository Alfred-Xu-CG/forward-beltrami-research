# Route B — Beurling transform, FFT, padding, and nonuniform evaluation

## 1. Whole-plane equation

Let `F=u+iv` solve `F_barz=mu F_z` on the complex plane, with compactly
supported `mu`. Introduce `h=F_barz`. A Cauchy-transform identity gives

`F_z=1+B h`,

where `B` is the Beurling transform. Substitution gives the fixed-point equation

`h=mu(1+B h)`,

or the linear-looking operator equation

`(I-M_mu B)h=mu`.                                   (B.1)

The equation is linear in `h` when `mu` is fixed, but the operator is dense and
singular-integral based.

## 2. Periodic Fourier discretization

On a uniform `n_x` by `n_y` torus, the discrete Fourier transform diagonalizes
the periodic operator. For nonzero frequency `k=(k_x,k_y)`, the implementation
uses

`B_hat(k)=(k_x-i k_y)/(k_x+i k_y)`,

with the zero mode set to zero. Hence

`B_h=hifft2(B_hat .* fft2(h))`.                        (B.2)

The zero mode is a mathematical nullspace, not a numerical accident. A
quasi-periodic affine lift must retain it separately. If `m=mean(h)`, the code
uses

`F(z)=z+m conjugate(z)+C(h-m)`,

so that the two periods are `1+m` and `i(1-m)`.

## 3. Fixed-point solver

Starting from `h^0=0`, the implemented iteration is

`h^{r+1}=mu .* (1+B h^r)`.                         (B.3)

The stopping criterion is the maximum norm

`||h^{r+1}-h^r||_infty <= tolerance`.

The iteration is contractive only in regimes controlled by `||mu||` and the
operator norm of `B`; a finite iteration count is not a universal convergence
theorem.

## 4. Bounded window and zero padding

For a bounded sample window, the code embeds `mu` into a larger zero-padded
array and applies the periodic symbol on the larger box. This approximates the
free-space transform because periodic copies are farther apart. It remains a
periodic approximation on a finite box, and its error contains box-tail,
frequency, interpolation, quadrature, and principal-value components.

The experiment with padding factors `2,4,8` at 512² gave relative errors
`7.07e-2`, `4.41e-3`, and `2.76e-4` against an independent direct reference.
This confirms convergence with padding but also shows that padding factor 2 is
not accurate enough for a strict solver claim.

## 5. Implicit VJP

Let `A=I-M_mu B` and `Ah=mu`. For a real loss with cotangent `g_h`, solve

`A^* lambda=g_h`.                                  (B.4)

Since

`A dh=dmu+dM_mu(Bh)`,

the real-inner-product coefficient VJP is

`dL/dmu=lambda .* conjugate(1+B h)`.                  (B.5)

The code implements the adjoint solve with GMRES and the conjugate Fourier
multiplier. This is a valid derivative of the discrete operator when the solve
converges. It is not a derivative of an unimplemented bounded-domain theorem.

## 6. Nonuniform mesh alternatives

On scattered points the FFT no longer diagonalizes the operator. Direct pair
evaluation has `O(N^2)` cost. Treecode/FMM methods approximate distant clusters,
while near source-target pairs require special quadrature or principal-value
cancellation. Particle-mesh methods regain FFT speed by interpolating between
points and a uniform grid, but then introduce interpolation and deconvolution
errors.

The treecode controls agreed with direct GPU sums at roughly `10^-7` relative
error for separated clouds. A paired offset of `10^-4` produced approximately
`4.53e-4` error, identifying the near singularity as the limiting component.

## 7. Topology and decision

The Beurling fixed-point equation does not by itself constrain every discrete
triangle determinant. A map can have a small operator residual and still fold
after reconstruction. Consequently a separate P1 orientation audit is needed.
Route B is an excellent fast operator for periodic/uniform settings, but a
general bounded-domain hard-bijective layer requires a production singular
quadrature/FMM backend, boundary closure, and a topology certificate.
