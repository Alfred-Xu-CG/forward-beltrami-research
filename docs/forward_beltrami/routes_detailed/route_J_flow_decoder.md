# Route J — learned flow matching and hard latent decoders

## 1. Generative formulation

Let `z` be a low-dimensional positive latent representation and let

`U=D_theta(z)`

be a decoder producing vertex coordinates. Conditional flow matching learns a
time-dependent vector field `v_theta(t,z,c)` for condition `c` by minimizing

`E_{t,z_0,z_1} ||v_theta(t,z_t,c)-(z_1-z_0)||^2`,

where `z_t` is a chosen interpolation between endpoints. Sampling solves

`dz/dt=v_theta(t,z,c)`

with Euler or another ODE integrator.

## 2. Hard topology parameterization

The decoder does not directly output unrestricted vertex coordinates. It uses
positive increments, triangular couplings, or compositions whose diagonal
derivatives are positive. If each layer is orientation preserving, the
composition is locally orientation preserving.

## 3. Training/inference distinction

The learned model approximates a distribution of maps. A forward Beltrami
solver would instead receive a specific `mu` and need to solve for `U`. This
requires an encoder, conditioning mechanism, or optimization/projection in the
latent space:

`z*=argmin_z sum_T |mu_T(D_theta(z))-mu_T^target|^2`.

That latent inverse is a new nonlinear problem and is not automatically fast or
globally unique.

## 4. Evidence and limitation

The high-resolution flow-matching controls produced 512² maps with zero flips
and minimum determinant around `1.57e-6`. These results validate the hard
decoder parameterization, not arbitrary Beltrami expressivity. Route J is best
viewed as a learned parameterization or initializer.
