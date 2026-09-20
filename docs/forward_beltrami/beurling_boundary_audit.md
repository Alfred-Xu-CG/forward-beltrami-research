# Route B: Periodic vs zero-padded Beurling audit

## What is implemented

`qcopt.forward.beurling` implements the flat-torus Fourier symbols

\[
 \widehat{\mathcal B h}(k)=
 \frac{k_x-i k_y}{k_x+i k_y}\widehat h(k),
 \qquad
 \widehat{\bar\partial^{-1}h}(k)=
 \frac{2}{i(k_x+i k_y)}\widehat h(k),
\]

with the zero Fourier mode explicitly set to zero. This is an exact discrete
periodic multiplier for the sampled Fourier grid, not an exact free-space or
rectangle-boundary operator. `zero_padded_beurling_apply` embeds a bounded-window
field in a larger periodic box and crops the result; it is a whole-plane
truncation approximation, not a boundary-value solve.

## TDD evidence

Fourier-mode tests pass for the multiplier, the zero mode is annihilated by both
operators, and a compact source produces a measurable difference between the
periodic and zero-padded operators. The test suite also checks that the
difference is not silently treated as solver residual.

## 512x512 boundary audit

The source is a Gaussian centered at `(0.04, 0.5)` with `sigma=0.03`, so it is
close to the left boundary but compact inside the sampled window. Results:

| operator | max difference from periodic | L2 difference | opposite-boundary difference | time |
|---|---:|---:|---:|---:|
| zero padding x2 | 0.520353 | 0.028432 | 0.520353 | 0.300 s |
| zero padding x4 | 0.521334 | 0.028542 | 0.521334 | 1.133 s |

The opposite-boundary response is approximately `0.52` for both padding sizes.
Thus the ordinary periodic FFT and a centered zero-padded whole-plane
approximation are materially different at this resolution. Increasing padding
does not impose the rectangle boundary condition; it only changes the
free-space truncation window. A bounded-domain Beltrami solver still needs a
boundary formulation (for example a matched Cauchy/\bar∂ BVP, reflection
extension, or an independent finite-element/Krylov discretization).

## Consequence for the original concern

The concern about implicit periodicity is valid. Setting `mu=0` outside a
bounded domain is a whole-plane modeling choice, while an unpadded FFT evaluates
the periodic extension of the sampled field. These coincide only under special
compatibility/decay conditions. Uniform grids enable the fast multiplier; a
general unstructured mesh does not inherit this diagonal Fourier representation
and requires a nonuniform transform, a matrix-free integral operator, or a
different discretization/preconditioner.

This report does not yet claim convergence of the Daripa/Gaidashev-Khmelev
iteration for a rectangle. That is the next Route B milestone, and it must be
tested with the same residual/topology contract as the other routes.

## 512² disjoint free-space direct cross-check

To separate periodic-image error from the singular self-term issue, a new
`512x512` source grid (`262,144` samples) was compared against `4,096`
targets in a disjoint block on the opposite side of the square. The source
density was a smooth compact Gaussian-modulated field, and the independent
AI-GPU direct evaluator processed `1,073,741,824` source-target interactions
in `0.302 s` without periodic wrapping. Because source and target sets are
disjoint, no principal-value self-term is involved.

Against that direct whole-plane quadrature control, centered zero-padded FFT
errors decreased systematically with the embedding box:

| padding factor | relative L2 error | max absolute error | FFT time |
|---:|---:|---:|---:|
| 2 | `7.0711e-2` | `1.3440e-3` | `0.252 s` |
| 4 | `4.4131e-3` | `8.4448e-5` | `1.016 s` |
| 8 | `2.7581e-4` | `5.2804e-6` | `4.347 s` |

Receipt:
`artifacts/beurling_free_space_fft_direct_512/beurling_free_space_fft_direct_audit.json`.
This is positive evidence that zero-padding can converge to a free-space
whole-plane operator when source/target geometry is separated; it does not
turn the operator into a rectangle-boundary solve, and the production
singular/nonuniform NUFFT/FMM gate remains open.

## Matched rectangle FD versus LSQC control

The rectangle question was then tested with the same manufactured smooth map
and the same full Dirichlet boundary values. At both `64x64` and `96x96`
cells, facewise LSQC with the exact manufactured P1 `mu` recovered the map to
`1.57e-15` and `2.33e-15` RMS error, respectively. The central-difference FD
operator used its own grid-sampled `mu` and reached equation residuals
`2.13e-11` and `5.42e-11`, but its recovered-map RMS errors were `0.359` and
`0.357`; the FD and LSQC maps differed by the same amount. This is a useful
negative result: a small discrete [1m[0mequation residual[0m does not certify the rectangle
map, because the naive centered complex FD system retains a checkerboard-like
nullspace/selection ambiguity. A boundary-adapted staggered, least-squares, or
P1-compatible operator is still required. Receipt:
`artifacts/rectangle_matched_solver_audit/rectangle_matched_solver_audit.json`.

The same matched experiment was extended to a `256x256`-cell mesh (`131,072`
triangles). The collocated FD solve took `10.47 s`, reached equation residual
`2.86e-10`, but still had map RMS error `0.35494`; facewise LSQC took `1.04 s`,
had primal residual `5.55e-15`, and recovered the manufactured map to
`9.10e-15`. Thus the checkerboard/selection ambiguity is not a coarse-grid
artifact and cannot be repaired merely by increasing resolution. Receipt:
`artifacts/rectangle_matched_solver_audit_256/rectangle_matched_solver_audit.json`.

## Boundary-adapted conductivity rectangle control

To remove the collocated first-derivative ambiguity, a conductivity finite
element control was added. It solves the SPD conductivity equation for
`u=Re(f)` and recovers `v=Im(f)` from the weak relation
`grad(v)=J A(mu) grad(u)`, using the supplied compatible Dirichlet boundary
data. On the manufactured map
`f(x,y)=x+0.08 sin(2 pi y)+i y`, the 65/129/257-node-per-axis audits used
4,096/16,384/65,536 cells (8,192/32,768/131,072 triangles). RMS map error
decreased from `3.65e-5` to `9.19e-6` to `2.31e-6`, while the minimum
triangle determinant increased from `0.99956` to `0.99997`. The Beltrami
equation residuals were `4.13e-3`, `2.06e-3`, and `1.03e-3`, and the
boundary tangential compatibility defect decreased at the same rate. In the
same runs the collocated central-difference matrix was exactly singular and
returned NaNs at all three resolutions, directly exposing the checkerboard
failure rather than hiding it behind a small residual. Receipt:
`artifacts/rectangle_conductivity_audit_256/rectangle_conductivity_audit.json`.

This closes the rectangle boundary-adapted benchmark control, not the whole
Route B goal: the conductivity solve is a Dirichlet reference/control and
does not by itself provide a boundary-free arbitrary-`mu` differentiable
decoder.

## 1024x1024 remote confirmation

The same NumPy audit ran on `ai-codex-mihomo-codex` at `1024x1024` (1,048,576
samples). The maximum periodic-versus-zero-padded difference was `0.531205`
for padding x2 and `0.532185` for padding x4; the opposite-boundary values
were identical to those maxima. Peak resident memory was about `1.77 GiB`.
This confirms that the boundary discrepancy persists at realistic resolution
and is not a coarse-grid artifact.

## Periodic fixed-point audit at 512x512

The periodic Neumann iteration `h = mu(1+B h)` was run on two coefficients:

- mean-zero Fourier mode with amplitude `0.2`: 15 iterations, fixed-point
  residual `3.28e-11`, reconstructed equation residual `6.57e-12`;
- constant coefficient `0.2+0.1i`: 2 iterations and fixed-point residual exactly
  zero, but reconstructed equation residual `0.223607` because the periodic
  Cauchy inverse removes the constant mode.

This is a direct high-resolution demonstration that fixed-point convergence is
not sufficient evidence of a valid Beltrami map. The constant mode must be
handled as a torus affine/quasi-periodic part (or by a rectangle/whole-plane
normalization); dropping it silently produces a wrong map.

## Zero-padded whole-plane fixed-point audit

The bounded-window solver was also checked on a compact Gaussian coefficient at
`48x40`, with padding factor four. The Neumann iteration converged, but the raw
reconstructed residual was `5.0265489e-4`, not machine precision. This is not
an iteration failure: it equals the mean of the padded `h` field, because the
periodic FFT inverse sets the zero Fourier mode of `\bar\partial^{-1}` to zero.
After adding this explicitly reported normalization defect, the local residual
was `1.26e-12`. The result is therefore a useful whole-plane truncation
diagnostic, not a normalized bounded-domain solution. A production solver must
either carry the missing affine/logarithmic normalization as an additional
unknown or impose a boundary condition that determines it; silently dropping
the mode would give a false bijectivity/residual certificate.

The explicit constant-coefficient lift is `f(z)=z+mu*conj(z)`, with
`f_z=1`, `f_bar=mu`, and quasi-periods `1+mu` and `i(1-mu)`. This restores the
Beltrami equation only by changing the torus modulus/period data; it is not a
periodic zero-mean correction. Any torus neural layer must therefore expose the
affine period parameters instead of discarding the zero mode.

## Matrix-free GMRES baseline

`periodic_beltrami_gmres` solves the same periodic operator without assembling a
dense matrix. At `512x512`, a mean-zero Fourier coefficient with amplitude
`0.35` converged in `22` FFT-based iterations and `2.57 s`; operator residual
was `4.40e-11` and the reconstructed map residual was `4.41e-11`. This is a
useful differentiable-layer reference target, but it remains a torus solver;
the earlier boundary audit still applies before interpreting it as a rectangle
or whole-plane method.

At `256x256`, a tolerance sweep on the same amplitude-`0.45` mean-zero mode
gave 12, 18, 23, and 29 GMRES iterations for relative tolerances
`1e-4`, `1e-6`, `1e-8`, and `1e-10`. The operator and map residuals tracked one
another (`4.50e-5` down to `5.73e-11`). This supports a matrix-free implicit
solve interface, but does not yet provide the adjoint/VJP needed for a
memory-bounded neural layer. Receipt:
`artifacts/beurling_gmres_tolerance_audit/beurling_gmres_tolerance_audit.json`.

The missing differentiation piece is now implemented for the periodic linear
problem. `periodic_beltrami_implicit_vjp` solves the conjugate-transpose
matrix-free GMRES system and applies the exact real-inner-product VJP; a finite
difference test agrees to `2e-6`. At `256x256`, forward solve took `0.462 s`
and the adjoint VJP `0.319 s` at tolerance `1e-10`, without storing an
unrolled iteration history. This closes the implicit-differentiation gate for
the torus operator only; rectangle boundary and zero-mode normalization remain
separate issues.

## Torus affine hard layer

For constant `mu`, `torus_affine_grid` constructs the half-open periodic
connectivity together with the explicit affine lift periods `1+mu` and
`i(1-mu)`. The lifted Jacobian is `1-|mu|^2>0`; seam edges must be interpreted
with those periods, not by naive Euclidean differences between fundamental-cell
coordinates. This is a valid hard torus layer for the constant/affine mode,
while a positive-weight periodic embedding for spatially varying `mu` remains
open.

The affine torus layer now also exposes the inverse period parameterization:
given complex periods `p` and `q`, it recovers
`a=(p-iq)/2`, `b=(p+iq)/2`, and `mu=b/a`, rejecting degenerate or
non-quasiconformal moduli. This makes target-modulus learning explicit instead
of hiding the affine anti-holomorphic mode inside a zero-mean FFT correction.

## Correctness-preserving warm starts

`periodic_beltrami_gmres` accepts an optional initial guess while retaining the
same residual-certified linear root. At `256x256`, the zero guess required 29
iterations and the simple `h_0=mu` warm start required 28; both ended at
`5.72e-11` operator residual and agreed to below `1e-8`. This is a small but
important Route-N baseline: a learned initializer may reduce Krylov work, but
the final solver and residual certificate remain responsible for correctness.

## Nonuniform-grid cost audit

As a control, a blocked direct quadrature was run on jittered points. It took
`0.057 s`, `0.300 s`, and `1.046 s` for `N=1024`, `2048`, and `4096`,
respectively, with the expected superlinear/O(`N^2`) growth. This routine
omits the principal-value self term and is explicitly marked diagnostic; its
large output variability is evidence that naive point quadrature is not an
accurate singular-integral solver. The result still establishes the practical
gap motivating NUFFT/FMM or a mesh-native discretization: an unstructured mesh
does not inherit the uniform FFT's diagonal multiplier. Receipt:
`artifacts/beurling_nonuniform_cost_audit/beurling_nonuniform_cost_audit.json`.

## Learned-initializer toy

A six-example least-squares fit of the scalar initializer `h0=alpha*mu`
learned `alpha=1.000000000000007-8.8e-19i`. On a held-out `128²` coefficient,
GMRES iterations fell from 27 to 26 while both residuals stayed at
`3.98e-11` and the final roots differed by `3.0e-16`. This is deliberately
modest evidence: it shows how a learned initializer can be inserted without
delegating correctness, but not a meaningful learned preconditioner speedup.
Receipt: `artifacts/learned_initializer_audit/learned_initializer_audit.json`.

An analytic truncated-Neumann initializer was added as a stronger
correctness-preserving baseline: on the same `128²` held-out mode, order two
reduced GMRES iterations from `27` to `24`, with residual `3.98e-11` and root
difference `3.64e-16`. It costs two extra FFT operator applications to build,
so it is evidence for a transparent preconditioner/initializer, not yet a
learned speedup. Receipt:
`artifacts/learned_initializer_audit_v2/learned_initializer_audit.json`.

A two-feature learned polynomial initializer `c0*mu+c1*mu*(B mu)` was also
fit from six solved examples. At `256²`, it reduced GMRES iterations from 25 to
23, included initializer cost in the measured inference time (`0.519 s` cold
versus `0.503 s` seeded), and preserved the certified root to `4.74e-16`.
This is a small net speedup (`1.033x`), not evidence of a full learned
preconditioner; training cost and more heterogeneous coefficients remain open.
Receipt: `artifacts/learned_polynomial_initializer_audit/learned_polynomial_initializer_audit.json`.

A heterogeneous stress audit then trained on 12 two-frequency fields and tested
four held-out amplitudes `0.24, 0.38, 0.52, 0.62` at `256²`. The seed reduced
GMRES iterations by two in every case and preserved the root to roughly
`1e-13` or better, but the end-to-end speedups were `1.021x`, `1.084x`,
`0.849x`, and `1.078x` (mean `1.008x`). Thus the initializer is correctness-
preserving and sometimes faster, but not yet a robust learned speedup at high
distortion. Receipt:
`artifacts/learned_polynomial_initializer_stress_audit/learned_polynomial_initializer_stress_audit.json`.

## True Krylov preconditioner baseline

The GMRES interface now also accepts a truncated-Neumann operator as an actual
matrix-free preconditioner, rather than only using it as an initial guess. On
four heterogeneous `256x256` fields with amplitudes `0.25, 0.40, 0.55, 0.70`,
order-3 preconditioning reduced iterations from `15, 23, 35, 58` to
`4, 6, 9, 15`. Including all extra FFT applications, wall-time speedups were
`1.021x, 1.156x, 1.203x, 1.249x`; residuals stayed below `4.0e-9` and root
differences versus cold solves stayed below `5.0e-9`. This is the first
cost-inclusive evidence that a real preconditioner can outperform a warm start
at higher distortion. It is still a transparent analytic baseline, not a
learned/general preconditioner. Receipt:
`artifacts/neumann_preconditioner_audit/neumann_preconditioner_audit.json`.

### Realistic-resolution random-field stress test

The same cost-inclusive preconditioner was then tested on three independently
generated smooth heterogeneous coefficients at `512x512` (the largest
resolution used in this audit). Cold GMRES required `13, 15, 22` iterations.
Order-1 Neumann preconditioning reduced these to `7, 8, 12` and produced
wall-time speedups of `1.058x, 1.054x, 1.091x` (median `1.058x`). Orders 2 and
3 reduced the iteration counts further to `5, 5, 9` and `4, 4, 7`, but their
extra matrix-free applications made them slower overall: median speedups were
`0.955x` and `0.923x`, respectively. All nine solves converged, with residuals
below `5.0e-9` and root differences from the cold solves below `4.5e-9`.
Thus the iteration reduction survives realistic resolution, but a robust
cost-inclusive advantage is currently limited to a modest order-1 effect; the
result is not evidence for a learned/general preconditioner.
Receipt:
`artifacts/neumann_preconditioner_random_audit_512/neumann_preconditioner_random_audit.json`.

## Anderson/DEQ forward equilibrium

The periodic fixed-point equation now also has a short-memory Anderson solver,
so its forward path need not expose GMRES or store all iteration states. On a
`128x128` grid with amplitudes `0.25, 0.40, 0.55`, depth-6 Anderson converged
in `15, 23, 36` iterations with fixed-point residuals below `9e-10`; roots
matched GMRES to `2.6e-12`, `7.9e-11`, and `1.7e-9`. The same final state was
fed to the implicit transpose solve, with DEQ-style adjoint residuals below
`2.2e-9`. Anderson was slower than GMRES in this small test, so this is a
forward-equilibrium/differentiation-pattern control, not a speed claim or a
general nonlinear KKT layer. Receipt:
`artifacts/beurling_anderson_deq_audit/beurling_anderson_deq_audit.json`.

## Particle-mesh general-mesh control

To test whether “nonuniform mesh means no fast method” is too strong, a
cloud-in-cell particle-mesh operator was added: scattered samples are
deposited to a uniform grid, the FFT multiplier is applied, and the result is
bilinearly gathered. On `1,048,576` jittered points and an exact Fourier-mode
truth, grid sizes `256`, `512`, and `1024` took `0.741 s`, `0.797 s`, and
`0.962 s`; mean errors decreased from `1.32e-3` to `3.40e-4` to `9.51e-5`.
This is not a production singular-integral proof, but it is concrete evidence
that a general mesh can use a fast approximate operator through gridding. The
tradeoff is interpolation/deposition error and a separate treatment of
nonperiodic boundaries; NUFFT/FMM or a matched Galerkin operator is still
needed for a high-accuracy solver.
Receipt: `artifacts/beurling_particle_mesh_audit/beurling_particle_mesh_audit.json`.

## High-resolution scattered treecode versus GPU direct reference

To separate the “nonuniform mesh” question from the periodic FFT question, a
`16,384`-source / `16,384`-target scattered cloud was evaluated with no grid
wrapping. A GPU direct blocked sum on the AI RTX A6000 served as an independent
reference. For a separated source/target geometry, the order-12 Barnes--Hut
treecode (`theta=0.22`) took `3.78 s` and had relative L2 error
`2.88e-7` (maximum absolute error `1.00e-8`). For a near-field geometry with
the two clouds almost touching, it took `7.30 s` and had relative L2 error
`3.65e-7` (maximum absolute error `7.90e-8`). The direct GPU references took
`0.0418 s` and `0.0393 s`, respectively, and all outputs were finite.

This is stronger evidence that arbitrary scattered coordinates can use a fast,
accurate nonperiodic approximation without a uniform FFT grid. It is still a
Barnes--Hut control rather than a production FMM/NUFFT: the speed comparison
is against a GPU direct kernel, and singular self-interaction/PV quadrature
for a true same-mesh solve remains a separate gate. Receipts:
`artifacts/beurling_scattered_highres_16384_separated/beurling_scattered_highres_crosscheck.json`
and
`artifacts/beurling_scattered_highres_16384_near/beurling_scattered_highres_crosscheck.json`.

## CIC window deconvolution control

The particle-mesh baseline now includes an explicit inverse of the combined
`sinc^4` response of CIC scatter and gather, clipped near Nyquist to avoid
noise amplification. On the same `1,048,576` jittered points, the mean errors
at grid sizes `256`, `512`, and `1024` fell from
`1.318e-3, 3.398e-4, 9.511e-5` to
`1.838e-4, 4.575e-5, 1.354e-5`, respectively. This is a substantial
low-frequency accuracy improvement at essentially the same FFT complexity,
but it remains a periodic gridding approximation rather than an exact NUFFT or
singular Galerkin method. Receipt:
`artifacts/beurling_particle_mesh_deconvolution_audit/beurling_particle_mesh_audit.json`.

A four-point cubic B-spline scatter/gather variant was then added. On the same
`1,048,576` jittered points, cubic plus `sinc^8` deconvolution achieved mean
errors `2.43e-8`, `1.75e-9`, and `2.09e-11` at grid sizes `256`, `512`, and
`1024`, respectively; the 1024² run took `3.35 s` for the cubic pass. This is
strong evidence that a general point cloud can use a fast high-order gridding
operator for smooth/low-frequency data. It is still not a production singular
NUFFT/FMM certificate: kernel truncation, near-singular fields, and boundary
extension require separate audits. Receipt:
`artifacts/beurling_particle_mesh_cubic_audit/beurling_particle_mesh_audit.json`.

The complementary dispersion audit prevents overclaiming this result. On a
256² grid, cubic/deconvolved mean error was `5.33e-9` for mode `(3,-2)`,
`9.20e-4` for `(40,-35)`, `0.417` for `(90,-80)`, and `0.973` for
`(120,-115)`. Thus compact-kernel gridding is an excellent low-frequency
general-mesh approximation but deteriorates near Nyquist; it is not uniformly
accurate for singular or high-frequency fields. Receipt:
`artifacts/beurling_particle_mesh_dispersion_audit/beurling_particle_mesh_dispersion_audit.json`.

## Independent scattered-source cross-check

The particle-mesh result was also compared with an independent blocked
O(N²) quadrature using disjoint source and target point sets, avoiding the
same-point PV self-term ambiguity. At 16,384 source and 16,384 target points
on separated rectangles, the direct reference took `14.40 s`; cubic/sinc⁸
particle-mesh evaluations took `0.043`, `0.056`, and `0.113 s` on 128², 256²,
and 512² grids. Relative L2 discrepancy was approximately `0.139` at all three
grids. This is a roughly two-order-of-magnitude speed advantage, but the
non-vanishing discrepancy is important negative evidence: the direct reference
omits the PV self term and the particle-mesh operator is periodic, so this is
not a production NUFFT/FMM accuracy certificate. Receipt:
`artifacts/beurling_particle_mesh_direct_crosscheck_16384/beurling_particle_mesh_direct_crosscheck.json`.

## GPU throughput control

On the available `ai-codex-mihomo-codex` GPU, an NVIDIA RTX A6000, a legacy
PyTorch FFT control processed a `2048x2048` periodic grid (4,194,304 points)
in 20 repeated Beurling applications at `3.44 ms` per application. This is
only a throughput measurement; it does not replace the CPU correctness tests,
and the remote PyTorch installation uses the older `torch.rfft/irfft` API.
Receipt: `artifacts/ai_gpu_beurling_2048.json`.

## Discretization-matched finite-difference symbol

The central-difference rectangle operator now exposes its own periodic symbol
using `sin(2*pi*k/n)/h` rather than the continuous derivative `k`. This gives
a reproducible way to distinguish a continuous Beurling multiplier from the
actual FD/P1 stencil symbol before building a preconditioner. The unit test
applies the symbol formula directly to a Fourier mode; a full P1 two-face
block-symbol derivation remains open.

## P1 two-face block symbol

That derivation is now implemented for the regular periodic triangulation. For
each vertex Fourier mode, the two face orientations produce 2-by-1 symbols
`A(k)` and `B(k)` for `d_z` and `d_bar`; the weighted pseudoinverse
`P=(B*WB)^-1B*W` yields the rank-one block symbol `S=A@P`. On a `256x256`
mode grid there was exactly one derivative-null mode (the constant mode), the
projector idempotence error was `4.19e-16`, and the largest block spectral norm
was `1.0000000000000024`. The continuous scalar-ratio dispersion error was
non-negligible: low-frequency mean/max `0.0373/0.2929`, high-frequency
mean/max `0.4643/2.0`. Therefore substituting the continuous multiplier into a
P1 solver is not discretization-matched, even though the block pseudoinverse
has an exact realizable-face projector. Receipt:
`artifacts/p1_symbol_audit/p1_symbol_audit.json`.

## Batched P1 GPU symbol control

The same two-face block symbol was implemented as a batched torch kernel, with
the constant derivative-null mode masked explicitly. Local complex128 output
matched the NumPy reference on a 32² mode grid to `5.55e-16` for the symbol and
`4.48e-16` for the projector. On the AI server's NVIDIA RTX A6000
(`torch 1.7.1+cu110`), a 2048² batch (`4,194,304` modes) ran in `6.01 ms` per
complex64 batch over 20 repeats, or about `6.98e8` modes/s; the explicit
projector idempotence error was `2.99e-7` in complex64. This closes the P1 GPU
batch-throughput gate for the periodic regular triangulation, but it does not
solve physical boundaries, unstructured meshes, or the high-frequency
dispersion identified above.
Receipt: `artifacts/p1_symbol_gpu_audit_ai_2048/p1_symbol_gpu_audit.json`.

## Device-native scattered direct control

To separate the geometry limitation from the FFT limitation, a blocked Torch
implementation of the whole-plane kernel was added. It accepts arbitrary
scattered source and target points, performs no periodic wrapping, and keeps
the block matrix multiplication on the selected device; source values and
weights remain differentiable. A 192-point NumPy parity check gave maximum
complex error `4.73e-16`. On the AI RTX A6000, the nonuniform disjoint-source
control processed `16,384` points (`268,435,456` interactions) in `0.0789 s`
and `32,768` points (`1,073,741,824` interactions) in `0.2930 s`, with finite
outputs and `3.40e9`/`3.66e9` interactions per second. This demonstrates that
general scattered meshes do not force a CPU-only implementation or a hidden
periodicity assumption. It is explicitly an `O(N^2)` differentiable GPU
baseline, however, so it does not close the production NUFFT/FMM gate.
Receipt: `artifacts/beurling_torch_gpu_audit_ai/beurling_torch_gpu_audit.json`.

## Zero-mode affine-period adjoint

The periodic implicit layer now exposes the missing affine channel explicitly.
For periods `p=1+mu` and `q=i(1-mu)`, the real-inner-product VJP is
`g_mu=g_p+i*g_q`; a finite-difference test matches this formula. This does not
turn the zero-mean FFT inverse into a bounded-domain solver, but it prevents a
torus layer from silently discarding period/modulus gradients.

The affine channel was extended to a complete constant-`mu` map objective:
the VJP combines cotangents on sampled map vertices with the two quasi-periods.
At `512x512` (262,144 vertices), evaluation took `0.0061 s`; an independent
directional finite difference agreed to relative error about `1e-8` (absolute
directional difference `2.78e-5`). This closes the constant/affine zero-mode
coupling for map losses, but not the spatially varying nonlinear torus solve.
Receipt: `artifacts/torus_zero_mode_map_vjp_audit/torus_zero_mode_map_vjp_audit.json`.

## Spatially varying zero-mode-safe map lift

The periodic GMRES equation returns `h=f_bar`, whose mean is not generally
zero even when `mu` varies in space. The map lift now writes
`f(z)=z+mean(h)*conj(z)+C(h-mean(h))`, retaining the induced quasi-periods
`1+mean(h)` and `i(1-mean(h))`. On a `256x256` spatially varying coefficient,
the solve converged with residual `5.17e-10`; the recovered mean was
`0.2200000000+0.059999999997i`. A map-objective VJP was composed with the
implicit GMRES adjoint and checked against two fresh nonlinear solves along a
constant coefficient direction: directional error `1.06e-7`, adjoint residual
`4.60e-10`. This closes the previously untested spatially varying
zero-mode/map-objective channel, but it is not yet a nonlinear DEQ or arbitrary
triangulated-torus theorem. Receipt:
`artifacts/torus_spatial_zero_mode_map_audit/torus_spatial_zero_mode_map_audit.json`.

## Nonuniform whole-plane treecode control

The local environment has no `finufft`, `pynufft`, or `pyfmmlib` backend, so a
transparent Barnes--Hut Taylor treecode was added as an independent control.
It uses scattered source/target points directly, never wraps them periodically,
and evaluates far clusters by complex source moments while summing leaves
directly. On `4096` nonuniform sources and targets, the separated case reached
relative L2 error `4.75e-9` (order 12, opening angle `0.2`); a near-field case
with target offsets of roughly `0.02` reached `7.78e-13`. The near-field run
was slower than blocked direct quadrature (`10.68 s` versus `0.93 s`), while
the separated treecode was `2.60 s` versus `0.92 s`: this is an accuracy and
non-periodicity control, not yet a production FMM speed result. Receipt:
`artifacts/beurling_treecode_audit/beurling_treecode_audit.json`.

The treecode now also has an exact transpose of its fixed admissibility
decisions for source-value VJPs; it does not approximate the adjoint by merely
swapping source and target trees. On the same `4096`-point separated case,
forward relative error against direct quadrature was `1.51e-8`, and a
finite-directional VJP check had absolute error `3.89e-10`. Aggregating
accepted multipole adjoints before source backpropagation reduced the VJP time
to `3.68 s` (forward `2.06 s`). A 1024/2048/4096 scaling receipt gives
forward/VJP times `(0.607,0.761)`, `(1.239,1.728)`, and `(2.116,4.077)` s.
This is a faster differentiable general-mesh control, but not yet a production
FMM backward backend. Receipts:
`artifacts/beurling_treecode_vjp_audit/beurling_treecode_vjp_audit.json` and
`artifacts/beurling_treecode_scaling_audit/beurling_treecode_scaling_audit.json`.

The independent comparison was extended to `32,768` scattered sources and
targets. The order-12, opening-angle-`0.22` treecode took `8.39 s`, while an
AI RTX A6000 blocked direct sum evaluated `1,073,741,824` interactions in
`0.299 s`. Relative L2 error was `2.53e-7` and maximum absolute error was
`7.52e-9`; all outputs were finite. Receipt:
`artifacts/beurling_scattered_highres_32768_separated/beurling_scattered_highres_crosscheck.json`.
This is stronger realistic scattered evidence that a general-mesh operator
need not inherit FFT periodicity, but the treecode remains a Barnes--Hut
control rather than a production FMM and does not address singular
self-interaction on coincident target/source sets.

The same independent check was repeated for a deliberately near-field cloud
(targets displaced by small offsets rather than separated into distant
clusters). The order-12 treecode took `16.46 s`; the AI RTX A6000 direct sum
took `0.299 s` for the same `1,073,741,824` interactions. Relative L2 and
maximum absolute errors were `4.37e-7` and `6.05e-8`, respectively, with
finite outputs throughout. Receipt:
`artifacts/beurling_scattered_highres_32768_near/beurling_scattered_highres_crosscheck.json`.
Together with the separated case, this is a realistic near/far control rather
than a single favorable geometry. It still does not cover coincident
singular self-interaction or replace a production FMM/singular quadrature.

To make the near-field test quantitative, a paired cloud with prescribed
source/target offsets was also evaluated. At `16,384` points and offset
`1e-4`, the order-12 treecode took `69.23 s` and agreed with the independent
A6000 direct sum to relative L2/max errors `4.53e-4/5.84e1`; the direct sum
used `268,435,456` interactions in `0.0799 s`. At `32,768` points and offset
`1e-3`, treecode time was `170.33 s` and relative L2/max errors were
`1.72e-3/2.40e2`; the direct sum used `1,073,741,824` interactions in
`0.299 s`. Receipts:
`artifacts/beurling_scattered_highres_16384_paired_1e-4/beurling_scattered_highres_crosscheck.json`
and
`artifacts/beurling_scattered_highres_32768_paired_1e-3/beurling_scattered_highres_crosscheck.json`.
The deterioration as targets approach individual sources is diagnostic
evidence for the singular-quadrature gate, not a production PV certificate.

### Float64 CPU reference for the paired cloud

The `16,384`-point, `1e-4` paired cloud was then recomputed with an independent
blocked `complex128` CPU direct reference. The reference itself took `19.91 s`.
Against that reference, the baseline order-12 treecode at opening angle
`0.22` had relative L2 error `5.49e-13` and maximum absolute error
`6.92e-9`; tightening to `(theta, order)=(0.12,16)` and `(0.08,24)` reduced
the relative errors to `3.15e-15` and `3.18e-15`. Adding `near_radius=0.01`
to the baseline made no numerical difference (`5.49e-13`). Thus the earlier
`4.53e-4` discrepancy against the A6000 direct receipt is attributable to the
external float32/device reference in this extreme near-offset test, not to an
unavoidable treecode truncation floor. This does **not** close the singular
principal-value gate: the paired source and target sets are disjoint, whereas
the production solver must still define and discretize coincident-target PV
terms. Receipt:
`artifacts/beurling_treecode_cpu_reference_16384_paired_1e-4/beurling_treecode_cpu_reference_audit.json`.

## Reflection-compatible rectangle extension

The roadmap's reflection route was implemented as an explicit boundary-model
control. A coefficient on a half-open rectangle is copied to a doubled
periodic cell; one reflection conjugates `mu`, while two reflections restore
`mu`. The construction rejects incompatible complex boundary traces rather
than silently projecting them. On a `256x256` rectangle (a `512x512` doubled
cell), a smooth coefficient with real reflection-compatible traces converged
in 13 fixed-point iterations. The map-level Beltrami residual was
`1.65e-11`, the minimum base-rectangle Jacobian determinant was `0.72089`,
and the horizontal/vertical symmetry-axis violations were
`9.67e-9/2.15e-8`. Receipt:
`artifacts/beurling_reflection_rectangle_256/beurling_reflection_rectangle_audit.json`.

This is positive evidence that FFT periodicity can be combined with a
reflection boundary model without assuming that the original rectangle is a
torus. It is deliberately not universal: arbitrary rectangle boundary maps,
non-real-compatible traces, corner normalization, and all rectangle-to-
rectangle homeomorphisms are outside this symmetry class.

This directly answers the periodicity concern: zero extension plus FFT is a
particular approximation whose image interactions must be audited, but a
general-mesh whole-plane operator does not mathematically require a uniform
periodic grid. The unresolved engineering gate is a high-accuracy production
NUFFT/FMM or singular-quadrature implementation with a comparable cost audit.

## FINUFFT scattered prototype and resolution-matched control

The available FINUFFT 2.5.1 backend was exercised on the same nonuniform
source/target geometry.  On the original 16,384-point random point cloud, a
box-length/mode sweep remained far from the independent nonperiodic GPU direct
reference: the best relative L2 discrepancy was about `0.126` (box `L=4`,
`128^2` modes), while increasing modes at fixed box length did not monotonically
reduce the error.  A resolution-matched sweep (`L=2,4,8` with `512^2,1024^2,2048^2`
modes, constant physical cutoff) gave relative errors `0.368, 0.317, 0.307`.
Receipts:
`artifacts/beurling_finufft_scattered_16384/finufft_smooth_box_sweep.json` and
the resolution script in `tmp/remote_finufft_resolution_audit.py`.

To separate point-source truncation from the boundary model, a smooth uniform
quadrature cloud with `65,536` sources and `4,096` targets was also compared
against an independent AI-GPU direct sum.  The same resolution-matched FINUFFT
sequence produced relative errors `4.51e-2`, `1.50e-2`, and `1.41e-2` for
`(L,modes)=(2,512),(4,1024),(8,2048)`, with finite outputs and runtimes
`1.47 s`, `5.60 s`, and `22.27 s`.  Receipt:
`artifacts/beurling_finufft_continuum_65536/finufft_continuum_resolution.json`.

This is useful positive evidence that FINUFFT can accelerate a smooth
scattered-density whole-plane approximation, but it is not a production
certificate: the remaining error is a combined finite-box/image,
Fourier-truncation, and quadrature effect, and the random point-cloud result
shows that a singular/discrete density is substantially harder.  The result
therefore strengthens—not removes—the requirement for a boundary-aware
high-accuracy NUFFT/FMM or singular Galerkin backend before Route B can close.

### FINUFFT box-tail versus fixed physical cutoff

The smooth `65,536`-source cloud was rerun on the AI host with the physical
Fourier cutoff held fixed at `pi*modes/L = 804.25`, while the free-space box
was increased from `L=8` through `12` to `16` and the mode grids from
`2048^2` through `3072^2` to `4096^2`. Relative error against the same
independent direct reference changed only from `0.0141276` to `0.0140020` to
`0.0139502`; all outputs were finite. The receipt is
`artifacts/beurling_finufft_box_tail_65536/beurling_finufft_box_tail_audit.json`.

This controlled ablation rules out a simple finite-box tail as the dominant
error in this smooth case. It does not validate a rectangle boundary solver:
the remaining error can still contain Fourier truncation, point-cloud
quadrature, target-side interpolation, and the missing coincident-target
principal-value treatment. A singular-aware Galerkin/NUFFT/FMM backend remains
the Route B production gate.

## Fixed-point residual versus sampled-map residual floor

The roadmap calls out a subtle diagnostic: convergence of the iteration
`h <- mu*(1+B h)` is not the same statement as convergence of a sampled map
under a second, possibly inconsistent derivative discretization.  The audit
`artifacts/beurling_fixed_point_residual_floor_512/beurling_fixed_point_residual_floor_audit.json`
uses a smooth `512²` coefficient with amplitude `0.35`.  The fixed-point and
Fourier-consistent Beltrami residuals fall from `3.50e-1` at iteration 0 to
`2.41e-14` by iteration 16, while the ordinary second-order central-difference
residual levels off at `2.50e-5`.  Thus a tiny fixed-point residual does not
certify a tiny residual for a separately sampled finite-difference map.

The resolution sweep
`artifacts/beurling_fixed_point_residual_floor_sweep/resolution_sweep.json`
repeats the converged check at `256²/512²/1024²`.  The central-difference
residuals are `1.00e-4/2.50e-5/1.26e-5`, while the spectral residuals remain
near machine precision (`1.06e-14/2.41e-14/3.65e-14`).  This is direct evidence
for a separate discretization floor and reinforces the need to match the
operator, derivative, and boundary discretizations when using a Beurling layer
inside a differentiable model.  The sweep remains a periodic-torus audit, not
a rectangle boundary solver.
