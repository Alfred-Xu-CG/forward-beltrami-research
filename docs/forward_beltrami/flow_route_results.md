# Routes C/G: finite-element safe flow prototype

`safe_pl_flow` represents a continuous P1 velocity by vertex values and applies
explicit Euler residuals. Before each update it computes the exact per-face
determinant quadratic and chooses a step below the first area-margin crossing;
then it runs the independent global injectivity audit. This is the finite-
element safety part that a BHF implementation can reuse. It is deliberately
not called an exact BHF reconstruction: the BHF-specific operation that maps a
target Beltrami differential to a compatible velocity field is still a global
analytic/BVP problem.

On a `256x256` mesh with four repeated smooth velocity proposals and a
determinant margin of `0.2`, all four maps passed the global audit. The active
steps were

`0.996618, 0.009966, 0.00009966, 0.0000009966`,

and the minimum audited determinant stayed above `0.2000000159999864`. The run
took `33.8 s`, dominated by repeated boundary/topology checks. The rapidly
shrinking steps are important negative evidence: a naive explicit flow can be
hard-safe but computationally impractical near the fold barrier. It needs
continuation, a better velocity parameterization, or an implicit/barrier step;
safe-step alone does not solve the target-BC reconstruction problem.

## Explicit invertible layer family

For Route F, `affine_shear_matrix` provides a mesh-native hard layer based on
two triangular shears. Its matrix
`[[1,alpha],[beta,1+alpha*beta]]` has determinant exactly one for every finite
parameter pair, and the inverse and log absolute Jacobian are analytic. A
1,000-point randomized round trip returned below `1e-12` error. This family is
globally bijective and differentiable, but intentionally low-expressivity; it
should be composed with safe PL residual layers or richer monotone spline
shears rather than presented as a Beltrami solver.

`monotone_axis_map` / `separable_monotone_map` add a richer hard decoder using
positive interval increments and piecewise-linear inversion. It is a genuine
rectangle-to-rectangle homeomorphism (up to the half-open sampling convention)
and mapped `1,048,576` vertices in `0.041 s` in a `1024x1024` benchmark. Its
separable Jacobian cannot express general QC coupling, but it is a useful
coarse latent decoder or flow layer whose positivity guarantee is independent
of post-hoc fold repair.

## Differentiable latent training toy

`positive_increment_knots` decodes logits through softplus and normalized
cumulative sums. A 128-interval Adam experiment reduced knot MSE from
`2.16e-3` to `2.83e-10` in 600 steps; the minimum decoded increment remained
`4.46e-3`, gradients stayed finite, and maximum knot error was `1.02e-4`.
This is a separable toy rather than a coupled QC decoder, but it verifies the
important J/F contract: optimization can operate on unconstrained latent
logits while every forward decode remains strictly monotone. Receipt:
`artifacts/monotone_latent_training_audit/monotone_latent_training_audit.json`.

## Coupled monotone/shear composition

The hard decoder is now composed with the determinant-one affine shear, so the
forward map is coupled while its inverse remains analytic up to the monotone
piecewise-linear interpolation. On a `1024x1024` grid (`1,048,576` vertices),
the composed map and inverse took `0.096 s` and had maximum round-trip error
`5.55e-16`. This is a stronger hard-bijective building block than the
separable map, but the affine shear is still a restricted coupling and cannot
represent arbitrary spatially varying Beltrami fields. Receipt:
`artifacts/coupled_decoder_audit/coupled_decoder_audit.json`.

## Conditional flow-matching integration control

A conditional latent velocity network was trained on 64-interval positive-
increment/shear latents and integrated with 32 explicit Euler steps. Training
loss decreased from `1.062` to `0.214`; a held-out latent RMSE was `0.0527`.
At the endpoint, the hard positive-increment/shear decoder was evaluated on a
`512x512` grid (`524,288` faces): minimum determinant `1.941e-6`, zero flips,
and finite output. The latent integration took `0.0103 s`. Receipt:
`artifacts/flow_matching_hard_decoder_audit/flow_matching_hard_decoder_audit.json`.

This demonstrates actual flow-matching integration with topology inherited from
the decoder, but it is a toy conditional distribution with affine coupling; it
does not establish arbitrary-Beltrami expressivity or a full diffusion model.

## 128² BHF near/far flow stress

The Duffy-corrected near/far vertex assembly was run for one explicit flow step
on a `128x128` mesh (`32,768` faces). The map remained certified with zero
flipped faces and minimum determinant `0.6343274870`; the determinant-safe bound
was `13.8893` and the accepted step was `0.2`. The complete step took
`554.42 s` on the local CPU. This is a realistic-resolution no-folding control,
but its direct face-by-vertex quadrature cost is already too high for an
interactive or neural-network layer. A scalable global PV/FMM or GPU assembly
and a convergent adaptive integrator are still required. Receipt:
`artifacts/bhf_near_far_flow_128/bhf_near_far_flow_audit.json`.

## Spatially varying triangular coupling

To remove the affine-shear expressivity bottleneck while retaining a hard
Jacobian sign, a triangular layer was added: `x'=X(x)` and `y'=Y_x(y)`, where
each x-control column uses strictly positive y-increments. The inverse first
recovers `x` and then uses the same interpolated column to invert `y`; the
Jacobian is positive because `det Df=X'(x) * dY_x/dy`. On a `512x512` grid
(`263,169` vertices, `524,288` faces) with 96 spatial control columns and 128
y intervals, the inverse round-trip error was `1.13e-14`, the minimum face
determinant was `3.13e-8`, and the independent injectivity audit reported zero
flips and certified the map in `28.4 s`. This is a richer hard coupling than a
global affine shear, but it remains triangular and therefore does not prove
arbitrary Beltrami expressivity. Receipt:
`artifacts/triangular_spatial_decoder_audit/triangular_spatial_decoder_audit.json`.

To test non-triangular spatial coupling, four alternating positive triangular
layers were composed (x-conditioned y layer, then y-conditioned x layer, and
repeat). On a 512²-cell mesh (524,288 faces), the explicit inverse had maximum
round-trip error `1.89e-15`; the sampled P1 map had minimum face determinant
`9.61e-7`, zero flips, and passed the independent injectivity audit. This is a
meaningful expansion beyond a single triangular layer, but it still does not
prove arbitrary-QC expressivity or a mesh-independent sampled-P1 theorem.
Receipt:
`artifacts/alternating_triangular_decoder_audit_512/alternating_triangular_decoder_audit.json`.

## GPU-native BHF near/far assembly

The NumPy reference's `554.42 s` one-step cost at 128² was dominated by the
quadratic seven-point far quadrature and per-incident-face Duffy corrections.
`qcopt.forward.bhf_torch` now evaluates both pieces in blocked torch kernels,
including the local Duffy tensor product and scatter-add correction. On the AI
RTX A6000, one safe flow step took `5.38 s` for the assembly and `9.92 s`
end-to-end at 128² (`32,768` faces), giving minimum determinant `0.6343275`
and zero flips. At 256² (`131,072` faces), assembly took `36.21 s` and the
complete step `44.14 s`, with minimum determinant `0.6343523` and zero flips.
The accepted step remained `0.2` in both cases.

An independent 32² float32 GPU-versus-NumPy Duffy reference comparison gave
relative L2 error `1.66e-7` and maximum absolute error `4.97e-9`. Receipts:
`artifacts/bhf_torch_gpu_32/bhf_torch_parity.json`,
`artifacts/bhf_torch_gpu_128/bhf_torch_gpu_audit.json`, and
`artifacts/bhf_torch_gpu_256/bhf_torch_gpu_audit.json`.

This closes a substantial BHF scalability bottleneck for regular planar
meshes, but not the mathematical route: the implementation retains the same
PV quadrature and normalized-kernel assumptions as the NumPy control, and a
global convergence theorem, arbitrary-target atlas coupling, and prescribed
Beltrami reconstruction remain open.

## Tensor-native BHF differentiation

The same near/far/Duffy assembly is also exposed as
`bhf_near_far_apply_torch_differentiable`. It keeps image vertices, face
derivatives, and variations in the Torch graph (the implementation uses a
real-pair representation where older Torch complex indexing is unavailable).
On the local Torch 2.5 CPU control, the loss was the squared velocity norm and
the image VJP was finite at all tested resolutions:

| grid | faces | forward | backward | gradient L2 |
|---|---:|---:|---:|---:|
| 16² | 512 | 0.343 s | 0.440 s | 2.6479e-2 |
| 32² | 2,048 | 2.137 s | 3.138 s | 6.1163e-2 |
| 64² | 8,192 | 28.604 s | 17.703 s | 1.6665e-1 |

The independent `tests/test_forward_bhf_torch.py` VJP check agrees with a
finite-difference directional derivative. The original complex-valued path is
not backward-compatible with the AI host's legacy Torch `1.7.1+cu110` complex
autograd bridge, so a second implementation represents every complex value as
two real channels and returns an `(N,2)` real pair. That fully real path runs
forward and backward on the RTX A6000:

| grid | faces | forward | backward | gradient L2 |
|---|---:|---:|---:|---:|
| 16² | 512 | 0.250 s | 0.689 s | 4.7212e-2 |
| 32² | 2,048 | 0.736 s | 1.998 s | 1.1106e-1 |
| 64² | 8,192 | 3.092 s | 8.326 s | 3.0980e-1 |
| 128² | 32,768 | 14.381 s | 40.869 s | 1.0065 |

Receipts are in `artifacts/bhf_torch_autograd_gpu_real_16/`,
`artifacts/bhf_torch_autograd_gpu_real_32/`, and
`artifacts/bhf_torch_autograd_gpu_real_64/`, and
`artifacts/bhf_torch_autograd_gpu_real_128/`. The real-pair output matches the
complex differentiable path to machine precision on the local control, and a
real-pair finite-difference VJP test passes independently. The
128² run uses checkpointed far blocks: dense kernel graphs are recomputed in
backward instead of retained, reducing memory at the cost of the measured
40.869 s backward. This closes the single-assembly GPU autograd/runtime gate,
but not the larger
mathematical route: a nonlinear BHF flow still needs a stable multi-step time
integrator, atlas coupling, and an implicit adjoint through the full
trajectory.

## Multi-step memory-bounded BHF flow

`recompute_bhf_flow_real_pair` wraps several explicit BHF updates in a custom
autograd function. The forward stores only the initial image and static mesh
data; backward reconstructs the trajectory and differentiates the replay, so
it does not retain one dense near/far interaction graph per time step. On the
AI RTX A6000, four steps at `64x64` (8,192 faces) took `3.043 s` forward and
`12.806 s` backward, with zero flipped faces and minimum signed-area ratio
`0.85959`. A `128x128` two-step run (32,768 faces) took `9.768 s` forward and
`44.701 s` backward, again with zero flips and minimum ratio `0.86018`.
Receipts:
`artifacts/bhf_flow_recompute_gpu_64/bhf_flow_recompute_audit.json` and
`artifacts/bhf_flow_recompute_gpu_128/bhf_flow_recompute_audit.json`.

The recompute VJP matches a directly unrolled two-step VJP on the local
control. This is a memory-bounded engineering layer, not an implicit adjoint:
backward time grows with the number of replayed steps, and a global BHF
convergence theorem, adaptive safe integrator, and atlas coupling remain open.

## Determinant-root safe step

The recompute layer now has an optional determinant-root step controller. For
each P1 face it evaluates

\[
J_T(t)=c_T+b_Tt+a_Tt^2
\]

for the current map and BHF velocity, finds the first positive root of
`J_T(t)=margin`, and applies the minimum root times a safety factor. The root
selection is run without gradient; backward therefore uses a fixed-active-set
VJP and remains nonsmooth when the active face changes.

In a stronger AI-GPU stress with requested step `2.0`, variation scale `0.2`,
and three steps, the accepted histories were `[2.0, 2.0, 0.72252]` at 64²
(`8,192` faces) and `[2.0, 2.0, 0.73573]` at 32². Both runs had zero flipped
faces, finite gradients, and minimum signed-area ratios `0.01342` and
`0.01055`, respectively. Receipts:
`artifacts/bhf_flow_recompute_safe_gpu_64_stress/bhf_flow_recompute_audit.json`
and `artifacts/bhf_flow_recompute_safe_gpu_32_stress/bhf_flow_recompute_audit.json`.
The replay backward now differentiates the selected quadratic root with the
active face fixed; a local finite-difference VJP check passes, and the AI-GPU
64² receipt is repeated at
`artifacts/bhf_flow_recompute_safe_gpu_64_vjp/bhf_flow_recompute_audit.json`
with finite gradient and the same clipped step `0.72252`. This closes a
practical no-folding step-control gate for the planar P1 flow, but not a smooth
global flow theorem: active-set switches, multiple roots, and boundary/atlas
coupling still require separate treatment.

## Realistic 256² adaptive-safe multi-step stress

The determinant-root controller was exercised on the same `256x256` mesh with
`131,072` faces, two steps, requested step `2.0`, and variation scale `0.2`.
Both steps were accepted at `2.0`; the forward/recompute-backward times were
`114.22/518.56 s`, the gradient was finite (`L2=1576.51`), and the minimum
signed-area ratio was `0.18236` with zero flipped faces. Receipt:
`artifacts/bhf_flow_recompute_safe_gpu_256_2step/bhf_flow_recompute_audit.json`.

A stronger requested-step-`10.0` run activated the root limiter: accepted steps
were `[3.90091, 0.267559]`, with `0` flips and minimum signed-area ratio
`0.06892`. It took `114.35 s` forward and `538.05 s` backward; the gradient
remained finite but its L2 norm rose to `2.23e6`. Receipt:
`artifacts/bhf_flow_recompute_safe_gpu_256_step10/bhf_flow_recompute_audit.json`.

These are realistic no-folding controls for the adaptive mechanism. They also
show that differentiating a fixed active determinant root becomes extremely
stiff near the fold boundary; a smooth barrier or generalized active-set VJP
is still required for a numerically useful neural layer.

## Conservative smooth-safe step prototype

To remove the active-face switch, a differentiable controller was added for
the real-pair replay layer. On each face it uses the certified lower bound

`det(A+tB) >= c-|b|t-|a|t^2`,

solves the resulting quadratic inequality, and combines all face bounds with a
pairwise smooth lower envelope. Because the smooth envelope is never larger
than the exact minimum, this is conservative rather than a heuristic barrier.
The controller was tested on the AI RTX A6000 at realistic resolution:

| mesh / steps | forward / backward | accepted steps | min area ratio | flips |
|---|---:|---|---:|---:|
| 64² / 4, requested 2 | `2.92 / 16.15 s` | `2.0000, 2.0000, 0.72253, 0.02708` | `0.004562` | 0 |
| 128² / 2, requested 2 | `9.71 / 43.80 s` | `2.0000, 2.0000` | `0.182295` | 0 |
| 256² / 2, requested 2 | `114.18 / 518.56 s` | `2.0000, 2.0000` | `0.182356` | 0 |

A `64²`, requested-step-`10` stress activated the smooth controller with
steps `[4.07421, 0.283442]`, zero flips, and minimum ratio `0.007757`.
Receipts:
`artifacts/bhf_flow_smooth_safe_gpu_64_4step/bhf_flow_recompute_audit.json`,
`artifacts/bhf_flow_smooth_safe_gpu_128_2step/bhf_flow_recompute_audit.json`,
`artifacts/bhf_flow_smooth_safe_gpu_256_2step/bhf_flow_recompute_audit.json`,
and `artifacts/bhf_flow_smooth_safe_gpu_64_step10/bhf_flow_recompute_audit.json`.

The smooth-safe VJP was checked against fresh central differences while
clipping was active. At `32²`, relative errors were `2.65e-3` and `3.56e-3`
for epsilons `1e-3` and `3e-4`; at `64²` they were `2.84e-2` and `5.38e-3`
for the same two epsilons. Smaller epsilons became float32 roundoff-limited.
Receipts:
`artifacts/bhf_smooth_safe_vjp_32/bhf_smooth_safe_vjp_audit.json` and
`artifacts/bhf_smooth_safe_vjp_64/bhf_smooth_safe_vjp_audit.json`.

This materially improves the differentiability story at active-set switches,
while preserving a facewise determinant certificate. It is still a replay
adjoint rather than an implicit adjoint, and the conservative envelope can be
overly small on highly heterogeneous fields; global BHF PV convergence,
scalable far-field integration, and atlas coupling remain open.

## Realistic 256² multi-step recompute flow

The same real-pair recompute layer was run on the AI RTX A6000 at `256x256`
cells (`66,049` vertices and `131,072` faces) for two explicit BHF steps. The
forward pass took `114.08 s`; replay-based backward took `538.09 s`, with a
finite gradient (`L2=409.80`). The accepted steps were `[0.02, 0.02]`, the
independent injectivity audit found zero flipped faces, and the minimum signed
area ratio was `0.86017`. The remote CUDA process peaked at approximately
`17.5 GiB` allocation. Receipt:
`artifacts/bhf_flow_recompute_gpu_256_2step/bhf_flow_recompute_audit.json`.

This is the first realistic multi-step BHF differentiability control beyond
`128²`. It demonstrates that recomputation remains feasible without retaining
one dense interaction graph per step, but its `538 s` backward time is not yet
compatible with an interactive neural layer; implicit adjoints, better far
field acceleration, and larger-step adaptive integration remain open.
