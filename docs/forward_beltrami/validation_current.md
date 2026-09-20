# Current validation receipt

Command:

```powershell
$env:PYTHONPATH='D:\QC_optimization\src'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
C:\Users\xuzhehao\anaconda3\python.exe -m pytest -q
```

Historical result on 2026-09-19: **258 passed, 1 warning in 115.68 s**. The
intermediate result after the differentiable BHF additions was **263 passed,
1 warning in 131.88 s**. The latest result after the R2 boundary audit was
**264 passed, 1 warning in 123.74 s**; the intermediate fully real-valued BHF
pair-kernel result was **265 passed, 1 warning in 123.11 s**. The latest
checkpointed-kernel/VJP regression was **265 passed, 1 warning in 122.71 s**.
After adding the fixed-R2 projection control, the current full suite was
**266 passed, 1 warning in 138.80 s**. After adding the recompute-flow layer,
the current full suite was **267 passed, 1 warning in 141.58 s**. After adding
the determinant-root safe-step regression, the current full suite is **268
passed, 1 warning in 124.88 s**. After adding the fixed-active-root VJP
regression, the current full suite is **275 passed, 1 warning in 127.55 s**.
After adding the conservative smooth-safe BHF controller and its tests, the
current full suite is **278 passed, 1 warning in 123.82 s**.
The warning is the pre-existing
Paramiko/Cryptography Blowfish deprecation. Full stdout for the previous
checkpoint is preserved at
`D:\QC_optimization\tmp\validation\full_pytest_after_tutte_safe_trust_region_20260919.log`;
the earlier receipt remains at
`D:\QC_optimization\tmp\validation\full_pytest_after_variable_mmatrix.log`.
The latest full stdout is preserved at
`D:\QC_optimization\tmp\validation\full_pytest_after_smooth_safe_bhf_clean_20260919.log`.
After adding the paired-offset scattered Beurling cloud control and its unit
test, the current full suite is **279 passed, 1 warning in 144.13 s**. Full
stdout is preserved at
`D:\QC_optimization\tmp\validation\full_pytest_after_paired_beurling_20260919.log`.
After adding the reflection-compatible rectangle extension and its tests, the
current full suite is **282 passed, 1 warning in 127.30 s**. Full stdout is
preserved at
`D:\QC_optimization\tmp\validation\full_pytest_after_reflection_rectangle_20260919.log`.
After adding the symmetric-versus-directed Tutte stress and its test, the
current full suite is **283 passed, 1 warning in 130.62 s**. Full stdout is
preserved at
`D:\QC_optimization\tmp\validation\full_pytest_after_tutte_directed_20260919.log`.
After adding the rough-band Neumann preconditioner generator and test, the
current full suite is **284 passed, 1 warning in 125.38 s**. Full stdout is
preserved at
`D:\QC_optimization\tmp\validation\full_pytest_after_neumann_rough_20260919.log`.
After adding the boundary-free rectangle negative control and test, the
current full suite is **285 passed, 1 warning in 126.67 s**. Full stdout is
preserved at
`D:\QC_optimization\tmp\validation\full_pytest_after_rectangle_boundary_free_20260919.log`.
After adding the directed Tutte implicit layer and its VJP test, the current
full suite is **286 passed, 1 warning in 129.03 s**. Full stdout is preserved
at
`D:\QC_optimization\tmp\validation\full_pytest_after_directed_tutte_implicit_20260919.log`.
After adding the random-incompatible compatibility projection audit and test,
the current full suite is **287 passed, 1 warning in 141.18 s**. Full stdout
is preserved at
`D:\QC_optimization\tmp\validation\full_pytest_after_random_projection_20260919.log`.
After adding the nonuniform Delaunay directed Tutte audit and test, the current
full suite is **288 passed, 1 warning in 125.88 s**. Full stdout is preserved
at
`D:\QC_optimization\tmp\validation\full_pytest_after_directed_unstructured_20260919.log`.
After adding the near-aware Beurling traversal and its regression test, the
current full suite is **289 passed, 1 warning in 125.55 s**. Full stdout is
preserved at
`D:\QC_optimization\tmp\validation\full_pytest_after_beurling_near_correction_20260919.log`.
The latest environment-guarded verification rerun is also **289 passed, 1 warning in 141.23 s**;
full stdout is preserved at
`D:\QC_optimization\tmp\validation\full_pytest_current_20260919.log`.
The latest post-Route-D verification is again **289 passed, 1 warning in
131.49 s**; its full stdout is preserved at
`D:\QC_optimization\tmp\validation\full_pytest_current_20260919_routeD.log`.

The new forward-route tests include the zero-padded Beurling normalization
diagnostic, compatibility-kernel dimension audit, Tutte expressivity test,
GMRES solver and implicit Beurling VJP, sphere benchmark and atlas, sparse
implicit Tutte VJP, safe PL flow, direct nonuniform quadrature control, and
coarse-to-fine folding counterexample, plus the nonperiodic rectangle FD BVP
and monotone/spherical hard-layer checks, particle-mesh Beurling gridding,
positive nonuniform Tutte weights, finite-direction M-matrix fits, active-root
derivatives, and differentiable positive-increment latent decoding.
The latest additions also cover the published arbitrary-base BHF kernel,
identity-base consistency, fixed degree-five triangle quadrature, and explicit
normalization under source/evaluation collisions, as well as the weighted-Tutte
edge-weight VJP, conditioning-safe M-matrix dictionary selector, deconvolved
particle-mesh Beurling control, certified large-deformation continuation, the
nonlinear realizability projection prototype, the truncated-Neumann GMRES
initializer, and the matched central-difference symbol.
The latest additions also cover the P1 two-face block symbol/projector and the
positive-weight quasi-periodic torus Tutte embedding, cubic B-spline particle
mesh deconvolution/dispersion, local Duffy BHF pole quadrature, the torus
affine-period zero-mode VJP, and the learned two-feature polynomial GMRES
initializer with an end-to-end wall-time measurement. Route A now also has a
finite-difference check for the analytic sparse projection Jacobian. Route E
now includes the integer wide-stencil positive-weight regression. Route B now
tests independent source/target particle-mesh evaluation, and Route K tests
the affine map-plus-period zero-mode VJP at 512². Additional D-drive receipts
cover the 16,384-point direct cross-check, heterogeneous initializer stress,
coupled prolongation through 1024², and conditional flow-matching hard decode
at 512². Route C now includes vertex/interior Duffy near-field tests and the
adaptive two-step near/far flow control. The realistic 64² one-step BHF flow
audit is recorded at
`artifacts/bhf_near_far_flow_audit_64/bhf_near_far_flow_audit.json`.

The batched P1 symbol GPU control is recorded at
`artifacts/p1_symbol_gpu_audit_ai_2048/p1_symbol_gpu_audit.json`. On the AI
server's NVIDIA RTX A6000 it processed all 4,194,304 Fourier modes in
`6.009 ms` per batch (`6.98e8` modes/s, complex64), with finite outputs and
projector idempotence error `2.99e-7`. A local 32² NumPy parity check gave
symbol and projector errors below `6e-16`. This is a throughput/control result
for the periodic regular P1 symbol only; it does not remove the physical-boundary
or general-mesh accuracy limits of the FFT route.

The device-native arbitrary-scattered Beurling control is recorded at
`artifacts/beurling_torch_gpu_audit_ai/beurling_torch_gpu_audit.json`.
Its local 192-point NumPy parity error is `4.73e-16`; the AI RTX A6000
processed 16,384 and 32,768 nonuniform source/target points in `0.0789 s` and
`0.2930 s`, respectively, without periodic wrapping. This is an O(N²)
differentiable GPU baseline, not a production FMM/NUFFT replacement.

The 96² realistic determinant-safe BHF flow is recorded at
`artifacts/bhf_near_far_flow_audit_96/bhf_near_far_flow_audit.json`.
It took `216.16 s` for one near/far velocity assembly, accepted `dt=0.2`,
retained minimum determinant `0.63431`, and had zero flipped faces.
The new `128²` one-step Duffy near/far flow stress is recorded at
`artifacts/bhf_near_far_flow_128/bhf_near_far_flow_audit.json`.
It took `554.42 s`, retained minimum determinant `0.634327`, and had zero
flipped faces; this is a realistic no-folding control but not a scalable
production BHF integrator.
The conservative smooth-safe replay controller is recorded at
`artifacts/bhf_flow_smooth_safe_gpu_64_4step/bhf_flow_recompute_audit.json`,
`artifacts/bhf_flow_smooth_safe_gpu_128_2step/bhf_flow_recompute_audit.json`,
and `artifacts/bhf_flow_smooth_safe_gpu_256_2step/bhf_flow_recompute_audit.json`.
The `256²` two-step run took `114.18 s` forward and `518.56 s` backward,
retained minimum area ratio `0.182356`, had zero flips, and a finite gradient.
A requested-step-`10` `64²` stress clipped smoothly to `[4.07421,0.283442]`
with zero flips and minimum ratio `0.007757`. Directional VJP receipts are
at `artifacts/bhf_smooth_safe_vjp_32/bhf_smooth_safe_vjp_audit.json` and
`artifacts/bhf_smooth_safe_vjp_64/bhf_smooth_safe_vjp_audit.json`; the larger
finite-difference errors at epsilons below `3e-4` are float32 roundoff, while
the `32²/64²` checks at epsilons `1e-3` and `3e-4` remain finite and within
`3.56e-3` and `5.38e-3` relative error, respectively. This is a local smooth
active-clipping control, not yet an implicit BHF adjoint or global PV theorem.
The latest run adds the explicit north/south chart velocity chain-rule test,
the zero-mode-safe spatially varying torus map lift, and its map-objective
cotangent regression. The corresponding high-resolution receipts are
`artifacts/sphere_chart_velocity_gluing_audit/sphere_chart_velocity_gluing_audit.json`
and
`artifacts/torus_spatial_zero_mode_map_audit/torus_spatial_zero_mode_map_audit.json`.
The Route H larger common-mesh control is recorded separately at
`artifacts/barrier_highres_audit_96/barrier_highres_audit.json` (96x96 cells,
18,432 faces, 16 iterations).
The device-native matrix-free Route H/K control is recorded at
`artifacts/barrier_kkt_matrixfree_gpu_96/barrier_kkt_matrixfree_audit.json`.
On the AI server's RTX A6000 it solved the 96², 18,432-face nonlinear
barrier problem with forward stationarity `9.54e-12`, zero flipped faces, and
an implicit-vs-fresh-solve directional error of `3.47e-6` using `cg_rtol=1e-12`.
The GPU path keeps Hessian-vector products and CG vectors on device; it does
not copy the KKT solve back to SciPy/CPU.
The Route A medium-mesh projection probe is recorded at
`artifacts/compatibility_projection_128_probe/compatibility_projection_128_probe.json`
(128x128 cells, 32,768 faces, 5 evaluations).
The Route A realistic 256² projection probe is recorded at
`artifacts/compatibility_projection_256_probe/compatibility_projection_256_probe.json`
(131,072 faces, 5 evaluations).
The Route A realistic 512² projection stress audit is recorded at
`artifacts/compatibility_projection_highres_512/compatibility_projection_highres_audit.json`
(524,288 faces, 522,242 free coordinates, 6,789,146 Jacobian nonzeros,
`304.57 s`, residual `2.887e-6`, zero flips, minimum determinant
`0.9999996154`; the five-evaluation budget was exhausted without convergence).
The increased-budget realistic 512² continuation is recorded at
`artifacts/compatibility_projection_highres_512_nfev12/compatibility_projection_highres_audit.json`.
It converged in 6 evaluations (the solver terminated before the cap), with
residual `9.5897e-8`, zero flips, minimum determinant `0.9999999995`, and
wall time `384.24 s`. This improves the manufactured fixed-boundary evidence,
but does not establish arbitrary-μ projection or a global manifold chart.
The realistic random-incompatible projection audit is recorded at
`artifacts/compatibility_projection_random_256/compatibility_projection_random_highres_audit.json`.
At `256²`, eight evaluations left relative residual `0.7081` (absolute
residual `25.7786` versus target norm `36.4053`), while the returned map had
zero flipped faces, minimum determinant `0.63327`, and an independent passing
injectivity audit. This is explicit evidence that the layer projects arbitrary
facewise fields onto a realizable subset rather than solving them exactly.
The Route A 128² sparse projection VJP control is recorded at
`artifacts/compatibility_projection_vjp_audit_128_directional/compatibility_projection_vjp_audit.json`.
The tangent-direction fresh-projection error was `8.47e-7`; an arbitrary
off-manifold direction gave `5.56e-5`, so the latter is retained as negative
evidence against treating Gauss--Newton normal equations as an exact general
projection derivative.
The Route A 256² dual-holonomy reconstruction audit is recorded at
`artifacts/holonomy_reconstruction_audit_256/holonomy_reconstruction_audit.json`
(manufactured compatible field versus bounded random field).
The corresponding project-level derivation of the simply-connected P1 disk
holonomy sufficiency, one-complex-scale dimension, rectangle boundary contract,
and multiply-connected period warning is recorded at
`docs/forward_beltrami/compatibility_holonomy_theorem.md`.
The Route D positive-row theorem qualification is recorded at
`docs/forward_beltrami/tutte_positive_row_theorem.md`; it ties the directed
256² and 12,384-vertex unstructured audits to the planar 3-connected,
strictly-convex-boundary hypotheses and keeps arbitrary-`mu` expressivity
separate from topology.
The Route E cone phase diagram is recorded at
`artifacts/mmatrix_cone_phase_diagram_256/mmatrix_cone_phase_diagram_audit.json`.
At `rho=0.2/0.4`, radius-1 positive directions represented all sampled sites
to machine precision; `rho=0.6` required radius 2; `rho=0.8/0.9` remained
partially outside the radius-4 cone.
The non-periodic scattered Beurling treecode control is recorded at
`artifacts/beurling_treecode_audit/beurling_treecode_audit.json` (4096 source
and target points in separated and near-field cases).
The realistic scattered treecode cross-check is recorded at
`artifacts/beurling_scattered_highres_16384_separated/beurling_scattered_highres_crosscheck.json`
and
`artifacts/beurling_scattered_highres_16384_near/beurling_scattered_highres_crosscheck.json`.
At 16,384 source/target points, the order-12 treecode agrees with independent
AI-GPU direct references to relative L2 errors `2.88e-7` (separated) and
`3.65e-7` (near), with treecode times `3.78 s` and `7.30 s`; this remains a
Barnes--Hut control rather than a production FMM/NUFFT.
The new `32,768`-point independent cross-check is recorded at
`artifacts/beurling_scattered_highres_32768_separated/beurling_scattered_highres_crosscheck.json`.
It gives relative L2/max errors `2.53e-7/7.52e-9`; the treecode took `8.39 s`
and the AI-GPU direct reference took `0.299 s` for `1,073,741,824` interactions.
The matched near-field cross-check is recorded at
`artifacts/beurling_scattered_highres_32768_near/beurling_scattered_highres_crosscheck.json`.
It gives relative L2/max errors `4.37e-7/6.05e-8`; the treecode took `16.46 s`
and the same direct GPU reference took `0.299 s`. The separated/near pair is a
realistic far/near control, while coincident singular self-interaction and a
production FMM/singular-quadrature backend remain outside this receipt.
Paired-offset controls are recorded at
`artifacts/beurling_scattered_highres_16384_paired_1e-4/beurling_scattered_highres_crosscheck.json`
and
`artifacts/beurling_scattered_highres_32768_paired_1e-3/beurling_scattered_highres_crosscheck.json`.
Their relative L2 errors are `4.53e-4` at offset `1e-4` and `1.72e-3` at
offset `1e-3`, explicitly exposing near-source deterioration of the
Barnes--Hut control rather than hiding it inside a separated benchmark.
The near-aware traversal follow-up is recorded at
`artifacts/beurling_treecode_near_correction_16384_paired_1e-4/beurling_treecode_near_correction_audit.json`
and the `near_radius=0.05` repeat at
`artifacts/beurling_treecode_near_correction_16384_paired_1e-4_r005/beurling_treecode_near_correction_audit.json`.
At `theta=0.22`, order 12, the relative error stayed exactly
`4.52699785e-4` (baseline `60.53 s`; near radius `0.05`, `60.98 s`), showing
that ordinary opening-angle refinement already resolves the paired source
cluster and that a singular/PV treatment is still required.
The independent `complex128` CPU reference audit is recorded at
`artifacts/beurling_treecode_cpu_reference_16384_paired_1e-4/beurling_treecode_cpu_reference_audit.json`.
Against this reference, the same treecode has relative errors `5.49e-13`
(`theta=0.22`, order 12) and `3.15e-15` (`theta=0.12`, order 16), so the
earlier `4.53e-4` A6000 comparison is dominated by the external float32
reference. Coincident-target principal-value quadrature remains unresolved.
The reflection-compatible rectangle audit is recorded at
`artifacts/beurling_reflection_rectangle_256/beurling_reflection_rectangle_audit.json`.
On a `256²` base / `512²` doubled grid it converged in 13 iterations, with
map-level residual `1.65e-11`, minimum determinant `0.72089`, and symmetry-axis
violations below `2.2e-8`; its limitation is the restricted reflection
boundary class.
The FINUFFT 2.5.1 prototype is recorded at
`artifacts/beurling_finufft_scattered_16384/finufft_smooth_box_sweep.json` and
`artifacts/beurling_finufft_continuum_65536/finufft_continuum_resolution.json`.
The fixed-physical-cutoff box-tail follow-up is recorded at
`artifacts/beurling_finufft_box_tail_65536/beurling_finufft_box_tail_audit.json`;
its `L=8/12/16` errors are `0.0141276/0.0140020/0.0139502`.
On a smooth `65,536`-source uniform quadrature cloud and `4,096` targets, a
resolution-matched `(L,modes)=(2,512),(4,1024),(8,2048)` sweep gave relative
errors `4.51e-2`, `1.50e-2`, and `1.41e-2` against an independent GPU direct
sum; the original random point-cloud control remained around `0.31`. This is
fast scattered whole-plane evidence, not a production singular NUFFT/FMM or a
boundary-correct rectangle solver.
The new disjoint free-space `512²` direct-GPU/zero-padded-FFT audit is recorded
at `artifacts/beurling_free_space_fft_direct_512/beurling_free_space_fft_direct_audit.json`.
An independent AI RTX A6000 direct sum evaluated `1,073,741,824`
source-target interactions in `0.3021 s`; padding factors `2/4/8` gave
relative errors `7.0711e-2/4.4131e-3/2.7581e-4` on 4,096 targets. Because the
source and target blocks are disjoint, this isolates finite-box/periodic-image
error from singular self-quadrature; it remains a bounded-window quadrature
control rather than an exact rectangle boundary solver.
The fixed-point residual-floor audit is recorded at
`artifacts/beurling_fixed_point_residual_floor_512/beurling_fixed_point_residual_floor_audit.json`,
with its `256²/512²/1024²` resolution sweep at
`artifacts/beurling_fixed_point_residual_floor_sweep/resolution_sweep.json`.
After 16 iterations the Fourier-consistent residuals were
`1.06e-14/2.41e-14/3.65e-14`, while central-difference map residuals were
`1.00e-4/2.50e-5/1.26e-5`; this explicitly records a sampled-operator floor
separate from fixed-point convergence.
The rectangle ILU-GMRES audit is recorded at
`artifacts/rectangle_fd_iterative_audit_512/rectangle_fd_iterative_audit.json`.
At 256²/512², forward GMRES converged for both constant and variable μ, with
map differences from direct solves below `1e-11`; transpose relative residuals
were below `7e-11`. The 512² CPU configuration is not faster than sparse
direct, so this is implicit-differentiation evidence rather than a speed claim.
The matched rectangle FD-versus-LSQC audit is recorded at
`artifacts/rectangle_matched_solver_audit/rectangle_matched_solver_audit.json`
(64² and 96² cells; identical manufactured boundary data).
The realistic `256²` extension is recorded at
`artifacts/rectangle_matched_solver_audit_256/rectangle_matched_solver_audit.json`:
the FD residual was `2.86e-10` with map RMS error `0.35494`, while P1 LSQC
recovered the manufactured map at `9.10e-15`. This remains negative evidence
against the collocated FD inverse, not completion of Route B3.
The new boundary-adapted conductivity rectangle control is recorded at
`artifacts/rectangle_conductivity_audit_256/rectangle_conductivity_audit.json`.
It covers 65/129/257 nodes per axis (up to 131,072 triangles), gives RMS map
errors `3.65e-5`, `9.19e-6`, and `2.31e-6`, and independently exposes the
collocated central-difference matrix singularity as NaN output on the same
manufactured variable-coefficient problem.
Its source-value VJP audit is recorded at
`artifacts/beurling_treecode_vjp_audit/beurling_treecode_vjp_audit.json`.
The boundary-free identity-boundary negative control is recorded at
`artifacts/rectangle_boundary_free_control_129_257/rectangle_boundary_free_control.json`.
At `129²/257²`, the d-bar residuals were `0.27165/0.27356` (ratio `1.007`),
while the minimum triangle determinants were `0.56297/0.56270` and no faces
flipped. This isolates μ–boundary incompatibility from local orientation.
The optimized fixed-tree VJP scaling audit is recorded at
`artifacts/beurling_treecode_scaling_audit/beurling_treecode_scaling_audit.json`
(1024/2048/4096 scattered points).
The BHF interior-edge local Duffy control is recorded at
`artifacts/bhf_edge_target_audit/bhf_edge_target_audit.json` (64² and 128²
meshes).
The global incident-face BHF PV convergence audit is recorded at
`artifacts/bhf_global_pv_cancellation_audit/bhf_global_pv_cancellation_audit.json`
(64²/128²/256² vertex and interior-edge targets).
The half-million-face extension is recorded at
`artifacts/bhf_global_pv_cancellation_audit_512/bhf_global_pv_cancellation_audit.json`.
At 512², order-16 to order-24 changes were `3.20e-16` for the vertex target
and `7.09e-11` for the interior-edge target.
The unstructured M-matrix cone audit is recorded at
`artifacts/mmatrix_unstructured_stencil_audit/mmatrix_unstructured_stencil_audit.json`
(4,096 Delaunay vertices, one-ring versus two-ring directions).
The globally assembled unstructured positive-edge decoder audit is recorded at
`artifacts/mmatrix_global_unstructured_audit/mmatrix_global_unstructured_audit.json`

The wide-stencil manufactured-solution consistency audit is recorded at
`artifacts/mmatrix_consistency_order_audit_512/mmatrix_consistency_order_audit.json`.
At `128²/256²/512²`, machine-small tensor-fit residuals gave empirical order
approximately two, while positive but inexact cone fits plateaued with mesh
refinement. This separates approximation-order evidence from mere positivity.
The variable-tensor extension is recorded at
`artifacts/mmatrix_variable_consistency_audit_512/mmatrix_variable_consistency_audit.json`.
The cone-exact variable field converged at orders `1.9990/1.9997`, whereas a
rotating Beltrami tensor with fixed-direction fit residual RMS `0.34669`
saturated at operator error about `22.076` through 512².
(4,056 vertices, 7,854 triangles, convex-square boundary, zero flips).
The spatial triangular hard-decoder audit is recorded at
`artifacts/triangular_spatial_decoder_audit/triangular_spatial_decoder_audit.json`
(512² cells, 524,288 faces).
The alternating non-triangular triangular-layer audit is recorded at
`artifacts/alternating_triangular_decoder_audit_512/alternating_triangular_decoder_audit.json`
(four layers, 512² cells, 524,288 faces).
The true matrix-free GMRES preconditioner audit is recorded at
`artifacts/neumann_preconditioner_audit/neumann_preconditioner_audit.json`
(256², four heterogeneous amplitudes).
The random-field preconditioner stress audits are recorded at
`artifacts/neumann_preconditioner_random_audit/neumann_preconditioner_random_audit.json`
(six heterogeneous 256² smooth fields; costs included) and
`artifacts/neumann_preconditioner_random_audit_512/neumann_preconditioner_random_audit.json`
(three heterogeneous 512² smooth fields; costs included). In the latter,
order-1 has median speedup `1.058x`, while orders 2 and 3 have median
speedups `0.955x` and `0.923x`; all roots agree with cold solves to below
`4.5e-9`.
The higher-resolution two-case audit is recorded at
`artifacts/neumann_preconditioner_random_1024/neumann_preconditioner_random_audit.json`.
At `1024²`, cold iterations were `13/15`; order-1 reduced them to `7/8`,
order-2 to `5/5`, and order-3 to `4/4`. Cost-inclusive median speedups were
`1.052x/1.008x/0.907x`, with residuals below `1.7e-9` and root differences
below `1.4e-9`.
The finite rough-band control is recorded at
`artifacts/neumann_preconditioner_rough_512/neumann_preconditioner_rough_audit.json`.
At `512²`, cold iterations were `18/31`; orders 1/2/3 reduced them to
`9/16`, `6/11`, and `5/8`, with cost-inclusive median speedups
`1.187x/1.185x/1.128x`. Residuals stayed below `2.5e-9` and root differences
below `4.1e-9`.
The local orbifold cone-chart audit is recorded at
`artifacts/orbifold_cone_chart_audit/orbifold_cone_chart_audit.json`

The realistic global two-chart cone-atlas audit is recorded at
`artifacts/orbifold_global_cone_atlas_audit_512/orbifold_global_cone_atlas_audit.json`.
Across mixed north/south exponents `(0.65,1.4)` and `(1.4,0.65)`, the 512-point
equator seam displacement was `2.48e-16`, inverse round-trip error was below
`1.56e-14`, and all `261120` faces retained positive outward orientation.
This is an analytic atlas control, not an arbitrary-orbifold or BHF solve.
(256 radial rings, 512 angular samples).
The 64² augmented/hard-pin LSQC comparison is recorded at
`artifacts/fast_lsqc_benchmark_current/benchmark.json`.
The nonlinear barrier-KKT implicit-adjoint audits are recorded at
`artifacts/barrier_kkt_implicit_audit_32/barrier_kkt_implicit_audit.json` and
`artifacts/barrier_kkt_implicit_audit_48/barrier_kkt_implicit_audit.json`.
The 64² matrix-free Newton-CG barrier-KKT audit is recorded at
`artifacts/barrier_kkt_matrixfree_audit_64/barrier_kkt_matrixfree_audit.json`.

The 96² matrix-free Newton-CG barrier-KKT audit is recorded at
`artifacts/barrier_kkt_matrixfree_audit_96/barrier_kkt_matrixfree_audit.json`.
The 128² AI-GPU matrix-free Newton-CG stress audits are recorded at
`artifacts/barrier_kkt_matrixfree_gpu_128/barrier_kkt_matrixfree_audit.json`
and
`artifacts/barrier_kkt_matrixfree_gpu_128/barrier_kkt_matrixfree_beta3e3_audit.json`.
At 32,768 faces they retain zero flips and stationarity `7.64e-12`, but the
fresh-solve implicit directional errors are `5.53e-4` (`beta=1e-3`) and
`2.50e-4` (`beta=3e-3`), with 260 and 452 adjoint HVPs respectively.
The tighter-CG follow-up is recorded at
`artifacts/barrier_kkt_matrixfree_gpu_128_rtol1e-12/barrier_kkt_matrixfree_audit.json`.
At `beta=1e-3`, `cg_rtol=1e-12` used 310 adjoint HVPs and reduced the fresh-
solve directional error to `3.83e-6`; stationarity was `6.95e-12`, with zero
flips and minimum determinant `4.37e-5`.
The realistic `256²` AI-GPU baseline is recorded at
`artifacts/barrier_kkt_matrixfree_gpu_256/barrier_kkt_matrixfree_audit.json`.
It used `131,072` faces and `130,050` free coordinates, reached forward
stationarity `4.18e-12` with zero flips and minimum determinant `1.10e-5`, and
used 534 adjoint HVPs. The fresh-solve implicit directional error was `9.93e-4`
(`cg_rtol=1e-10`), so this confirms matrix-free resolution scaling but not
high-accuracy backward convergence at 256².
The tighter-CG 256² follow-up is recorded at
`artifacts/barrier_kkt_matrixfree_gpu_256_rtol1e-12/barrier_kkt_matrixfree_audit.json`.
With `cg_rtol=1e-12`, it used 637 adjoint HVPs, retained zero flips and minimum
determinant `1.10e-5`, and reduced the fresh-solve directional error to
`4.55e-6` with stationarity `2.65e-12`.
The higher latent-resolution conditional flow-matching audit is recorded at
`artifacts/flow_matching_hard_decoder_intervals256/flow_matching_hard_decoder_audit.json`.
It used 256 positive-increment intervals and a 512² decoder (524,288 faces):
held-out latent RMSE `0.09174`, zero flipped faces, and minimum determinant
`1.571e-6`.
It used `600` CG iterations as an explicit resolution-scaled cap, reached
stationarity `9.54e-12`, zero flips, and implicit directional error `4.68e-11`.
The 256² certified continuation counterexample is recorded at
`artifacts/safe_continuation_audit_256/safe_continuation_audit.json`.

The 256² automatic waypoint-planner control is recorded at
`artifacts/safe_rotation_waypoint_planner_audit_256/safe_rotation_waypoint_planner_audit.json`.
The `60°→120°→180°` path reached the endpoint with error `4.01e-15`, minimum
determinant above `0.999999999999928`, and zero flips.
The adaptive arbitrary-target waypoint control is recorded at
`artifacts/general_waypoint_planner_256/general_waypoint_planner_audit.json`.
At 256², direct one-segment continuation stalled with error `129.57`, while
automatic two-half-segment path subdivision reached error `6.47e-15`, zero
flips, and minimum determinant `0.9999999999999432`.
The 128² positive Tutte edge-weight expressivity audit is recorded at
`artifacts/tutte_weight_learning_audit_128/tutte_weight_learning_audit.json`.
The realistic 512² positive edge-weight expressivity stress is recorded at
`artifacts/tutte_weight_learning_audit_512/tutte_weight_learning_audit.json`
(524,288 faces, 787,456 edges, three updates, MSE ratio `0.0542`, minimum
determinant `2.704e-6`, zero flips).
The realistic 256² positive Tutte edge-weight expressivity audit is recorded at
`artifacts/tutte_weight_learning_audit_256/tutte_weight_learning_audit.json`
(197,120 edges, eight optimization steps, zero flips).
The determinant-aware positive-weight trust-region audit is recorded at
`artifacts/tutte_weight_learning_safe_256/tutte_weight_learning_safe_audit.json`.
The symmetric-versus-directed positive-row Tutte stress is recorded at
`artifacts/tutte_symmetric_nonsymmetric_256/tutte_symmetric_nonsymmetric_audit.json`.
On `256²`, symmetric, directed-logspread-1, and directed-logspread-3 solves
all retained zero flips and independent injectivity certificates; minimum
signed areas were `9.32e-3`, `1.97e-3`, and `4.94e-10`, respectively. The last
case is a numerical stress result, not a nonsymmetric Tutte theorem.
The differentiable directed-row layer audit is recorded at
`artifacts/tutte_directed_implicit_256/tutte_directed_implicit_audit.json`.
At `256²` (`65,025` interior rows), logit spreads 1 and 3 had finite output
and gradients, zero flips, and minimum signed areas `3.66e-3` and `1.07e-10`;
the convex-parallelogram boundary audit certified both maps. A small-mesh
finite-difference test independently checked the logit VJP.
The general-mesh directed implicit audit is recorded at
`artifacts/tutte_directed_unstructured_12000/tutte_directed_unstructured_audit.json`.
It used a nonuniform Delaunay mesh with `12,384` vertices, `24,382` faces,
and `12,000` interior rows. Logit spreads 1 and 3 had finite gradients, zero
flips, passing convex-boundary audits, and minimum signed areas
`3.97e-3`/`8.01e-12`.
It accepted all 12 updates, reduced MSE to `3.0077e-6` (ratio `0.00316`),
and retained minimum determinant `9.31e-6` with zero flips.
The Anderson/DEQ equilibrium audit is recorded at
`artifacts/beurling_anderson_deq_audit/beurling_anderson_deq_audit.json`
(128², three amplitudes).
The Route L spatial-`mu` decoder boundary audit is recorded at
`artifacts/torus_spatial_mu_decoder_audit_256/torus_spatial_mu_decoder_audit.json`
(256² separable recovery and off-diagonal shear control), with the realistic
512² confirmation at
`artifacts/torus_spatial_mu_decoder_audit_512/torus_spatial_mu_decoder_audit.json`.

The 512² cross-direction torus decoder audit is recorded at
`artifacts/torus_cross_direction_decoder_audit_512/torus_cross_direction_decoder_audit.json`.
The realistic 512x256 orbifold exponent sweep is recorded at
`artifacts/orbifold_cone_exponent_sweep_512/orbifold_cone_exponent_sweep_audit.json`.
Five north/south exponent pairs retained seam error `2.48e-16` and zero
flipped faces; the worst inverse round-trip error was `1.38e-11`.
The 65k-vertex unstructured M-matrix decoder stress is recorded at
`artifacts/mmatrix_global_unstructured_audit_65k/mmatrix_global_unstructured_audit.json`.
It used 131,022 Delaunay triangles and 65,000 interior unknowns, completed in
`117.71 s`, retained zero flips, and had tensor-fit mean/p95 residuals
`0.1246/0.7716`.
The new one-/two-ring realistic unstructured comparison is recorded at
`artifacts/mmatrix_global_unstructured_wide_12000/mmatrix_global_unstructured_wide_audit.json`.
At `12,384` vertices, two-ring directions reduced fit residuals to
`0.00291/7.44e-12` (mean/p95) but introduced `1,641` face flips and minimum
determinant `-4.93e-4`, while one-ring retained zero flips. This is explicit
negative evidence against equating local positive tensor fits with a global
hard-injective decoder on a nonplanar wide graph.
The planar barycentric enrichment controls are recorded at
`artifacts/mmatrix_barycentric_enrichment_8000/mmatrix_barycentric_enrichment_audit.json`
and
`artifacts/mmatrix_barycentric_enrichment_12000/mmatrix_barycentric_enrichment_audit.json`.
They retained zero refined-face flips at `49,146`/`73,146` refined faces, with
tensor-fit mean/p95 residuals `0.0192/0.0699` and `0.0187/0.0652`.
The anisotropic face-center ablation is recorded at
`artifacts/mmatrix_barycentric_enrichment_12000_anisotropic_centers/mmatrix_barycentric_enrichment_audit.json`.
Its center-cone residual was `0.608/2.297` (mean/p95), all-node residual
`0.414/1.918`, and minimum determinant `1.60e-14`, despite zero flips.
The GPU-native BHF assembly is recorded at
`artifacts/bhf_torch_gpu_128/bhf_torch_gpu_audit.json` and
`artifacts/bhf_torch_gpu_256/bhf_torch_gpu_audit.json`; one-step assembly took
`5.38 s`/`36.21 s` at 128²/256² and both maps had zero flips. The independent
32² float32 parity receipt is
`artifacts/bhf_torch_gpu_32/bhf_torch_parity.json` (relative error `1.66e-7`).
The realistic 512² GPU receipt is
`artifacts/bhf_torch_gpu_512_1step/bhf_torch_gpu_audit.json`; it used 524,288
faces, took `1044.19 s` for assembly and `1066.53 s` end-to-end, retained
minimum signed-area ratio `0.634365`, and had zero flipped faces. The copied
velocity field is `artifacts/bhf_torch_gpu_512_1step/velocity_0.npy`. This is
an acceleration/topology receipt, not a scalable BHF theorem.
The positive-stencil active-set canonicalization stress is recorded at
`artifacts/mmatrix_stencil_width_audit_256_smooth/mmatrix_stencil_width_audit.json`,
`artifacts/mmatrix_stencil_width_audit_512_smooth/mmatrix_stencil_width_audit.json`,
and the scan-order check at
`artifacts/mmatrix_smooth_scan_audit_256/mmatrix_smooth_scan_audit.json`.
At 256²/512², raster-neighbor continuation retained tensor-fit residuals below
`1.4e-15` while reducing scalar operator errors from `447.21/897.30` to
`6.71e-2/1.68e-2`; reverse-column scanning reproduced the 256² result.
The 256² safe-step active-set audit is recorded at
`artifacts/safe_step_active_set_audit_256/safe_step_active_set_audit.json`.
Across 47 random trials, 8 crossed an active-face boundary; fixed-active
derivatives stayed near finite differences, while switch cases showed errors
up to `2.92e-2`.
The R2 rectangle-side boundary/holonomy audit is recorded at
`artifacts/compatibility_boundary_theorem_audit_512/compatibility_boundary_theorem_audit.json`.
At 512² it retained winding one at four interior samples, zero flipped faces,
minimum signed area ratio `0.65459`, and aligned holonomy reconstruction error
`1.44e-15`.
The fixed-R2-boundary nonlinear projection receipt is
`artifacts/compatibility_rectangle_projection_audit_256/compatibility_rectangle_projection_audit.json`.
At 256² it reached relative face-`mu` error `3.01e-9`, boundary error `0`,
zero flipped faces, and minimum signed area ratio `0.65460`.
The tensor-native differentiable BHF controls are recorded at
`artifacts/bhf_torch_autograd_cpu_16/bhf_torch_autograd_gpu_audit.json`,
`artifacts/bhf_torch_autograd_cpu_32/bhf_torch_autograd_gpu_audit.json`, and
`artifacts/bhf_torch_autograd_cpu_64/bhf_torch_autograd_gpu_audit.json`.
Torch 2.5 CPU forward/backward assemblies were finite through 64², with
forward/backward times `0.343/0.440 s`, `2.137/3.138 s`, and
`28.604/17.703 s` at 16²/32²/64². The focused VJP test matched a finite
difference directional derivative. The AI RTX A6000 legacy Torch 1.7.1
backward limitation is explicitly recorded at
`artifacts/bhf_torch_autograd_gpu_64/bhf_torch_autograd_gpu_audit.json`; the
GPU forward remains covered by the preceding BHF GPU receipts. The fully
real-valued legacy-Torch GPU autograd controls are recorded at
`artifacts/bhf_torch_autograd_gpu_real_16/bhf_torch_autograd_gpu_audit.json`,
`artifacts/bhf_torch_autograd_gpu_real_32/bhf_torch_autograd_gpu_audit.json`,
and `artifacts/bhf_torch_autograd_gpu_real_64/bhf_torch_autograd_gpu_audit.json`.
They completed forward/backward at 16²/32²/64² with 0.250/0.689 s,
0.736/1.998 s, and 3.092/8.326 s respectively, all finite, by keeping the
entire operator in two real channels. The realistic 128² receipt is
`artifacts/bhf_torch_autograd_gpu_real_128/bhf_torch_autograd_gpu_audit.json`:
32,768 faces completed in 14.381 s forward and 40.869 s backward with finite
gradient. Its far-field block graphs are checkpointed and recomputed during
backward to avoid retaining the full dense interaction graph.
The memory-bounded multi-step BHF receipts are
`artifacts/bhf_flow_recompute_gpu_64/bhf_flow_recompute_audit.json` and
`artifacts/bhf_flow_recompute_gpu_128/bhf_flow_recompute_audit.json`.
They retain zero flipped faces at four steps/64² and two steps/128²; the
128² run used 32,768 faces and took `9.768 s` forward plus `44.701 s`
recompute-backward.
The new realistic `256²` two-step receipt is
`artifacts/bhf_flow_recompute_gpu_256_2step/bhf_flow_recompute_audit.json`.
It used `131,072` faces and completed in `114.08 s` forward plus `538.09 s`
recompute-backward, with finite gradient, zero flips, and minimum signed-area
ratio `0.86017`.
The realistic `256²` adaptive-safe receipts are
`artifacts/bhf_flow_recompute_safe_gpu_256_2step/bhf_flow_recompute_audit.json`
and
`artifacts/bhf_flow_recompute_safe_gpu_256_step10/bhf_flow_recompute_audit.json`.
The requested-step-`2.0` run accepted `[2.0,2.0]` with minimum ratio `0.18236`;
the requested-step-`10.0` run clipped to `[3.90091,0.267559]` with minimum
ratio `0.06892`, zero flips, and finite gradients (the latter had gradient L2
`2.23e6`).
The determinant-root safe-step stress receipts are
`artifacts/bhf_flow_recompute_safe_gpu_32_stress/bhf_flow_recompute_audit.json`
and `artifacts/bhf_flow_recompute_safe_gpu_64_stress/bhf_flow_recompute_audit.json`.
With requested step `2.0` and variation scale `0.2`, the third step was clipped
to `0.73573`/`0.72252` at 32²/64², while both runs retained zero flipped faces.
The fixed-active-root VJP GPU replay is recorded at
`artifacts/bhf_flow_recompute_safe_gpu_64_vjp/bhf_flow_recompute_audit.json`;
it retained the clipped step `0.72252`, zero flips, and finite gradient.
Adding positive `(1,1)` and `(1,-1)` graph edges reduced face-`mu` relative
error from `1.02535` to `0.36217`, while retaining zero flipped faces and a
positive minimum lifted determinant `2.77e-6`.

The direct induced-μ Tutte-layer audit is recorded at
`artifacts/tutte_mu_fit_128_lr002/tutte_mu_fit_audit.json` and
`artifacts/tutte_mu_fit_256_lr0005/tutte_mu_fit_audit.json`. The stable runs
reduced μ MSE from `0.0228557` to `0.00978713` at `128²` and from
`0.0228584` to `0.00871626` at `256²`, with finite gradients, zero flips,
and passing injectivity audits. The conditioning/near-degeneracy controls are
at `artifacts/tutte_mu_fit_256_lr002/tutte_mu_fit_audit.json` and
`artifacts/tutte_mu_fit_128/tutte_mu_fit_audit.json`; they are negative
optimizer evidence, not evidence that the positive-row topology theorem
fails.

The new unstructured BHF GPU controls are at
`artifacts/bhf_unstructured_gpu_128/bhf_unstructured_gpu_audit.json` and
`artifacts/bhf_unstructured_gpu_256/bhf_unstructured_gpu_audit.json`. They
cover `32,768` and `131,072` Delaunay faces, finite real-pair VJPs, Duffy-order
agreement below `1.92e-7`/`1.14e-7`, and zero flipped faces for candidate steps
through 1.0.

The Route A random-projection budget follow-up is at
`artifacts/compatibility_projection_random_highres_256_nfev20/compatibility_projection_random_highres_audit.json`.
It used 11 evaluations under a 20-evaluation budget and produced relative
residual `0.7081017153`, zero flips, and minimum signed-area ratio `0.63326`.

The independent `256²` direct-map barrier baseline is at
`artifacts/barrier_highres_audit_256/barrier_highres_audit.json`; the
unconstrained case had 508 flips, while LIM/SLIM/AMIPS-style cases had zero
flips with wall times `301.1/338.3/356.5 s`. Its target parameterization is
intentionally recorded as different from the matrix-free KKT control, so it is
not treated as a closed common-target comparison.

The strict common-target follow-up is at
`artifacts/barrier_common_target_256/barrier_common_target_audit.json`.
Using the matrix-free KKT target at `256²`, four-step direct-map runs produced
`2,754` flips without a barrier and zero flips for LIM/SLIM/AMIPS variants;
their wall times were `159.0/167.1/167.3 s`.

The extended Route L graph receipt is
`artifacts/torus_extended_graph_decoder_512/torus_extended_graph_decoder_audit.json`.
It uses `262,144` periodic vertices and `524,288` lifted faces, with map/face-μ
relative errors `0.00681/0.15281`, zero flips, and minimum lifted determinant
`2.93e-6`.

The differentiable BHF multigrid prototype is recorded at
`artifacts/bhf_multigrid_32_128/bhf_multigrid_audit.json`. A `32x32` coarse
stage followed by a `128x128` fine correction achieved relative map error
`1.80e-7` against four fine-grid steps, finite gradients, zero flips, and
forward/backward times `13.78/50.95 s` versus `26.92/93.46 s` for the
fine-only reference. This validates a local coarse-to-fine mechanism, not a
general solver guarantee.

The `512²` wide-cone experiment has no receipt because it was terminated after
approximately 63 minutes and 460 MB resident memory. It is tracked as a
resource-bounded engineering observation rather than an accuracy result.

The attempted `64x64 -> 256x256` BHF multigrid extension was stopped after
about 27 minutes with roughly 24 GB GPU memory in use and no receipt. The
smaller `32x32 -> 128x128` receipt remains the validated multigrid result.

After the summary-layer additions, the full repository regression suite was
rerun with the safe single-thread MKL/OpenMP environment: `289 passed, 1
warning` in `121.02 s`. The warning is the existing Paramiko/Blowfish
deprecation warning and is unrelated to the numerical routes.
