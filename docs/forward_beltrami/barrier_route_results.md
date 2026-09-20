# Route H: high-resolution barrier comparison

The direct-map barrier family was compared under one fixed `64x64` mesh
(`8,192` faces), identity initialization, the same smooth-twist target, Adam
learning rate, 24-iteration budget, rectangle boundary projection, and an
independent injectivity audit.

| method | time (s) | final data loss | min determinant | flips | certified |
|---|---:|---:|---:|---:|---|
| direct / none | 9.36 | 4.50e-4 | -2.20e-3 | 116 | no |
| LIM-style | 36.63 | 4.52e-3 | 4.41e-5 | 0 | yes |
| SLIM-style | 40.14 | 5.21e-3 | 4.25e-5 | 0 | yes |
| AMIPS-style | 39.03 | 2.97e-3 | 3.78e-6 | 0 | yes |

All safeguarded methods accepted every iteration, with at most three
backtracks. The unconstrained method achieved lower image loss but folded 116
faces, so that loss is not a valid registration result. Among the safeguarded
methods AMIPS obtained the lowest data loss in this particular setting but also
the smallest determinant margin. This is evidence for the safety/quality
tradeoff, not a universal optimizer ranking and not an implicit KKT backward.
Receipt: `artifacts/barrier_highres_audit/barrier_highres_audit.json`.

## Larger common-mesh control

The same four-way comparison was repeated at `96x96` cells (`18,432` faces)
for 16 iterations. The unconstrained method reached data loss `5.59e-4` but
folded `188` faces. LIM, SLIM, and AMIPS all remained independently certified
with zero flips; their final losses were `5.23e-3`, `5.83e-3`, and `4.67e-3`,
with minimum determinants `2.37e-5`, `9.55e-6`, and `3.19e-6`. Wall times were
`15.6 s`, `61.8 s`, `66.4 s`, and `61.2 s`, respectively. This strengthens the
resolution-dependent safety/quality tradeoff evidence, while still not being
an implicit KKT backward or a universal optimizer ranking. Receipt:
`artifacts/barrier_highres_audit_96/barrier_highres_audit.json`.

## Sparse augmented KKT versus hard-pin reference

The weighted LSQC implementation was also benchmarked at `64x64` cells
(`8,192` faces, `4,225` vertices) with one warmed repetition. The augmented
system had dimension `24,838` and `2,739,987` LU nonzeros; the hard-pin fast
system had dimension `8,446` and `595,104` LU nonzeros. Forward times were
`0.387 s` versus `0.0415 s` (`9.32x`), complete forward-plus-VJP steps were
`0.3755 s` versus `0.0482 s` (`7.79x`), and maximum map/gradient differences
were `2.25e-12`/`8.81e-11`. This quantifies sparse assembly/constraint
overhead while retaining an independent augmented reference. It is still a
linear LSQC comparison, not an implicit nonlinear barrier-KKT layer. Receipt:
`artifacts/fast_lsqc_benchmark_current/benchmark.json`.

An actual nonlinear barrier-KKT prototype is now available. It minimizes a
direct-map quadratic objective plus a `-log(det)` barrier with fixed rectangle
boundary, then differentiates the converged stationarity equation by solving
the dense Hessian adjoint rather than unrolling optimization. At 32² and 48²
cells, stationarity infinity norms were `6.1e-18` and `1.1e-18`, no faces
flipped, and the minimum determinants were `5.80e-4` and `2.62e-4`. The
implicit target-gradient directional errors against fresh re-solves were
`1.20e-7` and `7.92e-7`, respectively. This closes a concrete nonlinear KKT
backward control, while matrix-free 256²-scale KKT and nonsmooth active-set
handling remain open. Receipts:
`artifacts/barrier_kkt_implicit_audit_32/barrier_kkt_implicit_audit.json` and
`artifacts/barrier_kkt_implicit_audit_48/barrier_kkt_implicit_audit.json`.

The dense-Hessian limitation was then removed in a matrix-free prototype. At
`64x64` cells (`4,225` vertices, `8,192` faces, `7,938` free coordinates),
Newton-CG reached stationarity infinity norm `1.17e-11` in four outer steps;
the implicit adjoint CG converged in 101 Hessian-vector products, with
adjoint time `0.91 s`. The minimum face determinant was `1.74e-4` with zero
flips, and a fresh matrix-free re-solve gave an implicit directional error of
`1.34e-5`. This is now a realistic-resolution matrix-free KKT control, though
active-set nonsmoothness, global boundary guarantees, and larger-scale GPU
parallelism remain open. Receipt:
`artifacts/barrier_kkt_matrixfree_audit_64/barrier_kkt_matrixfree_audit.json`.

The same matrix-free layer was raised to `96x96` cells (`9,409` vertices,
`18,432` faces, `18,050` free coordinates) with a resolution-scaled CG cap of
`600`. Newton-CG converged in four outer steps with stationarity
`9.54e-12`; the adjoint converged in `155` HVPs and `2.44 s`. The minimum face
determinant was `7.76e-5` with zero flips. A fresh plus/minus re-solve gave an
implicit directional error of `4.68e-11`. The initial 120-iteration cap failed
with `info=120` at this size; making the cap explicit exposes the true
resolution-dependent Krylov cost instead of silently treating the failure as a
topology problem. Receipt:
`artifacts/barrier_kkt_matrixfree_audit_96/barrier_kkt_matrixfree_audit.json`.

## 128² AI-GPU conditioning stress

The device-native matrix-free layer was raised to `128x128` cells (`32,768`
faces and `32,258` free coordinates) on an RTX A6000. With barrier weight
`1e-3`, Newton-CG reached stationarity `7.64e-12` in four outer steps and the
adjoint CG required `260` HVPs. The minimum determinant was `4.37e-5` with zero
flips; a fresh plus/minus solve gave implicit directional error `5.53e-4`.
Increasing the adjoint cap from `200` to `500` changed `info=200` to
convergence but did not materially change the error, indicating that the
dominant issue is conditioning/nonlinear sensitivity rather than a simple
iteration cap. With a stronger barrier `3e-3`, the error improved to
`2.50e-4`, but the adjoint required `452` HVPs and the forward solve required
`608` HVPs. This is realistic GPU evidence that the matrix-free layer scales in
memory and remains fold-free, while backward accuracy degrades substantially
before a general production DEQ/KKT layer is obtained. Receipts:
`artifacts/barrier_kkt_matrixfree_gpu_128/barrier_kkt_matrixfree_audit.json` and
`artifacts/barrier_kkt_matrixfree_gpu_128/barrier_kkt_matrixfree_beta3e3_audit.json`.

### Tighter adjoint-CG tolerance

The same `128²`, `beta=1e-3` AI-GPU problem was rerun with
`cg_rtol=1e-12` and `cg_maxiter=900`. The forward stationarity remained
`6.95e-12`, zero faces flipped, and the minimum determinant was `4.37e-5`.
Adjoint HVPs increased from `260` to `310`, while the fresh-solve directional
error fell from `5.53e-4` to `3.83e-6`. Receipt:
`artifacts/barrier_kkt_matrixfree_gpu_128_rtol1e-12/barrier_kkt_matrixfree_audit.json`.
This isolates Krylov tolerance/conditioning as a major numerical error source,
but does not address active-set nonsmoothness or larger-resolution scaling.

## 256² AI-GPU conditioning stress

The same matrix-free Newton-CG layer was raised to `256x256` cells on the AI
GPU (`66,049` vertices, `131,072` faces, and `130,050` free coordinates), with
`beta=1e-3`, `cg_maxiter=600`, and `cg_rtol=1e-10`. The forward solve reached
stationarity infinity norm `4.18e-12` in four outer Newton steps; the implicit
adjoint converged (`info=0`) after `534` Hessian-vector products and took
`3.14 s`. The minimum face determinant was `1.10e-5` with zero flips. A fresh
plus/minus re-solve gave finite directional derivative `-0.248763` versus the
implicit value `-0.249756`, absolute error `9.93e-4`. This is a realistic
131,072-face matrix-free KKT baseline: memory and forward topology remain
tractable, but backward accuracy is again conditioning-limited at this
resolution. Receipt:
`artifacts/barrier_kkt_matrixfree_gpu_256/barrier_kkt_matrixfree_audit.json`.

The same `256²` run was repeated with `cg_rtol=1e-12` and `cg_maxiter=900`.
Forward stationarity improved to `2.65e-12`; the minimum determinant remained
`1.10e-5` with zero flips. The adjoint required `637` HVPs and converged with
`info=0`; the fresh-solve directional error dropped from `9.93e-4` to
`4.55e-6` (implicit `-0.249755720` versus finite `-0.249751170`). This
separates the previous 256² discrepancy into a dominant Krylov-tolerance
component, while the remaining error and HVP growth still leave conditioning,
active-set smoothness, and solver-comparison gates open. Receipt:
`artifacts/barrier_kkt_matrixfree_gpu_256_rtol1e-12/barrier_kkt_matrixfree_audit.json`.

## Independent 256² direct-map baseline

For an independent high-resolution compute baseline, the same `256x256`
structured mesh and smooth-twist target used by the direct-map comparison
were run for eight Adam steps from the identity with the same rectangle
projection and independent injectivity audit. The unregularized direct map
(`none`) ended with data loss `7.3116e-4` but `508` flipped faces and failed
the certificate. LIM-style, SLIM-style, and AMIPS-style safeguards all kept
zero flips and passed the certificate, with final data losses
`6.7177e-3`, `6.9916e-3`, and `6.8305e-3`; their wall times were
`301.1 s`, `338.3 s`, and `356.5 s`, respectively. Minimum determinants were
`4.21e-6`, `3.14e-7`, and `2.26e-6`. Receipt:
`artifacts/barrier_highres_audit_256/barrier_highres_audit.json`.

This is a realistic same-resolution direct-map safeguard baseline, not a
head-to-head same-objective comparison with the matrix-free KKT receipt above:
the KKT control uses a different zero-boundary target parameterization. The
remaining common-target comparison gate is therefore explicit rather than
silently claimed closed.

## Common-target direct baseline at 256²

The direct baselines were then rerun against the exact zero-boundary target
used by the matrix-free KKT experiment (`_target(vertices, amplitude=0.08)`),
with the same `256²` mesh, identity initialization, four-step budget, and
independent rectangle injectivity audit. The unconstrained case reached data
loss `3.7963e-4` but folded `2,754` faces. LIM, SLIM, and AMIPS retained zero
flips and certification, with losses `8.1137e-4`, `8.6251e-4`, and
`8.4326e-4`; wall times were `159.0 s`, `167.1 s`, and `167.3 s`, and minimum
determinants were `4.889e-6`, `5.528e-6`, and `4.234e-6`. Receipt:
`artifacts/barrier_common_target_256/barrier_common_target_audit.json`.

This closes the missing same-target forward/topology baseline at this finite
budget. It still does not compare final objective quality after equal
convergence, nor does it resolve active-set nonsmoothness in the implicit KKT
backward.
