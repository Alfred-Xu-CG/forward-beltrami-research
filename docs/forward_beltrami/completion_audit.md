# Forward Beltrami route completion audit

This is an evidence audit against the roadmap's Definition of Done. `partial`
means that a route has a reproducible implementation and realistic-resolution
evidence, but at least one mathematical, topology, differentiation, or
scalability gate is still open.

| route | evidence now present | unresolved gate |
|---|---|---|
| A realizable BC | shared-edge compatibility matrix and nullity study through 256²; 256² dual-holonomy/primal-edge reconstruction with machine-precision manufactured recovery and large random-field failure; a project-level simply-connected P1 disk holonomy sufficiency/dimension derivation; fixed-boundary nonlinear projection with analytic sparse Jacobian through 256x256/131072 faces; 512² sparse stress assembly with 6.79M Jacobian nonzeros; increased-budget 512² compatible projection converges in 6 evaluations with residual `9.59e-8`, zero flips, minimum determinant `0.9999999995`, and `384.24 s` wall time; new 256² random facewise μ projection remains at relative residual `0.7081` after 8 evaluations with zero flips; 128² sparse projection normal-equation VJP with tangent/off-manifold directional checks; 512² R2 rectangle-side manufactured reconstruction with winding one, zero flips, and aligned error `1.44e-15`; fixed-R2 projection through 256² with relative `mu` error `3.01e-9` and zero flips | publication-grade holonomy theorem/independence conditions, uniqueness/manifold chart, exact second-order off-manifold VJP, arbitrary tangential-boundary optimization, arbitrary incompatible-target projection, hard global injectivity theorem, and differentiable projection manifold chart |
| B Beurling/FFT | periodic multiplier, zero-padding boundary audit through 1024², zero-mode counterexample, GMRES/VJP, FD symbol, 256² P1 block symbol, 1024² CIC/cubic particle mesh plus sinc⁴/sinc⁸ deconvolution, 256² frequency-dispersion audit, 512² rectangle FD, ILU-GMRES and transpose control through 512², matched 64²/96² rectangle FD-versus-LSQC control exposing a centered-FD ambiguity, 65/129/257-node boundary-adapted conductivity FEM control with second-order map convergence and explicit collocated singularity, a 129²/257² identity-boundary boundary-free negative control with residual `0.27165→0.27356` and zero flips, 16,384-point disjoint-source direct cross-check, 4096-point nonperiodic treecode forward/optimized exact-VJP control with scaling receipt, 16,384-point separated/near treecode versus AI-GPU direct references at relative errors `2.88e-7/3.65e-7`, 32,768-point separated/near direct-GPU cross-checks with relative errors `2.53e-7/4.37e-7`, paired-offset near-source controls exposing errors `4.53e-4`/`1.72e-3`, and a near-aware traversal follow-up that leaves the `16,384`-point offset-`1e-4` error unchanged, batched 2048² AI-GPU P1 symbol throughput/parity audit, device-native arbitrary-scattered direct GPU control through 32,768 points with `4.73e-16` small-case parity, FINUFFT 2.5.1 resolution-matched smooth-cloud control at 65,536 sources (`4.51e-2/1.50e-2/1.41e-2`), and a disjoint `512²` free-space direct-GPU cross-check (`1,073,741,824` interactions in `0.3021 s`) where zero-padded FFT errors fall from `7.07e-2` to `2.76e-4` as padding grows 2→8 | production NUFFT/FMM or equivalent high-accuracy scattered operator, stronger rectangle preconditioner, and a boundary-free arbitrary-`mu` rectangle decoder |
| C sphere BHF | Möbius truth velocity, atlas checks, arbitrary-base BHF kernel, 128² triangle quadrature, local Duffy pole quadrature on 8192 faces, vertex-target near/far assembly through 128², interior-target and interior-edge Duffy subdivision through 128², adaptive two-step determinant-safe flow through 32², realistic 64²/96² one-step determinant-safe BHF flow, 64² one-step orientation audit, 261120-face sphere audit, 512x256 chart-coordinate/velocity transition check, 64/128/256² plus 512² global incident-face Duffy convergence control, GPU-native regular-planar near/far/Duffy assembly through 256² (`36.21 s` assembly) and 512² (`1044.19 s` assembly, zero flips), tensor-native CPU VJP through 64² with finite-difference agreement, fully real-pair legacy-Torch GPU autograd through 128² with checkpointed far blocks, memory-bounded recompute flow through 128² two-step/64² four-step controls, determinant-root active-step clipping/fixed-active VJP through 64² with zero flips, and a conservative smooth-safe controller through realistic 256² with active-clipping directional checks | full adaptive global PV theorem, scalable nonlinear BHF time integrator beyond the regular planar control, multi-step implicit adjoint beyond replay, quantitative smooth-controller efficiency/generalization, active-set switch theorem for the exact controller, nonlinear-operator atlas coupling, BHF GPU/sphere scaling |
| D Tutte/Floater | uniform and positive weighted decoder through 512², sparse implicit boundary VJP, 256² edge-weight VJP, 64²/128² positive-weight expressivity optimization with zero flips, realistic 256²/197,120-edge expressivity optimization with zero flips, new 512²/787,456-edge three-update stress with MSE ratio `0.0542` and zero flips, expressivity counterexample, a 256² symmetric-versus-directed positive-row stress with zero flips and directed minimum determinant `4.94e-10`, a 256² directed positive-row sparse implicit layer with finite boundary/logit VJPs and zero flips, a 12,384-vertex nonuniform Delaunay directed implicit audit with finite VJPs and zero flips, and a literature-backed positive-row theorem qualification under planar 3-connected/convex-boundary hypotheses | richer composed decoder, graph-hypothesis verification on every target mesh, arbitrary-μ fitting, conditioning control, and stable high-resolution learned-weight training |
| E M-matrix | SPD tensor derivation, fixed-stencil failure, spectral decomposition, 256² finite-direction audit, conditioning-safe selector, 256² integer wide-stencil residual/positivity audit, 128²/256²/512² constant-tensor manufactured-solution order audit, 128²/256²/512² variable-tensor cone-exact versus rotating-Beltrami consistency audit, 4096-vertex nonuniform Delaunay one-/two-ring cone audit, a 4,056-vertex globally assembled positive-edge harmonic decoder with zero flips, 256²/512² stencil-width stress showing active-set noise despite exact local fits, and a raster-neighbor canonicalizer reducing operator errors to `6.71e-2`/`1.68e-2` | variable tensors outside the positive cone, scan-order-independent canonical/regularized conductance selection, boundary closure beyond the square control, and a general QC decoder theorem |
| F mesh flow | affine shear, positive monotone decoder, coupled monotone/shear composition at 1024², 512² spatial triangular positive-increment decoder, four-layer alternating triangular composition at 512², inverse checks, and coupled positive-latent prolongation | arbitrary-QC expressivity and a mesh-independent sampled-P1 theorem |
| G certified residual | exact determinant-root step, active-face derivative, 256² flow audit, 64²/256² large-deformation direct-vs-staged counterexamples, 256² automatic `60°→120°→180°` waypoint control, and 128²/256² adaptive arbitrary-target rotation-plus-shear path control with direct-path stall and zero-flip staged recovery | arbitrary homotopy construction, active-set nonsmooth handling and a solver-level continuation theorem |
| H optimization/barrier | existing LIM/SLIM/AMIPS/LSQC evidence plus common 64² and 96²/18,432-face comparisons with independent fold audits, a 64² augmented-KKT versus hard-pin timing/gradient comparison, dense nonlinear log-det barrier-KKT implicit-adjoint controls through 48², matrix-free Newton-CG KKT adjoints through 64² and 96² CPU with fresh-solve checks, a device-native 96² AI-GPU HVP/CG control with zero flips and directional error `3.47e-6`, a 128² AI-GPU stress with baseline errors `5.53e-4`/`2.50e-4`, a tighter-CG 128² control reducing the `beta=1e-3` error to `3.83e-6`, a 256² AI-GPU baseline with 131,072 faces, zero flips, stationarity `4.18e-12`, and fresh-solve directional error `9.93e-4`, and a tighter 256² `cg_rtol=1e-12` control reducing that error to `4.55e-6` | nonsmooth active-set handling, larger/common solver comparison, GPU conditioning/accuracy beyond this control, and a target-solver comparison at larger resolution |
| I coarse-to-fine | folding counterexample, positive latent prolongation, 512² and 1024² smooth-map audits, coupled injective prolongation with zero flips, 512² spatial triangular coupling, and 512² alternating triangular composition | mesh-independent prolongation theorem and arbitrary-QC latent expressivity |
| J diffusion/decoder | differentiable positive-increment latent training, 1024² coupled decoder, 512² spatial triangular decoder, 512² alternating triangular composition, coupled coarse-to-fine hard decode, conditional flow-matching integration with zero flips, and a new 256-interval/512² flow-matching stress with zero flips and minimum determinant `1.57e-6` | arbitrary-QC expressivity, full diffusion integration, and coupled high-resolution training comparison |
| K DEQ/implicit | sparse Tutte adjoint, Beurling and rectangle transpose VJPs, GMRES tolerance/initializer, explicit affine-period zero-mode VJP, 512² constant-affine map-plus-period VJP, 256² spatially varying zero-mode/map-objective composed VJP, 128² Anderson equilibrium/implicit-adjoint control, 32²/48² dense nonlinear barrier-KKT controls, 64²/96² matrix-free Newton-CG KKT control, a 96² device-native AI-GPU HVP/CG control with fresh-solve directional error `3.47e-6`, a 128² tighter-CG GPU control with `3.83e-6` error, a 256² AI-GPU baseline with 131,072 faces and `9.93e-4` error, and a tighter 256² control with `4.55e-6` error | general nonlinear DEQ/root-KKT coupling, nonsmooth active-set handling, and GPU accuracy/conditioning beyond this control |
| L torus | affine and spatially varying-`h` period lift, period-to-modulus inverse, 512² positive-weight quasi-periodic Tutte embedding, 512² spatial-`mu` separable/shear decoder boundary audit, and 512² positive graph decoder with diagonal cross-direction edges | arbitrary triangulated-torus theorem, exact variable-coefficient consistency, and a general hard decoder |
| M orbifold sphere | closed two-cap 256×128 assembly, seam/orientation certificates, realistic 256-ring×512-sample local cone charts, and realistic 512×256 global two-chart cone atlas with mixed exponents, inverse error `<1.56e-14`, and zero flips | arbitrary cone-angle transition law, orbifold normalization/global chart theorem, and BHF comparison |
| N learned initializer | scalar learned initializer, order-2 Neumann warm start, a two-feature polynomial initializer at 256², a 12-example heterogeneous stress audit (mean 1.008x, one 0.849x slowdown, roots preserved), a cost-inclusive true order-3 matrix-free Neumann preconditioner audit at 256², a six-case 256² random-smooth-field stress test showing only 1.023x median order-1 speedup and order-2/3 slowdowns, a realistic three-case 512² stress test reducing cold iterations 13/15/22 to 7/8/12 with order-1 median speedup 1.058x but order-2/3 median speedups 0.955x/0.923x, a two-case 1024² stress test reducing cold iterations 13/15 to 7/8 with cost-inclusive median speedups 1.052x/1.008x/0.907x for orders 1/2/3, and a two-case 512² rough-band stress reducing cold iterations 18/31 to 9/16 with median speedups 1.187x/1.185x/1.128x | robust learned/general speedup beyond Neumann structure and construction-cost generalization |

The active goal is not complete while any unresolved gate above remains. All
receipts referenced here are under `D:\QC_optimization\artifacts` and all
validation logs are under `D:\QC_optimization\tmp`.

Route B numerical clarification: an independent blocked `complex128` CPU direct
reference on the `16,384`-point offset-`1e-4` cloud gives treecode relative
errors `5.49e-13` at `(theta,order)=(0.22,12)` and `3.15e-15` at `(0.12,16)`.
The earlier `4.53e-4` AI-GPU comparison is therefore a float32/device-reference
effect for this disjoint near-source control, not a treecode truncation floor;
coincident principal-value quadrature is still not closed.

Route B FINUFFT box-tail clarification: the `65,536`-source smooth-cloud
fixed-cutoff sweep `(L,modes)=(8,2048),(12,3072),(16,4096)` produced relative
errors `0.0141276/0.0140020/0.0139502`. Increasing the free-space box alone
does not remove the discrepancy, so the periodic/finite-box concern is not
the sole bottleneck; singular quadrature and a boundary-aware operator remain
unresolved.

Route D theorem clarification: the positive-row Tutte/Floater embedding result
is conditional on a planar 3-connected graph, strictly convex boundary, and
strictly positive row-stochastic neighbor weights; it does not require
`w_ij=w_ji`. The project note
`docs/forward_beltrami/tutte_positive_row_theorem.md` records the theorem
scope, its failure modes, and the realistic structured/unstructured receipts.

Route C addendum: the realistic `128²` one-step Duffy near/far audit took
`554.42 s` while retaining positive determinants and zero flips; direct
face-by-vertex quadrature is therefore still a scalability bottleneck.

Route C unstructured-mesh clarification: jittered rectangular Delaunay meshes
were tested at `128²` and `256²` with the differentiable real-pair near/far/Duffy
operator. Both quadrature orders 8 and 16 had finite gradients, order-to-order
velocity differences below `1.2e-7`, and candidate steps through 1.0 retained
zero flipped faces with positive independently checked boundary orientation.
This shows the implementation is not dependent on a uniform grid, but it is
still a one-step control; global PV cancellation, multi-step adjoints, and
production scaling remain unresolved.

Route G addendum: the `256²` active-set audit observed 8 switches in 47
trials, with fixed-active derivative errors near `1e-6` and switch errors up
to `2.92e-2`; generalized nonsmooth differentiation remains open.

Route M addendum: the realistic `512x256` five-pair cone exponent sweep kept
seam error at `2.48e-16` and zero flips, but remains an analytic atlas control,
not a proof for arbitrary orbifold transitions or BHF coupling.

Route B addendum: FINUFFT 2.5.1 was exercised on the realistic scattered
geometry and on a `65,536`-source smooth quadrature cloud.  The latter's
resolution-matched relative errors were `4.51e-2`, `1.50e-2`, and `1.41e-2`
for boxes `L=2,4,8`; the random point-cloud control stayed around `0.31`.
This is a useful fast scattered prototype, but finite-box/truncation and
singular-density accuracy remain open, so the production NUFFT/FMM gate stays
unresolved.

Route B addendum: the disjoint `512²` free-space audit compares an independent
AI RTX A6000 direct sum against zero-padded FFT values on 4,096 targets. With
`1,073,741,824` direct interactions in `0.3021 s`, relative errors were
`7.0711e-2`, `4.4131e-3`, and `2.7581e-4` for padding factors `2`, `4`, and
`8`. This confirms that periodic-image/finite-box error is controllable by
padding in this separated case, while leaving singular quadrature and physical
rectangle boundary enforcement as separate unresolved gates.

Route E addendum: the `66,024`-vertex unstructured Delaunay stress retained
zero flips but showed tensor-fit mean/p95 residuals `0.1246/0.7716`; this is
not an arbitrary-μ consistency result.

Route B addendum: the `256²/512²/1024²` fixed-point residual-floor sweep
reached Fourier-consistent residuals `1.06e-14/2.41e-14/3.65e-14` after 16
iterations, while central-difference map residuals remained
`1.00e-4/2.50e-5/1.26e-5`. This separates periodic fixed-point convergence
from sampled derivative/discretization error; it does not close the
production NUFFT/FMM or physical rectangle-boundary gates.

Route C addendum: a `256²`, two-step real-pair recompute flow on the AI RTX
A6000 completed with `114.08 s` forward and `538.09 s` backward, finite
gradient, zero flipped faces, and minimum signed-area ratio `0.86017`. This is
realistic multi-step evidence, but not an implicit adjoint or a scalable BHF
time integrator theorem.

Route C multigrid addendum: the differentiable coarse-to-fine prototype
(`32x32` coarse, `128x128` fine, two coarse plus two fine steps) reduced
forward/backward time from `26.92/93.46 s` for four fine-grid steps to
`13.78/50.95 s`, with relative map error `1.80e-7`, finite gradients, zero
flips, and minimum signed-area ratio `0.85958`. Receipt:
`artifacts/bhf_multigrid_32_128/bhf_multigrid_audit.json`. The universal
multigrid/homeomorphism and million-vertex memory gates remain open.

The receipt was rerun with CUDA peak-memory instrumentation: peak allocated
memory was `4,913,521,152` bytes (`~4.91 GB`), with fine-only versus
coarse-to-fine forward/backward times `19.94/97.95 s` versus `13.81/51.33 s`.

The attempted `64x64 -> 256x256` extension used about `24 GB` GPU memory and
was stopped after `27 min` without a receipt. This marks the direct-quadrature
multigrid scaling bottleneck explicitly; it is not a failure of the smaller
coarse-to-fine differentiability control.

Route E resource addendum: the intended `512²` radius-4/6/8 positive-cone
sweep was interrupted at about `63 min` and `460 MB` resident memory without
a JSON receipt. This is a bounded-cost non-result; it does not establish
failure of the underlying local cone fit.

A `256²` radius-4/6/8 follow-up was stopped after about `23 min` and `1,400 s`
CPU without a receipt. Profiling identifies the three-direction active-set
enumeration as the dominant assembly cost. This is recorded as a resource
limit and motivates a fixed active-set or spatially continuation-regularized
implementation.

Route B addendum: an order-12 nonperiodic treecode at `32,768` scattered
source/target points agreed with an independent AI-GPU direct sum to relative
L2 error `2.53e-7` and max error `7.52e-9`; the direct reference took `0.299 s`
for `1,073,741,824` interactions and the treecode took `8.39 s`. This confirms
that general-mesh whole-plane evaluation need not use a periodic FFT grid, but
the production FMM/singular-quadrature gate remains unresolved.

A matched near-field cloud at the same resolution gave relative L2/max errors
`4.37e-7/6.05e-8` (treecode `16.46 s`, direct GPU `0.299 s`). The pair is a
stronger far/near numerical control, but does not close the singular
self-interaction or production FMM gate.

Paired offset controls reached the singular regime more directly: the
`16,384`-point/`1e-4` offset case had relative L2 error `4.53e-4`, and the
`32,768`-point/`1e-3` case had `1.72e-3`. The deterioration is consistent with
Barnes--Hut expansion error near individual sources and reinforces that these
receipts are diagnostics, not a production principal-value solver.

The reflection-extension subroute was also exercised at a `256²` base mesh:
the doubled `512²` periodic solve converged in 13 iterations, had map-level
residual `1.65e-11`, minimum determinant `0.72089`, and axis violations below
`2.2e-8`. This closes only the restricted reflection-compatible boundary
control; arbitrary rectangle boundary data and production singular quadrature
remain unresolved.

The nonperiodic FD-versus-LSQC matched control was extended to `256²` cells:
the FD residual was `2.86e-10` but map RMS error remained `0.35494`, whereas
P1 LSQC had RMS error `9.10e-15`. This is realistic-resolution negative
evidence for the naive collocated B3 discretization, not completion of the
boundary-adapted solver.

Route E addendum: a `12,384`-vertex one-/two-ring unstructured comparison
reduced the local tensor-fit mean/p95 residual from `0.1290/0.7951` to
`0.00291/7.44e-12`, but the two-ring positive graph introduced `1,641` face
flips and minimum determinant `-4.93e-4`. The experiment explicitly records
that local positive cone fidelity is insufficient once the decoder graph is
nonplanar.

Route E addendum: planar barycentric enrichment at `8,384` and `12,384`
original vertices reduced tensor-fit mean/p95 residuals to
`0.0192/0.0699` and `0.0187/0.0652`, respectively, while retaining zero
refined-face flips and exact boundary values. The result is a promising
planar refinement control, not yet an arbitrary-`mu` convergence or global
QC decoder theorem.

Route E ablation: anisotropic fitting at the added barycentric center rows
kept zero flips but produced center-cone mean/p95 residual `0.608/2.297` and
minimum determinant `1.60e-14`; uniform center rows were more robust. Center
geometry/cone compatibility therefore remains an unresolved gate.

Route D addendum: determinant-aware log-weight backtracking on `256²` accepted
12/12 updates, reduced MSE from `9.5043e-4` to `3.0077e-6`, and retained zero
flips with minimum determinant `9.31e-6`. This strengthens positive Tutte
optimization evidence but does not establish arbitrary-μ expressivity or
remove the per-trial sparse-solve cost.

Route C addendum: the realistic `256²` determinant-root adaptive-safe flow kept
zero flips for requested steps `2.0` and `10.0`; the latter clipped to
`[3.90091, 0.267559]` with minimum ratio `0.06892`. Its finite but very large
gradient (`L2=2.23e6`) confirms that fixed-active root differentiation is stiff
near the fold boundary, so a smooth/global adjoint gate remains open.

The conservative smooth-safe prototype now supplies a differentiable lower
envelope for the determinant roots. It retained zero flips at `64²/128²/256²`
and had finite gradients; the `256²` receipt is
`artifacts/bhf_flow_smooth_safe_gpu_256_2step/bhf_flow_recompute_audit.json`.
Active-clipping directional checks are recorded at
`artifacts/bhf_smooth_safe_vjp_32/bhf_smooth_safe_vjp_audit.json` and
`artifacts/bhf_smooth_safe_vjp_64/bhf_smooth_safe_vjp_audit.json`. This closes
the local hard-switch differentiability control, but not the global implicit
adjoint, PV, or atlas gates.

Route E cone-phase result: the `256²` `rho`/angle sweep found exact local
positive-cone coverage at `rho=0.2/0.4` with radius 1 and at `rho=0.6` with
radius 2, but only `70.1%--88.1%` at `rho=0.8` and `30.3%--41.1%` at
`rho=0.9` with radius 4. This is a quantitative local feasibility boundary,
not a global vector-map injectivity or boundary-closure theorem.

Route D induced-μ clarification: a direct facewise-μ objective through the
positive implicit layer reached MSE `0.0087163` from `0.0228584` on a
`256x256` mesh (10 steps, lr `0.005`) with zero flips and a passing
injectivity certificate; the analogous `128x128` run reached `0.0097871`.
The result separates two claims that must not be conflated: positive-row
weights provide a hard-topology mechanism, while the observed μ reduction is
only finite-step optimization evidence and does not prove arbitrary-μ
expressivity or convergence. Learning-rate stress produced conditioning and
near-degeneracy failures, so this route remains open.

Route H 256² comparison clarification: the new direct-map safeguard baseline
uses `131,072` faces and an independent smooth-twist target. The unconstrained
case folds (`508` faces), while LIM/SLIM/AMIPS-style barriers remain certified
but incur `301--356 s` wall times. Because the matrix-free KKT receipt uses a
different target parameterization, these are not a same-objective head-to-head
solver ranking; the common-target comparison remains incomplete.

Route H common-target clarification: a four-step direct baseline now uses the
exact matrix-free KKT target on the same `256²` mesh. The unconstrained case
folds `2,754` faces; all three safeguarded variants remain certified with
zero flips. This supplies the missing finite-budget same-target baseline, but
not an equal-convergence optimizer ranking or a smooth active-set adjoint
theorem.

Route L extended-cone clarification: adding four two-step lattice directions
 to the existing axis-plus-diagonal positive periodic graph at `512²` reduced
 map/face-μ errors to `0.00681/0.15281` while retaining zero lifted-face
 flips. This materially improves expressivity, but it remains a regular-grid
 control and does not prove arbitrary triangulated-torus consistency or an
 arbitrary-μ hard decoder theorem.

Route A projection-budget clarification: increasing the random `256²`
facewise-μ projection budget from eight to 20 evaluations changed the
relative residual only from `0.7081017289` to `0.7081017153`; the solver
terminated at 11 evaluations and retained zero flips with minimum area ratio
`0.63326`. This is realistic evidence of a projection residual floor for one
incompatible target, not a proof that every incompatible field has the same
floor or that the projection manifold has been characterized.
