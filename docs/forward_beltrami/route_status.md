# Forward Beltrami route audit (live)

This table is a work ledger, not a completion claim. A route is marked
`partial` until its mathematical scope, differentiability, topology, and
realistic-resolution evidence are all covered.

| route | current evidence | status | remaining gate |
|---|---|---|---|
| A realizable facewise BC | shared-edge compatibility matrix; 256x256 residual `1.94e-18`; random small-mesh incompatibility; nullity study; 256² dual-holonomy/primal-edge reconstruction with `1.22e-15` manufactured-map error and huge random-field cycle residual; a project-level simply-connected P1 disk holonomy sufficiency/dimension derivation; nonlinear fixed-boundary projection with analytic sparse Jacobian through 256x256/131072 faces and independent no-flip audits; 128² sparse projection normal-equation VJP with tangent-direction error `8.47e-7` and explicit off-manifold error `5.56e-5`; 512² stress assembly with 6.79M Jacobian nonzeros; increased-budget 512² compatible projection converges in 6 evaluations with residual `9.59e-8`, zero flips, min determinant `0.9999999995`, and `384.24 s` wall time; new 256² random facewise μ projection remains at relative residual `0.7081` after 8 evaluations but retains zero flips and min determinant `0.6333`; R2 rectangle-side holonomy reconstruction through 512² with `1.44e-15` aligned error, winding one, zero flips, and min determinant `0.65459`; fixed-R2 nonlinear projection through 256² with relative `mu` error `3.01e-9`, zero flips, and min determinant `0.65460` | partial | formal publication-grade holonomy proof/independence conditions, projection uniqueness/manifold chart, exact second-order off-manifold VJP, arbitrary tangential-boundary optimization, arbitrary incompatible-target projection, and hard global guarantee |
| B Beurling/FFT | exact periodic multiplier; zero-padding audit at 512/1024; fixed-point zero-mode counterexample; GMRES tolerance sweep; nonuniform O(N²) control; 1024² CIC/cubic particle-mesh gridding with sinc⁴/sinc⁸ deconvolution; implicit VJP; matched central-difference symbol; 256² P1 two-face block symbol/projector audit; 512² nonperiodic sparse FD BVP, reusable factorization, boundary/coefficient VJPs; ILU-GMRES plus transpose control through 512²; matched 64²/96² rectangle FD-versus-LSQC audit exposing a checkerboard ambiguity; 65/129/257-node boundary-adapted conductivity FEM control with second-order map convergence and singularity exposure of collocated FD; identity-boundary boundary-free negative control through 257² with residual `0.27165→0.27356` and zero flips; 16,384-point disjoint-source direct cross-check; 4096-point nonperiodic Barnes–Hut treecode forward plus optimized exact fixed-tree source-value VJP and 1024/2048/4096 scaling; 16,384-point separated/near scattered treecode versus AI-GPU direct references with relative errors `2.88e-7/3.65e-7`; 32,768-point separated/near AI-GPU direct cross-checks with treecode relative errors `2.53e-7/4.37e-7`; batched P1 torch symbol with 2048² AI-GPU throughput `6.01 ms` and local NumPy parity; device-native arbitrary-scattered direct kernel through 32,768 points at `3.66e9` interactions/s with NumPy parity `4.73e-16`; FINUFFT 2.5.1 scattered prototype with resolution-matched smooth-cloud errors `4.51e-2/1.50e-2/1.41e-2` at 65,536 sources; new `512²` disjoint source/target direct GPU cross-check with zero-padded FFT errors `7.07e-2/4.41e-3/2.76e-4` for padding 2/4/8 | partial | production NUFFT/FMM or singular-quadrature backend, stronger rectangle preconditioner, plus a boundary-free arbitrary-`mu` rectangle solver |
| C sphere BHF | analytic Möbius truth velocity; normalized and arbitrary-base BHF kernels; seven-point piecewise-affine quadrature on a 128x128 (32768-face) mesh; local Duffy pole quadrature on 8192 faces; vertex-target near/far assembly through 128x128; interior-target and interior-edge Duffy subdivision through 128x128; adaptive two-step determinant-safe flow on 32x32; realistic 64² and 96² one-step determinant-safe BHF flow, with 96² runtime `216.16 s`, min determinant `0.63431`, and zero flips; 261120-face orientation audit; north/south coordinate and velocity chain-rule gate through 512x256; 64/128/256² and 512² global incident-vertex/edge Duffy PV convergence audit on 524,288 faces; GPU-native near/far/Duffy assembly through 256² with `36.21 s` assembly, `44.14 s` full step, and 512² with `1044.19 s` assembly/`1066.53 s` full step, zero flips, and 32² GPU/reference error `1.66e-7`; tensor-native CPU VJP through 64² with finite-difference agreement; fully real-pair GPU autograd through 128² on legacy Torch with checkpointed far blocks and `14.38/40.87 s` forward/backward; memory-bounded recompute flow through 128² two-step/64² four-step with zero flips; determinant-root safe-step stress through 64² with active-root clipping, fixed-active VJP, and zero flips; conservative smooth-safe controller through 64²/128²/256² with zero flips and finite VJPs, plus active-clipping directional checks | partial | full adaptive global PV theorem, scalable nonlinear BHF integration beyond the regular planar control, implicit adjoint beyond recomputation, quantitative smooth-controller efficiency/generalization, active-set switch theorem for the exact controller, and atlas coupling of the nonlinear BHF operator |
| D Tutte/Floater | hard convex-boundary decoder; 512x512 positive nonuniform-weight solve; sparse implicit VJP; 256x256 learned-edge-weight VJP; 64x64 and 128x128 positive-weight expressivity optimization with zero flips; realistic 256x256/197,120-edge expressivity optimization with zero flips; new 512x512/787,456-edge three-update expressivity stress with MSE ratio `0.0542`, min determinant `2.704e-6`, and zero flips; non-harmonic expressivity audit; 256² symmetric-versus-directed positive-row stress with zero flips and directed minimum determinant `4.94e-10`; 256² directed positive-row sparse implicit layer with finite boundary/logit VJPs, zero flips, and minimum determinant `1.07e-10` at logit spread 3; new 12,384-vertex nonuniform Delaunay directed implicit audit with finite VJPs, zero flips, and minimum determinant `8.01e-12` at spread 3; literature-backed positive-row theorem qualification for planar 3-connected graphs with convex boundary | partial | verify graph/boundary hypotheses for every target mesh, control conditioning as logits approach zero-weight limits, and arbitrary-μ expressivity/training stability |
| E M-matrix QC | SPD tensor; fixed-stencil counterexample; spectral decomposition; 256² finite-direction audit; conditioning-safe selector; integer wide-stencil fit through max step 6 with positive weights; 4096-vertex nonuniform Delaunay one-/two-ring cone audit; 4,056-vertex globally assembled positive-edge harmonic decoder with convex boundary and zero flips; 128²/256²/512² constant-tensor manufactured-solution audit; new 128²/256²/512² variable-tensor control showing second-order convergence for cone-exact fields and zero-order saturation for a rotating Beltrami field with residual `0.34669`; new 256²/512² positive-stencil-width stress exposing active-set conductance noise and operator errors up to `897.3` despite machine-small tensor fits; raster-neighbor canonicalization reduces those errors to `6.71e-2`/`1.68e-2` at 256²/512² while retaining machine-small fits | partial | variable tensors outside the positive cone, scan-order-independent canonical/regularized conductance selection, boundary closure beyond the square control, and a general QC decoder |
| F mesh-native flow | safe P1 flow; implicit Tutte layer; determinant-one affine shear; differentiable positive-increment decoder; 1024² coupled monotone/shear round-trip; 512² spatially varying triangular positive-increment decoder with zero flips; four-layer alternating triangular spatial composition at 512² with zero flips; coupled positive-latent prolongation with zero flips at 1024² | partial | arbitrary-QC expressivity and a mesh-independent sampled-P1 theorem |
| G certified PL residual | exact determinant-root safe step; active-face reporting and fixed-active derivative; 256x256 multi-step audit; 64x64 and 256x256 direct-vs-staged 180° continuation counterexamples; 256² automatic `60°→120°→180°` waypoint planner; new adaptive arbitrary-target rotation-plus-shear planner at 256² with direct-path stall error `129.57`, two successful half-parameter segments, zero flips, and final error `6.47e-15` | partial | arbitrary homotopy construction, nonsmooth switch handling, and H/KKT coupling |
| H barrier/optimization | existing LIM/SLIM/AMIPS and exact LSQC evidence; common 64x64 and 96x96/18,432-face comparisons with independent fold audits; 64² augmented-KKT versus hard-pin LSQC comparison with 9.32x forward and 7.79x complete-step speedups; dense nonlinear log-det barrier-KKT implicit adjoint through 48²; matrix-free Newton-CG KKT adjoint through 64² and 96² CPU with fresh-solve error `4.68e-11`; device-native Torch CG on AI RTX A6000 through 96² with 18,432 faces, zero flips, forward stationarity `9.54e-12`, and implicit directional error `3.47e-6`; 128² GPU stress with baseline implicit errors `5.53e-4`/`2.50e-4`; tighter `cg_rtol=1e-12` at 128² reduces the `beta=1e-3` error to `3.83e-6` with 310 adjoint HVPs; new 256² GPU baseline has 131,072 faces, stationarity `4.18e-12`, zero flips, minimum determinant `1.10e-5`, and implicit directional error `9.93e-4` with 534 adjoint HVPs; tighter 256² `cg_rtol=1e-12` lowers the error to `4.55e-6` with 637 adjoint HVPs | partial | nonsmooth active-set handling, larger/common solver comparison with a target-solver baseline, and GPU conditioning/accuracy beyond this control |
| I coarse-to-fine | bilinear prolongation truth and explicit folding counterexample; positive-increment parameter prolongation; 512²/1024² fine-grid audits; coupled injective prolongation with analytic inverse and zero flips; 512² spatial triangular coupling audit; 512² alternating triangular composition audit | partial | mesh-independent prolongation theorem and arbitrary-QC latent expressivity |
| J diffusion/flow + decoder | positive-increment hard decoder; 1024² mapping and coupled round-trip benchmarks; 128-interval Adam latent training; new 256-interval flow-matching training/integration on a 512² decoder with zero flips and minimum determinant `1.57e-6`; 1024² coupled coarse-to-fine hard decode; 512² spatial triangular hard decoder; 512² alternating triangular composition; conditional flow-matching integration with zero flips at 512² | partial | arbitrary-QC latent expressivity, full diffusion integration, and coupled high-resolution training comparison |
| K DEQ/implicit solver | sparse Tutte adjoint; periodic and rectangle transpose/coefficient VJPs; GMRES tolerance sweep; 256x256 Beurling implicit VJP; explicit torus affine-period VJP; 512² constant-affine map-plus-period VJP; 256² spatially varying zero-mode/map-objective composed VJP; 128² depth-6 Anderson equilibrium with implicit adjoint; 32²/48² dense nonlinear barrier-KKT implicit adjoint controls; 64²/96² matrix-free Newton-CG KKT adjoint; 96² device-native GPU HVP/CG control with fresh-solve directional error `3.47e-6`; 128² tighter-CG GPU control with `3.83e-6` directional error; new 256² GPU matrix-free KKT baseline with 131,072 faces, zero flips, and `9.93e-4` fresh-solve directional error; tighter 256² CG reduces that error to `4.55e-6` | partial | general nonlinear DEQ/root-KKT coupling, nonsmooth active-set handling, and GPU accuracy/conditioning beyond this control |
| L torus hard layer | explicit affine/quasi-periodic zero-mode lift; spatially varying `h` mean retained in map and periods; periodic connectivity helper; period-to-modulus parameterization; 512² positive-weight periodic Tutte embedding with zero flips; 512² spatial-`mu` axis-only control; 512² positive graph decoder with `(1,1)/(1,-1)` edges reducing cross-direction `mu` error from `1.025` to `0.362` with zero flips | partial | arbitrary triangulated-torus theorem, exact variable-coefficient consistency, and a general hard decoder |
| M orbifold sphere | 256x128 closed two-cap positive-Tutte assembly; chart orientation fix; seam mismatch `6.75e-16`; outward orientation audit; 256-ring×512-sample radial cone chart for `alpha=0.65,1.4` with zero flips; realistic 512x256 global two-chart cone atlas with mixed north/south exponents, seam `2.48e-16`, inverse error `<1.56e-14`, and zero flips | partial | arbitrary cone-angle transition law, orbifold normalization/global chart theorem, and comparison with BHF |
| N learned initializer/preconditioner | GMRES warm start; six-example learned scalar initializer; order-2 truncated-Neumann initializer; two-feature polynomial initializer at 256² gives 25→23 iterations and 1.033x on one case; 12-example heterogeneous stress audit preserves roots but has mean 1.008x and one 0.849x slowdown; true order-3 matrix-free Neumann preconditioner at 256² gives 4/6/9/15 iterations across amplitudes 0.25–0.70 with cost-inclusive speedups up to 1.249x; six 256² random smooth μ fields show only 1.023x median for order-1 and slowdowns for orders 2/3; three 512² random smooth fields reduce cold iterations 13/15/22 to 7/8/12 with order-1 median speedup 1.058x, while orders 2/3 have median slowdowns 0.955x/0.923x and all roots remain within 4.5e-9; two-case 1024² audit reduces cold iterations 13/15 to 7/8, with median speedups 1.052x/1.008x/0.907x; new two-case 512² rough-band stress reduces cold iterations 18/31 to 9/16 and median speedups 1.187x/1.185x/1.128x for orders 1/2/3 | partial | learned/general preconditioner training, robustness beyond Neumann structure, and cost-inclusive generalization |

The route ledger is intentionally conservative: numerical certificates are not
formal topology proofs, and an analytic truth map is not an arbitrary-μ solver.

Route C addendum: the 128² Duffy near/far one-step flow audit completed in
`554.42 s` with minimum determinant `0.634327` and zero flips. This confirms
no-folding at that resolution but also exposes the current direct quadrature
scaling bottleneck; it does not close the scalable BHF-integrator gate.

Route G addendum: a 256² active-set scan found 8 active-face switches in 47
trials. Fixed-active derivatives matched finite differences at `1e-7`--`1e-6`,
whereas switch cases reached absolute discrepancies up to `2.92e-2`; the
safe-step map is therefore only piecewise differentiable.

Route B addendum: the `16,384`-point paired-offset cloud was recomputed against
an independent blocked `complex128` CPU direct reference. The order-12,
`theta=0.22` treecode error was `5.49e-13` (and `3.15e-15` at
`theta=0.12`, order 16), while the earlier A6000 comparison reported
`4.53e-4`; this isolates the latter as external float32/device-reference error
for that disjoint near-source test. Coincident-target principal-value
quadrature and a production FMM/NUFFT backend remain open.

Route D addendum: literature-backed Tutte/Floater conditions allow strictly
positive, row-stochastic directed weights on a planar 3-connected graph with a
strictly convex boundary; symmetry is needed for an SPD energy, not for the
conditional planar embedding theorem. The new theorem note records this
boundary/graph qualification and reclassifies the remaining gate as hypothesis
verification, conditioning, and arbitrary-`mu` expressivity rather than a
search for a directed-weight topology counterexample.

Route M addendum: a `512x256` five-pair cone-exponent sweep preserved seam
error `2.48e-16` and zero flips for all cases; the worst inverse error was
`1.38e-11`, with minimum orientation proxy `3.38e-10`. This broadens the
analytic atlas evidence but does not close arbitrary transition/BHF coupling.

Route E addendum: a `66,024`-vertex unstructured Delaunay decoder with
`65,000` interior unknowns completed in `117.71 s`, retained exact boundary and
zero flips, but had local tensor-fit mean/p95 residuals `0.1246/0.7716` and
minimum determinant `6.59e-15`. Positive unstructured decoding scales, while
one-ring cone fidelity remains the limiting gate.

Route B addendum: the new `256²/512²/1024²` fixed-point residual-floor sweep
separates iteration convergence from sampled-map consistency. At iteration 16,
the Fourier-consistent residuals were `1.06e-14/2.41e-14/3.65e-14`, whereas
second-order central-difference residuals were `1.00e-4/2.50e-5/1.26e-5`.
This is evidence for a discretization floor even after the periodic fixed
point has converged, not evidence that the rectangle boundary problem is
solved.

Route C addendum: the memory-bounded real-pair BHF recompute layer completed a
`256²`, two-step AI RTX A6000 run (`131,072` faces) with `114.08 s` forward,
`538.09 s` backward, finite gradient, zero flipped faces, and minimum signed
area ratio `0.86017`. This extends multi-step differentiability evidence beyond
`128²`, but the backward cost and lack of an implicit adjoint remain open.

Route B addendum: the order-12 nonperiodic treecode was independently checked
at `32,768` scattered source/target points against an AI RTX A6000 direct sum.
The treecode took `8.39 s`, the direct reference took `0.299 s` for
`1,073,741,824` interactions, and the relative L2/max-absolute errors were
`2.53e-7/7.52e-9`. This strengthens the general-mesh nonperiodic evidence but
does not close the production FMM or singular self-quadrature gate.

The same `32,768`-point check on a near-field cloud took `16.46 s` for the
treecode and `0.299 s` for the GPU direct reference, with relative L2/max
errors `4.37e-7/6.05e-8`. The separated and near receipts together provide a
realistic far/near control; coincident singular targets and production FMM
accuracy remain open.

Paired offsets expose the singular regime more directly: the `16,384`-point
offset-`1e-4` control gave relative L2 error `4.53e-4` (`69.23 s` treecode,
`0.0799 s` GPU direct), while the `32,768`-point offset-`1e-3` control gave
`1.72e-3` (`170.33 s` versus `0.299 s`). These remain diagnostic controls;
singular quadrature and production FMM accuracy are unresolved.

The reflection-compatible rectangle control now doubles a `256²` coefficient
to `512²`, converges in 13 iterations, reaches map-level residual `1.65e-11`,
and retains minimum determinant `0.72089`; symmetry-axis violations are below
`2.2e-8`. It is a valid restricted boundary model, not a universal rectangle
solver: incompatible traces, arbitrary boundary homeomorphisms, and corner
normalization remain open.

The matched nonperiodic rectangle control was extended to `256²/131,072`
faces: central FD residual `2.86e-10` still coexisted with map RMS error
`0.35494`, while P1 LSQC recovered the manufactured map at `9.10e-15`.
This realistic negative control confirms that the collocated FD operator is
not a valid rectangle d-bar inverse despite sparse direct solves; a
boundary-adapted P1, staggered, or mixed formulation is still required.

Route E addendum: on a `12,384`-vertex unstructured Delaunay mesh, expanding
from one-ring to two-ring positive directions reduced tensor-fit mean/p95
residuals from `0.1290/0.7951` to `0.00291/7.44e-12`, but the wide decoder graph
produced `1,641` original-face flips and minimum determinant `-4.93e-4`.
Positive local cone fidelity therefore does not imply a global hard-injective
map once the graph ceases to be planar.

Route E addendum: planar barycentric enrichment was tested at `8,384`
original vertices/`49,146` refined faces and `12,384` original vertices/
`73,146` refined faces. Tensor-fit mean/p95 residuals were
`0.0192/0.0699` and `0.0187/0.0652`; both retained zero refined-face flips
and exact boundary values. This is a promising planar refinement direction,
but face-center rows are still uniform and the minimum determinant was only
`2.80e-7`/`7.45e-8`.

Route E ablation: fitting anisotropic conductances also at the barycentric
face-center rows produced center-cone mean/p95 residual `0.608/2.297` and
all-node residual `0.414/1.918`; the refined map kept zero flips but its
minimum determinant fell to `1.60e-14`. Uniform center rows are therefore
currently the more robust planar enrichment choice.

Route D addendum: a determinant-aware log-weight trust-region optimizer on a
`256²` mesh accepted all 12 updates, reduced MSE `9.5043e-4 → 3.0077e-6`, kept
minimum determinant `9.31e-6`, and retained zero flips. This improves stability
over the oscillating unconstrained Adam stress, but each backtracking trial is
a sparse solve and arbitrary-μ expressivity remains open.

Route C addendum: realistic `256²` adaptive-safe BHF recompute runs completed
with zero flips. Requested step `2.0` accepted `[2.0, 2.0]` and retained
minimum ratio `0.18236`; requested step `10.0` activated root clipping with
`[3.90091, 0.267559]` and minimum ratio `0.06892`. Both gradients were finite,
but the clipped-root gradient norm reached `2.23e6`, exposing active-root
stiffness.

Route C addendum: the conservative smooth-safe controller was run on AI-GPU
`64²/128²/256²` meshes with zero flips and finite gradients. The `256²`
two-step run took `114.18/518.56 s` forward/backward and retained minimum area
ratio `0.182356`; a requested-step-`10` `64²` stress clipped smoothly to
`[4.07421,0.283442]` with minimum ratio `0.007757`. Directional checks under
active clipping gave relative errors `2.65e-3/3.56e-3` at `32²` and
`2.84e-2/5.38e-3` at `64²` for epsilons `1e-3/3e-4`. This removes the hard
active-face selection from the local VJP, but remains a conservative replay
adjoint rather than a global implicit BHF derivative.

Route N addendum: the realistic `1024²` random-smooth audit completed two
heterogeneous cases. Cold GMRES took 13/15 iterations; order-1 took 7/8,
order-2 took 5/5, and order-3 took 4/4. However, including preconditioner
construction, median wall-clock speedups were only `1.052x/1.008x/0.907x`
for orders 1/2/3. All runs converged with residuals below `1.7e-9` and
root differences below `1.4e-9`. This strengthens the resolution evidence
while confirming that iteration reduction alone is not a production speedup
claim; learned/general preconditioners and rough-coefficient robustness remain
open.

Route N rough-coefficient addendum: a deterministic high-frequency Fourier-band
control at `512²` completed two cases. Cold iterations were `18/31`; orders
1/2/3 reduced them to `9/16`, `6/11`, and `5/8`, with cost-inclusive median
speedups `1.187x/1.185x/1.128x`. Residuals stayed below `2.5e-9` and root
differences below `4.1e-9`. This is encouraging for this finite rough-band
model, but it is not evidence for arbitrary measurable coefficients or a
learned/general preconditioner.

Route B boundary-free addendum: applying the conductivity FEM with canonical
identity boundary data to a nontrivial manufactured coefficient was tested at
`129²` and `257²`. The d-bar residuals were `0.27165` and `0.27356` (ratio
`1.007`), while the minimum triangle determinants stayed `0.56297` and
`0.56270` with zero flips. Thus local orientation preservation survives, but
the canonical boundary choice is incompatible with the coefficient and the
residual does not converge away. This is a realistic negative control, not a
claim that no boundary-adapted solver can exist.

Route D differentiability addendum: a directed positive-row Tutte decoder was
implemented with sparse forward factorization and an implicit transpose solve.
At `256²` (`65,025` interior rows), logit spreads 1 and 3 produced finite
boundary/logit VJPs, zero flipped faces, and minimum signed areas
`3.66e-3` and `1.07e-10`; the convex-parallelogram boundary audit certified
both maps. The spread-3 result is numerically close to the injectivity
threshold, so this is evidence for a differentiable directed layer, not a
general nonsymmetric positivity theorem.

Route C scaling addendum: the GPU-native blocked near/far/Duffy assembly was
run at `512²` (`524,288` faces) on the AI RTX GPU. One determinant-safe step
required `1044.19 s` for assembly and `1066.53 s` end-to-end; the minimum
signed-area ratio was `0.634365`, with zero flipped faces and a passing
injectivity audit. This closes the realistic-resolution topology/throughput
measurement, but the roughly 17.8-minute direct quadrature cost is itself a
production bottleneck; it does not establish scalable BHF integration.

Route C unstructured addendum: the real-pair differentiable near/far/Duffy
assembly was run on jittered rectangular Delaunay meshes, not a translationally
regular grid. At `128²` (`16,641` vertices, `32,768` faces), near orders 8 and
16 took `6.10 s` and `15.76 s`; both VJPs were finite and their relative
velocity difference was `1.92e-7`. At `256²` (`66,049` vertices,
`131,072` faces), the corresponding times were `63.75 s` and `109.56 s`, with
relative order difference `1.14e-7`. Candidate velocity steps `0.25/0.5/1.0`
retained zero flipped faces at both resolutions; the `256²` minimum signed-area
ratios were `0.2860/0.2854/0.2840`, and the independently ordered boundary
polygon remained positive. Receipts:
`artifacts/bhf_unstructured_gpu_128/bhf_unstructured_gpu_audit.json` and
`artifacts/bhf_unstructured_gpu_256/bhf_unstructured_gpu_audit.json`.
This closes a meaningful mesh-native differentiability/near-quadrature control,
but not the global adaptive PV theorem or scalable multi-step implicit adjoint.

Route B near-field addendum: an exact-near-aware tree traversal was added and
tested on the existing `16,384`-point paired-offset-`1e-4` cloud. At
`theta=0.22`, order 12, both `near_radius=0.01` and `0.05` left the relative
L2 error exactly at `4.52699785e-4` (baseline `60.53 s`; radius `0.05`,
`60.98 s`). The opening criterion already refines the paired source cluster
to leaves, so this correction cannot address the remaining error. The result
points to singular/PV quadrature or a production FMM as the required next
step, not a larger ad-hoc near radius.

Route B FINUFFT box-tail addendum: on the realistic `65,536`-source smooth
cloud, holding the physical cutoff fixed at `804.25` while increasing
`(L,modes)` from `(8,2048)` to `(12,3072)` and `(16,4096)` changed relative
error only `0.0141276 -> 0.0140020 -> 0.0139502`. The finite-box tail is
therefore not the dominant error in this control; point-cloud quadrature,
target interpolation, Fourier truncation, and coincident-target PV treatment
remain coupled. Receipt:
`artifacts/beurling_finufft_box_tail_65536/beurling_finufft_box_tail_audit.json`.

Route C differentiable multigrid prototype: a coarse-to-fine BHF layer was
implemented with a differentiable PyTorch bilinear prolongation between a
`32x32` coarse grid and a `128x128` fine grid. Two BHF steps were run on the
coarse grid and two correction steps on the fine grid, and the result was
compared with four steps entirely on the fine grid. The coarse-to-fine path
had `13.78 s` forward and `50.95 s` backward versus `26.92 s` and `93.46 s`
for the fine-only reference, finite gradients, relative map error `1.80e-7`,
zero flipped fine faces, and minimum signed-area ratios `0.85958` (multigrid)
and `0.85958` (reference). Receipt:
`artifacts/bhf_multigrid_32_128/bhf_multigrid_audit.json`. This shows that
multilevel scheduling can reduce cost without severing local backpropagation,
but it is not yet a million-vertex memory result, a general unstructured-mesh
prolongation, or a global homeomorphism theorem.

The receipt was rerun with peak-memory instrumentation: the same result was
`19.94 s`/`97.95 s` for the fine-only reference and `13.81 s`/`51.33 s` for
coarse-to-fine, with peak CUDA allocation `4,913,521,152` bytes (about
`4.91 GB`).

The next `64x64 -> 256x256` multigrid run was allowed to use the AI GPU for
about `27 min` (roughly `24 GB` device memory) but produced no receipt before
being stopped. This is a resource-bounded scaling observation: the `32x32 ->
128x128` prototype is valid, while the same direct-quadrature design does not
yet scale to a 256² fine grid within a practical interactive budget.

The planned `512²` radius-4/6/8 positive-cone sweep was stopped after about
`63 min` and approximately `460 MB` resident memory without a receipt. This
is recorded as a resource-bounded engineering limit, not as an algorithmic
failure; a lower-resolution control is being used to separate cone
expressivity from the cost bottleneck.

The follow-up `256²` radius-4/6/8 control was likewise stopped after about
`23 min` (roughly `1,400 s` CPU) before emitting a receipt. The cause is the
combinatorial enumeration of all one-, two-, and three-direction active sets
in the vectorized nonnegative cone fit; it is a concrete assembly-cost limit,
not evidence that the cone fit is mathematically impossible.

Route J latent-resolution addendum: conditional flow matching was rerun with
`256` positive-increment intervals, `600` training steps, and `64` Euler
integration steps, while decoding on a `512²` mesh (`524,288` faces). The
held-out latent RMSE was `0.09174`; the decoded map remained finite with zero
flips and minimum determinant `1.571e-6`. This extends hard-topology evidence
to a substantially higher latent dimension, but remains a toy conditional
distribution rather than arbitrary-Beltrami expressivity or a full diffusion
model.

Route H/K conditioning addendum: on the AI RTX GPU at `128²` and
`beta=1e-3`, tightening device-native CG tolerance from `1e-10` to `1e-12`
increased adjoint HVPs from `260` to `310` and reduced fresh-solve directional
error from `5.53e-4` to `3.83e-6`. Forward stationarity stayed below `7e-12`,
zero faces flipped, and minimum determinant stayed `4.37e-5`. This identifies
adjoint-CG conditioning as the dominant source of the earlier error while
quantifying the extra HVP cost; active-set nonsmoothness and larger-resolution
conditioning remain open.

Route D general-mesh addendum: the same directed implicit layer was run on a
nonuniform Delaunay mesh with `12,384` vertices, `24,382` faces, and `12,000`
interior rows. Logit spreads 1 and 3 both had finite VJPs, zero flips, and
passing convex-boundary injectivity audits; minimum signed areas were
`3.97e-3` and `8.01e-12`, respectively. This demonstrates that the sparse
implicit layer is not tied to a uniform FFT-like grid, while the spread-3
conditioning warning persists.

Route E cone-phase addendum: a `256²` `rho`/angle scan found 100% exact local
coverage for `rho=0.2/0.4` with radius-1 directions, radius-2 recovery at
`rho=0.6`, but only `70.1%--88.1%` coverage at `rho=0.8` and `30.3%--41.1%`
at `rho=0.9` even with radius 4. This quantifies the mesh-cone boundary before
global assembly; it is not an injective vector-map theorem.

Route D induced-μ addendum: direct differentiation of facewise μ through the
positive directed implicit layer reduced μ MSE by factors `0.428` (`128²`,
20 steps, lr `0.02`) and `0.381` (`256²`, 10 steps, lr `0.005`) while both
runs retained zero flips and passed independent injectivity audits. A
`256²`, lr `0.02` run retained topology but worsened the loss, and a
`128²`, lr `0.25` stress reached numerical near-degeneracy. Route D therefore
has positive hard-topology and differentiable μ-fitting evidence, but remains
partial on arbitrary-μ expressivity, conditioning, and convergence.

Route A projection-budget addendum: the `256²` random facewise-μ projection
was extended from the earlier eight-evaluation run to a `max_nfev=20` budget.
The sparse trust-region solver stopped at 11 evaluations, with relative
residual `0.7081017153` versus `0.7081017289` previously, zero flips, minimum
signed-area ratio `0.63326`, and an independent injectivity certificate. The
near-identical residual after the larger budget is evidence for a
compatibility/projection floor in this random target, not merely insufficient
iteration; it does not yet provide a characterization of the projection image.
Receipt:
`artifacts/compatibility_projection_random_highres_256_nfev20/compatibility_projection_random_highres_audit.json`.

Route H 256² direct-baseline addendum: an independent eight-step direct-map
comparison on a `131,072`-face mesh found that the unregularized map reached
data loss `7.31e-4` but produced `508` flipped faces, whereas LIM/SLIM/AMIPS
style safeguards retained zero flips with final losses
`6.72e-3/6.99e-3/6.83e-3` and wall times `301/338/356 s`. This quantifies the
high-resolution cost/topology tradeoff, but the target parameterization differs
from the matrix-free KKT control, so the common-target solver comparison gate
remains open.

Route H common-target addendum: the direct baselines were rerun with the exact
zero-boundary target used by the 256² matrix-free KKT control, same identity
initialization, and four steps. The unconstrained map folded `2,754` faces;
LIM/SLIM/AMIPS retained zero flips with losses
`8.11e-4/8.63e-4/8.43e-4` and wall times `159/167/167 s`. This closes the
finite-budget same-target topology/compute baseline, while equal-convergence
objective ranking and active-set KKT smoothness remain open. Receipt:
`artifacts/barrier_common_target_256/barrier_common_target_audit.json`.

Route L extended-cone addendum: an eight-direction positive periodic graph
with lattice offsets through `(2,1)` and `(1,2)` was tested at `512²`. Relative
map/face-μ errors were `0.00681/0.15281`, versus `0.01679/0.36217` for the
four-direction axis-plus-diagonal graph; zero lifted-face flips were retained
with minimum determinant `2.93e-6`. This confirms a direction-cone
expressivity bottleneck, not a failure of periodic hard orientation. Receipt:
`artifacts/torus_extended_graph_decoder_512/torus_extended_graph_decoder_audit.json`.
