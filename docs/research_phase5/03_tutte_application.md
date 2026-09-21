# Route I application experiments: legality, engineering cost, and instance fitting

## 1. Questions and scope of the evidence

This chapter evaluates three different questions: (Q1) are the proposed ground-truth maps and image pairs legal inputs; (Q2) can the implemented layer complete a forward/backward workload across mesh, batch, and composition sizes; and (Q3) can raw learnable parameters fit a shared deformation or image pair through actual solves and interpolation? A positive answer to one does not answer the others. In particular, legal ground truth is not fitting accuracy, a finite gradient is not an accurate VJP, and surrogate forward/backward timing is not registration time.

The map and derivative definitions, exact topology hypotheses, numerical rejection rules, and backend distinctions are given in [the methods chapter](02_tutte_fast_solver.md). The [official TutteNet audit](01_tuttenet_code_audit.md) prevents interpreting our mesh or legacy SciPy reference as an exact reproduction of official center-split/KLU execution. The [worklog](WORKLOG.md) preserves development failures and preliminary probes. Here, the evidence population is fixed to the named receipts below; later investigations are not silently substituted for failed rows.

All experimental counts were recomputed from local JSON/JSONL, with the CSVs used as readable summaries. Engineering `status="ok"` means the entire configured workload completed, including its final audit. Formal-instance `status="success"` means the runner completed its prescribed optimization budget. **Neither status means that the objective threshold was reached.** Failures remain in every requested-run denominator; unavailable accuracy and timing fields are not replaced by zero.

## 2. Data, coordinates, and experimental variables

### 2.1 Control and query discretizations

The source is the unit square, triangulated by `structured_rectangle(N-1, N-1)`. Here \(N\) is the number of control vertices per side, not the number of cells. There are

\[
n=N^2,\quad n_I=(N-2)^2,\quad n_B=4(N-1),\quad
n_F=2(N-1)^2.
\]

Each cell uses the lower-left/upper-right diagonal. For square query resolution \(R\), \(Q=R^2\), and the query coordinates are \(q_{ab}=(b/(R-1),a/(R-1))\), including the endpoints. Thus \(R=256\) and \(512\) mean 65,536 and 262,144 coordinate queries respectively. They do not change the number of control unknowns for a fixed \(N\).

We use \(B\) for batch size, \(L\) for the number of composed layers, \(s\) for a pseudorandom seed, and \(\sigma\) for the Gaussian latent-logit scale called `strength` or `target_strength` in the receipts. It is not a prescribed displacement, Beltrami magnitude, condition number, or distortion bound. `dtype` specifies public tensors and iterative arithmetic; the direct/reference backends still solve internally in float64 as described below. Coordinates and map errors are in unit-source-domain coordinates, not physical pixels or millimetres.

### 2.2 Analytic ground-truth maps

The eight analytic maps are defined on \(q=(x,y)\in[0,1]^2\). Put \(c=(1/2,1/2)\), \(r=q-c\), and let \(\mathcal R_\theta\) be planar rotation through \(\theta\) radians. Their formulas and conservative continuous determinant lower bounds are:

| Map | Formula | Continuous lower bound for \(\det Df\) |
|---|---|---:|
| `affine_anisotropic` | \((1.15x,0.82y)\) | 0.943 |
| `shear` | \((x+0.25y,y)\) | 1 |
| `rotation_stretch` | \(c+\mathcal R_{0.25}\operatorname{diag}(1.15,0.85)r\) | 0.9775 |
| `twist` | \(c+\mathcal R_{0.60\|r\|^2}r\) | 1 |
| `sinusoidal_shear` | \((x+0.08\sin(2\pi y),y)\) | 1 |
| `local_compression` | \(c+[1-0.25e^{-\|r\|^2/0.12}]r\) | \(0.75^2\) |
| `local_expansion` | \(c+[1+0.25e^{-\|r\|^2/0.12}]r\) | \(1-0.5e^{-1.5}\) |
| `boundary_sliding` | \((x+0.06\sin(2\pi x),y)\) | \(1-0.12\pi\) |

These bounds concern the continuous formulas. Each sampled control P1 map is independently checked rather than inferred safe from the bound. Only local compression and boundary sliding remain in the closed unit square throughout the sampled meshes; the other six can be topologically valid geometric targets without being admissible for this in-range image sampler. Definitions are implemented in [the shared benchmark module](../../src/qcopt/neural_bijection/benchmarks.py).

### 2.3 Positive directed targets and images

Random Tutte targets are generated on CPU in float64 using strictly positive directed row-softmax weights, ordered per-side softmax boundary segments, and a direct solve. One seeded Torch generator draws interior logits first and boundary logits second, both as \(\sigma\mathcal N(0,1)\). The target includes both control coordinates \(Y^*\) and their dense P1 evaluations \(F^*(q)\). Independent topology auditing is separate from its generating linear system. Its membership in the directed family is numerical, up to the CPU-double solve; membership in the symmetric-conductance family is not assumed.

The four scalar image families are deterministic functions sampled at the query grid and independently normalized by \((I-\min I)/(\max I-\min I)\). No random image noise is added. For reproducibility, their unnormalized intensities are:

* Checkerboard: \(\operatorname{mod}(\lfloor8x\rfloor+\lfloor8y\rfloor,2)\).
* Smooth blobs: \(e^{-[(x-.30)^2+(y-.34)^2]/.018}+.75e^{-[(x-.70)^2+(y-.62)^2]/.032}+.45e^{-[(x-.52)^2+(y-.18)^2]/.010}\).
* Medical phantom: define \(E(a,b,u,v)=1-((x-a)/u)^2-((y-b)/v)^2\), and \(S(t)=(1+e^{-t})^{-1}\). The intensity is \(.22S(55E(.50,.51,.39,.46))+.58S(70E(.39,.53,.16,.25))+.38S(70E(.64,.45,.12,.18))-.20S(80E(.51,.67,.08,.10))\).
* Textured: \(.40\sin(6\pi x+.3)+.27\cos(10\pi y-.5)+.18\sin(8\pi(x+.7y))+.15\cos(14\pi(.4x-y))+.35e^{-[(x-.67)^2+(y-.31)^2]/.020}\).

The “medical” case is a synthetic phantom, not clinical registration evidence. The checkerboard includes discontinuities, flat regions, and periodic ambiguity; low image error and unique geometric recovery are different questions even for a topology-preserving map.

For registration, \(F\) is the **fixed-to-moving backward coordinate map**. With moving image \(I_m\), the fixed image is generated once as

\[
I_f(q)=\mathcal B\bigl(I_m,F^*(q)\bigr),
\]

where \(\mathcal B\) is bilinear interpolation with `align_corners=True`; the Torch normalized grid is \(2F^*(q)-1\). The fitting loss samples \(I_m\) at the learned \(F_\theta(q)\), not at an inverse. Formal image targets have height one; map-fitting targets have width one and height 0.9. Image fitting fixes the height at its represented unit-height initialization, whereas map fitting learns height from an initial value of one.

The sampler's underlying `padding_mode="border"` is disclosed. The instance objective first rejects coordinates outside the unit square by more than \(64\epsilon_{\rm dtype}\), so unrestricted out-of-domain padding is not an optimization strategy. Sub-tolerance floating excursions remain subject to the sampler's boundary behavior. Nonfinite images/maps are rejected. No inverse map, inverse-consistency test, mask-based extension, or landmark objective is present.

## 3. Metrics and interpretation

### 3.1 Topology and geometric margins

For oriented source face \(t=(i,j,k)\), define

\[
J_t=[y_j-y_i\;y_k-y_i][x_j-x_i\;x_k-x_i]^{-1},\quad
d_t=\det J_t,\quad a_t=a_t^{\rm source}d_t.
\]

`flip_count` is \(\#\{t:d_t\leq0\}\), including degeneracies. `minimum_signed_area` is \(\min_ta_t\), and `minimum_area_ratio` is \(\min_td_t\). `boundary_order_min_gap` is the minimum Euclidean length of a consecutive target boundary edge; it is a separation margin, **not** by itself a boundary-order proof.

`global_injectivity_certificate` records the independent [floating-point injectivity audit](../../src/qcopt/injectivity.py): one source boundary loop, positive determinant margin, simple counterclockwise target boundary, and valid interior cyclic links/winding. Its default base tolerance is \(10^{-12}\), scaled using the target coordinate range; interior winding is compared to \(2\pi\) with tolerance at least \(10^{-8}\). The stricter “certificate” boolean and `flip_count=0` are not synonymous, because small positive determinants can fail a tolerance margin. The source disk and the exact PL global-inversion hypotheses are described in the methods chapter. This remains a numerical audit, not an exact-predicate theorem for every floating input.

In engineering runs the independent audit is on every layer/sample of the final identical-parameter repeat; the solver additionally screens every returned control map. In formal optimization it is recomputed at every recorded objective evaluation, including LBFGS line-search trials. `all_iterates_certified` means all those observed control maps passed, not that the continuous path between optimizer iterates was checked. Neither statement certifies a new P1 interpolation of a composed dense field on the original mesh: composed maps have a refined affine partition. The engineering field `composite_original_mesh_p1_certificate` is deliberately null.

### 3.2 Map and Beltrami error

Given a fitted control map \(Y\) and a separately generated target \(Y^*\), the [common metrics](../../src/qcopt/neural_bijection/metrics.py) are

\[
e_{\rm map}=\sqrt{\frac1n\sum_i\|y_i-y_i^*\|_2^2},\qquad
e_{{\rm map},\infty}=\max_i\|y_i-y_i^*\|_2.
\]

For \(f=u+iv\), each triangle has

\[
f_z=\tfrac12(u_x+v_y)+\tfrac i2(v_x-u_y),\quad
f_{\bar z}=\tfrac12(u_x-v_y)+\tfrac i2(v_x+u_y),\quad
\mu_t=f_{\bar z}/f_z.
\]

The coefficient errors are unweighted face averages, not area-weighted errors:

\[
e_\mu=\sqrt{\frac1{n_F}\sum_t|\mu_t-\mu_t^*|^2},\qquad
e_{\mu,\infty}=\max_t|\mu_t-\mu_t^*|.
\]

These formal-instance accuracy fields are measured on **control vertices/faces**, even though the fitting objective is dense. The legality suite instead reports minimum/median/95th-percentile/maximum \(|\mu_t|\), using NumPy quantiles. It has no fitted prediction and therefore reports no map or coefficient error against itself. All inverse-consistency fields remain null/blank because no inverse was computed.

### 3.3 Objectives and threshold observations

The supervised objective and scalar image objective are respectively

\[
\mathcal L_{\rm map}=\frac1{2Q}\sum_q\|F_\theta(q)-F^*(q)\|_2^2,
\qquad
\mathcal L_{\rm image}=\frac1Q\sum_q|\mathcal B\bigl(I_m,F_\theta(q)\bigr)-I_f(q)|^2.
\]

The factor two in the first denominator comes from averaging the two coordinate components. Dense vector RMSE would be \(\sqrt{2\mathcal L_{\rm map}}\); it is not identical to control-vertex \(e_{\rm map}\). The engineering-only scalar surrogate is \(\mathcal L_{\rm eng}=(2BQ)^{-1}\sum_{b,q}\|F_b(q)\|^2\). It supplies a real adjoint workload, but there is no fitting target, image sampler, or optimizer in that experiment.

Formal thresholds are \(\tau_{\rm map}=10^{-4}\) and \(\tau_{\rm image}=10^{-3}\), chosen in the separate seed-20260921 pilot before the seed-20260922 formal runs, as recorded in the worklog. If \(j\) is the zero-based first independently audited evaluation with \(\mathcal L_j\leq\tau\), `evaluations_to_threshold` is \(j+1\). Solve count and observation wall time are those at that evaluation, before its not-yet-executed adjoint. If no evaluation hits, all three threshold fields are missing, not zero and not a timeout estimate. `best_objective` is the minimum over audited evaluations; `final_objective` is the explicit final evaluation. For LBFGS, a threshold hit or best value in a line-search trial is not necessarily an accepted optimizer state.

### 3.4 Residual and conditioning proxies

Engineering receipts independently recompute each primal residual from native realized weights and returned coordinates, accumulating on CPU in float64, without the solver's matvec. For directed rows this is \(r_i=y_i-\sum_jp_{ij}y_j\); for symmetric conductances it is \(r_i=\sum_j\widehat c_{ij}(y_i-y_j)\). Relative residual divides each coordinate-column norm by the corresponding Dirichlet RHS norm. Reported maxima range over layer/sample/coordinate columns. Adjoint residuals, when available, come from the solver diagnostics and are not an independent dense-reference VJP comparison.

Minimum supported probability and normalized conductance contrast \(\max_e\widehat c_e/\min_e\widehat c_e\) are descriptive conditioning proxies, not measured matrix condition numbers. Neither a relative residual near \(10^{-5}\) nor a finite gradient establishes map, \(\mu\), or VJP accuracy. Native float32 acceptance and independent float64 replay need not have identical residuals.

## 4. Protocol, counting, timing, and memory

### 4.1 Exact matrices

The legality suite uses \(N\in\{11,17,25,33,49\}\): eight analytic maps per mesh (40 cases); positive directed targets with \(\sigma\in\{0.25,1,3\}\) and \(s\in\{1701,1702,1703\}\) per mesh (45 cases); and four image families at \(R\in\{256,512\}\) (8 cases). Map generation is CPU float64; image sampling is float32. Random legality targets have height one and use a minimal 2-by-2 query table because only control geometry is being audited.

The normal engineering product is

\[
N\in\{11,17,25,33,49\},\quad B\in\{1,4,8\},\quad
L\in\{1,2,4\},\quad R\in\{256,512\}.
\]

There are 90 configurations per backend. CPU runs compare `reference` and `direct`; GPU runs compare `directed_iterative` and `symmetric`, giving 180 requests on each host. All are configured for public float32, one Torch thread, seed 1701, \(\sigma=0.15\), one warmup, and two timed repeats; only completed rows reach both repeats, while warmup failures retain no measured-repeat result. A fresh subprocess is used per configuration. Layer \(\ell\) draws latent values on CPU in float64 with seed \(s+2\ell\) and boundary logits with seed \(s+2\ell+1\), then casts explicitly. Directed backends share these underlying logits; symmetric latents have different shapes and define a different operator family. All composition heights are exactly represented as one. Repeats do not update parameters.

The adversarial engineering product is \(N\in\{17,25,49\}\), \(B\in\{1,4\}\), \(L\in\{1,4\}\), \(R=256\), three seeds 1701–1703, and the two GPU backends: 72 configurations **at each** \(\sigma\in\{0.5,1,3\}\). These runs have zero warmups and one repeat, so their timings are cold/single observations, not comparable warm steady-state estimates. The independent source targets in the legality suite are not identical to these batched composition inputs.

Formal optimization has 13 task configurations per device: map fitting at \((N,R)=(17,256),(25,256)\) with Adam and LBFGS, plus Adam at \((25,512)\); medical-phantom image fitting with the same five configurations; and checkerboard, smooth blobs, and textured images with Adam at \((25,256)\). Each is a single-instance, single-layer run (\(B=L=1\)). CPU requests three backends (`direct`, `directed_iterative`, `symmetric`); GPU requests the latter two. Therefore 26 JSON receipts contain \(13\times3+13\times2=65\) backend rows. The legacy `reference` is an engineering baseline only, not an omitted formal optimization result.

All formal runs use float32, seed 20260922, \(\sigma=0.25\), neutral zero latent/side logits, and one shared prebuilt CPU-double target per receipt. Adam runs 40 updates at learning rate 0.03, with 41 evaluations including the final one. LBFGS runs five outer steps at learning rate 0.8, `max_iter=5` per outer step, history size 10, strong-Wolfe search, and zero gradient/change stopping tolerances; closure counts are data dependent. There is no encoder/CNN or training dataset. Every evaluation invokes the actual decoder solve, dense interpolation, objective, and independent audit; every update/closure requiring gradients invokes the implicit backward.

The target is detached, checked for control/dense consistency, and left unchanged. Within one receipt, the same object is supplied to every requested backend. Across receipts, shape/sum/L2 summaries agree for matching task/seed/mesh/resolution/strength; this is a reproducibility check, not a cryptographic identity or elementwise equality proof. Object IDs must not be compared across processes. Shared-target setup is recorded once, separately from training, rather than regenerated for each backend.

### 4.2 Actual environment and provenance

| Evidence | Host/device | Python; Torch; NumPy; SciPy | Recorded OS / commit |
|---|---|---|---|
| Legality | `element`, CPU | 3.12.7; 2.5.1+cu124; 1.26.4; 1.13.1 | OS not recorded in this receipt; `cc60609d64dd94fb36bb99ef2a0d01923865e597`, `git_dirty=false` |
| Normal CPU engineering | `element`, x86_64 CPU | 3.12.7; 2.5.1+cu124; 1.26.4; 1.13.1 | Linux 6.8.0-44, glibc 2.39; `02019feb41dea28a0bf3328ecd2ff07c0b8cfdb0` |
| Normal GPU engineering | `ai`, NVIDIA RTX A6000 | 3.11.7; 2.4.0+cu121; 1.26.4; 1.11.4 | Linux 5.15.0-88, glibc 2.31; `02019feb41dea28a0bf3328ecd2ff07c0b8cfdb0` |
| Adversarial GPU and formal GPU | `ai`, NVIDIA RTX A6000 | 3.11.7; 2.4.0+cu121; 1.26.4; 1.11.4 | Linux 5.15.0-88, glibc 2.31; `e31586cc9e7012b7d16bb52dce9fad3061f3aa82` |
| Formal CPU | `turing`, CPU | 3.11.7; 2.8.0+cu128; 1.26.4; 1.11.4 | Linux 6.8.0-138, glibc 2.39; `e31586cc9e7012b7d16bb52dce9fad3061f3aa82` |

All recorded Torch thread counts are one; normal/adversarial engineering also records `OMP_NUM_THREADS=MKL_NUM_THREADS=1`. Formal GPU records CUDA build/runtime 12.1, `CUDA_VISIBLE_DEVICES=2`, and 51,033,931,776 total device bytes. CPU model names, total host RAM, GPU driver version, and cross-run machine load are **not established by these receipts**. The engineering environment records only `x86_64` as its CPU description. Formal CPU and CPU engineering are different hosts/software stacks. Except for the legality receipt's dirty flag, Git HEAD alone does not certify a clean worktree.

The CPU direct path uses float64 sparse SuperLU factorization with float32 returned coordinates; the legacy CPU reference also normalizes probabilities in float64 and factors \(A\) and \(A^T\) separately. GPU directed uses float32 BiCGStab (`rtol=1e-5`, `atol=0`, primary limit 500) with the CUDA-float32-only stationary fallback of up to 10,000 iterations. Symmetric uses unpreconditioned float32 CG (`rtol=1e-5`, `atol=0`, limit 1000) with candidate/periodic residual replacement. In the original `02019fe` matrix its internal and public thresholds coincide. The later committed `e688abc` correction uses \(0.99\) times the requested threshold for CUDA-float32 recurrence/candidate stopping only, while retaining the unchanged requested threshold for a fresh final residual acceptance check; CPU and float64 are unchanged. CPU directed has **no** stationary fallback. These arithmetic and algorithm differences must accompany any CPU/GPU comparison.

### 4.3 Solve and timing semantics

A global solve means one sample's primal or adjoint system with both coordinate RHS columns, not one solve per x/y coordinate. For \(T\) completed engineering repeats, measured primal and adjoint counts are each \(TBL\); warmup counts are reported separately. Completed module hooks are counted, and factorization counts multiply by the independently tested backend contract: two per reference sample-forward, one per direct sample-forward, zero matrix-free. A fallback is another attempt at the same logical system, not another counted global solve; its time remains charged, while its reported iteration count excludes the failed primary attempt.

Adam's 40 updates therefore have 41 primals and 40 adjoints, or 81 global solves, plus one separately reported target setup solve per receipt. LBFGS counts actual closures, not outer steps. At threshold observation \(j\), the formal traces count \(2(j+1)-1\) solves because the current evaluation's backward has not yet run.

Engineering forward time includes boundary construction, all control solves and their internal screens, and coordinate composition. Loss and backward are separately timed; end-to-end is their sum. It excludes setup, table preparation, the later independent audit, and the separate dense-only replay. Replay times re-evaluate coordinates with detached control maps and autograd enabled, without a solver; they are neither an additive decomposition nor a subtraction-based estimate of solve time. Throughput is \(B/\operatorname{mean}_t T_{{\rm eng},t}\), not multiplied by layers. CUDA wall-clock intervals synchronize completed device work.

Formal `forward_seconds` includes the decoder, dense map, and loss/image warp. `backward_seconds` includes implicit adjoint and parameter gradients. `audit_seconds` includes host transfer and independent control-map metrics. Adam step time is explicit; LBFGS optimizer-only overhead subtracts closure forward/backward/audit totals from its enclosing step interval and clamps the remainder at zero. `wall_seconds` encloses the optimization loop; `wrapper_wall_seconds` additionally encloses per-backend setup/preparation and final reporting. Observation wall time is recorded after the current audit. Nested intervals must not be added together. The experiments did not repeat each formal optimization for timing uncertainty.

### 4.4 Memory semantics

RAM fields are Linux `/proc/self/status` snapshots: `VmRSS` is approximate resident memory at the observation and `VmHWM` is the **process-lifetime** resident high-water mark. They include Python/libraries, setup, and earlier allocations; they are not incremental solver memory. Fresh engineering subprocesses prevent earlier configurations sharing an HWM. Formal receipts run several backends sequentially in one process, so a later backend's HWM can include an earlier backend's peak and is not isolated attribution.

GPU fields are Torch allocator `allocated` and `reserved` baselines/peaks, not whole-device memory usage. Engineering resets peak statistics after warmup and measures forward/backward, including the resident setup baseline, before independent audit and dense replay. Formal GPU resets peaks before each backend run and includes that backend's setup; shared CPU target generation is excluded. Reserved memory can exceed live tensor allocation. All displayed MiB values use \(2^{20}\) bytes. Missing GPU values on CPU are not zero-cost GPU measurements.

## 5. Ground-truth legality: 93/93, with small margins

The [legality JSON](raw_results/route1_validity_cc60609.json) and [CSV](raw_results/route1_validity_cc60609.csv) contain 40/40 analytic, 45/45 positive-Tutte, and 8/8 image cases accepted in 10.367025 s total on `element`. All geometric cases have zero nonpositive faces and pass the independent numerical audit; the random cases also verify positive represented probabilities. This is **93/93 ground-truth legality**, not 93 successful fits or a solver-reliability sweep.

Across analytic samples, the minimum area ratio is 0.563854701565084 and the minimum target face area is \(1.2236430155492252\times10^{-4}\), both for local compression at \(N=49\). Maximum sampled \(|\mu|\) is 0.28230759980783493. Only 10/40 analytic mesh/map pairs are in-range for unit-square sampling (the two admissible families at five mesh sizes).

All 45 random targets are in-range, but their worst margins are near-degenerate:

| Quantity | Minimum/maximum over the 45 random target cases | Case where specified |
|---|---:|---|
| Minimum target face area | \(4.951017696528849\times10^{-13}\) | \(N=33,s=1703,\sigma=3\) |
| Minimum area ratio | \(8.314899655844294\times10^{-10}\) | \(N=25,s=1701,\sigma=3\) |
| Maximum \(|\mu|\) | 0.999999939043383 | \(N=49,s=1703,\sigma=3\) |
| Minimum boundary gap | \(2.677056021305191\times10^{-9}\) | Across cases |
| Minimum supported probability | \(5.107571539000306\times10^{-9}\) | Across cases |
| Maximum row-sum rounding error | \(4.440892098500626\times10^{-16}\) | Across cases |

For the maximum-\(|\mu|\) case, median and 95th percentile are 0.9442249103028622 and 0.9994447244403218. Thus positive orientation does not imply a useful uniform distortion margin or float32 robustness. CPU-double acceptance cannot be transferred to the different float32 adversarial composition population.

Each image passes determinism, finiteness, exact normalized endpoint range, identity sampling, a positive-x one-pixel sampling shift on a cropped in-range domain, and a separate NumPy four-corner bilinear oracle. The legality image pairs use the analytic boundary-sliding map, independently checked as a strictly ordered product map before the oracle comparison. This differs from the random-Tutte target used for formal fitting. The largest identity/shift error is \(3.0517112463712692\times10^{-5}\); the largest pair-oracle error is \(1.5616416931152344\times10^{-5}\). Acceptance uses \(8\epsilon_{32}R\), not exact pixel equality. These errors validate the sampling convention within tolerance, not inverse consistency or image registration performance.

## 6. Engineering results and stress failures

### 6.1 Normal matrix

Sources are [CPU JSONL](raw_results/route1_engineering_cpu_02019fe.jsonl), [CPU CSV](raw_results/route1_engineering_cpu_02019fe.csv), [GPU JSONL](raw_results/route1_engineering_gpu_02019fe.jsonl), and [GPU CSV](raw_results/route1_engineering_gpu_02019fe.csv). Counts below are configurations, not repeat or layer/sample counts.

| Backend/device | Completed/requested | At 256 | At 512 | By \(N=11,17,25,33,49\), each /18 | Maximum independently replayed primal relative residual, completed only |
|---|---:|---:|---:|---|---:|
| reference / CPU | 90/90 | 45/45 | 45/45 | 18,18,18,18,18 | \(2.3177243963\times10^{-7}\) |
| direct / CPU | 90/90 | 45/45 | 45/45 | 18,18,18,18,18 | \(2.2907115114\times10^{-7}\) |
| directed_iterative / GPU | 72/90 | 36/45 | 36/45 | 18,18,18,18,0 | \(1.0001160366\times10^{-5}\) |
| symmetric / GPU | 88/90 | 43/45 | 45/45 | 18,18,18,18,16 | \(1.0013179914\times10^{-5}\) |

Thus CPU reference/direct jointly complete 180/180. All 18 normal directed GPU \(N=49\) configurations fail during warmup: the stationary fallback exhausts 10,000 iterations following primary nonconvergence/breakdown. They have no accepted measured forward/backward result. The two symmetric failures are \((N,B,L,R)=(49,4,4,256)\) and \((49,8,4,256)\), both during warmup. Both report final CG nonconvergence after 161 iterations with a displayed residual `1.000e-05`; display rounding does not make the actual comparison pass. The corresponding 512 cases complete, but their dense-query loss and adjoint workload differ; these receipts alone do not isolate the cause of that contrast. Success at larger query resolution is not monotone solver reliability.

All completed configurations have finite parameter gradients and passed independent per-layer control-map audits. There are 910 audited layer/sample maps per CPU backend, 728 for GPU directed, and 862 for GPU symmetric; these are final-repeat diagnostics, not independent randomized replicates. Their respective minimum area ratios are 0.46607198577421827, 0.4660715657127926, 0.48068437431925304, and 0.505981571749544. The small replay residual excess over \(10^{-5}\) for some GPU results is disclosed mixed-precision evaluation, not a tightened independent acceptance guarantee. The engineering experiment does not measure VJP finite-difference error or fitted map/\(\mu\) accuracy.

Representative full forward/surrogate/backward times in seconds are shown below. Each cell is the median of **two** timed repeats after one warmup; `failure` is not a missing zero. These coordinates were selected to show small, intermediate, and largest workloads, not to report a fitted scaling law.

| \((N,B,L,R)\) | CPU reference | CPU direct | GPU directed | GPU symmetric |
|---|---:|---:|---:|---:|
| (17,1,1,256) | 0.046973 | 0.041115 | 0.253139 | 0.119309 |
| (25,4,2,256) | 0.800899 | 0.691557 | 2.148924 | 1.081934 |
| (49,8,4,256) | 11.775958 | 10.169247 | failure | failure |
| (49,8,4,512) | 13.296564 | 11.617937 | failure | 15.208104 |

Across the 90 matched CPU configurations, the median of the paired ratios \(T_{\rm reference}/T_{\rm direct}\) is 1.1274383, with range [0.9671611, 1.2350139]; direct is faster in 86/90 pairs. This supports a bounded engineering benefit from the improved direct path, not a guaranteed speedup or a twofold reduction in total runtime from halving factorization calls. Geometry checks, interpolation, and backward remain charged.

The largest completed symmetric GPU case has 780.903809 MiB peak allocated and 1054 MiB peak reserved, and dense-only replay takes 0.010952 s versus 15.208104 s for the whole surrogate forward/backward. Replay is a separate forward-only measurement; it cannot simply be subtracted to isolate the solver. Across completed normal GPU cases, directed's maximum peak allocated is 778.720215 MiB. The CPU process-HWM maxima are 1660.695313 MiB for reference and 1735.25 MiB for direct, including process setup and warmup.

These tables do **not** establish that a GPU kernel is intrinsically slower or faster than a CPU kernel. Hosts, software, solver families, arithmetic, synchronization, and CPU geometry audits differ. In particular, reference/direct do not have GPU implementations here, and symmetric does not solve the same parameterized problem as directed. Two repeats and one random seed are insufficient for a stable hardware performance claim.

### 6.2 Adversarial scale sweep

The raw pairs are [0.5 JSONL](raw_results/route1_adversarial_strength_0p5_e31586c.jsonl)/[CSV](raw_results/route1_adversarial_strength_0p5_e31586c.csv), [1.0 JSONL](raw_results/route1_adversarial_strength_1p0_e31586c.jsonl)/[CSV](raw_results/route1_adversarial_strength_1p0_e31586c.csv), and [3.0 JSONL](raw_results/route1_adversarial_strength_3p0_e31586c.jsonl)/[CSV](raw_results/route1_adversarial_strength_3p0_e31586c.csv). Every row below has 36 requested configurations; the per-\(N\) entries have denominator 12. Failure categories are the **first observed exception**, so a run failing early is not tested for every later possible failure.

| \(\sigma\) / GPU backend | Complete /36 | Complete at \(N=17,25,49\), each /12 | Recorded failures | Smallest area ratio among accepted layer/sample maps |
|---|---:|---|---|---:|
| 0.5 directed | 25 | 12,12,1 | 11 stationary-budget failures | 0.08091057869 |
| 0.5 symmetric | 36 | 12,12,12 | None | 0.08833678851 |
| 1.0 directed | 26 | 12,12,2 | 10 stationary-budget failures | 0.002516484610 |
| 1.0 symmetric | 32 | 12,12,8 | 4 CG nonconvergences | 0.005623327379 |
| 3.0 directed | 3 | 3,0,0 | 22 positive-face screen failures; 6 boundary-order failures; 5 stationary-budget failures | \(1.037551556\times10^{-8}\) |
| 3.0 symmetric | 23 | 10,12,1 | 10 boundary-order failures; 3 CG nonconvergences | \(8.993083611\times10^{-9}\) |

All exceptions occur in the measurement stage (there is no warmup). Every completed case passes the independent layer audits. A positive-face-screen failure is not necessarily a proven fold: the predicate also rejects small/degenerate margins. A boundary-order failure demonstrates that positive theoretical softmax segments can round to collapsed cumulative vertices. These are explicit rejections, not repaired maps or successes hidden by padding.

The sweep totals 216 configurations, with 145 completions and 71 failures. Success fractions are descriptive for this fixed grid of seeds and settings, not estimates over a specified random population. Increasing strength is not a monotone measure of numerical difficulty; for example directed completes 25 cases at 0.5 and 26 at 1.0. At strength 3, even accepted maps have extremely small area ratios. No accuracy or topology claim is assigned to a failed output, and the accepted-only timing population is strongly selection biased.

## 7. Actual instance optimization

### 7.1 Completed runs, thresholds, and missing accuracy

The validated [65-row formal summary](raw_results/route1_instance_formal_summary.csv) is derived from the 26 JSON receipts linked in the next table. There are **56 completed runs and 9 failures**, all failures being CPU `directed_iterative`. None is dropped. The 56 complete runs all have certified observed control maps; across their recorded traces the minimum area ratio is 0.05902046180244738 and the maximum flip count is zero. Failed receipts do not preserve a completed trace or final fitted-map accuracy, so neither “all iterates valid” nor final errors are inferred for them.

Below, D denotes CPU direct; I-C CPU directed iterative; S-C CPU symmetric; I-G GPU directed iterative; S-G GPU symmetric. Cells give **final objective**, rounded to six significant digits; `fail` is a recorded solver failure, and a dagger marks a completed run that never reached its threshold. The full-precision values, best values, errors, and timings remain in the linked receipts/CSV. A final objective is not substituted for the best value when computing threshold hits.

| Task, \(N,R\), optimizer; raw receipts | D | I-C | S-C | I-G | S-G |
|---|---:|---:|---:|---:|---:|
| Map 17,256 Adam — [CPU](raw_results/formal_map_adam_n17_256_cpu.json), [GPU](raw_results/formal_map_adam_n17_256_gpu.json) | 2.43646e-5 | 2.43756e-5 | 1.79188e-5 | 2.42933e-5 | 1.79188e-5 |
| Map 25,256 Adam — [CPU](raw_results/formal_map_adam_n25_256_cpu.json), [GPU](raw_results/formal_map_adam_n25_256_gpu.json) | 4.52957e-5 | fail | 2.40938e-5 | 4.56739e-5 | 2.40939e-5 |
| Map 25,512 Adam — [CPU](raw_results/formal_map_adam_n25_512_cpu.json), [GPU](raw_results/formal_map_adam_n25_512_gpu.json) | 4.52254e-5 | fail | 2.39790e-5 | 4.57757e-5 | 2.39791e-5 |
| Map 17,256 LBFGS — [CPU](raw_results/formal_map_lbfgs_n17_256_cpu.json), [GPU](raw_results/formal_map_lbfgs_n17_256_gpu.json) | 3.42015e-5 | fail | 6.96887e-5 | 4.11094e-5 | 7.01460e-5 |
| Map 25,256 LBFGS — [CPU](raw_results/formal_map_lbfgs_n25_256_cpu.json), [GPU](raw_results/formal_map_lbfgs_n25_256_gpu.json) | 4.26181e-5 | fail | 2.01540e-4 † | 3.86782e-5 | 2.01530e-4 † |
| Medical 17,256 Adam — [CPU](raw_results/formal_image_medical_adam_n17_256_cpu.json), [GPU](raw_results/formal_image_medical_adam_n17_256_gpu.json) | 4.82736e-4 | fail | 6.07423e-5 | 3.35057e-4 | 6.05983e-5 |
| Medical 25,256 Adam — [CPU](raw_results/formal_image_medical_adam_n25_256_cpu.json), [GPU](raw_results/formal_image_medical_adam_n25_256_gpu.json) | 5.96766e-4 | 9.72417e-4 | 1.35882e-4 | 5.24555e-4 | 1.35556e-4 |
| Medical 25,512 Adam — [CPU](raw_results/formal_image_medical_adam_n25_512_cpu.json), [GPU](raw_results/formal_image_medical_adam_n25_512_gpu.json) | 7.48710e-4 | fail | 1.63301e-4 | 8.28770e-4 | 1.63302e-4 |
| Medical 17,256 LBFGS — [CPU](raw_results/formal_image_medical_lbfgs_n17_256_cpu.json), [GPU](raw_results/formal_image_medical_lbfgs_n17_256_gpu.json) | 2.17905e-4 | 2.32439e-4 | 1.37849e-4 | 1.69130e-4 | 1.41191e-4 |
| Medical 25,256 LBFGS — [CPU](raw_results/formal_image_medical_lbfgs_n25_256_cpu.json), [GPU](raw_results/formal_image_medical_lbfgs_n25_256_gpu.json) | 3.36413e-4 | fail | 2.10558e-4 | 3.31359e-4 | 2.73963e-4 |
| Checkerboard 25,256 Adam — [CPU](raw_results/formal_image_checkerboard_adam_n25_256_cpu.json), [GPU](raw_results/formal_image_checkerboard_adam_n25_256_gpu.json) | 0.207836 † | fail | 0.0113694 † | 0.108272 † | 0.0112415 † |
| Blobs 25,256 Adam — [CPU](raw_results/formal_image_smooth_blobs_adam_n25_256_cpu.json), [GPU](raw_results/formal_image_smooth_blobs_adam_n25_256_gpu.json) | 2.90052e-4 | fail | 5.91643e-5 | 2.91892e-4 | 5.91644e-5 |
| Textured 25,256 Adam — [CPU](raw_results/formal_image_textured_adam_n25_256_cpu.json), [GPU](raw_results/formal_image_textured_adam_n25_256_gpu.json) | 7.26211e-4 | 6.53576e-4 | 2.45559e-4 | 7.39163e-4 | 2.45524e-4 |

The CPU-directed failures comprise five shadow-residual breakdowns (checkerboard, medical Adam 17/256, medical Adam 25/512, medical LBFGS 25/256, blobs), three alpha-denominator breakdowns (map Adam 25/256 and map LBFGS at 17/256 and 25/256), and one 500-iteration nonconvergence (map Adam 25/512). The CUDA fallback is not available on CPU. The observed contrast therefore cannot be explained as GPU hardware alone curing those cases, nor does GPU 13/13 establish reliability beyond these small formal meshes.

Threshold counts and observed cost-to-threshold are:

| Backend/device | Complete/requested | Threshold hits/requested | Map hits /5; image hits /8 | Median evaluations / global solves / seconds to threshold, hits only |
|---|---:|---:|---|---|
| direct / CPU | 13/13 | 12/13 | 5; 7 | 21.5 / 42 / 3.064679 |
| directed_iterative / CPU | 4/13 | 4/13 | 1; 3 | 21.5 / 42 / 5.267443 |
| symmetric / CPU | 13/13 | 11/13 | 4; 7 | 17 / 33 / 3.364961 |
| directed_iterative / GPU | 13/13 | 12/13 | 5; 7 | 20.5 / 40 / 9.050356 |
| symmetric / GPU | 13/13 | 11/13 | 4; 7 | 17 / 33 / 3.513855 |

There are 50 threshold hits among 65 requests (50/56 completed). Six completed runs miss: the four completed checkerboard runs and the two symmetric \(N=25\) map-LBFGS runs. The latter finish near \(2.0153\times10^{-4}\), above \(10^{-4}\), while all checkerboard best values remain above \(10^{-3}\). This is an optimizer/representation/observation outcome within the fixed budget, not proof of an expressivity lower bound. The hit-only medians pool different tasks and omit failures/misses; they describe the recorded observations and must not be used as uncensored head-to-head speed rankings.

### 7.2 Geometric accuracy and real optimization cost

To separate objective values from geometry, the following medians use **completed rows only**, separately for map and image tasks. `Time` is optimization-loop wall seconds, not the wrapper, target setup, or engineering surrogate. Each accuracy number is the median of per-run final control-map/face errors defined in Section 3, not a pooled per-vertex average. The small CPU-directed survivor sets are explicitly shown.

| Task / backend-device | Complete/requested | Median time (s) | Median \(e_{\rm map}\) | Median \(e_\mu\) |
|---|---:|---:|---:|---:|
| Map / direct CPU | 5/5 | 5.225936 | 0.0100562 | 0.138476 |
| Map / directed CPU | 1/5 | 6.622615 | 0.00860394 | 0.113790 |
| Map / symmetric CPU | 5/5 | 6.784867 | 0.00785884 | 0.123760 |
| Map / directed GPU | 5/5 | 14.872292 | 0.0103069 | 0.138497 |
| Map / symmetric GPU | 5/5 | 7.446955 | 0.00785886 | 0.123760 |
| Image / direct CPU | 8/8 | 6.907738 | 0.0229523 | 0.159596 |
| Image / directed CPU | 3/8 | 12.164036 | 0.0155300 | 0.135683 |
| Image / symmetric CPU | 8/8 | 9.276718 | 0.0199919 | 0.149882 |
| Image / directed GPU | 8/8 | 19.640318 | 0.0228364 | 0.159632 |
| Image / symmetric GPU | 8/8 | 9.717181 | 0.0201129 | 0.149969 |

Image fitting does not identify the generating map uniquely. For example, CPU direct's checkerboard result remains topologically accepted but has control-map RMSE 0.29242 and \(\mu\)-RMSE 0.67002; its maximum map error is 0.5707466051 and maximum coefficient error is 1.1176646802. A coefficient **difference** exceeding one is possible even if each map individually has \(|\mu|<1\). Symmetric GPU's checkerboard map RMSE is much smaller (0.0071044), yet its image objective 0.0112415 still misses the prescribed threshold. Neither example justifies replacing the image criterion with a topology or map metric after seeing the result.

There is actual 512-squared fitting evidence, but only for \(N=25\) Adam map/medical cases, not 512-squared LBFGS or all four image families. For map 25/512, CPU direct, GPU directed, and GPU symmetric take 8.181004, 21.880101, and 9.933935 s respectively, while their final objectives are approximately \(4.5225\times10^{-5}\), \(4.5776\times10^{-5}\), and \(2.3979\times10^{-5}\). For medical 25/512, the corresponding wall times are 8.925026, 19.312075, and 9.423194 s, with final objectives in the table above. CPU directed fails both 512 cases; CPU symmetric completes both. All successful 40-update formal Adam runs have 81 global solves; successful LBFGS totals are 61, 63, 67, or 69 in these receipts, rather than being determined by the five outer steps.

Maximum recorded formal process HWM among completed rows is 706.589844 MiB for CPU direct, 673.953125 MiB for CPU directed survivors, 725.335938 MiB for CPU symmetric, 1234.324219 MiB for GPU directed, and 1272.3125 MiB for GPU symmetric. These formal HWM values are process-order confounded as explained above. Maximum GPU peak allocated is 60.370605 MiB for directed and 60.334473 MiB for symmetric; the maximum reserved peak across completed formal GPU runs is 82 MiB. These are \(B=L=1\) instance workloads, not the 8-by-4 multilayer engineering peak.

The table does not establish a fair hardware-only speedup. CPU direct versus GPU iterative differs in host, Torch/SciPy version, arithmetic, matrix representation, stopping behavior, and synchronization. Symmetric can optimize differently because it changes the parameterized operator family. In addition, a small error perturbation can alter an image optimizer's trajectory or an LBFGS line search, so near-equal initial maps do not require identical final objectives. Independent VJP accuracy is investigated separately in the methods/worklog; it is not measured by this application receipt set.

## 8. Separately identified extension experiments

These later observations are kept outside the original 65-row formal and 360-configuration normal engineering denominators. They do not overwrite earlier failures or threshold misses.

### 8.1 Checkerboard: a 200-update GPU extension

The [extended checkerboard receipt](raw_results/extended_image_checkerboard_adam200_n25_256_gpu.json) uses the same \(N=25,R=256\), seed 20260922, target strength 0.25, Adam learning rate 0.03, threshold \(10^{-3}\), float32 A6000 environment, and `e31586c` commit as its 40-update formal counterpart. All target shapes/sum/L2 summaries match exactly. It is a fresh run from neutral parameters for 200 updates, not a resumed optimizer checkpoint; each backend records 201 primals, 200 adjoints, and 401 global solves.

| Backend | Final / best image objective | First threshold evaluation / solves / time (s) | Final control \(e_{\rm map}\) / \(e_\mu\) | Loop wall time (s) |
|---|---|---|---|---:|
| directed_iterative | 0.06744971871 / 0.06661999971 | Not reached | 0.12083009 / 0.43103417 | 123.107438 |
| symmetric | 0.000303242967 / 0.000303242967 | 100 / 199 / 26.668467 | 0.005587178 / 0.10879884 | 53.213322 |

Both complete and certify every recorded control map. The symmetric result shows that its earlier checkerboard miss was not a demonstrated inability to reach the chosen threshold: it does reach it under this longer run. Directed does not reach the threshold within 200 updates; topology remains valid despite substantial geometric error. This is one budget extension, not a general convergence comparison or a proof of failure under all longer budgets.

The first 41 objective observations are not bitwise replayed across processes. At update 40 the extended/formal directed objectives are 0.1205154508/0.1082715765; symmetric objectives are 0.01133211143/0.01124150306. The maximum absolute discrepancy across those 41 observations is 0.01292727143 and 0.00065233931 respectively. This discloses numerical-path sensitivity under the CUDA computation; the receipt does not isolate every cause or establish deterministic trajectories. Matching target summaries does not imply matching floating-point solver/optimizer paths.

### 8.2 1024-squared coordinate engineering, not image optimization

The [1024 JSONL](raw_results/route1_engineering_symmetric_1024_e31586c.jsonl) and [CSV](raw_results/route1_engineering_symmetric_1024_e31586c.csv) contain exactly \(N\in\{25,49\}\) by \(L\in\{1,4\}\), with \(B=1\), symmetric float32 CUDA, seed 1701, strength 0.15, one warmup and two repeats. Environment is `ai`/A6000 at `e31586c`, matching the earlier GPU stack. All four complete, with every layer's control map independently certified. Here \(Q=1,048,576\); these are coordinate-composition/squared-surrogate/backward workloads, not million-pixel registration optimization.

| \(N,L\) | Mean end-to-end (s) | GPU peak allocated / reserved (MiB) | Process HWM (MiB) |
|---|---:|---|---:|
| 25,1 | 0.251740 | 212.919922 / 232 | 1101.660156 |
| 25,4 | 0.845966 | 419.437012 / 506 | 1309.839844 |
| 49,1 | 0.730039 | 213.228027 / 232 | 1113.660156 |
| 49,4 | 2.784996 | 420.614746 / 508 | 1317.683594 |

The means equal the two-sample medians here. They are four selected configurations, not a full 1024 matrix; none establishes original-mesh P1 validity for the composed dense samples. Memory units and measurement windows remain those of Section 4.

### 8.3 Explicit float64 directed reliability: independently recomputed extension

The existing float64 path was subsequently exercised with explicit float64 inputs, without a production solver edit, hidden promotion, or tolerance relaxation. [Reliability JSONL](raw_results/route1_directed_float64_reliability_gpu_e31586c.jsonl) and [CSV](raw_results/route1_directed_float64_reliability_gpu_e31586c.csv) record \(N=49\), \(B\in\{1,4,8\}\), \(L\in\{1,2,4\}\), seeds 1701/1702, \(R=256\), strength 0.15, one warmup/two repeats: **18/18** complete. The host/software/commit are `ai`/A6000/`e31586c`; Torch and OMP threads are one, while `MKL_NUM_THREADS` is unrecorded/null in these receipts. All 182 final-repeat layer/sample maps are certified, and all gradients are finite. Independent primal replay residual is at most \(9.9541990198\times10^{-11}\); reported adjoint residual is at most \(9.9943800575\times10^{-11}\). Maximum primal/adjoint iteration counts are 121/124; all methods are BiCGStab with no fallback, using the float64 default tolerance \(10^{-10}\).

For \(B=L=1\), the two seeds' mean full-stack times are 1.429784–1.488336 s and peak allocated memory is 16.757324 MiB. For \(B=8,L=4\), they are 18.284466–20.762145 s and 311.116699 MiB. These are surrogate full-stack times, not optimization times. They include decoder boundary validation and each forward's strict positive-face screen, but exclude `post_measurement_audit_seconds` and hence the later independent global-injectivity metric audit. Float64 preserves the underlying CPU-generated double random numbers rather than rounding them to float32; it is not the identical represented float32 operator solved more carefully.

One additional [512-squared receipt](raw_results/route1_directed_float64_n49_b8_l4_512_gpu_e31586c.json) uses \(N=49,B=8,L=4,s=1701\) and the same strength, float64 settings, warmup, and repeats. It completes in mean end-to-end time 18.214490 s with 1212.616699 MiB peak allocated GPU memory. Maximum reported primal/adjoint relative residuals are \(9.9336525148\times10^{-11}\) and \(9.9959910707\times10^{-11}\); all 32 control maps are independently certified. This is one supplemental case, not a 512-squared reliability sweep, and is additional to the 18 configurations above.

A separate [CPU-direct/GPU-double comparison](raw_results/route1_directed_float64_direct_vjp_e31586c.jsonl) uses \(N=49,B=4,L=1,R=256\), strength 0.15, seeds 1701/1702/1703, identical float64 inputs, and the squared dense-coordinate surrogate. For flattened CPU reference tensor \(a\) and GPU tensor \(b\), it reports \(\max|a-b|\) and \(\|a-b\|_2/\|a\|_2\). Across these three observations, maximum dense coordinate component error is \(3.4478526700\times10^{-9}\), row-logit VJP relative L2 error is at most \(4.5395653111\times10^{-9}\), and **boundary-logit** VJP relative L2 error is at most \(1.4586200406\times10^{-9}\). The last is not a direct test of arbitrary boundary-coordinate cotangents. These quantities compare flattened tensors componentwise/under the Euclidean norm; this is a reference VJP comparison, not an independent finite-difference experiment.

Those comparison timings are 1.897036–1.938931 s for CPU direct and 2.855915–3.189213 s for GPU double, including full coordinate forward/backward and recorded result transfers. The small comparison JSONL lacks its own full environment/commit fields, so its provenance relies on the accompanying investigation and filename rather than a self-contained environment receipt. These observations suggest a bounded explicit-precision reliability option, **not a speed winner**. Independent recomputation verified the 18/18 configuration count, all 182 final-repeat global audits, iteration/residual extrema, the 512-squared receipt, and the stated map/VJP discrepancies; it required only the timing-scope clarification above. They leave the normal float32 directed \(N=49\) result at **0/18**, and do not establish arbitrary-strength, other-device, or optimization-trajectory reliability.

The reliability investigation identifies cancellation in the float32 transpose operation \(x-\operatorname{scatter}(P_{II}^Tx)\), and observed variation when replaying the same rounded oracle solution, as evidence for the difficult adjoint acceptance scale. This is a measured arithmetic explanation on the tested cases, not a theorem that no float32 vector can satisfy the tolerance, that more iterations can never help, or that float64 universally fixes all positive-weight systems. Strong-logit boundary/area failures remain a separate issue.

### 8.4 Explicit-float64 directed conditioning beyond normal strength

The independently reviewed [conditioning JSONL](raw_results/route1_directed_float64_adversarial_gpu_e31586c.jsonl) and [CSV](raw_results/route1_directed_float64_adversarial_gpu_e31586c.csv) test whether the Section 8.3 option survives more extreme positive weights. They contain the exact Cartesian product

\[
N=49,\quad R=256,\quad B\in\{1,4\},\quad L\in\{1,4\},
\quad s\in\{1701,1702\},\quad \sigma\in\{0.5,1,3\},
\]

for 24 fresh float64 CUDA subprocesses. Each requests one warmup and one measured repeat; BiCGStab retains `rtol=1e-10`, zero absolute tolerance, a 500-iteration limit, and no float32 stationary fallback. Thus this is a conditioning/availability extension at one mesh size and one A6000, not a repeat of the full normal matrix.

| Strength | Completed / requested | Certified final-repeat layer/sample maps | Maximum independent primal replay | Maximum native adjoint residual | Maximum successful primal / adjoint iterations | First failure |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 8/8 | 50/50 | \(9.9606876652\times10^{-11}\) | \(9.9573462605\times10^{-11}\) | 136 / 150 | none |
| 1.0 | 7/8 | 34/34 | \(9.9723730948\times10^{-11}\) | \(9.8974122578\times10^{-11}\) | 299 / 301 | one 500-iteration primal nonconvergence |
| 3.0 | 0/8 | not reached | not available | not available | not applicable | four 500-iteration primal nonconvergences; four primal shadow-residual breakdowns |

All 15 completed configurations have finite gradients. Their 84/84 final layer/sample maps have zero flips and pass the independent audit. At strengths 0.5 and 1.0, respectively, the minimum signed areas are \(1.8730496639\times10^{-5}\) and \(8.5180972586\times10^{-7}\); the minimum target/source area ratios are 0.086310128515 and 0.0039251392168; and the minimum boundary-edge lengths in unit-square coordinates are 0.0029479889555 and 0.00044736723669. These shrinking margins are measured geometry, not condition numbers or a uniform distortion bound.

Every failure occurs in warmup, before any measured repeat. The strength-1, \(B=4,L=4,s=1702\) receipt completes two forward module calls and fails the third primal invocation: one RHS column remains at \(3.0269253665\times10^{-10}\) after 500 iterations. Every strength-3 receipt fails its first-layer primal invocation; the duplicated \(L=1\)/\(L=4\) diagnostics share that same seeded first layer and are distinct workloads, not independent random matrices. Failed rows have no measured timing, GPU allocator peak, complete map, gradient, or topology result; none is imputed.

Successful end-to-end values are synchronized coordinate forward, surrogate coordinate loss, and backward. Forward includes boundary validation and the strict positive-face screen; setup, warmup, later independent global-injectivity/metric audit, and dense-only replay are excluded. With one measured repeat, the observations support availability diagnosis rather than a stable performance comparison. The receipts self-report commit and software, while clean-worktree state, physical GPU ordinal, idle probes, and UTC bracket were operator-observed rather than embedded receipt fields. Independent review reproduced all 24 identities, the 15/9 split, failure attribution, 84 audits, residual/iteration extrema, solve counts, timing/memory table, and JSONL-to-CSV projection. The result directly falsifies any interpretation that explicit float64 is a universal remedy for arbitrary positive-weight contrast.

### 8.5 Symmetric CUDA threshold-boundary mechanism and clean committed rerun

The two original normal-matrix failures in Section 6.1 were not silently retried into the old artifact. A separate investigation repeatedly evaluated the represented operator \(Kx\) for the exact \(N=49,L=4,R=256,s=1701,\sigma=0.15\) cases. In the instrumented unmodified cohort, B=4 completed 4/5 and B=8 completed 7/10 fresh processes. All 114 reached CG invocations produced bitwise-varying results across 32 repeated CUDA `index_add_` evaluations. In the four decisive failures, a right-hand side had been deactivated at at most 0.9999971 times the requested threshold, but its first final evaluation rose to \(1.0000158\times10^{-5}\) through \(1.0001745\times10^{-5}\); repeated evaluations reached \(1.0009400\times10^{-5}\). This localizes those failures to threshold-boundary reevaluation variability on the tested operator; it is not a theorem that every CUDA scatter is nondeterministic or that CG otherwise converges.

Before confirmation, the implementation fixed an internal factor 0.99. If \(\tau_{\rm req}\) is the requested absolute-plus-relative residual threshold, CUDA float32 continues iterations until the internal tests reach

\[
\tau_{\rm int}=0.99\,\tau_{\rm req},
\]

but return still requires a fresh represented true residual no larger than \(\tau_{\rm req}\). CPU and float64 use \(\tau_{\rm int}=\tau_{\rm req}\). The instrumented candidate completed B=4 5/5 and B=8 10/10 (120/120 reached CG invocations); its maximum first-final residual was \(9.9045965\times10^{-6}\), and the maximum over 32 immediate rechecks was \(9.9082818\times10^{-6}\). The historical diagnostic [JSONL](raw_results/route1_symmetric_gpu6_reliability.jsonl) and [CSV](raw_results/route1_symmetric_gpu6_reliability.csv) preserve every original/candidate failure, separate exact forward and adjoint residual fields from instrumented all-solve fields, use solve-weighted iteration means, and leave unmeasured bitwise counts null. Their historical patched receipts lacked dirty-state metadata and are mechanism evidence, not the production provenance claim.

Production provenance comes from committed code `e688abc` in the clean [postfix full-matrix JSONL](raw_results/route1_engineering_symmetric_postfix_e688abc.jsonl) and [CSV](raw_results/route1_engineering_symmetric_postfix_e688abc.csv). Every row records the same before/after commit, `git_dirty=false`, physical GPU6, fresh-process identity, and UTC bracket. The exact original 90-cell product completes **90/90**, including both formerly failing \(N=49,L=4,R=256\) cells. All 910/910 final-repeat layer/sample maps are certified with zero flips and finite gradients. The extrema are:

| Quantity | Clean `e688abc` full matrix |
|---|---:|
| Native primal / adjoint relative residual | \(9.8962736956\times10^{-6}\) / \(9.9085782495\times10^{-6}\) |
| Independent CPU-double primal replay | \(9.9016578909\times10^{-6}\) |
| Primal / adjoint iterations | 144 / 162 |
| Minimum signed area / area ratio / boundary-edge length | \(1.1218366129\times10^{-4}\) / 0.50598157175 / 0.01216763258 |
| Maximum GPU allocated / reserved | 780.903809 / 1054 MiB |
| Maximum process-lifetime HWM | 1156.339844 MiB |

The formerly failing B=4 and B=8 256-squared cells have two-repeat mean end-to-end times 8.077845 s and 15.413787 s. The largest \(N=49,B=8,L=4,R=512\) cell completes in 16.088929 s with the displayed 780.903809 MiB allocation. These sequential observations are not randomized causal speed comparisons with the old run.

The separate clean [repeated JSONL](raw_results/route1_symmetric_gpu6_reliability_postfix_e688abc.jsonl) and [CSV](raw_results/route1_symmetric_gpu6_reliability_postfix_e688abc.csv) contain five fresh processes for each formerly problematic batch size at \(N=49,L=4,R=256\): **10/10** complete. Their 240/240 maps are certified; native primal/adjoint maxima are \(9.8999753391\times10^{-6}\)/\(9.9021499409\times10^{-6}\), and CPU-double primal replay is at most \(9.9027081605\times10^{-6}\). B=4 process-mean end-to-end time has mean/range 8.225484/8.144267–8.448442 s; B=8 has 15.603689/15.369637–15.785957 s. These are repeated executions of one seeded operator family, not ten independent deformation samples.

Across the clean full and repeated cohorts, 1,150 final layer/sample maps pass the numerical audit. This establishes bounded normal-strength availability for the committed correction on one A6000 and preserves the public residual criterion. It does not prove universal CG convergence, deterministic CUDA reductions, VJP accuracy from residual alone, or reliability at the stronger logits that Section 6.2 already shows can fail.

## 9. Reproducibility checks and pending work

The executable protocols are [legality](../../experiments/phase5/route1_validity_suite.py), [engineering measurement](../../experiments/phase5/route1_engineering_benchmark.py), [Cartesian sweep](../../experiments/phase5/route1_engineering_sweep.py), [instance runner](../../experiments/phase5/route1_instance_benchmark.py), and [formal summary validator](../../experiments/phase5/route1_instance_summary.py). The last validates schemas, solve arithmetic, threshold traces, target summaries, and duplicate experiment identities before flattening; it does not validate scientific truth or artifact provenance by itself. JSONL resume retains failures as recorded configurations rather than retrying them into success. No accidental duplicate identity in a resumable unique-configuration sweep is treated as a replicate; the postfix repeated cohort is explicitly indexed and reported as repeated execution of one seeded configuration. JSONL/CSV row counts are checked for every stated matrix/cohort.

This chapter's verification re-parses the named receipts, checks their Cartesian counts and unique configurations, recomputes completion/failure/threshold counts, validates the 26 formal receipts using the summary function, and verifies local links. Counts in the tables are exact; displayed continuous measurements are rounded explicitly and should be recomputed from full-precision JSON for downstream statistics. There are no confidence intervals: normal engineering has two repeated timings at one seed, adversarial engineering has one cold timing per configuration, the symmetric postfix reliability cohort repeats one identical seeded configuration per batch, and formal fitting has one target seed and one run per configuration.

The following follow-ups remain **pending in this evidence population**:

* Additional checkerboard controls beyond the recorded 200-update extension: alternative objectives, initializations, seeds, or budgets have no result assigned here. The original 40-update misses remain in their original matrix.
* Broader solver reliability: the normal float32 \(N=49\) directed failures, higher-strength directed/symmetric failures and near-degeneracy, and CPU-directed formal breakdowns remain. The two original normal symmetric warmup failures remain preserved as historical negative evidence, while the clean committed `e688abc` matrix and repeated cohort close only those exact normal-strength cases. Other seeds, strengths, devices, and tolerances remain untested or failure-prone.

No pending numerical outcome is guessed. The repository data scan found no ACDC/NIfTI/NRRD/MHA dataset (the current worktree file inventory likewise contains no `.nii`, `.nii.gz`, `.nrrd`, `.mha`, or `.mhd` data). No download was initiated for this optional real-data step, and no real-data result is claimed. Inverse consistency, real-image generalization, clinical validity, a learned encoder, and original-mesh P1 validity of composed resamples were not evaluated.

## 10. Bounded application conclusion

The evidence establishes a usable but bounded Route I prototype: independently checked legal targets; completed CPU reference/direct engineering across the specified 256/512 matrix; a clean 90/90 committed symmetric CUDA matrix; real single-instance fitting through solves, dense warps, and implicit gradients; and numerical topology checks on every recorded completed optimization evaluation. It also establishes concrete limits: normal float32 \(N=49\) directed GPU unavailability in the original matrix (0/18), higher-strength Krylov/boundary/near-degeneracy failures, nine CPU-directed optimization failures, checkerboard threshold misses, and a symmetric map-LBFGS miss at \(N=25\).

The separately recorded extensions show a symmetric checkerboard threshold hit with a longer budget, four completed 1024-squared symmetric coordinate workloads, bounded clean-commit closure of the symmetric residual-headroom correction, and an explicit-float64 directed option that succeeds 18/18 at normal strength but only 15/24 in the higher-strength extension. Consequently, these results support continued application and solver study under explicit dtype, mesh, tolerance, and budget restrictions. They do not establish uniform numerical availability, universally fast GPU execution, exact Beltrami recovery, a fixed distortion margin, image-to-map identifiability, or completion/closure of Route I before the route-level checker verdict.
