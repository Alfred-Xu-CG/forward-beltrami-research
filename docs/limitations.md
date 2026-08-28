# Limitations and unresolved defects

## Location and environment

- Development and validation now run directly in `D:\QC_optimization`; the
  earlier NTFS permission blocker has been resolved.
- The validated Anaconda environment contains two incompatible Intel OpenMP runtimes. Reproducible commands set `MKL_THREADING_LAYER=SEQUENTIAL` before process start. This makes the reference stable but prevents multithreaded MKL timing conclusions.

## Solver and gradients

- Numeric factorization is rebuilt for every changed μ. The implementation caches immutable mesh geometry conceptually but does not yet implement symbolic factorization reuse, CHOLMOD/PARDISO, GPU sparse solves, or iterative preconditioning.
- The PyTorch layer is CPU-only and transfers through NumPy. It is a ground-truth reference, not yet a high-throughput neural-network layer.
- No direct runtime/accuracy comparison with the SBN surrogate was run because compatible SBN source code and weights were not part of the supplied workspace.
- Weighted LSQC becomes ill-conditioned near `|μ|=1`; unweighted LSQC avoids that explicit singular weight but changes the least-squares metric.

## Injectivity

- The certificate uses floating-point predicates. It checks positive faces, boundary simplicity/orientation, one-ring branch index, and rectangle side order, but it is not an exact-arithmetic proof.
- A certificate is topological/numerical and does not guarantee a useful quality margin. The initial μ experiment passed while nearly collapsed; this was fixed experimentally by adding symmetric-Dirichlet regularization, but the distinction remains fundamental.
- No continuous collision detection is required for the final PL disk map, but a production optimizer may need exact predicates or interval-certified line search near degeneracy.

## Baselines and experiments

- `lim_style`, `slim_style`, and `amips_style` are controlled energy/barrier variants, not exact reproductions of the complete published solvers. Claims are restricted to these implementations.
- Hyperparameters were held fixed and not tuned per method. Forty iterations establish behavior, not converged best possible performance.
- I and S are analytic soft glyphs sampled at vertices. A paper should add quadrature samples, raster interpolation, landmark noise, multiple shapes, and train/test instances.
- The direct no-barrier method is allowed to fold; safeguarded variants use certified line search. This is intentional but means wall times include different acceptance work.
- Area sum preservation is necessary but insufficient for arbitrary prescribed factors. Checkerboard, spike, and random targets have unknown exact PL feasibility; only the manufactured target is a valid accuracy ground truth.

## Multi-chart scope

- The older cylinder experiment still has two pre-matched disk charts and is
  only a transition-formula baseline.  The new high-genus path instead uses
  every target triangle as a native PL chart and stores one global surface
  point `(face id, barycentric coordinates)`, so it does not need a fixed
  source-chart/target-chart assignment or a cut seam.
- The new method is a **refinement method**.  It requires a topology-valid
  initial homeomorphism `F0`.  Official experiments obtain `F0` and held-out
  truth from the published common-refinement map, then perturb it by a smooth
  diffeomorphic flow.  Automatically finding the correct homotopy class and a
  base homeomorphism between unrelated raw meshes remains unsolved here.
- The official landmark pairs are deterministic synthetic samples of the
  published correspondence, not independently annotated semantic landmarks.
  They validate recovery and flow mechanics, not landmark detection.
- Curvature-only matching is not identifiable on the heterogeneous published
  pairs: it reduces its own descriptor residual but worsens held-out dense
  correspondence in all 9 real trials.  More discriminative intrinsic
  descriptors, functional-map initialization, or learned features are needed.
- The numerical homeomorphism certificate samples face-chart orientation and
  checks CFL, forward/inverse replay, topology, and overlay projection.  Its
  `degree_one` result is an isotopy inference from the supplied base
  homeomorphism; it is not an exact preimage-counting degree proof.
- The reported CFL check is a shared numerical policy based on a
  first-percentile representative face scale.  It was conservative in a direct
  genus-5 P2-field Jacobian sampling check, but it is not a proof-grade bound
  for every possible sliver triangle.
- A sharp three-edge connected-sum genus-2 stress construction exposes a PL
  cone/seam failure.  The primary exact-genus-2 benchmark therefore uses a
  smoothed implicit double torus and the stress failure is retained rather
  than silently counted as a success.
- The official archive supplies genus-3 and genus-5 meshes but no genus-2 real
  pair.  Exact genus-2 evidence is manufactured, while real-data evidence is
  genuinely higher genus but not genus 2.
- Curvature smoothing, field construction, and the independent audit are CPU
  reference implementations.  The largest genus-5 trials take minutes, and
  the audit should be parallelized/cached before interactive use.

## Scientific claims not supported

This prototype does not establish that μ optimization is universally faster, more accurate, or uniquely bijective. It does not establish clinical or real-data utility. It supports exact facewise gradients, strong synthetic solver accuracy, a useful μ-LBS tradeoff on the tested tasks, and the necessity of transition-aware rather than coordinate-equality seams.
