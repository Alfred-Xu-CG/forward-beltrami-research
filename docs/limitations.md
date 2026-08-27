# Limitations and unresolved defects

## Location and environment

- The requested `D:\QC_optimization` directory is owned by `BUILTIN\Administrators`; the current user has read/execute only. Development therefore lives at `C:\Users\xuzhehao\Documents\Codex\QC_optimization`. The code is relocatable, but final copying to D remains blocked by the external NTFS ACL.
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

- The implemented experiment has two matched charts with identical triangulation and known overlap correspondences. It does not solve atlas discovery, nonmatching mesh interpolation, or unknown surface correspondence.
- Affine target transitions are enforced exactly. Nonlinear transitions currently have tested residual/Jacobian and Gauss-Newton primitives but are not yet used in the full surface experiment.
- The cylinder is represented through planar chart coordinates and transitions; no extrinsic 3D curvature estimator or real scanned surface dataset is included.
- Seam C0 compatibility and Beltrami covariance are tested separately. Higher-order seamlessness, holonomy around many-chart cycles, and topology beyond this two-chart cylinder remain future work.

## Scientific claims not supported

This prototype does not establish that μ optimization is universally faster, more accurate, or uniquely bijective. It does not establish clinical or real-data utility. It supports exact facewise gradients, strong synthetic solver accuracy, a useful μ-LBS tradeoff on the tested tasks, and the necessity of transition-aware rather than coordinate-equality seams.
