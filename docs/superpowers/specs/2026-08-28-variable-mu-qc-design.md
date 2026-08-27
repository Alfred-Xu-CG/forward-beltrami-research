# Variable-μ Computational QC Research Prototype

## Purpose and novelty boundary

Build a reproducible CPU prototype for differentiating through quasiconformal reconstruction on triangular meshes, comparing μ-space and direct-map optimization under large distortion, and extending local QC reconstruction to transition-aware multi-chart registration.

Xu and Lui's SBN/LSQC work already establishes differentiability and demonstrates equiareal and inconsistent registration. This project therefore does **not** claim the first differentiable LSQC method. Its candidate contribution is an exact residual-controlled sparse implicit layer, used as ground truth for surrogates and combined with independent discrete-injectivity auditing, fair direct-map safeguards, and multi-chart compatibility.

## Research questions

1. Can sparse variable-μ LBS/LSQC deliver accurate gradients for every face without differentiating through factorization iterations?
2. Which parameterizations and safeguards preserve a discrete homeomorphism under large deformation?
3. Does μ-space optimization retain an advantage after direct-f receives SLIM-, LIM-, and AMIPS-style barriers?
4. Can chart-local QC maps be coupled by transition compatibility rather than incorrect raw-coordinate equality?

## Scope and non-claims

- Deterministic SciPy/NumPy CPU reference plus PyTorch custom autograd; not a production GPU solver.
- A "certificate" is a documented floating-point discrete disk-map audit, not exact arithmetic.
- Baselines are labelled `slim_style`, `lim_style`, and `amips_style`; they do not claim to reproduce all engineering details of the original solvers.
- Normalized face-area targets may still be infeasible for a PL rectangle homeomorphism. Manufactured-feasible and unproven stress targets are separated.
- The first atlas experiment uses matched synthetic charts and known transitions; automatic atlas/correspondence discovery is excluded.

## Solver design

Geometry and sparsity patterns are cached; numeric matrices are rebuilt and factorized for each changed μ. The same factorization serves the forward solve and transpose adjoint solve.

### LBS

With linear constraints `Cx=d`, solve

\[
\begin{bmatrix}K(\mu)&C^T\\C&0\end{bmatrix}
\begin{bmatrix}x\\\lambda\end{bmatrix}=
\begin{bmatrix}0\\d\end{bmatrix}.
\]

The face tensor is

\[
A(\mu)=\frac1{1-|\mu|^2}
\begin{bmatrix}(1-\rho)^2+\tau^2&-2\tau\\-2\tau&(1+\rho)^2+\tau^2\end{bmatrix}.
\]

Rectangle sliding fixes `u=0/1` on left/right and `v=0/1` on bottom/top, leaving tangential boundary coordinates free. Fully fixed boundary is retained for manufactured tests.

### LSQC

Unweighted real residuals are

\[
r_1=(1-\rho)u_x-\tau u_y+\tau v_x-(1+\rho)v_y,
\]
\[
r_2=-\tau u_x+(1+\rho)u_y+(1-\rho)v_x-\tau v_y.
\]

Rows receive `sqrt(face_area)`. Weighted LSQC additionally uses `(1-|μ|²)^(-1/2)`; the repository's historical `lsqc.m` factor `1-|μ|⁴` is neither this weighted form nor the unweighted form. With two complex pins, solve the augmented least-squares KKT system

\[
\begin{bmatrix}-I&B&0\\B^T&0&C^T\\0&C&0\end{bmatrix}
\begin{bmatrix}r\\x\\\lambda\end{bmatrix}=
\begin{bmatrix}0\\0\\d\end{bmatrix},
\]

without forming normal equations.

### Exact adjoint

For `M(μ)y=b`, solve `M(μ)^T z=∂L/∂y`. Each face parameter obeys

\[
\frac{\partial L}{\partial\theta_T}=-z^T\frac{\partial M}{\partial\theta_T}y,
\quad\theta_T\in\{\Re\mu_T,\Im\mu_T\}.
\]

Analytic local derivatives return every facewise gradient. Finite differences and double-precision PyTorch `gradcheck` validate them. Raw parameters use radial squashing

\[
\mu=k_{\max}\tanh(r)p/r
\]

with its continuous zero limit.

## Injectivity

Three separate layers are used: `|μ|<k_max`, training barriers, and an independent acceptance audit. Certified μ line search interpolates raw parameters from the last certified iterate, resolves the full QC system, and accepts only an audited map. The audit checks positive oriented faces, simple consistently oriented boundary, and unit interior one-ring branch index. Rectangle mode also checks side membership and strict side order. Free LSQC checks boundary simplicity directly.

## Experiments

### Solver/gradient validation

Analytic affine maps, refined smooth manufactured maps, directional-derivative step sweeps, weighted/unweighted conditioning near `|μ|=1`, and mesh-size runtime/memory scaling.

### I-to-S registration

Differentiable smooth analytic fields define a thick `I` and `S`. Compare μ-LBS sliding boundary, μ-LSQC free boundary, unconstrained direct-f, and direct-f plus LIM-, SLIM-, and AMIPS-style energies under the same meshes, initialization, objective budget, and seeds. Report mismatch, soft Dice, folds, minimum area ratio, maximum dilation, certificate, time, and iterations. No winner is predetermined.

### Prescribed area

Use (a) targets induced by known difficult bijections and hence feasible up to discretization and (b) normalized checkerboard/spike/random stress fields with unknown exact feasibility. Report log/relative area error, distortion, folds, certificate, and cost; do not call an unproven target's residual a solver failure.

### Multi-chart registration

Matched source/target cylinder atlases satisfy

\[
g_d\circ\tau^S_{dc}=\tau^T_{dc}\circ g_c.
\]

Affine/conformal transitions enter exact KKT rows; nonlinear transitions use Gauss-Newton. Tests distinguish C0 compatibility from differential/Beltrami covariance. Registration uses landmarks and scalar curvature-like features. Report transition-aware seam residual, a deliberately wrong raw-coordinate-equality control, angular distortion, landmarks, and per-chart injectivity.

## Acceptance criteria

- Sparse forward and adjoint residuals `<1e-10` on small validation cases.
- Median directional-gradient relative error `<1e-5` in a stable step window; PyTorch gradcheck passes.
- Manufactured μ reconstructs its target modulo the declared gauge.
- Every result called bijective passes the independent numerical audit.
- Rectangle constraint residual `<1e-10` with strict side order.
- Affine hard seam residual `<1e-10`; nonlinear/covariance tests `<1e-8` where conditioned.
- Experiments save seed, configuration, metrics JSON/CSV, figures, logs, and artifact manifest.
- Final report includes negative results, scaling limits, and unresolved defects.

## Runtime

Use `C:\Users\xuzhehao\anaconda3\python.exe` with NumPy, SciPy, PyTorch CPU, Matplotlib, scikit-image, and pytest. Experiment modules accept fixed seeds; generated artifacts are never hidden test fixtures.
