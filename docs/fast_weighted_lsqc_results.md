# Fast weighted LSQC engineering result

## Implemented scope

The production fast path solves weighted LSQC for constraints whose every row
selects exactly one coordinate. This covers:

- free-boundary LSQC with two complex pins;
- fully fixed boundary vertices;
- axis-aligned rectangle boundaries whose vertices can slide tangentially.

Moving constraints, learned transitions, joint correspondence, mixed linear
rows, and unweighted LSQC deliberately remain on the augmented reference path.

The forward system is assembled directly as

$$
Q(\mu)
=
\begin{bmatrix}
K(\mu) & S\\
-S & K(\mu)
\end{bmatrix},
$$

then exact coordinate pins are eliminated before factorization. The
$\mu$-independent matrix $S$ is assembled only from oriented boundary edges;
interior edge terms cancel exactly. SuperLU uses the symmetric
`MMD_AT_PLUS_A` ordering and retains numerical pivoting. The same saved LU
factorization is reused for the transpose adjoint solve.

The weighted-LSQC VJP evaluates each field's residual and two derivative
actions in the same face-gradient kernel, then uses the exact
$d(B_w^T B_w)$ product rule. It has no Python loop over faces and does not
materialize the full facewise fourth-order derivative tensor.

## Verification

The implementation is checked against the augmented weighted-LSQC reference:

- direct Hessian versus $B_w^T B_w$;
- maps for two pins, fixed boundary, and rectangle sliding;
- analytic gradients for all three constraint families;
- central finite differences and PyTorch double-precision `gradcheck`;
- one forward factorization with reuse in backward;
- an irregular jittered mesh with random $|\mu_T|=0.9$.

The high-distortion test checks solver and gradient equivalence. It does not
claim the resulting map is injective; injectivity must still be audited or
enforced by the separate certificate/barrier machinery.

## Measured result

Environment: Windows, Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1, sequential MKL.
Each timing is the median of seven warmed repetitions on 2026-08-31.

| Grid | Faces | Aug./fast dimension | Aug./fast LU nonzeros | Forward speedup | Backward speedup | Full step speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 288 | 918 / 334 | 35,374 / 10,570 | 3.95x | 1.12x | 3.23x |
| 24 | 1,152 | 3,558 / 1,246 | 223,892 / 59,537 | 6.32x | 1.44x | 5.19x |
| 36 | 2,592 | 7,926 / 2,734 | 631,470 / 151,031 | 6.41x | 1.61x | 6.89x |
| 48 | 4,608 | 14,022 / 4,798 | 1,340,493 / 316,782 | 8.03x | 1.65x | 8.77x |

For the largest case, the median augmented/fast times were 213.26/26.56 ms
forward and 6.96/4.21 ms backward. Independently timed complete steps were
255.74/29.15 ms. The maximum map discrepancy was $1.11\times10^{-12}$ and the
maximum facewise gradient discrepancy was $2.50\times10^{-12}$.

Raw reproducible output is stored in
`artifacts/fast_lsqc_benchmark/2026-08-31/benchmark.json` and
`benchmark.csv`. The exact rerun command is in the repository README.

## API

- `qcopt.lsqc.solve_lsqc_fast`: NumPy/SciPy forward solve.
- `qcopt.adjoint.lsqc_fast_mu_vjp`: exact all-face VJP.
- `qcopt.autograd.lsqc_fast_layer`: bounded facewise $\mu$ components.
- `qcopt.autograd.lsqc_fast_from_raw`: radial squashing plus the fast layer.

The augmented `solve_lsqc`/`lsqc_layer` functions remain the general and
independent numerical reference.
