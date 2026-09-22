# Phase VI worklog

Start: 2026-09-22T18:59:43Z. The `START_TIME.json` file records the goal creation time. No Phase V clock is reused.

## Route A — fixed-grid monotone construction

Question: Can a dense rectangular P1 map be generated from latent logits with no global linear solve, while guaranteeing a homeomorphism on the supplied fine triangulation?

Exact claim: On the standard lower-left to upper-right diagonal triangulation, define vertex images \(F(i,j)=(X_i,V_{ij})\), where \(X_0=0<X_1<\cdots<X_{N-1}=1\) and, for every column \(i\), \(V_{i0}=0<V_{i1}<\cdots<V_{i,N-1}=1\). The P1 extension maps the unit square homeomorphically onto itself. A row-wise analogue follows by swapping coordinates.

Assumptions: Complete rectangular triangulation, consistent face orientation, strictly realized coordinate inequalities, finite coordinates, shared boundary values on adjacent faces. The implementation uses normalized positive increments with a uniform floor and checks inequalities in the represented dtype.

Proof sketch: Within either triangle of a cell, the first output coordinate is affine in source \(x\) alone with positive derivative. The second output coordinate has positive source-\(y\) derivative: in the lower triangle it is the positive increment in the right column, and in the upper triangle the positive increment in the left column. Thus every affine face has positive determinant. More directly, \(X(x)\) increases from 0 to 1; for each fixed \(x\), \(y\mapsto F_2(x,y)\) is continuous, piecewise affine, strictly increasing from 0 to 1. Therefore every target \((u,v)\) has one preimage, found first from \(X(x)=u\), then from \(F_2(x,y)=v\). Compact-to-Hausdorff continuous bijectivity gives a homeomorphism.

What falsifies it: A face with zero/negative represented determinant, noninjective vertical fiber, failed boundary coverage, or a finite-precision collapse at intended resolutions.

Smallest decisive test: Random and extreme float32 logits on 17² and 257² grids; independent signed face-area check; first-order directional finite difference for VJP; then time a 257² forward and backward.

Prior work: Phase V's fixed-grid positive Tutte decoder and exact composition implementation provide comparators. This explicit triangular family is deliberately more restricted than unrestricted Tutte weights; expression on cross-coupled targets must be measured, not assumed.

Baseline: Local Windows Anaconda torch 2.5.1+cpu. The initial baseline pytest invocation imported Torch before NumPy LAPACK and aborted in `numpy.linalg.eigvalsh`; importing NumPy and executing a tiny eigvalsh before pytest made the existing symmetric/composition selection pass (47 passed, 2 CUDA-unavailable skips). This is a known runtime load-order issue documented in Phase V, not a Phase VI algorithm failure. New Route A focused tests: 3 passed.

First dense CPU measurement: `tools/phase6_benchmark_monotone.py --side 257 --batch 1 --repeat 3 --device cpu --dtype float32`, with `OMP_NUM_THREADS=4`, on local Intel64 Family 6 Model 60 CPU. The control mesh has 66,049 vertices and 131,072 faces. The three complete decode+map-loss forward times were 0.9605/0.9378/0.9473 ms; backward times were 0.6108/0.6020/0.6170 ms. Minimum signed face-area ratio was 0.29679. The untrained random-logit map had target RMSE 0.02127, which is only a workload value, not fitted accuracy. These timings exclude image warping and neural prediction; no CPU process peak was measured. A GPU and actual-training comparison remain necessary.
