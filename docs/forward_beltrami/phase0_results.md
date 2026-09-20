# Forward Beltrami Phase 0 Results

## Scope

This phase establishes the common manufactured-map and metric layer used by all
future routes. It does **not** claim that any solver reconstructs an arbitrary
Beltrami coefficient, nor that a small equation residual is a topology proof.
The topology result is recorded independently through the existing floating-point
injectivity audit.

## Local reproducibility

The deterministic runner was executed at `256 x 256` cells (`131072` faces) in
`artifacts/forward_phase0/local_256`. Both the affine and smooth-shear maps had
zero flipped faces and passed the independent topology audit. The measured
minimum determinants were `1.0401999999998985` and
`0.9999999999999716`, respectively. Re-loading `maps.npz` in a separate Python
process reproduced every topology flag and minimum determinant to below
`1e-12`.

The focused TDD suite contains six tests and passed with exit code 0. The full
repository suite passed with `144 passed, 1 warning`; the warning is the
pre-existing Paramiko/cryptography Blowfish deprecation warning. Running with
`-W error` exposes that third-party warning as a collection error, so it is
tracked as an environment issue rather than silently treated as a code failure.

## Remote realistic-resolution runs

| host | implementation | grid | faces | min det | flipped faces | equation residual | peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| turing | self-contained C++17 P1 loop | 512x512 | 524288 | 0.9999999999999432 | 0 | 5.93e-17 | 4.5 MiB |
| element | self-contained C++17 P1 loop | 512x512 | 524288 | 0.9999999999999432 | 0 | 5.93e-17 | 4.0 MiB |
| ai | NumPy vectorized P1 loop | 1024x1024 | 2097152 | 0.9999999999998863 | 0 | 5.93e-17 | 757 MiB |

The turing and element default Python interpreters do not provide NumPy. Their
SSH sessions remained healthy, and an equivalent C++17 implementation was used
without installing packages or changing host configuration. The ai run used
the available NumPy environment. All result JSON files were streamed back to
the D drive; no experiment result was written to C.

## Interpretation

The common P1 geometry and audit layer scales to at least two million faces and
has the expected unit-determinant behavior for the smooth manufactured shear.
This is a benchmark-infrastructure result, not evidence that BHF, Beurling,
Tutte, M-matrix, or any other forward route is complete. The next phase must
use this same metric contract to compare actual reconstruction methods, with
periodic/toroidal and rectangle-boundary cases kept separate.
