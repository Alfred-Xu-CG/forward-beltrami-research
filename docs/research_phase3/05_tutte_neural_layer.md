# Directed Tutte neural layer: theorem scope and differentiability

Status: boundary/modulus API and VJP are implemented; topology and expressivity
claims are conditional and remain under large-mesh benchmarking.

## 1. Directed barycentric system

Let (G=(V,E)) be a triangulated disk with boundary cycle (B) and interior
vertices (I). For every interior vertex (i), choose positive directed
weights (p_{ij}>0) on graph neighbors with (sum_jp_{ij}=1). Given boundary
coordinates (y_binmathbb R^2), the decoder solves
\[
x_i-\sum_{j\in I}p_{ij}x_j=\sum_{b\in B}p_{ib}y_b,
\qquad i\in I.
\tag{T-1}
\]
In block form (M x_I=C y_B). If every interior support path reaches the
boundary, (M) is a nonsingular irreducible (M)-matrix and the forward map is
unique. A sparse factorization or iterative solve supplies (x_I).

## 2. What the directed Tutte theorem actually says

The classical topological guarantee requires more than positive logits. The
relevant hypotheses are a simple triangulated disk, a boundary polygon that is
strictly convex at its corners (collinear side subdivisions are only weakly
convex), a boundary-reachable directed support/3-connected-to-boundary
condition, and positive interior weights. Under these hypotheses, the straight
line drawing has no inverted faces and the interior lies in the boundary
polygon's interior.

The current `DirectedTutteSystem.from_mesh` checks only that there is one
boundary loop. The implicit routine now rejects repeated vertices, negative
turns, non-positive orientation, and degenerate target cycles, so it enforces a
positively oriented weakly convex boundary (including collinear subdivisions).
It still does not certify 3-connected-to-boundary support for an arbitrary
graph. Thus its VJP is mathematically valid for the linear system, while a
global homeomorphism claim still requires the graph certificate as a caller
obligation.

For a rectangle with many collinear samples, use the strict convex four-corner
polygon as the theorem object and treat the side subdivisions as a separate
boundary parameterization. This avoids falsely applying a strictly convex
polygon theorem to a vertex list containing collinear triples.

## 3. Learnable rectangle modulus

For logits (z\in\mathbb R^{4\times m}), each side uses a softmax vector of
positive segment lengths. The side sums are width, height, width, height, so
the final implicit segment closes the rectangle exactly. The new helper uses
\[
H(m)=\varepsilon+\operatorname{softplus}(m)>0
\]
as a tensor-valued height. If width is also a tensor, both aspect ratio and
boundary sampling have a gradient path; Python float defaults remain constants.

The independent finite-difference check at (m=-0.3) gave
\[
\partial_m\sum_k\|y_k\|^2=1.651961931851348,
\quad
\text{central difference}=1.651961931692370,
\]
with absolute discrepancy (1.6\times10^{-10}). Side closure was exact to
roundoff. This validates the latent parameterization, not the topology theorem.

## 4. Backward pass

For a loss (L(x_I,y_B)), solve the transpose system
\[
M^T\lambda=\partial L/\partial x_I.
\]
Then boundary and probability derivatives follow from
\[
d x_I=M^{-1}(dC\,y_B+C\,d y_B-dM\,x_I),
\]
or its adjoint contraction. The implementation differentiates through the
sparse solve without storing an unrolled iteration history. It still stores the
factorization and the interior solution, so memory is (O(|E|)) for sparse
factorization but can be much larger due to fill-in.

## 5. Expressivity statement

For an already valid straight-line PL embedding with fixed boundary, each
interior star has the origin in the strict convex hull of its neighbor rays;
positive local barycentric weights therefore exist. This is a surjectivity
statement onto valid embeddings. It does **not** show that arbitrary bounded
Beltrami coefficients, arbitrary target boundaries, or every positive-logit
decoder output represent a prescribed QC map. The distinction is essential for
neural expressivity claims.

## 6. Route conclusion

Directed Tutte is a promising differentiable linear layer with a clean adjoint
and a learnable rectangle modulus. Its hard-bijective guarantee is conditional
on boundary and graph certificates that are not currently enforced by the
generic API. Therefore it cannot yet be called a universal fold-free forward
Beltrami solver; a fair benchmark must report certificate failures, solve cost,
VJP error, and peak memory rather than only determinant statistics.

## 7. Expressivity stress experiment

To test expressivity rather than only topology, use a valid target with the same
identity rectangle boundary,
\[
F^\star(x,y)=\bigl(x+\alpha\sin(\pi x)\sin(\pi y),
 y+0.5\alpha\sin(\pi x)\sin(\pi y)\bigr).
\]
For \(\alpha\in\{0.05,0.15,0.30\}\), the target is sampled on a 16x16-cell
mesh, directed logits are optimized for 120 Adam steps, and the target/output
facewise Beltrami coefficients are compared:

| alpha | map RMSE | facewise mu RMSE | minimum output face determinant |
|---:|---:|---:|---:|
| 0.05 | 1.2707e-3 | 1.9526e-2 | 2.8304e-3 |
| 0.15 | 2.4450e-3 | 6.3278e-2 | 5.5467e-4 |
| 0.30 | 4.1212e-3 | 8.9111e-2 | 3.7499e-6 |

The fit remains orientation-preserving in these runs, but the conditioning and
distortion margin deteriorate rapidly with target distortion. This supports a
conditional expressivity statement, not universal exact representation. The
optimization itself is an additional cost absent from a direct forward solver.

The theorem-scope comparison follows [Haas et al.'s directed Tutte
formulation](https://www.cs.tufts.edu/research/geometry/pdf/haas04planar.pdf)
and the classical [Tutte planar embedding theorem](https://www.cs.harvard.edu/~sjg/papers/tutte.pdf);
the implementation's weakly convex rectangle subdivision is treated as a
specialized boundary parameterization rather than silently identified with a
strictly convex vertex polygon.

## 8. Same-resolution production-feasibility comparison

On the same CPU process and zero-logit square boundary, the directed Tutte
layer completed a forward and implicit backward pass in 0.8651 s and 0.4131 s
at 129x129 vertices, with minimum face determinant
\(6.1035\times10^{-5}\). At 257x257 vertices the corresponding times were
4.0842 s and 1.8012 s, with minimum *raw* triangle signed area
\(1.5259\times10^{-5}=2/256^2\); the measured resident-set increase was about
121.4 MB. A separate CPU implicit VJP prototype was measured in separate fresh
processes on the same smooth
coefficient: at 65x65 it took
0.7575 s forward, 0.7517 s backward, and increased process RSS by 18.52 MB;
at 129x129 it took 3.2267 s, 3.0190 s, and 59.85 MB; at 257x257 it took
14.1279 s, 12.1254 s, and 247.25 MB. Its gradients were finite, and an
independent 7x7 directional finite-difference check had relative error
9.37e-10. Independently recomputed fixed-(P_1) conjugacy residuals were
0.02416, 0.02366, and 0.02352 at 65, 129, and 257 respectively. The
prototype is differentiable on CPU, but this does not establish exact
discrete conjugacy or GPU readiness. The MBM VJP rows and separate 7x7
finite-difference row can be reproduced with
`src/qcopt/experiments/phase3_mbm_vjp_benchmark.py`; each resolution was run
once in a fresh process with `OMP_NUM_THREADS=1` and no warm-up repetition.
The normalized MBM face determinants are not directly comparable to the raw
Tutte triangle areas. RSS means the process-resident-set increase sampled
before/after the isolated run, not allocator peak. The CPU was an Intel i7-4770
with one OpenMP thread (NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.5.1+cpu).
These are an unmatched prototype cost contrast: Tutte and MBM use different
inputs, boundary parameterizations, and sparse systems, and the MBM timings
include geometry and stiffness assembly. They are not a production GPU
comparison. Within this limited contrast, Tutte has the cheaper backward path
and lower RSS at 257x257, while MBM retains the stronger continuum
interpretation but still lacks a structure-preserving global discrete
certificate.

As a nonuniform-mesh stress test, a Delaunay triangulation with 4,056
vertices, 7,854 faces, and 256 boundary samples was decoded twice. With
directed-logit spreads 1 and 3, the forward-plus-backward times were 1.772 s
and 1.611 s, respectively; gradients and outputs were finite, and the
independent injectivity audit found zero flipped faces and no boundary
intersections. The minimum signed-area ratios were (3.4651\\times10^{-3}) and
(5.5470\\times10^{-11}). The second value is a severe conditioning warning,
not a robust topology margin. This is realistic unstructured stress evidence
under the caller's graph certificate, not a proof for arbitrary Delaunay
meshes or arbitrary Beltrami fields.

Both prototypes are unbatched. Passing a leading batch dimension is rejected
by the public APIs, so a neural training batch would currently require a Python
loop or a new batched sparse-solve implementation. The Tutte implicit VJP also
copies tensors to NumPy/SciPy and back on every pass; the MBM VJP does the same.
CUDA was unavailable in the local measurement environment, and the remote GPU
probe showed occupied devices, so no CPU-to-GPU transfer or GPU kernel result
is claimed here.

At 129x129 with 128 logits per side, the learnable-modulus boundary helper
composed with the same implicit solve in 0.9229 s forward and 0.4133 s
backward. The modulus-logit gradient of the squared-coordinate loss was
2627.4139 and finite; side closure is algebraic because the fourth side is the
implicit closing segment. This confirms a usable latent-to-VJP path, not a
claim that the loss landscape is well conditioned.

## 9. Supporting Beurling/FFT audit

The whole-plane Beurling transform is
\[
(Bg)(z)=-\frac1\pi\operatorname{p.v.}\int_{\mathbb C}
\frac{g(w)}{(z-w)^2}\,dA(w).
\]
An (N_x\times N_y) FFT does not evaluate this integral on a bounded domain
directly. It diagonalizes the *periodized* convolution on a flat torus: the
kernel is replaced by the lattice sum
\[
K_{\mathrm{per}}(z)=\sum_{m\in\mathbb Z^2}K(z+mL),
\]
and the sampled field is periodic across opposite sides. Setting the
coefficient to zero outside the computational window removes the physical
coefficient there, but it does not remove the periodic image interactions.

Centered zero-padding approximates the free-space operator by increasing the
period (L); it is not an exact bounded-domain boundary condition. On a
128x128 uniform grid with a smooth coefficient supported in a central disk
(radius 0.34), periodic FFT and padding-factor-2 differed by 5.19% in the
central radius-0.18 region and 27.5% in the outer annulus. Padding factors 4
and 8 changed the central result by only (2.03\times10^{-4}) relative to
the factor-8 result; periodic versus the factor-8 result still differed by
5.53% centrally and 29.5% in the outer annulus. Runtime grew from 0.0202 s
(factor 2) to 0.325 s (factor 8) on this CPU.

On a general nonuniform mesh there is no exact uniform-grid FFT diagonalization.
The direct quadrature reference in `beurling_direct.py` costs (O(N^2)); a
particle-mesh, NUFFT, or treecode approximation can reduce that cost but needs
separate quadrature, near-singular treatment, and boundary-error validation.
Neither the periodic nor the zero-padded Beurling prototype currently supplies
a hard piecewise-affine homeomorphism certificate. This is why the route is
supporting evidence rather than a surviving production candidate in this
phase.

## 10. Legacy BHF/multilevel status (supporting only)

The earlier BHF prototype evolves a piecewise-affine map by explicit steps

$$
U^{k+1}=U^k+\tau_k V(U^k;\nu),
$$

where `V` is assembled from near-field singular quadrature and blocked
far-field interactions. A determinant-margin step controller and replay-based
VJP were implemented, and a coarse-to-fine schedule used

$$
U_{\mathrm{out}}=\Phi_f^{s_f}\,P\,\Phi_c^{s_c}(U_0).
$$

Legacy measurements on jittered meshes reached 16,641 and 66,049 vertices with
zero observed flips; the 32-to-128 coarse-to-fine prototype used about 4.91 GB
peak CUDA allocation and matched a four-step fine reference to relative map
error \(1.80\times10^{-7}\). A 64-to-256 attempt used about 24 GB and did not
finish. These are valuable multilevel engineering clues, but no Phase III
proof establishes a bounded-domain principal-value identity, arbitrary-mesh
homeomorphism, smooth active-set differentiation, or production memory bound.
Accordingly BHF/FMM/GPU remains a supporting branch, not a certified candidate
or a route closed by the failed large run.
