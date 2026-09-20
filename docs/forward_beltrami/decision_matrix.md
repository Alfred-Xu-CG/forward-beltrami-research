# Forward Beltrami route decision matrix (live)

This is a research decision aid, not a final selection. “Promising” means the
route has a correctness-preserving kernel that survived realistic-resolution
checks; “open” means a required theorem/operator is still missing.

| route | strongest demonstrated property | current bottleneck | practical interpretation |
|---|---|---|---|
| Realizable BC | compatibility nullity one for manufactured maps, zero for generic fields; sparse-Jacobian fixed-boundary projection through 64²/8,192 faces | nonlinear projection/chart and rectangle boundary | valid latent manifold, but not arbitrary facewise `mu` |
| Periodic Beurling/GMRES | FFT forward, matrix-free GMRES, implicit VJP, spatially varying zero-mode-safe map lift/VJP at 256², 1024² particle-mesh control, 32,768-point separated/near GPU-direct treecode checks, cubic/sinc⁸ low-frequency gridding, 2048² A6000 throughput control, and a 129²/257² identity-boundary conductivity negative control with residual saturation | nonlinear DEQ/root layer, production high-accuracy scattered operator, high-accuracy boundary/PV treatment, and near-Nyquist dispersion | excellent differentiable torus reference plus a general-mesh accuracy control; current treecode is not yet the production fast backend |
| Zero-padded whole-plane | boundary discrepancy and zero-mode defect measured | normalization at infinity and singular quadrature | useful diagnostic, not a rectangle solver |
| Rectangle FD BVP | 512² sparse direct solve plus boundary/coefficient VJPs | reusable factor memory and higher-order discretization | strongest nonperiodic control so far |
| BHF/PL flow | determinant-root and conservative smooth-safe updates through 256² with finite VJPs and zero flips, analytic Möbius velocity, global incident-vertex/edge Duffy PV convergence through 512², and GPU-native 512² assembly on 524,288 faces with zero flips | arbitrary-target PV assembly, active-set/implicit adjoint theory, nonlinear BHF-operator/atlas integration, and roughly 17.8-minute direct 512² quadrature cost | topology-safe differentiable flow plumbing is available for controlled surfaces; direct quadrature scaling and arbitrary-μ reconstruction remain limiting |
| Tutte/Floater | convex-boundary hard injectivity, positive nonuniform weights at 512², sparse implicit VJP, and 64² learned positive-edge expressivity audit | interior expressivity and stable heterogeneous training | robust decoder, not arbitrary QC solver |
| M-matrix QC | exact SPD spectral decomposition, finite-direction audit, and positive integer wide-stencil residual study through max step 6 | boundary closure, stencil conditioning, and unstructured connectivity | adaptive regular-grid stencil is plausible; a general QC decoder is not yet established |
| Mesh flow/shears | determinant-one affine and positive-increment layers | coupled expressivity | good hard geometric building blocks |
| Sphere/orbifold | two-chart gate and closed 256x128 two-cap assembly with `6.75e-16` seam mismatch | arbitrary cone-point data and BHF comparison | viable surface route, separate from rectangle |
| Learned init | 256² two-feature polynomial seed preserves certified roots; 512² and 1024² cost-inclusive Neumann stress tests preserve roots and reduce iterations; a finite rough-band 512² control reaches median speedups 1.187x/1.185x/1.128x, while 1024² smooth-field medians are only 1.052x/1.008x/0.907x for orders 1/2/3 | robust learned/general speedup, arbitrary rough-coefficient robustness, and cost-inclusive training | safe acceleration adjunct, never the certificate |

The current evidence supports a hybrid research architecture: use a rectangle
BVP or compatible-BC solver as the correctness reference, use implicit VJPs, and
compose it with explicit monotone/shear or Tutte layers when hard topology is
more important than arbitrary `mu` expressivity. The open gates above remain
mandatory before claiming a complete forward Beltrami layer.
