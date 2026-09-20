# Route E — M-matrix discretization and positive directional cones

## 1. From Beltrami coefficient to SPD tensor

For `mu=mu_1+i mu_2`, `|mu|<1`, define

`A(mu)=1/(1-|mu|^2) * [[1-2mu_1+|mu|^2, -2mu_2],`
`                         [-2mu_2, 1+2mu_1+|mu|^2]]`.       (E.1)

This is symmetric positive definite and is the conductivity tensor associated
with the divergence-form elliptic representation of the Beltrami equation.

## 2. Positive stencil representation

For mesh-realizable unit directions `d_k`, seek conductances `c_k>=0` such that

`A=sum_k c_k d_k d_k^T`.                         (E.2)

Writing each symmetric matrix as a three-vector yields a dictionary matrix

`D=[(d_kx^2,d_kx d_ky,d_ky^2)^T]_k`.

The code solves the nonnegative least-squares cone problem

`c*=argmin_{c>=0}||Dc-a||_2`.                    (E.3)

The residual is a direct measure of local representability, not a generic
solver residual. A positive residual means the tensor lies outside the finite
direction cone or that the numerical active-set fit failed.

## 3. M-matrix operator

For a scalar field `q` and an integer stencil vector `r_k`, a flux-form operator
has the schematic form

`(L_hq)_i=sum_k c_{i,k}(q_{i+r_k}-q_i)/|r_k|^2`,

with symmetric neighbour averaging when required by consistency. Nonnegative
off-diagonal couplings and positive diagonal dominance produce an M-matrix
structure, which supports a discrete maximum principle and monotonicity.

## 4. Why SPD does not imply mesh-native positivity

Every SPD matrix has the spectral decomposition

`A=lambda_1e_1e_1^T+lambda_2e_2e_2^T`,

with positive eigenvalues. However, eigenvectors `e_i` are generally not mesh
edges. Restricting to a finite stencil replaces the full SPD cone by a polygonal
positive cone. Rotating Beltrami fields can leave that cone even though
`|mu|<1` everywhere.

The route experiments show exact radius-1 fits for mild anisotropy, recovery by
radius 2 for some magnitude-0.6 fields, and large residuals for magnitude
0.8--0.9 even at radius 4. A two-ring unstructured graph improved local fit but
was not planar and produced flipped original faces.

## 5. Assembly and complexity

For `m` candidate directions, enumerating all feasible one-, two-, and
three-direction active sets costs combinatorially in `m`. The 256² radius-4/6/8
follow-up ran about 23 minutes without a receipt; the 512² run ran about 63
minutes. This is an assembly/selection bottleneck before solving any global
linear system.

## 6. Decision

Route E is a useful diagnostic and a possible local decoder when the coefficient
field stays inside a known positive cone. It cannot be advertised as a general
Beltrami solver without planar enrichment, selection regularization, boundary
closure, and a proof that the resulting vector map remains injective.
