# Common notation and acceptance criteria

## 1. Domain, mesh, and finite-element map

Let `Omega` be a bounded planar domain with a positively oriented conforming
triangulation `M=(V,F,E)`. A vertex `i` has source position `x_i=(x_i,y_i)`
and unknown image position `U_i=(u_i,v_i)`. The P1 basis functions satisfy
`phi_i(x_j)=delta_ij`, and the discrete map is

`F_U(x)=sum_i U_i phi_i(x)`.

On a face `T=(i,j,k)`, `grad(phi_i)` is constant. Therefore

`J_T(U)=DF_U|_T=sum_{a in T} U_a (grad phi_a)^T`

is constant on `T`. We use the row convention

`J_T=[[u_x,u_y],[v_x,v_y]]`.

## 2. Complex notation and the discrete coefficient

Identify `(x,y)` with `z=x+iy` and `(u,v)` with `F=u+iv`. Define

`F_z=0.5*(u_x+v_y+i*(v_x-u_y))`,

`F_barz=0.5*(u_x-v_y+i*(v_x+u_y))`.

The facewise coefficient is

`mu_T(U)=F_barz/F_z`.

The identity

`det(J_T)=|F_z|^2-|F_barz|^2`

implies `det(J_T)>0` whenever `F_z != 0` and `|mu_T|<1`.

## 3. What the numerical certificates mean

The independent face audit computes every mapped signed area and reports the
minimum area ratio and number of faces with nonpositive orientation. A zero
flip count is a local discrete certificate. It is not automatically a global
homeomorphism theorem: the boundary, graph topology, domain connectivity, and
absence of self-overlap still matter.

For a layer `Y=S(theta)`, a VJP is the covector `dL/dtheta` satisfying

`dL = <dL/dtheta,dtheta>`

for the real Euclidean inner product. “Finite gradient” means the computed VJP
is finite on the stated test, not that every active-set branch is smooth.

## 4. Acceptance target

The desired solver must specify an input class `A` of coefficients/mesh/boundary
data and prove or experimentally validate, for every accepted input:

1. existence of a P1 output;
2. piecewise-affine representation on the requested mesh;
3. global homeomorphism under explicit hypotheses;
4. forward and reverse cost bounds;
5. differentiability of the accepted computational branch; and
6. refinement/error control.

The route monographs distinguish these six requirements instead of treating a
single zero-flip experiment as evidence for all of them.
