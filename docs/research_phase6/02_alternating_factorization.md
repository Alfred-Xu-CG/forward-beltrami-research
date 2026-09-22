# Two triangular homeomorphisms as a dense forward layer

This note states the mathematical contract behind the Phase VI two-layer candidate. It distinguishes an exact continuous factorization, its fixed-grid piecewise-affine (P1) approximation, and a sampled image grid. It does **not** claim that every quasiconformal map is representable by two layers or that a sampled composition is automatically P1-bijective on the original triangulation.

## 1. Definitions and notation

Let (Q=[0,1]^2). Write a map (F:Q\to Q) as (F=(F_1,F_2)), with (x) horizontal and (y) vertical. A *homeomorphism* is a continuous bijection whose inverse is continuous. On a compact domain such as (Q), a continuous bijection onto (Q) is automatically a homeomorphism. An orientation-preserving (C^1) diffeomorphism has an invertible derivative (DF), with determinant (det DF>0), and a continuously differentiable inverse.

A vertical triangular map has the form

\[
V(x,y)=(X(x),v(x,y)),
\]

where (X) is strictly increasing from 0 to 1 and, for each fixed (x), (y\mapsto v(x,y)) is strictly increasing from 0 to 1. A horizontal triangular map has the form

\[
H(u,v)=(h(u,v),Y(v)),
\]

where (Y) is strictly increasing from 0 to 1 and, for each fixed (v), (u\mapsto h(u,v)) is strictly increasing from 0 to 1. Both are homeomorphisms of (Q). The implemented layer uses the diagonal triangulation of every rectangular cell and realizes each map as a continuous affine function on every triangle, determined by its fine-grid vertex values. The variable (N) denotes the number of vertices along one side, so the control mesh has (N^2) vertices and (2(N-1)^2) triangles. A query grid with (M^2) points is a separate object.

## 2. Exact two-factor theorem

**Theorem.** Let (F=(F_1,F_2):Q\to Q) be an orientation-preserving (C^1) diffeomorphism that fixes the boundary pointwise. Suppose

\[
\partial_y F_2(x,y)>0\quad\text{for every }(x,y)\in Q.
\]

Then (F=H\circ V), where (V) is a vertical triangular homeomorphism and (H) is a horizontal triangular homeomorphism. One canonical choice is

\[
V(x,y)=(x,F_2(x,y)),\qquad
H(u,v)=\bigl(F_1(V^{-1}(u,v)),v\bigr).
\]

**Proof.** Because (F) fixes the bottom and top edges, (F_2(x,0)=0) and (F_2(x,1)=1). By the strictly positive derivative, for each (x), the function (y\mapsto F_2(x,y)) is strictly increasing and covers ([0,1]). Hence (V) is a bijection of (Q), continuous with continuous inverse, and is vertical triangular. Define (H=F\circ V^{-1}). Its second coordinate is (v), so it is triangular in the other direction. At fixed (v), write (y=y(u,v)) for the inverse vertical fiber. Differentiating (F_2(u,y(u,v))=v) with respect to (u) gives (\partial_u y=-F_{2,x}/F_{2,y}). Consequently

\[
\partial_u H_1
=F_{1,x}-F_{1,y}\frac{F_{2,x}}{F_{2,y}}
=\frac{\det DF}{F_{2,y}}>0.
\]

The two endpoints of each horizontal fiber map to 0 and 1 because (F) fixes the left and right boundaries. Thus (H) is a horizontal triangular homeomorphism. Its definition gives (H\circ V=F). For the fixed canonical choice (V=(x,F_2)), (H) is uniquely determined. □

This is a sufficient representation theorem, not a theorem that *all* orientation-preserving maps satisfy the hypothesis. In fact, a two-layer vertical-then-horizontal composition of the implemented family necessarily has (F_{2,y}>0) wherever its affine derivatives exist: (F_2=Y(v(x,y))), and both factors have positive fiber derivatives. A map with (F_{2,y}<0) on an open set is therefore outside this two-layer family, even if it is orientation-preserving. More layers or a different construction would be needed for such a target.

## 3. Fine-grid P1 discretization

Take uniform vertices (x_i=i/(N-1), y_j=j/(N-1)). For one vertical layer, output vertices have coordinates

\[
V_h(x_i,y_j)=(X_i,v_{ij}),\qquad
0=X_0<X_1<\cdots<X_{N-1}=1,\quad
0=v_{i0}<v_{i1}<\cdots<v_{i,N-1}=1.
\]

On both triangles of every cell, (\partial_x(V_h)_1>0) and (\partial_y(V_h)_2>0), so the affine determinant is strictly positive. More importantly, every vertical fiber of the whole P1 extension is strictly increasing and covers ([0,1]), which proves global bijectivity without inferring it from local determinants alone. Swapping (x) and (y) gives the horizontal layer proof. The neural decoder uses a softmax of finite logits plus a fixed positive spacing floor, followed by cumulative sums. It checks that strict inequalities survive in the represented floating-point dtype. This is a structural guarantee for each layer, conditional on successful represented checks, not a post-hoc fold repair.

If (F) in the theorem is (C^2), sample the canonical (V,H) at the grid vertices and use their P1 interpolants (V_h,H_h). Strict fiber inequalities hold at the sampled vertices, so each P1 factor is a homeomorphism. On a shape-regular uniform triangulation, ordinary affine interpolation of a (C^2) function has a uniform function-value error of order (h^2), where (h=1/(N-1)). Because (H) is Lipschitz on compact (Q),

\[
\|H_h\circ V_h-H\circ V\|_\infty
\le \|H_h-H\|_\infty+\operatorname{Lip}(H)\|V_h-V\|_\infty
=O(h^2).
\]

This is an **existence/approximation** statement. It does not say that finite-step image-only training recovers those interpolants, nor that a fixed spacing floor represents arbitrarily small target derivatives. The implemented floor fraction is 0.01; each normalized interval is at least (0.01/(N-1)). Targets with stronger compression may require a smaller floor or more layers, balanced against floating-point resolution.

## 4. What exact composition means computationally

The continuous composition (H_h\circ V_h) is piecewise affine because the plane can be subdivided by the preimages under (V_h) of edges of the triangulation used by (H_h). That subdivision generally cuts through the **original** triangles. The layer stores two P1 control maps and evaluates a query (q) by first locating (q) in the first grid, applying (V_h), then locating (V_h(q)) in the second grid and applying (H_h). No intermediate image is resampled; only the final coordinate is used to sample the moving image. The chain-rule VJP is defined almost everywhere. A query exactly on a P1 edge receives the derivative of one selected affine branch.

If one samples (H_h\circ V_h) only at the original (N^2) vertices and P1-interpolates those samples on the original mesh, one has constructed a **different map**. The two-layer homeomorphism theorem does not certify this exported map. For a particular checkpoint, checking all original-grid face orientations and the fixed boundary may certify it; this is a checkpoint-specific result, not a parameterization-wide guarantee. The main neural-layer output remains the exact composition representation.

## 5. Beltrami and error measurements

On an affine piece, write (J=DF=\begin{psmallmatrix}u_x&u_y\\v_x&v_y\end{psmallmatrix}). The complex derivatives are

\[
f_z=\tfrac12[(u_x+v_y)+i(v_x-u_y)],\qquad
f_{\bar z}=\tfrac12[(u_x-v_y)+i(v_x+u_y)],\qquad
\mu=f_{\bar z}/f_z.
\]

The evaluation code locates each pixel-center query in each structured triangle, computes that triangle's **exact affine Jacobian**, and multiplies layer Jacobians in the composition order. Its reported map RMSE is (\sqrt{\frac1{2Q}\sum_{q=1}^Q\|F(q)-F_\star(q)\|_2^2}). Its Beltrami RMSE is (\sqrt{\frac1Q\sum_{q=1}^Q|\mu_F(q)-\mu_\star(q)|^2}). Uniform pixel-center sampling approximates an area integral; it is not an exact integral over every refined affine piece. The maximum (\lvert\mu\rvert) and minimum Jacobian determinant in that evaluation are likewise **sampled** quantities. Individual P1-layer face orientations are checked exactly from their vertex coordinates in represented arithmetic.

## 6. Current evidence boundary

On a 257×257 control mesh, a two-layer latent optimized directly against an independently generated smooth target with local 16-cycle detail reached a vertex map RMSE of about (6.93\times10^{-6}) after 1,000 Adam steps. On 512×512 pixel-center queries, the same saved checkpoint has map RMSE (1.35\times10^{-5}), sampled Beltrami RMSE 0.00836, sampled minimum exact-composition Jacobian determinant 0.5366, and maximum sampled (\lvert\mu\rvert=0.3574) versus target maximum 0.3605. The particular exported original-grid P1 interpolation has zero nonpositive faces and minimum signed area ratio 0.5465. Those numbers are direct-map-supervised *expressivity* evidence, not image-only generalization. Separate image-to-latent training and timing must be evaluated on the same data and scale.
