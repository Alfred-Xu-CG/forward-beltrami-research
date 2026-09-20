# Codex 24 小时目标模式：Forward Beltrami / Hard-Bijective Neural Geometry
## Phase II Research Reset — Master Handoff

> **这是给 Codex 直接执行的主计划文件。**
>
> 你收到本文件后，不要只把它当成一份“参考意见”。请先完成下面的仓库内落地动作，然后严格以本文件作为未来 24 小时目标模式的最高优先级研究规划。
>
> 项目仓库：`Alfred-Xu-CG/forward-beltrami-research`

---

# 0. 收到本文件后，Codex 必须立即完成的仓库落地动作

## 0.1 保存长期规划

将本文件的**完整内容**保存为：

```text
docs/research_phase2/PLAN.md
```

若目录不存在则创建。

后续 24 小时内：

- `docs/research_phase2/PLAN.md` 是当前研究阶段的 authoritative plan；
- legacy `docs/forward_beltrami/*` 只作为旧证据、旧实现和参考资料；
- 不得因为旧 `route_status.md` / `completion_audit.md` 中有 `partial` 项目就自行恢复旧路线。

---

## 0.2 创建或更新根目录 `AGENTS.md`

在仓库根目录创建或更新：

```text
AGENTS.md
```

内容以本文件 **“Appendix A — Root AGENTS.md”** 为基础。

要求：

- `AGENTS.md` 保持精简；
- 不要复制整个本文件；
- 它只负责长期工作原则、anti-bureaucracy、skill policy、remote fallback、model routing；
- 它必须明确链接：
  ```text
  docs/research_phase2/PLAN.md
  ```
- 如果根目录已经存在 `AGENTS.md`，不要粗暴覆盖其中与本项目仍然必要的安全/仓库规则；合并后以本计划为当前研究优先级。

---

## 0.3 建立本阶段文档目录

创建：

```text
docs/research_phase2/
```

本阶段只允许形成下列长期文档：

```text
00_state.md
01_mbm_lbs_theory.md
02_positivity_boundary.md
03_primal_dual_qc.md
04_tutte_latent.md
05_results.csv
06_final_decision.md
```

必要时可有少量临时草稿，但：

- 不再建立新的十几条 route 文档；
- 不再建立新的 completion ledger；
- 不再每次 experiment 生成一个独立 audit 目录。

---

# 1. 终极目标

研发一个适合神经网络的 differentiable geometric layer：

\[
\boxed{
z\ \text{(unconstrained / learnable latent)}
\longrightarrow
f_h:\Omega\to\Omega'
}
\]

其中 \(f_h\) 在明确假设下保证为**离散 bijective mapping / PL homeomorphism**，并且：

1. forward 足够快；
2. backward 可通过 implicit/adjoint differentiation；
3. high-resolution image 上 peak memory 可控；
4. 输出没有 folded faces；
5. global boundary/topology 有明确 theorem 或严格适用条件；
6. 几何 accuracy 可量化；
7. 可作为 diffusion / flow matching / registration 网络的一层。

本阶段不是继续“所有路线都做一点”。

本阶段要完成的是：

\[
\boxed{
\text{从杂乱探索}
\longrightarrow
\text{收敛到 1–2 个真正值得下一阶段继续的 architecture}
}
\]

---

# 2. 纠正旧研究模式

旧的：

- `docs/forward_beltrami/`
- `routes_detailed/`
- `summary/`
- `route_status.md`
- `completion_audit.md`
- `src/qcopt/forward/*`
- artifacts / experiment scripts

只视为：

- legacy evidence；
- reusable code；
- negative results；
- prior prototypes。

它们**不再定义当前 agenda**。

特别是：

- 不再默认 BHF 是 production 主线；
- 不再默认 periodic Beurling/FMM 是当前核心工程；
- 不再同时推进 A–N 十几条 route；
- 不再为了“把 partial 填满”追加实验；
- 不再为了 receipt 更漂亮而跑 512²/1024²；
- 不再自动采用旧 `09_comparative_decision.md` 的 coarse-BHF hybrid；
- 不再因为某条旧路线“理论上有趣”就继续消耗资源。

已有正确结果保留。

**不要为了简化而删除已有安全措施或重写 legacy history。**

---

# 3. 一个母问题、两条主线、两条 supporting theory

给定 quadrilateral / rectangle domain \(\Omega\) 上 Beltrami field

\[
\mu,\qquad \|\mu\|_\infty<1,
\]

定义

\[
A(\mu)=
\frac{1}{1-|\mu|^2}
\begin{pmatrix}
|1-\mu|^2 & -2\operatorname{Im}\mu\\
-2\operatorname{Im}\mu & |1+\mu|^2
\end{pmatrix},
\]

满足

\[
A=A^\top>0,\qquad \det A=1.
\]

最终需要回答：

> 能否从 learnable latent 得到 \(\mu\)、\(A\)、positive weights 或其它几何变量，再通过一个高效、可微、hard-bijective 的 discrete decoder 输出 mapping？

---

# 4. Main Branch A — MBM-LBS

## Mixed-Boundary Modulus Linear Beltrami Solver

目标：

\[
\mu
\to A(\mu)
\to \text{mixed anisotropic harmonic solve}
\to (u,v,M)
\to \text{rectangle PL map}.
\]

核心问题：

1. continuum mixed BVP 是否直接得到 exact QC quadrilateral map；
2. target rectangle modulus \(M_\mu\) 是否可以直接由 PDE/energy/flux 得到；
3. P1 FEM 是否对 compatible P1 Beltrami maps exact；
4. stiffness matrix 何时是 positive graph Laplacian / M-matrix；
5. mixed boundary 是否自动给 ordered rectangle boundary；
6. 若不自动，什么最小 boundary closure 可以保 topology；
7. 最终能否形成高速 linear implicit neural layer。

---

# 5. Main Branch B — Primal–Dual QC Electrical Mapping

目标：

\[
\mu
\to A(\mu)
\to \text{positive planar conductance / discrete Hodge star}
\to u_{\rm primal}
\to v_{\rm dual}
\to \text{rectangle tiling / refined PL map}.
\]

核心问题：

1. 能否离散地保留 generalized Cauchy–Riemann / harmonic conjugacy；
2. positive conductance 是否能同时带来：
   - convex-combination structure；
   - modulus；
   - boundary order；
3. primal/dual coordinate 如何变成实际 image warp；
4. 是否应接受 diamond / medial / orthodiagonal refinement，而不是强制使用旧 P1 mesh。

---

# 6. Supporting Theory C — Exact fixed-P1 realizability

已有 holonomy theorem 保留。

将其重写成 compatible derivative-space：

\[
(a,b)\in
\operatorname{range}
\begin{pmatrix}
D_z^h\\D_{\bar z}^h
\end{pmatrix},
\qquad
\mu=b/a.
\]

只再研究一个关键连接：

\[
h=Q_h[\mu(1+S_hh)]
\]

与 compatibility defect

\[
r_{\rm comp}
=
(I-Q_h)[\mu(1+S_hh)].
\]

目标：

> 把 discrete holonomy / realizability 与 P1-compatible discrete Beurling operator 统一起来。

不要再做 generic trust-region projection sweep。

---

# 7. Supporting Theory D — Directed Tutte

研究：

1. 任意 nondegenerate convex-boundary PL homeomorphism 是否存在 strictly-positive **directed** barycentric weights 精确重现；
2. 若成立，positive directed weights 是否构成 valid-map space 的冗余 global parameterization；
3. MBM-LBS 的 \(K(A)\) 何时本身就是 Tutte/Floater matrix；
4. unknown boundary tangential coordinates 如何纳入 hard-bijective decoder；
5. 是否形成：
   \[
   \text{unconstrained logits}
   \to
   \text{positive weights}
   \to
   \text{sparse linear solve}
   \to
   \text{PL homeomorphism}.
   \]

---

# 8. 当前冻结方向

除非 Main Branch A/B 的结果明确要求，否则 24 小时内不主动深入：

- planar BHF quadrature/FMM/GPU；
- generic Beurling treecode/FINUFFT；
- sphere/orbifold；
- torus；
- high-genus；
- generic barrier/SLIM/AMIPS；
- generic learned preconditioner；
- OT/Brenier；
- circle packing / curvature flow implementation；
- full diffusion training；
- 1024² stress；
- ad-hoc M-matrix finite-direction enumeration；
- 论文式写作。

它们不是被永久否定，只是当前不是主要瓶颈。

---

# 9. 文献工作：只做 question-driven research

不要启动宽泛综述。

每篇论文必须回答一个具体问题。

## 9.1 TEMPO

必须逐式回答：

- 从 Beltrami equation 到 generalized Laplace system 的推导；
- \(A(\mu)\) convention；
- rectangle target height \(h\) 如何处理；
- \(h\) 是 mixed BVP 直接决定还是后验 fitting；
- hybrid first-/second-order reconstruction 解决什么误差；
- point-cloud/MLS 的特有部分；
- 我们 MBM-LBS 真正新增什么。

禁止只写摘要。

## 9.2 Quadrilateral conjugate-function / modulus

查清 isotropic mixed BVP：

\[
\Delta u=0,\quad
u|_{\Gamma_L}=0,\quad
u|_{\Gamma_R}=1,\quad
\partial_nu|_{\Gamma_{B,T}}=0
\]

如何产生 canonical rectangle map 和 modulus。

然后核对 anisotropic \(A\) 情况：

\[
M_\mu
=
\int_\Omega \nabla u^\top A\nabla u\,dA
\]

是否成立。

若 reciprocal mixed problem 的 tensor 并不是同一个 \(A\)，必须纠正。

## 9.3 \(\sigma\)-harmonic mapping theory

只查：

- global homeomorphism theorem 的精确假设；
- 是否要求 full Dirichlet boundary homeomorphism；
- 是否适用于 mixed boundary；
- ellipticity / convex target / regularity 条件。

目标是避免误用 theorem。

## 9.4 Floater/Tutte

查清：

- directed positive row weights 是否需要 symmetry；
- graph hypotheses；
- convex boundary；
- 任意合法 embedding 是否可以反推出 strictly-positive directed weights；
- 是否已有等价 theorem。

若已有，直接引用，不包装 novelty。

## 9.5 Electrical rectangle / discrete analytic / orthodiagonal

只回答：

- positive planar conductance network 如何产生 rectangle tiling；
- primal harmonic 与 dual harmonic conjugate；
- energy/effective conductance 与 modulus；
- orthodiagonal convergence assumptions；
- anisotropic \(A(\mu)\) 是否已有推广。

## 9.6 Compatible FEM / DEC / de Rham

只做 feasibility note：

- RT / Nédélec / DEC 是否能让
  \[
  q_h=A\nabla u_h,\quad \nabla\cdot q_h=0
  \]
  导出 exact discrete conjugate；
- 是否需要 primal/dual scalar spaces；
- 下一阶段是否值得实现。

---

# 10. MBM-LBS 连续理论任务

研究：

\[
\nabla\cdot(A\nabla u)=0
\]

with

\[
u=0\ \Gamma_L,\qquad u=1\ \Gamma_R,
\]

\[
n^\top A\nabla u=0
\quad\Gamma_B\cup\Gamma_T.
\]

再定义：

\[
\nabla v=JA\nabla u.
\]

必须明确：

1. 为什么 \(JA\nabla u\) curl-free；
2. \(v\) 是否存在且只差常数；
3. top/bottom 上 \(v\) 是否 constant；
4. 为什么 \(f=u+iv\) 满足原 Beltrami equation；
5. target 是否必为
   \[
   [0,1]\times[0,M_\mu];
   \]
6. \(M_\mu\) 的 flux / energy 公式；
7. constant-\(\mu\) affine sanity check；
8. target modulus 与 \(\mu\)-induced conformal structure 的关系。

每个结论必须标记：

- `proved here`
- `literature theorem`
- `conjecture/open`
- `false/counterexample`

---

# 11. Complementary mixed solve

定义 normalized \(w\)：

\[
\nabla\cdot(A\nabla w)=0
\]

with

\[
w=0\ \Gamma_B,\qquad
w=1\ \Gamma_T,
\]

\[
n^\top A\nabla w=0
\quad\Gamma_L\cup\Gamma_R.
\]

研究 exact case 下是否：

\[
v=M_\mu w.
\]

核对：

\[
E_A(u),\quad E_A(w),\quad M_\mu.
\]

如果 reciprocal relation 实际需要：

- \(A^{-1}\)；
- rotated tensor；
- dual Hodge star；

必须给出正确形式。

---

# 12. P1 FEM formulation

构造：

\[
K_{ij}(A)
=
\sum_T|T|\,
\nabla\phi_i^\top A_T\nabla\phi_j.
\]

对 \(u\)：

- left/right Dirichlet；
- top/bottom natural Neumann。

vertical/conjugate coordinate 比较：

### A3-a
独立 complementary mixed solve。

### A3-b
从 solved \(u\) 的 flux / discrete conjugacy recover \(v\)。

### A3-c
只有数学上真的正确时，才尝试把 modulus \(M\) 作为 unknown scalar 加入 block system。

**不要为了“一次 solve”制造错误 formulation。**

两次 sparse linear solve 如果：

- 快；
- 可 implicit diff；
- 保 topology；

仍然完全可以成为 neural layer。

---

# 13. 修正旧 `rectangle_conductivity.py`

优先审计：

旧代码是否在求出 interior \(u\) 之前就使用 `boundary_values` 的 interior entries 计算 \(JA\nabla u\)。

正确顺序应为：

1. assemble \(K_A\)；
2. solve \(u\)；
3. 用 solved \(u\) 重算 face \(\nabla u\)；
4. 构造 \(JA\nabla u\)；
5. recover \(v\)。

若旧实现确认有误：

- 写一个简洁 issue note；
- 修正；
- 只重跑依赖该逻辑的最小关键测试；
- 不回头重跑全部旧 artifacts。

---

# 14. Exact P1 recovery theorem

目标 theorem：

> 若存在一个 rectangle-to-rectangle P1 homeomorphism \(f_h=u_h+iv_h\)，其逐面 exact Beltrami coefficient 为给定 \(\mu_T\)，则正确的 mixed FEM / conjugate formulation 应 exact recover 该 map（up to normalization）。

必须：

1. 证明或找到已有等价 theorem；
2. 明确 interior-edge flux continuity；
3. 明确 boundary；
4. 明确 uniqueness；
5. 做 4×4 / 8×8 manufactured exact test。

不要先跑 256²。

---

# 15. Positive convex-combination / M-matrix

## 15.1 Row-wise lemma

正式证明：

\[
K_{ii}>0,\quad
K_{ij}\le0\ (i\ne j),\quad
K\mathbf1=0
\]

当且仅当对应 row 可写成：

\[
x_i=\sum_{j\ne i}p_{ij}x_j,\qquad
p_{ij}\ge0,\quad
\sum_jp_{ij}=1.
\]

区分：

- nonnegative；
- strictly positive；
- irreducible。

不要把它写成“bijection 的必要充分条件”。

正确逻辑：

\[
\text{M-matrix row}
\iff
\text{convex-combination row},
\]

而

\[
\text{positive interior rows + ordered convex boundary}
\Rightarrow
\text{PL injectivity}
\]

是在 Floater/Tutte 假设下的充分条件。

---

# 16. \(K(A)\) 何时满足正权重结构？

研究：

- anisotropic nonobtuse；
- anisotropic Delaunay；
- intrinsic Delaunay；
- modified metric cotangent weights；
- AD-LBR 只作为参考。

输出最准确的：

\[
\text{mesh geometry}+A_T
\Longrightarrow
K_{ij}\le0
\]

条件。

对 regular image triangulation 做小型 phase diagram：

- \(|\mu|\)；
- \(\arg\mu\)；
- diagonal orientation；
- M-matrix yes/no。

不要做大网格。

---

# 17. Branch A 的决定性命题：mixed boundary monotonicity

研究：

> 当 \(K\) 是 irreducible positive planar graph Laplacian / M-matrix，采用 complementary mixed boundary conditions 时，free tangential boundary trace 是否必然沿 rectangle 每条 side strictly monotone？

执行顺序：

1. targeted literature；
2. tiny graph analytic examples；
3. proof attempt；
4. 自动搜索最小 counterexample；
5. 必要时 random small tests。

绝对不要默认答案为 yes。

若出现 counterexample：

- 立刻记录；
- 不要通过调数据掩盖；
- 转向 boundary closure。

---

# 18. Boundary closure 只比较三种

### B0 — natural mixed FEM
最高 PDE fidelity。

### B1 — positive 1D boundary chain

\[
q_i=
\alpha_iq_{i-1}
+(1-\alpha_i)q_{i+1},
\quad0<\alpha_i<1.
\]

研究：

- side order；
- square sparse system；
- 与 natural Neumann 的偏差；
- 是否适合作 learnable boundary latent。

### B2 — primal-dual electrical boundary
让 boundary order 从 dual potential / rectangle tiling 自然产生。

---

# 19. Primal–Dual QC Electrical Mapping

## PD1. 先做 \(\mu=0\)

构造最小 orthodiagonal / positive conductance quadrilateral：

\[
\sum_jc_{ij}(u_i-u_j)=0.
\]

定义 current：

\[
I_{ij}=c_{ij}(u_i-u_j).
\]

在 dual graph 上积分 harmonic conjugate \(v^\*\)。

验证：

- rectangle tiling；
- modulus；
- no overlap；
- boundary order；
- refinement trend。

这是 Branch B unit test。

## PD2. 再推 anisotropic \(A(\mu)\)

研究：

> 能否从 facewise \(A\) 构造 positive planar conductances / discrete Hodge star，使 energy 逼近
> \[
> \int\nabla u^\top A\nabla u\,dA
> \]
> 并保留 primal-dual conjugacy？

候选：

- local metric pullback；
- anisotropic cotangent；
- intrinsic Delaunay；
- DEC Hodge star；
- \(A\)-metric orthodiagonalization；
- compatible FEM。

只选 1–2 个最有理论依据的实现。

## PD3. 允许改变 mesh representation

接受：

\[
u:V_{\rm primal}\to\mathbb R,
\qquad
v^\*:V_{\rm dual}\to\mathbb R.
\]

研究 diamond / medial / barycentric refinement：

- primal edge + dual edge 形成 cell；
- target cell 是 rectangle/orthogonal quad；
- 再 triangulate 成 PL map。

如果需要改变原 image triangulation，是允许的。

---

# 20. Supporting Theory：P1 projected Beurling

不再做 FMM/FINUFFT。

完成：

\[
Q_h=D_{\bar z}^h(D_{\bar z}^h)^\dagger,
\qquad
S_h=D_z^h(D_{\bar z}^h)^\dagger,
\]

以及：

\[
h=Q_h[\mu(1+S_hh)].
\]

测试：

1. exact realizable P1 \(\mu\)；
2. random bounded incompatible \(\mu\)。

记录：

\[
\|(I-Q_h)[\mu(1+S_hh)]\|
\]

与实际 BC error。

问题回答清楚即停止，不继续工程化。

---

# 21. Directed Tutte

## T1. theorem-first

对 arbitrary valid convex-boundary PL homeomorphism \(Y\)，研究是否存在：

\[
p_{ij}>0,\qquad
\sum_jp_{ij}=1,\qquad
Y_i=\sum_jp_{ij}Y_j.
\]

结合 linear-system uniqueness，判断 directed positive weights 是否能精确重现任意合法 embedding。

必须查 prior art。

## T2. unknown boundary

研究：

1. positive boundary increments + interior Tutte；
2. boundary 1D positive chain + interior Tutte 联立。

目标：

\[
\text{unconstrained logits}
\to
\text{positive weights}
\to
\text{sparse solve}
\to
\text{PL homeomorphism}.
\]

---

# 22. 最终 neural-layer 候选

### Layer A — BC-primary

\[
z\to\mu\to\text{MBM-LBS}\to f.
\]

### Layer B — map-primary

\[
z\to\text{positive directed Tutte weights + boundary logits}\to f.
\]

### Layer C — metric-primary

\[
z\to w\to A=e^{S(w)}
\to\text{positive harmonic / primal-dual decoder}\to f.
\]

最终评价：

\[
\boxed{
\text{hard bijection}
+
\text{accuracy}
+
\text{forward speed}
+
\text{backward speed}
+
\text{peak memory}
+
\text{batchability}
+
\text{neural usability}
}
\]

---

# 23. Accuracy / topology / gradient 指标

## Correctness

- flipped faces；
- min signed area / Jacobian；
- boundary side membership；
- boundary monotonicity；
- global degree / injectivity audit；
- theorem hypotheses 是否真的满足。

## Accuracy

- map error；
- facewise \(\mu\) error；
- first-order Beltrami residual；
- conjugacy residual：
  \[
  \|\nabla v-JA\nabla u\|;
  \]
- modulus error；
- refinement convergence。

## Differentiability

优先：

- implicit VJP；
- transpose solve；
- matrix-free adjoint；
- custom sparse solve。

不要默认 unroll 数百步。

small double-precision gradient check 目标通常：

\[
\lesssim10^{-5}
\]

relative error；若 conditioning 特别差要解释。

## Performance

记录：

- assembly；
- setup/factorization；
- forward solve；
- backward solve；
- peak RAM/VRAM；
- matrix 是否随 latent 变化；
- factorization 是否可复用。

---

# 24. Resolution policy

### Tiny
16²–64²：proof/counterexample/gradient，优先本地。

### Medium
128²–256²：tiny correctness 通过后。

### Large
512²+：medium 同时通过 topology + accuracy 后，优先 remote。

24 小时目标不是强行达到 1024²。

---

# 25. 24 小时执行节奏

## 0–1.5 h
完成仓库落地动作、research reset、读取 legacy 核心结果，不跑大实验。

## 1.5–5 h
TEMPO + modulus + mixed BVP + sigma-harmonic scope。

输出：

```text
docs/research_phase2/01_mbm_lbs_theory.md
```

## 5–8 h
修正 conductivity 实现；P1 exact recovery；tiny tests。

## 8–11 h
M-matrix criterion + mixed-boundary monotonicity theorem/counterexample。

输出：

```text
docs/research_phase2/02_positivity_boundary.md
```

## 11–15 h
Primal-dual theory + isotropic tiny prototype + anisotropic feasibility。

输出：

```text
docs/research_phase2/03_primal_dual_qc.md
```

## 15–18 h
Directed Tutte theorem/prior art + hard decoder relation。

输出：

```text
docs/research_phase2/04_tutte_latent.md
```

## 18–21 h
只比较 surviving architectures；64²–256²。

## 21–23 h
做最小 differentiable neural layer。

不要 full diffusion training。

## 23–24 h
形成明确 decision。

---

# 26. 实验预算

默认：

- tiny：≤2 分钟；
- medium：≤10 分钟；
- large：≤30 分钟。

只有明确高信息价值时超时，并先写一句理由。

失败分类只需：

- theorem counterexample
- discretization inconsistency
- topology failure
- solver convergence failure
- conditioning failure
- gradient failure
- scalability failure
- environment failure
- promising

不要为每类 failure 建 workflow gate。

---

# 27. Remote 三主机策略

有三台远程主机，经 VPN 访问，VPN 约每 12 小时可能断开。

规则：

1. tiny test 不 SSH；
2. medium/large CPU/GPU 才 remote；
3. SSH failure 最多快速重试 2 次，总等待约 2 分钟；
4. 若失败，标记该 host 暂时 unavailable；
5. **绝不暂停目标模式**；
6. 立即转本地 reduced-resolution / theory / coding；
7. 45–90 分钟后或自然 phase transition 再检查；
8. 不持续 ping / polling。

长 remote job（>10 min）：

- 若允许，使用 `tmux` / `screen` / `nohup`；
- 输出简洁 log；
- 不高频 polling；
- VPN 断开时本地继续；
- 不干扰用户已有进程、端口、服务。

---

# 28. 模型与推理强度分配

**最低模型基线：GPT-5.6。**

除 GPT-6 Astra 外，不使用低于 GPT-5.6 的模型。

如果具体环境没有完全同名模型，使用可用的同等级 GPT-5.6 配置；不要自动降到 5.5、5.4、5.3 或其它更低系列。

## 28.1 Coordinator

默认：

**GPT-5.6 Sol — medium/high**

负责：

- 维持研究方向；
- 合并 subagent 结果；
- 时间预算；
- 判断是否 pivot。

## 28.2 Hard math / route redesign

**GPT-6 Astra — high**

用于：

- mixed modulus theorem；
- boundary monotonicity；
- primal-dual geometry；
- Tutte universality；
- 关键反例后的重规划。

真正概念性卡死：

**GPT-6 Astra — xhigh/max**

触发：

- 两个独立 proof attempt 失败；
- theorem 与 counterexample 矛盾；
- 核心 formulation >45 min 无法厘清；
- experiment 推翻主假设。

不要用 max 处理 routine code/debug。

## 28.3 Literature scout

普通 theorem / paper 定位：

**GPT-5.6 Terra — medium**

关键 prior-art / theorem 判断：

**GPT-5.6 Sol — high**

必要时：

**GPT-6 Astra — high**

## 28.4 Numerical implementation

常规实现：

**GPT-5.6 Sol — medium**

复杂 sparse / adjoint / DEC / custom autograd：

**GPT-5.6 Sol — high**

routine tests / utilities：

**GPT-5.6 Luna — medium**

## 28.5 Experiment / logs

小规模 experiment analysis：

**GPT-5.6 Terra — medium**

机械 log parsing / results table：

**GPT-5.6 Luna — medium**

## 28.6 Final synthesis

**GPT-6 Astra — high**

只给它：

- 核心 theory notes；
- `05_results.csv`；
- 关键 code diffs；
- theorem/counterexample。

不要把全部旧 artifact 灌入上下文。

---

# 29. 遇到研究失败时：升级而不是轻易放弃

第一次失败：

- 缩小问题；
- 检查 convention；
- 找最小 counterexample。

第二次独立失败：

调用：

**GPT-6 Astra high/xhigh**

给一个短 problem packet：

- precise statement；
- 两个失败 attempt；
- 最小 evidence；
- 相关 theorem。

要求：

1. proof idea；
2. counterexample；
3. corrected formulation；
4. closest prior art。

仍无结果：

- 标成 open；
- 推 weaker theorem；
- 做 falsifiable experiment；
- pivot 到另一主 branch。

不要把“没立即证明”写成“路线不可行”。

---

# 30. Multi-agent 规则

同时最多 2–3 个独立子任务。

适合并行：

- proof attempt；
- counterexample search；
- independent implementation sanity check。

不允许多个 agent 同时改同一个核心文件。

只在高价值 ambiguity 使用 Best-of-2：

1. mixed-boundary monotonicity；
2. anisotropic primal-dual formulation。

---

# 31. Skill policy

若这些 skill 存在：

### 默认禁用 `ars/experiment-agent`

本阶段是开放式算法研发，不是 controlled reproduction。

只保留其“超时、资源不破坏”的思想，不实际调用。

### `academic-paper`

前 23 小时禁用。

### `academic-research-suite`

只轻量用于：

- theorem；
- prior art；
- equation convention。

### `ars/deep-research`

默认不用。

只有核心 theorem/prior-art 普通 targeted search 无法解决时才调用。

### `verification-before-completion`

只在真实 phase milestone 和 24h final handoff 使用。

### `systematic-debugging`

只用于真实 software bug / test failure。

### `writing-plans` / `brainstorming`

当前已经有完整计划。

除非出现重大科学 pivot，不重复调用。

### `using-superpowers`

只做最小 routing。

不得因为“可能相关”自动启动一串 skill。

---

# 32. Anti-bureaucracy 全局规则

默认**不新增**：

- SHA/checksum/hash receipt；
- manifest；
- frozen contract；
- schema freeze；
- baseline framework；
- completion gate；
- release gate；
- per-experiment gate；
- duplicate audit layer；
- artifact integrity system；
- speculative infrastructure。

只有当 agent 能明确说明：

1. 一个具体失败场景；
2. 为什么 Git/versioning；
3. types；
4. ordinary tests；
5. DB 主键/唯一约束/事务（如适用）；

都不足以防止该失败时，才允许新增。

**不要为了简化删除已有安全措施。**

workflow gate 只放在：

- destructive irreversible action；
- cross-system side effect；
- security boundary；
- production data migration；
- formal release/publish。

研究 residual 阈值不是 workflow gate。

Preflight 不得挤掉实际 code / simulation / measurement。

---

# 33. Artifact 与 Git 规则

Git 已经负责版本历史。

不要新增 hashing layer。

本轮长期保留：

- 5–6 个 theory/result notes；
- 一个 `results.csv`；
- 必要 figures；
- 必要 tests。

不要每个 run 一个 artifact directory。

临时 logs 放：

```text
tmp/
```

或不提交。

Meaningful milestone commit 即可，不为每个实验 commit。

---

# 34. 建议新代码目录

需要时再创建，不先造空壳：

```text
src/qcopt/qc_rectangle/
    tensor.py
    mixed_fem.py
    conjugate.py
    positivity.py
    primal_dual.py
    tutte.py
    implicit.py
```

tests：

```text
tests/qc_rectangle/
    test_tensor.py
    test_mixed_exact.py
    test_modulus.py
    test_positivity.py
    test_boundary_order.py
    test_primal_dual.py
    test_implicit_gradient.py
```

---

# 35. 最小 experiment matrix

### E1 Constant affine QC
验证 exact map、modulus、energy/flux、BC。

### E2 Manufactured compatible P1
验证 exact recovery theorem。

### E3 Smooth continuum \(\mu(x,y)\)
看 refinement convergence。

### E4 Incompatible random facewise \(\mu_T\)
区分 continuum-valid 与 fixed-P1-realizable。

### E5 Boundary monotonicity adversarial search
tiny graphs；目标是 theorem/counterexample。

### E6 Primal-dual isotropic
复现 rectangle tiling。

### E7 Primal-dual anisotropic
只在 E6 清楚后。

### E8 Neural toy
只对 surviving decoder 做一次 end-to-end fitting/registration。

---

# 36. 一个统一 results table

只记录：

```text
question_id
method
resolution
runtime_forward
runtime_backward
peak_memory
mu_error
beltrami_residual
conjugacy_residual
modulus_error
min_det
flip_count
boundary_order_min_gap
gradient_error
notes
```

保存到：

```text
docs/research_phase2/05_results.csv
```

不要 receipt explosion。

---

# 37. 24 小时交付物

必须交：

```text
docs/research_phase2/00_state.md
docs/research_phase2/01_mbm_lbs_theory.md
docs/research_phase2/02_positivity_boundary.md
docs/research_phase2/03_primal_dual_qc.md
docs/research_phase2/04_tutte_latent.md
docs/research_phase2/05_results.csv
docs/research_phase2/06_final_decision.md
```

`06_final_decision.md` 必须明确：

1. 当前第一候选 neural layer；
2. 第二候选；
3. 哪些 theorem/假设被证伪；
4. 哪些旧路线继续冻结；
5. 下一轮最高价值任务，不超过 3 个。

禁止把所有方向继续标成 `partial`。

---

# 38. 本轮成功标准

以下任意一个都算实质进展：

### Success A

在明确 mesh/tensor 条件下证明：

\[
\mu
\to
\text{linear system}
\to
f_h
\]

有 hard PL homeomorphism guarantee。

### Success B

找到 mixed-boundary monotonicity 的真实反例，证明 vanilla mixed-LBS 不能作为 hard layer，从而明确 boundary closure / primal-dual 的必要性。

### Success C

构造 anisotropic primal-dual rectangle decoder，在明确离散假设下输出 nonoverlapping tiling / PL map。

### Success D

证明 directed positive Tutte weights 是足够广泛的/global redundant valid-map parameterization，并形成高速 implicit layer。

### Success E

将 fixed-P1 compatibility projector 与 discrete Beurling fixed point 统一，解释 exact-BC 与 projection error。

“又完成很多 512² audit”不算成功。

---

# 39. 最终执行原则

优先级：

\[
\boxed{
\text{correct formulation}
>
\text{proof/counterexample}
>
\text{small decisive experiment}
>
\text{implementation}
>
\text{scaling}
>
\text{polish}
}
\]

但 production layer 最终必须同时满足：

\[
\boxed{
\text{bijection guarantee}
+
\text{accuracy}
+
\text{forward speed}
+
\text{backward speed}
+
\text{memory}
+
\text{neural usability}
}
\]

> **Do not optimize the old research archive. Solve the new scientific problem.**

---

# Appendix A — Root `AGENTS.md`

将以下内容保存/合并到仓库根目录 `AGENTS.md`。

---

## Mission

This repository is in **research convergence mode**.

The goal is a fast, memory-efficient, differentiable neural layer that maps a learnable latent representation to a **guaranteed-bijective discrete deformation**, while retaining quasiconformal / Beltrami geometry when useful.

Authoritative plan:

```text
docs/research_phase2/PLAN.md
```

Legacy route documents and artifacts are evidence and reusable code, not the current agenda.

## Current priorities

Only these are primary:

1. Mixed-Boundary Modulus Linear Beltrami Solver (MBM-LBS).
2. M-matrix / positive-convex-combination conditions and mixed-boundary monotonicity.
3. Primal-dual / electrical-network QC rectangle mapping.
4. Directed positive Tutte weights as a hard-bijective latent coordinate system.
5. Fixed-P1 compatibility/projected-Beurling only as supporting exactness theory.

Do not expand old A–N routes merely because they are marked partial.

## Research method

Prefer:

1. precise mathematical formulation;
2. proof or smallest counterexample;
3. small decisive numerical experiment;
4. implementation;
5. medium benchmark;
6. large benchmark only when earlier steps justify it.

Negative results are progress.

When a core idea fails twice independently, escalate to a frontier reasoning model before abandoning it.

## Anti-bureaucracy

By default, DO NOT add:

- checksums or SHA manifests;
- custom artifact-integrity systems;
- frozen contracts or schema freezes;
- new baseline frameworks;
- workflow/completion/release gates for ordinary research;
- duplicate audit layers;
- per-run artifact directories;
- speculative infrastructure unrelated to a concrete failure.

Only add one when there is a concrete failure scenario and Git/versioning, types, ordinary tests, and normal constraints are insufficient.

Do not remove existing safety mechanisms merely to simplify the repository.

Workflow gates belong only at irreversible, destructive, cross-system, security, production-data, or formal-release boundaries.

Preflight checks must not crowd out actual code execution, simulation, or measurement.

## Skills

If these skills exist:

- Do **not** use `ars/experiment-agent` during this open-ended research phase.
- Do **not** use `academic-paper` until conclusions are stable.
- Use `academic-research-suite` only lightly for theorem/prior-art verification.
- Use `ars/deep-research` only when a core theorem/prior-art problem cannot be resolved with targeted search.
- Use `verification-before-completion` only at real phase milestones/final handoff.
- Use `systematic-debugging` only for actual software bugs/test failures.
- Do not repeatedly invoke `writing-plans` or `brainstorming`; the current plan already exists.
- Keep `using-superpowers` to minimal routing.

## Experiments

Every experiment must answer a named research question.

Scale order:

- tiny/local first;
- medium only after tiny correctness;
- 512²+ only after medium correctness and accuracy.

Default budgets:

- tiny: ~2 min;
- medium: ~10 min;
- large: ~30 min unless a written high-information reason justifies more.

Use one compact results table. Do not create one audit directory per run.

## Remote compute

Three remote hosts may be available through a VPN.

- Use remote compute for meaningful medium/large jobs, not tiny sanity checks.
- On SSH failure: retry at most twice over ~2 min, mark host temporarily unavailable, and continue locally.
- Do not pause the objective waiting for VPN restoration.
- Recheck at phase transitions or after ~45–90 min.
- Long jobs may use `tmux`/`screen`/`nohup` and concise logs.
- Never kill unrelated processes or modify shared services/ports.

## Model routing

Minimum model baseline: **GPT-5.6**.

- routine implementation: GPT-5.6 Sol medium;
- complex sparse/adjoint/DEC implementation: GPT-5.6 Sol high;
- routine literature extraction / experiment summary: GPT-5.6 Terra medium;
- mechanical log parsing / utility tests: GPT-5.6 Luna medium;
- hard mathematical reasoning / route redesign: GPT-6 Astra high;
- genuine conceptual impasse after two attempts: GPT-6 Astra xhigh/max.

Do not automatically fall back to models below GPT-5.6.

Do not use maximum reasoning for routine work.

## Git and files

Use Git as normal version history. No extra hashing layer.

Meaningful milestone commits are enough.

Do not refactor legacy code merely for cleanliness.

Prefer isolated new modules and minimal corrections to confirmed bugs.

Temporary logs may remain untracked.

## Scientific integrity

Always distinguish:

- theorem vs numerical evidence;
- continuum guarantee vs fixed-mesh guarantee;
- local positive Jacobian vs global homeomorphism;
- exact Beltrami reproduction vs metric/harmonic approximation;
- solver residual vs induced Beltrami error.

Do not claim `|mu|<1` alone guarantees a sampled P1 map is fold-free.

Do not claim a continuous diffeomorphism implies sampled P1 interpolation is fold-free.

Do not use post-hoc fold repair as the primary topology guarantee.


