# Codex 18 小时目标模式计划书
## Phase V — Tutte / MVC / Structure-Preserving Primal–Dual Neural Bijections

> **给 Codex 直接执行的 master handoff**
>
> 仓库：`Alfred-Xu-CG/forward-beltrami-research`
>
> 最终目标：
>
> \[
> \boxed{z\ \text{(learnable latent)}\longrightarrow f_h}
> \]
>
> 其中 \(f_h\) 应尽可能同时满足：离散 bijective / PL-homeomorphic、快速 forward/backward、低显存、可 batch、能用于 256²–512² image registration，并保留有意义的 QC/metric geometry。

本阶段只深入三个方向，并**严格顺序执行**：

1. **Route I — Fast Tutte neural layer**
2. **Route II — MVC canonical coordinates + geometric incremental optimization**
3. **Route III — Primal–dual / RT / Whitney / Hodge**

一个 route 未经独立 checker 闭合前，不得开始下一个 route。

---

# 0. 收到本文件后的立即动作

## 0.1 保存主计划

将本文件完整保存为：

```text
docs/research_phase5/PLAN.md
```

Phase II/III 只作为 prior evidence，不再定义当前 agenda。

## 0.2 更新根 `AGENTS.md`

把本文最后 `Appendix A — Root AGENTS.md` 合并到根目录：

```text
AGENTS.md
```

保留已有必要安全规则，但以本阶段的顺序路线、独立 checker、深度优先和 no-early-finalization 为当前执行规则。

## 0.3 创建真实 wall-clock guard

第一次启动时创建：

```text
docs/research_phase5/START_TIME.json
```

记录真实 UTC：

```json
{
  "start_iso8601_utc": "...",
  "start_unix_seconds": ...
}
```

再创建：

```text
tools/check_phase5_elapsed.py
```

规则：

- `START_TIME.json` 已存在时绝不覆盖；
- 每次用真实系统时钟算 elapsed；
- 禁止用手写 `T+6h` 代替真实时间。

只有：

```text
elapsed >= 17h15m
```

才能开始 final independent review。

只有：

```text
elapsed >= 18h
```

才能把阶段状态写成 `complete`。

否则必须输出：

```text
EARLY_FINALIZATION_DENIED
```

**Success criterion is NOT a stop criterion.**

若某 route 提前成功，剩余时间用于 adversarial test、prior-art check、solver optimization、conditioning、larger realistic experiment 和独立复核，不得提前结束。

---

# 1. 本阶段长期文件

仅保留：

```text
docs/research_phase5/
    PLAN.md
    START_TIME.json
    WORKLOG.md

    01_tuttenet_code_audit.md
    02_tutte_fast_solver.md
    03_tutte_application.md
    04_tutte_checker.md

    05_mvc_theory.md
    06_mvc_optimization.md
    07_mvc_application.md
    08_mvc_checker.md

    09_primal_dual_theory.md
    10_primal_dual_prototype.md
    11_primal_dual_neural.md
    12_primal_dual_checker.md

    13_cross_route_results.csv
    14_final_independent_review.md
    15_final_decision.md
```

临时日志放 `tmp/`。禁止重建 route ledger、completion ledger、每实验一个 artifact folder。

---

# 2. 18 小时硬时间表

## 0:00–0:30
Setup、wall-clock guard、共用 benchmark。

## 0:30–5:45
**Route I — Tutte**，至少 5 小时 15 分。

Route I checker 通过前不得开始 MVC。

## 5:45–10:45
**Route II — MVC**，至少 5 小时。

MVC checker 通过前不得开始 primal-dual。

## 10:45–16:30
**Route III — Primal–dual / RT / Whitney / Hodge**，至少 5 小时 45 分。

## 16:30–17:15
共同输入 cross-route experiment + engineering comparison。

## 17:15–18:00
Independent final review + synthesis。

不得为了“按时出结论”把 Route III 压缩成文献综述。未闭合的问题应诚实写 `not closed`，不能浅尝辄止后假装完成。

---

# 3. 模型与推理强度

最低模型基线：

\[
\boxed{\text{GPT-5.6}}
\]

不得自动降到更低系列。

- coordinator：**GPT-5.6 Sol high**
- 核心工程：**GPT-5.6 Sol high**
- routine utilities/tests：**GPT-5.6 Sol medium**
- 重大数学推导：**GPT-6 high**（若可用）
- 独立 theorem checker：**GPT-6 high/xhigh**（若可用）
- 两次独立尝试仍卡住：最高可用 GPT-6 effort；若不可用则 GPT-5.6 Sol maximum effort

不要把 max effort 用于格式化和日志汇总。

---

# 4. Builder–Checker–Adjudicator

每个 route 都必须：

```text
Builder
  ↓
Independent Checker
  ↓
Adjudicator if disagreement
  ↓
Route closure
```

Checker 不能只看 tests pass，必须主动检查：

- sign / indexing / off-by-one；
- necessary vs sufficient；
- theorem vs code；
- local vs global；
- symmetric vs directed；
- exact map vs sampled map；
- circular validation；
- gradient formula；
- timing/memory；
- overclaim。

高风险词：

```text
unique
iff
necessary
sufficient
universal
exact
guaranteed
canonical
bijective
```

一旦出现，必须触发 checker。

---

# 5. 共用 benchmark

三个 route 必须复用同一 benchmark。

## 5.1 Control mesh 与 dense image resolution 分开

控制网格至少：

```text
11, 17, 25, 33, 49
```

有余力加 65。

dense query/image：

```text
256×256 mandatory
512×512 preferred
```

必须明确：

\[
\boxed{\text{control DOF}\neq\text{image pixel count}}
\]

## 5.2 Synthetic valid deformation suite

至少：

- affine anisotropic stretch；
- shear；
- rotation+stretch；
- twist；
- sinusoidal shear；
- local compression；
- local expansion；
- boundary sliding；
- random positive-Tutte high-distortion maps。

所有 ground truth 必须先证实合法。

## 5.3 Synthetic images

至少：

- checkerboard；
- smooth blobs；
- medical-style phantom；
- textured image。

固定统一 warp convention。

## 5.4 Optional real data

若 ACDC/其它数据已在本地/远端，做一个小 real case；没有则不花时间下载大型数据。

## 5.5 共用指标

Topology：

```text
flip_count
min_signed_area
min_area_ratio
boundary_order_min_gap
global injectivity/degree certificate when applicable
```

Accuracy：

```text
map_RMSE
max_map_error
mu_RMSE
max_mu_error
image_similarity
inverse_consistency
```

Optimization：

```text
iterations_to_threshold
global_solves_to_threshold
wall_time_to_threshold
final_objective
best_objective
```

### Mandatory instance-optimization protocol

在任何 CNN/encoder training 之前，所有可学习 decoder 必须先通过一个**单实例 optimization**：

- 不使用 CNN；
- 直接把 latent 参数设成 `torch.nn.Parameter`；
- 固定一个 synthetic target map 或 image pair；
- 用 Adam、LBFGS 或其它明确 optimizer 直接优化 latent；
- 每一步都真正经过 decoder、dense warp 和 loss；
- 记录 forward / backward / optimizer step 的 wall time；
- 记录达到固定误差阈值所需的 global solve 数；
- 每一步检查 hard-topology 条件。

这优先回答：

\[
\boxed{
\text{decoder + solver + backprop 本身到底能不能稳定、快速地优化一个真实 case？}
}
\]

从而把 network capacity 与 geometric layer 本身分开。

Solver：

```text
setup_time
forward_time
backward_time
solver_iterations
residual
conditioning proxy
peak_CPU_RAM
peak_GPU_VRAM
batch_throughput
```

Gradient：

```text
finite_difference_error
adjoint_identity_error
```

统一写入：

```text
13_cross_route_results.csv
```

---

# ============================================================
# ROUTE I — FAST TUTTE NEURAL LAYER
# ============================================================

# 6. Route I 核心问题

不再主要问“为什么 Tutte injective”，而问：

\[
\boxed{\text{怎样把 Tutte 变成 high-resolution deep-learning 可用的 hard-bijective layer？}}
\]

必须回答：

1. TutteNet 原文和官方代码到底怎样求解；
2. 为什么它实际快；
3. forward/backward 真正成本在哪里；
4. control resolution 与 dense image resolution 如何解耦；
5. directed 和 symmetric-conductance family 哪个更适合；
6. direct / iterative solver 哪个更好；
7. 1/2/4 layer composition 的表达能力和成本；
8. 256²/512² end-to-end neural registration 是否可行。

---

# 7. Route I-A：TutteNet 原文 + 官方代码深度审计

必须逐代码读：

```text
TutteNet paper
GitHub: GitBoSun/TutteNet
dependency: flaport/torch_sparse_solve
```

回答：

### 7.1 控制网格实际 size

明确：

- `MESH_RESOLUTION`
- total vertices
- interior DOFs
- boundary DOFs
- 每层 system size

### 7.2 每层预测变量

核对：

- edge weights；
- boundary parameters；
- 是否 symmetric；
- sigmoid / normalization；
- square/circle boundary mapping。

### 7.3 Sparse backend

核对：

- CPU/GPU；
- dtype；
- KLU/LU；
- batch 是否真正并行；
- forward factorization；
- backward transpose solve；
- 是否复用 forward factors；
- symbolic analysis 是否可缓存。

### 7.4 Dense query

核对：

- point location；
- barycentric interpolation；
- `Delaunay.find_simplex`；
- 哪部分可以预计算。

### 7.5 Multiple layers

核对：

- 每层是否重新 solve；
- composition；
- query propagation；
- cost scaling。

输出：

```text
01_tuttenet_code_audit.md
```

Checker 必须逐条对 paper/code。

---

# 8. Route I-B：实现 image-registration-oriented Tutte decoder

针对 square/rectangle registration 实现，而不是完全照抄 TutteNet。

### Boundary

固定四角。

每条 side positive segment logits：

\[
\Delta s_k
=
\frac{e^{a_k}}{\sum_r e^{a_r}}
\]

保证 side order。

高度可 fixed 或：

\[
H=\epsilon+\operatorname{softplus}(m).
\]

### Interior parameterization：至少比较两类

#### Directed row-softmax

\[
p_{ij}=\operatorname{softmax}_j(\ell_{ij}).
\]

优点：自由度大。

缺点：矩阵一般 nonsymmetric。

#### Symmetric conductance

\[
c_{ij}=c_{ji}>0.
\]

优点：reduced Laplacian SPD，可 CG。

缺点：存在 detailed-balance / expressivity restriction。

不得预先宣布哪一个更好。

---

# 9. Route I-C：三个 solver backend

## Backend 0 — official-style reference

CPU sparse direct，仅作 correctness oracle。

## Backend 1 — improved direct

研究并尽量实现：

- fixed sparsity symbolic cache；
- x/y 多 RHS 共用 factor；
- transpose solve复用 numeric factors；
- backward 不重新 factorize \(A^T\)。

## Backend 2 — matrix-free iterative

Directed：

- GMRES / BiCGStab 等 nonsymmetric solver。

Symmetric conductance：

- CG。

operator 用 fixed neighbor table：

```text
gather → weight → reduce
```

不必 assemble CSR。

---

# 10. Route I-D：custom implicit backward

系统：

\[
A(\theta)X=B(\theta).
\]

adjoint：

\[
A^T\Lambda=G.
\]

对 row-softmax logits验证：

\[
\boxed{
\frac{\partial L}{\partial\ell_{ij}}
=
p_{ij}\Lambda_i^T(Y_j-Y_i)
}
\]

必须做：

- double precision finite difference；
- directional derivative；
- boundary-logit gradient；
- modulus gradient；
- batch gradient。

---

# 11. Route I-E：dense warp 预计算

固定 source control mesh 与 fixed 256²/512² pixel locations。

预计算：

```text
pixel -> source triangle id
pixel -> source barycentric weights
```

每 forward dense warp只做：

```text
gather deformed triangle vertices
+ barycentric weighted sum
```

不得每次用 CPU Delaunay 搜索全部像素。

---

# 12. Route I-F：工程 benchmark

control mesh：

```text
11 / 17 / 25 / 33 / 49
```

batch：

```text
1 / 4 / 8
```

image：

```text
256²
512² preferred
```

layers：

```text
1 / 2 / 4
```

比较：

- reference direct；
- improved direct；
- matrix-free iterative。

记录 forward/backward/memory/gradient/topology/dense warp cost。

---

# 13. Route I-G：Mandatory instance optimization

Route I 不再把 CNN training 当作首要验收。

先做一个最干净的单实例 optimization：

\[
\theta=
\{\text{interior Tutte latent, boundary latent/modulus}\}
\]

直接设成：

```python
torch.nn.Parameter
```

给定固定 valid target map 或 synthetic registration pair，在：

```text
256×256 mandatory
512×512 preferred
```

上优化。

至少包含：

## I1 — supervised map fitting

给定 ground-truth valid deformation \(F^\star\)，直接最小化：

\[
\mathcal L_{\rm map}
=
\|F_\theta-F^\star\|^2.
\]

## I2 — image registration instance

只给 moving/fixed synthetic images，直接优化 latent：

\[
\min_\theta
\mathcal L_{\rm image}
\bigl(I_m\circ F_\theta^{-1},I_f\bigr)
+
\text{必要的几何 regularization}.
\]

要求：

- 不使用 CNN；
- 从 identity / neutral initialization 开始；
- Adam 与至少一种 LBFGS/准二阶方案择一比较；
- 每一步真正调用 Tutte solve 和 dense warp；
- 每一步记录 topology；
- 记录 total global solves；
- 记录 time-to-threshold；
- control mesh 至少 17、25 两种；
- 256² mandatory。

这个实验优先回答：

\[
\boxed{
\text{如果只有一个 case，Tutte latent 是否容易优化，forward/backward 是否足够便宜？}
}
\]

## Optional I3 — tiny neural predictor

只有 I1/I2 已经可靠收敛且时间允许时，才增加一个很小的 encoder：

\[
(I_m,I_f)\to\theta\to F_\theta.
\]

CNN training 不是 Route I closure 的必要条件。

---

# 14. Route I closure

必须完成：

- official code audit；
- fast solver backend；
- implicit gradient；
- control/query scaling；
- 256² supervised map-fitting instance optimization；
- 256² image-registration instance optimization；
- conditioning/adversarial tests；
- independent checker。

输出：

```text
02_tutte_fast_solver.md
03_tutte_application.md
04_tutte_checker.md
```

CNN training 不是 mandatory closure criterion；instance optimization 是 mandatory。

Checker通过才进入 MVC。


# ============================================================
# ROUTE II — MVC CANONICAL COORDINATES + GEOMETRIC UPDATE
# ============================================================

# 15. Route II 核心问题

MVC 不是另一个 competing decoder。

它主要研究：

\[
\boxed{\text{怎样给 Tutte 的冗余 positive-weight space 一个规范表示和更好的优化几何？}}
\]

必须回答：

1. canonical subset 的准确 theorem；
2. MVC prior art 与 deep-learning prior art；
3. MVC encode–decode 是否 exact；
4. raw Tutte latent 到底有多少冗余；
5. covariance logit lift 与 MVC derivative 有何关系；
6. geometric retraction 能否减少 global solve 次数或改善 conditioning；
7. 如何真正适配 network training。

---

# 16. Route II-A：理论与 prior art

必须认真读/定位：

- Floater 2003 Mean Value Coordinates；
- Floater/Gotsman injective tiling morphing；
- Neural Cages / Deep Cage / cage deformation；
- TutteNet；
- Generative Escher Meshes；
- 搜索：
  - MVC + neural deformation；
  - barycentric coordinates + deep learning；
  - canonical Tutte coordinates；
  - differentiable barycentric parameterization；
  - natural/Riemannian gradient on embedding weights。

输出：

```text
05_mvc_theory.md
```

---

# 17. Route II-B：canonical subset 精确定义

记 Tutte decoder：

\[
D(p,b)=Y.
\]

对一个合法 map \(Y\)，MVC encoder：

\[
E(Y)=\bigl(p^{MVC}(Y),Y_B\bigr).
\]

必须在明确 assumptions 下证明或引用：

\[
\boxed{D(E(Y))=Y.}
\]

定义：

\[
\mathcal H=\text{合法 map space},
\]

\[
\mathcal C=E(\mathcal H).
\]

则：

\[
E:\mathcal H\to\mathcal C
\]

injective，且：

\[
D|_{\mathcal C}
\]

是 inverse。

同时明确：

\[
\boxed{D\text{ 在全部 positive weights 上不是 injective。}}
\]

禁止再泛称 “Tutte latent 和 map 一一对应”。

---

# 18. Route II-C：最小必要的冗余/gauge 检查

冗余不是本阶段主目标。

只保留支持 MVC canonicalization 所必要的事实：

1. raw logits 有 row-shift gauge：
   \[
   \ell_i+c\mathbf1
   \]
   不改变 softmax；
2. degree \(d_i>3\) 的一般 one-ring 中，同一个 vertex 位置通常存在多组 positive barycentric weights。

只做最小数值检查来确认这些 gauge/null directions。

**不要求**大规模 Jacobian-SVD 或完整 fiber-dimension study，除非它直接解释 optimization failure。

把主要时间投入：

\[
\boxed{
\text{MVC neural layer}
+
\text{instance optimization}
+
\text{fast forward/backward}
}
\]

而不是抽象冗余量化。

---

# 19. Route II-D：MVC encoder

对 interior vertex，按 cyclic neighbor order：

\[
r_j=Y_j-Y_i,
\qquad
l_j=\|r_j\|,
\]

\[
\theta_j=\angle(r_j,r_{j+1}).
\]

MVC：

\[
\widetilde p_j
=
\frac{
\tan(\theta_{j-1}/2)+
\tan(\theta_j/2)
}{
l_j
},
\]

\[
p_j=
\frac{\widetilde p_j}
{\sum_k\widetilde p_k}.
\]

canonical logits 可取：

\[
\ell_j=
\log\widetilde p_j
-
\operatorname{mean}_k\log\widetilde p_k.
\]

减 mean 只是固定 softmax 行平移 gauge。

必须 robust 处理：

- cyclic ordering；
- orientation；
- nearly degenerate one-ring；
- angle wrap；
- boundary-adjacent interior vertices。

---

# 20. Route II-E：encode–decode benchmark

对 Route I synthetic valid maps：

\[
Y\to E(Y)\to D(E(Y)).
\]

control resolutions：

```text
11 / 25 / 49
```

测：

- max vertex error；
- map RMSE；
- min weight；
- max logit spread；
- covariance conditioning；
- triangle quality。

在 assumptions 满足时，double precision round-trip应接近机器精度。

如果不行：

- 不允许用 optimizer“拟合到差不多”；
- 必须查 theorem / one-ring ordering / implementation。

---

# 21. Route II-F：MVC angles、boundary angles、row logits 必须区分

### TutteNet boundary angles

全局中心射向 target boundary，用于摆 boundary vertices。

### MVC interior angles

one-ring neighbor rays：

\[
Y_j-Y_i.
\]

### Tutte row logits

softmax 参数。

canonical MVC logits：

\[
\ell_j^{MVC}
=
\log
\left(
\frac{
\tan(\theta_{j-1}/2)+\tan(\theta_j/2)
}{
l_j
}
\right)
+C_i.
\]

所以：

\[
\boxed{\text{MVC logit 是 local angle + distance 的函数，而不是 angle 本身。}}
\]

---

# 22. Route II-G：covariance lift

当前 valid map \(Y\)。

定义：

\[
r_{ij}=Y_j-Y_i,
\]

\[
C_i=
\sum_jp_{ij}r_{ij}r_{ij}^T.
\]

给目标一阶 vertex direction：

\[
d.
\]

定义：

\[
b_i=
d_i-\sum_jp_{ij}d_j.
\]

lift：

\[
\boxed{
\delta\ell_{ij}
=
r_{ij}^TC_i^{-1}b_i.
}
\]

必须：

1. 完整推导；
2. finite difference 验证；
3. 与 decoder Jacobian pseudoinverse 比较；
4. 验证合适 weighted norm 下的最小范数性质；
5. 分析 \(C_i\) condition number；
6. 给 degeneracy counterexample。

---

# 23. Route II-H：MVC derivative 与 covariance lift

必须验证：

\[
dE(Y)[d]
\]

一般不等于：

\[
L_Yd
\]

(covariance lift)。

但差值应在 decoder Jacobian kernel 中：

\[
J_D\left(dE-L_Y\right)\approx0.
\]

解释：

> 同一个 map tangent 有多个 latent lifts；MVC derivative 与 covariance lift 只是不同 gauge。

这一步必须由独立 checker 复核。

---

# 24. Route II-I：合法 retraction

canonicalize：

\[
\ell=E(Y).
\]

给 geometric direction：

\[
d.
\]

lift：

\[
\delta\ell=L_Yd.
\]

定义：

\[
\boxed{
R_Y(\alpha d)
=
D(
\operatorname{softmax}(\ell+\alpha\delta\ell),
b_\alpha
).
}
\]

固定 boundary 时：

\[
b_\alpha=b.
\]

要求验证：

\[
R_Y(0)=Y,
\]

\[
\left.
\frac{d}{d\alpha}
R_Y(\alpha d)
\right|_0
=d.
\]

Topology 来自 decoder，而不是 Euler vertex step。

---

# 25. Route II-J：四种优化方式公平比较

同一 target、同一 decoder、同一 solver budget。

### O1 — TutteNet-style positive edge weights
bounded sigmoid/raw Laplacian entries。

### O2 — row-softmax directed logits
消掉 row scale gauge。

### O3 — MVC canonical re-encoding + ordinary gradient
每次 decode 后：

\[
Y\to E(Y)
\]

重新 gauge-fix。

### O4 — MVC canonical + covariance lift/retraction
从 vertex/map direction构造 latent step。

统一报告：

- global solves to threshold；
- iterations；
- wall time；
- map error；
- \(\mu\) error；
- min det；
- min area；
- logit spread；
- \(C_i\) condition；
- solver condition。

---

# 26. Route II-K：first variation 不能直接作为最终 map

可以用：

\[
dY
\]

做 predictor。

但 accepted state 必须来自合法 decoder：

\[
Y_{\rm new}=D(\ell_{\rm new}).
\]

禁止：

\[
Y+\alpha dY
\]

直接作为最终结果并声称继承 Tutte theorem。

---

# 27. Route II-L：incremental solver acceleration

latent 小变后，不应默认“从头”求解。

比较：

### warm-start iterative

新系统：

\[
A'Y'=B'
\]

用旧 \(Y\) 做 initial guess。

### exact correction equation

\[
A'(Y'-Y)
=
(B'-B)-(A'-A)Y.
\]

### local low-rank update

单行/少量 row 改变时研究 Sherman–Morrison / Woodbury。

必须回答：

- 什么 update locality 下真有收益；
- 全局 CNN 更新时低秩是否失效；
- predictor + correction是否减少迭代；
- 是否能安全减少 full solves。

不得为了证明“incremental快”而只选择极端局部 case。

---

# 28. Route II-M：把 MVC 直接 develop 成 differentiable neural layer

MVC 不能只作为离线 encoder / analysis tool。

必须实现至少一个可直接插入 PyTorch computational graph 的 MVC-based layer。

优先实现两个 API。

## Layer M1 — MVC canonicalization layer

\[
\ell
\overset{D}{\longrightarrow}
Y
\overset{E_{\rm MVC}}{\longrightarrow}
\ell_c.
\]

即：

\[
\boxed{
P_{\rm MVC}=E_{\rm MVC}\circ D.
}
\]

用途：

- 将任意 positive Tutte representation重新 gauge-fix 到 canonical MVC representation；
- 让后续 optimization / recurrent update 在 canonical latent 上继续。

要求：

- one-ring cyclic order可预计算；
- MVC 公式用 Torch differentiable ops 实现；
- finite-difference gradient check；
- forward/backward timing；
- 检查：
  \[
  D(P_{\rm MVC}(\ell))\approx D(\ell).
  \]

## Layer M2 — geometric retraction/update layer

输入当前 valid map/latent 与 geometric direction：

\[
(Y,d)
\quad\text{or}\quad
(\ell,d).
\]

通过 covariance lift：

\[
d\to\delta\ell,
\]

再：

\[
\ell_{\rm new}
=
\ell_c+\alpha\delta\ell,
\]

最后 hard-valid decode：

\[
Y_{\rm new}
=
D(\ell_{\rm new}).
\]

目标：

\[
\boxed{
\text{把 first-variation / vertex update 变成始终经过合法 decoder 的更新层。}
}
\]

该 layer 必须支持 backward。

---

# 29. Route II-N：Mandatory MVC instance optimization

和 Route I 一样，先不用 CNN。

至少做两个 case：

## M-Opt-1 — supervised map fitting

给定 \(F^\star\)，直接优化：

- MVC canonical latent；
- 或 geometric update/retraction variables。

目标：

\[
\min \|D(\theta)-F^\star\|^2.
\]

## M-Opt-2 — 256² image-registration instance

固定同一个 Route I synthetic image pair。

直接优化 MVC/Tutte latent 或 geometric-update variables。

和 Route I raw Tutte parameterization在：

- 相同 target；
- 相同 control mesh；
- 相同 initialization；
- 相同 optimizer budget；

下比较：

```text
iterations
global_solves
wall_time
map/image error
mu error
min area
conditioning
```

必须回答：

\[
\boxed{
\text{MVC canonical/retraction 是否真的让 optimization loop 更容易收敛或更快？}
}
\]

如果没有改善，也必须明确报告。

---

# 30. Route II-O：Optional tiny-network test

只有 MVC instance optimization 已经稳定成功后，再尝试：

\[
(I_m,I_f)
\to
\text{MVC latent or geometric update}
\to
\text{MVC/Tutte neural layer}
\to
Y.
\]

这不是 mandatory closure，但若时间允许应完成一个最小 proof-of-integration。

---

# 31. Route II closure

必须有：

- prior art；
- canonical theorem；
- exact round-trip；
- differentiable MVC canonicalization layer；
- differentiable covariance-retraction/update layer；
- supervised map-fitting instance optimization；
- 256² image-registration instance optimization；
- raw Tutte vs MVC optimization comparison；
- incremental solver benchmark；
- independent checker。

大规模 latent-redundancy 量化不是 closure requirement。

输出：

```text
05_mvc_theory.md
06_mvc_optimization.md
07_mvc_application.md
08_mvc_checker.md
```

Checker通过才进入 Route III。

---

# ============================================================
# ROUTE III — PRIMAL–DUAL / RT / WHITNEY / HODGE
# ============================================================

# 32. Route III 必须回答的核心问题

过去这条路线不能再停在：

> “DEC 很 structure-preserving，RT 保 flux，很有希望。”

本阶段必须回答：

\[
\boxed{\text{它到底能给 hard-bijective neural decoder 带来什么具体能力？}}
\]

拆成两个互补角色：

### Role A — accurate structure-preserving reference / teacher

优先保持：

- conductivity energy；
- flux conservation；
- conjugacy；
- discrete exactness。

不强求 hard bijection。

### Role B — positive hard-bijective approximation

把：

\[
A(\mu)
\]

映射成：

\[
c_e>0
\]

的 planar conductance operator，再通过 Tutte/electrical decoder输出 hard-bijective map。

本 route 要量化：

\[
\boxed{\text{geometry accuracy vs positivity/topology}}
\]

而不是只列方法名。

---

# 33. Route III-A：targeted literature table

至少定位：

- Hersonsky mixed graph rectangle tiling；
- Mercat discrete Riemann surface/Hodge；
- DEC；
- Raviart–Thomas mixed FEM；
- FEEC exact sequence；
- Data-driven Exterior Calculus；
- Conditional Neural Whitney Forms；
- recent compatible-discretization neural methods；
- anisotropic Delaunay/discrete maximum principle；
- orthodiagonal rectangle-tiling convergence。

输出表格：

```text
work
unknowns
exact structure
learned quantity
linear system
backprop
guarantee
relevance
what it does NOT guarantee
```

不要写长泛泛综述。

---

# 34. Route III-B：先证明一个 fundamental limitation

固定有限 directions：

\[
e_1,\dots,e_m.
\]

positive scalar conductance tensor cone：

\[
\mathcal C
=
\left\{
\sum_{k=1}^m
c_k e_ke_k^T:
c_k\ge0
\right\}.
\]

它是 finitely generated polyhedral cone。

二维 PSD cone：

\[
\mathbb S_+^2
\]

不是有限生成 polyhedral cone。

因此必须严谨证明：

\[
\boxed{
\text{固定有限 directions + positive scalar conductances 不能 exact 覆盖全部 SPD orientations。}
}
\]

Checker 要确认 statement 的精确版本。

意义：

> 终止“不断加几个 fixed directions 直到 exact arbitrary \(A\)”的无效探索。

之后只有：

1. approximation；
2. adaptive directions/connectivity；
3. full/block Hodge。

---

# 35. Route III-C：实现一个 full structure-preserving reference

至少实现一个，不允许都停在 note。

优先：

## Candidate 1 — Whitney/DEC full Hodge

构造：

\[
L_A=B_0^TH_AB_0
\]

其中：

\[
H_A
\]

来自 local tensor \(A_T\) 的 edge-form mass/Hodge。

要求：

- SPD；
- fixed topology；
- energy consistency；
- differentiable；
- matrix-free potential。

若这一实现无法正确回答 flux问题：

## Candidate 2 — RT0 mixed reference

\[
q_h\in RT_0
\]

显式保证 normal flux continuity。

至少有一个完整 prototype。

---

# 36. Route III-D：量化 discrete conjugacy 改善

对于 structure-preserving reference，测：

- flux balance；
- edge/dual exactness；
- stream integration residual；
- energy consistency；
- manufactured continuum convergence。

与 ordinary P1 MBM 对比：

\[
\|\nabla v-JA\nabla u\|.
\]

目标：

> 用数字说明 compatible discretization到底保住了什么，而不是只说“structure-preserving”。

---

# 37. Route III-E：positive conductance projection

在固定 planar control graph 上：

给 target：

\[
A_T^\star
\]

求：

\[
c_e>0
\]

使 discrete energy/operator尽量接近 target。

至少比较：

### deterministic local fitting / NNLS

### learned local map

小 MLP：

\[
\operatorname{features}(A_T)
\to
s_e,
\]

\[
c_e=\epsilon+\operatorname{softplus}(s_e).
\]

loss可包括：

- tensor mismatch；
- directional energy mismatch；
- teacher operator mismatch。

因为输出 positive，global Laplacian保持 Tutte sign structure。

---

# 38. Route III-F：planarity 是硬条件

引入 wide stencil时必须检查：

- graph edges是否 crossing；
- 是否对应 actual planar triangulation/cell complex；
- boundary cycle；
- embedding theorem assumptions。

不能：

\[
\text{positive weights}
\]

就直接套 Tutte theorem。

允许研究 enriched fixed planar mesh：

- cell centers；
- diamond refinement；
- additional noncrossing edges。

---

# 39. Route III-G：positive fixed-planar coverage study

在 Beltrami disk采样：

\[
|\mu|
\in
\{0,0.2,0.4,0.6,0.8,0.9\},
\]

\[
\arg\mu
\in[0,2\pi).
\]

转：

\[
A(\mu).
\]

对：

1. standard triangular grid；
2. center-split enriched planar grid；
3. 一个额外合理 planar refinement；

测：

- best positive tensor approximation error；
- worst orientation；
- distortion threshold；
- operator error；
- resulting map \(\mu\) error。

必须得到一个真正的：

\[
\boxed{\text{fixed-planar positive approximation frontier}}
\]

而不是几个随机例子。

---

# 40. Route III-H：hard-bijective hybrid decoder

至少做一个 end-to-end prototype：

\[
\boxed{
\mu/w
\to
A(\mu)
\to
c_e(A)>0
\to
L(c)
\to
\text{Tutte/harmonic map}
}
\]

要求：

- fixed planar graph；
- hard-valid rectangle boundary；
- positive conductances；
- 适用 theorem 下 PL injectivity。

这就是 primal-dual/Hodge 路线最实际的 neural hope。

---

# 41. Route III-I：full-Hodge teacher + positive student

比较：

### full compatible reference

更准确表达：

\[
A(\mu)
\]

但未必 hard injective。

### positive projected decoder

hard-bijective，

但：

\[
A_h\approx A.
\]

可以训练：

\[
\mathcal L
=
\mathcal L_{\rm operator}
+
\lambda_\mu\mathcal L_\mu
+
\lambda_E\mathcal L_{\rm energy}.
\]

把 primal-dual 变成：

\[
\boxed{\text{structure-preserving teacher + hard-valid student}}
\]

而不是抽象理论。

---

# 42. Route III-J：autograd / matrix-free

固定 topology：

\[
L(c)x
=
B_0^T\operatorname{diag}(c)B_0x.
\]

forward：

- CG（SPD）；
- warm-start；
- preconditioner。

backward：

同类 adjoint solve。

edge gradient：

\[
\boxed{
\frac{\partial\mathcal L}{\partial c_e}
=
-(B_0\lambda)_e(B_0x)_e
+\text{direct terms}.
}
\]

必须 finite difference 验证。

---

# 43. Route III-K：256² realistic synthetic experiment

同 control mesh、同 image pair 比较：

### T1
Route I direct Tutte latent。

### P1
QC-informed positive-Hodge/Tutte。

若 full-Hodge reference可运行：

### P-ref
structure-preserving teacher/reference。

比较：

- map/image accuracy；
- \(\mu\) fidelity；
- topology；
- forward/backward；
- memory；
- latent dimension；
- solve iterations。

---

# 44. Route III closure

不能以“有希望”结束。

必须得到以下至少一个：

### Outcome A
QC-informed positive Hodge/Tutte neural layer：可运行、可反传、hard-bijective，并有量化 accuracy。

### Outcome B
明确 approximation frontier：fixed planar positive graph在何种 anisotropy下无法准确表示 \(A\)，并实现 full-Hodge teacher + positive student。

### Outcome C
真正 structure-preserving RT/Whitney prototype，显著改善 conjugacy，并明确 topology 还缺什么 theorem。

无论哪个，都必须有 code + synthetic experiment + checker。

输出：

```text
09_primal_dual_theory.md
10_primal_dual_prototype.md
11_primal_dual_neural.md
12_primal_dual_checker.md
```


# ============================================================
# CROSS-ROUTE / EXECUTION RULES
# ============================================================

# 45. Common-input final comparison

只比较真正 surviving implementations。

至少：

### T1
Fast directed Tutte。

### T2
MVC/retraction-enhanced Tutte optimization。

### P1
QC-informed positive Hodge/Tutte。

若 full-Hodge/RT reference 可运行：

### P-ref
structure-preserving reference。

必须使用同样的：

- ground truth；
- images；
- control mesh；
- image resolution；
- hardware；
- stopping threshold。

禁止各 route 选自己的最好案例。

---

# 46. 最终最重要的不是单次 solve time

最终报告必须包含：

\[
\boxed{\text{time-to-useful-solution}}
\]

而不仅是：

\[
\text{one forward solve time}.
\]

包括：

- network/encoder；
- solver；
- dense warp；
- backward；
- number of global solves；
- optimization/training convergence；
- conditioning failures。

一个 solve 快但需要 1000 次更新，不一定更好。

---

# 47. Scaling-and-squaring 暂时只做 deferred reference

不扩展成第四 route。

理由：

- continuous diffeo exponentiation有价值；
- 但 fixed-grid sampled PL map不自动继承 hard no-fold guarantee；
- deformation field 的重新插值会改变几何映射。

若三个主路线明确需要，可做 baseline，但不得挤占 Route III 时间。

---

# 48. Remote compute

三台 remote 主机仍可使用。

规则：

- tiny correctness本地；
- 256²/512² training/benchmark优先 remote GPU/CPU；
- VPN/SSH 失败：
  - 快速重试最多两次；
  - 不暂停目标模式；
  - 转本地/theory/code；
  - 45–90 分钟后再检查；
- 长任务用 tmux/screen/nohup；
- 不高频 polling；
- 不干扰已有进程、端口和服务。

---

# 49. Route-specific checker

## Tutte checker

检查：

- 官方 paper/code理解；
- solver backend真实设备和算法；
- direct/iterative fairness；
- gradient；
- theorem hypotheses；
- dense query预计算；
- hard topology；
- timing/memory。

## MVC checker

检查：

- canonical subset说法；
- MVC公式；
- one-ring assumptions；
- fiber dimension；
- Jacobian nullspace；
- covariance lift；
- MVC derivative vs lift；
- prior art；
- optimization fairness。

## Primal-dual checker

检查：

- incidence signs；
- \(B_1B_0=0\) vs conservation；
- Hodge SPD/positivity；
- finite-direction impossibility theorem；
- planarity；
- full-Hodge vs positive-Hodge；
- injectivity theorem；
- autograd；
- “conservation ≠ bijection”。

---

# 50. Anti-error rules

任何高风险词出现：

```text
unique
iff
necessary
sufficient
universal
exact
guaranteed
canonical
bijective
```

必须独立复核。

每个核心 numerical identity 至少两条独立路径验证。

例如：

- gradient：adjoint vs finite difference；
- solve：residual vs independent backend；
- MVC round-trip：reconstruction + barycentric residual；
- Hodge energy：local quadrature + global operator；
- topology：theorem assumptions + Jacobian/boundary check；
- timing：GPU synchronized timer + wall clock。

禁止用同一代码路径生成 expected 和 measured value。

---

# 51. Anti-bureaucracy

默认禁止新增：

- hashes/checksums；
- manifest；
- frozen contract；
- completion ledger；
- per-run artifact folders；
- speculative framework。

允许的 gate：

- route checker；
- real 18h finalization clock；
- destructive/security/release boundary。

不删除已有合理安全措施。

---

# 52. Skill policy

默认禁用：

```text
ars/experiment-agent
```

`academic-paper`：

T+17h 前禁用。

`academic-research-suite`：

只做 targeted prior-art/theorem。

`deep-research`：

仅在关键 prior-art普通搜索无法解决时。

`systematic-debugging`：

真实 software bug。

`verification-before-completion`：

每个 route closure 和 final review。

不要重复调用 writing-plans；本计划是 authoritative。

---

# 53. WORKLOG

每个重大 subtask开始前，最多写：

```text
UTC time:
Route:
Question:
Precise claim/hypothesis:
Smallest decisive test:
What falsifies it:
```

结束后：

```text
Result:
Checker needed:
Next:
```

WORKLOG 不写长篇总结。

时间必须来自真实 clock。

---

# 54. Common code organization

允许按需建立：

```text
src/qcopt/neural_bijection/
    tutte/
    mvc/
    primal_dual/
```

但不要先创建空壳。

共享：

- mesh；
- synthetic data；
- metrics；
- timing；
- dense warp；
- solver diagnostics。

实验脚本集中：

```text
experiments/phase5/
```

---

# 55. Final independent review

真实 elapsed \(\ge17h15m\) 后才能开始。

Final checker必须检查：

1. 哪个 route真正有 hard topology theorem；
2. 哪个只有 empirical no-flip；
3. 哪个 solver真正 GPU/batched；
4. 哪个只是 CPU reference；
5. Tutte/MVC prior art 是否 overclaim；
6. Primal-dual是否把 conservation误写成 bijection；
7. benchmark是否 common-input；
8. timing是否含 dense warp + backward；
9. memory是否真实测量；
10. 公式/代码是否一致；
11. 是否混淆 expressivity 和 optimization；
12. 是否混淆 directed weights 与 symmetric conductances。

输出：

```text
14_final_independent_review.md
```

---

# 56. T+18h final decision

wall-clock gate通过后才写：

```text
15_final_decision.md
```

必须回答：

## A. Fastest practical hard-bijective layer
当前工程上最可行的是谁？

## B. Best optimization parameterization
raw weights / row-softmax / MVC canonical / covariance retraction，哪个在相同预算下最好？

## C. Best QC-informed route
Primal-dual/Hodge最终最适合成为：

- direct decoder；
- hard-valid approximate decoder；
- teacher；
- preconditioner；
- initialization；

中的哪一种？

## D. Main bottleneck
到底是：

- solver；
- conditioning；
- expressivity；
- topology；
- anisotropic approximation；
- dense warp；
- network prediction？

## E. 下一轮任务
只允许 3 个。

---

# 57. 本阶段成功标准

不是要求三个路线都变成论文级成功。

而是三个都必须深入到足以做技术判断。

## Tutte 至少：
- 官方机制复现；
- fast backend；
- correct implicit gradient；
- 256² supervised map-fitting instance optimization；
- 256² image-registration instance optimization；
- hard topology；
- solver/memory benchmark。

## MVC 至少：
- canonical encode/decode；
- differentiable canonicalization layer；
- differentiable covariance-retraction/update layer；
- 256² instance optimization；
- raw Tutte vs MVC optimization comparison；
- incremental solver study；
- optional tiny-network integration if time permits。

## Primal-dual 至少：
- 一个真正 compatible reference operator；
- 一个 positive hard-valid approximation/hybrid 或明确 obstruction；
- autograd；
- 256² experiment或很强 negative result；
- 说明“这条路线究竟带来什么”。

**文献笔记本身不算路线完成。**

---

# 58. 最终研究原则

现在不再问：

> “哪个方法理论最漂亮？”

而问：

\[
\boxed{\text{哪个 representation 把合法性放在 latent / solver structure 本身？}}
\]

以及：

\[
\boxed{\text{它能否被快速求值和快速反传？}}
\]

三条路线分别解决：

### Tutte
\[
\boxed{\text{positive barycentric equilibrium 给 hard topology}}
\]

### MVC
\[
\boxed{\text{给冗余 Tutte latent 一个规范表示和更好的优化几何}}
\]

### Primal-dual/Hodge
\[
\boxed{\text{把 QC metric/conjugacy 信息重新注入 positive hard-valid operator}}
\]

理想最终方法很可能不是三选一，而是：

\[
\boxed{
\text{QC/metric latent}
\to
\text{geometry-informed positive weights}
\to
\text{fast Tutte/electrical implicit solve}
}
\]

并用：

\[
\boxed{\text{MVC/canonical geometry}}
\]

改善优化和训练。

---

# Appendix A — Root AGENTS.md

将下面内容合并到仓库根 `AGENTS.md`。

---

## Mission

This repository is in **Phase V sequential deep-research mode**.

Authoritative plan:

```text
docs/research_phase5/PLAN.md
```

Goal: develop a fast, memory-efficient, differentiable neural layer producing a guaranteed-bijective discrete deformation.

Routes must be explored **sequentially and deeply**:

1. Fast Tutte neural layer.
2. MVC canonical coordinates + geometric incremental optimization.
3. Primal-dual / RT / Whitney / Hodge.

Do not start the next route until the current route passes its independent checker.

## Real wall-clock requirement

This is an 18-hour real wall-clock phase.

Use:

```text
docs/research_phase5/START_TIME.json
tools/check_phase5_elapsed.py
```

Do not simulate `T+Nh` labels.

Do not mark complete before real elapsed time reaches 18 hours.

A successful result is not a stop criterion.

## Research depth

Before a core theorem, solver design, or experiment, state:

```text
Question:
Precise claim/hypothesis:
Assumptions:
What falsifies it:
Smallest decisive test:
```

Prefer:

1. precise formulation;
2. proof/counterexample;
3. independent check;
4. minimal decisive code;
5. realistic benchmark;
6. engineering optimization.

## Independent checker

Each route requires an independent checker.

High-risk words require review:

```text
unique
iff
necessary
sufficient
universal
exact
guaranteed
canonical
bijective
```

The builder may not be the sole reviewer.

## Model routing

Minimum baseline: GPT-5.6.

- coordinator: GPT-5.6 Sol high;
- core implementation: GPT-5.6 Sol high;
- utilities/tests: GPT-5.6 Sol medium;
- hard mathematics: GPT-6 high if available;
- independent theorem checker: GPT-6 high/xhigh if available;
- conceptual impasse after two attempts: highest available GPT-6 reasoning, otherwise GPT-5.6 Sol maximum effort.

Do not automatically fall below GPT-5.6.

## Route order

Route I Tutte must close before Route II MVC.

Route II MVC must close before Route III primal-dual.

No parallel route exploration.

Independent checker work for the current route is allowed.

## Experiments

Use one shared benchmark.

Mandatory realistic scale:

```text
256×256 image/query
```

512×512 preferred when resources allow.

Always distinguish control mesh resolution from dense image/query resolution.

Every route must execute code and produce numerical evidence; literature-only exploration is insufficient.

## Tutte rules

Audit TutteNet paper, official code, and `torch_sparse_solve`.

Implement and benchmark:

- reference direct solve;
- improved direct/factor reuse where possible;
- matrix-free iterative solve;
- custom implicit backward;
- fixed-query dense-warp precomputation.

Do not call CPU reference code production-ready.

## MVC rules

MVC is primarily a canonicalization / optimization-geometry route, not a separate hard decoder.

Distinguish:

```text
D(E(Y)) = Y
```

from the false general claim:

```text
E(D(p)) = p.
```

Analyze redundancy, decoder Jacobian nullspace, covariance lift, canonical re-encoding, trainability, and incremental correction.

Treat covariance lift as one latent gauge/right-inverse, not the derivative of MVC unless proved.

## Primal-dual rules

Do not stop at “DEC/RT is promising”.

Implement at least one concrete compatible reference operator and one positive hard-valid approximation/hybrid.

Do not equate:

- conservation with bijection;
- SPD with M-matrix;
- positive wide stencil with planar graph;
- fixed-direction failure with route failure.

Quantify anisotropy/accuracy vs positivity/topology.

## Anti-bureaucracy

Do not add by default:

- hashes/checksums;
- manifests;
- frozen contracts;
- completion ledgers;
- per-run artifact trees;
- speculative frameworks.

Allowed gates:

- route checker;
- real 18h finalization clock;
- destructive/security/release boundaries.

## Skills

- do not use `ars/experiment-agent`;
- do not use `academic-paper` before final synthesis;
- use research/deep-research only for targeted prior-art/theorem questions;
- use systematic debugging only for real code failures;
- use verification-before-completion at each route closure and final review.

## Remote compute

VPN failure does not pause the objective.

Retry briefly, then continue locally.

Use remote compute for meaningful medium/large runs.

Avoid high-frequency polling.

Never interfere with unrelated processes or ports.

## Scientific integrity

Always distinguish:

- theorem vs evidence;
- exact vs approximate;
- expressivity vs trainability;
- control-map bijection vs dense-query evaluation;
- continuous diffeo vs sampled PL homeomorphism;
- directed positive Tutte weights vs symmetric conductance family;
- discrete exactness/conservation vs spatial injectivity.

Do not use post-hoc fold repair as the primary topology guarantee.


