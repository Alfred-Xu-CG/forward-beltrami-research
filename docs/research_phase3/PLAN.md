# Codex 24 小时目标模式计划书 v3
## Phase III — Deep Verification, Structure-Preserving Discretization, and Hard-Bijective Neural Layer

> **这是给 Codex 直接执行的 master handoff。**
>
> 你收到本文件后，必须先把它保存到仓库中，再开始工作。
>
> 仓库：`Alfred-Xu-CG/forward-beltrami-research`

---

# 0. 收到本文件后的立即动作

## 0.1 保存 master plan

把本文件完整保存为：

```text
docs/research_phase3/PLAN.md
```

若目录不存在则创建。

从此刻起，`docs/research_phase3/PLAN.md` 是未来 24 小时目标模式的 authoritative plan。

`docs/research_phase2/` 视为 **90-minute reconnaissance**：可以复用其中代码和线索，但不得把其中的 `06_final_decision.md` 当成已经完成的结论。

## 0.2 更新根目录 `AGENTS.md`

把本文件最后的 **Appendix A — Root AGENTS.md** 合并进仓库根目录：

```text
AGENTS.md
```

要求：

- 保留现有仍必要的安全规则；
- 以本阶段研究优先级覆盖旧研究路线优先级；
- `AGENTS.md` 只保留工作原则，不复制整个 PLAN；
- 明确指向 `docs/research_phase3/PLAN.md`。

## 0.3 创建本阶段长期交付物

只保留以下长期文件：

```text
docs/research_phase3/
    PLAN.md
    00_correction_audit.md
    01_continuum_mbm_theorem.md
    02_discrete_conjugacy.md
    03_electrical_primal_dual.md
    04_mmatrix_delaunay.md
    05_tutte_neural_layer.md
    06_results.csv
    07_independent_checks.md
    08_final_decision.md
    WORKLOG.md
```

`WORKLOG.md` 只记录：时间、当前 research question、做了什么、下一步、是否触发 checker / model escalation。每次不超过几行。

禁止重新产生 route ledger、completion ledger、每实验一个 artifact 目录、hash manifest。

---

# 1. 为什么 Phase II 不能被视为“完成”

上一轮 quick reconnaissance 方向基本正确，但结束过早，并出现了多个可避免错误：

1. electrical rectangle 的 modulus/energy bookkeeping 出现 boundary/off-by-one 错误；
2. four-direction tensor decomposition 被错误称为 unique，实际有自由参数；
3. local anisotropic nonobtuse condition 被写成 global M-matrix 的必要条件，而它只是强充分条件；
4. Tutte theorem 的“strict convex boundary”表述与 rectangle side 上有 collinear subdivision vertices 的实现不完全一致；
5. rectangle boundary latent 文档声称可学习 aspect ratio，但实现中的 width/height 是非 learnable Python float；
6. final decision 一边声称 Tutte universality 已证明，一边又把“证明或证伪它”列为下一任务；
7. mixed-boundary monotonicity 仍是核心 open question，却没有触发更高强度推理和独立 checker；
8. primal-dual anisotropic route 只排除了一个固定 stencil，就过早降级整条路线。

这些错误的共同原因是：

\[
\boxed{
\text{推进速度过快}
+
\text{作者兼任审稿人}
+
\text{缺少独立交叉检查}
+
\text{toy self-consistency 被误当 theorem validation}
}
\]

本阶段必须修正工作方式。

---

# 2. 本阶段最高原则

终极目标仍然是：

\[
\boxed{
z\ \text{(learnable latent)}
\longrightarrow
f_h:\Omega\to\Omega'
}
\]

其中 \(f_h\) 是：

- guaranteed bijective / PL homeomorphism；
- differentiable；
- fast forward；
- fast backward；
- memory-efficient；
- high-resolution；
- geometry/QC accurate。

但本轮最高优先级变为：

\[
\boxed{\textbf{深度和正确性优先于快速“完成”}}
\]

研究优先级：

\[
\boxed{
\text{correct theorem/formulation}
>
\text{independent verification}
>
\text{counterexample search}
>
\text{minimal code}
>
\text{medium benchmark}
>
\text{large benchmark}
}
\]

---

# 3. 24 小时是研究预算，不是“最多 24 小时”

## 3.1 禁止提前结束

**Success criterion is NOT a stop criterion.**

除非满足以下之一，否则不得在 T+22h 之前写 `08_final_decision.md` 或宣布目标模式完成：

1. 用户明确要求停止；
2. 整个执行环境完全无法继续，且 local/theory/coding 也不能进行；
3. 本计划 mandatory closure questions 全部已有正式证明并通过独立 checker、明确 counterexample 并通过 checker、或被可靠 prior art 完全解决且实现/验证完成。

即使某个方向成功，剩余时间也必须用于 adversarial tests、independent derivation、theorem hypothesis audit、performance analysis、alternative discretization、neural-layer feasibility 和 checker review。

**不得因为已有“足够写总结的结果”而提前结束。**

## 3.2 最终综合只能在 T+22h 后开始

T+0 至 T+22h：

- 不允许写最终排名；
- 不允许创建 final decision；
- 不允许因为出现第一候选而停止其它 mandatory closure questions。

T+22h 至 T+24h：

- independent final checker；
- reconcile contradictions；
- final synthesis。

---

# 4. Mandatory Independent Checker Agent

这是本轮最重要的流程修改。

任何承重结论都不能由提出它的 agent 自己批准。

## 4.1 Builder / Checker / Adjudicator

### Builder
负责推导、查文献、实现、实验。

### Independent Checker
必须是独立 agent/context。

只收到：

- precise claim；
- assumptions；
- builder 的最终 derivation / code diff / experiment；
- 必要最小背景。

不要把 builder 的完整思维过程灌给 checker。

Checker 的目标是**主动找错**，不是确认作者。

### Adjudicator
只有 Builder 和 Checker 不一致时启用。

优先使用最强推理模型。

输出：哪一方正确、错在哪里、正确 formulation、是否需要新 experiment。

---

# 5. 哪些东西必须经过 checker

以下任何一项写进长期文档前必须独立检查：

1. 新 theorem / lemma；
2. “necessary / sufficient / iff / unique / exact / universal” 等词；
3. modulus / energy / flux identity；
4. edge/boundary count；
5. orientation/sign convention；
6. M-matrix / Delaunay condition；
7. global homeomorphism claim；
8. continuum-to-discrete implication；
9. implicit VJP 公式；
10. 一个 route 被证伪或降级的结论；
11. performance comparison 中决定候选排序的数字。

Checker 结果写入：

```text
docs/research_phase3/07_independent_checks.md
```

只记录 claim、checker finding、fix，不写长篇审稿报告。

---

# 6. Checker 必须主动检查的高风险模式

## 6.1 自由度错误

看到 unique decomposition / unique solution，先做 dimension count 与 rank count。

## 6.2 local vs global

看到 element nonobtuse、triangle positive、row positive，必须问它是 global theorem 的必要条件、充分条件，还是 local sufficient condition。

## 6.3 necessary vs sufficient

禁止从 \(A\Rightarrow B\) 写成 \(A\Leftrightarrow B\)。尤其检查：nonobtuse vs Delaunay、M-matrix vs injectivity、positive Jacobian vs homeomorphism、positive stencil vs arbitrary anisotropy。

## 6.4 circular validation

禁止用同一实现产生 expected value 和 measured value。

例如 rectangle tiling 至少同时验证：total boundary flux、graph energy、target rectangle area、independently reconstructed dual potential。

## 6.5 boundary/off-by-one

所有 rectangular/grid theorem 必须有 2×2、3×2、3×3 手工可算例子，核对 primal rows、dual cells、parallel paths、effective conductance、boundary edge contribution。

## 6.6 theorem hypothesis vs implementation

如果 theorem 要 strictly convex polygon、convex polygon with boundary subdivision、3-connected graph、simple boundary cycle，必须明确实现满足哪一个。

## 6.7 continuum vs fixed mesh

每个结论标记：continuum、fixed-P1、graph/tiling、refined PL、numerical approximation。

---

# 7. 模型与推理强度：整体提升深度

**最低模型基线：GPT-5.6。**

不得自动降级到低于 GPT-5.6 的模型。

若 GPT-6 在当前 Codex 环境可用，则按下列规则调用；若不可用，则使用 GPT-5.6 Sol 的最高可用 reasoning effort。

## 7.1 Coordinator

默认：**GPT-5.6 Sol — high**

职责：维持研究深度、控制 24h、检查 mandatory questions、强制 checker、不允许提前结束。

## 7.2 Hard mathematical builder

优先：**GPT-6 — high**

用于 continuum MBM theorem、mixed boundary monotonicity、discrete conjugacy、anisotropic primal-dual、Tutte theorem scope、anisotropic Delaunay。

若 GPT-6 不可用：**GPT-5.6 Sol — high / max available effort**。

## 7.3 Independent mathematical checker

优先：**GPT-6 — high/xhigh** 独立上下文。

若只有 GPT-5.6：Builder 和 Checker 使用独立 GPT-5.6 Sol high 上下文，Checker 采用 adversarial review prompt。

## 7.4 Code implementation

核心 sparse / DEC / adjoint：**GPT-5.6 Sol — high**。

routine utilities/tests：**GPT-5.6 Sol — medium**。

## 7.5 Code checker

关键 solver 实现使用第二个独立 **GPT-5.6 Sol — high**，检查 math-to-code mapping、indexing、boundary conditions、transpose/adjoint、factor-of-two、shape、circular tests。

## 7.6 Final synthesis

T+22h 后使用 **GPT-6 — high**（若可用），并再次过独立 checker。

---

# 8. Research escalation protocol

核心问题第一次失败：

1. 缩小 statement；
2. 找 special case；
3. 做 dimension/sign check；
4. 找最小 counterexample。

第二次独立失败：立即升级最高 reasoning，提供短 problem packet：exact statement、assumptions、两个失败 attempt、counterexample evidence、closest literature。

要求返回：proof、counterexample、corrected theorem、missing assumption、decisive next test。

**不得以“暂时没证明”作为结束理由。**

---

# 9. Mandatory closure questions

以下问题在 24h 内必须深入到“证明 / 反例 / 明确受限 theorem / 实现 feasibility”之一。

---

# 10. CQ1 — Continuum MBM 是否可以严格闭合？

研究：

\[
\nabla\cdot(A(\mu)\nabla u)=0
\]

with left/right Dirichlet and top/bottom zero flux，再定义：

\[
\nabla v=JA\nabla u.
\]

必须尝试证明：

> 对合适 quadrilateral domain、\(\|\mu\|_\infty<1\) 和适当 boundary regularity，mixed problem + stream conjugate 给出的 \(f=u+iv\) 就是 normalized QC homeomorphism onto \([0,1]\times[0,M_\mu]\)。

优先 proof route：

1. measurable Riemann mapping / normalized QC existence；
2. image quadrilateral 的 conformal rectangle uniformization；
3. conformal postcomposition 不改变 \(\mu\)；
4. real part 满足同一 scalar mixed BVP；
5. mixed BVP uniqueness；
6. stream conjugate uniqueness；
7. 得出当前 construction 等于 canonical QC rectangle map。

Checker 必须检查 domain regularity、trace、corner behavior、orientation、normalization、measurable vs smooth assumptions。

若需要更强 regularity，写最小安全 theorem，不要过度声称。

交付：`01_continuum_mbm_theorem.md`。

---

# 11. CQ2 — Modulus / energy / reciprocal problem 必须独立核验

验证：

\[
M_\mu
=
\int_\Omega \nabla u^\top A\nabla u\,dA
=
\text{total right-side flux}
\]

以及 complementary problem 的正确 tensor 和 reciprocal identity。

至少用：

1. constant real \(\mu\)；
2. constant complex \(\mu\) 的解析/高精度 reference；
3. smooth manufactured continuum case。

禁止只用同一个 FEM code 自洽验证 energy。

---

# 12. CQ3 — Exact P1 recovery theorem

严格研究：

> 如果已存在一个 continuous P1 rectangle homeomorphism \(f_h\)，逐面 exact BC 为 \(\mu_T\)，那么其 \(u_h\) 是否 exact 满足 mixed FEM weak system？

必须推 shared-edge tangent derivative continuity、\(A\nabla u\) normal flux continuity、natural boundary、corner/side、uniqueness。

然后区分：\(u_h\) exact recovery、\(v_h\) 是否由 independent complementary solve exact recovery。

至少做一个**非平凡 manufactured compatible P1 map**，不能只用 constant real \(\mu\)。

---

# 13. CQ4 — Structure-preserving discrete conjugacy

上一轮只做 independent complementary P1 solve；这不够。

本轮必须深入至少一个真正 structure-preserving route。

## Route D1 — primal-dual incidence/Hodge

使用离散 chain complex：\(d_0u\) 作为 primal edge 1-form，构造 \(j=*_{A}d_0u\) 作为 dual current，通过 conservation 在 simply connected dual complex 上积分得到 \(v^\ast\)。

必须明确 primal/dual incidence matrices、Hodge star、exactness、gauge、boundary。

## Route D2 — compatible FEM / de Rham

至少写 feasibility derivation：RT、Nédélec、DEC、mixed finite elements。

目标不是同时实现全部，而是选择一个最有希望的。

交付：`02_discrete_conjugacy.md`。

---

# 14. CQ5 — 修复并真正验证 electrical rectangle theorem

当前 `electrical_rectangle.py` 的 9×7 modulus 0.75 不能继续作为 theorem validation。

必须：

1. 手工推导 unit-conductance \(n_x\times n_y\) graph effective conductance；
2. 计算 total boundary flux；
3. 计算 graph energy；
4. 按 Hersonsky convention 重建 tiling；
5. 验证 rectangle area 与 graph energy；
6. 修正 boundary/dual-cell indexing；
7. 加 2×2、3×2、3×3 regression tests；
8. 扩展至少一个 nonuniform positive conductance planar grid；
9. 若可行，扩展一个不规则 planar quadrilateral graph。

所有 expected quantities 必须有独立计算路径。

交付：`03_electrical_primal_dual.md`。

---

# 15. CQ6 — Anisotropic primal-dual 不允许用一个 fixed stencil 就结束

上一轮 fixed axis/diagonal stencil 只能证明该固定 direction cone 不覆盖 arbitrary \(A(\mu)\)。

本轮至少深入比较：

1. adaptive rotated directions；
2. \(A\)-metric orthogonalization；
3. intrinsic / anisotropic Delaunay；
4. diagonal vs non-diagonal Hodge star；
5. DEC / compatible FEM。

如果无法得到完全 positive diagonal Hodge，必须回答：是 representation 本质 obstruction，还是 fixed graph obstruction？

four-direction decomposition 要写成完整一参数 family，不得再称 unique。

---

# 16. CQ7 — M-matrix / anisotropic Delaunay 从 local 推进到 edge/global

必须纠正“每个 transformed triangle nonobtuse 是 global M-matrix 必要条件”。

研究 interior edge：

\[
K_{ij}=K_{ij}^{T_1}+K_{ij}^{T_2}.
\]

查并推导 anisotropic Delaunay-type condition。

目标区分：

- local nonobtuse：strong sufficient；
- edge-based anisotropic Delaunay：weaker sufficient；
- global M-matrix：actual assembled condition。

在 regular image triangulation 上研究 diagonal choice、local edge flips、\(A(\mu)\) orientation、distortion magnitude。

交付：`04_mmatrix_delaunay.md`。

---

# 17. CQ8 — Mixed-boundary monotonicity不能继续靠随机搜索

上一轮“10,000 random samples no counterexample”不算 close。

必须做：

1. targeted theorem search：planar resistor networks、discrete harmonic measure、response matrices、total positivity、maximum principle / boundary variation；
2. proof attempt；
3. exhaustive tiny graph search；
4. continuous theorem route via canonical QC map；
5. 明确区分 continuum monotonicity 与 discrete independent-complementary monotonicity。

若最后仍 open，要明确是哪一层 open。

---

# 18. CQ9 — Directed Tutte theorem scope与 prior-art audit

检查：convex polygon boundary、boundary subdivisions/collinear side vertices、strictly convex vs convex、3-connected requirement、triangulated disk 条件、directed nonsymmetric weights。

把当前 theorem 改成与 rectangle boundary latent 一致的最弱准确版本。

“universality”只有在 prior-art 和 theorem scope 确认后才能使用。

---

# 19. CQ10 — Tutte 不只证明 topology，要验证 QC expressivity

给定已知 valid QC / PL target \(F^\star\)，优化 directed Tutte logits，使：

\[
\mu(F_\theta)\approx\mu(F^\star)
\]

或 \(F_\theta\approx F^\star\)。

至少包括 smooth low distortion、twist/shear、larger but valid distortion、varying boundary sampling。

报告 map error、\(\mu\) error、min det、condition number/solve residual、forward/backward time、gradient quality。

若表示 theorem 成立，这里重点不是“能不能表示”，而是训练条件数、优化 landscape 和参数化实用性。

---

# 20. CQ11 — Tutte boundary latent必须支持 learnable modulus

当前 width/height 是 Python float，不是 neural variable。

实现至少一个：

\[
H=\epsilon+\operatorname{softplus}(m)
\]

或由 QC modulus / MBM modulus 提供 \(H\)。

要求 gradient 对 \(m\) 有效、rectangle closure exact、side order hard guarantee，并与 interior Tutte implicit VJP 组合。

---

# 21. CQ12 — Neural production feasibility必须实际测

本轮最后 3–4 小时比较至少两个 surviving candidates：

### Candidate A
Directed Tutte + learnable modulus/boundary latent

### Candidate B
MBM / structure-preserving QC decoder

同 resolution、同 machine 测：forward、backward、peak memory、sparse setup、solve residual、min det、map/\(\mu\) accuracy、batch limitation、CPU↔GPU copy、factorization reuse possibility。

SciPy CPU prototype可做 reference，但不得描述成 production-ready。

---

# 22. Deep-thinking protocol

每个核心 task 开始前，Builder 在 `WORKLOG.md` 写短 research card：

```text
Question:
Exact claim:
Known assumptions:
What would falsify it:
Smallest decisive test:
Relevant prior work:
```

不超过 10 行。

目的不是流程化，而是阻止没想清楚就写代码。

---

# 23. Proof protocol

任何 theorem claim 必须包含：

1. precise domain；
2. exact assumptions；
3. exact conclusion；
4. proof structure；
5. one sanity case；
6. one potential failure case；
7. prior-art comparison；
8. independent checker result。

禁止只写 roughly/generically/expected 然后标 proved。

---

# 24. Numerical validation protocol

每个核心 numerical identity 至少两条独立验证路径。

例如 modulus：flux、energy、geometry area。

VJP：analytic adjoint、finite difference、必要时 autograd reference。

Topology：theorem assumptions、face orientation、boundary order、degree/injectivity audit。

---

# 25. Remote 三主机与 VPN

- tiny 本地；
- medium/large 可 remote；
- SSH 失败最多重试 2 次，总等待约 2 分钟；
- 失败后立刻切 local/theory/coding；
- 不暂停目标模式；
- 45–90 分钟后再检查；
- 长任务使用 `tmux`/`screen`/`nohup`；
- 不高频 polling；
- 不碰用户已有进程/端口。

---

# 26. Skill policy

默认禁用：

- `ars/experiment-agent`
- `academic-paper`（T+22h 前）

轻量使用：

- `academic-research-suite`：targeted theorem/prior-art
- `ars/deep-research`：核心 theorem 普通搜索解决不了时
- `systematic-debugging`：真实 software bug
- `verification-before-completion`：T+12h midpoint 和 T+22h final checker

不要为每个 hypothesis 重新调用 writing-plans / brainstorming。

---

# 27. Anti-bureaucracy 继续生效

默认不新增 hash、checksum、manifest、frozen contract、schema freeze、baseline framework、gate、artifact integrity system、per-run artifact folder。

只有明确失败场景且 Git/types/tests 不足时才允许。

不删除已有合理安全措施。

---

# 28. 代码与文档修改范围

允许：

- 修 Phase II 的明确错误；
- 新增 `src/qcopt/qc_rectangle/`；
- 新增必要 tests；
- 复用 legacy prototype。

不要：

- 大规模 refactor legacy repo；
- 重跑全部旧 artifacts；
- 更新旧 completion ledger；
- 花时间美化 README。

---

# 29. 推荐代码目录

按需建立：

```text
src/qcopt/qc_rectangle/
    tensor.py
    continuum_reference.py
    mixed_fem.py
    discrete_conjugate.py
    electrical.py
    hodge.py
    mmatrix.py
    tutte_layer.py
    implicit.py
```

不要预先造空文件。

---

# 30. 24 小时时间安排

## T+0–2h — Correction Audit

必须首先修/确认：electrical modulus/energy、four-direction non-uniqueness、local/global M-matrix wording、Tutte theorem scope、learnable aspect ratio mismatch、final-decision contradictions。

独立 checker 后才能继续。

输出 `00_correction_audit.md`。

## T+2–6h — Continuum MBM theorem

高推理模型完成 CQ1/CQ2，Builder + Independent Checker。

输出 `01_continuum_mbm_theorem.md`。

## T+6–11h — Discrete conjugacy

完成 CQ3 exact P1 recovery、CQ4 primal-dual/DEC/compatible construction。至少一个 prototype。

输出 `02_discrete_conjugacy.md`。

## T+11–14h — Electrical / primal-dual

完成 CQ5 corrected theorem reproduction、CQ6 anisotropic feasibility。

输出 `03_electrical_primal_dual.md`。

## T+14–17h — M-matrix / Delaunay / boundary

完成 CQ7、CQ8。

输出 `04_mmatrix_delaunay.md`。

## T+17–20h — Tutte neural layer

完成 CQ9、CQ10、CQ11。

输出 `05_tutte_neural_layer.md`。

## T+20–22h — Fair architecture benchmark

完成 CQ12，更新 `06_results.csv`。

## T+22–23h — Independent Final Review

Checker 不看 builder 的 final ranking。检查 theorem、formulas、code、results table、contradictions、overclaims。

输出 `07_independent_checks.md`。

## T+23–24h — Final Synthesis

只有现在才允许写 `08_final_decision.md`。

必须给 strongest theorem、strongest counterexample、first candidate、second candidate、genuinely open problems、speed/memory bottleneck、next 3 tasks only。

---

# 31. T+12h Mandatory Midpoint Review

无论进度如何，约 T+12h 做一次独立 review：

1. 有没有 theorem 被写得过强？
2. 有没有 numerical self-consistency 被误当 truth？
3. 有没有 route 因 toy negative result 被过早放弃？
4. 有没有最关键 open question 被绕开？
5. 是否需要 GPT-6 / max reasoning escalation？
6. 剩余 12h 如何重新分配？

只写 1–2 页，不停工太久。

---

# 32. 最终“完成”定义

24h 结束时可以说“本轮完成”，只意味着：

- 24 小时研究预算已使用；
- mandatory tasks 有清楚状态；
- checker 完成；
- final decision 完成。

不意味着最终科学问题完全解决。

**不得在 24h 前因为“已经有第一候选”而结束目标模式。**

---

# Appendix A — Root AGENTS.md

将以下内容合并到根目录 `AGENTS.md`。

## Mission

This repository is in **deep-verification research mode**.

Authoritative plan:

```text
docs/research_phase3/PLAN.md
```

The objective is a fast, memory-efficient, differentiable neural layer that maps a learnable latent variable to a **guaranteed-bijective discrete deformation**.

Legacy Phase II results are reconnaissance, not final conclusions.

## No early completion

Do not declare the 24-hour objective complete before T+22h unless the user explicitly stops it or no local/theoretical/coding work is possible.

A success criterion is not a stop criterion.

Final synthesis begins only after the independent final review.

## Mandatory independent checking

Critical claims require an independent checker agent. The author of a theorem/formula/result may not be its only reviewer.

Check especially: uniqueness; necessary vs sufficient; local vs global; boundary/off-by-one; sign/orientation; continuum vs fixed mesh; theorem hypotheses vs implementation; circular validation; modulus/energy/flux identities; VJP formulas.

Disagreements require adjudication.

## Research method

Prefer precise formulation, independent check, proof/counterexample, small decisive experiment, implementation, medium benchmark, large benchmark.

A toy negative result must not kill a broader route without checking whether it only invalidates one representation.

## Deep-thinking requirement

Before implementing a core idea, write a short research card with Question, Exact claim, Assumptions, What falsifies it, Smallest decisive test, Prior work.

Do not code a theorem you have not stated precisely.

## Model routing

Minimum baseline: **GPT-5.6**.

- coordinator: GPT-5.6 Sol high;
- core implementation: GPT-5.6 Sol high;
- utilities/tests: GPT-5.6 Sol medium;
- hard mathematics / route redesign: GPT-6 high if available;
- independent theorem checker: GPT-6 high/xhigh if available;
- conceptual impasse after two attempts: highest available GPT-6 reasoning, otherwise GPT-5.6 Sol maximum available effort.

Do not automatically fall back below GPT-5.6.

## Escalation

After two failed independent attempts on a core question, do not stop. Escalate with a concise problem packet and request proof, counterexample, corrected theorem, missing assumption, closest prior art, decisive next test.

## Skills

- Do not use `ars/experiment-agent`.
- Do not use `academic-paper` before T+22h.
- Use research/deep-research skills only for targeted theorem/prior-art.
- Use systematic debugging only for real software bugs.
- Use verification-before-completion at midpoint and final independent review.

## Anti-bureaucracy

By default, do not add checksums/hashes/manifests, frozen contracts/schema freezes, duplicate audit systems, per-run artifact folders, ordinary-research workflow gates, speculative infrastructure.

Use Git, types, normal tests, concise logs.

Do not remove existing safety mechanisms merely to simplify.

## Experiments

Every experiment must answer a named research question. Tiny first, medium after correctness, large only when justified.

Every core numerical identity needs an independent validation route.

Do not validate code against quantities derived by the same code path.

## Remote compute

VPN/SSH failure does not pause the objective. Retry briefly, then continue locally. Use remote resources for meaningful medium/large jobs. Avoid high-frequency polling.

## Scientific integrity

Always distinguish theorem vs evidence; continuum vs fixed mesh; local Jacobian vs global homeomorphism; exact BC reproduction vs approximation; one failed stencil vs failure of an entire route; sufficient vs necessary conditions.

Do not use post-hoc fold repair as the primary topology guarantee.
