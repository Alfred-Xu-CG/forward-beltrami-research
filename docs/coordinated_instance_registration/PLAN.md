# 下一轮 Goal：多尺度协同的严格保拓扑逐实例配准
## 从“安全但可能低效的更新”走向“能求到好配准的优化算法”

**交付对象：Codex。默认窗口：24 个真实小时；若用户在启动时指定其他时长，以最新时长为准。**

**仓库：** `Alfred-Xu-CG/forward-beltrami-research`。不要假设 `master` 就包含最新工作；先用当前 worktree、用户指定分支和必要的远端读取确认，禁止为追求“最新”而覆盖未提交修改。

**一份文件即可启动。** 将本文件保存到 `docs/coordinated_instance_registration/PLAN.md`；将末尾 Appendix A 合并到根目录 `AGENTS.md`。保留已有仍适用的安全规则，不覆盖用户的全局配置。

---

# Approved revisions — 2026-10-01

This section is authoritative over conflicting inherited wording below.
The user authorized actual execution for 24 hours, with possible later extension.
Active checkout: D:/QC_optimization/.worktrees/phase6-dense-homeomorphism-plan.
Branch: codex/coordinated-instance-registration, initial base 4ca9f09.
Window start 2026-10-01 11:26:23 UTC; deadline 2026-10-02 11:26:23 UTC.

1. Main line is multiscale coordinated feasible INSTANCE OPTIMIZATION. Compare radial
   latent and analytic feasible-step entry using identical exact four-corner constraints.
   For proposal p with slack s>0, alpha_max=min_{(Cp)_k<0} s_k/(-(Cp)_k), infinity if none.
   Choose alpha=min(alpha_trial,theta*alpha_max), 0<theta<1, and candidate=Y+alpha*p*e.
   Image-objective acceptance is separate. No inner geometry line search or QP is needed.
   Measure the extra margin restriction; radial surjectivity does not ensure good conditioning.
2. Normalize coefficient scales or calibrate physical proposal displacements per level.
   Equal raw Adam rates are not automatically fair: P_l^T aggregates different supports.
   A diagonal P_l^T W P_l preconditioner is optional, not a new required framework.
3. Run a small existing-objective versus RegWSI-style-objective comparison early, alongside
   geometry, because prior proxy/anatomy conflicts have already been reported. Every method
   within each geometry ablation shares objective/evidence/initialization/physical units.
4. Synthetic benefit without real benefit means "geometry benefit not yet transferred".
   It is NOT automatically an evidence bottleneck. First distinguish target-class mismatch,
   directions, initialization, boundary, image resolution, regularization and actual convergence.
5. Specify tissue/background masks, out-of-bounds queries and noncorresponding tissue.
   Do not lower loss by dropping difficult tissue or shrinking effective overlap. Preserve
   fixed eligible-case and required-landmark evaluation denominators. Reuse existing masks.
6. Predeclare dataset, metric/units, pair/specimen aggregation, tail/failure measure and budgets.
   ACROBAT 2023 official-compatible score is mean of per-pair 90th-percentile TRE.
   Missing hidden labels mean official competitiveness NOT TESTED. Existing untouched labelled
   independent specimens can be confirmation; new downloads are not mandatory.
   Thresholds 2x/25%/10%/5% are guides; small means alone do not establish noninferiority.
7. Local/global supports, normalization, limited extra directions, schedules, radial versus
   analytic-step are ordinary main-line variants. At most one DIFFERENT mechanism is active
   at once after diagnosis; sequential replacement is allowed.
8. Hard topology is a mandatory feasibility condition. q_ref means the reference determinant,
   twice signed triangle area, so identity q/q_ref=1. Declare interpolation and validate
   the actual exported map; nonaligned resampling does not inherit an old certificate.
9. Roles: coordinator integrates evidence and controls changed variables; geometry/optimizer
   builder owns operators/stages/scales/supports; application/baseline builder owns datasets,
   groups/common evidence/RegWSI/real experiments/performance; temporary checker independently
   derives/recomputes critical results read-only. At most two builders plus one checker,
   no recursive delegation or overlapping file edits. Use real runtime model selection.
10. All local code, logs, caches, outputs and reports stay on D:. Research SSH MUST use
    ClearAllForwardings=yes for each existing alias; preserve occupied port 18082 and others'
    jobs. Recheck GPU processes/capacity before a job. No installed skills are read/invoked/
    followed in this round, per the user; use direct code, derivations and primary sources.
11. Time tables are adaptable; don't abandon a decisive high-information test merely to meet
    an intermediate slot. Use actual elapsed time; no premature closure after a successful pilot.


---

# 0. 先读这一页：本轮到底要解决什么

> **给定真实病理图像对和共同的图像驱动初始化，在实际输出网格上始终保持严格数字拓扑，用尽可能少的计算优化出准确的非刚性对应。**

本轮优先实现 **per-case instance optimization**，不要求先训练新的形变网络。

- 主方法：**多尺度、单方向协同的可行实例优化**；比较显式径向参数化与解析可行步长。
- 对照：已有 F1/F2 的逐实例优化；可执行的 RegWSI/DeeperHistReg。
- 研究组织：几何能力测试与真实病理实验**并行**，互相决定下一步。
- 只允许一个由已观察瓶颈触发的替代分支，不恢复十几条平级路线。
- 神经网络暂时可以提供冻结特征、匹配或初始化；只有实例优化取得价值后，才讨论摊销、提案学习或少步蒸馏。

**本轮明确不以 inverse consistency、cycle consistency 或训练更大 CNN 为目标。**
它们不是“不重要”，而是在当前阶段没有证据表明应优先于正确对应、协同更新能力和时间—准确性。

最重要的判断顺序是：

1. 在已知合法目标上，新的更新是否真正比 F1/F2 更有效？
2. 在相同真实图像证据下，这个优势是否转化为更好的配准？
3. 若没有转化，是可行域、优化路径、对应信号，还是测量协议出了问题？
4. 下一步只修那个得到证据支持的瓶颈。

**不要承诺必然突破或 SOTA。不要把一个被证伪的实现等同于整条路线不可能。**

---

# 1. 信息来源与优先级

本计划区分三种依据：

- **[U] 用户提供的讨论与既有探索报告。**它们支持本轮转向多尺度协同实例优化、保留 F1/F2、减少低价值工程工作的决定。关于旧标本重复使用、proxy 与 anatomy 排序冲突等，先当作已报告线索；需要据其作新数值结论时，核对实际记录。
- **[L] 原始文献与官方工具说明。**见文末参考。它们支持既有原理、能力和 prior art，不证明我们的新算法已经有效。
- **[P] 本计划提出的算法、资源分配和通过标准。**均是待验证研发设计，不是文献定理，不可在报告中改写成“已知结论”。

当前对话的任务优先级覆盖旧研究路线的优先级，但不覆盖系统、开发者、安全、权限、数据许可等更高优先级约束。

本轮不要求重新审完全部旧 repo。最初只读与当前几何 primitive、实例优化、图像损失、初始化、地标评估、时间和数据分组直接相关的内容。重复的报告不再次整篇加载。

---

# 2. 最小启动动作：不要让前置检查吞掉研发

在最初约 30 分钟内完成可执行的工作，而不是只写计划。

1. 确认 worktree、分支和现有未提交修改；选一个隔离的新工作分支，不重写历史。
2. 保存本 PLAN，并合并 Appendix A 的短规则。若有 nested `AGENTS.md` / `AGENTS.override.md`，只检查实际涉及路径的有效规则，避免旧指令偷偷恢复已冻结任务。[O3]
3. 确认本地可跑的最小 PyTorch/NumPy 测试、现有 F1/F2 evaluator、现有 RegWSI/DeeperHistReg 入口、已有真实图像与标签路径。环境缺项不要求一次补齐。
4. 将已有“反复看过标签”的病例标为 **development-only**。同时安排可用的新 specimen/patient 分组用于一次确认；无法得到新组时如实报告，不能停止所有开发，也不能虚构独立泛化。
5. 启动一个已有真实配准对的 affine + 已有非刚性 baseline；与新几何原型并行。
6. 只验证一次实际模型路由。模型不可用不反复猜 API 名，也不把文字角色扮演当真实独立 agent。

**不要在这里重新安装整套工具、重构整个项目、重跑数百个旧测试、制作哈希清单。**

---

# 3. 精确问题与表示：本轮不允许再混淆

## 3.1 地图和边界

参考矩形规则网格顶点为 \(X_i\)，输出顶点为 \(Y_i\)。完整 fixed→moving 采样映射记为

\[
F_Y(x)=A\bigl(\mathcal I_hY(x)\bigr),\qquad \det DA>0.
\]

\(A\) 是冻结的图像估计初始仿射；\(\mathcal I_h\) 是**明确指定**的 P1 或 Q1 地图插值。

第一组公平消融沿用当前可运行表示，不同时重写所有 P1/Q1 图像采样器。每个结果必须写明表示。若用 P1 采样损失，不能把 Q1 图像损失当成相同实验。

边界先统一使用共同初始仿射下的固定 residual boundary；若真实残差或 target-capacity test 显示边界限制是瓶颈，才增加第二组 R2：四角固定、边上顶点只沿对应边严格有序地滑动。所有比较方法使用相同边界族，不能只给新方法更自由的边界。

## 3.2 数字拓扑空间

一个 cell 的目标顶点按 SW、SE、NE、NW 为 \(a,b,c,d\)。要求

\[
\begin{aligned}
q_1&=\det(b-a,d-a)>0,\\
q_2&=\det(b-a,c-b)>0,\\
q_3&=\det(c-d,c-b)>0,\\
q_4&=\det(c-d,d-a)>0.
\end{aligned}
\]

这些是两种对角线对应的全部角三角形符号。配合共享边与正确边界，构成比单一 P1 剖分更强的要求。[L1,L6]

可用归一化 Jacobian \(q_k/A_k^{ref}\) 做诊断。若施加绝对下界 \(\epsilon_k=\eta A_k^{ref}\)，它是**额外几何限制**，必须在方法间一致，并披露被排除的合法压缩。

**四角正性并不表示 P1 与 Q1 是同一函数。**它可以保证各自解释的合法性，但查询、损失、精确细化、导出必须与声明的解释一致。

## 3.3 本轮主问题

\[
\min_{Y\in\mathcal H_h^{digital}} E(Y;I_f,I_m,A,\text{matches}).
\]

目标是优化得到好地图，不是只输出一张合法地图。

\[
\text{no folding}\ne\text{good correspondence};\qquad
\text{lower proxy}\ne\text{lower anatomical error}.
\]

本轮不优化网络参数 \(\theta\) 作为默认主任务。优化变量是每个病例自身的尺度系数、方向幅度或安全提案。

---

# 4. 主算法：单向联合几何约束 + 多尺度实例优化

## 4.1 给定方向，面积约束对全部幅度精确线性

当前顶点为 \(Y\)，固定非零方向 \(e\)（默认单位向量），同时更新

\[
Y_i^+=Y_i+u_i e.
\]

对候选角三角形 \(t=(i,j,k)\)，令 \(E_j=Y_j-Y_i\)、\(E_k=Y_k-Y_i\)。则

\[
q_t(u)=q_t(0)
 +(u_j-u_i)\det(e,E_k)
 +(u_k-u_i)\det(E_j,e).
\]

没有二次项，因为 \(\det(e,e)=0\)。将全部四角和适用的边界间隔写成

\[
s+Cu>0,\qquad s_k=q_k(0)-\epsilon_k>0.
\]

这里 \(C\) 是局部稀疏算子，不必显式组装大矩阵。

**关键假设：**同一个联合子步内，共享约束的顶点必须沿同一方向。不能令每点有任意不同方向，仍然使用这个线性公式。

默认首先只用 horizontal / vertical 两方向。斜方向仅在明确的 directional limitation 证据后添加，不先优化角度网络。

边界规则：horizontal 子步可允许上下边切向幅度，左右边和四角固定；vertical 子步相反。斜方向默认所有矩形边界点固定，除非另外推导了与侧边约束相容的构造。

## 4.2 显式径向参数化：不用内层 QP

对于满足边界线性等式的任意原始幅度 \(z\)，定义

\[
g(z)=\max\left(0,\max_k\frac{-(Cz)_k}{s_k}\right),\qquad
u(z)=\frac{z}{1+g(z)}.
\]

由 \((Cz)_k\ge-g(z)s_k\)，有

\[
s_k+(Cu)_k\ge \frac{s_k}{1+g(z)}>0.
\]

给任意严格可行 \(u\)，\(g(u)<1\)，且

\[
z=\frac{u}{1-g(u)}
\]

可精确恢复它。这是该**固定当前几何、固定方向、固定边界子空间**的可行域内部满射，不是任意二维地图的一层 universality theorem。

它属于凸域径向参数化家族，应与 RAYEN 等 prior art 正确区分；不要把分母本身作为独立 novelty。[L2]

需要检查：零输入保持零、正齐次性、几何依赖、数值可行性、接近边界的梯度衰减，以及输入目标不在该切片时的局限。

## 4.3 最小多尺度结构

使用

\[
z=P_\ell c_\ell,
\]

其中 \(P_\ell\) 把粗系数插值为**最细输出网格**上的幅度场。它插值的是尚未接受的提案，不是把一张已经宣称合法的地图随意重新采样。

第一版使用已有固定层次：例如 \(17,33,65,129,257\)，实际以当前输入大小和可用实现为准。不要求最终像素格点总是 \(2^k+1\)，但最终坐标约定必须一致。

先实现一个简单可重复的“粗→中→细→中→粗”schedule；不要一开始训练调度策略。

- 每个子步先 horizontal，再 vertical；
- 保留 F1 作少量局部后平滑；
- F2 作一个可比较的 patch 校正，不预设它必然更好或更坏；
- 粗步也使用**最细网格全部相关约束**；
- 细层后允许回粗层，而不是只上采样一次就结束。

这借鉴 subspace correction 和多层配准思想，不应宣称经典 SPD multigrid 收敛率自动适用非凸图像目标。[L3,L4]

## 4.4 全局最坏约束仍可能限制进展

上述公式仍然是一次共享 scaling，不能把它描述成完全消除了 F2 的最坏单元问题。

默认只记录与诊断有关的少量量：

- 接受幅度／原始幅度比例；
- 决定最小余量的单元是否长期固定在少数位置；
- 限制来自哪一尺度和区域；
- 低尺度误差是否通过粗校正减少。

若几个薄单元连续阻止整图更新，先尝试**局部支持域 + 交错非重叠更新**，而不是直接把全局系数改为每点系数。

局部构造要求：每个并行支持域的**受影响 cell 集合**互不重叠；包括过渡带与patch边缘的全部四角约束。只保证活动顶点ID不同不够。重叠块顺序执行并刷新当前几何。

同一方向的多个支持域也可先合成一个幅度场，再对整体施加一个合法算子；不得先各自安全化后任意相加。

---

# 5. 实例优化循环：不能把 trial candidate 当作已接受状态不断累加

以下是本轮的默认执行语义，而非需要照抄的API：

```text
Y = shared valid initialization
for each outer cycle, until per-case compute budget is exhausted:
    for selected scale and direction:
        anchor = detach(Y)           # 本stage的已接受地图固定
        build C(anchor), s(anchor), P_scale
        initialize stage coefficients c = 0 or a justified warm start
        perform a small number of optimizer steps on c:
            z = P_scale(c)
            u = radial_safe(z; C, s)
            candidate = anchor + u * direction
            evaluate original moving evidence at A(map(candidate, x))
            optimize E(candidate), not a repeatedly rewarped raster
        compare candidate against anchor with the SAME full objective
        verify the actual candidate under the declared output checks
        accept useful feasible candidate; otherwise retain anchor
    if stagnating, diagnose once and change scale/support/proposal if justified
return best accepted state by a predeclared LABEL-FREE selection rule
```

重要细节：

1. 一个stage内每次计算都从同一个 `anchor` 出发；不可把每次Adam试探都累加到地图上，然后仍按旧C/s解释。
2. stage接受后，下一stage才更新C、s、当前图像残差。几何权重不允许长期使用identity的版本。
3. 优化历史不需要完整autograd图。当前是per-case solver，不是训练整个optimizer；detach已接受状态不意味着当前decoder不能反传。
4. 对新显式算子仍做局部VJP／directional finite difference；如果未来训练跨stage网络，才单独处理全链梯度、checkpoint或隐式方法。
5. 图像/描述子应从原始对应尺度的moving field通过当前完整地图查询。避免“上次warped图再warp一次”的累计模糊。
6. 正则化默认施加在**累计地图**上；只正则当前增量可能让总畸变失控。
7. 必要的image-objective acceptance/backtracking允许存在；它与几何安全证明不同。拓扑无需靠图像loss决定。
8. 若本轮新增外层迭代数较多，允许每case不同stage数，但不能靠test landmarks选最佳停止时刻。

复杂地图仍可存为同一顶点表。对于P1，当前网格上的增量可解释为精确复合；无需显式维护不断增长的独立规则地图因子。

**真正新增独立factor只作为后备**：必须先证明单表优化因几何条件而停滞，而非匹配错误；且保留factorization或重新认证最终导出。不得采样后默认安全。

---

# 6. 三个科学问题：每项实验必须归入其中一个

## R1 — 表示／可达性

能否构造latent准确产生一个预先已知、满足四角条件的目标？

这里允许用目标地图构造latent，但明确标记为 **capacity/oracle test**，不是image registration。

## R2 — 优化效率

从相同初始化出发，达到相同地图误差或同一图像目标，哪个更新机制耗时更少、需要更少顺序stage、梯度更健康？

满射或有限步存在性，不回答这个问题。

## R3 — 真实对应

固定图像证据与预算后，改良的优化是否真的降低解剖TRE及尾部错误？

只降低NCC/MIND/machine-match误差不回答这个问题。

**每次失败先定位R1/R2/R3，不允许直接得出“F1/F2很弱”或“只剩loss问题”。**

---

# 7. 几何与真实图像：两条小规模实验并行，不能二选一

## 7.1 小而有区分力的几何实验

最少包含：

1. 固定边界的宽区域单方向平移/剪切：验证联合运动优势和显式逆编码。
2. 局部旋转/弯曲：不能只挑新方法天然一层能解的目标。
3. 粗运动 + 细局部压缩：检验回到粗尺度是否有价值。

目标来自解析构造或独立可信reference。每个目标先检查它确实属于声明的digital空间；若不属于，不可用它证明合法decoder有缺陷。目标过滤规则必须在比较前固定并报告。

使用同一物理目标的 \(33^2,65^2,129^2,257^2\) 采样，必要时513²；不要把目标频率随网格增长而偷偷改变。

最低比较：

- 现有F1实例优化；
- F1/F2混合实例优化（使用合理调参，而不是故意弱化）；
- 新单向径向层与解析可行步长，单尺度；
- 新单向显式层，多尺度回访。

完整矩阵不必一次全部跑完：先33²/65²定位，随后只扩大有信息价值的对照。

记录物理尺度归一化的地图RMSE、可行性、达到目标误差的时间、stage数和峰值显存。至少一个目标做\(H^1\)/导数误差检查，防止只对少量地标好看，但不把导数指标扩成新优化目标。

## 7.2 尽早跑真实病理

第一对真实图像与已有基线必须在最初几个小时启动，不等待“理论完全漂亮”。

- 第一组：旧development病例，便于debug，但明确不blind。
- 第二组：可获得的新specimen/patient-disjoint确认病例，冻结配置后只做一次主评分。
- 同一组织不同stain方向不是独立患者；相关pair按物理组织分组。
- 如果独立数据获取受阻，继续development算法工作，并把泛化状态写成未测；不能伪装取得正式竞争结论。

**不要在本轮强行新建大型WSI数据工程项目。**优先已有合法本地文件与已能运行的baseline，必要的新数据获取并行且有时限。

---

# 8. 公平比较：先隔离几何更新，再谈完整pipeline

## 8.1 几何更新消融

所有方法共享：

- 同一个图像驱动初始仿射；
- 同一固定/移动图像、输入分辨率、坐标单位和有效区域；
- 同一特征、匹配点及置信度；
- 同一regularized objective与权重；
- 同一边界族、输出分辨率；
- 相近的调参和时间预算。

优先从 affine/identity residual开始；如果使用既有neural warm start，则所有对照都使用同一个warm start，分开报告其成本。

## 8.2 RegWSI/DeeperHistReg

RegWSI是方法，不是数据集。它结合学习特征初始化和逐实例非刚性优化，是合理的主要应用基线。[L5]

尽量复用已工作代码、特征、初值及目标设置。不能一边换loss、匹配、初始化、图像大小和输出空间，一边把差异归因于新几何层。

设置两类comparison：

1. **Common-evidence study**：用同一证据比较F1/F2和新协同优化；可加无硬约束位移优化只作诊断，不把它冒充合法候选。
2. **Native practical baseline**：按照合理可运行的原方法配置跑RegWSI/DeeperHistReg，保留它的原生pipeline优势，独立评价其输出拓扑与实际时间。

不强制为了完美统一objective而重写第三方全部工程。无法统一的差异列出来，结论限制到匹配的范围。

没有更新调研，不写“打败当前全领域SOTA”。这轮主要目标是有价值、可重复的同协议比较。

---

# 9. 目标函数与停止规则：不为次要指标增加新工程

默认使用一份已有强baseline所采用、已经能运行的目标：

\[
E(Y)=E_{image}(I_f,I_m\circ F_Y)
+\lambda E_{machine-match}(F_Y)
+\gamma E_{regularity}(Y).
\]

第一版不发明新特征或十项loss。尽早并行进行已有目标与RegWSI风格目标的小规模对照；各几何方法使用共同目标。

本轮不加入：inverse consistency loss、cycle loss、反向网络、multi-stain循环训练、拓扑损失罚项（拓扑已是硬可行条件）。

每case停止由：

- 时间／评价次数上限；
- 同一objective上的进展停滞；
- 一次有依据的尺度/支持域切换后仍停滞；

决定。它是计算停止准则，**不是证明配准已经正确**。

在development中可预先保存1/2/4/8等少量预算的轨迹，再离线评价TRE以诊断proxy问题；禁止在盲测时用TRE选最终迭代。盲测最终选择规则在查看标签前固定。

多尺度改变image objective时，跨层“进步”须用同一个参考评价尺度复算；不能拿低分辨率loss与高分辨率loss直接相减。

---

# 10. 最重要的新环节：瓶颈裁决，而不是持续“更努力”

每90–120分钟，或完成三项有信息量的实验后，Coordinator用不超过约10行作一次决定：

1. 我们刚改变了哪个科学判断？
2. 当前阻塞属于R1、R2、R3、运行环境还是测量协议？
3. 下面哪个最小测试能区分两个最可能解释？
4. 下一个阶段继续、改构造、换证据，还是暂停这个实现？

若没有新信息，不得用相邻学习率、更多轮数、更密网格填充下个两小时。

## 10.1 触发式决策表

| 观察 | 必须先区分 | 下一项最小测试 | 禁止的错误反应 |
|---|---|---|---|
| 合法目标无法拟合 | 不可表达还是latent条件差 | 显式逆编码/构造latent，与梯度优化分开 | 直接归咎于图像 |
| 单向目标能直编、但优化很慢 | 饱和、梯度缩放或搜索方向 | 近identity VJP、幅度sweep、同一physical step比较 | 无限制加深 |
| 全局g由少数坏cell控制 | 真实总畸变还是局部控制范围过大 | 局部支持域交错版本，与全局同方向比较 | 随便逐点scaling |
| 粗运动残差长期存在 | 局部传播还是目标不匹配 | coarse revisit vs same number fine passes | 单纯升级分辨率 |
| proxy持续下降，TRE变差 | 证据信号/regularization不匹配 | 同一安全solver下两种小规模evidence对照 | 继续只训/优化更久 |
| safe和unconstrained都同样错 | 可能不是topology瓶颈 | 已知target fitting；重新核对初始化/方向 | 换第三种安全公式 |
| 安全solver差，oracle能逼近 | 可能是强约束或optimization bottleneck | 同一梯度proposal的接受率与step cost | 宣称地图空间不够 |
| 新方法变快但TRE无增益 | 初始化或应用分辨率限制 | 同初值、真实高分辨ROI少量比较 | 报kernel快等于配准快 |
| 几何核只占总时间小部分 | feature/I/O才是瓶颈 | 一次profile并停止无关核优化 | 为1%成本花几小时 |
| 找不到新blind数据 | 证据不可用，不是模型能力问题 | 继续development＋准备一次确认 | 冒充SOTA或停工 |

## 10.2 每次转向的最小说明

在 `PROGRESS.md` 写四行即可：

```text
Observed failure:
Two plausible causes:
Smallest discriminating test:
Decision if result A / result B:
```

这不是一个新的gate系统，也不要求用户批准每个实验。

## 10.3 升级Astra的条件

- 两次**实质不同**的尝试仍不能解释同一核心失败；
- checker与builder在承重公式上冲突；
- 发现可能需要改变“固定网格/单方向/输出表示”的证据；
- 怀疑结果被数据或坐标假象解释。

给Astra一个短problem packet：准确命题、相关公式/最多必要代码、最小失败结果、两个候选解释、待做决定。要求可执行的区分实验，不要让它重新规划十条路线。

**最高模型不能补齐缺失数据；更高effort也不保证研究成功。**官方同样强调先检查信息与权限，不要把所有缺陷都归为推理强度不足。[O6]

---

# 11. 备选路线：只在诊断触发时打开一个

优先级顺序：

1. **局部化／分块单向显式层**：当全局最坏约束是主要瓶颈。
2. **改变尺度基或少量方向**：当粗运动／方向限制已被capacity实验支持。
3. **零保持的凸近端／中心化障碍子问题**：当径向条件数或提案投影质量明显限制，先只在小patch／粗层试，并测额外成本。
4. **有界畸变凸块或成熟连续流参考**：用于回答“是否存在明显更强的联合求解方向”，不自动替换数字输出保证。

Lipschitz residual保留为小型可靠对照，不把它的更强范数限制误称为必要的no-fold条件。

Stripe/Progressive层次构造、Yee/DEC、全局层次结构组合优化、独立factor系统暂列后续研究。只有主路线遭到明确结构性阻碍，而且它们直接针对该阻碍时，才动用预留预算，不多于一个分支。

默认约70%的**builder执行资源**用于主实现/对照/真实数据，约20%用于一个针对性替代，余下用于整理与复核。比例是防漂移指引，不是新的成本记账项目。

---

# 12. Inverse consistency：明确降级，不留模糊空间

本轮不做以下事项：

- 为提高round-trip分数训练第二个反向网络；
- 写通用逆P1/Q1查询器；
- 添加cycle consistency项再重新调所有权重；
- 为一个逆指标改变主representation；
- 以forward/reverse不互逆否定已经证明的单向同胚；
- 以inverse consistency较好替代真实对应准确性。

现有便宜cycle数字若顺手已有，可放附表，但**不是验收指标，不要求改善，不占独立研究预算**。

例外只有两个：

1. 发现fixed→moving/moving→fixed方向错误，需要一次最小诊断；
2. 正式数据集评分必须使用反向坐标，此时做最小正确转换以完成必要评价。

这两个例外不是打开inverse-consistency研究分支的许可。

同样降级：极致通用API、漂亮README、全局抽象、千次梯度probe、CPU/GPU全硬件矩阵、gigapixel图像导出美化、穷尽所有模型／loss超参。

---

# 13. Checker：检查会影响结论的东西，而不是创建第二个官僚系统

## 13.1 几何checker（Astra high，一次主证明＋必要修正）

独立推导：

- 四角面积与单向幅度的仿射关系；
- radial可行性与逆参数化的适用范围；
- 边界切向等式及顺序不等式；
- 并行支持域的受影响cell覆盖；
- P1/Q1函数及导出约定。

必须主动检查：方向是否真共享；\(s>0\)；近边界数值余量；没有把full-polytope surjectivity写成低维coarse space的surjectivity；未把planar morphing theorem搬成digital空间的定理。

## 13.2 实现checker（Sol high，读有限diff＋重跑最小例子）

不调用生产代码的同一个helper计算expected答案。重点检查：

- 2×2、3×3等可手算的四角与边界；
- 刻意会破坏另一条对角线的输入；
- float32输出实际坐标；
- 非正方形与边缘支持域；
- 一次新算子的directional finite difference；
- 阶段anchor是否被错误累加；
- 局部／完整地图的插值和affine顺序。

改动后跑受影响测试；不反复重跑全部legacy test suite。导出topology验证可复用已有经过独立核验的代码，不重建exact arithmetic基础设施。

## 13.3 科学checker（Sol high；最终决策由Astra high）

核对：

- 真实数据、患者/标本分组和标签暴露；
- 同初始化、同单位、同目标与预算是否真的成立；
- 失败和超时是否计入；
- 时间是否包含实际宣称的阶段；
- 是否拿teacher/合成capacity代替图像推断；
- 关键负结果是否在最终结论中保留。

输出只需要“claim—依据—问题—修正”，不写长篇审稿式ledger。Astra终审只看核心图表、代码片段和checker分歧，不灌入旧几百万token档案。

**两个agent赞同不构成证明。构造独立数值检查也不替代统计与实验设计。**

---

# 14. 模型和effort：只使用 GPT-6.1 Sol 与 GPT-6 Astra

本计划编写时官方模型ID分别是 `gpt-6.1-sol` 和 `gpt-6-astra`。两者的reasoning effort均支持low、medium、high、xhigh、max；API支持不等于当前账户的Codex客户端一定开放全部配置。[O1,O2]

| 工作 | 默认模型/effort | 何时升级 | 明确不做 |
|---|---|---|---|
| 总协调、每轮决策 | GPT-6.1 Sol / medium | 非平凡分歧升high | 全程Astra max |
| 文件定位、表格解析、已有记录提取 | GPT-6.1 Sol / low | 证据冲突升medium | 用高推理读重复日志 |
| 主要算法实现、实例优化脚本 | GPT-6.1 Sol / medium | 几何/autograd/并发难点升high | 每个函数都高强度 |
| 针对性文献核对 | GPT-6.1 Sol / medium | 改变novelty/正确性的争议交Astra high | 全面重复综述 |
| 新几何主证明及独立审查 | GPT-6 Astra / high | 具体未解争议才xhigh | 自动展开无关理论 |
| 数据/性能/代码checker | GPT-6.1 Sol / high | 关键科学冲突交Astra | 自动审计全部repo |
| 核心瓶颈裁决 | GPT-6 Astra / high | 一次有边界的xhigh重试 | 两个高模型无限往复 |
| 最终科学验收 | GPT-6 Astra / high | 必要时要求补一项测试 | 以文档完整替代结果 |

默认`standard`执行，不自动启用Pro/Fast/Ultrafast等额外消耗模式。max是罕见例外：必须说明它要解决哪个会改变主决策的问题。

同一时间最多两个builder＋一个临时checker；不递归spawn，不让多个agent修改同一文件。机械任务合并给现有agent，不为每个grep新开子代理。

每次委派给一个明确问题，附必要源代码与结果；结论通常不超过约800–1200词，完整证明或关键反例可例外。**不要求输出私有思维链**，只要可检查的推导、计算与结论。

## 14.1 实际配置，不是文字扮演

官方Codex支持在agent文件中指定 `model` 与 `model_reasoning_effort`。[O4]

若当前客户端支持，可用既有配置机制新增/复用少量角色。例如：

```toml
# 示例：.codex/agents/geometry_checker.toml
# 仅在当前安装版支持此schema时使用；不要覆盖整个已有配置。
name = "geometry_checker"
description = "Independently checks the new digital-topology update, not the entire repository."
model = "gpt-6-astra"
model_reasoning_effort = "high"
developer_instructions = """
Check the precise supplied claim independently. Return the derivation or smallest
counterexample. Separate fixed-P1, four-corner digital constraints, and exported
interpolation. Do not edit the builder's implementation or open unrelated routes.
"""
```

同理worker可用Sol medium，code/data checker可用Sol high。Coordinator保持Sol medium。

读取当前版本/配置一次并确认实际角色。不要因为prompt说“Astra checker”就声称已经调用Astra。若模型或subagent功能不可用：

- 在两种用户允许模型中选择可用者并报告；
- 无独立agent时可以做独立脚本验证和另一上下文复查，但不能声称完成真实独立agent审查；
- 不自购第三方API、不暴露key、不偷偷换到其他模型；
- 若两个允许模型都不可用，报告这一能力限制；不把重复连接失败变成研究工作。

---

# 15. 时间、token与远程算力

## 15.1 24小时是实际窗口，不是阶段名称

继承已有可靠的goal计时机制，只记录一次真实UTC start、deadline与actual end。若此前已有时间检查器则复用，不新增hash／不可改写contract体系。

- 不把active tool time、多个agent的时间和wall-clock相加；
- 不用“逻辑上完成T+24阶段”冒充经过24小时；
- 发现阶段性好结果，不结束整轮工作，转向反例、确认病例、预算对照；
- 不为了凑时长sleep、空poll或制造低价值benchmark；
- 若客户端／工具／额度停止，标`interrupted`，保留下一条可执行命令，不标`complete`。

**这个Markdown能规定行为，但不能替运行平台创造24小时执行能力。**最终报告区分“时间窗口结束”“计划完成程度”“科学目标是否达成”。

不因找不到新患者数据而暂停本地算法开发；也不因开发有进展而宣称独立应用验证完成。

## 15.2 时间分配建议

| 真实elapsed | 主工作 | 并行工作 |
|---|---|---|
| 0–1h | 最小state/角色/数据确认；已有baseline实际运行 | 单向约束短推导 |
| 1–4h | 单向显式prototype与checker | 第一组真实pair、共同loss/单位校验 |
| 4–9h | 几何capacity/效率与第一轮实例优化 | 查看真实失败模式，不等全部synthetic结束 |
| 9–15h | 多尺度回访、局部支持域（如触发）、matched F1/F2 | 既有强baseline同数据实际测量 |
| 15–20h | 冻结本轮survivor配置；确认病例 | 一个有依据的替代，或扩大有价值分辨率 |
| 20–24h | 必要复核、准确性—成本曲线、最终结论 | 未结束作业正常收尾，不加新路线 |

这是资源指引，不是阶段冒名。遇到关键反例可提前改变内容，不把实际时间改写。

文献/重复文档/环境工程合计不应挤掉主要算法和真实实验。每约两小时检查一次是否正在做与R1/R2/R3无关的工作；不另写计费程序。

## 15.3 三台remote与VPN

沿用用户已授权配置。只用空闲资源，不杀其他任务，不改VPN/共享端口（特别是18082），不改变系统权限以绕过连接/安全限制。

SSH失败：两次短重试、总等待不超过约两分钟；转本地小规模、理论或代码。约60分钟或自然阶段切换才再查。三台共享VPN时不要对每台重复高频探测。

长job可用已有tmux/nohup/调度器，只跟踪自己的PID；中断后先确认旧job状态，避免重复启动。不得把未知退出误记为实验成功。

将大任务拆成有信息价值的子批次，以便一个病例超时不占掉全部窗口。10分钟是初期单个pilot的默认软上限，不是人为阉割所有强baseline；若原方法合理配置需更久，可以明确投入，所有方法按完整成本比较。

---

# 16. 最小记录，而不是再次制造审计工程

长期文件仅建议：

```text
docs/coordinated_instance_registration/
    PLAN.md
    PROGRESS.md       # 当前判断、有限pivot记录、实际时间、数据与角色限制
    METHOD.md         # 存活算法的必要推导、伪代码、来源
    RESULTS.csv      # 一个表，逐病例/方法/预算结果
    REVIEW.md        # checker的实质发现，不是全repo ledger
    REPORT.md        # 最终判断与下一步
```

代码复用现有mesh/采样/topology工具。按需新增小模块，例如 `coordinated_update.py` 与一个实例优化入口；不要先铺一整套空目录或通用benchmark框架。

一个CSV行至少包含：method、case/group、development/confirmation、map representation、input/output resolution、common initialization、预算、实际stage数、主要误差、最小角Jacobian、invalid/failed状态、time、peak memory、结果文件路径。没有标签时填NA，不填0。

保存关键地图、baseline输出和必要优化曲线，以便独立重算；无需每一次无效超参都一个artifact文件夹。模型／原图可按原许可留在现有数据盘，不把大数据复制进repo。

允许正常Git commit id和版本记录；“不新增hash”不禁止Git自身标识，不禁止普通比较baseline。禁止的是为它们另外造一套完整性／冻结／门禁系统。

---

# 17. Skill政策（下列旧建议由 Approved revisions 第10条覆盖：本轮不使用已安装skills）

- 默认不调用可选 `ars/experiment-agent`：其受控复现约束若限制自主改脚本，会错配本轮任务。
- 默认不调用 `academic-paper`：只写工作note与必要文献归属，结果稳定后再写论文。
- `academic-research-suite` / `ars/deep-research` 只用于一个明确承重theorem/prior-art问题，不自动跑完整文献六阶段。
- `systematic-debugging` 只在真实软件故障时调用；数学反例不是software bug。
- 不为每个小实验重复 `brainstorming` / `writing-plans`；使用本计划的短瓶颈裁决。
- 保留最小skill routing；若更高优先级指令要求某skill或审批，必须遵守，不声称本文件能禁用平台强制规则。
- 不修改用户全局skill文件、审批策略或远端安全配置以节省时间。

---

# 18. 本轮明确的通过标准：不要用漂亮但不相关的结果“过关”

以下门槛是**本项目决策阈值，不是文献规定，也不是自动release gate**。在开发比较前选定主要误差/预算，不能看确认标签后改阈值。

## T — 几何正确

新算子完整保护全部四角与声明边界；actual output可验证；checker独立复算关键例子；新算子局部梯度通过合理容差测试。

一个合法identity或大量拒绝更新不能单独达到整个项目成功。

## O — 有效优化

在至少三类目标（包含非单向目标）上做同时间/同误差比较。一个有价值的正结果例如：达到同误差中位时间至少快2倍，或同预算地图误差改善至少25%，且没有通过更差畸变、改变目标或弱化对照获得。

这些只是筛选数值，不要求每例都达到。若达不到，必须解释是否只是单方向特殊例子有利、优化参数差、或新方法本身无优势。

## A — 真实应用进展

至少实际完成一组同初始化、同证据、同预算的真实病例比较，包含失败与困难pair。主要指标为规定坐标单位下的解剖TRE，辅以尾部和失败率；image proxy只作解释。

若新法比最强已有安全实例优化在development中提高约10%主要TRE或形成明显时间—准确性优势，并在新独立组确认相同方向，可视为值得继续的应用信号。小样本、无独立组或不确定性太大，只能称development evidence。

若误差在预先确定的非劣范围内（例如5%，须结合标注噪声）而速度显著更好并具有数字保证，也可能有价值。不要把这种Pareto改进改称accuracy SOTA。

## C — 竞争性与科学结论

- 与RegWSI/DeeperHistReg的同协议比较尚未完成，不能称competitive；
- 没有当前强对手／官方兼容独立评估，不能称SOTA；
- 一个漂亮定理、零folding、通过全部测试、工作满24小时，都不能代替A；
- 如果O显著改善但A不变，报告“几何收益尚未转化为应用收益”；原因经区分实验支持后再归因，不自动判为对应证据瓶颈；
- 若新法不如F1/F2，接受结果；将其作为反证淘汰该实现，不为了维护新idea反复调整评价。

---

# 19. 最终验收必须直接回答的九个问题

由Astra high对核心结果和checker发现进行一次聚焦验收：

1. 新方法究竟优化什么变量？每case是否真的独立优化？
2. 新的单向联合层相比F1/F2，有无可重复的time-to-accuracy优势？
3. 这种优势是否只存在于新方法可一层精确编码的特制shear？
4. 地图是否合法是对哪张网格、哪种插值、哪个实际导出文件说的？
5. 真实病理改善来自哪一项变化？有无相同初始化/证据的对照？
6. proxy改善与TRE改善是否一致？不一致时是否停止了无意义加深？
7. 数据有没有重复标本／标签暴露／只保留成功初始化的selection bias？
8. 时间是否计入实际宣称的工作，强baseline是否合理而没有被削弱？
9. 本轮还没有回答什么？下一轮最多三项任务是什么？

必须给出最强正结果和最强反结果。

**不能声称流程保证了科研成功或保证永不犯错。**这套流程的作用是让假设较早接受区分实验、让错误不轻易升级成结论，并把大部分资源投向真正影响结果的环节。

---

# 20. 必要参考：每篇只解决一个明确问题

## 数学与配准

**[L1]** Liu et al., *On Finite Difference Jacobian Computation in Deformable Image Registration*，arXiv:2212.06060 / IJCV。问题：二维数字四差分与四角几何的关系。不要只引用中心差分。
https://arxiv.org/abs/2212.06060

**[L2]** Tordesillas et al., *RAYEN: Imposition of Hard Convex Constraints on Neural Networks*。问题：凸可行域径向参数化的prior art；不能把我们的简单gauge公式包装成首次硬约束层。
https://arxiv.org/abs/2307.08336

**[L3]** J. Xu, *Iterative Methods by Space Decomposition and Subspace Correction*, SIAM Review, 1992。问题：局部／块／粗子空间的组织方式；其SPD理论不直接证明非凸配准收敛。
https://doi.org/10.1137/1034116

**[L4]** Haber & Modersitzki, *A Multilevel Method for Image Registration*, SIAM J. Sci. Comput., 2006。问题：多层continuation与线性子问题multigrid的区别。
https://doi.org/10.1137/040608106

**[L5]** RegWSI, *Whole Slide Image Registration using Combined Deep Feature- and Intensity-Based Methods*。问题：实际可执行的强初始化＋实例非刚性优化基线。
https://arxiv.org/abs/2404.13108

**[L6]** Lipman, *Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing*。问题：局部正向与边界条件如何推出全局性；不要泛化到任意重采样。
https://arxiv.org/abs/1310.0955

**[L7]** Alamdari et al., *How to Morph Planar Graph Drawings*。问题：单向运动的既有理论；一般planar结论不是少层digital-rectangle结论。
https://arxiv.org/abs/1606.00425

以上是定向阅读起点，不是要求本轮把所有证明重做。与本轮决策无关的文献分支立即停止扩展。

## 官方Codex与模型配置（本计划编写时核对）

**[O1]** GPT-6.1 Sol模型与effort：
https://developers.openai.com/api/docs/models/gpt-6.1-sol

**[O2]** GPT-6 Astra模型与effort：
https://developers.openai.com/api/docs/models/gpt-6-astra

**[O3]** AGENTS.md发现、层级与长度限制：
https://developers.openai.com/codex/guides/agents-md

**[O4]** Subagents与agent-level model/effort：
https://developers.openai.com/codex/subagents

**[O5]** 配置键与plan-mode effort，实际版本优先：
https://developers.openai.com/codex/config-reference

**[O6]** 使用量与reasoning level：缺信息／权限不是提高effort能解决的问题。
https://help.openai.com/en/articles/20001516-managing-usage-with-gpt-6-astra-in-work-and-codex

---

# Appendix A — 合并到根目录 AGENTS.md 的短规则

完整PLAN不放进AGENTS；只合并下文并链接本文件。不要复制多份互相冲突的长期规则。

# AGENTS.md — Coordinated Instance Registration

## Current objective

Read `docs/coordinated_instance_registration/PLAN.md` as the current project plan.
Legacy documents are evidence, not a queue of unfinished work.

The immediate objective is accurate, fast **per-case image registration** whose
actual exported map satisfies the declared digital-topology conditions. A new
neural network, inverse consistency, or a new infrastructure framework is NOT a
prerequisite for this phase.

## Priorities

1. Correct correspondence / anatomical accuracy on real images.
2. Hard topology of the declared output, including boundaries and interpolation.
3. Time-to-accuracy and usable memory.
4. Understand whether remaining error is representation, optimization, or evidence.

Start real-data execution alongside geometry work; do not postpone it until all
proofs or synthetic experiments are complete. Reuse available data and working
baselines. Do not silently treat previously viewed specimens as blind tests.

## Active work

Main: multiscale, explicit unidirectional coordinated instance updates.
Controls: existing F1/F2 instance optimization with the SAME input evidence,
initialization, objective, boundary class, and comparable compute budget.
Supporting baseline: an executable, correctly configured RegWSI/DeeperHistReg.
At most ONE alternative mechanism may be active when a diagnosed bottleneck
justifies it. F1/F2 remain useful local/block components, not mandatory winners.

## Non-goals

Do NOT optimize inverse/cycle consistency, undertake inverse-field generation,
train a new large network, build a new SLAM-like matcher, develop Stripe/
Progressive/Yee/DEC frameworks, or perform enormous-resolution demos by default.
Do not add more metrics just because they are customary. Existing cheap metrics
can be retained, but cannot create a new research branch.

## Anti-rabbit-hole rule

Before spending substantial time, answer in a few sentences:
What decision can this task change? What is the smallest discriminating test?
What result would redirect the work? If no answer, do not do the task.

After two materially different failed interventions on the same hypothesis,
diagnose and escalate; do not repeat nearby hyperparameters indefinitely.
Every 90–120 minutes, or after three informative experiments, make a brief
continue/change/pause decision. No meeting-style report is required.

## Models and delegation

Use only `gpt-6.1-sol` and `gpt-6-astra` when actually supported by this runtime.
Coordinator and routine implementation: Sol medium.
Mechanical extraction: Sol low. Complex implementation/debug: Sol high.
New load-bearing geometry and ambiguous scientific pivots: Astra high.
Use Astra xhigh only for a specific unresolved decision; max is exceptional.
Verify the actual role configuration once. Never pretend that a prompt switched
the model or created an independent agent. Do not silently use another family.

At most two builders plus one temporary checker. No recursive delegation.
Give each worker a bounded question and relevant files, not the full archive.
Checkers do not approve their own code or merely repeat the author's conclusion.

## Checking that matters

Independently check new topology formulas, boundary/seam coverage, gradients of
new operators, map direction and coordinate units, and final empirical claims.
Run focused tests after edits and a final affected-suite check; do not repeatedly
run the entire legacy suite. Agreement between agents is not a proof.

The production map and evaluator must specify P1 or Q1. Four-corner positivity
supports both interpretations, but does not make the two functions identical.
Never assume arbitrary resampling preserves the guarantee.

## Experiments and optimization

Fix the accepted map as a stage anchor while optimizing that stage's latent.
Only commit a candidate after applying the declared safe operator; then refresh
geometry/evidence as needed. Do not accidentally compound trial candidates during
one latent solve. Per-case optimization does not need a graph through its entire
optimization history. Test the local decoder gradient separately.

Keep the same regularized objective for acceptance comparisons. A lower image
proxy is not proof of better anatomy. Never select a blind-test iterate by its
manual landmarks. Include all eligible failures and timeouts.

## Time, compute, and safety

Use the runtime's actual goal duration; default to the PLAN's 24-hour window.
Record real UTC start/deadline/end in an existing simple run record. Do not invent
T+24 labels, add token times to wall time, sleep to fill the window, or mark an
interrupted session complete. A Markdown file cannot extend a host's execution
limit: report interruption honestly and leave the next command.

Use local small experiments if remote VPN/SSH fails. Retry briefly, then return
to useful local work. Do not disrupt other users' GPUs, ports (including 18082),
processes, or VPN settings. Never bypass approval or network/security controls.
No extra paid model/compute service beyond the user's authorized environment.

## Minimal process

Use Git, normal tests, one result table, and short decision notes. No new hashes,
manifest system, frozen-contract framework, benchmark framework, or per-run audit
forest without a concrete failure ordinary tools cannot address. Do not delete
existing safety mechanisms. A comparison method is welcome; a new 'baseline
framework' is not. Keep existing necessary data/model artifacts.

Do not invoke optional `ars/experiment-agent` or `academic-paper` workflows in
this phase. Research skills are for targeted source checks, not mandatory full
review cycles. Higher-priority tool, security, and workspace rules still apply.

## Handoff

Separate: geometry established; useful instance optimizer; evidence bottleneck;
real-data competitiveness; formal SOTA not established. Never equate tests passed,
wall time consumed, or a theorem alone with successful registration. Finish with
actual results, the strongest adverse result, and at most three next actions.


