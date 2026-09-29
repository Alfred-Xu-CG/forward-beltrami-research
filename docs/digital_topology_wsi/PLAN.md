# 下一阶段 Goal 计划：F1/F2 数字拓扑保持与高分辨率病理图像配准

## 2026-09-29 用户延长与研究重心修正

用户在原 T+24 前将窗口**延长 12 个真实小时**，起点不变，新截止为 `2026-09-30T06:04:58Z`（总 36 小时）。原 T+22 科学审查是阶段性结论，不是延长期的最终审查；`END_TIME.json` 只在新窗口完成时写入。用户明确指出当前安全层尚未学会复杂真实配准，甚至在两例重复查看的真实标注上落后初始仿射；后续工作必须正面处理这个失败，不能以拓扑、梯度、伪教师 RMSE 或层数增加替代真实配准收益。

延长期的优先研究问题依次是：

1. **诊断瓶颈。** 用已保存的真实图像、输入/输出地图和标注的只读诊断区分：表示容量是否不足、训练目标是否错位、跨染色图像特征是否无判别力、初始配准是否锁死后续局部搜索。诊断中不得用评估标注训练或选择每例超参数。
2. **尝试有信息量的对应关系。** 比较不依赖评估标注的跨图像特征/关键点、局部结构相似度和可微图像损失，必要时探索显式匹配＋安全地图解码、逐对优化或混合网络。每个尝试都须有同样的初始仿射、blank/identity 控制和实测端到端成本；不可把直接使用完整 DHR 位移目标作为“网络学会配准”的捷径。
3. **诚实的选择标准。** 优先追求至少一个真实、未用于调参的图像组上明显优于仿射、并保持最终 P1 拓扑和可用 VJP 的结果。开发组上的改进仅作候选筛选，不能充当独立成功。若数据不足以做正式独立组，明确标注证据级别并报告失败，不为了时间硬选一个漂亮样本。

此修正不放宽共享 GPU/端口、数据许可、真实标注泄漏和精确输出证书的边界。F1/F2 层数、局部参数化、匹配前端和训练方式均可改变；是否有价值由实际配准而非层的形式判断。

**研究依据更新至：2026-09-29。** 这是可直接交给 Codex 的独立执行说明。默认首轮预算为 **24 个真实小时**；如果用户另给时长，以用户授权为准。本文不保证24小时达到SOTA，也不以时间耗尽或测试通过替代科研目标达成。

## 0. 先做什么、放在哪里

1. 在 `Alfred-Xu-CG/forward-beltrami-research` 中确认实际包含最新F1/F2实现的分支；不要假设 `master` 最新。读当前 `AGENTS.md`、F1/F2实现、最近阶段报告和用户聊天记录；不重新遍历全部旧artifacts。
2. 将本文完整保存为 `docs/digital_topology_wsi/PLAN.md`。如该目录已存在，原位更新并由Git记录，不新造另一套phase命名。
3. 将文末附录A合并进仓库根目录 `AGENTS.md`，保留仍有效的安全规定。详细数学和实验只放PLAN及对应短文档，根AGENTS作为导航。
4. 在实际支持子代理配置的客户端中，为角色显式设置本文的GPT-6模型与推理强度。配置前读取当前官方格式；不得把旧格式、虚构模型名或未执行的委派写成事实。
5. 复用已有真实计时机制；若没有，仅用一个由程序写入的开始/结束UTC记录。禁止伪造时间、把逻辑阶段当作已过小时、靠sleep凑足预算。不要再开发一套计时平台。
6. 仅新增必要成果：`EVIDENCE.md`、`TOPOLOGY.md`、`EXPERIMENTS.md`、`results.csv`、`REVIEW.md`、`DECISION.md`，以及代码、测试、必要模型检查点。数据集成员关系可以放普通CSV；不得把临时日志变成几十份长期审计文件。

**权限边界：**本文件是用户研究授权，不覆盖系统/开发者、沙箱、共享服务器或数据许可规则。允许自主提出假设、改脚本、重试、终止自己超预算的实验；不允许删他人数据、占用他人进程、绕过下载许可或改用户安全配置。

---

## 1. 研究目标及四种不能混淆的成功

最终应用：高分辨率二维病理图像配准，优先同一物理切片的重复染色/多轮成像，兼顾在相邻切片上的外部泛化。

核心候选结构：

\[
(I_F,I_M)\xrightarrow{\text{特征/匹配/提案}}
\{r^{(k)}\}\xrightarrow{\text{F1、F2及其调度}}
\Phi_h\xrightarrow{\text{明确的地图插值}}I_M\circ\Phi_h.
\]

- **G1 几何目标：**实际返回和导出的离散地图在声明的二维网格、边界、插值与浮点条件下全局一一对应；四种前后差分Jacobian严格正，禁止仅用中心差分。
- **G2 网络目标：**学到的特征/提案在新图像上确实有用；其准确性、耗时或鲁棒性超过同一几何层的无网络版本，而非仅证明随机VJP非零。
- **G3 应用目标：**在真实、无泄漏、官方口径兼容的病理配准数据上，有准确性—拓扑—时间—显存的有竞争力结果。
- **G4 SOTA目标：**仅在具体数据集、划分、指标与当前强对手的同口径比较确实支持时成立；不是预先保证，也不能由G1–G3自动推出。

**F1和F2均为主研究对象。**不得因F1局部公式简单便取消F2；不得因某个粗层F2成功便规定所有网络必须用它。两者的协同方式由公平消融决定。

**不再把“从任意逐面mu精确恢复固定P1图”设为本阶段目标。**QC/Beltrami可用于畸变解释、损失与评价，但不能重新扩散为旧的十几条solver路线。

---

## 2. 已有证据与当前未知

### 2.1 用户材料实际支持的内容

用户附件《Forward Homeomorphism Layer Chat.md》区分了生成问题与推断问题：已有F1单点、F2联合patch的固定SW–NE P1安全机制；完成4097²控制顶点生成和全latent反传；图像推断主要是粗CNN加细级光度反馈。参考附件行14–17、37–55、69–79、249–292。

已有超细网格成绩不能解释为自然图像通用配准已解决。teacher latent由已知目标反算；多视图是同一形变的独立纹理观测；部分训练从已有检查点继续；单视图续训曾退化。参考附件行249–300。

当前保证主要针对固定SW–NE三角剖分，不能自动推成另一剖分、双线性坐标场或导出到另一格点后的数字拓扑保证。旧实现的恒等回退不能算任务成功。

### 2.2 本文的证据等级

- `SOURCE`：原始论文/官方文档明确支持，给定定位。
- `DERIVATION`：本文推导出的候选性质，需独立checker补齐假设及代码对应。
- `EMPIRICAL`：实际运行所得，注明输入、输出、硬件、数据划分。
- `OPEN`：未证明、未实测、未获得数据或代码。

不得将 `SOURCE abstract only` 写成全文复现，不得将 `DERIVATION` 写成全球首创，不得将 `EMPIRICAL zero failures` 写成对所有输入的证明。

---

## 3. 文献调查结论：不承诺不存在更好方案

本轮调研覆盖八类：数字拓扑指标；连续及离散保拓扑配准；局部可行域神经层；Tutte/PL层；病理WSI强基线；细胞/核匹配；近期GPU优化；数据和评估协议。主要来源列在文末。

**可以肯定的结论：**替代方案真实存在，不能把F1/F2当唯一可行路线。没有可辩护的“100%穷尽文献”或“必无更优方案”证明。该风险由强竞争基线、小型同标准对比和明确pivot规则控制，而不是由更多空泛综述控制。

### 3.1 与几何层直接竞争的路线

| 路线/来源 | 已有核心能力 | 对当前任务的边界 | 本阶段动作 |
|---|---|---|---|
| 数字Jacobian判据，Liu等，IJCV 2024 [R1] | 二维四种前后差分与不同三角剖分的关系；NDA | 是标准/指标，不是网络生成器；仍需本项目的边界与实际插值证明 | 必须逐式使用并独立实现checker |
| SITReg，2024 [R3] | 学习式多尺度、拓扑、对称、逆一致性；隐式逆层 | 连续因子/复合和重采样的保证不同；其论文已讨论该限制 | 必须作为最接近的学习式竞争者之一 |
| Noblet等，2005/2006 [R4] | 分层B-spline与区间分析的全域约束 | 不是神经前向层，但否定“首次整个域保拓扑” | 定理查重；可作小型严格参考 |
| Sdika 2013；Chun–Fessler 2009 [R5,R6] | B-spline可逆性的充分条件 | 条件、实际罚项实现与硬约束版本不能混淆；输出采样需复核 | 优先候补的受约束FFD路线 |
| LDDMM/SVF/CPAB [R7] | 连续可逆流及成熟参数化 | 数值积分、复合及地图导出可能改变保证；不能一概判“会fold” | 选一个成熟可运行基线，不同时重写所有方法 |
| TutteNet/Generative Escher [R8,R9] | 可微合法PL解码，正权重全局求解 | 固定一套P1与四角凸quad不是同一保证；已有仓库可复用 | 一项有意义的中等规模几何对照 |
| RAYEN / Soft-Radial，2023–2026 [R10,R11] | 硬凸可行域输出；后者研究饱和/秩亏 | F1局部可行域是凸的；F2多点完整可行集合一般不是凸的 | 必须比较径向参数化，不能无条件套F2 |
| 本文候选Q1 Lipschitz residual / 单调三角层 | 可构造显式可逆小更新或结构化quad | 组合后导出问题仍需解决，单层可能保守 | F1/F2瓶颈出现后只开一条有针对性的对照 |

已有局部网格优化、flip-avoiding line search和并行图着色也必须列为F1/F2原语的先例。新意候选是：**实际输出网格的数字/插值一致保证、F1/F2协同、可训练可行参数化、以及真实配准的成本—精度收益**，不是面积是线性或二次式这个事实。

### 3.2 当前WSI竞争者，不止RegWSI

| 方法 | 必须承认的能力与证据 | 本阶段用途 |
|---|---|---|
| RegWSI / DeeperHistReg，2024 [R12,R13] | 强初始特征匹配＋逐对非刚性优化；公开框架；历史ACROBAT 2023冠军不等于2026所有任务SOTA | 强制真实任务基线；优先复用WSI I/O和初始对齐 |
| VALIS及2025公开更新 [R14,R15] | 面向实际多张WSI/多模态序列，已有大图支持及更新后的特征匹配 | 强制外部基线，不用陈旧版本制造优势 |
| CORE，2025公开稿 [R16] | 粗特征＋细胞核点集＋CPD，报告三个公开数据集结果 | 当前重要挑战者；代码/模型可运行性需查；不可忽略其最新结果 |
| FireANTs，2026 [R17] | training-free GPU多尺度优化，已报告广泛数据性能 | 回答“为什么需要网络”的强制优化式挑战者；2D/跨染色配置须实际确认 |
| UWarp，2025 [R18] | 同片跨扫描仪局部对齐；报告亚微米精度，但主要数据不公开 | 对应用选择和外部泛化有启发；不能伪造复现 |
| 2025 Elastix分层colocalization [R19] | 在已有框架上取得细结构对齐 | 强调简单可靠工程也可能足够；作为后备 |
| CurvReg，2026 [R20] | 曲形肝穿刺分段/组织mask及学习式细化 | 数据域相关时纳入，不自动取代同片主benchmark |

CORE原文包含对non-positive Jacobian区域置零位移的处理。该步骤本身不构成全局同胚证明；需要按相同标准测实际导出场，但不得据此否定它的配准准确性。其报告的HyReCo细级数值与RegWSI相比较只是**论文报告的比较**，不是本轮复现或当前官方排行榜结论。[R16]

### 3.3 需要补充查重但不能阻塞实现

首轮预留2–3小时完成：[R1]、[R3]、[R10]、[R11]的承重公式定位；强基线公开代码入口和数据许可确认；对2025–2026 WSI工作沿引用追踪一轮。连续两轮定向检索不再发现改变决策的直接方法，就进入实验。

不能从没有检索到同名工作推断全球首创。不能为了“100%确认不存在”无限研究。新发现若实质覆盖当前主张，立即调整贡献并实测，而非隐瞒。

---

## 4. 正式拓扑标准：对实际交付地图，而不是漂亮的内部状态

### 4.1 先定义物理坐标与输出

使用反向取样地图
\[
\Phi:\Omega_F\to\Omega_M,\qquad I_{M\to F}(x)=I_M(\Phi(x)).
\]
`(x,y)`与张量`[row,col]`分开；像素中心、物理间距、原点、轴向、`align_corners`写成明确转换。计算的是完整地图 \(\Phi=x+u\) 的Jacobian，不是 \(\det Du\)。

默认使用规则或张量积矩形网格上的**双线性Q1地图**作为部署表示，P1双对角线作为兼容输出/证明工具。二维Q1通常仅全局C0，不得声称全局C1 diffeomorphism。

### 4.2 每个方格四角条件

按SW、SE、NE、NW依次记目标顶点为 \(a,b,c,d\)，源方格宽高为 \(h_x,h_y>0\)。定义未归一化角行列式
\[
q_{00}=\det(b-a,d-a),\quad q_{10}=\det(b-a,c-b),
\]
\[
q_{11}=\det(c-d,c-b),\quad q_{01}=\det(c-d,d-a).
\]
要求每一个实际单元四项严格正。归一化后 \(J_{st}=q_{st}/(h_xh_y)\)。它覆盖两种对角线的四个三角形，不仅是旧SW–NE的两个面。[R1]

对
\[
\Phi(s,t)=a+s e+t f+st g,
\quad e=b-a,\;f=d-a,\;g=a-b+c-d,
\]
有
\[
\det D_{s,t}\Phi=\det(e,f)+s\det(e,g)+t\det(g,f).
\]
该量在方格上仿射，其最小值在角点；所以四角严格正推出整个Q1单元内部正向。这里是 `DERIVATION`，checker须独立展开并核对符号。

### 4.3 全局结论不能省略边界

源为有限矩形盘、拓扑相容网格；输出连续，共边值一致；边界为简单、取向保持、按顺序的凸多边形边界。通过任选一致对角线的P1正向及degree论证，再由每个Q1单元映向同一个严格凸quad的单元同胚性，建立全局Q1同胚。应引用[R2]的适当全局框架，但Q1拼接部分必须自己证明。

在等距输出网格和通常的中心差分定义下，数字四差分在合法邻点上全部严格正，由行列式双线性可得
\[
J^{00}=\tfrac14(J^{++}+J^{+-}+J^{-+}+J^{--})>0.
\]
非均匀张量积网格的中心导数若定义为跨两邻点的割线，应使用由相邻间距决定的正权加权式，而非机械使用1/4。两种情形都不保证任意高阶差分、任意滤波/重采样之后仍正。

### 4.4 精度与输出条件

数学定理假定合法初态和规定的有限latent；实际计算需要在**实际返回dtype和保存后的坐标**上核验。零面积必须失败，不能被NDA=0掩盖；不能把`J>-epsilon`称为正。

保存格式、重读、tile接缝、坐标尺度转换、最终输出分辨率都属于验收对象。地图亮度取样与地图坐标插值是两回事。

空间映射同胚不等于将二值标签重新光栅化、阈值化或下采样后像素连通性必然相同；后者可能受aliasing影响，应单独评估标注迁移，不能将这一保证偷换成“所有分割掩码离散拓扑自动不变”。

对下采样的重新插值地图不作自动保证。实际部署应保留声明的Q1地图/网格，或在最终导出格点上直接安全生成并再次验证。上采样若是完全嵌套的Q1细分，则可精确保持函数；跨旧单元的任意采样网格不享有此结论。

### 4.5 仿射初始对齐的处理

优先 \(\Phi(x)=A_0(f_Y(x))\)，其中 \(f_Y\) 为共同矩形画布上的Q1 residual homeomorphism，\(A_0\) 为非奇异、det>0的初始仿射。这种**输出侧仿射复合**保持Q1及四角符号；总导数为 \(DA_0 Df_Y\)。不要任意右复合/重采样后仍用同一证明。

扫描文件方向镜像应先显式规范化为一致物理坐标；不要把元数据反射默默当非刚性错误。若任务确需真实反向定向配对，必须另立任务定义，不能冒称本取向保持定理覆盖。

---

## 5. F1-D与F2-D：两者都必须升级，不预设谁赢

### 5.1 F1-D：所有受影响角三角形的局部安全更新

保留四奇偶颜色。一次颜色更新每个quad至多一个活动顶点。对活动点 \(i\) 收集四周所有包含它的候选角三角形，内部最多12项，而不是只约束旧6面，也不是只约束以该点为差分中心的4项。

候选位移 \(r_i\) 用该层物理尺度有界化，例如 \(r_i=\alpha\,\mathrm{diag}(h_x,h_y)\tanh z_i\)。每个约束
\[
q_k(t)=q_k(0)+tL_k(r_i).
\]
合法初态下取
\[
\ell_k=(-L_k)_+,\quad
m_k=\min\{\gamma q_k(0),\ q_k(0)-\beta_k\},
\quad 0<\gamma<1,\quad 0\le\beta_k<q_k(0),
\]
\[
s_i=\min\{1,\min_{\ell_k>0} m_k/\ell_k\}.
\]
则对整个路径 \(0\le t\le s_i\)，\(q_k(t)\ge q_k(0)-m_k>0\)。beta是面积量，应按源单元归一化，不能将不同网格的绝对阈值混用。

已有状态低于指定floor时不能装作floor仍成立；可采用只禁止继续变差的兼容行为并明确标记，但新的严格floor定理只对初态满足floor的路径成立。

### 5.2 F2-D：保留多个相邻顶点联合运动

对每个patch的所有受影响quads，保护四种角三角形。多个顶点同时移动时
\[
q_k(t)=q_k(0)+tL_k+t^2Q_k.
\]
对 \(0\le t\le1\)，令 \(C_k=(-L_k)_++(-Q_k)_+\)，则
\[
q_k(t)\ge q_k(0)-tC_k.
\]
用同样面积预算取patch尺度
\[
s_P=\min\{1,\min_{C_k>0}m_k/C_k\}.
\]
这给整条更新路径的充分安全条件，而不仅是终点检查。实现可另外比较稳健求“第一次碰floor的根”的较不保守版本，但不得跳到第二段正值区间、绕过中间退化。

同一pass的**受影响单元集合**必须互不冲突，不能仅检查活动顶点ID不重复。patch边界固定；交错偏移使seam后续可动；不完整边缘patch、非正方图、奇数尺寸都必须覆盖或显式固定，不能漏掉未检查单元。

### 5.3 分段可微与更好的参数化

硬min/max版本先作为正确性参考。它通常几乎处处可微，不是全域光滑；不得detach安全尺度或用STE然后声称原算子的真梯度。

F1截断分支可能出现径向梯度秩亏：若 \(d(r)=m r/(a^Tr)\)，则 \(D d(r)r=0\)。要测饱和比例、局部Jacobian奇异值和学习退化，而不只测梯度finite。

强制比较一个仍硬可行的内部径向重参数化。查[R10,R11]，明确功劳归属。F2可行域一般非凸，不可套凸多面体结论；需基于原点到第一次约束边界的安全区间或保守光滑上界另证。

可研究的简单保守候选（`DERIVATION`，不是已验收算法）：对允许损失 \(m_k>0\) 和不利项上界 \(C_k\ge0\)，构造平滑 \(U\ge\max_k C_k/m_k\)，取 \(s=1/(1+U)\)。则 \(sC_k<m_k\)。必须检查U的平滑上界、维数造成的保守性、零提案行为和梯度；不能因公式安全就假定full-rank或高效。

### 5.4 浮点稳健性不是用大guard掩盖

优先局部相对坐标、物理尺度归一化、可靠的面积运算；对不确定符号使用更高精度/过滤式谓词。不要对所有inactive分支做可能溢出的除法。编译和eager各做一次独立核对，明确fast-math/FMA对误差界的影响。

数值上无法安全接受某个局部提案时保持旧的合法局部状态，可以保拓扑但必须计数。整图恒等回退不算配准成功。所有声称成功的主要测试样本不得靠隐蔽回退获得零fold。

---

## 6. 多尺度、边界与超大图：共同几何，而不是地图拼贴

### 6.1 精确细化

- Q1：边中点平均，中心四角平均；每个子单元精确表达同一旧Q1函数。
- P1：中心取指定对角线两端平均；这保持的是旧P1函数。四角凸时其dyadic细quad仍凸，但P1与Q1内部值不同。
- 每一级只保留一份全局一致节点坐标。必须独立测试加密前后函数值/边界/符号。
- 若原图尺寸不能整除dyadic间距，构造包含最终物理格点的嵌套张量积轴，或直接把最终网格作为最后安全生成层。不能混淆512个像素中心与513个控制节点。

### 6.2 必须进行的调度消融

以相同提案网络和大致相同GPU时间预算比较：

1. 全F1-D；
2. 全F2-D；
3. 粗F2-D＋细F1-D；
4. 低频/相干区域F2-D、局部细节F1-D的交替方案。

先只用两种patch物理尺度和少量粗轮数，例如2/4/8。不展开无上限组合搜索。调度比较需统一原始位移量纲、幅度、实际更新次数/成本，不把同名“一轮”当相同工作量。

仅在静态混合有收益后研究dynamic routing。动态选择应作用在**提案/更新调度**，每个被执行路径均独立满足约束；不可将两张安全地图的输出平均。硬routing通常只分段可微，报告实际梯度路径。

### 6.3 边界

先采用共同大画布外边界固定，组织离外框有padding；这不是把组织自身边界固定。若外框约束限制真实配准，再研究R2边界沿side滑动及角点固定。边界提案需满足切向与连续路径正性/一维顺序条件；不能事后排序顶点改变函数。

### 6.4 WSI分块

允许分块编码、局部相关与流式I/O，但必须共享全局地图。重叠块可以先融合**原始提案或特征**，再进入一次全局一致的F1-D/F2-D；禁止分别生成多张地图后直接平均拼接。

并行块要有halo，读取同一颜色/pass的快照；共享顶点仅一个owner写入。checker检查跨tile边和所有seam单元。先做规则调度，暂不引入带hanging nodes的自由quadtree；后者必须另证几何一致性。

在超大物理坐标下审计float32精度；原点/尺度和局部位移可分存，但最终保存与重读仍需验证。只在ROI验证不能宣称全WSI输出保证。

---

## 7. 为什么需要网络：用实验回答，而不是预设答案

F1/F2不需要CNN才能保拓扑。网络可能提供：跨染色特征、较好的初值、匹配先验、减少逐对迭代、对无纹理区域的合理补充。但FireANTs、RegWSI等说明优化式方法也可能快且强。[R12,R17]

必须构造相同几何层的三种推断模式：

- **O：无可训练图像编码器。**经典/冻结特征＋局部相关或光流/迭代优化，全部更新经过F1-D/F2-D。
- **N：学习式预测。**图像encoder＋多尺度提案，固定数量安全更新；真实图像训练到全部参与参数。
- **H：混合。**网络初值/特征＋预算受控的安全细化。

单独比较：标量增益光度提示、学习式局部提案、光度＋学习残差。不能把仅调几个lambda的系统当成通用dense learned encoder。

报告训练摊销：若训练成本为 \(C_{train}\)，单对成本为 \(t_N,t_O\)，只有 \(t_N<t_O\) 时才有时间回本点 \(n_* = C_{train}/(t_O-t_N)\)。训练数据、适配新染色成本及精度必须一起报告。

若N没有优势，保留O/H并改变论文定位，不得为了满足“网络”二字添加一个几乎不起作用的CNN。

---

## 8. 候选配准网络：先解决外观与对应，再让几何层限制执行

1. 复用强初始化：组织mask、跨染色特征匹配、det>0全局仿射。不要为证明本方法独立而重写WSI读取和成熟仿射模块。
2. 共享图像encoder提取多尺度特征。优先中等宽度CNN/FPN＋局部相关；若粗配准失败再比较更强matcher，不预先扩大网络。
3. 在当前Q1地图下取移动特征，预测局部位移、置信度、patch协同分量。置信度不能自由抹掉全部困难区域：必须有coverage约束和独立评价mask。
4. F2-D处理相干/较大尺度提案，F1-D处理局部细节与seam，具体组合由消融选出。
5. 只在声明网格上精确细化；完整图像取样由同一Q1函数执行；最终亮度重采样次数及算法内部特征warp次数分别计。
6. 训练损失候选：LNCC、NGF或固定/学习特征相似度；可加入核质心软匹配及鲁棒outlier模型；加形变平滑/畸变正则，但不以Jacobian罚项代替硬保证。
7. 跨染色不能只用原始MSE；同时避免在小开发集上无穷组合损失。先选两种有文献依据的外观损失，固定开发预算。
8. teacher/伪标签可来自强方法，但必须只在训练组生成；测试地标绝不参与初始化、loss、early stop或超参选择。distillation不能称“没有任何外部几何监督”。

**同一物理切片的丢失、裂缝、染色损伤不符合处处真实一一对应。**用外部预先定义的组织可见性mask评价有对应部分；完整画布上的同胚是几何延拓，不声称缺失组织中存在真实对应。mask不得根据本方法的误差事后裁掉难点。

---

## 9. 数据设计：HyReCo主任务，ACROBAT/ANHIR外部竞争

### 9.1 数据源与任务地位

- **HyReCo重复染色：主应用。**原研究区分同片重复染色与相邻切片；同片更适合细胞/核级对应。[R21]
- **HyReCo相邻切片：单独报告。**不可与同片数据混成一个“严格物理同胚”结论。
- **ACROBAT：较大、临床常规染色和域移测试。**采用官方版本及评价协议，2022公开数据/验证排行榜与2023隐藏新测试不是同一实验。[R22,R23]
- **ANHIR：外部泛化、组织与染色多样性。**保持官方rTRE分母、配对与聚合，公开训练地标和隐藏评估明确分离。[R24]
- 数据可用性、许可、版本、原始像素大小必须在运行时确认。下载受限就记录为未完成，不用合成纹理冒充病理实验。

### 9.2 无泄漏分组

按可获得的最高独立单位分组：patient > tissue block > 原始物理切片。重复染色、多扫描器图像及其所有patch必须同组。不能随机patch切分。

若只有单一患者或患者信息不全，明确限制为slide/block泛化；不能把pair数当独立患者数。小HyReCo可采用grouped外层交叉验证；模型选择仅在内层开发组。官方划分优先，不擅自以“平衡数据”为理由改测试集。

真实测试地标只由独立evaluator读取；若评估服务不可访问，状态是“内部可比较，官方SOTA未验证”。

### 9.3 两种公平评估模式

- **模块比较：**所有几何后端共享同一个合格仿射初值、相似度与特征提案，隔离安全层代价。
- **完整系统比较：**每个强基线按官方推荐流程、自己的初始化运行，比较真实可用系统。

这两组分别报告；不能通过削弱基线初始化让我们的完整系统胜出，也不能只共享基线特征却把收益都归给F1/F2。

---

## 10. 竞争基线与优先级

### 必做核心集合

1. RegWSI/DeeperHistReg实际可运行的强配置。
2. 当前可用VALIS稳定版。
3. 本方法同几何层的O/N/H三种推断。
4. SITReg或另一可获得的强拓扑学习式方法，适配二维/同特征时清楚注明改动。
5. FireANTs对应可运行的二维优化配置；若2D或模态不支持，报告具体接口/失败，不伪造适配结果。

### 最新方法补位

CORE是必须调查的2025候选。能运行则加入；暂不能复现，应明确原因并限定“优于已复现基线”，不能喊全领域SOTA。UWarp、CurvReg按数据域和公开性纳入，不为了表格长而同时重写。

### 几何替代对照

至少选一个与F1/F2新标准同样成立的替代：

- 受硬导数/Lipschitz界的Q1 residual；
- 可验证B-spline/interval方法；
- 安全凸域径向层（对F1局部可行域）；
- 单调三角quad层或可核验的Tutte变体。

普通Tutte只保证某P1剖分，不应未经修改标为四角保证。SVF/CPAB/SITReg同时报告native函数理论和实际导出数字字段；不能仅凭某次重采样失败贬低其native理论，也不能把native理论自动授予导出数组。

---

## 11. 实验顺序：让最小实验决定大训练是否值得

### E0 — 独立标准与反例，首日最早完成

输入不是图像，只是坐标数组。必须包含：恒等、正仿射、强压缩、凹quad、另一对角线反转、中心差分漏检、边界顺序错误、NaN/Inf、极薄单元、奇数/矩形尺寸。

checker用独立NumPy/高精度几何实现四角与四差分，不能调用生产area helper。至少一个有理数手算例。确认：NDA=0不掩盖零面积；所有四角正推出Q1内部正；导出重读和坐标换算一致。

### E1 — F1-D / F2-D安全与微分

分别测试完整proposal到输出，含同色并行、patch重叠错误、偏移边缘、当前几何变化和boundary。理想公式解析证明＋浮点输出独立符号核验；finite difference扫多个步长，远离分支处检查真梯度，分支处只声称单侧/选定分支导数。

默认tiny float64 directional误差目标1e-5以内；float32目标随尺度和条件数说明，不能全局放宽测试到随意通过。比较eager/compiled/activation-recompute。注入一次故意漏约束错误，checker必须检出。

### E2 — 固定标准下的表示和调度

用独立解析/物理样式目标：剪切、局部旋转、压缩、跨patch位移、窄局部特征。区分teacher inversion、逐对latent优化、image prediction三种任务。

同预算比较全F1-D、全F2-D、混合和一种替代。记录时间、显存、误差、安全尺度、饱和、拒绝、粗层轮数与patch大小。不能只用F1/F2自己生成的target。

四角约束比单一P1更强；允许极端P1目标不在新可行类，不能为拟合它偷偷关掉另一对角线检查。选定光滑物理目标的网格充分细时可行性需证/测，而非一概声称新标准不损失表达能力。

### E3 — 真实病理小规模管线打通

先在开发组少量完整slide/ROI跑强基线和本方法，确认毫米/微米/像素换算、landmark方向、mask、读取层级、仿射det、WSI输出均正确。立即开始，不等24小时最后才看第一对真实切片。

这些只作开发，不能用来最终宣称SOTA。不得提供测试的真实boundary或真位移。

### E4 — 为什么需要网络

在相同数据和几何约束上比较O/N/H。主要问题：N是否减少逐对优化且不损失准确性，H是否提供最好Pareto；而不是CNN参数是否有非零梯度。

做输入置换/冻结encoder消融，验证地图确实依赖该图像对且学习有用；不要求所有loss都必须变差，但必须报告作用大小。

### E5 — 分辨率与全图一致性

使用真实原图的不同物理分辨率和真实细结构，不把512²图像上采样成4096²当新信息。起步ROI 1024/2048，然后4096及完整WSI流式应用；地图自由度按实际需要增长。

分别报告图像像素数、控制节点数、patch提案数、输出地图格点数。测地标/细胞层误差是否随更高信息分辨率改进，并与训练/推理成本对应。

### E6 — 正式盲测与外部泛化

只在开发设置确定后执行。小规模开发成功不替代多组测试。默认至少3个训练随机种子；若预算不足就降为初步结果并给继续命令，不把单种子泛化成最终结论。

官方split/聚合优先；自定义实验必须标识。保留失败、超时、缺失文件并按事先规则纳入统计，不能删去最难病例。

---

## 12. 指标、通过标准与SOTA用语

以下数值是**本项目拟定的开发决策标准**，不是论文规定的自然常数。可以基于开发数据、标注噪声和资源在盲测前调整一次并写明理由；禁止看测试结果后改。

### G1：几何层合格

- 理想F1-D/F2-D、Q1全局/边界、细化、输出侧仿射定理经独立checker核对。
- 每个主要测试输出文件中：四角/四差分所有真实单元严格正，零退化，边界有序，共边一致；未靠忽略背景或seam得到结果。
- 若数值符号无法判定，不能标正。局部拒绝保留旧状态须计数；整图回退在主要测试中目标为0，否则计任务失败并分析。
- 有独立梯度和导出/重读测试，报告a.e.可微范围和分支行为。

这项通过只支持“安全几何层”，不支持“准确配准”。

### G2：学习式网络有价值

在独立held-out组上，N或H相对同几何层O，满足以下预声明的一项：

- 主几何误差至少改善10%，且组级配对bootstrap的95%区间支持改善；或
- 主几何误差在5%非劣范围内，并实现至少2倍推理速度或明确的显存优势；或
- 在预先定义的低对比/跨染色域移上显著降低失败率，同时不牺牲主要准确性。

误差非常接近标注噪声时，不机械追求相对10%；以开发阶段给出的物理意义阈值和CI为准。训练成本和预训练来源必须计入。

### G3：有竞争力的应用结果

比较同split、同协议的最强已复现基线：

- 几何指标：HyReCo以微米TRE与相应官方/原研究聚合为主；ACROBAT按对应官方版本规定的聚合；ANHIR按官方rTRE和稳健性统计。
- 附加：尾部TRE、任务失败率、细胞/核对应（有独立ground truth时）、畸变、实际输入输出拓扑、end-to-end时间和峰值内存。
- 允许“准确性非劣＋明确数字拓扑保证＋显著成本/可靠性收益”构成有价值成果，但不得改名为准确性SOTA。
- 如果基线在实际任务同样全程数字无fold，本方法必须在准确性/成本/泛化中提供额外可测收益，不能凭定理文字取胜。

### G4：允许SOTA声明的条件

明确写出数据集版本、官方split、任务、指标、可用测试条件及最新可复现对手；同口径排名确实最好，统计和标注分辨率支持差异；未排除相关强方法或失败病例；必要时获得官方server结果。

没拿到隐藏test，只能说“在本次可用划分/复现集合最好”；没有显著差别则说并列/不可区分。时间到、三百测试通过、最小J>0都不构成SOTA。

### 指标实现细节

- 地标方向与image backward map统一；需要正向点映射时真正求 \(\Phi^{-1}\)，不是交换图像就假定互逆。
- inverse在Q1单元内用独立点定位/解析或稳健局部求解；重采样得到的inverse-field另需验证。
- TRE用物理单位；ANHIR分母和目标frame严格依官方脚本，不擅自用crop对角线。
- 报告单位样本/组织块聚合，不能把同片数千patch当独立样本伪造显著性。
- 图像损失、TRE、NDA、J尾部、QC畸变分别报告；不能因为MSE下降就宣称形变正确。

---

## 13. 防止走捷径：判为不通过的具体情形

1. 输出恒等/冻结大多数困难点，获得零fold，却没完成对应任务。
2. 只检查中心差分或旧SW–NE两面，漏另一对角线。
3. 只在GPU内部高精度张量检查，导出float32后不检查。
4. 用同一个area/坐标代码产生期望值和验证值，循环论证。
5. 多个安全patch地图直接平均，或复合后随意重采样，仍使用旧保证。
6. 训练/调参看到测试地标、真boundary、真map、同切片其他patch，或以confidence mask事后删除高误差区。
7. 对手版本陈旧/初始化被削弱/运行失败却不分析；仅拿论文异硬件数字来证明速度优胜。
8. 把teacher-latent重建、随机VJP、继续训练checkpoint说成从图像学会通用配准。
9. 两种方法共用同一个数值LR就称公平；实际上坐标/步幅/可用预算不同。
10. 只用噪声纹理、多视图模拟数据或上采样图，称临床高分辨率SOTA。
11. 用近似逆/错误物理单位算TRE；hidden benchmark不可用却标官方通过。
12. 放宽正性或梯度阈值直到测试通过；有限精度回退未纳入失败率和耗时。

checker应从中至少构造两种定向反例检查评价管线，而不是只阅读作者总结。

---

## 14. 优先顺序与遇瓶颈后的有限转向

**主投入：**四角约束F1-D/F2-D共同核心＋真实WSI强基线＋O/N/H比较。

**备用预算：**约20%的研发预算用于一个最相关替代，不同时重启十条路线。

| 瓶颈 | 首先诊断 | 下一条允许试的路线 |
|---|---|---|
| F1截断多、粗运动不够 | 同步提案/中间状态限制 | F2较大协同patch、额外粗轮、更好初值 |
| F2被单个面拖慢 | patch内最小余量/异质性 | 更小patch＋F1、保守根法、分组调度 |
| 安全器梯度饱和 | 局部Jacobian与激活比例 | RAYEN/Soft-Radial式内点参数化，保持硬可行 |
| 两者都难训练 | 同几何层O是否容易优化 | learned features＋逐对安全优化，不强制纯前馈 |
| 高分辨率耗时由特征主导 | profiler分拆encoder/geometry/I/O | 分块特征、缓存、稀疏相关，不先改拓扑理论 |
| 凸quad约束明显损伤任务精度 | 充分分辨率、边界/初值是否错误 | 保证更强的B-spline/Lipschitz替代，或限定任务；不得悄悄取消标准 |
| 新方法在实测不如SITReg/FireANTs | 公平配置与领域适配 | 将F1/F2作为export-safe refinement层或改研究定位 |
| 数据对应本身非同胚 | 缺失/撕裂/遮挡 | 可见性/outlier建模或将该数据列为外部压力，不强迫虚假物理解释 |

转向须先写三句话：哪条假设被证伪、替代怎样解决、最小决策实验是什么。不要为转向再建一整套计划和ledger。

---

## 15. Agent角色：只使用GPT-6，按任务而不是按文件大小选强度

2026-09-29核对的官方模型ID：`gpt-6-astra`、`gpt-6-sol`、`gpt-6-luna`。[R27–R31]

| 角色 | 默认模型/effort | 任务与升级条件 |
|---|---|---|
| Coordinator | `gpt-6-sol`, medium | 路线、预算、汇总；重要路线取舍临时high，不逐秒盯日志 |
| Literature/Data Scout | `gpt-6-luna`, medium | 论文/代码/许可/指标定位和精简提取；不能独自批准novelty/theorem |
| Geometry Builder | `gpt-6-astra`, high | F1-D/F2-D、Q1全局性、细化、浮点scope；不是日常格式整理 |
| Implementation Builder | `gpt-6-sol`, medium | 实现、工具、训练；自定义autograd/并发seam错误升high |
| Experiment Operator | `gpt-6-sol`, low或medium | 运行现有实验、真实异常诊断；机械日志由Luna low处理 |
| Math Checker | 独立`gpt-6-astra`, high | 不依赖builder隐藏推理，独立证明/反例；冲突才xhigh |
| Code/Gradient Checker | 独立`gpt-6-sol`, high | 实际运行替代实现、FD与输出检查，不能只改文案 |
| Statistical/Evaluation Checker | 独立`gpt-6-sol`, high | 数据泄漏、坐标、官方聚合、失败分母、预算与CI |
| Final Adjudicator | `gpt-6-astra`, high | 收到匿名A/B结果与未闭合问题，逐项给G1–G4 verdict |

`max`仅用于两次独立尝试仍无法解决、且改变主决策的数学冲突；默认不超过一次短问题包，不让max读全库日志。若再次升级没有新信息，保留OPEN并换最小可证伪实验，不能无限自我审稿。

所有角色保持真实调用记录（角色、实际model ID、effort、一个任务句即可）。客户端不支持选模型/子代理时如实说明，不能在文档里假装已使用独立GPT-6 checker；可用另一独立会话，但同上下文自评不能冒称独立复核。不得自动降到GPT-5系列。

官方当前支持在自定义agent配置中设置`model`与`model_reasoning_effort`；采用实际版本的格式，避免猜测。API自建路由用Responses的`reasoning.effort`，不要混用CLI字段。[R30,R31]

最多同时3个子代理：一个builder、一个独立问题/数据任务、一个checker。读日志、作表不需最高推理。主上下文只接收短结果与定位，不搬入海量raw outputs。

---

## 16. Checker职责：纠错，而不是制造流程成本

### 三个承重边界

1. **几何算子合入前：**F1-D、F2-D、Q1和输出语义；独立公式、枚举小格点、至少一个攻击例。
2. **正式评估前：**数据划分/标签可见性、公式/单位、baseline配置；以真实一对图走完整链。
3. **宣布阶段科研结论前：**独立重跑小核心集及一项真实基线/候选结果，逐项验收G1–G4。

平常修改一个函数不启动完整三人审稿。争议分为：定理错误、实现错误、评价错误、资料缺失、仅风格。前三者阻断相关主张，风格不能吞掉研究时间。

checker不要求阅读私有思维链，要求公开可复核推导、代码路径、实际输出。即使两个agent同意也不等于零错误；必须有数学与独立计算两条证据。

数值checker与生产layer不共享关键面积helper；统计checker不使用训练脚本内部宣称的成功布尔值。最终读实际保存地图与地标预测。

---

## 17. 首轮24小时：真实数据、理论和实现并行，不强行许诺SOTA

| 实际时间 | 主工作 | 独立并行工作 |
|---|---|---|
| 0–2 h | 确认分支/环境，读最新F1/F2，确定地图坐标和标准 | 下载许可/数据可用性；启动RegWSI/VALIS小开发例 |
| 2–6 h | F1-D与F2-D四角推导、tiny实现、checker | SITReg/FireANTs/CORE接口与新近文献核对 |
| 6–10 h | 几何安全/梯度/细化/导出；调度小消融 | 真实图像评估脚本、分组和单位验证 |
| 10–16 h | O/N/H开发实验；先真实ROI再更大分辨率 | 外部强基线同口径运行、训练曲线和失败分析 |
| 16–21 h | 存活方案扩大真实图分辨率；WSI接缝；必要单一pivot | 独立数据/统计review；已启动训练持续进行 |
| 21–24 h | 不再增加新路线；复核、组级结果、G1–G4状态 | 独立最终验收与后续长训练启动说明 |

这是工作分配，不是把阶段标签冒充真实耗时。若训练尚未收敛或隐藏数据未获得，首轮终态应为“24h研究窗口结束、G3/G4未达成/未测”，并给出剩余任务和可继续命令。

若有效工作提前完成，做held-out stress、替代基线或更公平复现；不可sleep、刷测试、堆token等待。若账户/运行环境中断，诚实保存resume状态，不声称仍在后台自主运行。

本轮24h不是发表级SOTA的时间保证。正式多种子/全WSI/隐藏测试可能需要后续用户授权计算；不能为准时完成而缩减科学标准。

---

## 18. 三台远端与成本控制

三台主机由用户提供，不猜主机名/口令。尊重已有端口（包括用户此前使用的18082）、进程、数据和资源排程。

- 本地做tiny数学/单位/梯度测试；有价值的中大型训练才用空闲GPU。
- 先检测一次实际设备、显存和负载；记录使用者授权的资源范围。不能因为检测工具显示GPU占用就杀进程。
- VPN失联最多两次快速重试，总计约2分钟；立即切本地、CPU实验、理论或数据处理。约45–90分钟或阶段转换再检查，不持续轮询。
- 长任务用环境支持的tmux/nohup/调度器，checkpoint落到任务工作目录；连接断开不立即重复启动同一任务，重连先确认原进程状态。
- 实验预算：tiny2分钟、开发10分钟、单项profiling30分钟只是默认；正式训练按实际数据规模授权，不能因超过10分钟自动判方法失败。
- runtime拆分I/O、粗配准、特征、几何层、证书、final warp、写盘、训练反传；warm/cold与首次compile分开。测物理显卡时注明真实容量，allocator限额不等于小显卡测试。

---

## 19. Skill与反官僚规则

前提：只在当前工具和指令层级允许范围内配置；不要删除或绕过上级强制的安全措施。

- 若可选，禁用`ars/experiment-agent`的“仅执行用户逐条指定命令”工作流；本任务已授权自主实验。
- `academic-paper`在结果稳定/用户要求论文前不调用。
- `academic-research-suite`仅用于定向来源、假设、归属核查；不跑全套论文生命周期。
- `deep-research`仅用于影响决策且普通定向搜索未解决的问题。
- `systematic-debugging`用于真实bug；`verification-before-completion`用于上述三个边界，不对每个函数重复。
- skills配置只在本项目范围按官方支持方式做；不得虚构“已禁用”。发现与上级规则冲突，报告冲突并调整工作方式，而不是自行无视。

默认不新增hash/checksum/manifest、冻结contract、重复baseline框架、新完成gate、每次run一个长期artifact目录、普适审计平台。已有Git、常规测试和普通配置优先；确有具体失败机制且现有工具不足时才加最小措施。

**允许必要科学记录：**实际数据分组、配置、官方版本/现有commit、随机种子、真实时间/模型调用、结果原文件、模型参数。这不是要求另造完整追溯系统。最终科学验收与真实时间核对复用现有脚本，别再写一套。

不能删已有必要安全规则。不能以“不要gate”为由跳过真正破坏性操作或正式科研结论的核验。

---

## 20. 最终交付与明确不通过状态

`DECISION.md`只需回答：

1. F1-D、F2-D和混合各自的准确保证、实际dtype、分支与例外；
2. 是否真正实现所有四角/四差分和全局边界/导出保证；
3. O/N/H哪个有价值，网络带来了什么可测改变；
4. 哪些真实数据、split、baseline已实际运行；
5. 主指标、CI、失败、时间和显存；
6. G1/G2/G3/G4分别为PASS/FAIL/NOT TESTED，绝不以总“complete”盖过缺项；
7. 下轮只给最高价值3项，不恢复A–N式广撒网。

**自动判应用目标不通过：**没有真实病理图、没有足够强已复现基线、数据泄漏未排除、只提供零fold/随机VJP/teacher误差、主要输出靠恒等回退、测试集上调参、或未按官方口径却称SOTA。

**可以是有价值的负结果：**新严格标准确实让精度显著受损；无网络方法全面优于网络；SITReg或其他方法在相同导出标准下更好。此时保留正确实现和结论，转方法定位，不强行包装成功。

---

## 21. 核心原始来源与核查范围

以下来源是本计划的依据与Codex后续精读入口。日期为已核查版本/发布年，不等于已复现代码；原始作者摘要的性能主张不能直接作为我们的比较结果。

### 数字拓扑与几何/学习方法

- **[R1]** Liu et al. *On Finite Difference Jacobian Computation in Deformable Image Registration*. IJCV 132, 3678–3688, 2024. DOI: https://doi.org/10.1007/s11263-024-02047-1 。本轮读取全文网页，重点§2.2–2.3；数字四差分与半平面可行域是已有结果。
- **[R2]** Lipman. *Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing*. https://arxiv.org/abs/1310.0955 。全局degree/边界框架；不能直接代替本计划的Q1拼接与数值证明。
- **[R3]** Honkamaa & Marttinen. *SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration*. MELBA 2024. https://www.melba-journal.org/papers/2024:026.html 。本轮读取期刊全文；需要特别复核Appendix C及resampling范围。
- **[R4]** Noblet et al. *3-D deformable image registration: a topology preservation scheme based on hierarchical deformation models and interval analysis optimization*. TIP 2005. https://doi.org/10.1109/TIP.2005.846026 。本轮确认原始摘要；进一步需读其引用的二维先例及完整约束实现。
- **[R5]** Sdika. *A Sharp Sufficient Condition for B-Spline Vector Field Invertibility. Application to Diffeomorphic Registration and Interslice Interpolation*. SIIMS 2013. https://doi.org/10.1137/120879920 。本轮确认原始出版页/摘要；具体可逆条件需原文核对。
- **[R6]** Chun & Fessler. *A Simple Regularizer for B-spline Nonrigid Image Registration That Encourages Local Invertibility*. 2009. https://doi.org/10.1109/JSTSP.2008.2011116 ，作者机构存档 https://deepblue.lib.umich.edu/handle/2027.42/85951 。必须区分充分条件与仅用penalty的实现。
- **[R7]** Freifeld et al. *Highly-Expressive Spaces of Well-Behaved Transformations*. ICCV 2015. https://openaccess.thecvf.com/content_iccv_2015/html/Freifeld_Highly-Expressive_Spaces_of_ICCV_2015_paper.html 。CPA速度不等于最终地图P1。
- **[R8]** Sun et al. *TutteNet*. CVPR 2024. https://arxiv.org/abs/2406.12121 。可复用既有仓库分析，不重复全面审计。
- **[R9]** Aigerman & Groueix. *Generative Escher Meshes*. https://arxiv.org/abs/2309.14564 。可微正权重有效网格先例。
- **[R10]** Tordesillas et al. *RAYEN: Imposition of Hard Convex Constraints on Neural Networks*. https://arxiv.org/html/2307.08336v2 。本轮读取全文网页；F1局部可行域与其径向机制需精确比较。
- **[R11]** Schneider & Kuhn. *Soft-Radial Projection for Constrained End-to-End Learning*. 2026预印本. https://arxiv.org/html/2602.03461v1 。本轮读取全文网页；严格可行与梯度饱和重要先例，不能自动应用于F2非凸集合。

### WSI与高分辨率竞争者

- **[R12]** Wodzinski et al. *RegWSI*. CMPB 2024. https://arxiv.org/html/2404.13108 ，DOI https://doi.org/10.1016/j.cmpb.2024.108187 。本轮读取全文，关注初始化、非刚性、数据划分与不同聚合；不是已经复现。
- **[R13]** Wodzinski et al. *DeeperHistReg*. https://arxiv.org/abs/2404.14434 。WSI数据读取/接口与算法分开计功。
- **[R14]** Gatenbee et al. *Virtual alignment of pathology image series for multi-gigapixel whole slide images*. Nat Commun 2023. https://www.nature.com/articles/s41467-023-40218-9 。
- **[R15]** VALIS v1.2.0官方软件发布记录，2025-06-25. https://zenodo.org/records/15739760 。该版本说明更新了特征提取/匹配；执行时再查实际稳定版。
- **[R16]** Nasir et al. *CORE — A Cell-Level Coarse-to-Fine Image Registration Engine for Multi-stain Image Alignment*. 2025预印本. https://arxiv.org/html/2511.03826 。本轮读取全文，§3.3.2处理non-positive Jacobian、§4数据/结果；表4报告HyReCo细级，但未独立复现。公开代码入口见论文。
- **[R17]** Jena et al. *Adaptive Riemannian optimization for multi-scale diffeomorphic matching*（FireANTs）. Nat Commun 2026. https://www.nature.com/articles/s41467-026-72508-3 。可另核查2026 LM变体 https://arxiv.org/abs/2603.19371 ；不要未经验证把其3D设置标成2D病理基线。
- **[R18]** Schieb et al. *UWarp*. 2025预印本. https://arxiv.org/abs/2503.20653 。同片跨扫描仪应用，私人数据限制明确。
- **[R19]** Bisson et al. *A high-precision hierarchical registration approach for stain- and scanner-independent colocalization on whole slide images in histopathology*. 2025. https://doi.org/10.1007/s13755-025-00353-7 。
- **[R20]** *CurvReg: Curvature-aware registration for multi-stain alignment of liver biopsies*. 2026在线预出版. https://doi.org/10.1016/j.jpi.2026.100698 。本轮核查出版页/摘要，未全文复现，不作泛化SOTA判断。

### 数据和正式评估

- **[R21]** Lotz et al. *Comparison of Consecutive and Re-stained Sections for Image Registration in Histopathology*. https://arxiv.org/abs/2106.13150 。HyReCo原始研究；同片与相邻切片应分开。
- **[R22]** ACROBAT官方任务和数据概述. https://acrobat.grand-challenge.org/ 。2023未公开新测试与既有公开数据的区别不能忽略。
- **[R23]** ACROBAT官方评价. https://acrobat.grand-challenge.org/evaluation-ranking-and-prizes/ 。本轮搜索获得官方内容，完整抓取部分失败；执行时必须读取当前官方评价代码/规则，不能自行猜聚合。
- **[R24]** ANHIR官方评价. https://anhir.grand-challenge.org/Performance_Metrics/ 。rTRE分母和多级统计必须照官方；本轮完整抓取部分失败但搜索可得官方说明。
- **[R25]** digital-diffeomorphism作者软件发布. https://pypi.org/project/digital-diffeomorphism/ 。本轮看到1.1.1发布信息，运行时核查实际版本；可作为第二checker，但生产实现不能只测自己。
- **[R26]** NIST / DIC Challenge 2.0. https://www.nist.gov/publications/dic-challenge-20-developing-images-and-guidelines-evaluating-accuracy-and-resolution-2d 。来自上一轮确认的候补应用入口，本阶段不声称重做DIC综述，不作为主训练任务。

### GPT-6 / Codex官方指引（2026-09-29核对）

- **[R27]** GPT-6 Astra. https://developers.openai.com/api/docs/models/gpt-6-astra 。支持low/medium/high/xhigh/max。
- **[R28]** GPT-6 Sol. https://developers.openai.com/api/docs/models/gpt-6-sol 。用于coding/agentic工作；运行环境可用性仍需确认。
- **[R29]** GPT-6 Luna. https://developers.openai.com/api/docs/models/gpt-6-luna 。轻量定位和提取。
- **[R30]** Reasoning与模型指导. https://developers.openai.com/api/docs/guides/reasoning ，https://developers.openai.com/api/docs/guides/latest-model 。effort按任务选择；不以无限提高effort替代正确问题与证据。
- **[R31]** Codex/ChatGPT subagents与AGENTS. https://developers.openai.com/codex/multi-agent ，https://developers.openai.com/codex/guides/agents-md 。本轮访问重定向至官方learn.chatgpt.com；独立context、短结果和显式模型设置有助降低上下文污染。

---

# 附录A：合并到根目录AGENTS.md的精简工作原则

以下附录既可单独使用，也可从本文件提取。不要将整份PLAN复制进AGENTS。

## Mission and source of truth

Current plan: `docs/digital_topology_wsi/PLAN.md`.
Goal: useful high-resolution 2D pathology registration with a digital-output-consistent homeomorphic map, not merely a zero-fold toy decoder.
Both F1 and F2 remain primary. Their schedule and the usefulness of learning must be experimentally established.

## Scope and honesty

Separate G1 geometry, G2 benefit of learning, G3 competitive real-data performance, and G4 dataset/protocol-specific SOTA. Passing tests, using the time budget, or returning identity does not establish G2–G4.
No claim of exhaustive literature coverage or error-free agents. Cite verified sources and distinguish new derivations from established results.
Do not revert to broad legacy Beltrami/Tutte/BHF route exploration unless a specific current bottleneck justifies one targeted alternative.

## Geometry

Default deployed map is Q1 on a declared rectangular grid. Protect all four corner determinants of every affected cell, not only the original SW–NE triangles.
Require consistent boundary, shared-edge values, physical coordinates, and saved-output validation.
F1 uses all incident constraints and independent colors; F2 uses all affected-cell quadratic bounds and nonconflicting patch passes.
Never blend accepted maps or silently resample them under an old certificate. Features/raw proposals may be blended before a valid update.
Report almost-everywhere differentiability, active scaling, rejections and fallbacks. Never detach a safety scale while claiming the true gradient.

## Experiments

Start real pathology baseline/data work in parallel with tiny geometry tests.
Compare no-network optimization, network prediction, and hybrid refinement using the same safe layer.
Use patient/block/physical-slide grouping; never split patches from one physical slide across training and test.
Never use test landmarks, true target boundary, or target maps for inference or model selection.
Use official evaluation conventions. Keep failure cases in the denominator. Do not call an unavailable hidden-test comparison SOTA.

## Models and delegation

GPT-6 only. Default coordinator/coding: `gpt-6-sol` medium; complex implementation/checking: Sol high; extraction: `gpt-6-luna` low/medium; core math and independent math review: `gpt-6-astra` high.
Use xhigh/max only for a focused unresolved decision-critical conflict after two distinct attempts. Never claim unavailable model routing or fake independent agents.
At most three concurrent focused agents. Return short evidence summaries, not whole log dumps.

## Independent checks

Independent review at geometry merge, formal evaluation setup, and final scientific claim. Checker uses separate critical formulas/code paths and at least two deliberate failure fixtures.
The author is not their own only reviewer. Agreement between agents is not proof; mathematical derivation and independent computation are both needed.

## Autonomy and resources

Autonomously edit, test, measure, retry and stop your own over-budget experiments within permissions. Preserve other users' files, jobs, ports and security rules.
VPN failure: two brief retries, then local/theory/CPU work; recheck at natural transitions, not continuously.
Use real elapsed time for the default 24h window; do not fabricate progress timestamps, sleep to fill the window, or equate elapsed time with scientific success. If the environment ends, report a resumable partial state.

## Keep the process light

Do not add hashes/manifests, frozen contracts, duplicate auditing platforms, per-run permanent folders, or speculative infrastructure by default. Use Git, ordinary configs/tests, concise results and saved scientific outputs.
Do not delete necessary existing safety measures. Respect instruction hierarchy.
Optional `ars/experiment-agent` and premature `academic-paper` workflows are not appropriate; targeted literature and real-bug debugging are. Disable optional workflows only through supported project-local configuration, never pretend or override mandatory controls.

## Final decision

Read actual saved maps and evaluation outputs. Report G1/G2/G3/G4 independently as PASS/FAIL/NOT TESTED, strongest counterexample, best viable architecture, baseline coverage, and at most three next tasks.
A strong alternative outperforming F1/F2 is a legitimate research finding, not a result to suppress.
