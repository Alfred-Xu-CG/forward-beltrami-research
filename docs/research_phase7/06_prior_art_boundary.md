# Phase VII 的原始文献定位：不预设首创性

本文件只核对与“latent → 高分辨率**固定源三角剖分的一张 P1 homeomorphism**，可全反传，尽量不解大系统”直接相交的文献。连续微分同胚、任意动态剖分上的 PL 映射、若干 P1 因子的复合、仅局部面正向的优化结果，均不可自动替代这个输出契约。以下不是完整系统性综述，也没有证明“领域里无人做过”。

| 原始工作 | 已明示的核心性质 | 与本轮 F1/F2 的实质区别和不能偷换之处 |
|---|---|---|
| [Lipman, *Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing*](https://arxiv.org/abs/1310.0955) | 对保持统一面取向的 simplicial map，加入一一对应边界等条件可推出全局单射与到目标域的满射。单独“全部面积正”并不够。 | 为我们的固定矩形边界证明提供**全局拓扑判据**，并非神经 decoder 构造。必须逐项满足边界假设；浮点面证书只是实现核验。 |
| [Aigerman–Groueix, *Generative Escher Meshes*](https://arxiv.org/html/2309.14564) | 将正的邻边 barycentric/Laplacian 权重和边界条件变为可微参数，解线性系统得到不重叠 2D 网格；其定理证明在所设平铺边界约束下覆盖所有有效配置。 | “可微 latent → 合法 P1 网格”及某种**完全表达性**均不是本轮首次提出。它依赖网格规模线性系统；本轮候选的差异是局部正向、多尺度、257²/1025² 真正固定原网格 forward/VJP 的实测。F1/F2 现仅有**目标依赖且不限制深度**的固定网格可达性证明，不能推成短网络训练后能快速编码任意目标，也不能据此声称整体实用表达性胜过该工作。 |
| [Sun 等, *TutteNet*](https://arxiv.org/html/2406.12121) | 每层通过正 Laplacian 稀疏系统得到 2D Tutte P1 图；多层在不同 3D 平面复合，形成 3D injective 函数，可由网络预测参数。文中分辨率消融列出 7、11、17、25 顶点每边，主学习例采用 11² 与 24 层。 | 它不是“没有神经可微同胚层”的反例，而是重要前例。其复合结果一般不再是某**一张预定 2D 原网格**上的 P1；小每层分辨率不等于其最终连续函数只能有低细节，也不能凭消融表断言后续工作没有更高控制分辨率。本轮仍需与其方法在准确度/显存对齐后的定量比较。 |
| [Alamdari 等, *How to morph planar graph drawings*](https://arxiv.org/abs/1606.00425) | 两幅同一平面图的直线无交叠嵌入之间，有 \(O(n)\) 个保持平面性的单向线性 morph 步；最坏情形阶数最优。 | 说明固定 connectivity 的合法嵌入可通过安全连续路径连接，避免笼统断言局部正向更新不可能到达目标。但它没有给出本轮四着色/patch 字典的常数深度、图像 latent 编码器或低显存神经实现。 |
| [Luo, *Spaces of Geodesic Triangulations of Surfaces*](https://link.springer.com/article/10.1007/s00454-021-00359-4) | 固定凸多边形边界与三角连接关系的平面嵌入空间不仅路径连通，而且可缩；可用正有向重心权重插值构造连续合法路径。 | 是一般连通性的重要前例，不是本轮新定理。具体规则网格的共线边界细点与角点划分边还须用下一行 Floater 的判据核对。文献证明路径时求解大系统；我们的实际局部 decoder 在给定 latent 后不求它。 |
| [Floater, *One-to-one piecewise linear mappings over triangulations*](https://doi.org/10.1090/S0025-5718-02-01466-7) | Proposition 3.3：每张单射 P1 图的内部顶点都有严格正的邻点凸组合权重。Theorem 6.1：对凸目标边界上的共线细分点，正权重解一一对应当且仅当没有内部划分边被整条映到目标边界。 | 本固定 SW–NE 正方网格的角点划分对角边连接**相邻两侧**，不落在同一方形边上；从而正权重线性插值给出合法嵌入路径。这填上 [固定网格可达性证明](07_fixed_mesh_reachability.md) 的弱凸边界假设缺口，但**不是**网络无需大系统即可快速求得目标 latent 的结论。 |
| [Guan, *Exact Reachability by Positive Vertex-Centroid Moves*](https://arxiv.org/abs/2607.22981) | 对若干配置空间的正向单顶点质心移动给出目标依赖的有限步精确可达；摘要明言证明为存在性且非定量。 | 再次说明“局部正向移动 + 有限步可达”不能据为本轮独占新意。该工作研究一般点配置/简单多边形与质心原语，而非本轮固定 triangulation、每面正向和 1025² 全反传实现；具体技术边界须读正文后再用于论文优先权结论。 |
| [Bellido–Mora-Corral, Hölder homeomorphism 的 PL 逼近](https://arxiv.org/abs/0806.3366)；[平面 bi-Sobolev homeomorphism 的 PL 逼近](https://arxiv.org/abs/1509.01045) | 对各自正则类存在 P1 homeomorphism 逼近；假设和收敛范数各不相同。 | 这类存在定理通常允许随目标构造剖分，不能直接推成**固定规则细网格**上的具体 F1/F2 latent 可达性，也不说明训练成本。我们的条件 \(C^{1,1}\) 正则同伦构造是更窄、实现相关的命题，不能包装成解决一般平面 homeomorphism 逼近。 |
| [CorticalFlow](https://arxiv.org/html/2206.02374) | 对 \(h\operatorname{Lip}(v)<1\)，连续 Euler 映射 \(x\mapsto x+hv(x)\) 是 Lipschitz homeomorphism；论文明确指出仅把受流推动的**顶点**以原边/面重新连线，仍可能自交。 | 是 coarse-to-fine 正向形变与网络的直接相近方向，但它自身已经清楚区分连续流保证和离散采样网格保证。F1/F2 的目标是直接约束原面，消除这层保证缺口。不能因 CorticalFlow 报告高顶点数就把它误列为同一 fixed-grid P1 契约，也不能忽视其前向思想已有先例。 |
| [SITReg, *Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration*](https://www.melba-journal.org/papers/2024:026.html) | 用受硬幅度界约束的多层 cubic B-spline 控制点，证明各小位移的 Lipschitz 收缩/可逆性，跨尺度作复合；图像注册与反传实际实现。原文还明确指出：若把复合映射重采样到图像分辨率，其严格可逆性可能丢失；须保存因子并以**真正复合**评估才保留保证。 | 这是“正向、多分辨率、神经网络、严格拓扑”极直接的前例，不能声称本轮首创这些要素。本轮 F1/F2 特定差异是**最终一张预定三角剖分上的 P1 同胚**，每次更新在这张嵌套细图上直接保面取向，而非保存多层 B-spline 因子后动态复合。单个受收缩界的 B-spline 位移采样到顶点并取 P1 插值也很可能仍满足收缩界；故“单步固定 P1 保证”不能作为独有新意，论文须核对其离散界对三角 P1 的直接适用性。 |
| [Aigerman 等, *Neural Jacobian Fields*](https://arxiv.org/pdf/2205.02904) | 网络在任意输入网格上预测候选 Jacobian 场，再用可微、可缓存预分解的 Poisson 线性求解恢复连续 P1 网格映射；报告重视细节精度与跨三角剖分泛化。 | 这是“神经预测 + 高细节 P1 输出 + 全反传”的重要直接先例，不能把这些单项算本轮新意。该论文明确需要全局 Poisson solve；在本次所核查的构造和摘要中没有发现对所有 latent 的平面固定网格单射保证。后一句仅是已查文本范围内的观察，不是对整篇论文的否定性定理。 |
| [Freifeld 等, *Highly-Expressive Spaces of Well-Behaved Transformations*](https://openaccess.thecvf.com/content_iccv_2015/html/Freifeld_Highly-Expressive_Spaces_of_ICCV_2015_paper.html) | 用连续的分片仿射**速度场**作积分，生成可用于图像配准的微分同胚；先于本轮展示了低维 latent、正向连续变形和高表达能力的结合。 | “分片仿射”修饰速度场，积分后的变换一般不是指定固定三角网的一张 P1 函数；如果取其网格顶点再做 P1 插值，仍须额外证明逐面正向。不可把本轮称为首个正向 latent 形变方法。 |

## 本轮暂可防守的差异与仍需验证的主张

1. F1 与 F2 的局部安全机制都在**最后的同一固定 2D 网格**直接更新顶点，精确 P1 加密不改变前一层连续函数；它们不求全局网格系统。这个结构差异成立，但“新定理/新方法”需要更全面的优先权检索与严格证明。
2. 对所列独立 high32/high64/局部旋转目标，已知目标教师 latent 在 1025² 网格可达到近机器精度；这给**特定分布的可表达性**，不是普适性。F2 由零直接优化在相同高频目标上曾远差于教师，显示参数化与可训练性之间有实质距离。
3. 257²、1025²及4097²真实控制网格的forward/VJP已有记录；4097²的F1/F2双机制实验使用**已知解析teacher latent**，图像训练则使用混合F2/F1机制与特定合成多视图。速度数字只有在相似目标误差、同硬拓扑、同GPU/软件和预处理口径下才可用于优势主张。早期257² F1 image-only编码器明显不及既有Phase VI A8；后续图像反馈改善后仍需要同预算、自然图像与新种子的配对比较。详见[主报告](REPORT.md)。
4. 从有限个相近工作不能推断“尚无人在高分辨率实现 guaranteed P1 neural layer”。最终新颖性主张至多暂写为：**对固定规则2D三角网格、无需全局网格求解器的多尺度latent构造及其4097²控制顶点全VJP实测组合，尚未在以上直接文献中找到同一输出契约与规模的报告**。这是检索观察，非排他性证明；继续扩充2024–2026相关工作和公开代码复核。

研究优先级不应因上面涉及 Beltrami/准共形的传统动机而重新偏回任意面 \(\mu\) 恢复。本轮评价用 fixed-grid P1 安全、独立目标的有效自由度、图像推断和同精度计算成本。
