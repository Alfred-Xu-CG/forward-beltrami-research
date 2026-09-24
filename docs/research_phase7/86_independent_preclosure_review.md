# Phase VII 结论独立复核记录（窗口末前）

本文不是替代[自包含主报告](REPORT.md)的第二套总结。复核方法是从源码、原始JSON、原始定理假设和重新运行的实验各自出发，检查主报告最重要的断言是否越过证据；没有外部审稿人或第二个代理参与，故“独立”限于**独立计算/重跑及不同证据路径**，不冒称同行评议。目标模式的21小时结束时还须核对时间、Git同步和剩余任务。

## 逐项判断

| 主张 | 复核依据 | 结论与必要限定 |
|---|---|---|
| F1/F2输出一张固定原网格P1图 | F1 [`_update_color`](../../src/qcopt/neural_bijection/dense/colored_vertex_relaxation.py)、F2 [二次patch面积预算](../../src/qcopt/neural_bijection/dense/patch_field.py)、[F1证明](23_uniform_isotopy_multilevel_approximation.md)、[F2证明](26_f2_uniform_isotopy_approximation.md) | 在正向起点、固定边界、全部有限latent、**精确算术**下，每子步保原三角面正向。精确dyadic加密的细三角形仍位于父三角形，连续P1函数不变；未把连续复合再采样当作P1。全局同胚还需边界一一对应及盘域假设，不能只引用局部面积。 |
| 实际float32输出仍拓扑安全 | [`certify_p1_or_identity`](../../src/qcopt/neural_bijection/dense/multilevel_forward_p1.py)逐实际原面/边界/有限性检查；4097²独立teacher重跑与64张大latent压力测试 | 在所支持网格、dtype及数值模型下，证书接受才输出候选，否则输出精确恒等图。**回退边界不可微且回退样本latent梯度为零**；数学局部安全是主要保证，证书是防浮点舍入的最后一道实现筛选。未证明低精度/任意硬件上永不回退。 |
| F1/F2对统一光滑同伦类的定量逼近 | [F1](23_uniform_isotopy_multilevel_approximation.md)、[F2](26_f2_uniform_isotopy_approximation.md)的面积、边长、时间速度界 | 给定统一\(m,L,K,M\)及边界恒等的\(C^{1,1}\)同伦，可选择**类别共同**的足够细seed和有限seed周期，每次二分后一个F1颜色周期或F2交错周期到达目标顶点样本，精确算术误差\(O(h^2)\)，工作\(O(V)\)。这是teacher latent的存在性，不是训练好的固定小CNN对该类的统一逼近。代码内正的浮点guard不等同理想实数算法；极小\(m\)还需更高精度/数值证书。 |
| 全体边界固定同胚的\(C^0\)稠密 | [相对边界光滑化与缩放共轭](30_all_boundary_homeomorphisms_c0_density.md)、[紧支撑光滑同伦](27_compact_smooth_diffeomorphism_corollary.md)、[Hatcher原文](https://ems.press/content/serial-article-files/50953) | 只能对**所有架构规模之并、逐目标**宣称；不含统一速度、固定网络深度或图像逆推。复核发现应明确该命题取`minimum_jacobian=None`，即无统一\(0.05h^2\)绝对floor；[密度文档](30_all_boundary_homeomorphisms_c0_density.md)和主报告已修正。实验可设0.05，不能反过来证明全群密度。 |
| 两种机制确实上4097²并向全部latent反传 | [F1原始重跑](phase7_teacher4097_F1_independent_recheck.json)、[F2原始重跑](phase7_teacher4097_F2_independent_recheck.json)、[协议](74_two_distinct_forward_mechanisms_4097_teacher_vjp.md) | 同一独立解析\(F_{.08}\)与teacher：两者顶点RMSE\(3.436\times10^{-10}\)、min\(J=.5660\)；全latent forward+VJP中位.5602/1.1391秒。该数据是已知目标teacher，不是未知图像几何恢复。早期报告没链接原始JSON；此次独立重跑补齐。 |
| 混合层4097² image-to-latent训练可行 | [原首视图300步](82_compiled_color_kernel_full_training_pareto.md)、[四视图对称融合300步](85_permutation_invariant_coarse_latent_fusion.md)、[新种子确认日志](phase7_confirm_trimmed_C4_firstspots_seed20270823.json) | 300/300完整image-only训练和128/128留出均通过证书；对称融合在“一幅spots＋三幅独立standard”的后续新seed128例地图RMSE\(3.208\times10^{-5}\)，原首视图为.004363。这个改善是**特定合成多视图/输入结构**，并非自然单图注册的普适结果；四幅重复同一图仍无法增加观测信息。 |
| 编译和内存优化不改变科学结论 | [热态/冷态及显存](82_compiled_color_kernel_full_training_pareto.md)、[大latent全VJP差](83_compiled_color_extreme_latent_stress.md)、[融合batch/受限allocator](85_permutation_invariant_coarse_latent_fusion.md) | `torch.compile`实测可显著加速并降低saved tensors，但首步有十秒到数十秒编译成本；4097²随机latent的编译/未编译VJP相对L2差最高0.71%，不能称梯度位级等价。allocated/reserved为PyTorch度量；7GiB allocator限额**不是物理8GB设备证据**，CPU暂存以速度及主机RSS换GPU容量。 |
| 新颖性可表述为已证领域首创 | [主文献定位](06_prior_art_boundary.md)：TutteNet、SITReg、Generative Escher Meshes、Neural Jacobian Fields等原始材料 | **不能**这样表述。正向、多尺度、神经、P1和严格拓扑分别已有前例；较可防守的是具体固定原网格P1输出契约、局部多尺度安全原语、特定统一光滑同伦逼近证明和1600万控制顶点全VJP组合。未做覆盖全部2024–2026论文及代码的系统性优先权检索。 |

## 可复算检查与未解决事项

- 重新运行[同一4097²解析teacher脚本](../../tools/phase7_verify_isotopy_pyramid.py)后F1/F2误差和中位时长与此前[74号表](74_two_distinct_forward_mechanisms_4097_teacher_vjp.md)一致到运行波动量级，原始重跑JSON已留D盘。
- 本机将仓库`src`加入`PYTHONPATH`后，重跑全部`tests/test_phase7_*.py`得到`64 passed`（22.21秒）；`src/qcopt/neural_bijection/dense`及修改过的探针可`compileall`。扫描本阶段Markdown相对文件链接后目前`MISSING_COUNT=0`，顺手修复两处第53号文档的单复数拼写误链。这些检查不证明所有数值/理论断言，只提供可复现的一致性底线。
- 剩余最重要的科学缺口：真实图像/不同成像模态上的单图或多图几何真值验证；不同GPU与物理小显卡上的容量和吞吐；对活动面切换的梯度稳定性分析；从大表达性teacher类到可学习低维latent的**有效**逼近复杂度；更完整的先例全文比较。对称粗latent融合在一坏三好的外观偏移中很有效，但[独立加性噪声配对](85_permutation_invariant_coarse_latent_fusion.md)几乎没有收益，不能把某一路线的teacher精确性或合成四视图成功替代这些证据。
