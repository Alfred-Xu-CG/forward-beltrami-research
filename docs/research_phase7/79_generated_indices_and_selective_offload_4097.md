# 4097²动态图索引与选择性暂存：初始化省内存，训练显存/速度仍不可兼得

## 问题、具体修改和有效对照

在固定西南—东北对角的三角网格，F1四着色安全顶点更新的每个内点恰有六个相邻原面。原实现对每个颜色预先保存顶点索引、局部latent索引，以及六条相对边的两个端点索引。后者是每个内点12个`int64`数，4097²层单独就约1.61 GB。新选项`SafeColoredVertexRelaxation(index_mode="generated")`只长驻颜色顶点索引和12个固定偏移；每次颜色更新临时计算`opposite=vertices+offsets`和对应latent索引。数学算子、四组顺序及局部安全缩放完全不变。默认仍为`buffered`。实现见[源码](../../src/qcopt/neural_bijection/dense/colored_vertex_relaxation.py)，9²/10²网格逐值相等的输出及logit VJP见[回归测试](../../tests/test_phase7_generated_incident_indices.py)。

第二项修改是[规模探针](../../tools/phase7_feedback_scale_probe.py)的`--offload-min-elements 4000000`：只在自动微分保存一个不少于四百万元素的CUDA张量时复制到pinned CPU内存，反向按需搬回。阈值按**元素数**而不是字节数，故不代表固定字节预算；同时保留全量`--offload-saved-tensors`和细层`--checkpoint-extra`作对照。CPU暂存改变存放位置，不改变局部同胚构造；其VJP等价只在实测轨迹中核实，不能声称活动集切换处有全域光滑梯度。

公平训练设置与[直接/重算基线](77_4097_multiview_checkpoint_memory_tradeoff.md)完全相同：AI主机RTX A6000，PyTorch2.5.1+cu124，float32，4097²=16,785,409控制顶点、33,554,432**原面**，512²图像查询，四幅独立合成纹理共享同一真实形变，32个训练例seed55101、batch1、固定抽样序列、image MSE×\(10^6\)、20步Adam、编码器学习率\(10^{-5}\)、四级增益学习率\(10^{-4}\)，从**同一编码器和已训练细层增益**起步。对照的单独16例测试seed20270215从未参与这20步。计时为同步的完整训练步中位，包含编码器、局部提示、P1生成、原面证书、查询和反传；显存来自`torch.cuda.max_memory_allocated/reserved`，是字节数换算的十进制GB，不能直接等同显卡物理容量。进程RSS取`resource.getrusage(...).ru_maxrss`，是单进程历史峰值而非仅pinned saved tensor字节数。两条新训练的[选择性暂存原始JSON](phase7_generated_selective4m_train20_fair.json)及[叠加重算原始JSON](phase7_generated_selective4m_checkpoint_train20.json)留在D盘仓库。

| 4097² C4完整训练策略 | 初始化allocated | 20步中位/步 | 训练allocated峰值 | 训练reserved峰值 | 进程RSS历史峰值 |
|---|---:|---:|---:|---:|---:|
| buffered，直接保存 | 约2.66 GB（单例setup） | .7608 s | 15.934 GB | 未记录 | 未记录 |
| generated，直接保存 | 约.50 GB（16例setup） | .7644 s | 16.074 GB | 18.346 GB | 约1.55 GiB（该运行） |
| generated，选择性暂存≥4m元素 | .502 GB | **3.6114 s** | **5.5115 GB** | **8.4515 GB** | **26.92 GiB** |
| generated，选择性暂存＋细层重算 | .502 GB | .9803 s | 13.5622 GB | 15.9195 GB | 约2.65 GiB |

上表`buffered`的初始化取单例实验而非16例训练，所以不作严格同一setup的差值推断；`generated`两条20步实验中的常驻训练数据一致。单例相同配置的setup对照更清楚：buffered约16.35 s/2.662 GB allocated，generated约.63 s/.345 GB；已在同型号GPU上看到几乎相同的热态推理时间。**这项优化主要降低常驻索引和构建时间，不自动降低完整反传峰值**：动态产生的gather索引仍会被自动微分保存，generated直接VJP allocated实测15.580 GB，对比buffered 15.445 GB。上表也出现相同现象。细层重算时少有达到4m阈值的saved tensor可供暂存，合用方案反而比单用细层重算13.992 GB/约.852 s更差；两种技巧不可假定收益可加。

选择性暂存方案比全量CPU暂存[既有结果](78_4097_saved_tensor_cpu_offload_extreme_memory_tradeoff.md)的4.333 s训练步稍快，allocated却不能像后者那样低至5.859 GB，因为两种方案的索引模式与数据setup不同，不能把这些数字解读为纯阈值消融。公平generated＋选择性暂存运行的20步**全部通过最终浮点原面证书**；对照16例亦全部接受，最小原面Jacobian（用float64对实际float32顶点重算）约.04999，query-map RMSE \(1.50884\times10^{-5}\)。与buffered直接保存的[20步权重](checkpoints/phase7_full_encoder4097_C4_direct20.pt)相比，[generated＋选择性暂存](checkpoints/phase7_full_encoder4097_C4_generated_selective4m20.pt)编码器权重最大差\(4.84\times10^{-8}\)，粗增益完全相同，细增益最大差\(7.45\times10^{-9}\)；[再叠加重算](checkpoints/phase7_full_encoder4097_C4_generated_selective4m_checkpoint20.pt)的对应差为\(2.98\times10^{-8}\)、零、\(7.45\times10^{-9}\)。这支撑相同训练梯度轨迹，却不是任意输入/不同显卡的bitwise等价定理。

结论：已能减少4097²完整训练的**allocated**显存到约5.51 GB，且所有实际20步拓扑接受，但reserved约8.45 GB、主机RSS约26.9 GiB、训练慢约4.75倍；**没有在8 GB或12 GB物理卡上验证**。更重要的未决瓶颈是无需把大批自动微分激活搬过PCIe的局部反向/融合实现，以及小卡上CUDA上下文、allocator、batch和长训练的真实容量测试。不能把“初始化只要.5 GB”误称为“完整VJP只要.5 GB”。
