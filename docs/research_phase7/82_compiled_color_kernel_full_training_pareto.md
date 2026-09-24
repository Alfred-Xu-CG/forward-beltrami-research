# 局部颜色算子编译：4097²完整训练热态速度与显存同时改善

## 问题、方法与适用范围

F1单次更新在固定西南—东北三角网的每个颜色\(c\)计算相邻六面有向面积及从latent到可行位移的缩放，随后只更新该颜色的内点。见[安全算子及逐颜色重算](../../src/qcopt/neural_bijection/dense/colored_vertex_relaxation.py)、[数学归纳证明](23_uniform_isotopy_multilevel_approximation.md)和[重算基线](80_per_color_checkpoint_vjp_memory.md)。这里研究：能否用PyTorch 2.5.1的`torch.compile`/Inductor对**每个颜色的同一前向表达式**做算子融合，让反向按颜色重算时既减少kernel启动/中间数组，也保持同一固定原网格P1输出？这不是新拓扑定理、不是大线性系统，也不改输出三角剖分。每个浮点输出依然经过实际原面的保守正向证书；编译器可改变浮点运算次序，不能仅凭符号表达式相同声称bitwise相等。

[规模探针](../../tools/phase7_feedback_scale_probe.py)的`--compile-color-update`只编译`SafeColoredVertexRelaxation._update_color`，可与`--checkpoint-colors`组合。不同层/颜色/梯度模式会导致特殊化；默认Dynamo缓存上限8在1025²实验曾触发回退到普通执行，热态完整VJP几乎不加速。本探针把上限提高到64后重新测量。诊断性的[独立颜色算子探针](../../tools/phase7_compile_color_probe.py)在1025²、batch1、float32随机latent下，热态完整F1 forward+向底图及latent反传从.03198s降到.00982s；输出最大坐标差\(4.77\times10^{-7}\)、均方根差\(1.39\times10^{-9}\)，底图梯度最大差.00554（参考梯度最大100.83；梯度RMSE约\(1.00\times10^{-5}\)），logit梯度最大差\(1.12\times10^{-5}\)。两者最小原面Jacobian在报告精度下同为.049895。[原始独立算子日志](phase7_compile_color1025.json)含编译启动约17.7秒和显存；单算子速度收益不能直接冒充全图像层收益。

共同完整层协议：AI主机空闲RTX A6000 48GB、PyTorch2.5.1+cu124、float32；4097²=16,785,409**控制顶点**、33,554,432**原三角面**，四幅独立合成纹理共用同一真实形变，512²图像查询；32个训练例seed55101，batch1，固定20步Adam抽样序列；编码器与四级增益同时参与image-only MSE×\(10^6\)反传，学习率\(10^{-5}\)/\(10^{-4}\)。从同一编码器与已训练细增益开始；另用未参与20步的16例seed20270215做验证。时间是CUDA同步完整训练步中位，包含编码器、局部提示、P1生成、证书、重采样、损失、VJP与Adam；CUDA显存为`max_memory_allocated/reserved`换算的十进制GB。**表内全是相同count16、batch1、20步配置**，故可以比较内存常驻和速度；每行20个训练输出及16个留出输出均通过证书。

| 颜色索引/激活策略 | 热态中位训练步 | allocated峰值 | reserved峰值 | 首步/编译成本 |
|---|---:|---:|---:|---:|
| generated，普通自动微分 | .7644 s | 16.074 GB | 18.346 GB | 无编译 |
| generated，逐颜色重算，不编译 | .8766 s | 7.885 GB | 10.815 GB | 无编译 |
| generated，编译、不重算 | **.3449 s** | 8.997 GB | 10.897 GB | 首步13.63 s |
| **generated，编译＋逐颜色重算** | **.3671 s** | **6.778 GB** | **9.030 GB** | 首步25.26 s |
| generated，编译＋逐颜色重算＋≥4m元素CPU暂存 | 1.6024 s | **4.022 GB** | **6.965 GB** | 首步37.13 s；主机RSS约15.33 GiB |

编译＋逐颜色重算相对未编译同机制热态约快2.39倍、allocated约降14%，相对传统generated直接反传约快2.08倍、allocated约降58%。与编译但不重算相比，重算的训练步慢约6.4%，allocated低约24.7%，是本实验较好的速度—显存折中。加选择性CPU暂存可再降GPU峰值，但要通过PCIe搬运大张量，训练步约慢4.36倍于纯编译＋重算；这是容量折中而非无代价加速。以上比值是**该GPU、该PyTorch版本、该batch、该图像任务**的实测，不能泛化成机器无关常数。编译缓存可能影响首次运行时间，不能只报热态数字：无训练的[4097²五次VJP日志](phase7_compile64_full4097_vjp5.json)中首次推理16.86秒、首次VJP40.12秒，随后热态推理.0844秒、完整forward+VJP.3681秒；未编译逐颜色重算对应约.1516/.8660秒。20步训练的首步时间13.63—37.13秒随编译图及先前磁盘缓存不同而变，短任务总时间未必获益。

更关键的长训练复核没有只停留在20步：**300步完整image-only训练**、32训练例、batch1、编译＋逐颜色重算，300/300步输出均通过最终原面证书；独立128例seed20270317也128/128接受，query-map RMSE \(1.490054\times10^{-5}\)、image MSE \(3.388914\times10^{-9}\)、最小实际原面\(J\)约.049726。训练中位步时.3655秒，首步13.72秒（这次已有编译磁盘缓存），[原始300步记录](phase7_generated_color_checkpoint_compiled_train300.json)与[训练后权重](checkpoints/phase7_full_encoder4097_C4_generated_color_checkpoint_compiled300.pt)可复核。原[未编译300步结果](75_full_encoder_4097_training_single_vs_four_views.md)的留出map RMSE同为约\(1.49005\times10^{-5}\)；两者编码器权重最大差\(2.38\times10^{-7}\)，粗/细增益最大差\(1.34\times10^{-7}\)/\(5.96\times10^{-8}\)。由于这次`--count 128`把更多评价数据预先常驻GPU，其300步allocated峰值7.955GB**不能**与表内count16的6.778GB作纯优化方法比较；该差异主要涉及数据常驻。300步数值相近证明的是本合成任务训练轨迹基本保留，不是一般训练收敛/单图像可识别定理。

在真实4097²控制网格上，编译＋逐颜色重算的**batch4**也完成20步/80个输出，全部接受：热态步时1.0626秒、allocated24.727GB、reserved34.880GB，相比[未编译batch4](81_batch_scaling_4097_color_checkpoint.md)的2.6667秒/27.696GB/40.049GB分别改善。吞吐约3.764例/秒，但首步因batch4形状的编译约77.48秒，短20步任务总耗时反而可能高于未编译方案。[原始batch4日志](phase7_generated_color_checkpoint_compiled_b4_train20.json)和[权重](checkpoints/phase7_full_encoder4097_C4_generated_color_checkpoint_compiled_b4_20.pt)留在D盘。batch4权重与同抽样未编译版本的最大编码器差\(9.76\times10^{-7}\)，不能称位级一致。

选择性暂存＋编译＋逐颜色重算还在同一48GB卡上以`torch.cuda.set_per_process_memory_fraction`设置**7 GiB allocator上限**完整运行同协议20步，reserved峰值6.965GB（约6.49 GiB），见[受限allocator原始记录](phase7_generated_color_checkpoint_compiled_selective4m_cap7_train20.json)。这只测试了PyTorch allocator的限额；CUDA上下文、驱动、其他库和实际8GB小卡的有效容量/性能没有被模拟。故**仍未声称已在物理8GB GPU上训练成功**，同时该方案使用约15GiB主机RSS、步时1.60秒，不能称总体低内存。

局限：所有性能仍在一类RTX A6000与合成四视图下，未测不同PyTorch版本、物理小显卡或真实单图像配准；编译不改变F1/F2表示定理，不消除图像不可辨识/噪声退化。若只看“编译后.365秒”，就会遗漏分钟量级冷启动、活动集浮点差异、主机内存以及不同batch的内存峰值。
