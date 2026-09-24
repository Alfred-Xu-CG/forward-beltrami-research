# 四视图对称粗 latent 融合：4097²全参数训练与分布偏移对照

## 问题与构造

[84号配对实验](84_4097_trained_multiview_channel_order_ood.md)发现：四幅观测中只有首通道换成稀疏光斑，其余三幅仍有标准独立纹理时，原网络的4097²地图误差可从\(1.50\times10^{-5}\)暴涨到.00455；只把同一组四幅图重新排列，让粗编码器读另一幅，误差就回到\(3.33\times10^{-5}\)。原因是原粗编码器只读首图，虽然细层局部光度提示汇总了全部通道。这里试验一个直接针对该瓶颈的候选层，而**不修改F1/F2拓扑算子**。

设共享的粗编码器对第\(c\)幅固定/移动图像对输出各尺度 latent 张量\(z^{(c)}=(z^{(c)}_j)_j\)。对四视图、每个latent坐标\(j\)，将四个实数排序为\(z^{[1]}_j\le z^{[2]}_j\le z^{[3]}_j\le z^{[4]}_j\)，定义**去极值平均**
\[
\bar z_j=\frac{z^{[2]}_j+z^{[3]}_j}{2}.
\]
对粗种子和各级粗latent都做这一步，随后运行原有的精确dyadic P1加密和局部安全更新；细层2×2光度提示仍按所有通道汇总。该融合在精确算术下对输入通道排列不变，但在排序值相等处不可微；其他几乎处处可用标准自动微分，梯度传到当前居中的两个通道。**无论融合得到什么有限实数latent，后续安全解码器仍由其自身的不变量保证固定原网格P1同胚**。

它还有一个很窄但严格的**单坏通道坐标界**：若某一坐标上的三个可信通道值均在\([a_j,b_j]\)，第四个值任意大或小，则排序后的第2/3个值仍在\([a_j,b_j]\)，所以\(\bar z_j\in[a_j,b_j]\)。证明只是四个数中最多一个能落在该区间下方、最多一个能落在上方。若三份可信latent逐坐标都离某个teacher latent不超过\(\varepsilon\)，融合latent同样如此；但本实验**没有证明**这些单通道预测本来接近teacher，也没有从latent误差界自动得到地图误差界或噪声鲁棒性。坏值若落在中间，它仍可能参与平均；观测分布、编码器系统偏差与安全解码器的局部敏感性仍需单独分析。

实现入口是[phase7_feedback_scale_probe.py](../../tools/phase7_feedback_scale_probe.py)的`--coarse-view-fusion trimmed`。同一入口还提供`first`原方案、`mean`对照。编码器共享权重，四个视图合并为一个batch送入一次前向，再按视图维度融合latent；并非生成四张P1地图后平均，后者没有所需的拓扑理由。输出始终是一张4097²固定网格的P1图。

## 公平训练协议和资源

原方案与新方案都从相同的[训练前检查点](checkpoints/phase7_extra_gains_4097_float32_train500.pt)与[粗图像模型](checkpoints/phase7_hybrid_feedback_w3_float32_train500.pt)开始，用相同32张独立`high128`真实形变/四视图训练图（seed55101）、同一300步batch1 Adam抽样序列，粗编码器学习率\(10^{-5}\)，各级增益\(10^{-4}\)，优化目标为**仅图像MSE乘\(10^6\)**；真地图不进入反传。4097²=16,785,409**控制顶点**、33,554,432原三角面；每图512²图像查询。float32、AI主机48GB GPU7、PyTorch2.5.1+cu124；F1采用generated索引、逐颜色checkpoint和局部`torch.compile`。新方案完整训练原始记录及[权重](checkpoints/phase7_full_encoder_C4_trimmed_compiled300.pt)留在仓库，原方案为[82号记录](phase7_generated_color_checkpoint_compiled_train300.json)。

| 同协议300步 | 训练热态中位步时 | CUDA allocated峰值 | CUDA reserved峰值 | 训练输出接受 | 新128例seed20270317 map RMSE / 接受 |
|---|---:|---:|---:|---:|---:|
| 只读首通道粗编码器 | .36555 s | 7.955 GB | 10.234 GB | 300/300 | 1.49005e-5 / 128/128 |
| 四视图去极值平均粗latent | .36908 s | 8.075 GB | 10.127 GB | 300/300 | 1.42438e-5 / 128/128 |

allocated/reserved为PyTorch峰值，**含128例评价数据常驻**，不是物理GPU总占用；表的同配置可直接对照。两种首步编译分别约13.72/13.42秒，已有磁盘编译缓存，不能推广冷启动。训练步时的约1%差异小于一般跨进程性能噪声，故这里只能说**未见明显热态代价**，不能断言融合总是等速；粗CNN远小于4097²细网格反传开销。两种模型都在300步和128例留出中通过实际浮点原面证书，最小留出归一化面积约.0497；未在物理8GB卡验证。

完整**batch1推理**另以相同16例数据常驻设置、各6次CUDA同步运行测量，范围含四视图编码、全部P1层、证书、512² dense query插值及图像重采样，不含数据生成、模型构造和编译。丢开每进程第1次特殊化/编译后，后5次中位：首通道方案.08541秒，对称trimmed方案.08868秒，约**3.8%额外热态推理时间**；本次allocated峰值3.206/3.250GB。首计时分别7.36/7.44秒，受既有磁盘缓存影响，也不代表真正无缓存冷启动。原始[首通道6次](phase7_full_forward_timing_C4_firstview_4097_6repeats.json)与[trimmed6次](phase7_full_forward_timing_C4_trimmed_4097_6repeats.json)保留每次耗时。此小差异仅是该GPU、PyTorch版本的单batch实测，不能泛化为复杂度常数。

另用同一训练前检查点、同一32例但**batch4、20步、16例评价**复核容量：[trimmed原始日志](phase7_full_encoder_C4_trimmed_compiled_b4_20.json)及[权重](checkpoints/phase7_full_encoder_C4_trimmed_compiled_b4_20.pt)显示20/20训练步的80个输出与16/16新例全部接受，热态训练步1.0906秒、allocated23.861GB、reserved33.387GB；原首通道同协议[日志](phase7_generated_color_checkpoint_compiled_b4_train20.json)为1.0626秒/24.727GB/34.880GB。两次均在48GB GPU而非物理24/32GB卡。trimmed首步14.10秒与旧batch4的77.48秒差异主要受已有编译磁盘缓存/特殊化影响，不能当架构冷启动优势。这里只能确认batch4可运行，不能由一对进程断言普遍的速度/显存排序。

同协议batch1、20步还运行了**选择性CPU saved-tensor暂存＋7GiB PyTorch allocator模拟上限**：[原始日志](phase7_full_encoder_C4_trimmed_compiled_offload_cap7_20.json)及[权重](checkpoints/phase7_full_encoder_C4_trimmed_compiled_offload_cap7_20.pt)显示20/20训练与16/16新例接受，热态1.647秒/步、CUDA allocated4.082GB、reserved6.531GB，进程RSS历史峰值约16.08GiB。相比[原首通道同协议](82_compiled_color_kernel_full_training_pareto.md)的1.602秒/4.022GB/6.965GB，改进并非无成本；即使allocator限额通过，**仍未在物理8GB或12GB设备验证**。这项资源测试只是说明新输入融合没有把已有容量折中破坏到无法运行。

## 同地图的新外观/新形变评价

下表固定**各自训练后**的权重，不重训；新地图种子20270531，每行128例，RMSE定义为512²查询点二维地图向量的均方根，image MSE为所有通道及像素的配准亮度平方差均值。`first spots + 3 standard`严格保持四个独立图像通道，只有首通道换成稀疏光斑；`4 duplicate standard`把同一标准纹理复制四份，不能增加观测方向。

| 评价条件 | 首通道方案 map RMSE | 去极值平均方案 map RMSE | 去极值平均方案 image MSE | 新方案原面证书 |
|---|---:|---:|---:|---:|
| 四独立标准纹理 | 1.49553e-5 | **1.43255e-5** | 2.59076e-9 | 128/128 |
| 未训练过的三载波`high128_tri`，四标准纹理 | 1.38476e-5 | **1.31994e-5** | 1.80246e-9 | 128/128 |
| 未训练过的局部包`high128_tiles`，四标准纹理 | 1.36077e-5 | **1.29483e-5** | 1.57855e-9 | 128/128 |
| 首幅spots、余三幅独立标准 | .00455307 | **.00003204** | 2.67878e-9 | 128/128 |
| 四幅重复同一标准纹理 | 未在本协议重测 | **.00049063** | 2.16403e-8 | 128/128 |

固定四幅独立standard纹理的同组新128例，另外对固定图和移动图各通道加入**独立高斯噪声**\(\sigma=.001\)（两模型使用相同随机种子和噪声实现），不重训时[首通道日志](phase7_trained_firstview_C4_noise001_new128.json)与[trimmed日志](phase7_trained_trimmed_C4_noise001_new128.json)分别得到map RMSE **.000171138 / .000171245**、各128/128通过证书。差异很小且本次trimmed略差，故不能把对一幅稀疏分布外纹理的鲁棒收益推广为独立加性噪声鲁棒收益；噪声导致的地图误差仍比无噪声大一个数量级。

新方案[完整训练日志](phase7_full_encoder_C4_trimmed_compiled300.json)、[新种子标准纹理](phase7_trained_trimmed_C4_standard_new128.json)、[首通道spots](phase7_trained_trimmed_C4_firstspots_new128.json)、[重复标准通道](phase7_trained_trimmed_C4_duplicate_standard_new128.json)可复算。作为未重训结构消融，旧首通道权重直接应用去极值平均时，标准/首通道spots分别为\(1.43496\times10^{-5}\)/\(3.22347\times10^{-5}\)；旧权重直接作简单四通道**均值**时，spots为\(9.27398\times10^{-5}\)。[相应日志](phase7_compiled_trained_C4_firstspots_trimmed_new128.json)、[均值日志](phase7_compiled_trained_C4_firstspots_mean_new128.json)、[标准trimmed日志](phase7_compiled_trained_C4_standard_trimmed_new128.json)说明结果不是只能靠额外300步训练才出现，但这些同一测试集上的消融不构成独立验证。首通道spots的未重训16例烟测[单列](phase7_compiled_trained_C4_firstspots_trimmed16.json)，没有用它调权重。

两种新形变族的训练后评价原始日志：[tri](phase7_trained_trimmed_C4_high128_tri_new128.json)、[tiles](phase7_trained_trimmed_C4_high128_tiles_new128.json)；旧首视图的对应值见[84号配对报告](84_4097_trained_multiview_channel_order_ood.md)。它们改变了高频几何，但仍共享标准合成纹理机制，不能代表自然图像内容或模态外推。

把已训练trimmed方案的**同一四幅图像**轮换通道后，[128例原始日志](phase7_trained_trimmed_C4_firstspots_rotated_new128.json)的地图RMSE为\(3.203944996\times10^{-5}\)，未轮换为\(3.203945205\times10^{-5}\)；实际最小原面\(J\)分别为.0496321/.0496352。宏观误差几乎不受顺序影响，但浮点运算和CNN的batch执行不是逐位排列不变量；不能把数学对称性写成bitwise保证，两次均须各自通过最终原面证书。

**设计定稿后的新种子确认。** 上面seed20270531参与发现首通道问题及选择trimmed融合，不能当独立盲测。于是保持已训练的两份检查点及所有参数不动，另用从未用于这些决策的seed20270823生成128张新地图及四幅新纹理，其中只有第一幅为spots、余三幅独立standard。原首通道方案的query-map RMSE为**.00436303**、image MSE .00081526；对称trimmed方案为**.00003208**、image MSE\(2.78182\times10^{-9}\)，约136倍几何误差差距。两者原面证书各128/128接受，最小实际归一化面积分别.047249/.049676。[原方案原始记录](phase7_confirm_original_C4_firstspots_seed20270823.json)与[trimmed原始记录](phase7_confirm_trimmed_C4_firstspots_seed20270823.json)单列，不能把两批128例合并声称为256次独立方法选择后的盲测；第二批才是预定结构下的确认。

## 结论边界

本组配对证据支持：把原先非对称的粗图像入口改为**共享编码器＋对称latent融合**，能在仍有三幅独立且有纹理的图像时大幅减少“坏首通道”引起的几何失败，同时保持已测标准族精度和4097²全链可训练性；主拓扑保证没有放松。它不解决四幅都低信息的情况，重复同一幅标准图时地图误差仍约\(4.91\times10^{-4}\)，也未验证自然图像、未知模态、不同图像强度尺度或多个坏通道。去极值排序本身在活动顺序切换点不可微，证书回退也有分支；“可反传”指所测训练轨迹及一般位置的一阶梯度。下一步应在独立纹理/真实多模态数据上比较置信度加权融合、对称特征融合与这条简单基线，并报告实际物理小显卡和batch扩展。
