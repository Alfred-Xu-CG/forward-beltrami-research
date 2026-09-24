# 4097²完整图像训练的batch扩展：逐颜色重算在48GB卡上跑通batch4

## 问题和公平口径

[逐颜色重算](80_per_color_checkpoint_vjp_memory.md)已把batch1四视图4097²完整训练的CUDA allocated峰值从同索引模式直接反传的16.074GB降到7.885GB。这里直接问：这个节省能否转成**实际多个样本同一步**的训练，而不是只显示单例VJP？[规模探针](../../tools/phase7_feedback_scale_probe.py)新增`--train-extra-batch`，默认1保持既有路径；batch2/4从相同32个训练例seed55101有放回抽取，四幅纹理共享每例的真实形变，但训练仍只看image MSE而不看真地图。输入四通道512²图像，输出每例4097²=16,785,409个控制顶点、33,554,432个原三角面的P1函数；不是一个4097²图像查询。batch大小改变每步处理的**独立训练例数**，没有改变单例控制网格。

AI主机启动前两块空闲的同型RTX A6000 48GB、PyTorch2.5.1+cu124、float32；batch1/2/4各做20步Adam，编码器学习率\(10^{-5}\)、四级增益\(10^{-4}\)、image MSE×\(10^6\)。直接batch2与重算batch2使用相同抽样序列与初值；batch4由于每步例数不同，**不是**同一优化轨迹。步时CUDA同步取中位，包含编码器、稠密固定P1前向、512²四通道图像查询/损失、全参数VJP、实际浮点原面证书与Adam；吞吐量为`batch/中位步时`，不是纯卷积吞吐。allocated/reserved均为PyTorch CUDA峰值，十进制GB；主机CPU内存不在显存数字中。batch2直接实验在GPU2及GPU7各重复一次，时间1.292/1.294s、峰值相同；batch4重算同时/单独运行时间2.662/2.667s、峰值相同，表中取**GPU7单独运行**数值以降低共享资源干扰。

| 策略 | batch | 20步中位/步 | 每秒训练例 | allocated峰值 | reserved峰值 | 接受的训练输出 |
|---|---:|---:|---:|---:|---:|---:|
| generated索引、直接反传 | 1 | .7644 s | 1.308 | 16.074 GB | 18.346 GB | 20/20 |
| generated索引、逐颜色重算 | 1 | .8766 s | 1.141 | **7.885 GB** | 10.815 GB | 20/20 |
| generated索引、直接反传 | 2 | **1.2936 s** | **1.546** | 28.817 GB | 34.037 GB | 40/40 |
| generated索引、逐颜色重算 | 2 | 1.4649 s | 1.365 | **14.488 GB** | **19.701 GB** | 40/40 |
| generated索引、逐颜色重算 | 4 | 2.6667 s | 1.500 | **27.696 GB** | 40.049 GB | 80/80 |

在batch2同任务/同初值对比，逐颜色重算使allocated约减49.7%、reserved约减42.1%，中位步时增加约13.2%。两个20步[直接权重](checkpoints/phase7_full_encoder4097_C4_generated_direct_b2_20.pt)与[重算权重](checkpoints/phase7_full_encoder4097_C4_generated_color_checkpoint_b2_20.pt)的编码器最大绝对差\(7.45\times10^{-8}\)，两组增益完全相同；这是本次轨迹核查而非普遍的bitwise等价定理。batch4重算在单块48GB卡上完成20步/80个实际同胚输出，训练allocated27.696GB、reserved40.049GB。**没有**运行batch4直接反传，也不从batch2线性外推其能否装入同一卡。batch2/4吞吐分别比batch1重算高约20%/31%；但batch4 reserved已接近该48GB卡的容量，不能说它低显存。

所有训练输出经实际float32顶点上的保守原面符号筛选，未回退；每组20步后各有独立16例seed20270215复核，全部接受。不同batch的20步见到20/40/80次有放回样本，因此留出map/image数字**不能**用于宣称batch4优化一定优于batch1。原始配置/峰值/评价：[batch2直接，GPU7单独](phase7_generated_direct_b2_train20_gpu7.json)、[batch2重算](phase7_generated_color_checkpoint_b2_train20.json)、[batch4重算，GPU7单独](phase7_generated_color_checkpoint_b4_train20_gpu7_solo.json)；另有GPU2/并行重复日志，供交叉核查。

局限：资源仍是48GB单卡，未做真实8/12/24GB设备、batch8或长时间训练；GPU allocator reserved与物理显存使用不同。该结果回答的是固定P1神经层在密控制网格上的**实际batch可扩展性**，不消除单图像形变不可辨识和噪声下地图误差，也不把20步性能实验当成已训练模型的泛化结论。
