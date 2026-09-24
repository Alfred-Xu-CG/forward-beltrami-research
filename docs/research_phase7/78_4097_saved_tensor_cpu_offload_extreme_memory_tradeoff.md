# 4097²完整图像层CPU暂存：allocated约5.86GB，但约5.7倍慢、占用大量主机内存

## 问题和方法边界

[细层activation checkpoint](77_4097_multiview_checkpoint_memory_tradeoff.md)把4097²四视图全参数训练CUDA allocated峰值从15.934GB降到13.992GB，仍不适配8–12GB GPU。这里测试PyTorch `torch.autograd.graph.save_on_cpu(pin_memory=True)`：在前向中把自动微分需留到反向的张量转到pinned主机内存，反向时再复制回GPU。它**没有消除**保存量，只把主要容量与带宽开销从GPU转到CPU RAM与PCIe；最终4097²地图、原面拓扑证书及图像512²查询照旧。选项为[探针](../../tools/phase7_feedback_scale_probe.py)的`--offload-saved-tensors`，只包需要VJP的前向/损失/反向，推理不使用。

实验与[直接/重算对照](77_4097_multiview_checkpoint_memory_tradeoff.md)相同：AI主机空闲RTX A6000、PyTorch2.5.1+cu124、float32，四幅独立合成纹理共享一张真实形变，batch1。单例seed20270317做3次CUDA同步完整forward+image-MSE VJP；另从完全相同初始编码器和增益出发，在seed55101的32训练图像中以同样抽样序列做20步全参数image-only训练，统计训练步中位与allocated峰值。20步用于计算代价/梯度轨迹检查，不用它宣称几何学习效果。直接保存与细层重算列来自同协议已记录实验；单例热态不同运行次数，几毫秒差不作因果解释。

| VJP存储策略 | 单例热态forward+VJP | 单例VJP CUDA峰值 | 20步训练中位 | 训练CUDA峰值 |
|---|---:|---:|---:|---:|
| 直接GPU保存 | **.756s** | 15.445GB | **.761s** | 15.934GB |
| 仅两个细层反向重算 | .847s | 13.504GB | .852s | 13.992GB |
| saved tensors全部暂存CPU | 4.263s | **5.366GB** | 4.333s | **5.859GB** |

CPU暂存使本次完整4097²训练的allocated峰值低于6GB，但20步中位训练时间是直接方案的约5.69倍。补充复测的单例VJP中，CUDA **reserved峰值为8,663,334,912字节（约8.07 GiB）**，明显高于allocated；训练reserved峰值本次未测。因此本实验**不能证明它能在8 GB物理显存设备上运行**，更不能以allocated单项推断容量可行性。第一次VJP用时18.08s，较热态4.26s明显更慢，不能把一次冷态混入中位。运行中用`ps`观察到该单训练进程主机RSS约**31.3百万KiB（约29.9GiB）**；这是一个时间点的进程常驻量，**不是峰值证明**，也不全由saved tensor造成，仍足以说明显存降低需要大量主机内存/传输。主机当时有可用内存约944GiB，所以没有与他人争抢内存的迹象；不能据此假设普通工作站也能承受。

单例开启/关闭暂存的前向地图、image MSE、最小原面 \(J\) 在报告精度下相同，VJP到24个实际使用编码器参数张量及四项增益均有限；20步两种策略训练后所有编码器权重最大绝对差 \(3.73\times10^{-8}\)，两个粗增益逐项相同，两个细增益最大差 \(7.45\times10^{-9}\)。这核查了本次梯度链与训练轨迹，**不**是任意latent和活动集切换点的光滑等价定理。[20步CPU暂存权重](checkpoints/phase7_full_encoder4097_C4_offload20.pt)与[直接保存权重](checkpoints/phase7_full_encoder4097_C4_direct20.pt)可重复差异计算。

阶段结论须把“降低allocated显存”与“确实可在小GPU容纳”、以及“快、总体低内存”分开：当前全量CPU offload只证实第一项，却明显损害速度与主机内存；是否适配8–12GB物理卡仍未实测。下一步若把它用作实际训练方案，应测试**选择性**保存/重算与块式局部VJP，同时计入CUDA reserved、pinned CPU内存峰值、PCIe吞吐、batch扩展及所有训练步的拓扑接受率；不能只展示CUDA allocated一列。
