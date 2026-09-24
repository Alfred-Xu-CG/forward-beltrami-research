# 四颜色局部反向重算：4097²完整训练约减半allocated显存、约15%增时

## 计算图分解及其不改变的数学对象

F1一次安全更新依序执行四组不共享原三角面的内点：\(Y^{c+1}=U_c(Y^c,z)\)，\(c=0,1,2,3\)。每个\(U_c\)只改变该组顶点的像，局部面积截断保证在精确算术下每步仍是同一原规则网格上的P1同胚。普通自动微分把四组的中间算子激活同时留在GPU，形成大常数的\(O(V)\)存储。本次在[安全层](../../src/qcopt/neural_bijection/dense/colored_vertex_relaxation.py)增加可选`checkpoint_colors=True`：前向只记录必要的颜色边界状态，反向从第四组到第一组逐组重新执行相同的\(U_c\)计算局部VJP。它**不**改变任一顶点正向值、安全缩放或原面证书；只是以重算交换自动微分激活。与[整个2049²/4097²反馈模块一次重算](77_4097_multiview_checkpoint_memory_tradeoff.md)相比，切割点更细，故反向不会同时建立所有四组的内部计算图。索引使用[动态生成模式](79_generated_indices_and_selective_offload_4097.md)，避免常驻巨大的相对边表。该构造仍只是\(O(V)\)复杂度的**常数**改进，并非子线性显存定理。

9²/10²动态索引与11²按颜色重算的[单元测试](../../tests/test_phase7_generated_incident_indices.py)比较输出和向输入地图、latent的VJP，双精度逐值或1e-12容差匹配。大网格实验为AI主机空闲RTX A6000、PyTorch2.5.1+cu124、float32、4097²=16,785,409控制顶点/33,554,432原三角面、四幅独立合成纹理、512²图像查询、batch1。训练仍是32例seed55101、20步Adam、image-only MSE×\(10^6\)，编码器学习率\(10^{-5}\)、增益\(10^{-4}\)，从与[直接基线](77_4097_multiview_checkpoint_memory_tradeoff.md)相同的编码器和已训练细级增益出发。全部计时CUDA同步，训练步包括编码器、细层、图像采样、原面证书、损失和反传；allocated/reserved为CUDA字节数换算的十进制GB。以下对照没有改变**控制网格规模**。

| 4097² C4完整训练策略 | 20步中位/步 | allocated峰值 | reserved峰值 | 主机RSS历史峰值 |
|---|---:|---:|---:|---:|
| buffered索引、直接反传 | **.7608 s** | 15.934 GB | 未记录 | 未记录 |
| buffered索引、整细层重算 | .8520 s | 13.992 GB | 未记录 | 未记录 |
| generated索引、直接反传 | .7644 s | 16.074 GB | 18.346 GB | 约1.55 GiB |
| **generated索引、按颜色重算** | **.8766 s** | **7.885 GB** | **10.815 GB** | **约1.55 GiB** |
| generated索引、按颜色重算＋仅≥4m元素暂存CPU | 2.1070 s | **4.575 GB** | **8.198 GB** | **约15.16 GiB** |

按颜色重算相对同为generated索引的直接训练，allocated约降50.9%，中位步时约增14.7%；对buffered直接基线则allocated约降50.5%，增时约15.2%。它不是“用稀疏小控制网插值得到4097²”：最终输出仍是所有16,785,409点的同一P1函数，全部33,554,432原面逐个检查。独立单例seed20270317的完整forward+VJP做3次，按颜色重算热态中位.8660 s、allocated7.388 GB、reserved10.138 GB；编码器24组有效梯度、粗/细增益梯度均有限，[原始VJP记录](phase7_generated_color_checkpoint_vjp3.json)。

仅选择性CPU暂存时训练3.611 s/5.512 GB allocated/8.452 GB reserved/[约26.9 GiB RSS](79_generated_indices_and_selective_offload_4097.md)；与按颜色重算**叠加**，时间改善到2.107 s且allocated再降为4.575 GB，但CPU内存仍显著。单例叠加策略热态完整VJP为2.260 s/4.081 GB allocated/6.963 GB reserved，[记录](phase7_generated_color_checkpoint_selective4m_vjp3.json)。不能由单次VJP显存替代训练显存。

按颜色重算单独20步[权重](checkpoints/phase7_full_encoder4097_C4_generated_color_checkpoint20.pt)与buffered直接基线相比，编码器最大权重差\(3.73\times10^{-8}\)，粗增益完全相同、细增益最大差\(7.45\times10^{-9}\)；叠加CPU暂存的[权重](checkpoints/phase7_full_encoder4097_C4_generated_color_checkpoint_selective4m20.pt)有相同量级差。两条20步训练每步最终浮点证书都接受，独立16例seed20270215也16/16接受，后者最小实际原面\(J\)约.04999、query-map RMSE约\(1.50884\times10^{-5}\)。[单独重算训练JSON](phase7_generated_color_checkpoint_train20.json)和[叠加训练JSON](phase7_generated_color_checkpoint_selective4m_train20.json)含精确配置、计时与峰值。这里的训练步数是**内存/梯度轨迹**实验，不足以证明更好的图像学习或跨域泛化；在`min`活动集切换点也不能宣称光滑VJP。

额外用PyTorch的`set_per_process_memory_fraction`在同一**48GB**卡上给allocator设8 GiB上限，叠加方案完成相同20步，reserved峰值仍8.198 GB（约7.635 GiB），见[受限allocator日志](phase7_generated_color_checkpoint_selective4m_cap8_train20.json)。这说明**该运行的PyTorch allocator**可在模拟上限内工作；CUDA上下文、库工作区、驱动及实际小卡架构未被这个API模拟。因此依然**没有在8GB物理GPU证明可训练**。下一步值得在真实8/12GB卡上实测，同时寻找不依赖高达约15GiB主机RAM的局部VJP或融合核。
