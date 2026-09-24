# 4097² 已训练四视图层：新形变族、稀疏纹理与通道顺序

## 决定性问题

[82号实验](82_compiled_color_kernel_full_training_pareto.md)已在4097²固定P1网格完成四视图image-only训练，但网络结构不完全对称：粗尺度卷积编码器只读取**第一**图像通道，细尺度局部光度提示才按所有四通道平均。因此即使四幅图像提供了充分的局部方向信息，只要首通道超出训练纹理分布，整体几何推断仍可能失败。本文用**同一组图像只改通道顺序**，把这种编码器瓶颈与拓扑安全器区分开。

## 固定设置与度量

完全固定已训练的[300步编译＋逐颜色重算检查点](checkpoints/phase7_full_encoder4097_C4_generated_color_checkpoint_compiled300.pt)，不在下列测试集上优化任何参数。训练只有32个不同解析目标地图，四幅合成纹理，300步Adam，`image-only`。评价新随机种子20270531，每组128张目标地图，4097²=16,785,409个**控制顶点**和33,554,432个原三角面；512²是**图像查询数**，不是控制分辨率。AI主机空闲48GB GPU7、PyTorch2.5.1+cu124、float32、batch1；此处只报告评价误差而不把含编译/数据设置的推理耗时用于跨方法速度比较。

`high128` 是训练时的目标形变族但新地图种子；`high128_tri` 改为三个高频载波，`high128_tiles` 改为空间局部高频包，两者均未参与300步训练。`standard` 为训练时纹理生成机制，新的种子保证图像内容未见。稀疏 `spots` 是六个有符号高斯光斑和很弱线性背景；固定图始终从对应移动图按**同一真实地图**重采样得到。`first spots + 3 standard` 中只有第一个通道改为 spots，另三个仍是彼此独立的 standard 纹理；`rotated` 把这个第一个通道移到末尾，**观测图像集合完全不变**，使粗编码器改读原来的第二个 standard 通道。局部2×2提示的通道平均和像素MSE除浮点求和顺序外不变。`4 duplicate spots` 将同一张 spots 图复制四次，不能当四幅独立观测。

地图误差为所有128张地图、所有512²查询点的二维坐标向量RMSE：\(\sqrt{(128Q)^{-1}\sum_{b,q}\|\hat F_b(q)-F_b(q)\|_2^2}\)。图像MSE是所有样本、通道、像素的移动图按预测图采样与固定图的平方差平均。`min J` 把**实际输出的float32顶点**转成float64，逐原三角面计算有向面积与原面积之比，再取128例全体最小值；接受数由正式浮点原面证书给出，失败时返回恒等图。指标均为整组`cohort_validation`，不是第一例的单独值。

| 未重训评价集 | query-map RMSE | image MSE | min J | 原面证书接受 |
|---|---:|---:|---:|---:|
| `high128`, 4 independent standard | 1.49553e-5 | 3.14922e-9 | .049667 | 128/128 |
| `high128_tri`, 4 independent standard | 1.38476e-5 | 2.33075e-9 | .049726 | 128/128 |
| `high128_tiles`, 4 independent standard | 1.36077e-5 | 2.10910e-9 | .049746 | 128/128 |
| `high128`, first spots + 3 independent standard | **.00455307** | .00091179 | .045869 | 128/128 |
| 与上一行**同图像集合**，顺序轮换使CNN读standard | **.00003325** | 3.08400e-9 | .049656 | 128/128 |
| `high128`, 4 duplicate spots | .00829942 | 8.09849e-7 | .049864 | 128/128 |

原始日志依次为 [standard](phase7_compiled_trained_C4_standard_new128.json)、[tri](phase7_compiled_trained_C4_high128_tri_new128.json)、[tiles](phase7_compiled_trained_C4_high128_tiles_new128.json)、[first spots](phase7_compiled_trained_C4_firstspots_new128.json)、[同图像通道轮换](phase7_compiled_trained_C4_firstspots_rotated_new128.json)、[重复 spots](phase7_compiled_trained_C4_duplicate_spots_new128.json)。[实验入口](../../tools/phase7_feedback_scale_probe.py)增加显式`--allow-multichannel-appearance-shift`与`--rotate-view-channels`；默认旧实验逻辑和参数不变。`--duplicate-image-channels`是原有信息量对照。

## 解释与局限

相同四幅图像仅改变顺序，就把地图RMSE从.00455降低到.0000333，约137倍。这是强的**架构级诊断**：首通道粗编码器的分布外失配，在该设置下压过了其他三个有信息量通道的局部提示；不能把此失败归咎于固定P1安全器。轮换并非对任意数据的最佳策略，只是本次明确可控的干预。四幅复制 spots 时图像MSE低而地图RMSE高，重申光度一致不等于几何真值可识别；复制通道不增加独立方向信息。

三种新形变族/种子的standard纹理误差仍在\(1.36\text{--}1.50\times10^{-5}\)，但它们共用标准纹理机制；这不是自然图像泛化。上表的拓扑全部通过说明**给定网络输出时解码器可靠**，并不说明图像预测总是正确。下一步应让粗编码器也做多视图对称或置信度感知融合，再在保持相同训练形变数、相同测试图像集合的条件下复核；不能只重新排序测试通道后声称问题已解决。
