# 融合安全细网格更新：有效的轻扰动加速与不可忽略的反向故障

## 问题与实验对象

当前 1025² 联合层的细阶段 [SafeColoredVertexRelaxation](../../src/qcopt/neural_bijection/dense/colored_vertex_relaxation.py) 用四个颜色次序更新顶点，每色在六个相邻面上算方向面积余量、局部最小步幅和安全复制。它的数学输出合同不因算子融合而改变；但 PyTorch eager 产生多个 gather/pointwise/autograd 中间张量。测试仅在**不改几何公式**的情况下将该模块交给 PyTorch2.5.1 的编译器，问是否有可用的 forward/VJP 性能收益。设备为 AI A6000 GPU2、float64 几何；所有数字均将启动编译与热启动推理分开。复现脚本：[单独细阶段](../../tools/phase7_compile_fine_update.py)、[完整联合层](../../tools/phase7_profile_joint_compiled.py)、[故障分段定位](../../tools/phase7_diagnose_joint_vjp_nan.py)。

单独细阶段，以规则单位图作 base、随机正态 logit 标准差 .2、batch1、1025² 控制顶点、2,097,152 面、八次热测：eager forward+一次对全部细 latent 的 VJP 中位31.88 ms，融合后7.38 ms；CUDA allocated 峰值分别983.6/625.2 MB。输出最大差 \(4.44\times10^{-16}\)，VJP 最大差 \(5.64\times10^{-13}\)，两者最小面 \(J\) 约.05。257² 时对应11.57→2.91 ms、63.6→39.8 MB，输出/梯度差分别 \(7.77\times10^{-16}/2.86\times10^{-14}\)。首次编译与预热约15–17秒，重复网络结构/shape 可摊销，动态 shape 或频繁重编译则未测。

## 完整层：轻扰动有效，但一般 latent 有 NaN VJP

完整层包含257²双周期粗 patch、两次精确 P1 细分、1025²细顶点更新、float64 实际输出筛选；输入 coarse/fine latent 均 float32。两套模型只有细阶段是否融合不同；seed91737、随机标准差 \(s\)、同一输出 cotangent，十次热测。按最后输出所有 2,097,152 面算最小 \(J\)。峰值是 CUDA allocated，下面“瞬态额外”从每次热测前常驻占用扣除。

| \(s\) | 版本 | 完整 forward+VJP 中位 | 峰值 allocated | 瞬态额外 | 最小 \(J\) | VJP 有限 |
|---:|---|---:|---:|---:|---:|---|
| .01 | eager | 83.54ms | 1.374GB | 1.038GB | .33046 | 是 |
| .01 | 仅细阶段融合 | 55.55ms | .927GB | .565GB | .33046 | 是 |
| .10 | eager | 69.89ms | 1.374GB | 1.038GB | .05 | 是 |
| .10 | 仅细阶段融合 | 59.41ms | .927GB | .565GB | .05 | **否** |

在 .01 样本，输出、粗/细 VJP 逐元素相等（按该 float32 latent 口径）；因此小扰动条件下融合有效。但 .10 样本，输出仍有限且与 eager 最大只差 \(8.05\times10^{-16}\)，两者都被浮点 P1 筛选接受，而融合版 coarse/fine VJP 含 NaN。**不能拿 59.41ms 宣称可用的 neural layer 加速。**这不是某一张图折叠，也不是输出筛选回退触发。

## 分段定位与目前的根因范围

最小稳定复现保留相同 random seed，并把粗/细 logit 标准差各设为 .05：eager 的粗图、精确细分、细输出、筛选输出各阶段 VJP 全有限；PyTorch 的 aot_eager 图整理后端在同一状态也全有限；只有 Inductor GPU 编译的**细阶段反向**失效：约130,004/130,050个 coarse latent 梯度及4,874/2,093,058个 fine latent 梯度为 NaN。NaN 首次出现在细阶段而非粗 patch 或细分，筛选前后相同。粗/细之一保持零时，Inductor VJP 有限；两者均 .01 时也有限。把细层绝对面积 floor 从 \(.05h^2\) 降到0，在 .05/.05 复现中仍产生约130,002个粗梯度和3,586个细梯度 NaN，因此不能把问题简单归咎于那个 floor 的 clamp。实际导致 Inductor 生成的哪个反向内核操作失效**尚未确定**；目前证据只把故障定位到该后端在变形 base 加非零细更新的活动集上，而不能指认为某一已确认 PyTorch 源码 bug。

研究决定：生产层继续使用已通过梯度测试与压力测试的 eager VJP，**暂不启用这个融合版本**。后续可将颜色 pass 或面积余量计算分段编译，定位失败的最小子图；只有通过 .05/.05 及更强混合 latent 压力集、有限差分/双精度参照和 image training 后，才能将它作为默认加速。另一方向是手写有限步自定义 VJP 或只对安全中间式做融合，但不能以丢失全部 finite latent 的可导性换速度。
