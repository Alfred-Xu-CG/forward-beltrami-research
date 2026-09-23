# Route A 百万控制顶点：激活重计算的速度—显存 Pareto

## 不改几何，只改反传存储

可复用的 `NestedP1PhotometricFeedbackLayer` 在每次细反馈中先从固定/移动图、当前1025²映射计算局部光度位移提议，再截取16个二维正弦模式，然后按四色解析QC/面积安全界更新顶点。默认实现只对安全更新部分作非重入 activation checkpoint：反向传播时重算安全更新而不保存其中全部中间张量，图像提示与谱步骤的激活仍由autograd保留。本文另外实现两种**数学映射完全不变**的策略：

- `full`：将光度提示、16模态谱化和安全更新作为一个无随机态的函数整体checkpoint，反向时均重算；
- `none`：完全不作checkpoint，保存更多中间激活以减少VJP时重算。

三种模式都先对输入257² P1图检查固定边界、全原面正面积和QC cap，并在输出1025²对全部2,097,152面和边界再次检查；策略选择不改变同胚条件或输出值。9→33的小网格测试在相同输入下比较了三档的全部输出，以及对粗map、固定图和移动图的梯度，均在预定float32容差内一致。1025²的同一冻结编码器、同一8例high64图上，三档 image/map/面$\mu$与最小面积完全一致；最后一次 image-only VJP 对编码器参数的梯度范数均为$4.693806\times10^{-5}$。范数相同不替代完整向量逐元素检查，小网格测试承担后者。

## 实际测量

AI主机同一空闲物理GPU2，batch1、512²最终图像/query、257²已训练粗网络＋两次1025²细反馈、float32、内部QC cap0.8。每档从新进程启动，8例保留评价后，对同一训练draw做10次完整 image-to-loss forward/VJP，无参数更新；GPU同步，丢首轮、余9次取中位。峰值含32训练＋8保留图常驻、整个细网格层和反传；`allocated` 是PyTorch活动张量峰值，`reserved` 是缓存分配器峰值，均非`nvidia-smi`总显卡进程占用。冷启动与8例评价不算进热启动毫秒。

| 激活策略 | 完整forward / VJP | allocated峰值 | reserved峰值 | 输出/梯度 |
|---|---:|---:|---:|---|
| 默认 `refiner` | 79.08 / 205.02 ms | 2.150 GB | 2.418 GB | 相同 |
| `full` 整次反馈重算 | 79.49 / 209.02 ms | **1.960 GB** | **2.196 GB** | 相同 |
| `none` 不重算 | **78.02 / 160.64 ms** | 3.050 GB | 3.402 GB | 相同 |

相对默认，`full`节约活动显存0.190 GB（约8.8%）、reserved 0.222 GB（约9.2%），VJP增约4ms（约2%）；`none`多占活动显存0.899 GB（约42%），却使VJP短约44ms（约21.6%）。这不是独立求解算法的比较；同一层可按部署显存预算选择模式。当前保持 `refiner` 为默认，因其已有1025²训练证据且处于两者之间；用户若显存接近2GB可选 `full`，若有足够显存并重视吞吐可选 `none`。本节**这组**测试只有batch1；随后追加的[batch 1/2/4/8 实测](33_route_a_million_control_batch_scaling.md)另列显存与吞吐。仍未测混合精度、torch.compile或多卡，不能外推这些配置的吞吐拐点。显存还有图像提示、静态查询、粗网络等固定成本，仅靠这一级checkpoint不可能消掉全部2GB。

复现：`tools/phase6_exact_coarse_fine_feedback.py --use-module --checkpoint-mode {refiner,full,none} --benchmark-repeats 10 --fine-cycles 64`；层参数为 `checkpoint_refiner` 与 `checkpoint_full_pass`。主原始JSON为 `raw_results/a8_high64_1025_checkpoint_{refiner,full,none}_grad_gpu2.json`；前两轮短测与未含reserved/梯度范数的JSON仍留作噪声对照，不作为表中主值。相关测试 `tests/test_phase6_nested_p1_feedback.py`。
