# 4097² 固定 P1 网格：编译颜色算子的极端有限 latent 压力测试

## 问题与对象

问题是：将 F1 单颜色更新交给 PyTorch `torch.compile`/Inductor 后，编译器的浮点重排是否在大幅度 latent 下破坏**实际返回的 float32 顶点表**的逐面正向？这是有限样本数值测试，不是对所有浮点输入的定理。F1 的精确算术保证、float32 后验原面证书与必要时返回恒等图的区别见 [主报告](REPORT.md) 和 [数值保证](22_numeric_certified_joint_layer.md)。

使用边长为 4097 的同一规则三角网格：16,785,409 个控制顶点、33,554,432 个原三角面。边界固定为单位正方形恒等映射。每次从独立标准正态张量采样内部顶点二维 latent，再乘幅度 \(a\in\{5,20\}\)。固定种子 1949，每个幅度 32 张地图；模型 `raw_span=2`、`motion_mode=radial`、`index_mode=generated`，对有向双面积设绝对目标底线 \(0.05h^2\)，其中 \(h=1/4096\)。输入是**随机 latent**，并非给定目标地图的 teacher latent；这里只测单个最细网格 F1 颜色周期，没有图像编码器、训练或多层组合。

对同一 latent，分别运行 eager 与仅编译 `_update_color` 的 F1。先把返回的 float32 顶点转成 float64，再从这些**已舍入的实际顶点**重新算所有原面有向双面积，报告最小归一化面积 \(J_{\min}=\min_T A_T/h^2\)。另外执行完整 `certify_p1_or_identity` 原面与边界检查，记录是否接受而非 fallback。计时是 GPU7 热态单样本 F1 forward（首个编译样本不计中位数），不包含图像编码、训练反传或其他层；硬件是 AI 主机的 48GB NVIDIA GPU，float32。脚本为 [phase7_stress_compiled_color_p1.py](../../tools/phase7_stress_compiled_color_p1.py)，原始输出为 [幅度 5](phase7_compiled_color4097_random_stress32.json) 和 [幅度 20](phase7_compiled_color4097_random_stress32_amp20.json)。

| latent 幅度 | eager / compiled 证书接受 | 两版本每组最小 \(J_{\min}\) | 顶点坐标最大绝对差 | eager / compiled 热态 forward 中位 |
|---:|---:|---:|---:|---:|
| 5 | 32/32 / 32/32 | 0.0494237 / 0.0494237 | \(5.3644\times10^{-6}\) | 55.48 / 10.60 ms |
| 20 | 32/32 / 32/32 | 0.0494385 / 0.0494385 | \(6.2585\times10^{-6}\) | 55.49 / 10.57 ms |

幅度 20 的所有 \(32\times4097^2\) 个顶点中，逐点二维最大坐标差超过 \(0.01h\) 的有 10 个；逐坐标均方根差为 \(9.13\times10^{-10}\)。编译版与 eager 版**不是逐位相同**；两个最低面积在表中相同也不表示每个三角面的面积相同。最低实际面积略低于名义 \(0.05h^2\) 是 float32 顶点舍入的结果，仍远大于 0，并且通过最终证书。编译首例约 2.8–7.1 秒，受已有编译缓存影响；热态 5.2 倍只适用于这个局部 F1 forward 基准，完整可训练网络的速度应看 [82 号实测](82_compiled_color_kernel_full_training_pareto.md)。

结论限于这 64 个已知有限随机样本：编译重排没有诱发可观察到的原面翻转或证书 fallback。实际层仍必须对每张返回地图运行浮点原面证书；这组压力样本不能取代一般拓扑证明，也不能将名义面积底线视为舍入后严格满足的下界。

## 相同 cotangent 对全部 fine latent 的 VJP 差异

只看顶点输出可能掩盖活动面切换附近的反传差异。因此用同一脚本新增`--vjp-count`：在4097²网格上分别采样16个新的独立随机latent（幅度1或5），每例生成相同标准正态cotangent\(w\)，计算eager与compiled对**所有内部二维latent**的\(g=\partial\langle w,Y(z)\rangle/\partial z\)。不在两版之间改变地图、cotangent或参数。对每例记录\(\|g_{\rm c}-g_{\rm e}\|_2/\|g_{\rm e}\|_2\)；表中位数在16例上取第8/9个排序值的平均。另报告所有坐标/案例的最大绝对梯度差。每张VJP对应的前向输出也用正式原面证书检查。全为float32，GPU7；这不是双精度真值比较，不能从两版差异判断哪一个梯度更准。

| latent幅度 | 两版证书接受 | 所有梯度有限 | 相对L2差中位 / 最大 | 最大逐坐标绝对梯度差 |
|---:|---:|---:|---:|---:|
| 1 | 各16/16 | 是 | .001706 / .003616 | .003120 |
| 5 | 各16/16 | 是 | .002048 / .007118 | .003605 |

[幅度1原始VJP](phase7_compiled_color4097_nominal_vjp16_certified.json)、[幅度5原始VJP](phase7_compiled_color4097_extreme_vjp16_certified.json)含每例相对误差和两版梯度RMS；[257²先行烟测](phase7_compiled_color257_vjp_sanity.json)也保留。两版梯度在这32例全部有限、整体相近，但**并不逐位甚至逐坐标都接近**；幅度5的单例相对L2差最高约0.71%。局部安全比例用min/夹断决定活动面，编译浮点重排可能改变接近切换点的路径，这是一个合理机制解释，当前实验没有逐面活动集追踪，不能声称已证明根因。300步image-only训练的几何误差和权重轨迹相近，仍不能替代应用损失下的更广泛梯度稳定性检查。

在加入每例原面证书之前的[4例探索](phase7_compiled_color4097_extreme_vjp4.json)、[16例幅度5探索](phase7_compiled_color4097_extreme_vjp16.json)与[16例幅度1探索](phase7_compiled_color4097_nominal_vjp16.json)也保留在仓库，但表格只采用**已逐例证书确认**的后续运行。
