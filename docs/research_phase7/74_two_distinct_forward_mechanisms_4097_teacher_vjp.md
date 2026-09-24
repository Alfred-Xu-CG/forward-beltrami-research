# 两种独立正向P1机制完成4097²真实控制网格的目标复现和VJP

## 任务、原语和teacher定义

在 \(\Omega=[0,1]^2\) 的固定SW–NE三角剖分上，给定与待测decoder**独立定义**的解析平面映射
\[
F_a(x,y)=\bigl(x+a\sin(2\pi x)\sin(\pi y),\quad
y-0.8a\sin(\pi x)\sin(2\pi y)\bigr),\qquad a=.08.
\]
所有边界点恒等。取每级网格对 \(F_a\) 的顶点采样，然后按相邻时间样本或父级P1精确二分延拓的差，反算有限 \(\operatorname{atanh}\) teacher latent；这不是从待测decoder输出反向生成目标。seed17²上分4轮由恒等图走到 \(I_{17}F_a\)，之后依次33²、65²、129²、257²、513²、1025²、2049²、4097²，每个级别恰好一个局部更新周期。最后真实输出 \(4097^2=16,785,409\) 个控制顶点，\(2(4096)^2=33,554,432\) 个原三角面，**不是**4097²个图像查询。

比较两套不同局部原语：F1每周期为四着色的独立单顶点径向更新；F2每周期为四种交错offset的4-cell局部patch联合更新。两者都先精确P1二分，再直接修改同一张最终网格的顶点表；不求全局线性系统，不把连续复合重采样冒充固定P1。每次有限latent的精确算术面积约束见[F1构造/统一类证明](23_uniform_isotopy_multilevel_approximation.md)和[F2二次面积预算](26_f2_uniform_isotopy_approximation.md)；本次还启用对**实际float32输出坐标**的逐原面保守符号筛选，未通过时按样本回退为单位图。所列目标全部通过筛选，没有回退。

实验代码为[同伦teacher探针](../../tools/phase7_verify_isotopy_pyramid.py)。本次修正其`--certified --dtype float32`路径，使证书层真正使用指定的float32而非构造函数默认的float64。AI远程同一空闲RTX A6000 GPU7、PyTorch2.5.1+cu124、batch1、float32，随机cotangent固定seed713。计时只包decoder forward和对**全部12个latent张量**的一次 \(\langle Y,Q\rangle/V\) VJP，CUDA同步三次取中位，不含构造teacher latent、图像CNN、数据转移或初始化；峰值是CUDA allocated，不是reserved。顶点RMSE为 \(\sqrt{V^{-1}\sum_i\|Y_i-I_hF_a(x_i)\|_2^2}\)，最小 \(J\) 是所有33,554,432个原三角形的有向面积除以恒等面面积。

| 4097²机制 | 全12个latent的forward+VJP中位 | CUDA allocated峰值 | 顶点RMSE / 最大坐标绝对误差 | 目标/输出最小 \(J\) | 12组VJP |
|---|---:|---:|---:|---:|---|
| F1四颜色单点 | **.5590s** | **12.561GB** | 3.436e-10 / 1.490e-8 | .566003 / .566003 | 全部有限且非零 |
| F2四交错patch | 1.1348s | 19.941GB | 3.436e-10 / 1.490e-8 | .566003 / .566003 | 全部有限且非零 |

两者实际复现精度相同，但当前实现的F2在此平滑目标约耗2.03倍时间、1.59倍峰值显存；这是同一GPU/任务上的实现比较，不是F2原语在其他形变/patch尺寸上的不可能性结论。F2在[1025²较强目标](26_f2_uniform_isotopy_approximation.md)中曾用更少粗轮达到F1/4粗轮受截断的目标，故应保留作为有条件的表示/稳定性替代。这里的 \(3.4\times10^{-10}\) 是**相对于同float32解析顶点采样**的误差，不是连续映射的全域逼近误差或图像配准精度。

最终独立重跑将相同CLI、3次同步forward＋全latent VJP的原始JSON存为[F1复核](phase7_teacher4097_F1_independent_recheck.json)和[F2复核](phase7_teacher4097_F2_independent_recheck.json)：本次中位分别.5602秒和1.1391秒，顶点RMSE均\(3.43608\times10^{-10}\)，目标/输出最小\(J\)均.566002607，allocated峰值分别12.543GB和19.941GB。与前表的细小时间/显存差属于重复运行波动；两个原始JSON补全可复算证据，不把teacher潜变量误说成图像网络推断。

此结果补足“只在小网格做teacher拟合”的缺口：两种确实不同的前向机制在4097²真实控制尺度做过完整VJP，而非仅对512²图像查询反传。不过teacher latent由已知解析 \(F_a\) 构造，不能据此声称**网络**在4097²自动恢复任意同胚。当前4097² image-to-latent训练/保留试验属于混合F2 seed＋F1细级模型，见[实际细级训练](65_trained_4097_extra_feedback_gains.md)、[四视图观测](68_multiview_observability_4097_p1.md)与[含噪边界](70_multiview_channel_count_noise_tradeoff.md)；纯F2的4097²图像训练仍未做。
