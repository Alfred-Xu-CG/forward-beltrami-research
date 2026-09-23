# 顶点位移势的正边化：避免细尺度 latent 的积分型梯度衰减

## 从一次明确的优化失败出发

[正边密度层](13_monotone_fiber_forward_layer.md)把每行 latent 先变成正边长，最后前缀求和。它的逆表示和百万点前向/VJP 都很快，但**可表示不等于易训练**。在独立给定的 1025² high128 水平纤维目标 \(F(x,y)=(x+10^{-4}\sin(256\pi x)\sin(256\pi y)\sin^2(\pi x),y)\) 上，完整 1,047,552 维边密度 latent 从零直接作 1200 步 Adam（余弦学习率从 0.0003 到 \(3\times10^{-8}\)，map 欧氏平方损失乘 \(10^6\)）后，全顶点向量 RMSE 仍为 \(2.07\times10^{-5}\)；零 latent 的 RMSE 是 \(3.06\times10^{-5}\)，而解析反求 teacher latent 的 float32 RMSE \(2.0\times10^{-9}\)。普通优化只追回约三成误差。这是具体协议的阴性结果，不是所有优化器或边密度层的不可训练性定理。[拟合脚本](../../tools/phase7_fit_fiber_latent.py)。

原因可以从恒等图附近的线性化看出。设一条水平行的 \(n\) 条正边概率为 \(p_j=\operatorname{softmax}(Lz)_j\)，\(X_r=\rho r/n+(1-\rho)\sum_{j<r}p_j\)。在 \(z=0\) 时，\(p_j=1/n\)，对微扰有
\[
\delta X_r=\frac{(1-\rho)L}{n}
\left(\sum_{j<r}\delta z_j-\frac rn\sum_{j=0}^{n-1}\delta z_j\right).
\]
除掉每行常数 logit 的规范自由度后，这仍是一个**离散积分算子**：细振荡 \(\delta z\) 的顶点位移幅度被频率抑制，map-loss 的反传自然弱化高频。这解释了一种训练条件数障碍，但不能单凭线性化断言所见训练曲线全由此造成；Adam 的学习率、规范方向及损失尺度也参与。

## 不把坏的顶点势直接相加：先正边化、再归一与积分

另取第 \(i\) 条内部行的细网格**顶点势** latent \(z_{ij}\)，\(j=1,\ldots,n-1\)，边界设 \(u_{i0}=u_{in}=0\)、内部 \(u_{ij}=A\tanh z_{ij}\)。\(A>0\) 是物理位移幅度范围。原始边密度 \(r_{ij}=1+n(u_{i,j+1}-u_{ij})/(1-\rho)\) 可能为负，所以**不能**简单输出 \(x+u\)。取 \(\beta>0\) 和 \(s_\beta(t)=\log(1+e^{\beta t})/\beta>0\)，定义
\[
a_{ij}=s_\beta(r_{ij}),\quad q_{ij}=a_{ij}/\sum_ka_{ik},\quad
d_{ij}=\rho/n+(1-\rho)q_{ij},\quad
X_{i0}=0,\quad X_{ij}=\sum_{k<j}d_{ik}.
\]
其余纵坐标为 \(i/n\)，边界恒等。每个合法有限 latent 即使提出严重折叠的 raw \(x+u\)，也有 \(d_{ij}>\rho/n\)，故全部原网格 P1 面 \(J_T>\rho\)、全局同胚。没有从已有 map 沿目标损失梯度做 line search；这是从任意顶点势 latent 一次性正向解码的光滑层。[实现](../../src/qcopt/neural_bijection/dense/monotone_fiber_p1.py)。

若希望在**已有**正边密度基图 \(B_{ij}\) 上加细节，不做地图复合或重采样。写同一行已有细边 \(b_{ij}=B_{i,j+1}^x-B_{ij}^x>0\)，\(\sum_jb_{ij}=1\)。令
\[
r_{ij}=1+\frac{u_{i,j+1}-u_{ij}}{(1-\rho)b_{ij}},\qquad
a_{ij}=b_{ij}\frac{s_\beta(r_{ij})}{s_\beta(1)},\qquad
d_{ij}=\rho b_{ij}+(1-\rho)\frac{a_{ij}}{\sum_ka_{ik}}.
\]
由 \(d_{ij}>\rho b_{ij}>0\)、\(\sum_jd_{ij}=1\)，直接在**原来相同的细顶点表**上重建每行 \(X_{ij}\)，得到另一张固定网格 P1 同胚。\(u=0\) 时数学上 \(a=b,d=b\)，基图严格保持。基图由多尺度边密度 latent 生成时，这一 `HybridMonotoneFiberP1Layer` 对**全部有限粗细 latent**保拓扑；无需大矩阵求逆。竖直纤维交换 x/y 即可。它仍不能单独表示任意二维同胚，若要修正双分量运动需与 F1/F2 的一般顶点安全原语结合。

## 表达误差与梯度条件数

设目标细势 \(u\) 满足所有 \(r_{ij}\ge m_r>0\)。因 \(0\le s_\beta(t)-t\le e^{-\beta m_r}/\beta\) 对 \(t\ge m_r\) 成立、且每行 \(\sum_jb_{ij}r_{ij}=1\)，归一后概率向量相对理想 \(b_{ij}r_{ij}\) 的 \(\ell^1\) 差不超过 \(2e^{-\beta m_r}/\beta\)。故新的顶点坐标与合法的理想候选 \(B^x+u\) 在该行的最大误差至多
\[
\|X-(B^x+u)\|_{\ell^\infty}
\le\frac{2(1-\rho)}{\beta}e^{-\beta m_r}.
\]
这是**有正余量的目标**的均匀近似界，不意味着任意大的 raw 位移仍被精确追随；对不合法 raw 目标，层继续安全，但必然产生非零修正误差。界中的 \(m_r\) 若接近 0 会变差；float32 误差另计。

在 \(u=0\) 线性化，\(r=1\)，\(\sum_j\Delta u_j=0\)，所以
\[
\delta X_{ij}=c_\beta\,\delta u_{ij},\qquad
c_\beta=\frac{s_\beta'(1)}{s_\beta(1)}
=\frac{\operatorname{sigmoid}(\beta)}{s_\beta(1)}\approx1.
\]
因此在 \(z=0\) 有 \(\delta X\approx A\delta z\)：其主导 Jacobian 是**顶点坐标局部恒等**，不是先前的频率衰减积分算子。这说明“势 latent”尤其适合在已有良好粗图附近学习弱细节；对强剪切从零训练，raw 边也可能大量进入 softplus 饱和区，梯度会受限，不能把局部线性化推广为任意状态的全局好训练保证。

## 已完成的实现核查与实验边界

9 个本地单元测试覆盖水平/竖直极端有限 latent 的全脸正向、边界固定、强剪切解析 oracle、双精度 finite-difference VJP，以及强剪切基图加独立正弦细节的 hybrid 输出；后者在 65² 上最大坐标误差低于 \(10^{-8}\)。Turing A40、PyTorch 2.8、float32、batch2、1025² 控制顶点的**纯顶点势层**中位 forward 0.000805 秒、forward+VJP 0.00217 秒，峰值 allocated 0.126/0.159 GB；随机极端细边导致最小 \(J_T=0.04987>0\)，低于精确算术的 \(\rho=0.05\) 是浮点舍入。强剪切 teacher latent 的最大坐标误差 \(4.77\times10^{-7}\)、顶点 RMSE \(7.19\times10^{-8}\)、最小 \(J_T=0.12036\)。这些数不含图像编码器、warp、loss。[测量脚本](../../tools/phase7_profile_monotone_fiber.py)。

百万点独立目标的直接 latent 拟合显示**频率/幅度与参数化适配**的差异：同一个 high128 目标、1200 步 Adam、余弦学习率初始 0.0003、顶点监督，完整 1025² 势 latent 从零收敛到 float32 顶点误差 0，而密度 latent 保留约 \(2.07\times10^{-5}\)；513² 势 latent 达 \(5.18\times10^{-6}\)，257² 势 latent 仅 \(2.78\times10^{-5}\)。同样势参数化对强剪切若使用未调适的 0.03 学习率，在 400 步后的 1025² 误差仍 0.00684，明显劣于密度层的约 0.0002–0.0004；这是不能选择性隐去的**反向失败**。这些是两种不同任务的超参数试验，并非相同预算下的完整 Pareto 排名，更不是图像反向推断成功。

同一强剪切基图 \(G_{0.28}\) 加独立 high128 细模式的进一步对照已经完成。基图是从解析目标构造的 teacher density latent，但**只保留 257 侧长的 65,280 个内部行边 latent**，升到 1025² 后的目标图误差约 \(3.1085\times10^{-5}\)；因此细层既要补局部高频，也要补粗基图差。目标为解析 \(G_{0.28}\) 与 high128 位移相加后直接采样在 1025² 顶点，全部面最小 \(J_T\) 约 0.08337。密度细残差与顶点势细残差分别有 1,047,552/1,046,529 参数，从零拟合相同 target map 1200 步 Adam、余弦学习率、map MSE×\(10^6\)、float32。前者在 Turing A40、初始学习率 0.003 下，RMSE 从 \(3.1085\times10^{-5}\) 到 \(2.3431\times10^{-5}\)；teacher 可表达误差 \(1.76\times10^{-8}\)。后者在 Element L40、初始学习率 0.0003 下，RMSE 降到 \(1.02\times10^{-8}\)，teacher 可表达误差 \(5.46\times10^{-9}\)。两者输出 min \(J_T\) 分别 0.12012/0.08337、梯度有限；每步中位约 0.00148/0.00167 秒、峰值 allocated 0.126/0.147 GB。这个结果强烈支持**在已有粗图附近**用顶点势修正高频的参数化选择；但两组跑在不同 GPU、PyTorch 版本，学习率分别调适，因此不能将速度差或优化差直接解释为严格同设备 Pareto 支配。基图由解析 oracle 提供，仍不是图像推断结果。[完整实验脚本](../../tools/phase7_fit_hybrid_fiber_detail.py)。

下一步是和一般双分量 F1/F2 的密集细 latent 做相同目标与训练预算比较，并把该混合结构连到图像编码器；图像纹理是否足以辨识高频位移是另外的问题。
