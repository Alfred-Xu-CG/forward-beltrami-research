# Route C：物理响应引导的 18 模态可微正 conductance 层

## 1. 为什么从边权 CNN 转向解析光度系数

[六频边字典实验](13_spectral_edge_dictionary.md) 表明，直接用目标几何作 oracle 可用18个 conductance 模态恢复约92%的32周期位移，但普通和六频 CNN 在图像训练后分别只恢复接近零与约1–3%。因此这里不扩大求解器，而改变 *图像到 latent* 的方式：预计算每个边权模态对平衡映射的灵敏度，再从图像差异求一个18维光度正规方程。该小系统不是大稀疏矩阵的逆；最终仍需在全部257²控制顶点上解一次正 conductance 系统。

本层使用单位正方形的 SW–NE 原网格三角剖分，$257^2=66049$ 顶点、131072面、197120条水平/竖直/东北对角边；固定恒等凸边界。18个固定边中点函数 $\phi_{ke}$ 来自六个频率 $1,2,4,8,16,32$ 在三类边上的方向模板，精确定义见13节。给定系数 $a\in\mathbb R^{18}$，逐边

\[
z_e(a)=-0.8+\sum_{k=1}^{18}a_k\phi_{ke},\qquad
c_e(a)=1+15\sigma(z_e(a))\in(1,16).
\]

内部顶点 $Y_I(a)$ 解 $A(c(a))Y_I=b(c(a))$，其中 $A$ 是整个细网格的对称正权 Dirichlet 图 Laplacian。精确实数解在固定凸边界下是 Tutte 型原网格 P1 homeomorphism，不是仅在512²图像 query 上正定向。实现用 float64 正弦预条件 CG 与隐式伴随，到真实相对残差约 $10^{-10}$，每次最终输出另检查全部131072面的正有向面积；边界—边界无效边虽然包含在张量中，对映射的梯度为零。

## 2. 解析预计算与运行时系数估计

在零系数处，边权统一为 $c_0=1+15\sigma(-0.8)$，解是恒等映射 $Y(0)=I$。对第 $k$ 个系数求导，$c'_e=15\sigma(-0.8)(1-\sigma(-0.8))\phi_{ke}$，从平衡方程得到

\[
A(c_0)V_{k,I}=b'_k-A'_k I=-(L(c'_k)I)_I,
\qquad V_k:=\left.\frac{\partial Y}{\partial a_k}\right|_{a=0}.
\]

右端可在图上直接组装；同一对称系统可批量解18个二维右端。一次性预计算的最大真实相对残差 $7.69\times10^{-13}$、25次 PCG 迭代。对第0、5、17模态，分别以 $a_k=\pm10^{-3}$ 独立做中心差分，预计算 $V_k$ 与差分的顶点逐分量最大差为 $1.34\times10^{-9}$、$4.56\times10^{-11}$、$3.12\times10^{-11}$。这个验证只说明响应求导实现准确；对大位移仍是零点一阶近似。

运行时仅在 **128² 光度拟合网格**采样 $V_k$ 和输入的固定/移动图像；最终 map 依旧在257²控制网格生成，最终图像仍在512²查询，绝不能把128²称为控制规模。对拟合点 $p$，令 $r(p)=I_f(p)-I_m(p)$、$G_k(p)=\nabla I_m(p)\cdot V_k(p)$。图像梯度用中心差分和真实单位坐标间距。求

\[
\widehat a=\underset{a\in\mathbb R^{18}}{\arg\min}\;
  Q^{-1}\|Ga-r\|_2^2+\lambda\|a\|_2^2,
\quad
\lambda=0.005\,\frac{\operatorname{tr}(G^\mathsf TG/Q)}{18},
\quad Q=128^2.
\]

这就是18×18 SPD 正规方程。为限制线性化失效时的极端值，用 $\widetilde a_k=4\tanh(\widehat a_k/4)$；最终求解以 $\widetilde a$ 对应的**真实非线性** $c_e$ 进行，而不是输出线性位移 $I+\sum a_k V_k$。即便估计不准，每条 $c_e$ 仍正，拓扑机制不变。18个模态包含已知目标频率32，是强任务先验；不能推广为任意 Beltrami 场的表示。

## 3. 从不训练到多步可训练层

不训练时，$\widetilde a$ 已构成可微 image-to-latent 前向。batch2 的完整 forward（128²图像拟合、257²PCG、262144 query、一次512²图像warp、损失）中位36.5 ms；回传到输入图像和可微标量 gain 的完整 VJP 中位31.4 ms，PyTorch peak allocated 195 MB，伴随真实相对残差 $5.12\times10^{-11}$；输入图像与 gain 梯度均有限。这个测时的 VJP 不等于18参数训练的测时，GPU共享负载及图结构不同。

进一步令 $a_k^{\rm final}=g_k\widetilde a_k$、$g_k=\exp(3\tanh\theta_k)$，只训练18个 $\theta_k$，初始全零即 $g_k=1$。正导纳及固定边界保证与 $\theta$ 数值无关；上式使增益有限，但频率32可被训练放大。训练数据是种子55101的32例 high32 合成纹理/形变，保留种子99317的8例；图像512²、fit128²、batch2、Adam rate0.01，只用最终 warped-moving 对 fixed 的像素 MSE，不在损失中使用目标 map 或真μ。目标几何只供评价。

| 配置 | 保留 image MSE | query-map RMSE | 原面 μ RMSE | 32周期振幅率 | 最大 $|\mu|$ | 最小面积比 |
|---|---:|---:|---:|---:|---:|---:|
| 固定18响应，未训练增益 | 0.0002781 | 0.001389 | 0.08177 | 0.485 | 0.208 | 0.694 |
| 18增益，300步图像训练 | **0.0001002** | **0.0008845** | 0.05771 | 0.858 | 0.316 | 0.559 |
| 18增益，1000步图像训练 | 0.0001083 | 0.0009029 | **0.05710** | **0.865** | 0.317 | 0.559 |

1000步不是更优的 image/map 选择。300步训练墙钟27.7 s，batch2 完整 forward/VJP 中位36.62/49.51 ms，peak allocated 340 MB；1000步76.4 s，35.92/35.84 ms、338 MB。VJP 计时抖动较大，不能凭相差14 ms认定后者数学上更快。训练期前向/伴随真实残差及逐例面证书在 JSON，全部被求解器检查。每个增益的一阶梯度穿过正 conductance、PCG 隐式伴随、图像采样，没有保存 Krylov 迭代轨迹。

在先声明的新种子1170031共128例上，300步层的 image MSE $1.360\times10^{-4}$、map RMSE 0.001042、面 μ RMSE0.05526、32周期振幅率0.880、最大 μ0.362、最小面积比0.508；1000步相应为 $1.392\times10^{-4}$、0.001048、0.05466、0.887、0.363、0.507。两者基本一致，支持300步早停。相同种子的 A8 两次反馈 image MSE $4.328\times10^{-5}$、map RMSE0.000590、面 μ RMSE0.13576、最大μ0.794、最小面积比0.117。即 C 本层在 μ 和面积余量上更好，但图像与位置精度较差；它不是单指标的全面胜利。A8 初始cap在该128例全满足，其0.8上界仍是条件性的。

## 4. 失败域与判断

把300步层直接用于六张从未用于训练的摄影/扫描灰度内容加同族合成形变96例时，image MSE **0.03254**、map RMSE **0.004913**、面 μ RMSE0.1120、细频振幅率仅0.277，最小面积比0.595。未训练的同一响应层在此域上 image MSE0.03417、map RMSE0.005240、细频振幅率0.153。训练仅有小幅收益，而 A8 在同批96例的 image MSE0.004166、map RMSE0.001072。光度正规方程依赖图像梯度与一阶化，照片纹理的响应分布明显不同；不能把 synthetic heldout/fresh 成绩外推为真实照片鲁棒性。低面 μ RMSE也不能弥补差的 image/map fit。

该层确实在257²控制侧完成 forward、隐式 VJP 和多步训练，且具有严格正导纳构造、明确计算/显存收益与可独立检查的拓扑。但它是**预先给出18个已知频率模态的任务特化层**，不是通用由任意 $\mu$ 求解 Beltrami 方程，也没有证明真实医学图像能泛化。改进方向应针对光度估计的域偏移和二次线性化，而非继续把求解残差当作主要误差来源。

## 5. 可复用的 PyTorch layer API

`src/qcopt/neural_bijection/dense/photometric_conductance.py` 提供 `PhotometricSpectralTutteLayer(side=257,fit_side=128)`。先将其移到目标 device，并调用一次 `prepare(device=...)` 以构造18个边基和响应；`forward(fixed,moving)` 接受同 device float32 `(B,1,512,512)` 图像，返回 `(B,257,257,2)` 原网格 P1 顶点 map；`forward_with_latent` 另返回 `(B,18)` 的图像条件系数。18个 `raw_mode_gains` 是普通可训练参数，图像 loss 可经最终图像查询/采样、正边权方程的隐式伴随回传到该参数及输入图像。响应只是由固定网格、频率、初始权重决定的可重算 buffer；加载权重后须再次 `prepare`，不能把预计算0.90 s 偷算成每步 forward 或反过来完全忽略初始化成本。

257²、batch2 的模块级独立核验：装载300步检查点后，新模块与研究脚本在同两张保留图像上的最大逐顶点坐标差 $5.96\times10^{-8}$、18维 latent 最大差 $6.51\times10^{-8}$；真实前向/伴随相对残差 $5.34\times10^{-11}$ / $1.25\times10^{-11}$，全部原面最小面积比0.559，输入固定图、移动图与18增益梯度范数分别 $5.71\times10^{-5}$、$6.95\times10^{-5}$、$5.83\times10^{-4}$，均有限。热身后五次完整 image-to-loss forward/VJP 中位36.24/51.07 ms、峰值已分配显存约215 MB；只常驻两幅图，因此低于训练32例常驻时的340 MB。这个 API 不是仅在工具脚本里能运行的伪层。

复现入口：tools/phase6_test_c_photometric_response.py、tools/phase6_train_c_photometric_calibration.py、tools/phase6_verify_photometric_layer257.py；原始逐例 JSON 以 raw_results/c_photometric_response257_* 和 c_photometric_calibration_* 命名，模块回归在 raw_results/c_photometric_reusable_module257_gpu2.json，检查点为 checkpoints/c_photometric_calibration_mean0005_{300,1000}.pt。
