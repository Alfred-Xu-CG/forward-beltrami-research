# Route C：多频谱边字典的容量与图像推断

## 数学构造

这项实验保留 [正 conductance 图 Laplacian、正弦预条件 CG 和隐式伴随](09_sine_pcg_and_conductance_synthesis.md)不变，只修改图像到边权的参数化。原网格为单位正方形 $257\times257$ 顶点 SW–NE 三角剖分，含 131072 个三角形和 197120 条边；边按水平、竖直、东北对角三组排序。单位凸边界固定，所有内部点满足 $\sum_{j\sim i}c_{ij}(Y_i-Y_j)=0$。对每条边 $e$ 的实数 logit $z_e$，取 $c_e=1+15\sigma(z_e)$，所以任何有限网络参数下 $1<c_e<16$，精确平衡解为原网格 P1 Tutte homeomorphism。实现的 float64 PCG 只给近似坐标，另检查真实残差和 131072 个面的正面积；隐式 VJP 解同一对称系统，不保留迭代轨迹。

普通 edge CNN 在每条边中点取图像特征后输出 $z_e^{\rm CNN}$。新增六个频率 $K=\{1,2,4,8,16,32\}$ 的已知边字典：水平边中点 $m=(x,y)$ 用 $\phi_k^{H}(m)=\cos(2\pi kx)\sin(2\pi ky)$，竖直边用 $\phi_k^{V}(m)=\sin(2\pi kx)\cos(2\pi ky)$，对角边用 $\phi_k^{D}(m)=(\phi_k^H+\phi_k^V)/\sqrt2$。图像网络除逐边 logit 外还预测三组各六个全局系数 $a_{gk}(I_f,I_m)$，于是

\[
 z_e=-0.8+z_e^{\rm coarse}+z_e^{\rm fine}
       +\sum_{k\in K}a_{g(e),k}\phi_k^{g(e)}(m_e),
 \qquad c_e=1+15\sigma(z_e).
\]

最后一个频率 32 恰好包含本轮预设目标族的细频；这是刻意有利的、任务特定先验，不能当成任意 Beltrami 场都能由六频描述。编码器共 4116 参数，对照普通 edge CNN 的 3090 参数。

## 图像层训练

使用固定 high32 协议：32 训练（种子55101），8 保留（种子99317），图像512²、控制257²、batch2、Adam 学习率0.003。解析目标只用于事后 map/μ/32周期投影评价，**不进入 image-only 的训练目标**。完整 forward 包括边权生成、float64 PCG、257² 顶点插值到262144个图像 query、一次移动图像采样；VJP 包括图像、插值和隐式求解到网络参数。

| 训练变体 | 步数 | 保留 image MSE | query-map RMSE | 面 μ RMSE | 32周期振幅率 |
|---|---:|---:|---:|---:|---:|
| 普通 edge CNN，像素损失 | 1000 | 0.0008912 | 0.002721 | 0.1524 | 近零 |
| 谱字典，像素损失 | 300 | 0.0003818 | 0.001629 | 0.1342 | 0.0317 |
| 谱字典，像素损失 | 1000 | 0.0004016 | 0.001658 | 0.1360 | 0.0150 |
| 谱字典，像素+0.1 教师细边损失 | 300 | 0.0003855 | 0.001628 | 0.1338 | 0.0356 |

谱字典使图像和 map 指标明显好于普通 CNN，但 32 周期真实振幅的恢复仍极少，且从300到1000步略退化，不能称达到高频几何准确。1000步 image-only 谱字典训练墙钟73.2秒，batch2 forward/VJP 中位数31.96/43.88毫秒、峰值已分配显存314 MB；这些共享GPU时间仅供量级判断。求解残差与各面证书在原始 JSON 中，不能以较低 image MSE 替代 map/Beltrami 评价。

## 同一字典的容量诊断

把网络完全移开，直接优化仅 18 个谱系数，针对预定训练图像例29的**解析目标顶点坐标与细边导数**做500步 oracle 拟合。所得顶点 map RMSE $4.84\times10^{-4}$，预测 x/y 位移的32周期振幅均约0.002261，真实约0.002458，即约92%。全部原面正面积，最小面积比0.583，真实相对 PCG 残差 $4.48\times10^{-12}$。独立 197120 个自由边 latent 的同例 oracle map RMSE 约 $4.94\times10^{-5}$，说明18系数字典有限但**足以产生**大部分细频；图像网络只恢复约1–3%的振幅，主要差距在 amortized 图像推断/优化，而不是该18维字典完全没有相应方向。这个 oracle 使用目标几何作损失，绝不是合格的 image-to-latent 神经层泛化成绩。

复现：tools/phase6_spectral_edge_encoder.py、tools/phase6_train_c_multisample.py、tools/phase6_fit_c_spectral_coeff_oracle.py；逐项证据在 raw_results/c_sine_pcg_spectralbank_*.json 和 raw_results/c_spectral_coeff_oracle257_trainitem29_500_gpu2.json。上述所有时间均在 AI 主机空闲 GPU2 上实测。
