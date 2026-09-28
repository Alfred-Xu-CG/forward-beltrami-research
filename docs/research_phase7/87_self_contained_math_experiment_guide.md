# Phase VII 方法与实验的自包含说明

本文件回答两个不同问题：**生成问题**是给定 latent \(z\)，能否快速输出一张固定三角网格上的分片仿射同胚并向 \(z\) 反传？**推断问题**是给定图像，网络能否预测产生正确几何的 \(z\)？前者的 teacher 实验知道真值地图；后者的 image-only 训练不知道真值地图。不得将两者的误差混为一谈。本轮没有把任意逐面 Beltrami 系数的精确恢复当作训练目标。

## 1. 几何对象、输入和输出

取 \(\Omega=[0,1]^2\)，\(N=2^k+1\)，\(h=(N-1)^{-1}\)。源顶点为 \(x_{ij}=(jh,ih)\)，\(0\le i,j<N\)。每个方格按西南—东北对角线切成 \((\mathrm{SW},\mathrm{SE},\mathrm{NE})\) 与 \((\mathrm{SW},\mathrm{NE},\mathrm{NW})\) 两个三角形，集合记为 \(\mathcal T_N\)。因此有 \(V=N^2\) 个**控制顶点**、\(F=2(N-1)^2\) 个**原三角形**；4097²分别为16,785,409和33,554,432。

解码器输出顶点像 \(Y=(Y_{ij})\in\mathbb R^{N\times N\times2}\)。若 \(x=\lambda_i x_i+\lambda_jx_j+\lambda_kx_k\in T\)，其中 \(\lambda\ge0\)、\(\sum\lambda=1\)，定义 \(f_Y(x)=\lambda_iY_i+\lambda_jY_j+\lambda_kY_k\)。这叫**一张固定原网格 P1 图**：每个原面各有一个常矩阵导数 \(Df_Y|_T\)，且相邻面上的函数连续。它不同于多张 P1 图的连续复合；后者通常要在新划分上才是 P1。

边界控制顶点始终固定 \(Y_i=x_i\)。面 \(T=(i,j,k)\) 的归一化有向面积为

\[
J_T(Y)=\frac{\det(Y_j-Y_i,Y_k-Y_i)}{\det(x_j-x_i,x_k-x_i)}.
\]

在本规则圆盘三角网、恒等边界、连续P1及所有原面 \(J_T>0\) 的条件下，映射是整个 \(\Omega\) 到自身的同胚。直观上正面积令每面局部不反向，而边界一一对应与拓扑度数排除了内部多重覆盖；**若边界不满足条件，仅有正面积不足以推出全局单射**。本轮的实数局部算子以归纳维持这些条件。float32/64实现另用对实际输出坐标的逐面保守符号证书；拒绝则返回恒等图。这个离散分支在切换处不可微，拒绝样本对latent的梯度为零。

一张512×512图像上的查询网格记 \(Q_{512}\)。在这些点用上述重心插值得到 \(f_Y(q)\)，再双线性重采样移动图。**512²是图像像素/查询数，绝不等于4097²控制顶点数**。

### 1.1 三种需要分开的输入—输出实验

| 任务 | 输入 | 输出 | 真值地图是否参与算法 |
|---|---|---|---|
| latent-only 解码器 | 每尺度两个分量的有限实数latent张量；固定源网格 | \(Y\in\mathbb R^{B\times N\times N\times2}\)，证书布尔值；可选 \(f_Y(Q)\) | teacher试验中用已知真值**构造**latent；解码器前向本身不读取真值 |
| 图像推断网络 | 固定图 \(I_F\)、移动图 \(I_M\in\mathbb R^{B\times C\times512\times512}\)；固定网格坐标 | 网络latent、\(Y\)、证书、512²查询地图和重采样图 \(I_M\circ f_Y\) | image-only训练中不读取真值；只在事后评价用真值 |
| VJP实验 | 上述输入与输出cotangent \(q\)，或图像损失 | \(\nabla_z\langle q,Y\rangle=(D_zY)^\top q\)，训练时还得到 \(\nabla_\theta\mathcal L\) | 随机cotangent测试不测图像推断；image-only反传不使用真值地图 |

这里 \(B\) 是batch，\(C\) 是独立纹理观测数；\(\theta\) 是编码器参数。VJP（vector–Jacobian product）只是反向自动微分的一次梯度乘积，不是另存一个巨大稠密Jacobian矩阵。

## 2. 为什么粗到细不会悄悄破坏固定P1输出

从 \(N\) 加密为 \(2N-1\) 时，保留旧顶点值；旧水平/竖直边的中点取端点平均；每个方格中心位于原SW–NE对角线上，取该**对角线两端**的平均。这样在每个子面上表达的仍是同一个旧P1函数。特别是中心点不可改为四角双线性平均，否则一般不保持旧P1函数。再在细网格上安全修改控制顶点，得到一张**新的、仍在细网格上P1**的函数。没有把两个连续映射复合后直接采样来声称P1。

## 3. F1：四颜色、单顶点的显式安全更新

固定一个当前内点像 \(p=Y_i\)。它周围六个面各有一条不含 \(i\) 的相对边 \(e_T=b_T-a_T\)，按取向选符号使当前有向双面积 \(A_T=\det(e_T,p-a_T)>0\)。给 \(i\) 两维latent \(z_i\)，先提出 \(r_i=\alpha h\tanh(z_i)\)，其中tanh逐坐标作用，常用 \(\alpha=2\)。若直接移动会降低某面的面积，其损失为 \(\ell_T=[-\det(e_T,r_i)]_+\)。取安全比例 \(0<\gamma<1\)（典型.85）和可选绝对双面积底线 \(\beta h^2\)（实验常用 \(\beta=.05\)），定义

\[
m_T=\min\{\gamma A_T,[A_T-\beta h^2]_+\},\qquad
s_i=\min\left\{1,\min_{T:\ell_T>0}\frac{m_T}{\ell_T}\right\}.
\]

空的内层最小值按1处理。输出 \(p'=p+s_ir_i\)。每个相关面满足 \(A'_T=A_T+s_i\det(e_T,r_i)\ge A_T-m_T\ge(1-\gamma)A_T>0\)。如果起始 \(A_T\ge\beta h^2\)，同时有 \(A'_T\ge\beta h^2\)；如果起点已低于floor，公式只禁止不利运动，不能自动修复该面。把内点按行列奇偶分四颜色，同一面在同一pass最多有一个活动顶点，因此组内能并行；四pass依次读最新地图。后续细级只给新顶点非零logit，旧顶点仍可在其他架构的粗seed阶段运动。

真实代码为防非活动除法及VJP溢出，用 \(g=\sqrt{\mathrm{eps(dtype)}}h^2\) 作为分母下限，即对 \(\ell_T>0\) 取 \(m_T/\max\{\ell_T,m_T,g\}\)。这**只会缩小位移，不破坏正面积**，但在极薄面上可能让按理想实数公式本可一步完成的teacher更新被截断。因此下述逼近定理只属于无正guard的理想实数算子；float实现的拓扑依赖最终证书，实际训练的分支不保证处处可导。

## 4. F2：四个交错patch pass，联合移动相邻顶点

F1一次移动一个顶点，窄可行域可能限制协同形变。F2把方格按偶数边长 \(p\)（例如4或8个cell）划为互不交叠的patch，每块边界固定、内部多个相邻点有共同尺度 \(s_P\)。对patch中一个原面，旧顶点 \(a,b,c\)、提议位移 \(\delta a,\delta b,\delta c\) 给出**精确**面积多项式

\[
A_T(s)=A_T(0)+sL_T+s^2Q_T,\quad
L_T=\det(\delta b-\delta a,c-a)+\det(b-a,\delta c-\delta a),\quad
Q_T=\det(\delta b-\delta a,\delta c-\delta a).
\]

对 \(0\le s\le1\)，令 \(C_T=(-L_T)_++(-Q_T)_+\)，则 \(A_T(s)\ge A_T(0)-sC_T\)。取与F1相同形式的预算 \(m_T=\min\{\gamma A_T(0),[A_T(0)-\beta h^2]_+\}\)，并令 \(s_P=\min_{T\subset P}\min\{1,m_T/C_T\}\)（\(C_T=0\)记尺度1）。于是patch内所有原面仍正向；patch边界没动，外面不受影响。四次偏移 \((0,0),(p/2,0),(0,p/2),(p/2,p/2)\) 覆盖所有内部顶点，使上一pass的seam也能在后续pass移动。latent提出的endpoint即使本身会折叠，也只执行被预算允许的安全部分。

实装还有与F1同量级的正分母guard；它不扩大位移，但会影响极薄面的一步可达性。F2的优点不是无条件比F1快：它能让相邻点联合运动；代价是四组patch内部的二次面积、更多中间张量和较大峰值显存。

## 5. 混合图像网络的具体计算图

4097²主要图像实验使用**F2的17²粗种子＋F1的细级**。17²seed在同一endpoint场上作4轮patch周期；33²、65²、129²、257²通过卷积编码器给出的早期latent作安全细化。编码器宽度16，输入第一幅固定/移动灰度图、两张坐标通道和由该图像对计算的两张局部2×2光流特征通道，共6通道。两层3×3 stem、四个池化上下文块及一层融合卷积在257²特征格上运行，1×1 head按层输出两个坐标的latent。4097²实验的编码器虽然声明513²/1025²两个head，实际image-feedback路径不使用这两head；约15,038个声明标量中实际有梯度的是14,970个编码器标量及4个增益。513²/1025²各作两轮细级反馈，2049²/4097²各作一轮；每轮的反馈在当前地图上重新计算。

513²、1025²、2049²、4097²不是由4097² CNN直接预测任意稠密latent，而是在**当前**P1地图 \(g\) 上用图像残差产生局部提示。对通道 \(c\) 定义 \(r_c(x)=I_{F,c}(x)-I_{M,c}(g(x))\)、\(v_c(x)=\nabla I_{M,c}(g(x))\)。在控制顶点附近 \(3\times3\) 窗口求

\[
G_x=\rho I_2+\frac1C\sum_c\langle v_cv_c^\top\rangle_{W_x},\quad
b_x=\frac1C\sum_c\langle v_cr_c\rangle_{W_x},\quad
d_x=G_x^{-1}b_x,\qquad\rho=1.
\]

这是每个顶点独立的**2×2显式逆**，不是尺寸随 \(N^2\) 增长的global linear solve。它来自一阶线性化 \(r_c\approx v_c^\top d\)，但只是假设附近共用小位移的近似。将 \(d_x/(2h)\) 截到 \([-.95,.95]\)、取atanh，再乘可学习的正增益，作为F1原始提议的logit；F1本身决定实际安全位移。局部提示可反传到图像取样、增益和先前地图。图像只有512²像素，即使控制网格4097²，也没有凭空产生额外观测信息。

四视图对称版本让**同一个**编码器分别处理四组图像，给每个latent坐标四个数 \(z_j^{(1)},\ldots,z_j^{(4)}\)。按值排序后取中间两个平均 \(\bar z_j=(z_j^{[2]}+z_j^{[3]})/2\)，再调用**同一个**安全P1解码器；不是平均四张地图。它在数学上对通道排列不变，排序相等点不光滑。若每个坐标有三个好预测落在区间 \([a_j,b_j]\)，融合值仍在该区间；但这不保证三个预测确实好，也不是地图误差定理。

训练的图像目标是

\[
\mathcal L_{\rm img}(\theta)=10^6\frac1{BCHW}\sum_{b,c,q}
\bigl[I_{M,b,c}(f_{D(E_\theta(I_F,I_M))}(q))-I_{F,b,c}(q)\bigr]^2.
\]

固定/移动图张量形状各为 \(B\times C\times512\times512\)；输出顶点表形状 \(B\times4097\times4097\times2\)，另有每样本接受/回退标志；用于损失的查询地图为 \(B\times512\times512\times2\)。训练使用Adam；真值地图不进入上式。图像模型的最细latent主要是上述局部提示加少数增益，**不是可自由选择的3353万个独立latent组成的通用逆编码器**。

## 6. 逼近数学结论的量词与边界

给一类边界恒等的光滑同胚同伦 \(F_t\)，\(F_0=\mathrm{id}\)，\(F_1=F\)，并假设对全类共同有 \(\det DF_t\ge m>0\)、\(\|DF_t\|\le L\)、\(DF_t\) 的空间Lipschitz常数至多 \(K\)、\(\|\partial_tF_t\|\le M\)。取足够细的**共同**seed间距 \(H_0\)、足够多但有限的共同seed轮数；理想无guard实数F1或F2之后每次dyadic加密只要一个四pass周期，就能选teacher latent使输出**恰为** \(I_hF\) 的顶点表，并有 \(\|I_hF-F\|_\infty\le Kh^2\)、全部层总工作 \(O(V_h)\)。这说的是**已知目标与其同伦时的存在性**，不是图像编码器从未知图片找到latent的算法。

具体充分条件可读作：若当前每面双面积至少 \(ah^2\)、边像长度至多 \(Eh\)，每点拟移至多 \(dh\)，则F1以 \(B_1=d(E+2d)\) 满足
\[
d<\alpha,\qquad B_1<\min\{a/6,\gamma a/2,a/2-\beta\}
\]
时整个周期不截断。F2以 \(B_2=4Ed+12d^2\) 满足
\[
d<p/2,\qquad B_2<\min\{a/8,\gamma a/2,a/2-\beta\}
\]
时一个交错周期不截断。选择seed足够细可令采样 \(I_{H_0}F_t\) 每面 \(a=m/2\)；把时间切得足够密有 \(d\le M/(N_0H_0)\)；每次二分后新顶点对旧P1的偏差不超过 \(KH^2\)，折算细网格 \(d\le2KH\)，故同一个足够细seed使全部后续级满足上述不等式。实际float代码还需预算大于 \(\sqrt{\mathrm{eps(dtype)}}h^2\) 才可排除guard截断；本轮3×3极薄面单测给出了理想/float一步可达性不一致的反例。

对所有边界固定同胚的更强 \(C^0\) 稠密命题，仅适用于**无正面积floor、无guard的理想实数算子，且允许架构随目标/误差改变**；它不推出固定网络、固定低维latent或本轮.05 floor实现的全群逼近。事实上若严格要求所有原面 \(J_T\ge\beta>0\)，把中心半径 \(r_0=1/8\) 的圆盘压缩至 \(\delta r_0\)（\(0<\delta<\sqrt\beta\)）的边界固定光滑同胚，任何这种P1近似的最大误差至少 \((\sqrt\beta-\delta)r_0\)。固定4097²网格的合法顶点表还是一个 \(d=2(4095)^2=33{,}538{,}050\)维开集；固定权重、局部Lipschitz、输入维数 \(m<d\) 的decoder不可能按全维Lebesgue测度**精确覆盖几乎所有**这些表。两条限制都不否定针对特定低维任务分布的近似。

## 7. 实验数据和指标先定义，再读数字

主要合成移动图是由随机相位的正弦/余弦、两个随机中心Gaussian bump及一项较高频的正弦余弦乘积组成的512²灰度纹理 \(I_M\)。对独立生成的真地图 \(F_*\) 在512²像素 \(q\) 上双线性重采样，得固定图 \(I_F(q)=I_M(F_*(q))\)。`high128`族的典型位移为
\[
F_*(x,y)=(x,y)+\bigl(a_x b(x,y)+a_f\phi(x,y),\ a_yb(x,y)+a_f\phi(x,y)\bigr),
\]
其中 \(b=\sin(2\pi x)\sin(2\pi y)\)，\(\phi=\sin(256\pi x)\sin(256\pi y)\)；各系数和纹理在训练、评价种子下重新抽样。`high128_tri`改成三个高频载波，`high128_tiles`改成16个局部包；`spots`用少数Gaussian斑点代替某通道纹理。四视图实验是四个**不同纹理**受**同一个** \(F_*\) 变形；复制四份同一纹理是阴性对照。32训练地图seed55101，通常128留出地图seed20270317或后续预定的新seed；训练/评价地图不同，但仍属于这些合成分布。自然图像未测试。

定义 \(\mathrm{mapRMSE}=\sqrt{(BQ)^{-1}\sum_{b,q}\|f_{Y_b}(q)-F_{*,b}(q)\|_2^2}\)；teacher `vertexRMSE`把 \(q\) 换成全部 \(V\) 控制顶点。\(\mathrm{imageMSE}=(BCHW)^{-1}\sum(I_M\circ f_Y-I_F)^2\)。`modeRMSE`只比较指定正弦基上的位移投影系数，**不是全部频率的地图误差**。`minJ`扫全部原面；`accepted`表示最终证书直接输出候选，`fallback`表示改为恒等图。热态时间取CUDA同步重复运行的中位；`allocated`/`reserved`是PyTorch十进制GB峰值，不含全部驱动或其他进程显存。不同表是否含CNN、512²查询、VJP、Adam会单列。

## 8. 主要实验：究竟验证了什么

| 实验与明确问题 | 网络/算法输入→输出、训练设置 | 结果：误差/拓扑 | 成本与严格解释 |
|---|---|---|---|
| [E1 F1/F2 teacher 4097²](74_two_distinct_forward_mechanisms_4097_teacher_vjp.md)：已知目标时两原语可否真实大网格前向/VJP？ | 独立解析 \(F_{.08}=(x+.08\sin2\pi x\sin\pi y,\ y-.064\sin\pi x\sin2\pi y)\)；从其17²至4097²顶点样本**反算12个teacher latent场**（seed4＋细级8），batch1，float32；输出全体4097²顶点，随机cotangent反传到全部12场；无CNN/图像。 | F1/F2 vertexRMSE均 \(3.436\times10^{-10}\)，目标/输出minJ=.5660，证书接受。 | 同AI RTX A6000 GPU7，含decoder forward＋**一次全latent VJP**，F1 .559s/12.561GB allocated，F2 1.135s/19.941GB；不含teacher生成、CNN或图像。只验证已知目标的可达/可导，不是逆推准确率。 |
| [E2 三模型同图像任务](49_same_task_three_forward_pyramids_and_ood.md)：F1、F2、混合谁更适合学习？ | 单幅512²固定/移动图→257²或1025²顶点。width16 CNN，`high32`合成训练32例；257²从零1000步，1025²迁移600步，batch2，仅imageMSE训练，128新例。 | 1025² mapRMSE：F1 .003258、F2 .002818、混合 .002912；各128例正向，无回退。 | 1025²完整训练步 .144/.253/.223s；allocated 1.745/2.670/2.920GB。F1/F2几何float32、混合float64；这是相同任务预算的实现比较，**非严格同dtype/同精度Pareto**。F2此设置最准、F1最快。 |
| [E3 high128 单图反馈](57_high128_trainable_feedback_layer.md)：细级图像残差是否有用？ | 单幅图→1025²混合层；32训练例、500步image-only及细修正500步；128例评价。 | 粗模型mapRMSE约 \(4.9\times10^{-4}\)；指定128周期模态经细修正约减半；换稀疏spots纹理mapRMSE约.0082；安全仍通过。 | 说明局部反馈有特定高频作用，但单图外观敏感。该组没有可与E1 teacher直接比较的统一速度数字，不能凭误差推算4097²成本。 |
| [E4 4097²图像训练](53_mesh_free_queries_and_4097_image_training.md)、[完整编码器](75_full_encoder_4097_training_single_vs_four_views.md)：整条图像→P1链能否训练？ | 先有high32迁移100步；主协议为C=1/4，32训练例，batch1，300步Adam，编码器及4个增益以imageMSE×\(10^6\)反传；输出4097²顶点，512²查询；128新例。 | high32迁移mapRMSE .001245；high128 C1完整再训mapRMSE \(4.905\times10^{-4}\to5.580\times10^{-4}\)（变差），C4 \(1.497\times10^{-5}\to1.490\times10^{-5}\)（微小改善）；300/300训练及128/128留出通过证书。 | high32早期全训练步1.089s/22.854GB；主C1 .713s/14.740GB、C4 .762s/15.934GB allocated。数字含CNN、解码器、查询、图像重采样、loss、backward、Adam，非纯layer时间。 |
| [E5 独立通道/重复通道](68_multiview_observability_4097_p1.md)、[通道数与噪声](70_multiview_channel_count_noise_tradeoff.md)：改善来自信息还是张量数量？ | 固定权重不重训；同一真地图，一/二/四/八/十六独立纹理，或四份完全重复；4097²输出，512²查询，新128例。 | C1/C2/C4的mapRMSE约 \(4.83\times10^{-4}/1.10\times10^{-4}/1.52\times10^{-5}\)；四份重复恢复C1误差；C8/C16无噪声为 \(6.28\times10^{-6}/5.39\times10^{-6}\)；加独立噪声 \(\sigma=.001\) 时C4/C8/C16为 \(1.73\times10^{-4}/8.94\times10^{-5}/5.66\times10^{-5}\)。各组所测样本安全接受。 | 旧实现C1/C4完整forward约.132/.143s，forward＋VJP约.785/.841s，峰值VJP 12.826/13.659GB。多独立纹理改善合成可观测性；重复不改善；噪声仍明显。 |
| [E6 计算/内存消融](77_4097_multiview_checkpoint_memory_tradeoff.md)—[82号](82_compiled_color_kernel_full_training_pareto.md)：能否较快较省显存地完整训练？ | 同C4、32训练图、16留出图预置、batch1、固定20步序列；变更F1索引生成、逐颜色activation重算、局部编译或CPU saved-tensor暂存，不变目标/拓扑公式。 | 各行20训练与16留出均过证书；编译300步的独立128例mapRMSE \(1.490054\times10^{-5}\)，与未编译300步近似。 | generated直接 .7644s/16.074GB allocated/18.346GB reserved；逐颜色重算 .8766s/7.885/10.815GB；编译不重算 .3449s/8.997/10.897GB；**编译＋重算 .3671s/6.778/9.030GB**；再CPU暂存1.602s/4.022/6.965GB，主机RSS约15GiB。均为热态完整训练步；首次编译约14—37s。 |
| [E7 batch/容量/数值](81_batch_scaling_4097_color_checkpoint.md)—[83号](83_compiled_color_extreme_latent_stress.md)：扩batch和编译是否稳？ | 4097² C4、batch4、20步完整image-only训练；另随机latent幅度5/20各32例正向、幅度1/5各16例随机VJP；前后实现对照。 | batch4全部80张训练输出过证书；随机正向各32/32过证书；VJP各16/16有限。编译与eager梯度相对L2最大差约0.36%/0.71%，**非逐位相同**。 | 编译＋重算batch4约1.063s/24.727GB allocated/34.880GB reserved。7GiB **PyTorch allocator上限**＋CPU暂存20步通过，不是物理8GB设备实测。 |
| [E8 对称四视图融合](84_4097_trained_multiview_channel_order_ood.md)—[85号](85_permutation_invariant_coarse_latent_fusion.md)：坏首通道能否不支配粗latent？ | 原粗CNN只读第一幅；新方案共享CNN读四幅、逐坐标去极值平均，再用同一4097²拓扑层。相同初始权重、32训练图、300步image-only、batch1；新128例及设计定稿后另128例。 | 标准四视图旧/新mapRMSE约 \(1.490\times10^{-5}/1.424\times10^{-5}\)；新seed一幅spots＋三幅standard为.004363/**.00003208**；两者各128/128通过。重复四图新方案仍.0004906；加噪 \(\sigma=.001\) 时旧/新.000171138/.000171245，**没有抗噪收益**。 | 300步热态训练旧/新.36555/.36908s、allocated7.955/8.075GB；完整batch1热态推理（编码＋P1＋证书＋查询＋重采样）.08541/.08868s、allocated3.206/3.250GB。训练峰值含128评价图常驻；首次编译另计。 |
| [E9 阴性边界](66_trained_4097_spatial_correction_negative_result.md)、[71号](71_single_image_exact_p1_nonidentifiability.md)—[73号](73_sparse_image_hint_dense_p1_negative_result.md)：为何低像素损失不足？ | 单图加入高频修正/真地图监督诊断；固定P1单图不可辨识反例；把最细局部提示降到2049²再上采样。 | 图像MSE可降但mapRMSE不一定降；精确反例有两张不同P1同胚产生同一非恒定图像对；稀疏提示mapRMSE由约 \(1.51\times10^{-5}\) 变 \(4.64\times10^{-5}\)。 | 稀疏提示forward＋VJP .756→.704s，小速度收益换几何精度；真地图监督诊断不能算image-only成果。 |

E1、E2、E4、E6、E8的时间**包含范围不同**，不可横向组成单一“最快算法”榜单。对主模型最接近使用者关心的两个数字是：4097²四视图、batch1完整热态推理约.085—.089s；同任务完整训练热态约.365—.369s/步、PyTorch allocated约7.96—8.08GB，reserved约10GB。它们是在48GB RTX A6000、PyTorch2.5.1+cu124、编译缓存热态与合成图像上测得；不是跨GPU保证、不是4097²任意地图求逆时间，也不是物理8GB卡训练可行性证据。

编号1—46还记录了为选择上述主模型而做的探索，并非另有46个相同协议的4097²网络。其问题、输入、输出如下；准确的各次计时须回到对应原文，不能跨任务拼在E1—E9的热态速度表中：

| 探索簇 | 输入→输出 | 解决的问题及最终定位 |
|---|---|---|
| [1—9号](01_refinement_reachability.md) | 解析目标或512²图像→多尺度F1 latent→257²/1025²固定P1 | 证明原始细化/局部顶点更新确能移动继承点并做初步VJP；从单个已知细模式走向空间表示，不能当未知几何的全面逆推。 |
| [10—15号](10_multimode_spectral_readout.md) | 图像或多尺度标量场→预设正弦字典系数、或正边增量fiber latent→1025²固定P1 | 预设字典可以高效读出指定模式；[单坐标fiber](13_monotone_fiber_forward_layer.md)保证一个受限同胚族，却不能表达一般双坐标耦合。 |
| [16—22号](16_dense_two_component_f1_fit.md) | 独立双坐标目标→直接优化稠密F1 latent；或endpoint latent→F2 patch→1025²固定P1 | 检查不预置单一正弦字典的双坐标细节、相邻顶点联合运动、有限精度证书；这些是表达/安全测试，不能直接代表图像推断。 |
| [23—31号](23_uniform_isotopy_multilevel_approximation.md) | 光滑同伦及目标顶点表→构造teacher latent→固定P1；或局部图像提示→该层 | 给出上文限定的统一类逼近证明，实测teacher并量化图像编码器与teacher的距离；[维数/条件数分析](31_latent_dimension_conditioning_tradeoff.md)解释潜在空间不能既任意低维又精确覆盖全维地图。 |
| [32—39号](33_unknown_carrier_fft_encoder.md) | 未知整数/非整数高频载波的图像→频谱/局部解调/空间CNN latent→1025²固定P1 | 检验“网络事先知道高频基函数”这一捷径；FFT/软连续频率只估计图像特征，最终仍需F1保拓扑。受噪声、载波错估和硬选择影响，未成为最终通用4097²方案。 |
| [40—46号](40_mixed_scale_forward_p1.md) | 局部低频窗、两个重叠窗或局部仿射目标的图像→小型局部图像前端＋F2/F1→1025²固定P1 | 测多尺度/局部仿射表达及image-only训练；[CPU流式数据](46_cpu_streaming_memory_scaling.md)还区分训练数据常驻成本与层自身激活。局部前端的部分hard选择不可反传，故不把它混同于最终端到端4097²链。 |
| [47—52号](47_hybrid_patch_seed_vertex_refinement.md) | 512²单图像对→17² F2 seed＋F1细级→257²/1025²固定P1 | 确立后续主混合架构，并发现光流特征在同分布可帮助但稀疏spots外观导致明显失败；这是开展多视图/对称输入实验的直接动机。 |

## 9. 严格结论：成功、未成功、下一步

**已经证实的限定命题**：F1和F2在精确算术下对全部有限latent逐步保持本固定网格的P1同胚；实际规定float32/64输出由最终证书过滤；两种机制各自在4097²控制网格实测完整teacher forward及全部latent VJP；混合图像网络在4097²合成多纹理任务上完成300步image-only训练及128例留出，反向确实到达参与的编码器参数。以当前实现，在同GPU上F1大teacher测试比F2快、省显存，而混合图像模型经过颜色重算和编译后有约.37s/步的热态训练实例。

**没有证实、不能暗示的命题**：给一幅自然图像即可准确恢复真形变；固定小网络能从图像实现所有理论teacher latent；所有有限latent处处有连续/光滑梯度；数学上的全群稠密性适用于固定.05 floor或固定dtype guard；7GiB allocator模拟等于物理8GB GPU验证；F1/F2已在同一4097²任务公平击败正导纳/Tutte全局求解基线。拓扑安全与图像可辨识是两件事。下一阶段最值得做的是独立自然/跨模态图像与几何真值、物理小显卡、分支梯度稳定性，以及同任务同精度的结构化求解基线。

方法源码：[F1](../../src/qcopt/neural_bijection/dense/colored_vertex_relaxation.py)、[F2](../../src/qcopt/neural_bijection/dense/patch_field.py)、[精确细化与证书](../../src/qcopt/neural_bijection/dense/multilevel_forward_p1.py)、[图像编码器](../../src/qcopt/neural_bijection/dense/forward_p1_encoder.py)、[局部提示](../../src/qcopt/neural_bijection/dense/photometric_hint.py)。实验入口：[teacher](../../tools/phase7_verify_isotopy_pyramid.py)、[图像/资源](../../tools/phase7_feedback_scale_probe.py)。各表链接到原始设置与JSON，不依赖未定义的内部缩写。
