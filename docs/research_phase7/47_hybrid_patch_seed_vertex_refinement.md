# F2 粗 patch＋F1 细顶点：真正长程到细节的单张 P1 层

## 问题与输出契约

先前的[两尺度层](22_numeric_certified_joint_layer.md)从257²粗控制点到1025²，已经训练过图像网络；但若把同一比例机械放大为1025²→4097²，则所谓“粗层”本身已有百万点。一个[单独的大尺度探针](../../tools/phase7_benchmark_joint_large.py)在AI A6000、batch1、float64坐标/float32 latent、两轮F2 patch、一次F1细更新、4097²实际控制顶点、3355万原三角面上测得forward 152.63 ms、VJP 623.60 ms、CUDA allocated峰值19.552 GB，双组梯度有限非零、最小归一化面Jacobian .98751；但其最大位移只有$9.74\times10^{-4}$。这能检验大张量程序，却**不能**证明大尺度形变能力。把F1 fine pass做通用PyTorch checkpoint反而在这一运行增加峰值至19.804 GB、VJP增至721.32 ms；没有凭直觉声称checkpoint必然节省显存。峰值只算前向/VJP，另做的全脸检查没有改变峰值。

因此新增[混合层](../../src/qcopt/neural_bijection/dense/multilevel_forward_p1.py)：从很粗的$S\times S$控制网格出发，F2在粗网格直接移动相邻顶点，再反复**精确P1延拓**和F1仅移动新顶点，直到最终固定规则网格$N\times N$。输出是**一张定义在原有固定三角剖分上的连续P1映射**，不是连续映射的复合再采样。即使$N=4097$，本节的起点仍为$S=17$。

## 数学构造与安全性

令$\mathcal T_n$为$[0,1]^2$的$n\times n$顶点规则网格，每个小方格沿西南—东北对角线分成两个三角面。顶点坐标表$Y\in\mathbb R^{B\times n\times n\times2}$经每面仿射插值形成$f_Y$。所有边界顶点精确等于原坐标。记$h_n=(n-1)^{-1}$及目标绝对双面积下界$\beta h_n^2$，此处$\beta=.05$。

粗层一个**endpoint field** $z^j\in\mathbb R^{B\times(S-2)\times(S-2)\times2}$给每个内部点提议位移
\[
\delta_i^j=\sigma_S\tanh z_i^j,\qquad
\sigma_S=0.5p/(S-1),
\]
其中$p$为偶数patch边长（以粗网格cell计），边界$\delta^j=0$。从已有安全图$Y^j$出发，F2先形成$C$个阶段目标$Y^j+(c/C)\delta^j$，每阶段执行四个交错、片内并行的patch pass。每个pass对所有参与顶点施加同一个安全比例$t\in[0,1]$；每个受影响面的映后有向双面积是$t$的二次多项式$A(t)=A(0)+Lt+Qt^2$，选取使全部面维持正向并守住面积余量的$t$。这不是解随$S^2$增长的全局线性系统。可顺序输入$J\ge1$个endpoint field：$Y^{j+1}=\mathrm{F2}(Y^j,z^j)$，初值$Y^0=\mathrm{id}$。代码默认$J=1,C=4$；增大$J$允许沿一般同伦分段行进，而不能把单个$z$对应的直线阶段偷换成一般路径。

对每一级$n\mapsto 2n-1$，先作$P(Y)$：旧点取旧值，水平/竖直边中点取相邻端点平均，一个旧方格中心取该格**西南与东北两个对角端点**平均。因为中心位于原三角剖分的对角线，$f_{P(Y)}=f_Y$作为整个连续函数严格相等；双线性四角平均在一般P1图上不具有这个性质。然后仅在至少一个索引为奇数的新内部顶点使用F1四颜色安全顶点更新，旧even/even点的latent强制为零。每个新点沿原始raw方向走到正向面积允许的最大安全半径以内；每个面维持$A_T\ge\beta h_{2n-1}^2$。经$K$级后得到$D_{S,J,C,N}(z^0,\ldots,z^{J-1};z_{2S-1},\ldots,z_N)$。

**精确算术命题。** 若初始网格单位图正向、边界固定、所有latent有限、各局部安全器遵循上述面面积不变量，则归纳可知每一步所有原三角面有向面积严格为正、边界为单位方形。平面三角网格的局部正向加单射边界在这里给出全局P1同胚（相关定理、边界假设见[文献](06_prior_art_boundary.md)）；因此最终输出是方形到自身的P1 homeomorphism。这个保证来自构造，不依赖训练损失、后处理折叠修复或$\|\mu\|<1$。计算负载在固定$J,C,p$时是$O(\sum_n n^2)=O(N^2)$，并非$O(N^2)$个未知量的系统求逆；完整反传仍要保留或重算$O(N^2)$的中间量。

**浮点契约。** 最后还在实际返回的float64坐标上逐面作带舍入误差阈值的正向检查、边界精确相等检查；失败的整样本返回单位图。这是运行时数值防线，不能把它写成对任意硬件和任意NaN输入的抽象实数定理。通过检查的样本保留普通自动微分；检查分支切换处不可微，回退样本梯度为零。下列teacher样本均未回退。

## 逼近范围：一定区分“架构并集”与固定网络

给定边界固定、空间与时间足够光滑且Jacobian具有正裕量的同胚路径$F_t$，先选足够细的**仍然有限**粗seed $S$。按[F2小步命题](26_f2_uniform_isotopy_approximation.md)，把时间分为足够多段：每段的采样顶点差$\delta^j=I_SF_{t_{j+1}}-I_SF_{t_j}$足够小，因而可以写为$\sigma_S\tanh z^j$，四个patch pass均不被安全器截断，精确抵达下一个采样图。这里需要可选的$J$，而不是只增大单个field内部的$C$。随后每次Dyadic P1延拓后，目标新顶点与延拓值之差为$O(h_n^2)$；按[F1定理](23_uniform_isotopy_multilevel_approximation.md)，足够细的起点使每级一轮新顶点安全更新精确抵达$I_nF_1$。所以**允许$S,J,N$依目标变化的混合架构并集**覆盖这个光滑同伦类的采样P1图。

结合[边界固定全群$C^0$论证](30_all_boundary_homeomorphisms_c0_density.md)，该并集还对边界固定的全部连续homeomorphism定性稠密；前提是相对边界的任意$C^0$小光滑化。独立文献核查确认[Hatcher正式发表版本](https://ems.press/content/serial-article-files/52184?nt=1)的Theorem B之后**明确写出**相对边界固定及可任意小的isotopy（文中第2页），早期10页预印本仅显式写前者。也可由[Yagasaki Proposition 3.1](https://arxiv.org/html/math/0010224v1)得到相对固定紧多面体时的PL映射$C^0$稠密，但PL到光滑仍需额外步骤，不能只引该命题就完成本证明。这里的稠密性仍不提供统一逼近率、固定17² seed的普适性、可学习latent的存在或低成本逆编码。

## 大网格教师实验

独立解析目标$F_a(x,y)=(x+a\sin(2\pi x)\sin(\pi y),\ y-0.8a\sin(\pi x)\sin(2\pi y))$；它在内部patch seam有非零位移。teacher只在本验证中读取目标顶点，令粗latent等于$\operatorname{atanh}((I_SF_a-\mathrm{id})/\sigma_S)$，细级latent等于$\operatorname{atanh}((I_nF_a-P(I_{(n+1)/2}F_a))/(2h_n))$，旧顶点掩码为零。这样测试的是decoder可达性与VJP，不是图像推断。真实目标最小面$J$在$a=.14$也为正，但该值**不能**反证[早期解析连续下界](24_global_smooth_isotopy_teacher_test.md)不适用所留下的全域正则性空白。

AI主机空闲RTX A6000 GPU7，PyTorch2.5.1+cu124，batch1，float64，$S=17$，粗endpoint field一个且$C=4$，每级一轮F1，最终做全原三角面证书；[复现脚本](../../tools/phase7_verify_hybrid_pyramid.py)。时间是CUDA同步后forward和VJP分别取中位，峰值为PyTorch CUDA allocated，不含纹理生成、图像CNN或训练。误差$\sqrt{N^{-2}\sum_i\|Y_i-F_a(x_i)\|^2}$；$J_{\min}$是每面映后有向双面积除以$h_N^2$的最小值。VJP用$N^{-2}\sum_i(Y_{i,x}+.37Y_{i,y})$，逐组检查所有粗/细latent梯度有限且至少一项非零。

| 原控制网格 / 面数 | $a$、$p$ | 目标/输出$J_{\min}$ | 顶点向量RMSE | forward / VJP | allocated峰值 |
|---|---|---:|---:|---:|---:|
| 257² / 131,072 | .08，4 | .564466 / 同值 | 4.80e-19 | 40.00 / 122.65 ms，5次 | 0.0873 GB |
| 1025² / 2,097,152 | .08，4 | .565891 / 同值 | 5.07e-19 | 46.02 / 153.20 ms，5次 | 1.393 GB |
| 1025² / 2,097,152 | .14，8 | .205058 / 同值 | 4.42e-18 | 59.09 / 120.76 ms，5次 | 1.392 GB |
| 4097² / 33,554,432 | .08，4 | .566248 / 同值 | 5.26e-19 | 194.94 / 737.68 ms，3次 | 22.183 GB |
| 4097² / 33,554,432 | .14，8 | .205050 / 同值 | 4.58e-18 | 193.37 / 736.78 ms，3次 | 22.183 GB |

最大坐标误差在这些运行中不超过$5.56\times10^{-17}$；全部潜变量组VJP有限非零，边界不动，无输出回退。$a=.14$的1025²/17²/4粗阶段已经精确，而[纯F1同样17²/4粗轮数](24_global_smooth_isotopy_teacher_test.md)曾被局部安全缩放、RMSE $3.715\times10^{-4}$；纯F1增至8粗轮也可精确。这体现patch粗seed联合移动相邻点的一项优势，不能把不同脚本的时间直接当成严格同条件排名。与[全级F2](26_f2_uniform_isotopy_approximation.md)相比，混合层只在粗层用较贵的patch pass，细级改用更简的顶点pass；数值时间和显存似乎更低，但随机cotangent及脚本计时口径不同，严谨速度结论需同设备同cotangent复跑。

**限制。** 本节4097²是教师latent测试而不是4097²图像训练；大网格20GB级VJP也不能宣称低端GPU可训练。任意拓扑安全与定性稠密性并不保证给定17²/4周期能表示所有目标。图像编码器能否在真实数据中找到多尺度latent、能否消除现有hard前处理、以及固定规模下的速度—精度优势都必须另测。
