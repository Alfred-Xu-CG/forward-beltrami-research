# 全体边界固定平面 homeomorphism 的 \(C^0\) 稠密性：存在性定理及复杂度边界

## 1. 精确命题

设 \(\Omega=[0,1]^2\)，\(\operatorname{Homeo}_{\partial}(\Omega)\) 是在每个边界点取恒等的自同胚集合，赋予一致距离
\(d_\infty(F,G)=\sup_{x\in\Omega}\|F(x)-G(x)\|_2\)。边界点逐点固定已限定其取向为正。

记 \(\mathscr D_{F1}\)、\(\mathscr D_{F2}\) 是[交错顶点 F1](23_uniform_isotopy_multilevel_approximation.md)与[交错 patch F2](26_f2_uniform_isotopy_approximation.md)的**理想guard-free实数算子、所有合法架构规模**之并：允许选择足够细但有限的 seed 网格、有限 seed 更新轮数、最终规则 dyadic 网格分辨率；在本定性全群命题中取**不设统一绝对面积底线** \(\beta=0\)，代码参数对应 `minimum_jacobian=None`，但实际float32/float64代码仍有正分母guard。每个有限 latent 通过已证明的理想局部安全更新产生一张固定规则三角剖分上的 P1 同胚，边界恒等。令 \(\mathcal R\) 是这些 P1 连续映射的并集。实验中常用的固定 \(\beta=0.05\) 只是数值配置，不能不经额外证明就带入下述对任意小 Jacobian 目标的密度断言；同理，不能把固定dtype的guard在任意小Jacobian目标下无条件视为不活动。

**定理（定性 \(C^0\) 稠密性）。**
\[
\overline{\mathcal R}^{\,d_\infty}
=\operatorname{Homeo}_{\partial}(\Omega).
\]
也就是说，对于任意 \(F\in\operatorname{Homeo}_{\partial}(\Omega)\) 和 \(\varepsilon>0\)，存在一个足够大的但有限的 seed、足够细的最终**固定规则网格**及有限 latent，使 F1 或 F2 输出 \(D(z)\) 满足
\[
\|f_{D(z)}-F\|_{L^\infty(\Omega)}<\varepsilon.
\]
这个定理是**表示集合**的存在性定理，绝不宣称一个固定小 seed、固定网络、固定潜变量维数或已训练图像编码器对整个群统一逼近；也不提供目标到 latent 的低成本逆编码。

## 2. 第一步：把边界固定 map 缩到内部支撑

取 \(c=(1/2,1/2)\)，\(S_r(x)=c+r(x-c)\)，\(0<r<1\)。定义
\[
G_r(x)=
\begin{cases}
S_r\circ F\circ S_r^{-1}(x),&x\in S_r\Omega,\\
x,&x\notin S_r\Omega.
\end{cases}
\]
因为 \(F|_{\partial\Omega}=\mathrm{id}\)，两支在内方形边界吻合，且反映射由同样的缩放公式拼成，所以 \(G_r\) 是 \(\Omega\) 自同胚；其非恒等支撑紧含于内部。不能只写“把 \(F\) 截断为零”：那样的线性混合一般不保单射，这里的**共轭加恒等延拓**才是保拓扑操作。

令 \(R=\max_{x\in\Omega}\|x-c\|_2=\sqrt2/2\)，\(\omega_F(s)=\sup_{\|x-y\|\le s}\|F(x)-F(y)\|\) 是 \(F\) 的一致连续模。内方形内令 \(x=S_ry\)，则
\[
\|G_r(x)-F(x)\|
\le (1-r)R+\omega_F((1-r)R).
\]
内方形外，\(x\) 距某一外边界点 \(b\) 至多 \((1-r)/2\)；由于 \(F(b)=b\)，
\[
\|G_r(x)-F(x)\|
\le (1-r)/2+\omega_F((1-r)/2).
\]
故 \(G_r\to F\) 一致；这个结论对极不光滑、非 Sobolev 的 homeomorphism 也成立。这里并未使用光滑近似定理。

## 3. 第二步：相对边界的光滑化

为 \(r<1\) 选一个光滑圆角闭 disk \(D\)，使 \(S_r\Omega\subset\operatorname{int}D\subset D\subset\operatorname{int}\Omega\)。将 \(G_r\) 限在 \(D\)，它在 \(D\) 的一条边界邻域已经恒等。[Hatcher, *The Kirby torus trick for surfaces*, Theorem B 及其相对边界、小同伦细化](https://ems.press/content/serial-article-files/50953)说明：每个光滑曲面之间的 homeomorphism 都可同伦到 diffeomorphism；若原图在边界邻域已经是 diffeomorphism，同伦可在边界邻域固定，并可任意 \(C^0\)-小。故对每个 \(\eta>0\)，存在光滑 \(Q:D\to D\)，在边界邻域恒等，且 \(\|Q-G_r\|_\infty<\eta\)。在 \(D\) 外令 \(Q=\mathrm{id}\)，得到 \(Q\in\operatorname{Diff}^{\infty}_c(\Omega)\)。

注意这里的正则性结论来自**二维曲面定理**，不应把适用于高维或只给拓扑同伦、无任意小控制的其他陈述替换进去。前一节的缩放把方形的尖角问题转移到内部一张光滑 disk，避免直接把有角的 \(\Omega\) 当作 Hatcher 定理的光滑边界曲面。

## 4. 第三步：接到 F1/F2 的固定网格正向逼近

给定 \(\varepsilon\)，先选 \(r\) 使 \(\|G_r-F\|_\infty<\varepsilon/3\)，再用上节取 \(Q\) 使 \(\|Q-G_r\|_\infty<\varepsilon/3\)。[光滑紧支撑推论](27_compact_smooth_diffeomorphism_corollary.md)对该 **特定** \(Q\) 提供一条空间/时间光滑且内部支撑的同胚路径，因而有 \(m_Q,L_Q,K_Q,M_Q\) 有限常数。随后选目标依赖的 seed 尺度与更新轮数，并选足够细的最终 \(h\)，使 F1 或 F2 的某个有限 latent **精确达到 \(I_hQ\) 的顶点表**，同时
\(\|f_{D(z)}-Q\|_\infty=\|I_hQ-Q\|_\infty<\varepsilon/3\)。
三角不等式给出所述结论。关键逻辑顺序是先固定光滑 \(Q\)，再选择表示它所需的架构常数，最后细化规则网格；不能颠倒量词，不能从 Hatcher 的同伦直接声称 F1/F2 latent 可达。

反向包含：每个有限 latent 输出都是边界固定的 P1 同胚；在 \(\operatorname{Homeo}_{\partial}(\Omega)\) 这个相对空间内取闭包，上述密度等式成立。如果把闭包放在所有连续映射的裸 \(C^0\) 空间，homeomorphism 极限可退化为非 homeomorphism，因此不应写成那个更强且错误的等式。

## 5. 算法含义、非含义与下一关键问题

这是相较“只覆盖某类 \(C^{1,1}\) 同伦路径”的**严格扩张**：定性表示能力覆盖所有边界固定连续平面 homeomorphism；最后输出仍是单张固定规则网格 P1，计算是一系列显式局部正向运算，不需要一个随 \(V\) 增长的全局线性系统。

但这个证明不会给粗糙 \(F\) 任何统一收敛率。为精度 \(\varepsilon\) 选的 \(Q\) 可以有越来越小的最小 Jacobian 和越来越大的二阶导/时间速度；所需 seed、轮数和最终顶点数可能急剧增长。仅在**固定有界光滑同伦类**上，[F1/F2 定理](23_uniform_isotopy_multilevel_approximation.md)的常数才可统一，且每级一轮/周期、整体 \(O(V)\)。此处的全群稠密性不等于对任意目标实现 \(O(V)\) 的**统一 \(\varepsilon\)-复杂度**、不等于普适低维 latent、不等于 image-to-latent 可识别。

接下来真正重要的是定量化：给一个在图像配准中自然的目标紧类（例如双 Lipschitz 常数、局部尺度、离散曲率/形变裕量有界），证明固定 seed 与层数下的误差/复杂度界；同时测量不提供 teacher 同伦时，编码器能否接近这些理论可达的目标。全群稠密性本身不应被当作网络实际成功的终点。
