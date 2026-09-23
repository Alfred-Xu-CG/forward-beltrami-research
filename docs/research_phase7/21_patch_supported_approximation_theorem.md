# 不求全局逆的二维局部类：一 pass 精确采样与 \(O(h^2)\) 逼近率

## 定义与结论边界

这是 [F2 patch 原语](19_residual_patch_multilevel_decoder.md) 的一个**严格受限**但非单轴的逼近结论：任意二维 \(C^{1,1}\) 位移场都可以出现在每个 patch 内，不预先选定正弦/多项式基函数；位移必须在所选固定 patch 的边界为零且导数足够小。它不证明跨 patch seam 的任意映射四 pass 可达，更不证明所有边界固定同胚在常数深度内可逼近。

单位方形 \(\Omega=[0,1]^2\) 按边长 \(q=1/m\) 划成 \(m^2\) 个闭正方形 patch。取嵌套 SW–NE 规则三角网格，边长 \(h=1/(n-1)\)，要求 \(p=q/h\) 为整数，因此每个 patch 恰含 \(p\times p\) 个细方格。记 \(I_hu\) 为顶点样本 \(u(v)\) 的固定三角网格 P1 插值。定义 \(C^{1,1}\) 向量场类：

1. \(u:\Omega\to\mathbb R^2\) 是连续 \(C^1\)，每个 patch 内的导数是 Lipschitz，统一常数 \(K\)：\(\|Du(x)-Du(y)\|_2\le K\|x-y\|_2\)；
2. \(u=0\) 在**每一条** patch 边界上，因此相邻 patch 的内部场可以不同而胶合为连续边界固定场；
3. \(\sup_x\|Du(x)\|_2\le\kappa_0\)。

设 patch 安全比例 \(\sigma=0.85\)、精确算术归一化面积 floor 为 0.05，取任何 \(\kappa\) 满足

\[
\kappa_0+2Kh\le\kappa,
\qquad 2\kappa+\kappa^2<\sigma.
\]

例如当 \(\kappa_0<\sqrt{1.85}-1\approx0.3601\) 时，全部足够细的网格满足条件。则**一个**未错位 `SafePatchFieldPass(n,p)` 从 identity 出发、接受合适的有限 latent，输出恰好是 \(\mathrm{id}+I_hu\) 的顶点表；对每个固定网格面有 \(J_T>1-\sigma=0.15>0.05\)，边界恒等，因而是一张固定原网格 P1 同胚。而且连续目标 \(F=\mathrm{id}+u\) 与输出 \(f_h=\mathrm{id}+I_hu\) 的一致误差满足

\[
\|f_h-F\|_{L^\infty(\Omega;\mathbb R^2)}\le Kh^2.
\]

因此对这个**明确的 patch-supported 小导数类**，一 pass、\(O(n^2)\) 的正向层实现了统一 \(O(h^2)\) 逼近；不需要随顶点数增长的线性系统，也不需要让深度随 \(h^{-1}\) 增长。适用类是受限的，不能写成“绝大部分 homeomorphism”。

## 证明

**P1 导数界。** 在任一细直角三角形上，\(D(I_hu)\) 的两个列向量分别是某条水平、竖直长度 \(h\) 网格边上的 \(u\) 差商。选该小方格的西南角 \(a\)，两条相关边的每个点离 \(a\) 不超过 \(\sqrt2h\)。用导数沿边积分，差商列与 \(Du(a)\) 对应列之欧氏差至多 \(\sqrt2Kh\)。两列组成的矩阵谱范数不超过 Frobenius 范数，故

\[
\|D(I_hu)\|_2\le\|Du(a)\|_2+2Kh\le\kappa.
\]

**有限 teacher latent。** Patch 边界上 \(u=0\)，任一内部顶点到某个 patch 边界的距离不超过 \(q/2\)，沿相应线段积分给 \(\|u(v)\|_2\le\kappa_0 q/2<q/2\)。patch 原始位移尺度为 \(s_0=0.5ph=q/2\)。于是每坐标 \(|u_c(v)|/s_0<1\)，可设 \(z_{v,c}=\operatorname{atanh}(u_c(v)/s_0)\)，是有限数；seam 上的 latent 为零或不参与更新。每个 patch 原始提议正好等于 \(u\) 的顶点样本。

**安全比例不裁剪。** 从 identity 沿原始位移 \(I_hu\) 走公共比例 \(t\) 时，一个细面归一化有向面积为

\[
J_T(t)=\det(I_2+tD_T(I_hu))=1+t\,\mathrm{tr}(M_T)+t^2\det M_T,
\qquad M_T=D_T(I_hu).
\]

因为 \(\|M_T\|_2\le\kappa\)，有 \(|\mathrm{tr}M_T|\le2\kappa\)、\(|\det M_T|\le\kappa^2\)。原语使用的“不利系数和”
\(C_T=(-\mathrm{tr}M_T)_++(-\det M_T)_+\le2\kappa+\kappa^2<\sigma\)。在 identity 基图，\(J_T(0)=1\)，故每个面允许的 \(\sigma J_T(0)\) 大于 \(C_T\)，额外 0.05 floor 的预算为 0.95，也不成为约束。所有 patch 的比例都是 1，输出恰为目标顶点表。每个面终态 \(J_T(1)\ge1-C_T>1-\sigma\)，固定 patch seam 与整个方形边界；由正向三角面加一一对应边界的平面 PL 定理得全局 P1 同胚。连续图也满足 \(\|Du\|_2<1\)，由 \(\|F(x)-F(y)\|_2\ge(1-\kappa_0)\|x-y\|_2\) 和边界固定可知是同胚。

**逼近误差。** 对任一三角形内的 \(x=\sum_i\lambda_i v_i\)，以 \(x\) 作一阶 Taylor 展开各顶点：\(u(v_i)=u(x)+Du(x)(v_i-x)+r_i\)，\(\|r_i\|_2\le(K/2)\|v_i-x\|_2^2\)。加权线性项相消；每个顶点与 \(x\) 距离至多 \(\sqrt2h\)，故
\(\|I_hu(x)-u(x)\|_2\le\sum_i\lambda_i(K/2)(2h^2)=Kh^2\)。常数只用本规则网格的直径，不含 patch 数目或图像分辨率。

## 对当前实验的意义和限制

对于 \(q=1/16\)，\(n=1025\) 时 \(p=64\)，[对齐窗口](17_coherent_patch_vs_vertex_depth.md)的一 pass F2 float32 精确采样是一个较大导数、却仍偶然通过实际面预算的例子；其 \(\|Du\|\) 上界不满足此保守定理，**不能**把数值例子当成定理实例。[半 patch 错位](18_unaligned_patch_and_fp32_margin.md)违反 \(u=0\) 于 patch seam 的假设，一个 pass 的不可能性由固定 seam 立即给出；重叠补偿或 coarse-to-fine 才有用。[训练实验](19_residual_patch_multilevel_decoder.md)又显示，可表示性与从零优化、图像推断是不同问题。

上述结论在**精确算术**成立。`float32` 对极薄面可能翻号，现有 0.05 floor 的实际安全范围应以逐面检查/误差预算另立数值论证，不由连续逼近定理自动推出。另一方面，本定理的输入类虽然无限维（每 patch 可放任意受限的二维 \(C^{1,1}\) 场），但要求每条固定 seam 上零位移，离“覆盖一般边界固定同胚”仍很远；扩大到跨 seam 场且保持与分辨率无关的 pass 数，是未解决的主问题。
