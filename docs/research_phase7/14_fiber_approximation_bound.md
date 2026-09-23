# 正边增量层的一个分辨率无关逼近界：明确限定在纤维单调目标类

## 定理所处理的对象

固定 \(\Omega=[0,1]^2\)、SW–NE 网格间距 \(h=1/n\)。目标是**独立于解码器给定**的
\[
F(x,y)=(g(x,y),y),\qquad g(0,y)=0,\quad g(1,y)=1,\quad
g(x,0)=g(x,1)=x.
\]
假设 \(g\in C^{1,1}\)、\(m\le\partial_xg\le M\)、\(\operatorname{Lip}(\partial_xg)\le K\)，其中 \(0<\rho<m\le M<\infty\)。这类包含 [强剪切 \(G_{0.28}\)](12_shear_beyond_contraction.md)，但**不包含一般双分量平面同胚**。

对内部行 \(i=1,\ldots,n-1\) 和边 \(j=0,\ldots,n-1\)，取目标细边增量
\[
d_{ij}=g((j+1)h,ih)-g(jh,ih),\quad
u_{ij}=n d_{ij}-\rho,\quad
\ell_{ij}=\log u_{ij}-n^{-1}\sum_{k=0}^{n-1}\log u_{ik}.
\]
因为 \(u_{ij}\ge m-\rho>0\)，\(\ell\) 有定义。设
\(B=\log((M-\rho)/(m-\rho))\) 并选解码器跨度 \(L>B\)。每行 \(|\ell_{ij}|\le B<L\)，于是有限的精确 teacher logit
\[
z^*_{ij}=\operatorname{atanh}(\ell_{ij}/L)
\]
使 [正边增量层](13_monotone_fiber_forward_layer.md)在**精确实数算术**下输出逐顶点的 \(F(ih,jh)\)；其 P1 延拓正向且全局一一对应。证明是 softmax 的平移不变性：
\(\operatorname{softmax}_j(\ell_i)=u_{ij}/\sum_k u_{ik}\)，而边界条件给 \(\sum_k u_{ik}=n(1-\rho)\)。故 \(\rho h+(1-\rho)q_{ij}=d_{ij}\)。这个表示结论使用**细网格全自由度**，不应误称为低维压缩。

## 解码器稳定性：常数不随最终控制网格增大

给同一行两组 logit \(z,z'\)，令 \(s_j=L\tanh z_j\)、\(p_j=\operatorname{softmax}(s)_j\)，\(P_r=\sum_{j<r}p_j\)。直接微分得
\[
\frac{\partial P_r}{\partial s_k}=p_k(\mathbf1_{k<r}-P_r),\qquad
\sum_k\left|\frac{\partial P_r}{\partial s_k}\right|
=2P_r(1-P_r)\le\frac12.
\]
均值定理和 \(\tanh\) 的 1-Lipschitz 性因此给出**与 \(n\) 无关**的解码器坐标稳定界
\[
\|D_h(z)-D_h(z')\|_{\ell^\infty(\text{顶点坐标})}
\le\frac{(1-\rho)L}{2}\|z-z'\|_{\ell^\infty}.
\]
这里右边是 latent 场的最大单坐标差；左边第二像坐标恒同，所以也是顶点二维欧氏最大差。P1 重心插值进一步把同一界扩展到整个 \(\Omega\) 的函数 \(L^\infty\) 范数。对任意 finite \(z,z'\)，两端输出都安全；该稳定界不是额外的拓扑假设。

## 限定光滑类的粗 log-density 逼近

把 \(u_{ij}\) 看作 \(\partial_xg\) 在长度 \(h\) 的水平边上平均后减 \(\rho\)。在行坐标 \(y=ih\)、边中点 \(x=(j+1/2)h\) 上，它具有一个连续扩展 \(u_h(x,y)\)，其值范围仍在 \([m-\rho,M-\rho]\)，且 Lipschitz 常数不超过 \(K\)（用移动平均的平移估计）。逐行减去 \(\log u_h\) 的离散边平均得到 \(\ell_h(x,y)\)，其 x/y 方向 Lipschitz 常数分别至多 \(K/(m-\rho)\)、\(2K/(m-\rho)\)。只要 \(L>B\)，\(z_h^*=\operatorname{atanh}(\ell_h/L)\) 的 Lipschitz 常数有仅依赖 \(m,M,K,\rho,L\) 的上界，例如用 \(\ell^\infty\) 的坐标距离可取
\[
C_z=\frac{3K}{L(m-\rho)[1-(B/L)^2]}.
\]
这个公式故意取宽松上界。把该连续 \(z_h^*\) 在一个间距至多为 \(H\) 的较粗矩形 logit 网格采样，再做与代码相同的双线性升采样，得到 \(z_H\)。每个细点是距离至多 \(O(H)\) 的四个粗采样值的凸组合，故有 \(\|z_H-z_h^*\|_\infty\le C_I C_z H\)，其中 \(C_I\) 是仅由坐标范数与采样端点约定决定的常数，不依赖细网格 \(n\)。本实现的粗 logit 采样端点对齐细网格首尾**内部行**与首尾**边中点**；此处不从越界位置外推。

结合上一节，令 \(I_hF\) 为在原三角剖分上的目标顶点 P1 插值，就有
\[
\|D_h(z_H)-I_hF\|_{L^\infty}
\le\frac{(1-\rho)L}{2}C_I C_z H,
\qquad
\|D_h(z_H)-F\|_{L^\infty}
\le C_1H+C_2h^2.
\]
第二项来自 \(g\in C^{1,1}\) 在形状规则的固定三角形上的标准线性插值误差；\(C_1,C_2\) 只依赖目标类上界及三角形形状。若让粗 latent 网格间距 \(H\to0\)、细控制间距 \(h\to0\)，便在**这个明确的纤维单调 \(C^{1,1}\) 类**上取得 \(C^0\) 逼近，同时每一步都是一张原固定网格 P1 同胚。若有共同的 \(m,M,K\) 且 \(L\) 统一选取，这也是该受限类的统一误差界；没有“绝大部分任意 homeomorphism”结论。

## 为什么高频实验不推翻定理

[多尺度实测](13_monotone_fiber_forward_layer.md)中 high128 在 33–129 粗层出现严重点抽样混叠，因其 \(\operatorname{Lip}(\partial_xg)\) 随频率平方迅速增大，所以上述常数 \(C_1\) 及 \(C_z\) 很大；\(H\) 未小到能控制误差。面积平均低通减少混叠但过滤真细节，也不与定理矛盾。定理只说明适合目标平滑度的 \(H\) 足够细时可逼近，**不**宣称固定 33² latent 能无损表达 128 周期或图像网络会从纹理中读出它。

还有两个独立缺口：把纤维层与 F1/F2 合成后对**一般双分量目标类**的定量逼近率；以及在目标 \(J_T\) 很薄时，前向安全步数、梯度和显存的统一可训练复杂度。固定网格的目标依赖层数[存在定理](07_fixed_mesh_reachability.md)没有解决这两个问题。
