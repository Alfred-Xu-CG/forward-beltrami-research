# 从显式同伦类到全部内部紧支撑光滑微分同胚：逐目标逼近推论

## 先说明量词

[ F1 定理](23_uniform_isotopy_multilevel_approximation.md)和[ F2 定理](26_f2_uniform_isotopy_approximation.md)假设已给出边界恒等的一条 $C^{1,1}$ 空间、足够光滑时间同伦 $F_t$，并有统一 $m,L,K,M$ 界。这个假设不是随意丢掉的：任意两个 P1 同胚虽然有拓扑路径，路径未必自动具备该定理需要的导数界。下面用一项**原始文献中的光滑 disk isotopy 定理**扩大目标类，但不声称固定一套浅网络参数就覆盖该类。

令 $\operatorname{Diff}^{\infty}_c(\Omega)$ 为 $\Omega=[0,1]^2$ 内部紧支撑的 $C^\infty$ 保向微分同胚群：每个 $F:\Omega\to\Omega$ 是光滑一一到上映射，逆亦光滑，且存在外边界的一条开邻域，使 $F$ 在其中恒等。这里“紧支撑”是对 $F-\mathrm{id}$ 而言；它比仅在边界四条边上取恒等**更强**。

[Dinkelbach–Leeb, 2009, Lemma 2.4(ii)](https://msp.org/gt/2009/13-2/gt-v13-n2-p15-s.pdf) 明确陈述：闭二维 disk 上一个保向微分同胚，若在边界邻域恒等，则可由**支撑在 disk 内部**的 isotopy 连到单位映射；该论文把光滑 disk 的原始依据指向 Munkres, *Differentiable isotopies on the 2-sphere*, Michigan Math. J. 7 (1960), Theorem 1.3。这里直接采用已核对全文的前一篇引理，不猜测原始论文 DOI。对我们的方形，因为 $\operatorname{supp}(F-\mathrm{id})$ 紧含于其内部，可选一张光滑圆角闭 disk $D$ 满足
$\operatorname{supp}(F-\mathrm{id})\subset\operatorname{int}D\subset D\subset\operatorname{int}\Omega$。将该 disk 上的光滑 isotopy 在 $D$ 外延为恒等，得 $\Omega$ 上一条边界恒等的光滑 $F_t$，从 $F_0=\mathrm{id}$ 到 $F_1=F$。

**推论（逐目标，不是统一全类）。** 对每个 $F\in\operatorname{Diff}^{\infty}_c(\Omega)$，这条 $F_t$ 在紧致的 $[0,1]\times\Omega$ 上具有

\[
m_F:=\min_{t,x}\det DF_t(x)>0,\quad
L_F:=\max_{t,x}\|DF_t(x)\|_2<\infty,\quad
K_F:=\max_{t,x}\|D^2F_t(x)\|<\infty,\quad
M_F:=\max_{t,x}\|\partial_tF_t(x)\|_2<\infty.
\]

行列式严格正来自每个 $F_t$ 都是保向微分同胚，紧致性使正的最小值存在。把这些**依赖 $F$** 的常数代入 F1/F2 定理，选任何 $0\le\beta<m_F/4$，则对该 $F$ 存在一个有限的粗 seed 侧长与轮数，此后每级 F1 一轮或 F2 一周期，使最终固定规则三角网 P1 输出恰为 $I_hF$，并满足 $\|I_hF-F\|_\infty\le K_Fh^2$；各最终分辨率的 decoder 前向工作/显存渐近 $O(V_h)$，无全局大线性系统。输出为单张原网格 P1 同胚，而不是若干连续 P1 因子复合后采样。

这个结论不是“每个连续 homeomorphism 都有 $O(h^2)$ 误差”，因为非光滑函数不满足二阶误差界；也不是同一个 $H_0,N_0$ 对所有 $F$ 通用，因为可把光滑变形压缩到任意小的空间区域，使 $K_F$ 任意大、$m_F$ 任意小。它没有构造高效的**目标到 latent 编码器**：所用 isotopy 是存在性结果，直接反算 teacher 需知道 $F_t$。实际神经网络只读图像时，推断困难仍可主导误差，参见[1025² 受控训练](25_1025_image_to_fine_latent.md)。

“绝大部分 homeomorphism”若没有指定测度、拓扑或正则类没有数学意义。本文证明的是整个明确的无限维群 $\operatorname{Diff}^{\infty}_c(\Omega)$ 的**逐元素**逼近（解码架构常数依目标而异），不是关于所有边界固定 homeomorphism 的测度一声明，也没有证明用单一固定 seed 深度对这个无界群作统一近似。今后若要升格为一般连续平面同胚的 $C^0$ 稠密性，必须另证它们可在**相同边界条件**下由上述内部紧支撑光滑群近似；这项相对边界光滑化步骤不由这里的 disk isotopy 或我们的 P1 层定理自动给出。
