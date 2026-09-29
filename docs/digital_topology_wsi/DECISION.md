# Phase VIII decision: recurrent F1/F2 homeomorphic image maps

This is the T+22-hour independent scientific adjudication for the 24-hour window started at `2026-09-28T18:04:58Z` in `START_TIME.json`. The window's elapsed-time closure is recorded separately in `END_TIME.json` when reached. The decision is about the tested digital-topology WSI system, not a claim that the research objective is solved. Full derivations, configurations and independent checks are in `TOPOLOGY.md`, `EXPERIMENTS.md` and `REVIEW.md`; the figures below retain their own experimental scope.

## 1. What the layer represents and why repeated updates are necessary

Let the fixed source rectangle have an $n\times n$ vertex grid $x_i$, spacing $h=1/(n-1)$, fixed cell connectivity and one fixed SW–NE diagonal per cell. The decoder stores **one** current image-vertex table $Y^{(k)}=(Y_i^{(k)})_i$. Its fixed-diagonal piecewise-affine (P1) interpretation is $F_k(x_i)=Y_i^{(k)}$; a separate bilinear (Q1) interpretation uses the same four cell vertices. The source indexing stays regular, but after one update the image mesh need not be geometrically regular. A single coarse seed update or one F1/F2 pass should not be treated as sufficient for a complex registration.

For an interior vertex $i$, form the **current** centered image edges

\[
E_i(Y)=\left[\frac{Y_{i+e_x}-Y_{i-e_x}}2,\frac{Y_{i+e_y}-Y_{i-e_y}}2\right],
\qquad r_i^{F1}=\alpha E_i(Y)\tanh z_i .
\]

At the identity, $E_i=hI$ and this reduces to the old $\alpha h\tanh z_i$. On a locally affine deformed mesh $Y(x)\simeq Ax+b$, it is approximately $\alpha hA\tanh z_i$: the proposal follows the *current* scale, axes and shear. The implemented dense F1 experiments use $\alpha=8$. F1 divides interior vertices into four source-index parity colors. Each color's vertices affect disjoint cells; after **each color and each round**, geometry and safety constraints are recomputed from the just-updated table. The image head predicts fresh logits once per full F1 round, while the same logits are used through its four colors. At an F2 offset the image head is evaluated again on that offset's current state.

For F1, each active vertex has 12 incident Q1 corner determinants $q_c(d)=q_c(0)+a_c^\top d$. With safety fraction $0<\gamma<1$, optional absolute floor $\beta_c$, and positive margin

\[
m_c=\min\{\gamma q_c(0),q_c(0)-\beta_c\},\quad
M_i=\max\left(0,\max_c\frac{-a_c^\top r_i^{F1}}{m_c}\right),\quad
d_i=\frac{r_i^{F1}}{1+M_i},
\]

every adverse exact-arithmetic corner loss is strictly less than its budget. If any required margin is nonpositive, the implemented branch freezes that local vertex; the nominal floor cannot be retroactively repaired. The implemented `minimum_jacobian=.05` enters these budgets through the **reference-cell area** $h^2$. Thus the raw movement has been adapted to current edges, but a fixed reference-normalized floor remains and can cause local freezes. A fixed-$h$ raw proposal with the same safety law is also topologically safe in exact arithmetic; current edges are geometrically covariant in the ideal no-floor regime, not experimentally proven more accurate.

F2 simultaneously proposes movement for the interior vertices of a width-$p$ patch using $r_i^{F2}=\alpha p E_i(Y)\tanh z_i$ (the tested dense patch has $p=8$, $\alpha=.5$). One **common** patch scale is bounded using every affected corner's quadratic path $q_c(t)=q_c(0)+L_ct+Q_ct^2$. Four staggered, nonoverlapping patch-offset passes make seam vertices eligible later; each pass sees the current table, and its perimeter vertices stay fixed within that pass. Applying independent F1 safety factors to simultaneously moving neighboring F2 vertices would be invalid. An alternating macro-round $Y^{k+1}=U_{F1}(U_{F2}(Y^k,z_{F2}^k),z_{F1}^k)$ is conditionally safe by induction, provided every subpass recomputes its own constraints and preserves the boundary. **A trained alternating F2→F1 stack at 257² was not run**; existing 257² learned F1 K16 and F2 K4 models are separate.

Repeated updates do not secretly reinterpolate compositions of maps defined on independent reference grids. If both $F_k$ and $F_{k+1}$ are valid P1 embeddings of the same source triangulation, define $G_k$ on the **deformed** triangulation $F_k(\mathcal T)$ by $G_k(Y_i^{(k)})=Y_i^{(k+1)}$. Barycentric coordinates on each corresponding triangle give the exact identity

\[
F_{k+1}=G_k\circ F_k,
\]

while the result stays P1 on the original indexed source triangulation. A standalone evaluation of $G_k$ or $F_k^{-1}$ at an arbitrary point would still require image-triangle location. Across dyadic resolutions, the implemented subdivision exactly preserves the old **Q1** function, not generally its coarse P1 function; the fine P1 table is separately valid when its four-corner/boundary conditions hold. At zero logits and strictly positive safety margins, the first derivative of a color update with respect to its free vertex logits is $\alpha E_i(Y)$; all such matrices are nonsingular under the incident positive-corner hypotheses. The inverse-function theorem makes a full F1 pass locally surjective onto nearby valid vertex tables, and finitely many such local neighborhoods cover a compact prescribed path inside the strictly Q1-positive fixed-boundary set. This **conditional free-latent** result does **not** give a bounded-depth approximation theorem, a claim about all P1 homeomorphisms, or a theorem that a shared CNN learns the needed logits.

## 2. Exact guarantee versus actual arithmetic

For each cell with image corners $a,b,c,d$ in source cyclic order, the Q1 map is $a+s(b-a)+t(d-a)+st(a-b+c-d)$. Its Jacobian determinant is affine in $(s,t)$, so strict positivity of **all four** corner determinants implies positivity throughout the cell. With one-to-one ordered fixed boundary and shared edge values, the exact-arithmetic Q1 map is a global homeomorphism of the rectangle; its fixed SW–NE P1 interpolation is also a global homeomorphism because its two triangle determinants are among those corner conditions. Positive determinants alone without the boundary/global hypotheses would not suffice. A valid P1 grid can fail the stronger Q1 corner test: on a fixed-boundary 3×3 example with the center shifted to $(1/4,1/4)$, both original-diagonal P1 triangles stay positive while one Q1 corner determinant is $-1/2$. Changing interpolation conventions without a new check is therefore impermissible.

F1, F2 and their properly sequenced mixture retain this theorem for finite ideal-arithmetic proposals from a valid starting map. The trained networks and saved maps use float32 coordinates, with float64 intermediate F1 geometry in the current-edge implementation. Rounding, denominator guards, a fixed absolute floor, output-affine materialization and unsupported extreme coordinate scales prevent an **all-finite-IEEE-input** theorem. The independent binary-value checker found strictly positive Q1 corners and exact ordered boundaries for the declared saved test outputs, including all **35,651,584** corners of 136 saved 257² round-end states and all **67,108,864** corners of a saved 4097² geometry-only map. These audits exclude unarchived within-color intermediate states and arbitrary future inputs. Among the 136 states, **16,489** corners were below the *nominal* `.05` normalized floor after float32 storage, although none was nonpositive. The 4097² map likewise had **3,168,114** below-floor corners. Therefore the supported stored-output observation is **strict positivity**, not exact satisfaction of `.05`.

The saved residual table followed by a positive-determinant affine is a homeomorphism **onto its affine image**. It need not cover the moving-image square, and separately rounding every composed affine vertex can invalidate a certificate for the factorized representation. The P1 image-loss gradient path, latent and parameter VJPs, fixed-boundary/export/readback and native-image tile consistency were exercised; nondifferentiable branch boundaries and frozen vertices remain. In the original-backend 8-case K16 replay, **43,329 of 8,323,200** local vertex/color attempts (`0.521%`) had no positive floor budget and froze. This is neither a whole-map fallback nor proof that all gradients vanish, but it is a repeated-depth capacity/gradient bottleneck. No full-map fallback was used as the principal guarantee.

## 3. What was trained and measured

The principal neural input is a pair of 512² grayscale pathology canvases after an *external initial-only* affine prewarp of the moving image. The width-16 encoder computes static features once. A frozen 17² F2 seed and 33²–257² new-vertex refinement supply a parent map; extra recurrent heads receive static features, fixed-image brightness, moving-image brightness sampled at the **current** internal-affine map, current displacement and round index. They output two logits per active interior vertex, not a target Beltrami field. The learned output is one 257² residual vertex table plus a positive internal affine and the stored external initial affine. The declared guarantee is its fixed-diagonal P1 map under the tested factorization, not a guarantee that it matches anatomy.

The 257² dense-head comparisons froze the earlier base and 17²/33²/65² recurrent heads, trained only the new shared dense head for 2,400 Adam steps on 102 ACROBAT training IDs, and inspected eight distinct but **previously opened** development pseudo-target IDs. These IDs cannot be repackaged as a new blind validation set. The pseudo-target is a saved image-only DeeperHistReg (DHR) field, which is neither ground-truth correspondence nor necessarily fold-free. Vertex pseudo-target RMSE is $\sqrt{N^{-1}\sum_i\|A(Y_i)-D_i\|_2^2}$ in normalized canvas units; the complete training-step time includes map construction and backward on batch four, whereas the forward-only figure excludes it. The timings exclude the external image-derived initial affine/prewarp, image I/O, saved-output certificate and native-scale export; they are **not** end-to-end WSI registration times. Reported peak is PyTorch CUDA **allocated** memory, not board occupancy.

| 257² learned dense head | 8-case actual / blank pseudo-target vector RMSE | Warmed forward | Complete train step | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| F1 K1, rate `.002` | `.008233 / .012427` | 59.7 ms | 72.5 ms | `.923` GB |
| F1 K16, rate `.002`, ordinary backward | `.008146 / .012474` | 118.6 ms | 361.1 ms | `7.124` GB |
| F1 K16, rate `.002`, round checkpointing | `.008162 / .012456` | not separately asserted here | 461.0 ms | `.995` GB |
| F1 K16, rate `.0002`, round checkpointing | `.008261 / .012411` | not separately asserted here | 488 ms | `.990` GB |
| F2 K1, rate `.002` | `.008269 / .012437` | 58.8 ms | 85.6 ms | `1.245` GB |

K16 is a **separately trained** model, not the K1 model unrolled more times. Its development pseudo-target advantage over K1 is small, while cost is much higher; reducing the K16 learning rate worsened this eight-case mean. Checkpointing traded roughly 86% lower measured K16 training allocation for roughly 28% higher complete-step time in its `.002` runs, not constant memory or identical trained weights. F2 K4 training was optimization-sensitive: its ordinary `.002` run deteriorated; `.0002` recovered much of the pseudo-target fit but did not beat F2 K1. The 136 saved K16 round-end states monotonically lowered the **mean pseudo-target error** `.008497989→.008261423`, but case 68 worsened at every round. None of this proves a useful image-registration benefit.

For scaling only, a 1025²-control-vertex, four-round **geometry-only** F2 current-edge run with round checkpointing measured 67 ms forward, `.384` s complete forward/VJP and `2.176` GB allocated peak; a 4097²-control-vertex, four-round **geometry-only** F1 checkpointed run measured `.829` s forward, `4.773` s forward/VJP and `12.55` GB peak on one timed repetition. Both saved maps had positive corners. Neither is a trained high-resolution image encoder. A separate fixed-diagonal P1 image-loss VJP on a saved 257² K16 model queried all 512² pixel centers and produced finite nonzero dense-head gradients; it did **not** constitute a new P1-trained registration model. Actual level-four original-TIFF RGB warps using the same saved P1 map were tile-size bitwise identical, but are not a whole-WSI out-of-core accuracy study.

## 4. Real-data decision and O/N/H interpretation

The local real-data evidence comprises one author HistoReg development pair and two repeatedly inspected BIRL/ANHIR sample pairs: kidney (69 joined landmarks) and lung lesion (78). The latter are 5%-scale JPEGs, not an official ANHIR test split or independent patient/block groups; no group-level confidence interval or official ANHIR rTRE can be inferred. The eight ACROBAT pseudo-target cases were also opened during architecture development. Landmark TRE below is the mean Euclidean error at original moving-JPEG pixel centers after fixed→moving P1 query, image-layout conversion and stored affine; lower is better. The fresh final-review rescoring deliberately omitted full DHR because this Windows interpreter lacks SimpleITK; prior DHR numbers retain their earlier protocol/provenance and are not silently merged with this rerun.

| Reused development pair | K16 actual-image P1 TRE | Same-model blank-image TRE | Initial affine TRE |
| --- | ---: | ---: | ---: |
| Kidney, 69 landmarks | **14.311984 px** | 15.929635 px | **13.917438 px** |
| Lesion, 78 landmarks | **9.606485 px** | **7.741336 px** | **8.532122 px** |

The **lesion** is the strongest counterexample to the inference “more safe/current-edge iterations imply better registration”: actual-image K16 is 24.1% worse than its own blank-input control and 12.6% worse than initial affine. Kidney improves over blank but remains 2.8% worse than affine. Earlier matched DHR development runs were also substantially more accurate, although their end-to-end cost and field topology differ. Thus the negative evidence is anatomical, not a failure of the saved output's homeomorphism audit.

`O` means per-pair image-only latent optimization under the safe decoder. Its 100-step global-NCC pilot improved the image objective but produced poor landmarks, so the objective is not a reliable anatomy selector here. `N` means amortized image-to-latent prediction; it is fast and can make image-dependent 257² safe maps, but neither the small-group transfer experiments nor the recurrent ACROBAT pseudo-target runs meet the predeclared independent-group benefit test. `H` in the available strong teacher-fit experiment means a safe affine-plus-residual fit *to a DHR map*; it approaches DHR landmarks on development pairs, demonstrating representation capacity, but requires generating the expensive teacher and is **not** a measured fast network-plus-budgeted-refinement inference method. These modes do not establish an accurate fast full registration system.

## 5. Independent verdict against the declared gates

| Gate | Verdict | Exact meaning |
| --- | --- | --- |
| G1 safe geometry | **PASS, scoped** | Conditional exact-arithmetic F1/F2 theorem, independent proof review, saved binary32 strict signs/boundary, export and gradient checks pass for declared output families. Not universal IEEE or `.05`-floor certification. |
| G2 benefit of learning | **NOT TESTED** | The required untouched specimen-group, same-geometry O/N/H comparison with prespecified accuracy/CI or cost criterion was not available. Development results are mixed or negative, not a formal G2 pass or formal rejection of every architecture. |
| G3 competitive real pathology | **FAIL** | Official-compatible leakage-safe group evaluation and adequate matched strong-baseline coverage are absent, triggering the PLAN's application-failure rule; inspected anatomy is not competitive. |
| G4 SOTA | **NOT TESTED** | No qualifying official split, hidden-test result, current strong-comparator ranking or statistical evidence. No SOTA claim is authorized. |

The independent final adjudicator recomputed 35,651,584 saved round-end corner values by a separate NumPy route, checked the corrected current-edge determinant identity, reran 28 focused tests, confirmed the actual F1-K16 and separate F2-K4 manifests, and agreed with the lesion/kidney P1 landmark figures. The coordinator's wider final focused suite passed **91 tests**. The corrected identity is $4\det E=[e,s]+[s,w]+[w,n]+[n,e]$, where $e,s,w,n$ are current right/down/left/up vectors from the active vertex. A previously written reverse cyclic order had the wrong sign on the identity grid; the source implementation's right-minus-left/down-minus-up vectors were correct. The independent review and repair are recorded in `REVIEW.md` and the targeted regression test.

## 6. Best present architecture and only three next tasks

The best **measured economical learned dense-head reference** is F1 K1, not K16 or F2 K4. A plausible system architecture is a strong image-only initial correspondence/affine stage followed by one certified residual vertex table and a *budgeted* number of current-state F1/F2 updates; this is a design direction, **not an accomplished competitive system**. More rounds are mathematically admissible and may increase free-latent coverage, but current image heads, losses and data have not translated that capacity into reliable anatomy. The current-edge basis fixes the specific fixed-$h$-on-irregular-mesh mismatch; it does not solve matching, conditioning, fixed-boundary limitations or numerical floor freezes by itself.

1. Obtain genuinely independent specimen groups and reserve untouched evaluation groups before tuning; record grouping, landmark direction and official-compatible metrics. Full ANHIR access is the first candidate, conditional on its provenance and terms audit.
2. Improve correspondence/initialization and run a matched, same-geometry O/N/H comparison with blank and affine controls, actual end-to-end time/memory, failure counts and group-level confidence intervals. Do not select by the reused two BIRL landmark sets.
3. Only then train an **actual** mixed F2→F1 recurrent 257² stack against F1 K1 and F2 K1 at matched compute, measuring anatomy, all-corner signs, local freezes, forward/VJP time and memory. Test whether relaxing the fixed absolute floor or adapting proposal conditioning helps without weakening the certified topology.

The scientific result of this window is a working and independently audited repeated **representation**, plus a strong negative application finding—not a solved fast high-resolution pathology registration network.
