# Comparative decision matrix

| Route | Forward speed | Backward | Local topology | General expressivity | Overall decision |
|---|---|---|---|---|---|
| Compatibility projection | sparse but solve-heavy | local VJPs; global projection derivative incomplete | audited zero flips | projects arbitrary data only approximately | preprocessor, not final solver |
| Beurling/FFT | very fast on periodic uniform grids | fixed-point/implicit VJPs available | no universal hard guarantee | whole-plane operator; boundary and singular errors remain | fast operator component |
| BHF | direct quadrature currently expensive | real-pair autograd and replay work | safe steps and zero flips in tests | promising, but PV/global theorem open | strongest research candidate |
| Tutte/Floater | sparse solve; scalable | sparse implicit VJP | strongest under graph hypotheses | restricted positive graph | hard decoder/baseline |
| M-matrix | local assembly can be sparse | possible through positive solves | not automatic for wide nonplanar graphs | finite cone obstruction | diagnostic/decoder component |
| Barrier/KKT | nonlinear solve and HVP cost | implicit adjoint measured | strong finite-budget safeguards | broad target objectives, not direct arbitrary mu | correction/certification layer |
| Triangular/latent flows | fast evaluation | straightforward autodiff | positive structure | restricted latent family | neural parameterization |

## Recommended architecture for future work

The most defensible next system is a hybrid:

1. encode the input coefficient on a compatible or learned latent manifold;
2. run multilevel BHF on coarse meshes;
3. prolongate with a topology-safe triangular or certified positive operator;
4. apply a small number of fine BHF corrections;
5. use a determinant-safe smooth controller and an independent certificate;
6. use checkpoint/recompute or an implicit adjoint with explicit failure rules.

This is a research architecture, not a claim that the final solver already
exists. A future completion claim must include arbitrary/general-mesh tests,
memory scaling, an end-to-end derivative check, and a global injectivity proof
or a clearly delimited theorem covering the actual domain class.
