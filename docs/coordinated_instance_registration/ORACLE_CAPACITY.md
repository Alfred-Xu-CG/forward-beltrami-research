# Can the implemented map class fit the persistent 7-to-8 errors?

## 1. The question and what this experiment is NOT

The original image-driven registration leaves a worst landmark error of about
53 pixels on the known MIIT 7-to-8 section pair. Every target individually lies
inside the output affine polygon, but that does not imply all targets can be
realized simultaneously by our fixed-grid, fixed-boundary map class.

This experiment asks that narrower representational question. It EXPLICITLY
reads manual annotations and is a **label-oracle capacity diagnostic**. It is
not image-only registration, a learned encoder, a held-out test, an inference
score, an initializer, or a training teacher. No oracle output is fed back into
the production registrations. The fit is an additional suffix after the
already computed original registration, not a fresh 300-step registration.

## 2. Inputs, variables, and output map

Let Omega=[0,1]^2. There are N=257 vertices per axis, at
X_ij=(j/256,i/256), including the four corners. The output is a table
Y_ij in R^2 on this SAME source mesh: 66,049 control vertices and 131,072
triangles. Fix each cell's top-left to bottom-right diagonal. Let f_Y be the
continuous piecewise-affine interpolant of Y, denoted P1-ac.

The incoming table Y^0 is the saved ORIGINAL image-only analytic registration,
not identity and not the retired shared-affine feature variant. Its frozen
image-derived affine A in R^(2-by-2) and b in R^2 are preserved as their original
float32 binary values, faithfully promoted to float64 for geometry. The complete
map is F_Y(q)=A f_Y(q)+b. All residual boundary vertices stay exactly at X_ij.

The source images are the published moving section 7 and fixed section 8,
with their original saved 512-canvas layouts. CSV x,y coordinates are interpreted
as zero-based native pixel centers, an explicit convention whose publisher
origin is not independently verified. For native coordinate x along an image
axis, with native size W, actual resized size R and canvas padding p, use

    q=((x+.5)*(R/W)+p)/512.

Apply this separately on x and y using their actual resize factors. There are
124 nominal IDs. A (+inf,+inf) annotation marks absence; the fixed annotation-only
intersection has exactly 107 available IDs, identical to earlier scoring.
All 107 are fitted and scored with equal weight. No large-error point is dropped.
Call their fixed-canvas locations q_k and moving-canvas targets y_k.

## 3. Exact fitting problem and forward decoder

The fitting objective is ONLY

\[
J(Y)=\frac1{107}\sum_{k=1}^{107}\|512\,[A f_Y(q_k)+b-y_k]\|_2^2.
\]

It has units of squared 512-canvas pixels. The root-mean-square error is sqrt(J);
it is not the mean Euclidean error. Images, machine points and distortion priors
do not enter J. They are evaluated separately AFTER fitting as diagnostics.

Require the original exact identity boundary and every cell's four normalized
corner determinants q_j/q_ref>.001, with q_ref=1/256^2. These are twice signed
triangle areas. The four determinants are

    det(b-a,d-a), det(b-a,c-b), det(c-d,c-b), det(c-d,d-a),

where a,b,c,d denote that cell's mapped top-left, top-right, bottom-right and
bottom-left vertices, distinct from the affine translation b above.
These constraints imply a global P1 homeomorphism with the declared boundary.

At one stage freeze its accepted anchor Y. The learnable variable is an interior
scalar coefficient field z_l at level l in {17,33,65,129,257}. Pad it with exact
zero boundary values and interpolate the RAW scalar proposal r=P_l z_l to the
257-square control grid. Interpolating a proposal is not resampling a previously
certified map. For direction e=(1,0) or (0,1), the candidate is

\[
Y'_i=Y_i+\sigma r_i e,\qquad
\sigma=\min(1,.95/g_Y(r)),
\quad g_Y(r)=\max(0,\max_j[-(C_Yr)_j/s_j]).
\]

Here s_j=q_j(Y)/q_ref-.001>0, and C_Y is the exact local linear determinant-change
operator for single-direction motion. Since det(e,e)=0, candidate determinant
slacks are exactly s+C_Y(sigma*r). No global mesh matrix is solved. The derivative
of the active sigma is INCLUDED in autograd; it is not detached.

Use x then y at each of the five levels: ten stages, 30 Adam gradients per stage.
The level-dependent learning rate is .004*16/(l-1), the original physical-rate
recipe. Each stage starts its coefficients at zero and holds its anchor fixed;
trial candidates are not compounded. Evaluate 31 trials to include the final
optimizer step. Select the best finite, actual-strictly-valid J candidate,
including the incoming anchor. There is no stopping or budget expansion based
on anatomical error. Final budget: 300 gradients, 310 decoder trials,
322 J evaluations (initial + ten anchors + trials + final), zero failed trials.

The gradient of J uses a fixed P1 point sampler and the existing safe decoder.
No Jacobian matrix of the full optimizer history is formed or retained.
Local decoder differentiation is distinct from end-to-end encoder training.

## 4. Predeclared witness criterion and actual result

A witness succeeds only if its ACTUAL saved map is valid and EVERY available
center has Euclidean error <=1 canvas pixel. Failure after the one fixed budget
would be inconclusive, not an impossibility theorem.

| Measurement | Incoming original registration | Label-oracle endpoint |
|---|---:|---:|
| Mean Euclidean error, canvas pixels | 3.265901 | .015168 |
| Pair p90 error, canvas pixels | 4.932224 | .021693 |
| Maximum error, canvas pixels | 52.713426 | .026773 |
| Root-mean-square error, canvas pixels | 6.689510 | .016373 |
| J, squared canvas pixels | 44.749548 | .000268082 |

The old worst center Pt121 improves from 52.713426 to .014186 canvas pixels.
Every one of the 107 points meets the criterion. Final minimum normalized
corner determinant is .013077228, strictly above .001; reference and boundary
tables and the original affine are exact. During the recorded fit, 131 trials
activate geometric clipping. This is not a fit performed by ignoring safety.

The saved table is therefore a concrete simultaneous SPARSE capacity witness
for this implemented mesh, boundary and floor. It does not determine true dense
anatomical correspondence between the 107 centers, and does not prove general
capacity on other specimens or convergence of image-only optimization.

## 5. What does the ORIGINAL image objective think of this map?

Evaluate the ORIGINAL full-512 objective only at the incoming and final tables:

    E1 = I + 3 R + .0001 S + .1 M + O.

I is the original transported eight-channel self-similarity image term with its
unchanged fixed foreground mask and denominator. R is cumulative P1 ARAP strain;
S is the original four-corner symmetric-Dirichlet penalty; M is the frozen
image-machine-correspondence robust term using the original 159 raw points,
eligibility/confidence and eight-canvas-pixel scale; O penalizes original moving
image out-of-bounds coordinates. None of these terms is used to fit the oracle.

| Term | Incoming value | Oracle value | Weighted change in E1 |
|---|---:|---:|---:|
| Image I | .368334800 | .375475556 | +.007140756 |
| ARAP R, weight 3 | .002629763 | .063753376 | +.183370839 |
| Shape S, weight .0001 | .022493091 | .862538629 | +.000084005 |
| Machine points M, weight .1 | .026702288 | .224072997 | +.019737071 |
| OOB O | .000730140 | .000729909 | -.000000230 |
| Total E1 | .379626708 | .589959148 | +.210332440 |

About 87.18% of this particular E1 increase is weighted ARAP, 9.38% is weighted
machine evidence and 3.40% is image evidence. The oracle map's residual
vertex-displacement RMS grows from 1.96845 to 6.96629 canvas pixels and its
maximum from 6.88177 to 55.36578 pixels.

This does NOT prove the prior is unreasonable or every accurate map must have
higher E1. A point-only fit may create costly deformation in unobserved regions;
other accurate maps may be smoother. It does show that mere sparse attainability
is insufficient and that this achieved witness is disfavored by the retained
production objective. No regularization-weight sweep follows automatically.

## 6. Runtime, memory, independent checks, and artifacts

The one freshly idle AI RTX A6000 GPU-6 call takes 2.89790 seconds for the fit
interval and 3.74883 seconds end-to-end. Fit timing includes stage construction,
trials, gradients and actual numerical geometry checks; it excludes final
oracle/E1/distortion evaluation and export. End-to-end includes loading/setup
and output checks, but not the already completed initial registration.
Allocated fit peak is 115.065 MiB, including 41.158 MiB resident at fit start
(the full-512 original evidence is already loaded). This is not whole-process
RAM, driver memory, encoder-training memory or a matched speed comparison.

Before real execution, focused author tests and an independent off-node P1
gradient calculation check physical units, original-map start, stage semantics
and counts. The latter's maximum gradient discrepancy is 4.73e-12. Independently
reconstructed original E1 matches the legacy implementation bitwise on synthetic
identity/nonidentity examples.

For the actual result, a separate literal CSV/native half-pixel conversion and
generic 3-by-3 triangle-affine solve reproduce all 107 point errors to 9.98e-13
pixels. All 262,144 saved corner signs, original affine and boundary values are
independently checked. Independent NumPy/PIL descriptor sampling, machine point
evaluation, SVD-based ARAP and explicit matrix-inverse shape evaluation reproduce
nonimage terms to 7e-17 and image costs to 2.98e-8. All ten recorded stage minima
are consistent with anchor-inclusive selection. Intermediate trial arrays were
not archived; their claims are source/record consistency, NOT independent binary
certification of every intermediate table.

Artifacts are the separate files
`outputs/coordinated_instance_registration/miit_7_to_8_label_oracle_capacity_t17.npz`
and its `.json`, generated by `tools/coordinated_miit_capacity_witness.py`.
They are explicitly marked label_oracle and landmarks_used. They are absent
from production prediction manifests and must never be scored as image-only
registration or used as production teachers.

A separately predeclared matched analytic-versus-F2 oracle comparison will ask
whether the coordinated mechanism reaches this known sparse motion more
efficiently. That is a new controlled geometry-efficiency question, not another
budget/weight search and not a production accuracy claim.
