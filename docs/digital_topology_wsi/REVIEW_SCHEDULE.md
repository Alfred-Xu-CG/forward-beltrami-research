# Independent F1/F2 schedule review

Scope: `ScheduledQ1Pyramid`, inherited Q1 update/refinement semantics, four random and four lesion-fit 257² archives, and their benchmark/fit reports. No production code was changed. Verdict: **no blocking geometry defect found**, with important distinctions about allocated latents, per-proposal versus per-stage motion, and the saved numerical Jacobian floor.

## Geometry and coverage

The fine-stage mask uses global interior indices and `(row odd) OR (column odd)`, exactly the complement of retained even/even coarse vertices. Zeroing logits before each F2 pass gives zero displacement at those old vertices; patch-shared scaling does not move a zero proposal. Enumerating patch interiors independently for offsets `(0,0),(2,0),(0,2),(2,2)` found complete coverage: 225 seed vertices and 736/3,008/12,160/48,896 newly introduced vertices at 33/65/129/257. No old fine-stage vertex was eligible. F2 may revisit a vertex in 1, 2 or 4 passes, whereas F1 visits it once per stage.

Dyadic Q1 refinement retains corners, averages edge endpoints, and uses the **four-corner average** at the cell center. These values reproduce the parent bilinear polynomial on every child cell in exact arithmetic; this is not diagonal P1 refinement or map composition. The determinant is affine within a Q1 cell, so its positive corner bounds extend to child corners. Child source-cell area and unnormalized determinants both scale by one quarter, preserving a normalized Jacobian floor in exact arithmetic. F1 then protects its 12 incident corner constraints; F2 bounds all four corners along the complete quadratic update path. Neither proof establishes exact preservation after arbitrary floating-point rounding.

Independent float64 17→33 checks for `11/12/21/22` found finite latent gradients, exactly zero gradients at masked old-node positions, exact retained coarse values, and zero difference between zero-fine output and direct Q1 refinement. A separate non-affine-cell evaluation at four off-grid points matched its refined Q1 function exactly for the chosen dyadic data. These are implementation checks, separate from the preceding polynomial argument.

## Latent counts and proposal calibration

| Schedule | Allocated scalars, matching JSON | Structurally used scalar entries | Independently recomputed best lesion RMSE |
|---|---:|---:|---:|
| `11111` | 172,618 | 130,050 | .0005581114013 |
| `21111` | 173,968 | 130,482 | .0005664555172 |
| `22211` | 203,548 | 141,970 | .0008090665369 |
| `22222` | 690,472 | 341,426 | .0008131072401 |

The JSON count is correct as **allocated tensor/optimizer entries**, not effective deformation dimension. Fine-stage old-node entries and per-pass positions outside patch interiors are structurally ignored. Even the used-entry count is not a Jacobian-rank or intrinsic-expressivity claim; repeated passes can control the same vertex. Every final fixed-boundary table has only 130,050 interior coordinate scalars.

At the tested four-cell patch width, F1's maximum raw component is `2h*tanh(z)` and F2's is `.5*4h*tanh(z)`: both have supremum `2h` per component, or `2sqrt(2)h` in vector norm. Both schedules use safety fraction .75 and normalized floor .05. **The total stage motions are not equalized**, because F2 uses four sequential, possibly overlapping passes. Nor are accepted steps equal: F2 shares a scale across a patch. A custom patch width other than four changes this raw-amplitude match; no equal-amplitude claim should be generalized to that setting.

## Independent saved-output audit

I converted every stored binary32 residual coordinate to a common dyadic integer scale and computed the four oriented corner determinants with arbitrary-precision integers, independently of the production certificate. All **eight** archives have 262,144 strictly positive corners and the complete exact identity boundary. All four lesion-fit archives retain bit-identical teacher affine coefficients; their exact determinant is `4369753492216013/4503599627370496 > 0`. Random outputs are square-to-square Q1 homeomorphisms; fit outputs are factored Q1 homeomorphisms onto that affine parallelogram. Recomputing residual-target RMSE in float64 reproduces each fit report up to its float32 reduction rounding.

**The saved floor is not exactly .05.** Across the eight binary32 tables, the exact stored-coordinate minimum normalized determinant ranges from approximately `.04996744357` to `.04999655997`, slightly below .05 but comfortably positive. Thus the exact-arithmetic floor theorem must not be promoted to a strict saved-output `.05` certificate. This is not a fold or a failure of the verified positive-orientation claim. A separately rounded composite table would also require its own audit.

The benchmark measures the stated allocated-latent implementation, including ignored slots, not a compact parameterization. Forward/VJP cost excludes image encoding, dense image sampling and saved-file auditing. The fit is teacher-assisted, uses teacher-coordinate error without landmarks, and compares limited optimization budgets; its ranking does not establish universal F1 expressivity superiority or independent network inference. Timing values were inspected, not independently rerun on the remote GPU.

## Deliberate attacks

1. **Remove the fine-stage mask.** On a 17→33 F2/F2 fixture, put nonzero logits only at old even/even vertices. The real pyramid changed the masked output by exactly zero. Calling the same fine F2 layer without the mask moved retained coarse coordinates by up to `.00622925` unit. This can remain topology-safe while violating the hierarchical-update contract, so a corner-sign certificate alone would miss it. An erroneous AND mask would instead cover only 256 of the required 736 new vertices at side 33, silently omitting every new edge midpoint.
2. **Substitute diagonal P1 refinement.** For the positive-Q1 cell with corners `(0,0),(1,0),(1.25,1.25),(0,1)`, the correct bilinear center is `(.5625,.5625)`. The diagonal endpoint average is `(.625,.625)`. That substitution changes the represented map despite apparently valid endpoints; the current implementation returns the correct four-corner average.

These attacks changed only temporary in-memory test data, not production code or saved maps.

## Addendum: analytic stress test, high resolution, and repeated seeds

### Independent analytic derivation and center obstruction

For `T_a=(x+a sin(pi*x)sin(pi*y),y)`, `det DT_a=1+a*pi*cos(pi*x)*sin(pi*y) >= 1-|a|pi`. Each horizontal slice is strictly increasing with fixed endpoints for `|a|pi<1`, proving the continuum square-to-square homeomorphism directly. In the exact sampled Q1 map, the second coordinate is still y and the horizontal derivative is

`1 + a * (sin(pi*x[i+1])-sin(pi*x[i]))/h * ((1-t)*sin(pi*y[j])+t*sin(pi*y[j+1]))`.

The sine secant has magnitude at most pi, so this particular sampled Q1 target also retains the lower bound before rounding. This special argument must not be generalized to arbitrary samples of a continuum diffeomorphism. Independent arbitrary-precision integer corner tests of the two saved targets found positive normalized minima `.6858520508` and `.2460327148`, and their stored coordinates matched separately evaluated float32 target formulas exactly.

The **single-seed** center cap and `.115` error obstruction are valid for the declared seed 17, patch width 4 and identity affine. Independently enumerating the seed patch interiors gives center `(8,8)` selection counts **`[0,0,0,1]`**. The first three passes keep it on a patch perimeter; only the `(2,2)` offset's `[6,10]^2` patch selects it. Its component proposal is `.5*4/16*tanh(z)` for F2, or `2/16*tanh(z)` for F1; the safe scale is at most one. Subsequent levels freeze this old vertex. Thus its horizontal displacement is bounded by `.125` and its error against an exact `.24` target by at least `.115`, regardless of training duration. Strict `<.125` refers to ideal finite-logit tanh, not saturation in machine arithmetic; `<=.125` is the useful cap. The stored target center displacement is `.2400000095367`, so its analogous lower bound is `.1150000095367`.

This is **not** a `.115` lower bound on grid RMS error. Nor is the center's visit count typical: seed `(3,3)` is selected by all four F2 passes. Extra seed repetitions, another patch width, or a nonidentity learned affine invalidate this particular one-stage cap. The `.10` target lies below the center cap but this does not prove its complete exact representability.

Independently recomputed best strict-interior RMSEs for `11111/21111/22211/22222` were `.00359079865/.00227365396/.00209454437/.00229675945` at amplitude .10 and `.05138993149/.02059336334/.01994605365/.01861521409` at .24. Center errors matched every displayed entry. All eight fitted maps passed independent integer-sign and exact-boundary checks, and all stored affine factors were identity. The reported finite-budget reversal of the small-DHR-residual ranking is supported; it is not an equal-cost optimum comparison or a universal expressivity ranking. The historical step-100 snapshots and GPU timings were inspected as reports, not independently rerun.

### Full high-resolution saved-file check

I loaded all four 1025² archives and both D:-resident 4097² archives. This was **not sampling** and did not call the production certificate. In 64-cell-row chunks, I promoted the exact binary32 inputs to float64 and enclosed every subtraction, endpoint-product extremum and final determinant subtraction with outward `nextafter` bounds. Thus a strictly positive interval lower endpoint certifies the exact stored-coordinate orientation, under the standard IEEE arithmetic assumptions. Every 1025² map had **4,194,304/4,194,304 positive corner intervals**; each 4097² map had **67,108,864/67,108,864**. There were zero nonpositive or ambiguous intervals, all coordinates were finite, and every outer boundary node exactly matched the ordered identity square.

The minimum normalized determinant lower enclosures were `.04980838951/.04981353879/.04981445521/.04980744421` for the four 1025² maps and **`.04869341850/.04878562689`** for the 4097² all-F1/hybrid maps. The growing binary32 deviation below .05 is real, not a certificate failure: positivity is verified, but a strict saved `.05` floor is not. The 4097² shortfall is about 2.6% of the nominal floor. The documented distinction between three-repeat timing runs and separate one-repeat saved-output runs is correct; the original timing-only reports had no saved-output certificate. Neither large-grid result establishes image-conditioned training, accuracy, or two independent decoder families. I did not independently rerun remote GPU timings or peak-memory measurements.

### Additional deliberate failure fixtures and API finding

1. Moving **only** the identity 17² center horizontally by .24, bypassing the safe update, creates four nonpositive corner determinants with normalized minimum **-2.84**. A globally valid coherent target does not license a locally isolated large proposal.
2. Sampling the same sine formula at amplitude **.4**, outside its premise, produces **17,988 nonpositive stored corner determinants** and minimum normalized determinant **-.256591796875** at 257². This exposes misuse of the analytic argument without `|a|pi<1`; the generator's amplitude guard rejects that premise.

The inspected target generator initially accepted even `side` values while naming index `side//2` the center. At side 4 and amplitude .1 that node is `(2/3,2/3)` and its reported displacement is `.0749999881`, not the geometric center's .1. This was a metadata/API defect, **not** a defect in the odd-side 257² experiments. The builder restricted the generator to odd sides. I then independently called the amended function on `(side=4,a=.1)` and `(side=257,a=.4)` with file writing mocked: both raised `ValueError` before any write. The identified even-side issue is corrected.

### Newly added repeated-seed implementation

The inspected `seed_repeats` extension applies independent latent groups to the **latest same vertex table**, using one F1 sweep or four F2 patch passes per repetition, before any dyadic refinement. It neither resets to identity nor composes separately resampled maps. Consequently every accepted pass supplies the same exact-arithmetic positivity/boundary preconditions to the next, and the preceding one-seed center obstruction no longer applies. It does not follow that optimization reaches any desired center displacement.

I independently reconstructed the old single-pass forward loop: `seed_repeats=1` matched it exactly for `11/12/21/22`. Float64 17→33 checks with four F1 seed repetitions and three F2 repetitions retained positive independently computed corners after each seed sweep/patch pass (4 and 12 checked outputs), and every separate seed latent had finite, nonzero gradient L1 norm. Structurally selected counts were 3,272 and 4,118, matching direct enumeration. These are focused regression checks, not a certificate for all future rounded repeated-depth outputs; saved-output audits remain necessary.
