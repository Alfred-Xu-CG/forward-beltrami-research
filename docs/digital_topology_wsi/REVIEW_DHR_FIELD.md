# Independent review of the saved DeeperHistReg development field

## Scope and provenance

This review reads the already saved `D:\QC_optimization_data\digital_topology_wsi\DHR_CD68_CD4_development_clean\HistoReg_CD68_to_CD4_development\Results_Final\displacement_field.mha`, its `postprocessing_params.json`, the run's `config.json` and `runtime.json`, and the 77 paired public HistoReg example landmarks. It did **not** rerun registration, change DeeperHistReg's image-only estimator, or use landmarks to choose its configuration. The landmarks are diagnostic labels for one development pair, not a blind or group-held-out test. The supplied JPEG resolution is known; original scanner pyramid level and physical pixel spacing are not.

The field is finite float32 `(component, row, col)=(2,991,747)`. The run used DeeperHistReg PyPI 1.0.1 with the reduced CPU configuration recorded in `BASELINE.md`; this evaluation does not establish published-setting performance.

## Coordinate convention, independently traced

DeeperHistReg `dhr_utils/warping.py` constructs the identity grid with `affine_grid(..., align_corners=False)` and samples the moving/source tensor with `grid_sample(source, identity + displacement, align_corners=False)`. Its `dhr_utils/utils.py::tc_df_to_np_df` converts normalized displacement to pixel displacement by multiplying x by `747/2` and y by `991/2` before the MHA saver stores the x and y planes. Therefore, on the saved padded canvas, for each fixed/output point `q=(x,y)`, the moving/source sampling coordinate is

```text
B(q) = q + (u_x(q), u_y(q)).
```

The saved map is fixed→moving (backward for image sampling). Forward moving→fixed landmark evaluation must solve `B(q)=p_moving_canvas`; adding `u` at a moving point would use the field in the wrong direction. The library's `transform_landmarks` routine is not an authoritative forward inverse of this saved field, so it was not used.

The `SimpleLoader` reduces images with PyTorch bilinear interpolation at scale factor `0.1`, `recompute_scale_factor=False`, `align_corners=False`. In continuous pixel-center coordinates, an original supplied-JPEG point `(x,y)` enters the canvas as `(0.1(x+0.5)-0.5, 0.1(y+0.5)-0.5)` plus left/top padding. The actual pads are `[row before,row after],[column before,column after]`: moving `[[1,1],[0,0]]`, fixed `[[0,0],[15,15]]`. Thus moving `(x,y)` enters as `(0.1(x+0.5)-0.5, 0.1(y+0.5)+0.5)`. The inverse fixed conversion is `( (q_x-15+0.5)/0.1-0.5, (q_y+0.5)/0.1-0.5 )`. The extra preprocessing ratio is recorded as 1. The rounded image sizes and pads yield the declared 747×991 canvas: moving 747×989 plus two y pixels; fixed 717×991 plus 30 x pixels.

This is an explicitly declared continuous Q1 interpolation of the **saved** field between its sample nodes. It is consistent with the saved tensor's backward sampling values and pixel-center convention. It does not claim that DeeperHistReg's unsaved full-resolution deformation or some other field interpolation has been measured.

## Inversion and landmark diagnostic

For each moving point, the evaluator examines every saved-grid cell whose four mapped vertices enclose the point in both coordinate bounds. In each candidate cell it solves the bilinear Q1 inverse quadratic, keeps all in-cell roots that reconstruct the moving point within `1e-6` canvas pixels, and deduplicates shared-edge roots. An affine/bilinear unit fixture, a two-root folded fixture, and an outside-domain fixture exercise the implementation. No initial guess from the fixed label is used. A point with zero or multiple roots would be reported as such and would suppress an aggregate 77-point TRE, rather than silently selecting a root.

| Saved-field diagnostic | Result |
|---|---:|
| Moving landmarks with unique inverse | 77/77 |
| Zero-root / multiple-root points | 0 / 0 |
| Forward predictions inside fixed JPEG | 77/77 |
| Largest inverse reconstruction residual | `1.61e-13` canvas pixels |
| Smallest local Q1 determinant at the 77 inverse points | `0.482669` |
| Mean TRE | `14.2325` supplied-JPEG pixels |
| Median TRE | `11.7720` supplied-JPEG pixels |
| 95th percentile TRE | `35.2819` supplied-JPEG pixels |
| Maximum TRE | `65.1085` supplied-JPEG pixels |

These are defensible *saved-field, one-pair development diagnostics* because every labeled point has one root, the inverse residual is tiny, and no point is omitted. Root uniqueness at 77 points does not establish global injectivity. Pixel-center scaling is derived from the actual resampler; using the simpler `0.1x` landmark convention would be a different evaluation convention. No micrometer TRE, official ANHIR rTRE, or group-generalization claim follows.

## Complete four-corner Q1 audit

Let `F(x,y)=(x+u_x,y+u_y)` on the saved 747×991 grid, and bilinearly interpolate each of its 738,540 cells. With `a=F10-F00`, `b=F01-F00`, `c=F11-F10-F01+F00`, its Jacobian determinant is `det(a+c t, b+c s)` for local horizontal/vertical coordinates `(s,t)`. The table counts determinants `<=0` at each corner; a cell enters the union once if any corner fails.

| Corner | Nonpositive cells | Minimum determinant |
|---|---:|---:|
| Top-left | 7,003 | −0.476811 |
| Top-right | 7,004 | −0.519980 |
| Bottom-left | 7,004 | −0.620162 |
| Bottom-right | 7,005 | −0.614719 |
| Any corner, union | **7,108** | — |

The prior `BASELINE.md` one-corner count of 7,003 and minimum −0.4768 reproduce exactly as the top-left row. The extra corners identify 105 more distinct failing cells. Since a bilinear cell's determinant is affine in `(s,t)`, its minimum occurs at a corner; all-four-corner positivity is the exact local Q1 criterion on this grid. Here the saved field fails that criterion. It cannot be presented as a guaranteed homeomorphism or as satisfying G1, even though its landmark TRE is low on this development pair.

## Reproduction and limits

`tools/digital_dhr_field_eval.py` is a separate read-only evaluator; `tests/test_digital_dhr_field_eval.py` returned `5 passed`. Full per-landmark roots, residuals, local determinants and TRE are saved as `D:\QC_optimization_data\digital_topology_wsi\DHR_CD68_CD4_development_clean\field_evaluation.json`. The command uses the original two JPEGs, two CSVs, saved MHA, postprocessing metadata and config; it does not call the registration pipeline.

The result tests the sampled field with a declared continuous interpolation. It does not validate a full-resolution warped output, mask overlap, physical-unit error, an official hidden test, or a high-resolution strong DeeperHistReg configuration. The current pair's patient/block/physical-slide identity remains unknown. The good diagnostic TRE must not be used to tune a method and then relabeled as held-out evidence.
