# Independent real-pair protocol review — 2026-09-29

Scope: `tools/digital_wsi_pair.py`, `tools/digital_wsi_pair_eval.py`, both focused test files, HistoReg sections of `EVIDENCE.md`, and PLAN §§4, 9, 11. Independent checker context; no production changes. **The reported original-file pixel TRE values are correct. This passes a one-pair development arithmetic check, not formal benchmark or leakage-safe generalization validation.**

## Independent recomputation

Read the original CSVs with Python's CSV reader; joined integer IDs directly; computed Euclidean distances with scalar `math.sqrt`, mean/median with `statistics`, and P95 by linear interpolation at sorted index `0.95*(77-1)`. No production evaluator/helper was called for these results. Independently obtained image centroids using binary-mask row/column sums and first moments, preserving the documented Pillow JPEG draft/thumbnail operations, threshold `<230`, and pixel-center resize conversion `(index+0.5)*original_size/thumbnail_size-0.5`.

| Map, CD68 moving → CD4 fixed | Translation xy (px) | det | Inside fixed / total | Mean TRE (px) | Median | P95 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| Identity | (0, 0) | 1 | 77/77 | 513.207232834 | 559.502010720 | 765.843298133 | 884.362651857 |
| Tissue centroid | (+107.829102234, −43.064511293) | 1 | 77/77 | 587.821596445 | 659.590984097 | 876.140635217 | 997.859762154 |

These reproduce the evidence table to its displayed precision. Independently measured centroids: CD68 `(3607.016405369, 4318.570356822)` from 290×384 thumbnail; CD4 `(3714.845507602, 4275.505845529)` from 278×384. The production estimator/evaluator were called only afterward for comparison and agree.

## Direction, coordinates, identity and isolation

- The [author README, example lines 84–99](https://raw.githubusercontent.com/CBICA/HistoReg/master/README.md) explicitly identifies CD68 as moving/source and CD4 as fixed/target. The [official ANHIR data description](https://anhir.grand-challenge.org/Data/) specifies ImageJ X,Y coordinates with top-left pixel origin and matched landmarks within sets; its current indexed content was checked because direct retrieval returned 403.
- Original files are RGB JPEGs, CD68 width×height 7470×9890 and CD4 7171×9916, without EXIF orientation metadata. CSV IDs are uniquely 1–77 on each side. All original points are inside their own image. `X` maps to width/column and `Y` to height/row; centroid code correctly reverses NumPy's row,column index order. No physical pixel spacing is established.
- The saved map is a **forward landmark transform**, moving→fixed. PLAN §4's image sampler uses fixed→moving: any later image warp must use the inverse affine. Current landmark TRE is consistent; there is no image-warp implementation here to validate.
- Both linear parts are identity, hence determinant exactly 1 and full-plane affine bijectivity. Neither implies source-rectangle coverage of the fixed rectangle or a nonrigid Q1 certificate. Visibility is a separate count and does not filter TRE.
- Estimators read only images and accept no landmark paths. The evaluator alone reads CSVs. This establishes code-path separation, not a blind test: the developers have now observed these labels/results, the example is public, and patient/block/physical-slide/set identity is unknown. Keep it development-only; do not select thresholds, architecture, stopping rules or hyperparameters on these TRE values and then call the same pair held out. No group-independent split can be verified.

## Deliberate failures exercised against the actual evaluator

The focused suite independently reran: **9 passed in 0.48 s**. Additional in-memory fixtures used the real image files and real CSV contents; only file input streams/transform JSON were substituted, leaving production parsing, validation, mapping and metrics intact.

| Deliberate perturbation | Observed outcome |
|---|---|
| Swap the two original CSV role arguments, retaining filenames | Rejected: moving landmark file role does not match moving image. |
| Append ID 1 a second time | Rejected: duplicate landmark ID 1. |
| Remove moving ID 77 | Rejected: moving and fixed landmark ID sets differ. |
| Swap X and Y in both CSVs | Rejected here: moving ID 38 outside 7470×9890 image. Bounds cannot detect every in-range xy swap. |
| Translate all predictions +20000 px in x | All 77 retained in TRE; inside count 0; mean 20251.667597896 px. No silent visibility exclusion. |
| Divide all CSV coordinates by two, preserve names/IDs and original image dimensions | **Accepted without warning**; all 77 inside; identity mean becomes 256.603616417 px, exactly half the correct value. |

## Findings requiring careful scope

1. **Input-scale/provenance gap:** bounds, matching IDs and filename suffixes cannot establish that CSV coordinates belong to the declared image resolution or specimen. The half-scale attack demonstrates this limitation. There is no observed error for the original author files, but formal evaluation needs ordinary dataset metadata that ties image/landmark scale and specimen/pair identity together, plus explicit conversion when resizing. Do not describe the present checks as detecting all wrong-frame or wrong-scale inputs. Renaming a wrong CSV can also defeat the filename role check.
2. **Unsupported wording:** `EVIDENCE.md` calls the JPEGs “level-zero files.” Their standalone JPEG format establishes only the resolution of the provided files, not their original scanner/WSI pyramid level. Use “provided JPEG resolution; original WSI level and physical spacing unknown.”
3. **Development visibility:** wording such as “held-out evaluation” must not imply a hidden test or independent specimen holdout for this pair. Label-free estimation is verified; absence of human/model-selection leakage in future iterations is not.

No numerical correction to the two current TRE rows is needed. Strong external baseline, full-resolution image export, physical-unit evaluation, official aggregation and grouped generalization remain untested. No commit or push was made by this checker.
