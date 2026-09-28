# Real pathology development baseline: DeeperHistReg

## Research question

Can a published WSI registration system run image-only on the author-provided HistoReg CD68 moving → CD4 fixed pair, producing a saved deformation for later same-pair comparison?

**Answer: yes, at a reduced development scale.** This is one pair and one configuration, not an official ANHIR score or a claim about the method's best accuracy. Landmarks were never supplied to the registration process or used to select the configuration.

## Input and software

| Item | Actual setting |
| --- | --- |
| Moving/source | `D:\QC_optimization_data\digital_topology_wsi\HistoReg_CD68_CD4\Images_CD68.jpg`, RGB JPEG, 7470 × 9890 pixels (width × height) |
| Fixed/target | `D:\QC_optimization_data\digital_topology_wsi\HistoReg_CD68_CD4\Images_CD4.jpg`, RGB JPEG, 7171 × 9916 pixels |
| Pair provenance | [HistoReg author example](https://github.com/CBICA/HistoReg); see `EVIDENCE.md` for terms and grouping limits |
| Baseline | [DeeperHistReg](https://github.com/MWod/DeeperHistReg) PyPI 1.0.1; inspected source checkout `42e7c9ddedb5932fbcbdf598fbc9b3a47baa47b6` |
| Runtime | Windows 10, CPU, 4 PyTorch threads; isolated D: Python 3.12.4 environment with PyTorch 2.5.1+cpu, NumPy 1.26.4, OpenCV 4.10.0, SimpleITK 2.5.6 |
| Terms | DeeperHistReg is CC-BY-SA; bundled SuperPoint/SuperGlue weights are for noncommercial research use. HistoReg data terms are recorded in `EVIDENCE.md`. |

VALIS was assessed first for practical installation. Its [installation instructions](https://valis.readthedocs.io/en/stable/installation.html) require Java/Maven and libvips system packages; this Windows host did not have Java. DeeperHistReg's PyPI wheel bundled its feature-model weights, so it was the first attempted executable baseline.

## Configuration and run

The run used DeeperHistReg's `default_initial_nonrigid_fast()` with its SuperPoint/SuperGlue multifeature rigid initialization and instance-optimization nonrigid stage. Changes for a CPU development run were: PIL/JPEG loader; each original image sampled at 0.1 in each spatial axis; preprocessing maximum resolution 768 (the recorded postprocessing ratio was 1); nonrigid registration size 512, five levels, 30 iterations per level (150 total) with the preset first five learning rates/regularization weights; final full-resolution image export disabled. This is a **reduced** configuration, so it is not a strong published-setting accuracy comparison. The exact settings are saved in `D:\QC_optimization_data\digital_topology_wsi\DHR_CD68_CD4_development_clean\config.json`.

The command run from the worktree was:

```powershell
D:\QC_optimization_data\digital_topology_wsi\dhr_clean_venv\Scripts\python.exe tools\digital_dhr_baseline.py --moving D:\QC_optimization_data\digital_topology_wsi\HistoReg_CD68_CD4\Images_CD68.jpg --fixed D:\QC_optimization_data\digital_topology_wsi\HistoReg_CD68_CD4\Images_CD4.jpg --output D:\QC_optimization_data\digital_topology_wsi\DHR_CD68_CD4_development_clean
```

All input-derived output was written on D:. The first attempt using the base Anaconda packages reached preprocessing and then terminated on duplicate OpenMP runtimes. An isolated environment with the pinned versions above resolved that startup conflict; the failed attempt did not produce a field.

## Measured result

| Measurement | Saved result |
| --- | ---: |
| Loaded and padded tensors, each | `[1, 3, 991, 747]` |
| Processed grayscale tensors, each | `[1, 1, 991, 747]` |
| Total pipeline including reading and writing field | 77.03 s |
| Image loading | 42.86 s |
| Preprocessing | 2.74 s |
| Feature initialization | 24.43 s |
| Nonrigid optimization | 5.95 s |
| Saved field | `Results_Final/displacement_field.mha`, float32 `(component, row, col) = (2, 991, 747)` |
| Finite values | all |
| Displacement range in sampled-grid pixels | x: −61.34 to 71.81; y: −88.65 to 33.72 |
| One-corner cell Jacobian audit | minimum −0.4768; 7,003 nonpositive of 738,540 cells |

The authoritative files are `D:\QC_optimization_data\digital_topology_wsi\DHR_CD68_CD4_development_clean\runtime.json`, `deeperhistreg.log`, and `HistoReg_CD68_to_CD4_development\Results_Final\displacement_field.mha` plus `postprocessing_params.json`. The saved field was re-read with SimpleITK for the finite/range/Jacobian observations. The negative one-corner determinants mean this sampled deformation does not meet this project's strict all-corner Q1 topology criterion; the count is not a clinical or registration-accuracy score. A subsequent independent **all-four-corner** and landmark inverse audit is recorded in `REVIEW_DHR_FIELD.md`: 7,108 cells fail at least one corner, and all 77 matched landmarks have a unique inverse with mean TRE 14.2325 supplied-JPEG pixels. Those later measurements are not part of the 77.03-second registration runtime.

### Coordinate convention

DeeperHistReg computes a backward sampling map: for a fixed/output sample at column `x`, row `y`, read the moving/source image at `(x + u_x, y + u_y)` in the **padded, 0.1-scaled registration canvas**. Its PyTorch path adds the displacement to an `affine_grid` and calls `grid_sample(..., align_corners=False)`. Its field saver converts normalized PyTorch displacement to grid pixels using `u_x = normalized_x × width/2`, `u_y = normalized_y × height/2`, and writes two scalar MHA planes, first x then y. `postprocessing_params.json` records source `pad_1=[[1,1],[0,0]]`, target `pad_2=[[0,0],[15,15]]`, and both source/target resample ratios `0.1`. Full-resolution landmark or image coordinates require undoing these pads and scales consistently; registration did not use the landmarks. The subsequent independent read-only evaluator performed this coordinate conversion; see `REVIEW_DHR_FIELD.md`.

## Limits and next comparison

This is a real paired-image development smoke run. It used only a tenth-scale image and 512-pixel nonrigid optimization and did not export a full-resolution warped JPEG/OME-TIFF. Public example landmarks were subsequently evaluated separately; they are **development diagnostics**, not held-out landmarks or an official ANHIR split. Its output field is a sampled displacement, not a guaranteed-bijective map. The patient/block/physical-slide group identity is still unknown. Before a formal method ranking, use an appropriate group-held-out metric and compare at the same pair, resolution, coordinate convention and hardware budget.
