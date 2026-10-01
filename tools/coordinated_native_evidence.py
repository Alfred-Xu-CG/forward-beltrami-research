"""Frozen native DHR preprocessing only; no affine, field, matcher or labels.

The installed PIL pair loader and configured preprocessing function are called
directly. This is not a hand-written normalization/CLAHE approximation. The
supported scope is equal unpadded L/RGB square canvases with all scale ratios1.
Native PIL retains RGBA, which native Grayscale cannot process; alpha is never
silently discarded here. Returned order is fixed, moving, coordinate metadata.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

from PIL import Image
import torch


def native_evidence_parameters(config=None):
    """Installed fast preset plus the existing common-512 baseline overrides.

    Passing its saved config.json instead uses that exact captured configuration.
    Registration flags/parameters are not executed by this helper.
    """
    import deeperhistreg
    if config is None:
        params = deeperhistreg.configs.default_initial_nonrigid_fast()
        params["loading_params"].update(loader="pil", source_resample_ratio=1., target_resample_ratio=1.)
        params["preprocessing_params"].update(initial_resolution=768, save_results=False)
    elif isinstance(config, (str, Path)):
        params = json.loads(Path(config).read_text(encoding="utf-8"))
    else:
        params = copy.deepcopy(config)
    params.update(device="cpu", echo=False, logging_path=None)
    return params


def load_native_preprocessed_pair(fixed, moving, *, config=None, device="cpu", expected_side=512):
    """Return native float32 frozen fixed/moving features and unchanged frame.

    ``expected_side`` is smaller only for focused fixture tests. Invalid native
    normalization (e.g. a constant channel) is reported, not replaced or repaired.
    Geometry/image dtype conversion can occur AFTER these frozen features return.
    """
    import deeperhistreg
    params = native_evidence_parameters(config)
    loading, preprocessing = params["loading_params"], params["preprocessing_params"]
    if loading["loader"] != "pil" or any(loading[k] != 1 for k in ("source_resample_ratio", "target_resample_ratio")):
        raise ValueError("native evidence requires PIL and source/target ratio1")
    required = ("normalization", "convert_to_gray", "clahe", "flip_intensity", "initial_resampling")
    if preprocessing["preprocessing_function"] != "basic_preprocessing" or not all(preprocessing[k] for k in required):
        raise ValueError("declare the native common normalized/gray/flipped/CLAHE basic preprocessing")
    modes = {}
    for name, path in (("fixed", fixed), ("moving", moving)):
        with Image.open(path) as image:
            modes[name] = image.mode
            if image.mode not in ("L", "RGB"):
                raise ValueError(f"native PIL mode {image.mode} is unsupported; RGBA is not silently converted")
            if image.size != (expected_side, expected_side):
                raise ValueError("equal declared square canvases required; no resize or padding")
    if preprocessing["initial_resolution"] < expected_side:
        raise ValueError("initial resolution would change the declared coordinate frame")
    # Use the EXACT loader/function objects referenced by the installed pipeline.
    native = importlib.import_module(deeperhistreg.direct_registration.DeeperHistReg_FullResolution.__module__)
    loader = native.pair_full_loader.PairFullLoader(str(moving), str(fixed),
        loader=native.loader_mapper["pil"], mode=native.pair_full_loader.LoadMode.PYTORCH)
    source, target, padding = loader.load_array(source_resample_ratio=1., target_resample_ratio=1.,
                                              pad_value=loading["pad_value"])
    if any(value != 0 for key in ("pad_1", "pad_2") for pair in padding[key] for value in pair):
        raise ValueError("native pair loader introduced padding")
    function = native.pre.get_function(preprocessing["preprocessing_function"])
    with torch.no_grad():
        source = source.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        # Native normalize divides by per-channel range without an epsilon.
        # CLAHE's uint8 cast can subsequently hide those NaNs as black pixels.
        # Reject the undefined input rather than claim its cast is meaningful.
        if any(bool((image.amax((-2, -1)) == image.amin((-2, -1))).any()) for image in (source, target)):
            raise ValueError("native channel normalization would be nonfinite for a constant channel")
        moving_feature, fixed_feature, _, _, post = function(source, target, None, None, preprocessing)
    if post["initial_resample_ratio"] != 1 or fixed_feature.shape != (1, 1, expected_side, expected_side) or moving_feature.shape != fixed_feature.shape:
        raise ValueError("native preprocessing changed the declared feature frame")
    if not bool(torch.isfinite(fixed_feature).all() and torch.isfinite(moving_feature).all()):
        raise ValueError("native preprocessing produced nonfinite features (constant-channel normalization is undefined)")
    metadata = dict(fixed=str(fixed), moving=str(moving), original_modes=modes,
        original_size=[expected_side, expected_side], feature_shape=list(fixed_feature.shape),
        feature_dtype=str(fixed_feature.dtype), device=str(fixed_feature.device),
        source_resample_ratio=1., target_resample_ratio=1., initial_resample_ratio=post["initial_resample_ratio"],
        padding=padding, coordinate_frame_unchanged=True,
        query_convention="unit pixel centers (x+.5)/width,(y+.5)/height; original moving features, no affine prewarp",
        loading_gaussian_sigma=.1, initial_gaussian_sigma=.1, initial_gaussian_kernel_size=1,
        smoothing_note="installed sigma.1 GaussianBlur has kernel1: exactly identity at ratio1",
        loading_params=copy.deepcopy(loading), preprocessing_params=copy.deepcopy(preprocessing),
        native_pipeline_source=native.__file__, native_preprocessing_source=native.pre.__file__,
        native_loader_source=importlib.import_module(loader.source_loader.__class__.__module__).__file__,
        affine_nonrigid_matcher_or_labels_loaded=False,
        scope="native frozen preprocessing only; not native NCC/pyramid/diffusion objective equivalence")
    return fixed_feature.detach(), moving_feature.detach(), metadata
