"""Frozen released MatchAnything ELoFTR image points; no labels or dense teacher.

The matcher is an external, fixed evidence source. The existing coordinated
optimizer remains responsible for deformation and its geometric certificate.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from tools.digital_affine_prewarp import warp_moving_to_fixed


SOURCE_REVISION = '6a7bcb589ec8da3a9e861e799122beaa5eba2193'
SOURCE_URL = 'https://huggingface.co/spaces/LittleFrog/MatchAnything'
WEIGHTS_URL = 'https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view'
IMAGE_SIDE = 512


def _affine(matrix, offset):
    matrix, offset = np.asarray(matrix), np.asarray(offset)
    if matrix.shape != (2, 2) or offset.shape != (2,):
        raise ValueError('affine shape must be 2x2 and 2')
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise ValueError('nonfinite affine')
    a = matrix.astype(np.float64)
    if not (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0] > 0):
        raise ValueError('the frozen affine must have positive determinant')
    return matrix, offset


def point_record(points0, points1, confidence, matrix, offset, *,
                 image_side=IMAGE_SIDE, fixed_support=None):
    """Adapt literal pixel-index outputs without geometric filtering or clipping."""
    if image_side != IMAGE_SIDE:
        raise ValueError('approved matcher input is exactly 512 square')
    matrix, offset = _affine(matrix, offset)
    p0, p1 = np.asarray(points0, dtype=np.float64), np.asarray(points1, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if p0.ndim != 2 or p0.shape[-1] != 2 or p1.shape != p0.shape or confidence.shape != (len(p0),):
        raise ValueError('match shape must be Nx2, Nx2 and N')
    # Check ALL outputs before domain filtering: a NaN is a failed extraction.
    if not np.isfinite(p0).all() or not np.isfinite(p1).all() or not np.isfinite(confidence).all():
        raise ValueError('nonfinite model point or confidence output')
    if ((confidence < 0) | (confidence > 1)).any():
        raise ValueError('model confidence must lie in [0,1]')
    q, p = (p0 + .5) / image_side, (p1 + .5) / image_side
    inside = ((q >= 0) & (q <= 1) & (p >= 0) & (p <= 1)).all(axis=1)
    network_matches = len(q)
    q, p, confidence = q[inside], p[inside], confidence[inside]
    world_target = p @ matrix.astype(np.float64).T + offset.astype(np.float64)
    eligible = ((world_target >= 0) & (world_target <= 1)).all(axis=1)
    positive = eligible & (confidence > 0)
    bins = np.minimum(np.floor(q[positive] * 4).astype(np.int64), 3)
    source_counts = np.unique(q, axis=0, return_counts=True)[1]
    target_counts = np.unique(p, axis=0, return_counts=True)[1]
    support_fraction = None
    if fixed_support is not None:
        support = np.asarray(fixed_support)
        if support.shape != (image_side, image_side) or support.dtype != np.bool_:
            raise ValueError('fixed support diagnostic must be a boolean 512 square')
        # Containing-pixel lookup; endpoint clamping is ONLY for this diagnostic.
        indices = np.minimum(np.floor(q * image_side).astype(np.int64), image_side - 1)
        support_fraction = float(support[indices[:, 1], indices[:, 0]].mean()) if len(q) else 0.
    return dict(
        status='ok' if int(positive.sum()) >= 8 else 'insufficient_matches',
        network_matches=network_matches, raw_matches=len(q),
        discarded_out_of_unit_domain=int((~inside).sum()),
        eligible_matches=int(positive.sum()),
        static_outside_original_moving_matches=int((~eligible).sum()),
        zero_confidence_matches=int((confidence == 0).sum()),
        confidence_mass=float(confidence.sum()),
        eligible_confidence_mass=float(confidence[eligible].sum()),
        occupied_quadrants_4x4=len(set(map(tuple, bins))),
        unique_source_coordinates=len(source_counts),
        unique_target_coordinates=len(target_counts),
        maximum_source_multiplicity=int(source_counts.max()) if len(source_counts) else 0,
        maximum_target_multiplicity=int(target_counts.max()) if len(target_counts) else 0,
        multiplicity_scope='exact retained fine point coordinates; reporting only; no deduplication',
        source_original_gray_support_fraction=support_fraction,
        source_support_diagnostic='unweighted retained-source containing pixel floor(512*q); q=1 uses index511; no point filtering',
        source_points_unit=q.tolist(), target_points_unit=p.tolist(),
        confidence=confidence.tolist(),
        coordinate_convention='fixed unit q -> affine-aligned moving unit p; full moving point is A p+b',
        global_geometric_ransac_used=False,
        tissue_support_filter_used=False,
        insufficient_match_policy='fewer than8positive staticeligible matches: extraction failure; no SG fallback',
    )


def read_matcher_gray(path: Path):
    """Released ordinary PIL grayscale convention, retaining the exact canvas."""
    with Image.open(path) as image:
        if image.size != (IMAGE_SIDE, IMAGE_SIDE):
            raise ValueError('matcher requires an existing 512x512 canvas; no implicit resize')
        gray = np.array(image.convert('L'), dtype=np.float32, copy=True) / 255.
    return torch.from_numpy(gray[None, None])


def _load_model(source_root: Path, checkpoint: Path, device: torch.device):
    """Load inference classes only, not Lightning/ROMA or released training pins."""
    source_root, checkpoint = Path(source_root).resolve(), Path(checkpoint).resolve()
    config_path = source_root / 'configs/models/eloftr_model.py'
    for required in (config_path, source_root / 'src/config/default.py', source_root / 'src/loftr/loftr.py', checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    sys.path.insert(0, str(source_root))
    try:
        cfg_module = importlib.import_module('src.config.default')
        model_module = importlib.import_module('src.loftr')
        for loaded in (cfg_module, model_module):
            if not Path(loaded.__file__).resolve().is_relative_to(source_root):
                raise RuntimeError('another src.config/src.loftr package is already loaded; use an isolated matcher process')
        config = cfg_module.get_cfg_defaults()
        config.merge_from_file(str(config_path))
        if config.DATASET.NPE_NAME != 'megadepth' or not config.LOFTR.COARSE.ROPE:
            raise ValueError('released ELoFTR positional-encoding configuration differs')
        config.LOFTR.COARSE.NPE = [832, 832, IMAGE_SIDE, IMAGE_SIDE]
        if config.LOFTR.MATCH_COARSE.THR != .1 or config.LOFTR.FP16 or config.DATASET.FP16:
            raise ValueError('released threshold.1/FP32 configuration differs')
        from yacs.config import CfgNode

        def lower_config(value):
            return {key.lower(): lower_config(item) for key, item in value.items()} if isinstance(value, CfgNode) else value

        model_config = lower_config(config.LOFTR)
        model = model_module.LoFTR(config=model_config)
        if not model.coarse_matching.mtd:
            raise ValueError('released mtd_spvs=True coarse matching configuration differs')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get('state_dict'), dict):
            raise ValueError('checkpoint must contain its released state_dict')
        state = payload['state_dict']
        # The released override strips matcher. prefixes, then enforces strict=True.
        matched = model.load_state_dict(state, strict=True)
        if matched.missing_keys or matched.unexpected_keys:
            raise RuntimeError('checkpoint model keys did not match completely')
        model.requires_grad_(False).eval().to(device)
        summary = dict(strict_keys=True, state_dict_entries=len(state),
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       coarse_threshold=float(config.LOFTR.MATCH_COARSE.THR),
                       configuration_force_nearest=bool(config.LOFTR.MATCH_COARSE.FORCE_NEAREST),
                       actual_coarse_match_policy='all thresholded coarse pairs after border removal (mtd_spvs=True); FORCE_NEAREST is unused by this released CoarseMatching class',
                       confidence_scope='released coarse confidence retained through TOPK=1 fine matching and local regression',
                       coarse_npe=list(config.LOFTR.COARSE.NPE),
                       inference_dtype='float32', weights_only=True,
                       model_config=model_config)
        return model, summary
    finally:
        sys.path.remove(str(source_root))


class FrozenMatchAnything:
    """One model load for a batch; point extraction never opens manual labels."""

    def __init__(self, source_root: Path, checkpoint: Path, *, device: str = 'cpu'):
        self.device = torch.device(device)
        self.source_root, self.checkpoint = Path(source_root), Path(checkpoint)
        started = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.model, model_summary = _load_model(self.source_root, self.checkpoint, self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.setup_report = dict(
            setup_seconds=time.perf_counter() - started,
            setup_peak_allocated_bytes=self._peak(),
            source_root=str(self.source_root), checkpoint=str(self.checkpoint),
            source_url=SOURCE_URL, source_revision=SOURCE_REVISION,
            author_weights_url=WEIGHTS_URL, model=model_summary,
            setup_scope='one cold model load/device transfer; no first forward or optimizer',
        )

    def _peak(self):
        return int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == 'cuda' else None

    def extract(self, fixed: Path, moving: Path, affine: Path, *, output: Path | None = None) -> dict:
        if self.model is None:
            raise RuntimeError('matcher is closed')
        fixed, moving, affine = Path(fixed), Path(moving), Path(affine)
        output = None if output is None else Path(output)
        if output is not None and output.exists():
            raise FileExistsError(output)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        fixed_gray, moving_gray = read_matcher_gray(fixed), read_matcher_gray(moving)
        fixed_support = ((1. - fixed_gray[0, 0]).numpy() > .04)
        with np.load(affine, allow_pickle=False) as archive:
            matrix = archive['post_affine_matrix'].copy()
            offset = archive['post_affine_offset'].copy()
        _affine(matrix, offset)
        with torch.inference_mode():
            fixed_gray, moving_gray = fixed_gray.to(self.device), moving_gray.to(self.device)
            aligned = warp_moving_to_fixed(moving_gray, torch.as_tensor(matrix, device=self.device),
                                           torch.as_tensor(offset, device=self.device), height=IMAGE_SIDE, width=IMAGE_SIDE)
            batch = {'image0': fixed_gray, 'image1': aligned}
            self.model(batch)
            p0, p1, confidence = [batch[key].detach().cpu().numpy() for key in ('mkpts0_f', 'mkpts1_f', 'mconf')]
        result = point_record(p0, p1, confidence, matrix, offset, fixed_support=fixed_support)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        result.update(
            fixed=str(fixed), moving=str(moving), affine=str(affine),
            post_affine_matrix=matrix.tolist(), post_affine_offset=offset.tolist(),
            image_side=IMAGE_SIDE,
            method='frozen released MatchAnything ELoFTR; cross-modality pretrained; threshold.1; no global RANSAC',
            matcher_input='PIL L/255 ordinary grayscale; moving same-affine bilinear BORDER prewarp; align_corners=False; no resize',
            source_revision=SOURCE_REVISION,
            inference_seconds=time.perf_counter() - started,
            extraction_seconds=time.perf_counter() - started,
            extraction_time_scope='raster/affine loading, grayscale, prewarp, model forward and point adaptation; model setup/JSON write excluded',
            peak_allocated_bytes=self._peak(),
            targets_manual_landmarks_or_dense_teacher_loaded=False,
            affine_estimated_or_changed=False,
            setup_report_scope='reported once by FrozenMatchAnything.setup_report, not charged perpair',
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            tick = time.perf_counter()
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            result['serialization_seconds'] = time.perf_counter() - tick
        return result

    def close(self):
        self.model = None
        gc.collect()
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source_root', 'checkpoint', 'fixed', 'moving', 'affine', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    matcher = FrozenMatchAnything(args.source_root, args.checkpoint, device=args.device)
    try:
        report = matcher.extract(args.fixed, args.moving, args.affine, output=args.output)
        print(json.dumps(dict(setup=matcher.setup_report, extraction={key: value for key, value in report.items()
              if key not in ('source_points_unit', 'target_points_unit', 'confidence')}), allow_nan=False))
    finally:
        matcher.close()


if __name__ == '__main__':
    main()
