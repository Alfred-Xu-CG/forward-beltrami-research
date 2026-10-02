"""Explicit prepared terminal-image evidence; original controls/points stay fixed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _array(value,side,name):
    array=torch.as_tensor(value)
    if array.dtype!=torch.float32 or array.shape!=(1,1,side,side) or array.requires_grad:
        raise ValueError(f'{name}: frozen float32 (1,1,{side},{side}) array required')
    if not bool(torch.isfinite(array).all() and (array>=0).all() and (array<=1).all()):
        raise ValueError(f'{name}: finite intensities/weights in [0,1] required')
    return array


def validate_terminal_tensors(terminal_evidence,args,source_mask,*,device,dtype):
    """No preprocessing: validate actual arrays and exact replicated old support."""
    source_side=terminal_evidence['source_image_side']
    side=args.image_side
    if (isinstance(source_side,bool) or not isinstance(source_side,int) or source_side<8
            or side<source_side or side%source_side or source_mask.shape!=(1,1,source_side,source_side)):
        raise ValueError('terminal/source sides must be an integer enlargement of original square support')
    fixed,moving,mask=[_array(terminal_evidence[k],side,k) for k in ('fixed','moving','fixed_mask')]
    factor=side//source_side
    expected=source_mask.detach().cpu().repeat_interleave(factor,-2).repeat_interleave(factor,-1)
    if not torch.equal(mask.cpu(),expected):
        raise ValueError('terminal mask must EXACTLY repeat the original source mask, without rethresholding')
    metadata=terminal_evidence.get('metadata',{})
    if not isinstance(metadata,dict):raise ValueError('terminal metadata must be a plain mapping')
    metadata=json.loads(json.dumps(metadata,allow_nan=False))
    metadata.update(source_image_side=source_side,terminal_image_side=side,
        mask_policy='exact integer repeat of original fixed grayscale inversion >.04 support',
        intensity_policy='prepared float32 inverted grayscale; no further grayscale conversion, normalization or quantization',
        source_mask_normalized_mass=float(source_mask.double().mean()),
        terminal_mask_normalized_mass=float(mask.double().mean()))
    return *(x.to(device=device,dtype=dtype) for x in (fixed,moving,mask)),metadata


def load_terminal_bundle(args,bundle_path):
    """Production pair bundle. Tiny same-raster checks call the core hook directly."""
    if args.image_side!=1024 or list(args.image_levels)!=[32,64,128,256,1024]:
        raise ValueError('terminal detail bundle requires declared 1024 final query raster and fixed prefix')
    bundle_path=Path(bundle_path)
    with np.load(bundle_path,allow_pickle=False) as archive:
        values={key:np.asarray(archive[key]).copy() for key in ('fixed','moving','fixed_mask')}
        metadata=json.loads(str(archive['metadata']))
    if (not isinstance(metadata,dict) or metadata.get('arm') not in ('direct_original','lift512')
            or metadata.get('source_image_side')!=512 or metadata.get('terminal_image_side')!=1024):
        raise ValueError('explicit direct-original/lift512 bundle and raster sides required')
    basename=lambda value:Path(str(value).replace('\\','/')).name
    if any(basename(metadata.get(key,''))!=Path(getattr(args,key)).name for key in ('fixed','moving')):
        raise ValueError('terminal bundle pair does not match the accepted512 image pair')
    for key,value in values.items():_array(value,1024,key)
    metadata={**metadata,'bundle_path':str(bundle_path),
              'path_check':'accepted512 raster basenames; relocated paths allowed, not content identity proof'}
    return {**values,'source_image_side':512,'metadata':metadata}


def optimize_terminal_detail(args,bundle_path):
    """Run the existing frozen-pose optimizer with one explicit terminal override."""
    from tools.coordinated_real_case import optimize
    return optimize(args,terminal_evidence=load_terminal_bundle(args,bundle_path))
