"""Explicit direct-original/lift512 terminal rendering, without annotations."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tools.coordinated_real_case import load_registration_evidence

ARMS=('direct_original','lift512')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def native_cases(manifest):
    path=Path(manifest).resolve()
    value=read_json(path)
    return {row['name']:{role:(path.parent/row[role]).resolve() for role in ('fixed','moving')}
        for row in value['rows']}


def decoder_cache(manifest):
    """Optional lossless historical JPEG decode, not a different resize kernel."""
    if manifest is None:return {}
    path=Path(manifest).resolve();value=read_json(path);result={}
    for row in value['rows']:
        name=row['source_name']
        if name in result:raise ValueError('unique source basename required for decoder bridge')
        if str(row['source']).replace('\\','/').rsplit('/',1)[-1]!=name:
            raise ValueError('decoder source basename mismatch')
        result[name]={**row,'decoded_rgb':str((path.parent/row['decoded_rgb']).resolve()),
            'decoder':row.get('decoder',value['decoder'])}
    return result

def input_pair(case,layouts,native,miit_source):
    name=case['name'];cfg=case['original_configuration'];layouts=Path(layouts)
    if case['cohort']=='miit':
        _,moving,_,fixed=name.split('_')
        sources={role:Path(miit_source)/section/'images/image.tif'
            for role,section in [('fixed',fixed),('moving',moving)]}
        saved=read_json(Path(cfg['fixed']).parent/(name+'_layout.json'))
        if saved['side']!=512:raise ValueError('accepted512 MIIT layout required')
        roles={role:saved[role] for role in ('fixed','moving')}
    else:
        sources=native[name]
        if name in ('histo','rat_kidney'):
            saved=read_json(layouts/(name+'_layout.json'))
            if saved['side']!=512:raise ValueError('accepted512 layout required')
            roles={role:saved[role] for role in ('fixed','moving')}
        else:
            roles={}
            for role,stain in zip(('fixed','moving'),name.split('_to_')):
                saved=read_json(layouts/(('cc10' if stain=='he' else stain)+'_layout.json'))
                if saved['side']!=512:raise ValueError('accepted512 lung layout required')
                roles[role]=saved['fixed' if stain=='he' else 'moving']
    for role in ('fixed','moving'):
        source_name=str(roles[role]['source']).replace('\\','/').rsplit('/',1)[-1]
        if source_name!=sources[role].name:raise ValueError('native source differs from accepted layout')
    return sources,roles

def render_original(source,layout,accepted,*,side=512,decoded=None):
    """Return direct2x intensity and metadata; reproduce the accepted RGB first."""
    wh=np.asarray(layout['original_wh']);n=np.asarray(layout['resized_wh']);p=np.asarray(layout['padding_xy'])
    scale=np.asarray(layout['effective_original_to_canvas_scale_xy'])
    if (any(v.shape!=(2,) for v in (wh,n,p,scale)) or not all(np.isfinite(v).all() for v in (wh,n,p,scale))
            or any(not np.equal(v,np.floor(v)).all() for v in (wh,n,p))
            or np.any(wh<=0) or np.any(n<=0) or np.any(p<0) or np.any(p+n>side)
            or not np.array_equal(scale,n/wh)):
        raise ValueError('exact integer original/accepted layout required')
    with Image.open(source) as source_image:
        if list(source_image.size)!=wh.tolist() or source_image.getexif().get(274,1)!=1:
            raise ValueError('native source dimensions/orientation changed')
        original=source_image.convert('RGB') if decoded is None else None
    if decoded is not None:
        if (decoded['source_name']!=Path(source).name or decoded['original_wh']!=wh.tolist()):
            raise ValueError('decoder bridge must preserve original source identity and dimensions')
        with Image.open(decoded['decoded_rgb']) as image:
            if image.mode!='RGB' or list(image.size)!=wh.tolist() or image.format!='PNG':
                raise ValueError('lossless full-resolution RGB PNG decoder bridge required')
            original=image.copy()
    with Image.open(accepted) as image:
        if image.size!=(side,side):raise ValueError('accepted canvas shape mismatch')
        accepted_rgb=np.asarray(image.convert('RGB'))
    rebuilt=Image.new('RGB',(side,side),(255,255,255))
    rebuilt.paste(original.resize(tuple(n.astype(int)),Image.Resampling.BILINEAR),tuple(p.astype(int)))
    if not np.array_equal(np.asarray(rebuilt),accepted_rgb):
        raise ValueError('original source/layout does not reproduce accepted512 RGB exactly')
    direct=Image.new('RGB',(2*side,2*side),(255,255,255))
    direct.paste(original.resize(tuple((2*n).astype(int)),Image.Resampling.BILINEAR),tuple((2*p).astype(int)))
    intensity=1.-np.asarray(direct.convert('L'),dtype=np.float32)/np.float32(255.)
    metadata=dict(source=str(source),original_wh=wh.tolist(),accepted_resized_wh=n.tolist(),
        accepted_padding_xy=p.tolist(),terminal_resized_wh=(2*n).tolist(),terminal_padding_xy=(2*p).tolist(),
        terminal_to_original_resize_ratio_xy=(2*n/wh).tolist(),upscaled_axes=(2*n>wh).tolist(),
        accepted512_reconstruction_exact=True,
        decoding=(dict(mode='historical_lossless_RGB_bridge',**decoded) if decoded is not None
                  else dict(mode='current_PIL_source_decode')),
        original_center_to_unit='((x+.5)*(accepted_resized_wh/original_wh)+accepted_padding_xy)/512')
    return intensity[None,None],metadata

def prepare_pair(case,output,sources,layouts,*,decoded_sources=None):
    output=Path(output);cfg=case['original_configuration'];started=time.perf_counter()
    paths={arm:output/(case['name']+'_'+arm+'.npz') for arm in ARMS}
    if any(path.exists() for path in paths.values()):raise FileExistsError('new render bundles required')
    fixed,moving,mask,_=load_registration_evidence(Path(cfg['fixed']),Path(cfg['moving']),512,
        preprocessing='raw_inverted',device='cpu',dtype=torch.float32)
    mask2=mask.repeat_interleave(2,-2).repeat_interleave(2,-1).numpy()
    original={};role_metadata={}
    for role in ('fixed','moving'):
        original[role],role_metadata[role]=render_original(sources[role],layouts[role],Path(cfg[role]),
            decoded=(decoded_sources or {}).get(Path(sources[role]).name))
    lift={role:F.interpolate(value,size=(1024,1024),mode='bilinear',align_corners=False).numpy()
        for role,value in [('fixed',fixed),('moving',moving)]}
    output.mkdir(parents=True,exist_ok=True)
    for arm,arrays in [('direct_original',original),('lift512',lift)]:
        metadata=dict(arm=arm,source_image_side=512,terminal_image_side=1024,
            fixed=str(cfg['fixed']),moving=str(cfg['moving']),affine=str(cfg['affine']),
            point_prediction_side=512,point_pixel_scale=512,point_robust_scale=8,
            roles=role_metadata,mask='exact2x replication of original512 fixed grayscale>.04',
            render=('original RGB/PIL BILINEAR double layout then PIL L/inversion' if arm=='direct_original'
                    else 'unchanged512 inverted float32 gray; torch bilinear align_corners=False; no uint8 roundtrip'),
            added_native_information=arm=='direct_original',uniform_native1024_resolution=False,
            annotations_read=False)
        with paths[arm].open('xb') as stream:
            np.savez(stream,fixed=arrays['fixed'],moving=arrays['moving'],fixed_mask=mask2,
                metadata=np.asarray(json.dumps(metadata)))
    return dict(name=case['name'],bundles={k:str(v.resolve()) for k,v in paths.items()},
        preparation_seconds=time.perf_counter()-started,roles=role_metadata,
        mask_normalized_mass_512=float(mask.mean()),mask_normalized_mass_1024=float(mask2.mean()),annotations_read=False)
