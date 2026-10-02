"""Exact quarter-turn descriptor-frame diagnostic, not registration or training.

The fixed image is an exact spatial rotation of the moving image, so no unknown
neighborhood deformation, interpolated raster, manual landmark or OOB mask is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tools.digital_mind_objective_probe import OFFSETS, self_similarity


def channel_permutation(turns):
    if isinstance(turns,bool) or not isinstance(turns,int) or turns not in range(4):
        raise ValueError("quarter turns must be an integer from zero through three")
    result=[]
    for x,y in OFFSETS:
        for _ in range(turns): x,y=-y,x
        result.append(OFFSETS.index((x,y)))
    return result


def probe(image):
    if image.ndim!=4 or image.shape[:2]!=(1,1) or image.shape[-2]!=image.shape[-1]:
        raise ValueError("one square scalar image required")
    if image.dtype!=torch.float64 or not bool(torch.isfinite(image).all()):
        raise ValueError("finite float64 diagnostic image required")
    moving,_=self_similarity(image)
    rows=[]
    for k in range(4):
        fixed,_=self_similarity(torch.rot90(image,k,(-2,-1)))
        transported=torch.rot90(moving,k,(-2,-1))
        permutation=channel_permutation(k)
        corrected=transported[:,permutation]
        def error(values):
            delta=(fixed-values).abs()
            return dict(mean_absolute=float(delta.mean()),maximum_absolute=float(delta.max()))
        rows.append(dict(quarter_turns=k,channel_permutation=permutation,
                         unpermuted=error(transported),permuted=error(corrected)))
    return rows


def run(output,image_path=None):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(1729)
    axis=torch.linspace(0,1,64,dtype=torch.float64)
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    texture=(.45+.18*torch.sin(31*x+2*y)+.11*torch.cos(9*y)+
             .04*torch.rand((64,64),generator=generator,dtype=torch.float64))[None,None]
    rows=[dict(name="seeded_anisotropic_64",side=64,results=probe(texture))]
    if image_path is not None:
        from tools.digital_q1_real_optimize import _read_gray_thumbnail
        image,_=_read_gray_thumbnail(image_path,512)
        rows.append(dict(name="existing_real_texture_exact_rotations",source=str(image_path),
                         side=512,results=probe(image.double())))
    result=dict(question="exact affine image correspondence and directional descriptor channel frames",
        dtype="float64",rows=rows,annotations_read=False,optimizer_executed=False,
        spatial_operation="exact torch.rot90 of the prepared raster; no rotation interpolation or OOB samples (optional image loading may bilinearly resize input)",
        fixed_image_definition="I_f(x)=I_m(A_k x+b_k); inverse pixel-index rotation has linear part (x,y)->(-y,x) for k=1",
        interpretation="equivariance mechanism only, NOT evidence of real MIIT causality, a new descriptor, or an accuracy improvement")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--image",type=Path,help="optional existing texture; exact rotations only")
    a=p.parse_args();print(json.dumps(run(a.output,a.image)))


if __name__=="__main__":main()
