"""Source P1 evaluation spectrum only; NOT the image-objective Hessian."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from qcopt.neural_bijection.dense.coordinated_fixed_sampling import FrozenP1Evaluator


def sampling_matrix(control_side, image_side):
    """SMALL diagnostic matrix with fixed boundary amplitudes eliminated."""
    if control_side>17:
        raise ValueError("explicit matrix is intentionally limited to tiny grids")
    evaluator=make_evaluator(control_side,image_side)
    interior=torch.zeros(control_side,control_side,dtype=torch.bool)
    interior[1:-1,1:-1]=True
    columns=torch.full((control_side**2,),-1,dtype=torch.long)
    columns[interior.flatten()]=torch.arange(int(interior.sum()))
    ids=columns[evaluator.vertex_ids]
    matrix=torch.zeros(image_side**2,int(interior.sum()),dtype=torch.float64)
    for slot in range(3):
        valid=ids[:,slot]>=0
        matrix[torch.arange(image_side**2)[valid],ids[valid,slot]]+=evaluator.weights[valid,slot]
    return matrix


def make_evaluator(control_side,image_side):
    centers=(torch.arange(image_side,dtype=torch.float64)+.5)/image_side
    y,x=torch.meshgrid(centers,centers,indexing="ij")
    return FrozenP1Evaluator(control_side,control_side,torch.stack((x,y),-1)[None],"ac")


def chain_spectrum(length):
    if not isinstance(length,int) or isinstance(length,bool) or length<1:
        raise ValueError("positive integer chain length required")
    angle=math.pi/(2*(length+1))
    return dict(interior_chain_length=length,minimum_gram_eigenvalue=math.sin(angle)**2,
                maximum_gram_eigenvalue=math.cos(angle)**2,gram_condition=1/math.tan(angle)**2,
                sampling_condition=1/math.tan(angle),exact_null_mode=False)


def run():
    torch.set_num_threads(1)
    rows=[]
    for control_side in (17,65,257,1025):
        image_side=control_side-1
        evaluator=make_evaluator(control_side,image_side)
        y,x=torch.meshgrid(torch.arange(control_side,dtype=torch.float64),
                           torch.arange(control_side,dtype=torch.float64),indexing="ij")
        # Deterministic non-affine data; no registration/topology claim for it.
        vertices=torch.stack((torch.sin(x+.2*y),torch.cos(.7*x-y)),-1)[None]
        actual=evaluator(vertices)
        expected=.5*(vertices[:,:-1,:-1]+vertices[:,1:,1:])
        rows.append(dict(control_side=control_side,image_side=image_side,
            control_vertices=control_side**2,queries=image_side**2,
            midpoint_max_abs_error=float((actual-expected).abs().max()),
            three_positive_weight_fraction=float(((evaluator.weights>0).sum(1)==3).double().mean()),
            **chain_spectrum(control_side-2)))
        del evaluator,vertices,actual,expected
    tiny=[]
    for control_side in (5,9):
        for ratio in (1,2):
            s=sampling_matrix(control_side,ratio*(control_side-1))
            eigenvalues=torch.linalg.eigvalsh(s.T@s)
            tiny.append(dict(control_side=control_side,samples_per_cell_axis=ratio,
                minimum_gram_eigenvalue=float(eigenvalues[0]),
                maximum_gram_eigenvalue=float(eigenvalues[-1]),
                gram_condition=float(eigenvalues[-1]/eigenvalues[0])))
    return dict(scope="unweighted fixed-boundary source P1 evaluation only, not full image/prior Hessian, anatomy or a topology certificate",
                diagonal="ac",large_grids=rows,tiny_explicit_matrices=tiny,
                interpretation="ratio2 uses quarter/threequarter source samples; it supplies no extra original-image information and is not an executed registration intervention")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=run()
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":main()
