"""One detached scalar logging transfer; no objective/geometry arithmetic.

Floating32 values embed exactly in floating64. Integer32 tensors and Python
integers through2**53 embed exactly too. Integer64 tensors are intentionally
unsupported instead of silently rounding a possibly large logging counter.
"""
from collections.abc import Mapping

import torch


def pack_trial_scalars(values):
    """Stack already-detached scalars, copy once, restore Python scalar types.

    Nonfinite floating values are preserved for the caller's EXISTING guards.
    Empty mappings require no copy. No input graph is silently detached here.
    """
    if not isinstance(values,Mapping) or any(not isinstance(key,str) for key in values):
        raise TypeError("string-keyed scalar mapping required")
    if not values:return {}
    tensors=[value for value in values.values() if isinstance(value,torch.Tensor)]
    device=tensors[0].device if tensors else torch.device("cpu")
    packed=[];types=[]
    for value in values.values():
        if isinstance(value,torch.Tensor):
            if value.numel()!=1 or value.requires_grad or value.device!=device:
                raise ValueError("already-detached scalar tensors on one device required")
            if value.dtype not in (torch.float32,torch.float64,torch.bool,torch.int32):
                raise TypeError("float32/64, bool or exact int32 tensor scalars required")
            kind=bool if value.dtype==torch.bool else int if value.dtype==torch.int32 else float
            packed.append(value.reshape(()).to(dtype=torch.float64))
        elif isinstance(value,(bool,int,float)):
            if isinstance(value,int) and abs(value)>2**53:
                raise ValueError("Python integer outside exact float64 logging range")
            kind=type(value)
            packed.append(torch.tensor(value,dtype=torch.float64,device=device))
        else:raise TypeError("tensor or Python real scalar required")
        types.append(kind)
    numbers=torch.stack(packed).cpu().tolist()  # ONE GPU->CPU copy for the trial.
    return {key:kind(number) for key,kind,number in zip(values,types,numbers,strict=True)}
