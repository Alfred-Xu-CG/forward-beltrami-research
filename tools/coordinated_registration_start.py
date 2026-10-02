"""Load a declared absolute P1 incumbent, without changing its affine or reference."""
from pathlib import Path
import numpy as np
import torch
from tools.digital_q1_dhr_distill import identity_vertices
from qcopt.neural_bijection.dense.digital_q1 import q1_corner_determinants
from qcopt.neural_bijection.dense.q1_filtered_sign import certify_q1_binary_map


def load_incumbent(path,matrix,offset,side,*,device='cpu',minimum_jacobian=.001):
    path=Path(path)
    certificate=certify_q1_binary_map(path)
    if not certificate['valid']:raise ValueError('incumbent binary map certificate failed')
    with np.load(path,allow_pickle=False) as saved:
        vertices=saved['vertices'].copy();reference=saved['boundary_reference'].copy()
        if (vertices.dtype!=np.float64 or vertices.shape!=(1,side,side,2)
                or str(saved['interpolation'].item())!='p1_ac'
                or not np.array_equal(saved['post_affine_matrix'],np.asarray(matrix))
                or not np.array_equal(saved['post_affine_offset'],np.asarray(offset))):
            raise ValueError('incumbent fine P1ac grid/dtype/original affine mismatch')
    expected=identity_vertices(side,device='cpu').double().numpy()
    if not np.array_equal(reference,expected):raise ValueError('incumbent must use exact unit-square reference')
    a=np.asarray(matrix,dtype=np.float64);b=np.asarray(offset,dtype=np.float64)
    if a.shape!=(2,2) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.det(a)<=0:
        raise ValueError('positive finite original affine required')
    result=torch.as_tensor(vertices,device=device,dtype=torch.float64)
    ratio=q1_corner_determinants(result)*(side-1)**2
    if not bool(torch.isfinite(result).all() and (ratio>minimum_jacobian).all()):
        raise ValueError('incumbent actual strict configured corner floor failed')
    return result,dict(path=str(path.resolve()),saved_binary_certificate=certificate,
        minimum_corner_ratio=float(ratio.min()),interpolation='p1_ac',control_vertices=side**2,
        scope='absolute saved incumbent; identity geometric reference and original affine retained')
