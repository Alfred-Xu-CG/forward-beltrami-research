"""The exact mesh's collinear rectangle boundary satisfies Floater's chord condition."""

import numpy as np
import pytest

from qcopt.mesh import structured_rectangle


@pytest.mark.parametrize("side", [3, 5, 17, 257])
def test_no_dividing_edge_maps_to_same_square_side(side):
    faces = structured_rectangle(side-1,side-1).faces
    edges = np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],
                                    faces[:,[2,0]])),axis=1)
    unique, multiplicity = np.unique(edges,axis=0,return_counts=True)
    boundary = lambda indices: (
        (indices//side==0)|(indices//side==side-1)|
        (indices%side==0)|(indices%side==side-1))
    dividing = unique[(multiplicity==2)&boundary(unique[:,0])
                      &boundary(unique[:,1])]
    # SW-NE connectivity has exactly the two corner-cutting interior chords.
    assert len(dividing)==2
    for first,second in dividing:
        row_a,col_a=divmod(int(first),side)
        row_b,col_b=divmod(int(second),side)
        assert row_a!=row_b and col_a!=col_b
        assert not (row_a==row_b==0 or row_a==row_b==side-1
                    or col_a==col_b==0 or col_a==col_b==side-1)
