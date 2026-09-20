import numpy as np

from qcopt.forward.benchmarks import affine_map, smooth_twist_map
from qcopt.forward.prolongation import prolongate_positive_increments, prolongate_regular_grid
from qcopt.mesh import structured_rectangle
from qcopt.injectivity import audit_injectivity


def test_regular_grid_prolongation_reproduces_an_affine_map():
    coarse = structured_rectangle(8, 6)
    fine = structured_rectangle(32, 24)
    matrix = np.asarray([[1.2, 0.1], [0.05, 0.9]])
    offset = np.asarray([0.2, -0.1])
    coarse_uv = affine_map(coarse.vertices, matrix, offset)

    fine_uv = prolongate_regular_grid(coarse_uv, 8, 6, 32, 24)

    assert np.max(np.abs(fine_uv - affine_map(fine.vertices, matrix, offset))) < 1e-12
    assert audit_injectivity(fine, fine_uv).certified


def test_regular_grid_prolongation_preserves_a_smooth_truth_map_on_refinement():
    coarse = structured_rectangle(16, 12)
    fine = structured_rectangle(64, 48)
    coarse_uv = smooth_twist_map(coarse.vertices, amplitude=0.08)

    fine_uv = prolongate_regular_grid(coarse_uv, 16, 12, 64, 48)

    assert audit_injectivity(fine, fine_uv).certified
    source_det = np.linalg.det(_jacobians(fine, fine.vertices))
    target_det = np.linalg.det(_jacobians(fine, fine_uv))
    assert np.min(target_det / source_det) > 0.5


def test_bilinear_prolongation_can_fold_even_when_coarse_diagonal_faces_are_positive():
    coarse = structured_rectangle(1, 1)
    fine = structured_rectangle(16, 16)
    coarse_uv = np.asarray(
        [
            [-0.43038134, -0.02790793],
            [0.70675741, -1.75678915],
            [1.51860469, -1.74314225],
            [0.22238447, -0.91419358],
        ]
    )

    fine_uv = prolongate_regular_grid(coarse_uv, 1, 1, 16, 16)
    coarse_det = np.linalg.det(_jacobians(coarse, coarse_uv))
    fine_det = np.linalg.det(_jacobians(fine, fine_uv))

    assert np.all(coarse_det > 0.0)
    assert np.min(fine_det) < 0.0


def test_positive_parameter_prolongation_cannot_create_a_negative_increment():
    coarse = np.asarray([0.2, 1.0, 0.4, 2.0])
    fine = prolongate_positive_increments(coarse, 32)
    assert np.all(fine > 0.0)
    assert np.isclose(np.sum(fine), np.sum(coarse))


def _jacobians(mesh, uv):
    p0, p1, p2 = (uv[mesh.faces[:, i]] for i in range(3))
    return np.stack((p1 - p0, p2 - p0), axis=-1)
