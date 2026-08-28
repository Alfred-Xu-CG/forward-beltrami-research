import numpy as np

from qcopt.common_surface_map import sample_common_refinement
from qcopt.surface_mesh import CommonRefinement, CommonRefinementRepair, SurfaceMesh


def _tetrahedron(scale):
    vertices = scale * np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return SurfaceMesh(vertices, faces)


def test_common_refinement_samples_are_located_on_original_surfaces():
    source = _tetrahedron(1.0)
    target = _tetrahedron(2.0)
    common = CommonRefinement(
        source,
        target,
        CommonRefinementRepair(0, 0, 0, 0, 0.0, 1e-6),
    )
    sampled = sample_common_refinement(common, source, target, maximum_samples=4)

    assert len(sampled.source_points) == 4
    assert len(sampled.target_points) == 4
    assert sampled.maximum_source_projection_error < 1e-14
    assert sampled.maximum_target_projection_error < 1e-14
    for index, face in enumerate(sampled.overlay_faces):
        expected_source = source.point_from_barycentric(face, np.full(3, 1.0 / 3.0))
        expected_target = target.point_from_barycentric(face, np.full(3, 1.0 / 3.0))
        assert np.allclose(sampled.source_points[index].position(source), expected_source)
        assert np.allclose(sampled.target_points[index].position(target), expected_target)


def test_original_surface_genus_mismatch_is_rejected():
    source = _tetrahedron(1.0)
    target = _tetrahedron(2.0)
    common = CommonRefinement(
        source,
        target,
        CommonRefinementRepair(0, 0, 0, 0, 0.0, 1e-6),
    )
    open_mesh = SurfaceMesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]]),
    )
    try:
        sample_common_refinement(common, source, open_mesh, maximum_samples=2)
    except ValueError as error:
        assert "topology" in str(error) or "genus" in str(error)
    else:
        raise AssertionError("mismatched original topology must be rejected")
