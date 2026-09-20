import numpy as np

from qcopt.experiments.compatibility_boundary_theorem_audit import side_sliding_map
from qcopt.experiments.compatibility_rectangle_projection_audit import target_with_same_r2_boundary
from qcopt.beltrami import face_beltrami
from qcopt.forward.compatibility_projection import project_facewise_mu
from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def test_side_sliding_boundary_stays_on_rectangle_sides_and_is_ordered():
    mesh = structured_rectangle(8, 8)
    mapped = side_sliding_map(mesh.vertices)
    report = audit_injectivity(mesh, mapped, rectangle=True)
    assert report.certified
    assert report.rectangle_side_violations == ()
    assert not report.flipped_faces
    assert np.isfinite(mapped).all()


def test_fixed_r2_boundary_projection_recovers_small_manufactured_field():
    mesh = structured_rectangle(8, 8)
    boundary_map = side_sliding_map(mesh.vertices)
    target = target_with_same_r2_boundary(mesh.vertices)
    result = project_facewise_mu(
        mesh,
        face_beltrami(mesh, target),
        boundary_map=boundary_map,
        initial_map=boundary_map,
        max_nfev=5,
    )
    report = audit_injectivity(mesh, result.map, rectangle=True)
    assert result.success
    assert result.residual_norm < 1e-5
    assert report.certified
