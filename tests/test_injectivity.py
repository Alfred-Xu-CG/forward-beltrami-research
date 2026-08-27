import numpy as np

from qcopt.injectivity import audit_injectivity
from qcopt.mesh import structured_rectangle


def test_identity_rectangle_receives_full_numerical_certificate():
    mesh = structured_rectangle(3, 2)
    report = audit_injectivity(mesh, mesh.vertices, rectangle=True)

    assert report.certified
    assert report.flipped_faces == ()
    assert report.boundary_intersections == ()
    assert report.bad_branch_vertices == ()
    assert report.rectangle_side_violations == ()
    assert np.isclose(report.minimum_signed_area_ratio, 1.0)


def test_single_face_flip_is_reported():
    mesh = structured_rectangle(1, 1)
    uv = mesh.vertices.copy()
    uv[3] = np.array([-0.25, 0.25])
    report = audit_injectivity(mesh, uv)

    assert not report.certified
    assert len(report.flipped_faces) >= 1
    assert report.minimum_signed_area_ratio < 0.0


def test_bow_tie_boundary_intersection_is_reported():
    mesh = structured_rectangle(1, 1)
    uv = mesh.vertices.copy()
    uv[0] = [0.0, 0.0]
    uv[1] = [1.0, 1.0]
    uv[3] = [0.0, 1.0]
    uv[2] = [1.0, 0.0]
    report = audit_injectivity(mesh, uv)

    assert not report.certified
    assert report.boundary_intersections


def test_collapsed_interior_one_ring_is_not_unit_branch_index():
    mesh = structured_rectangle(2, 2)
    uv = mesh.vertices.copy()
    center = 4
    uv[center] = uv[0]
    report = audit_injectivity(mesh, uv)

    assert not report.certified
    assert center in report.bad_branch_vertices


def test_rectangle_audit_rejects_tangential_side_order_reversal():
    mesh = structured_rectangle(2, 2)
    uv = mesh.vertices.copy()
    uv[1, 0] = 1.1
    report = audit_injectivity(mesh, uv, rectangle=True)

    assert not report.certified
    assert "bottom-order" in report.rectangle_side_violations
