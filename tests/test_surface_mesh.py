from pathlib import Path

import numpy as np
import pytest

from qcopt.surface_mesh import SurfaceMesh, load_common_refinement, load_obj


def _tetrahedron() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]
    )
    return SurfaceMesh(vertices, faces)


def _periodic_torus(nu: int = 5, nv: int = 4, x_offset: float = 0.0):
    vertices = []
    major, minor = 2.0, 0.55
    for i in range(nu):
        u = 2.0 * np.pi * i / nu
        for j in range(nv):
            v = 2.0 * np.pi * j / nv
            vertices.append(
                [
                    x_offset + (major + minor * np.cos(v)) * np.cos(u),
                    (major + minor * np.cos(v)) * np.sin(u),
                    minor * np.sin(v),
                ]
            )
    faces = []
    index = lambda i, j: (i % nu) * nv + (j % nv)
    for i in range(nu):
        for j in range(nv):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i + 1, j + 1), index(i, j + 1)
            faces.extend(((a, b, c), (a, c, d)))
    return np.asarray(vertices), np.asarray(faces, dtype=np.int64)


def _double_torus_combinatorics() -> SurfaceMesh:
    """Connected sum of two consistently triangulated tori."""

    va, fa = _periodic_torus(x_offset=-3.0)
    vb, fb = _periodic_torus(x_offset=3.0)
    removed_a = fa[0]
    removed_b = fb[0]
    fa = fa[1:]
    fb = fb[1:]

    vertices = np.vstack((va, vb))
    fb = fb + len(va)
    removed_b = removed_b + len(va)
    # Reverse the second boundary orientation when welding the two punctures.
    parent = np.arange(len(vertices))
    for source, target in zip(removed_b[[0, 2, 1]], removed_a):
        parent[source] = target

    remap = np.empty(len(vertices), dtype=np.int64)
    kept = []
    for old in range(len(vertices)):
        root = parent[old]
        if root == old:
            remap[old] = len(kept)
            kept.append(vertices[old])
    for old in range(len(vertices)):
        if parent[old] != old:
            remap[old] = remap[parent[old]]
    faces = remap[np.vstack((fa, fb))]
    return SurfaceMesh(np.asarray(kept), faces)


def test_closed_oriented_tetrahedron_topology_and_frames():
    mesh = _tetrahedron()
    report = mesh.topology

    assert report.closed
    assert report.connected
    assert report.oriented
    assert report.euler_characteristic == 2
    assert report.genus == 0
    assert np.all(mesh.face_neighbors >= 0)
    assert np.allclose(np.linalg.norm(mesh.face_normals, axis=1), 1.0)
    assert np.all(mesh.face_areas > 0.0)
    assert np.allclose(mesh.barycentric_gradients.sum(axis=1), 0.0, atol=1e-14)
    face = 2
    barycentric = np.array([0.17, 0.28, 0.55])
    point = mesh.point_from_barycentric(face, barycentric)
    assert np.allclose(mesh.barycentric_from_point(face, point), barycentric, atol=1e-14)


def test_surface_mesh_owns_immutable_geometry_and_cached_arrays():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = SurfaceMesh(vertices, faces)

    vertices[0, 0] = 7.0
    faces[0, 0] = 2
    assert mesh.vertices[0, 0] == 0.0
    assert np.array_equal(mesh.faces, [[0, 1, 2]])
    for array in (
        mesh.vertices,
        mesh.faces,
        mesh.face_areas,
        mesh.face_normals,
        mesh.face_frames,
        mesh.barycentric_gradients,
        mesh.face_neighbors,
        mesh.vertex_normals,
    ):
        assert not array.flags.writeable


def test_periodic_torus_and_connected_sum_have_expected_genus():
    torus = SurfaceMesh(*_periodic_torus())
    double_torus = _double_torus_combinatorics()

    assert torus.topology.euler_characteristic == 0
    assert torus.topology.genus == 1
    assert double_torus.topology.euler_characteristic == -2
    assert double_torus.topology.genus == 2
    assert double_torus.topology.closed
    assert double_torus.topology.connected
    assert double_torus.topology.oriented


def test_open_mesh_reports_boundary_and_no_closed_genus():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    mesh = SurfaceMesh(vertices, np.array([[0, 1, 2], [0, 2, 3]]))

    assert not mesh.topology.closed
    assert mesh.topology.boundary_edges == 4
    assert mesh.topology.genus is None


def test_nonmanifold_edge_is_rejected():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    with pytest.raises(ValueError, match="non-manifold"):
        SurfaceMesh(vertices, faces)


def test_isolated_vertex_is_rejected_instead_of_corrupting_euler_genus():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [9.0, 9.0, 9.0],
        ]
    )
    with pytest.raises(ValueError, match="isolated"):
        SurfaceMesh(vertices, np.array([[0, 1, 2]]))


def test_obj_loader_ignores_texture_indices_and_triangulates_polygon(tmp_path: Path):
    path = tmp_path / "textured_quad.obj"
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "vt 0.7 0.2",
                "vt 0.8 0.2",
                "f -4/2 -3/1 -2/2 -1/1",
            ]
        ),
        encoding="utf-8",
    )

    mesh = load_obj(path)
    assert mesh.n_vertices == 4
    assert mesh.n_faces == 2
    assert np.array_equal(mesh.faces, np.array([[0, 1, 2], [0, 2, 3]]))


def test_common_refinement_geometry_may_differ_but_connectivity_must_match():
    first = _tetrahedron()
    second = SurfaceMesh(first.vertices * np.array([2.0, 1.0, 0.5]), first.faces.copy())
    assert first.has_same_connectivity(second)

    changed_faces = second.faces.copy()
    changed_faces[0] = changed_faces[0, [0, 2, 1]]
    changed = SurfaceMesh(second.vertices, changed_faces)
    assert not first.has_same_connectivity(changed)


def test_paired_common_refinement_loader_repairs_obj_quantization(tmp_path: Path):
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    first_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # Quantized to a collinear first face.
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],  # Duplicate representation of vertex zero.
        ]
    )
    second_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    # Use the duplicate only in a face where merging restores the tetrahedron.
    represented_faces = faces.copy()
    represented_faces[1, 0] = 4

    def write(path, vertices):
        lines = ["v " + " ".join(map(str, vertex)) for vertex in vertices]
        lines += ["f " + " ".join(str(int(i) + 1) for i in face) for face in represented_faces]
        path.write_text("\n".join(lines), encoding="utf-8")

    first_path, second_path = tmp_path / "A.obj", tmp_path / "B.obj"
    write(first_path, first_vertices)
    write(second_path, second_vertices)
    common = load_common_refinement(first_path, second_path)

    assert common.source.has_same_connectivity(common.target)
    assert common.source.topology.genus == 0
    assert common.target.topology.genus == 0
    assert common.repair.merged_vertices == 1
    assert common.repair.perturbed_source_vertices >= 1
    assert common.repair.maximum_perturbation <= 5.1e-7
