"""Reference normalized Beltrami-holomorphic variation operator.

The formula implemented here is the normalized first variation at the
conformal base map, with ``f(0)=0``, ``f(1)=1``, and ``f(infinity)=infinity``.
It is a direct blocked quadrature reference and intentionally does not claim to
be the full nonlinear BHF flow for an arbitrary base coefficient.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def bhf_kernel_R(source_image: np.ndarray, evaluation_image: complex) -> np.ndarray:
    """Return the normalized BHF kernel ``R(F(z),F(w))``.

    The formula is the sphere/whole-plane variation kernel from the BHF
    theorem.  It is written as three simple-pole terms instead of combining
    them algebraically, because the separate terms are also the form used by
    the piecewise-affine triangle implementation in the reference paper.
    ``source_image`` may be a scalar or an array.
    """

    source_image = np.asarray(source_image, dtype=np.complex128)
    w = np.complex128(evaluation_image)
    return 1.0 / (source_image - w) - w / (source_image - 1.0) + (w - 1.0) / source_image


def arbitrary_base_bhf_variation(
    evaluation_points: np.ndarray,
    source_points: np.ndarray,
    evaluation_image: np.ndarray,
    source_image: np.ndarray,
    fz_source: np.ndarray,
    variation: np.ndarray,
    weights: np.ndarray,
    *,
    block_size: int = 256,
    collision_tolerance: float = 1e-14,
) -> np.ndarray:
    """Evaluate the arbitrary-base BHF first variation by blocked quadrature.

    This is a faithful reference discretization of

    ``-pi**-1 integral nu(z) R(f(z), f(w)) f_z(z)**2 dx dy``.

    ``source_image`` and ``evaluation_image`` are the current map values,
    while ``fz_source`` is the current Wirtinger derivative on source cells.
    The implementation is intentionally direct (``O(N_source*N_eval)``).  If
    an evaluation point coincides with a source image, the principal-value
    diagonal term is omitted; this makes the singular treatment explicit and
    keeps this routine a diagnostic/reference operator, not a production PV
    quadrature.  The regularizing terms still contribute on that diagonal.
    """

    z = np.asarray(evaluation_points, dtype=np.complex128)
    s = np.asarray(source_points, dtype=np.complex128)
    f_eval = np.asarray(evaluation_image, dtype=np.complex128)
    f_source = np.asarray(source_image, dtype=np.complex128)
    fz = np.asarray(fz_source, dtype=np.complex128)
    nu = np.asarray(variation, dtype=np.complex128)
    area = np.asarray(weights, dtype=np.float64)
    if z.ndim != 1 or s.ndim != 1 or f_eval.shape != z.shape:
        raise ValueError("evaluation points and images must be compatible 1D arrays")
    if any(x.shape != s.shape for x in (f_source, fz, nu, area)):
        raise ValueError("source arrays must have compatible 1D shapes")
    if not all(np.all(np.isfinite(x)) for x in (z, s, f_eval, f_source, fz, nu, area)):
        raise ValueError("all inputs must be finite")
    if np.any(area <= 0.0) or block_size < 1 or collision_tolerance < 0.0:
        raise ValueError("weights must be positive and options must be valid")
    # The normalization poles are removable in the full kernel but should not
    # be sampled as source images by a point quadrature.
    if np.any(np.abs(f_source) < collision_tolerance) or np.any(
        np.abs(f_source - 1.0) < collision_tolerance
    ):
        raise ValueError("source image points must avoid normalized poles 0 and 1")

    weighted = nu * (fz**2) * area
    result = np.zeros(z.shape, dtype=np.complex128)
    for start in range(0, z.size, block_size):
        stop = min(start + block_size, z.size)
        block_w = f_eval[start:stop]
        delta = f_source[None, :] - block_w[:, None]
        singular = np.abs(delta) <= collision_tolerance
        safe_delta = np.where(singular, 1.0 + 0.0j, delta)
        kernel = 1.0 / safe_delta - block_w[:, None] / (f_source[None, :] - 1.0)
        kernel += (block_w[:, None] - 1.0) / f_source[None, :]
        kernel[singular] = (
            -block_w[:, None] / (f_source[None, :] - 1.0)
            + (block_w[:, None] - 1.0) / f_source[None, :]
        )[singular]
        result[start:stop] = -(kernel @ weighted) / np.pi
    # The normalized map fixes 0 and 1 exactly.  Enforce this in floating point
    # even if a caller supplied slightly perturbed image coordinates.
    result[np.abs(f_eval) <= collision_tolerance] = 0.0
    result[np.abs(f_eval - 1.0) <= collision_tolerance] = 0.0
    return np.ascontiguousarray(result)


def triangle_bhf_variation(
    evaluation_points: np.ndarray,
    evaluation_image: np.ndarray,
    triangles: np.ndarray,
    triangle_image: np.ndarray,
    fz_triangle: np.ndarray,
    variation_triangle: np.ndarray,
    *,
    block_size: int = 128,
) -> np.ndarray:
    """Evaluate BHF after fixed degree-five quadrature on affine triangles.

    ``triangles`` and ``triangle_image`` have shape ``(n_tri, 3)`` and contain
    the source and current-map vertices.  The seven-point Dunavant rule turns
    each face into quadrature source/image samples, so an evaluation point that
    lies in a face is no longer represented by a single omitted point sample.
    The remaining global operator is the same direct arbitrary-base kernel and
    therefore still has quadratic pair cost; this is a stable mesh-native
    reference, not a fast production implementation.
    """

    tri = np.asarray(triangles, dtype=np.complex128)
    tri_image = np.asarray(triangle_image, dtype=np.complex128)
    fz = np.asarray(fz_triangle, dtype=np.complex128)
    nu = np.asarray(variation_triangle, dtype=np.complex128)
    if tri.ndim != 2 or tri.shape[1:] != (3,) or tri_image.shape != tri.shape:
        raise ValueError("triangles must have shape (n_tri, 3)")
    if fz.shape != (tri.shape[0],) or nu.shape != fz.shape:
        raise ValueError("face data must have one value per triangle")
    # Degree-five Dunavant rule on the reference triangle; weights sum to 1/2.
    bary = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [0.05971587178977, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.05971587178977, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.05971587178977],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ],
        dtype=np.float64,
    )
    ref_weights = np.array(
        [0.1125, 0.066197076394253, 0.066197076394253, 0.066197076394253,
         0.0629695902724135, 0.0629695902724135, 0.0629695902724135],
        dtype=np.float64,
    )
    source = np.einsum("qa,ta->tq", bary, tri).reshape(-1)
    source_image = np.einsum("qa,ta->tq", bary, tri_image).reshape(-1)
    jac = np.abs(
        np.imag((tri[:, 1] - tri[:, 0]) * np.conjugate(tri[:, 2] - tri[:, 0]))
    )
    weights = (jac[:, None] * ref_weights[None, :]).reshape(-1)
    return arbitrary_base_bhf_variation(
        evaluation_points,
        source,
        evaluation_image,
        source_image,
        np.repeat(fz, bary.shape[0]),
        np.repeat(nu, bary.shape[0]),
        weights,
        block_size=block_size,
    )


def duffy_triangle_kernel_integral(
    source_triangle: np.ndarray,
    image_triangle: np.ndarray,
    evaluation_image: complex,
    *,
    vertex: int = 0,
    order: int = 16,
) -> complex:
    """Integrate the BHF kernel over one affine triangle with a Duffy map.

    The selected source/image vertex is collapsed by the Duffy coordinates, so
    the Jacobian cancels the simple ``1/(f(z)-f(w))`` pole when the evaluation
    image equals that vertex. This is a local singular-quadrature primitive;
    assembling it over all triangles still requires a mesh-aware near-field
    strategy.
    """

    source = np.asarray(source_triangle, dtype=np.complex128)
    image = np.asarray(image_triangle, dtype=np.complex128)
    if source.shape != (3,) or image.shape != (3,):
        raise ValueError("triangles must contain three complex vertices")
    if vertex not in (0, 1, 2) or order < 2:
        raise ValueError("vertex must be 0, 1, or 2 and order at least two")
    order_vertices = [vertex, (vertex + 1) % 3, (vertex + 2) % 3]
    p0, p1, p2 = source[order_vertices]
    f0, f1, f2 = image[order_vertices]
    jacobian = abs(np.imag((p1 - p0) * np.conjugate(p2 - p0)))
    nodes, weights = leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    total = 0.0 + 0.0j
    for si, ws in zip(nodes, weights):
        for ti, wt in zip(nodes, weights):
            source_point = p0 + si * ((1.0 - ti) * (p1 - p0) + ti * (p2 - p0))
            image_point = f0 + si * ((1.0 - ti) * (f1 - f0) + ti * (f2 - f0))
            kernel = bhf_kernel_R(np.asarray([image_point]), evaluation_image)[0]
            total += ws * wt * jacobian * si * kernel
    return complex(total)


def vertex_near_far_bhf_variation(
    source_vertices: np.ndarray,
    image_vertices: np.ndarray,
    faces: np.ndarray,
    fz_face: np.ndarray,
    variation_face: np.ndarray,
    target_vertex: int,
    *,
    near_order: int = 16,
    far_block_size: int = 128,
) -> complex:
    """Assemble one BHF vertex velocity with Duffy near-field correction.

    All faces first use the stable degree-five quadrature.  Faces incident to
    ``target_vertex`` are then replaced by a Duffy integral whose collapsed
    vertex removes the simple pole.  This is a concrete near/far assembly
    primitive: it avoids applying a singular rule to every far face while
    retaining a convergent local treatment at the target vertex.
    """

    source_vertices = np.asarray(source_vertices, dtype=np.complex128)
    image_vertices = np.asarray(image_vertices, dtype=np.complex128)
    faces = np.asarray(faces, dtype=np.int64)
    fz = np.asarray(fz_face, dtype=np.complex128)
    nu = np.asarray(variation_face, dtype=np.complex128)
    if source_vertices.ndim != 1 or image_vertices.shape != source_vertices.shape:
        raise ValueError("source_vertices and image_vertices must be matching 1D arrays")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= source_vertices.size):
        raise ValueError("faces must be an in-range (n_faces,3) array")
    if fz.shape != (faces.shape[0],) or nu.shape != fz.shape:
        raise ValueError("face data must have one value per face")
    if target_vertex < 0 or target_vertex >= source_vertices.size:
        raise ValueError("target_vertex is out of range")
    if near_order < 2 or far_block_size < 1:
        raise ValueError("near_order must be at least two and far_block_size positive")
    triangles = source_vertices[faces]
    image_triangles = image_vertices[faces]
    evaluation_points = np.asarray([source_vertices[target_vertex]])
    evaluation_image = np.asarray([image_vertices[target_vertex]])
    far_and_near = triangle_bhf_variation(
        evaluation_points,
        evaluation_image,
        triangles,
        image_triangles,
        fz,
        nu,
        block_size=far_block_size,
    )[0]
    corrected = far_and_near
    incident = np.flatnonzero(np.any(faces == target_vertex, axis=1))
    for face_index in incident.tolist():
        local = np.flatnonzero(faces[face_index] == target_vertex)
        vertex = int(local[0])
        regular = triangle_bhf_variation(
            evaluation_points,
            evaluation_image,
            triangles[face_index : face_index + 1],
            image_triangles[face_index : face_index + 1],
            fz[face_index : face_index + 1],
            nu[face_index : face_index + 1],
            block_size=1,
        )[0]
        duffy = -nu[face_index] * fz[face_index] ** 2 / np.pi * duffy_triangle_kernel_integral(
            triangles[face_index],
            image_triangles[face_index],
            image_vertices[target_vertex],
            vertex=vertex,
            order=near_order,
        )
        corrected += duffy - regular
    return complex(corrected)


def all_vertex_near_far_bhf_variation(
    source_vertices: np.ndarray,
    image_vertices: np.ndarray,
    faces: np.ndarray,
    fz_face: np.ndarray,
    variation_face: np.ndarray,
    *,
    near_order: int = 16,
    far_block_size: int = 128,
) -> np.ndarray:
    """Vectorized far-field plus local Duffy corrections for all vertices."""

    source_vertices = np.asarray(source_vertices, dtype=np.complex128)
    image_vertices = np.asarray(image_vertices, dtype=np.complex128)
    faces = np.asarray(faces, dtype=np.int64)
    fz = np.asarray(fz_face, dtype=np.complex128)
    nu = np.asarray(variation_face, dtype=np.complex128)
    if source_vertices.ndim != 1 or image_vertices.shape != source_vertices.shape:
        raise ValueError("source_vertices and image_vertices must be matching 1D arrays")
    triangles = source_vertices[faces]
    image_triangles = image_vertices[faces]
    corrected = triangle_bhf_variation(
        source_vertices,
        image_vertices,
        triangles,
        image_triangles,
        fz,
        nu,
        block_size=far_block_size,
    )
    incident: list[list[int]] = [[] for _ in range(source_vertices.size)]
    for face_index, face in enumerate(faces.tolist()):
        for vertex in face:
            incident[vertex].append(face_index)
    for target_vertex, face_indices in enumerate(incident):
        for face_index in face_indices:
            local = np.flatnonzero(faces[face_index] == target_vertex)
            vertex = int(local[0])
            regular = triangle_bhf_variation(
                np.asarray([source_vertices[target_vertex]]),
                np.asarray([image_vertices[target_vertex]]),
                triangles[face_index : face_index + 1],
                image_triangles[face_index : face_index + 1],
                fz[face_index : face_index + 1],
                nu[face_index : face_index + 1],
                block_size=1,
            )[0]
            duffy = -nu[face_index] * fz[face_index] ** 2 / np.pi * duffy_triangle_kernel_integral(
                triangles[face_index],
                image_triangles[face_index],
                image_vertices[target_vertex],
                vertex=vertex,
                order=near_order,
            )
            corrected[target_vertex] += duffy - regular
    return np.ascontiguousarray(corrected)


def interior_point_near_far_bhf_variation(
    source_vertices: np.ndarray,
    image_vertices: np.ndarray,
    faces: np.ndarray,
    fz_face: np.ndarray,
    variation_face: np.ndarray,
    target_face: int,
    barycentric: np.ndarray,
    *,
    near_order: int = 16,
    far_block_size: int = 128,
) -> complex:
    """Near/far BHF assembly for a target strictly inside one source face.

    The target face is split into three affine subtriangles with the target as
    their first vertex; Duffy quadrature is then applied to those subtriangles.
    All other faces use the regular degree-five rule.  This extends the vertex
    control to arbitrary interior evaluation points while deliberately leaving
    edge/vertex multi-face principal-value cancellation as a separate case.
    """

    source_vertices = np.asarray(source_vertices, dtype=np.complex128)
    image_vertices = np.asarray(image_vertices, dtype=np.complex128)
    faces = np.asarray(faces, dtype=np.int64)
    fz = np.asarray(fz_face, dtype=np.complex128)
    nu = np.asarray(variation_face, dtype=np.complex128)
    bary = np.asarray(barycentric, dtype=np.float64)
    if bary.shape != (3,) or np.any(bary <= 0.0) or not np.isclose(np.sum(bary), 1.0):
        raise ValueError("barycentric coordinates must be strictly interior and sum to one")
    if target_face < 0 or target_face >= faces.shape[0]:
        raise ValueError("target_face is out of range")
    triangles = source_vertices[faces]
    image_triangles = image_vertices[faces]
    source_target = bary @ triangles[target_face]
    image_target = bary @ image_triangles[target_face]
    evaluation = np.asarray([source_target])
    evaluation_image = np.asarray([image_target])
    regular = triangle_bhf_variation(
        evaluation,
        evaluation_image,
        triangles,
        image_triangles,
        fz,
        nu,
        block_size=far_block_size,
    )[0]
    face = triangles[target_face]
    image_face = image_triangles[target_face]
    regular_target = triangle_bhf_variation(
        evaluation,
        evaluation_image,
        face[None, :],
        image_face[None, :],
        fz[target_face : target_face + 1],
        nu[target_face : target_face + 1],
        block_size=1,
    )[0]
    corrected = regular - regular_target
    duffy_total = 0.0 + 0.0j
    for local in range(3):
        nxt = (local + 1) % 3
        source_subtriangle = np.asarray([source_target, face[local], face[nxt]])
        image_subtriangle = np.asarray([image_target, image_face[local], image_face[nxt]])
        duffy_total += duffy_triangle_kernel_integral(
            source_subtriangle,
            image_subtriangle,
            image_target,
            vertex=0,
            order=near_order,
        )
    corrected += -nu[target_face] * fz[target_face] ** 2 / np.pi * duffy_total
    return complex(corrected)


def edge_point_near_far_bhf_variation(
    source_vertices: np.ndarray,
    image_vertices: np.ndarray,
    faces: np.ndarray,
    fz_face: np.ndarray,
    variation_face: np.ndarray,
    edge_vertices: tuple[int, int],
    edge_parameter: float,
    *,
    near_order: int = 16,
    far_block_size: int = 128,
) -> complex:
    """Assemble a BHF velocity at a point in a mesh edge.

    Every incident triangle is split into two target-centered subtriangles;
    Duffy quadrature is used only on those local pieces and the regular
    degree-five contribution is retained on all far faces.  For an interior
    edge this treats both sides of the shared edge with the same image target,
    which is the required local cancellation/control before a global PV rule.
    """

    source_vertices = np.asarray(source_vertices, dtype=np.complex128)
    image_vertices = np.asarray(image_vertices, dtype=np.complex128)
    faces = np.asarray(faces, dtype=np.int64)
    fz = np.asarray(fz_face, dtype=np.complex128)
    nu = np.asarray(variation_face, dtype=np.complex128)
    a, b = (int(edge_vertices[0]), int(edge_vertices[1]))
    if a == b or a < 0 or b < 0 or a >= source_vertices.size or b >= source_vertices.size:
        raise ValueError("edge_vertices must contain two distinct in-range vertices")
    if not (0.0 < edge_parameter < 1.0) or near_order < 2 or far_block_size < 1:
        raise ValueError("edge_parameter must lie in (0,1) and quadrature options must be positive")
    if image_vertices.shape != source_vertices.shape or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("source/image vertices and faces have invalid shapes")
    if fz.shape != (faces.shape[0],) or nu.shape != fz.shape:
        raise ValueError("face data must have one value per face")
    incident = np.flatnonzero(np.sum(np.isin(faces, [a, b]), axis=1) == 2)
    if incident.size == 0:
        raise ValueError("edge_vertices do not form a mesh edge")
    source_target = (1.0 - edge_parameter) * source_vertices[a] + edge_parameter * source_vertices[b]
    image_target = (1.0 - edge_parameter) * image_vertices[a] + edge_parameter * image_vertices[b]
    triangles = source_vertices[faces]
    image_triangles = image_vertices[faces]
    evaluation = np.asarray([source_target])
    evaluation_image = np.asarray([image_target])
    corrected = triangle_bhf_variation(
        evaluation, evaluation_image, triangles, image_triangles, fz, nu, block_size=far_block_size
    )[0]
    for face_index in incident.tolist():
        face = faces[face_index]
        third_candidates = [int(v) for v in face.tolist() if v not in (a, b)]
        if len(third_candidates) != 1:
            raise ValueError("edge_vertices must occur as an edge of every incident face")
        c = third_candidates[0]
        source_subtriangles = (
            np.asarray([source_target, source_vertices[a], source_vertices[c]]),
            np.asarray([source_target, source_vertices[c], source_vertices[b]]),
        )
        image_subtriangles = (
            np.asarray([image_target, image_vertices[a], image_vertices[c]]),
            np.asarray([image_target, image_vertices[c], image_vertices[b]]),
        )
        regular = triangle_bhf_variation(
            evaluation, evaluation_image, triangles[face_index : face_index + 1],
            image_triangles[face_index : face_index + 1], fz[face_index : face_index + 1],
            nu[face_index : face_index + 1], block_size=1
        )[0]
        duffy = sum(
            duffy_triangle_kernel_integral(st, it, image_target, vertex=0, order=near_order)
            for st, it in zip(source_subtriangles, image_subtriangles)
        )
        corrected += -nu[face_index] * fz[face_index] ** 2 / np.pi * duffy - regular
    return complex(corrected)


def normalized_bhf_variation(
    evaluation_points: np.ndarray,
    source_points: np.ndarray,
    variation: np.ndarray,
    weights: np.ndarray,
    *,
    block_size: int = 256,
) -> np.ndarray:
    """Evaluate the normalized first variation by blocked Cauchy quadrature."""

    z = np.asarray(evaluation_points, dtype=np.complex128)
    w = np.asarray(source_points, dtype=np.complex128)
    nu = np.asarray(variation, dtype=np.complex128)
    area = np.asarray(weights, dtype=np.float64)
    if z.ndim != 1 or w.ndim != 1 or nu.shape != w.shape or area.shape != w.shape:
        raise ValueError("points, variation, and weights must have compatible 1D shapes")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(w)) or not np.all(np.isfinite(nu)):
        raise ValueError("inputs must be finite")
    if np.any(area <= 0.0) or block_size < 1:
        raise ValueError("weights must be positive and block_size must be positive")
    if np.any(np.abs(w) < 1e-14) or np.any(np.abs(w - 1.0) < 1e-14):
        raise ValueError("source points must avoid normalization points 0 and 1")
    weighted = nu * area / (w * (w - 1.0))
    result = np.zeros_like(z)
    for start in range(0, len(z), block_size):
        stop = min(start + block_size, len(z))
        block = z[start:stop]
        # The BHF theorem uses the first kernel term ``1/(f(z)-f(w))``.
        # Keep that orientation here; using ``w-z`` silently flips the flow.
        delta = w[None, :] - block[:, None]
        kernel = np.zeros_like(delta)
        nonzero = np.abs(delta) > 1e-14
        kernel[nonzero] = 1.0 / delta[nonzero]
        integral = kernel @ weighted
        result[start:stop] = -(block * (block - 1.0) / np.pi) * integral
    result[np.abs(z) < 1e-14] = 0.0
    result[np.abs(z - 1.0) < 1e-14] = 0.0
    return np.ascontiguousarray(result)
