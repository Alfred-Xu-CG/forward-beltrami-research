"""RQ III-C/D: what does a full Whitney Hodge preserve beyond P1 potentials?

Run in a fresh CPU process; on Windows Conda set MKL_THREADING_LAYER=SEQUENTIAL
before Python starts. N means vertices per axis, not image/query resolution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qcopt.mesh import structured_rectangle
from qcopt.forward.whitney_hodge import WhitneyHodge
from qcopt.forward.discrete_conjugacy import assemble_facewise_conductivity, integrate_stream_from_face_flux, p1_conjugacy_residual


def git_state():
    """Record the ordinary Git provenance of the code used for a receipt."""
    commit = subprocess.run(
        ("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "-C", str(ROOT), "status", "--porcelain"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def manufactured(xy, a):
    k = np.array([1., 0.])
    ell = np.array([-a[1, 0], a[0, 0]])
    s, t = xy @ k, xy @ ell
    u = np.exp(s)*np.cos(t)
    v = np.exp(s)*np.sin(t)
    gu = np.exp(s)[:, None]*(np.cos(t)[:, None]*k-np.sin(t)[:, None]*ell)
    return u, v, gu


def run(n, a):
    mesh = structured_rectangle(n-1, n-1)
    p, f = mesh.vertices, mesh.faces
    b = mesh.boundary_loops[0]
    interior = np.setdiff1d(np.arange(len(p)), b)
    s = WhitneyHodge(p, f)
    aa = np.repeat(a[None], len(f), axis=0)
    tensor = torch.tensor(aa, requires_grad=True)
    exact_u, exact_v, _ = manufactured(p, a)
    bv = torch.tensor(np.column_stack((exact_u[b], exact_v[b])), requires_grad=True)
    start = perf_counter()
    uv = s.solve(tensor, b, bv)
    forward = perf_counter()-start
    start = perf_counter()
    ga, gb = torch.autograd.grad(uv.square().mean(), (tensor, bv))
    backward = perf_counter()-start
    values = uv.detach().numpy()
    u, v = values.T
    reference = assemble_facewise_conductivity(mesh, aa)
    ref_uv = np.column_stack((exact_u, exact_v))
    ref_uv[interior] = sla.spsolve(reference[interior][:, interior], -reference[interior][:, b] @ ref_uv[b])
    # Explicit independent sparse incidence/Hodge product.
    ht = s.hodge(tensor.detach()).coalesce()
    h = sp.coo_matrix((ht.values().numpy(), ht.indices().numpy()), shape=ht.shape).tocsr()
    k_whitney = s.b0_scipy.T @ h @ s.b0_scipy
    q = s.flux_cochain(tensor.detach(), uv[:, 0].detach())
    dual, dual_res = s.dual_stream(q)
    legacy = integrate_stream_from_face_flux(mesh, u, aa)
    grad_u = np.einsum("ti,tij->tj", u[f], mesh.gradients)
    grad_v = np.einsum("ti,tij->tj", v[f], mesh.gradients)
    flux = np.einsum("tij,tj->ti", aa, grad_u)
    jflux = np.column_stack((-flux[:, 1], flux[:, 0]))
    conjugacy = np.sqrt(np.sum(mesh.areas*np.sum((grad_v-jflux)**2, axis=1)))
    centers = p[f].mean(axis=1)
    _, center_v, _ = manufactured(centers, a)
    # Compare streams modulo one best constant; do not give face values nodal semantics.
    delta = dual-center_v
    delta -= np.sum(mesh.areas*delta)/np.sum(mesh.areas)
    # Geometric normal jumps of piecewise-constant A grad(u), NOT dual cochain divergence.
    b1 = s.b1_scipy.tocsc()
    integrated_jumps, jumps = [], []
    for e in s.interior_edges:
        adj = b1.indices[b1.indptr[e]:b1.indptr[e+1]]
        tangent = p[s.edges[e, 1]]-p[s.edges[e, 0]]
        normal = np.array([-tangent[1], tangent[0]])
        jump = abs((flux[adj[0]]-flux[adj[1]]) @ normal)
        integrated_jumps.append(jump)
        jumps.append(jump/np.linalg.norm(tangent))
    # Three-point quadrature of potential and gradient error against analytic field.
    lam = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    qp = np.einsum("qi,tij->tqj", lam, p[f])
    qu, _, qg = manufactured(qp.reshape(-1, 2), a)
    uh = np.einsum("qi,ti->tq", lam, u[f])
    l2 = np.sqrt(np.sum(mesh.areas[:, None]*(uh-qu.reshape(-1, 3))**2)/3)
    h1 = np.sqrt(np.sum(mesh.areas[:, None, None]*(grad_u[:, None]-qg.reshape(-1, 3, 2))**2)/3)
    energy_local = np.sum(mesh.areas*np.einsum("ti,tij,tj->t", grad_u, aa, grad_u))
    energy_hodge = float((s.b0_scipy @ u) @ (h @ (s.b0_scipy @ u)))
    torch.manual_seed(n)
    direction = torch.randn_like(tensor)
    direction = (direction+direction.transpose(-1, -2))/2
    db = torch.randn_like(bv)
    eps = 1e-6
    with torch.no_grad():
        fd = (s.solve(tensor+eps*direction, b, bv+eps*db).square().mean()-s.solve(tensor-eps*direction, b, bv-eps*db).square().mean())/(2*eps)
    vjp = (ga*direction).sum()+(gb*db).sum()
    difference = k_whitney-reference
    factor = sla.splu(reference[interior][:, interior].tocsc())
    def sparse_bytes(m):
        return int(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes)
    return {
        "n_vertices_per_axis": n, "vertices": len(p), "faces": len(f), "edges": len(s.edges),
        "p1_operator_relative_difference": float(sla.norm(difference)/sla.norm(reference)),
        "p1_solution_max_difference": float(np.max(np.abs(values-ref_uv))),
        "nodal_l2_error_u": float(l2), "gradient_l2_error_u": float(h1),
        "conjugacy_l2_whitney_and_p1": float(conjugacy),
        "conjugacy_max_whitney_and_p1": p1_conjugacy_residual(mesh, u, v, aa),
        "legacy_integrated_stream_conjugacy_max": p1_conjugacy_residual(mesh, u, legacy.values, aa),
        "primal_potential_curl_max": float(np.max(np.abs(s.b1_scipy @ (s.b0_scipy @ u)))),
        "weak_vertex_flux_balance_max": float(np.max(np.abs((s.b0_scipy.T @ q.numpy())[interior]))),
        "p1_weak_vertex_flux_balance_max": float(np.max(np.abs((reference @ u)[interior]))),
        "face_normal_flux_jump_max": float(max(jumps)),
        "integrated_face_normal_flux_jump_max": float(max(integrated_jumps)),
        "dual_stream_relative_residual": float(dual_res),
        "dual_stream_face_center_l2_mod_constant": float(np.sqrt(np.sum(mesh.areas*delta**2))),
        "legacy_primal_stream_edge_inconsistency": legacy.edge_inconsistency,
        "legacy_primal_stream_cycle_residual": legacy.cycle_residual,
        "energy_relative_error": float(abs(energy_hodge-energy_local)/energy_local),
        "implicit_vjp_fd_absolute_error": float(abs(fd-vjp)),
        "forward_seconds_cold": forward, "backward_seconds": backward,
        "hodge_csr_bytes": sparse_bytes(h), "stiffness_csr_bytes": sparse_bytes(reference),
        "lu_factor_sparse_bytes": sparse_bytes(factor.L)+sparse_bytes(factor.U),
        "whitney_quadrature_bytes": s.whitney_quadrature.numel()*8,
        "topology_guarantee": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 9, 17, 25])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = {"isotropic": np.eye(2), "anisotropic_det1": np.array([[2., .7], [.7, .745]])}
    results = []
    for name, a in cases.items():
        for n in args.sizes:
            row = {"case": name, "tensor": a.tolist(), **run(n, a)}
            results.append(row)
            print(json.dumps(row), flush=True)
    payload = {"research_question": "III-C/D: Whitney energy/dual exactness versus ordinary P1 and primal stream integration", "utc": datetime.now(timezone.utc).isoformat(), "git": git_state(), "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "device": "CPU float64", "MKL_THREADING_LAYER": os.environ.get("MKL_THREADING_LAYER"), "torch_threads": torch.get_num_threads(), "timing_scope": "single cold forward with assembly/factorization; backward reuses factor; diagnostics excluded", "memory_scope": "explicit arrays/factors only; not process peak nor allocator peak", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
