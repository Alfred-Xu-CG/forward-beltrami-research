"""Measure mesh-free versus mesh-validated fixed structured query setup."""
from __future__ import annotations

import argparse
import json
import time

import torch

from qcopt.mesh import structured_rectangle
from qcopt.neural_bijection.tutte.dense_warp import StructuredDenseQueryTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-side", type=int, default=1025)
    parser.add_argument("--query-side", type=int, default=512)
    parser.add_argument("--compare-mesh", action="store_true")
    args = parser.parse_args()
    if min(args.control_side, args.query_side) < 2:
        raise ValueError("sides must be at least two")
    cell_count = args.control_side - 1
    began = time.perf_counter()
    direct = StructuredDenseQueryTable.from_shape(
        cell_count, cell_count,
        height=args.query_side, width=args.query_side,
    )
    shape_seconds = time.perf_counter() - began
    result = dict(
        control_side=args.control_side,
        query_side=args.query_side,
        control_vertices=direct.control_vertices,
        control_faces=2 * cell_count ** 2,
        shape_factory_seconds=shape_seconds,
        mesh_factory_seconds=None,
        identical_tables=None,
    )
    if args.compare_mesh:
        began = time.perf_counter()
        mesh = structured_rectangle(cell_count, cell_count)
        checked = StructuredDenseQueryTable.from_mesh(
            mesh, height=args.query_side, width=args.query_side,
        )
        result["mesh_factory_seconds"] = time.perf_counter() - began
        result["identical_tables"] = bool(
            torch.equal(direct._vertex_indices_cpu, checked._vertex_indices_cpu)
            and torch.equal(direct._barycentric_cpu, checked._barycentric_cpu)
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
