"""Trace nonfinite gradients through the compiled-joint benchmark stages."""
from __future__ import annotations

import argparse
import json
import torch

from qcopt.neural_bijection.dense import (
    CoarsePatchFineVertexP1Layer, exact_dyadic_p1_refine,
)
from qcopt.neural_bijection.dense.multilevel_forward_p1 import (
    certify_p1_or_identity,
)


def stats(value: torch.Tensor | None) -> dict[str, object]:
    if value is None:
        return {"none": True}
    finite = torch.isfinite(value)
    good = value[finite]
    return {
        "shape": list(value.shape),
        "finite": int(finite.sum()),
        "elements": value.numel(),
        "nan": int(torch.isnan(value).sum()),
        "inf": int(torch.isinf(value).sum()),
        "finite_maxabs": float(good.abs().max()) if good.numel() else None,
    }


def trial(coarse_amplitude: float, fine_amplitude: float,
          compile_fine: bool, backend: str, fine_floor: float):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(91737)
    zc = (coarse_amplitude * torch.randn(
        1, 255, 255, 2, device=device, dtype=torch.float32,
        generator=generator)).requires_grad_()
    zf = (fine_amplitude * torch.randn(
        1, 1023, 1023, 2, device=device, dtype=torch.float32,
        generator=generator)).requires_grad_()
    model = CoarsePatchFineVertexP1Layer(
        257, 1025, coarse_patch_cells=16, compute_dtype=torch.float64,
        minimum_jacobian=.05, certify_output=True,
    ).to(device)
    if compile_fine:
        model.fine = torch.compile(model.fine, backend=backend)
    coarse = model.coarse(model.identity, zc.double())
    base = exact_dyadic_p1_refine(exact_dyadic_p1_refine(coarse))
    floor = base.new_full((1,), fine_floor / 1024 ** 2)
    fine = model.fine(base, zf.double(), area_floor=floor)
    certified, accepted = certify_p1_or_identity(
        fine, model.fine_identity)
    stages = {"coarse": coarse, "base": base, "fine": fine,
              "certified": certified}
    cotangent = torch.randn(
        1, 1025, 1025, 2, device=device, dtype=torch.float64,
        generator=generator)
    rows = []
    for name, output in stages.items():
        cot = (torch.ones_like(output) if name in ("coarse", "base")
               else cotangent)
        gc, gf = torch.autograd.grad(
            (output * cot).sum(), (zc, zf),
            retain_graph=True, allow_unused=True,
        )
        rows.append({"stage": name, "output": stats(output),
                     "coarse_grad": stats(gc), "fine_grad": stats(gf)})
    return {"coarse_amplitude": coarse_amplitude,
            "fine_amplitude": fine_amplitude,
            "compile_fine": compile_fine,
            "backend": backend,
            "fine_floor": fine_floor,
            "accepted": bool(accepted),
            "stages": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--case", choices=["all", "both"], default="all")
    parser.add_argument("--coarse-amplitude", type=float, default=.1)
    parser.add_argument("--fine-amplitude", type=float, default=.1)
    parser.add_argument("--fine-floor", type=float, default=.05)
    args = parser.parse_args()
    cases = ([(args.coarse_amplitude, args.fine_amplitude)] if args.case == "both" else
             [(0., 0.), (.1, 0.), (0., .1), (.1, .1)])
    for zc, zf in cases:
        for compile_fine in (False, True):
            print(json.dumps(trial(zc, zf, compile_fine, args.backend,
                                   args.fine_floor),
                             sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
