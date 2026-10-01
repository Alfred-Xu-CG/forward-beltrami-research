"""Independent Section 20 arithmetic check; no registration or production edits.

Reproduces 24 CPU comparisons using explicit integer-cell barycentric refinement,
independent of the repository's refinement/query samplers. JSON is numerical
evidence, not a proof, speed benchmark, or a bitwise-equivalence claim.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from tools.coordinated_real_case import corner_symmetric_dirichlet
from tools.digital_mind_safe_optimize import strain_penalty


def identity(n):
    y, x = torch.meshgrid(torch.arange(n + 1) / n,
                          torch.arange(n + 1) / n, indexing="ij")
    return torch.stack((x, y), -1)[None]


def prolong(vertices, factor, diagonal):
    """Direct barycentric construction at integer local coordinates."""
    n = vertices.shape[1] - 1
    output = []
    for i in range(n * factor + 1):
        row = []
        for j in range(n * factor + 1):
            ci, cj = min(i // factor, n - 1), min(j // factor, n - 1)
            u, v = (j - cj * factor) / factor, (i - ci * factor) / factor
            a, b = vertices[:, ci, cj], vertices[:, ci, cj + 1]
            d, c = vertices[:, ci + 1, cj], vertices[:, ci + 1, cj + 1]
            if diagonal == "ac":
                value = ((1-u)*a + (u-v)*b + v*c if u >= v
                         else (1-v)*a + u*c + (v-u)*d)
            else:
                value = ((1-u-v)*a + u*b + v*d if u+v <= 1
                         else (1-v)*b + (u+v-1)*c + (1-u)*d)
            row.append(value)
        output.append(torch.stack(row, 1))
    return torch.stack(output, 1)


def phis(vertices):
    n = vertices.shape[1] - 1
    a, b = vertices[:, :-1, :-1], vertices[:, :-1, 1:]
    c, d = vertices[:, 1:, 1:], vertices[:, 1:, :-1]
    jacobians = [torch.stack((n*x, n*y), -1) for x, y in
                 ((b-a, d-a), (b-a, c-b), (c-d, c-b), (c-d, d-a))]
    # Independent inverse-based phi, not production's determinant identity.
    return torch.stack([j.square().sum((-1, -2)) +
                        torch.linalg.inv(j).square().sum((-1, -2)) - 4
                        for j in jacobians], -1)


def reduced_shape(vertices, factor, diagonal):
    p = phis(vertices)
    i, j = (1, 3) if diagonal == "ac" else (0, 2)
    return (((factor-1)/(2*factor))*(p[..., i]+p[..., j]) +
            p.sum(-1)/(4*factor)).mean()


def mathematical_strain(vertices):
    n = vertices.shape[1] - 1
    residual = vertices - identity(n)
    return .5*((n*(residual[:, :, 1:]-residual[:, :, :-1])).square().sum(-1).mean()
               + (n*(residual[:, 1:]-residual[:, :-1])).square().sum(-1).mean())


def reduced_strain(vertices, factor):
    batch, n = vertices.shape[0], vertices.shape[1] - 1
    fine_n = n * factor
    residual = vertices - identity(n)
    dx = n*(residual[:, :, 1:]-residual[:, :, :-1])
    dy = n*(residual[:, 1:]-residual[:, :-1])
    weights = torch.full((n+1,), float(factor*factor))
    weights[0] = weights[-1] = factor*(factor+1)/2
    return ((dx.square().sum(-1)*weights[None, :, None]).sum() +
            (dy.square().sum(-1)*weights[None, None, :]).sum()) / (
                2*batch*fine_n*(fine_n+1))


def run():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(327)
    quad = torch.tensor([[[[0., 0.], [1., .1]], [[-.1, 1.], [1.1, 1.2]]]])
    random = identity(4).repeat(2, 1, 1, 1)
    random[:, 1:-1, 1:-1] += .018*torch.randn(2, 3, 3, 2)
    near = torch.tensor([[[[0., 0.], [1., 0.]], [[0., 1.], [1., .001]]]])
    rows = []
    for name, base in (("quad", quad), ("random_fixed", random), ("near_margin", near)):
        for diagonal in ("ac", "bd"):
            for factor in (1, 2, 3, 4):
                vertices = base.clone().requires_grad_()
                fine = prolong(vertices, factor, diagonal)
                full_shape = corner_symmetric_dirichlet(fine)
                small_shape = reduced_shape(vertices, factor, diagonal)
                full_strain, small_strain = mathematical_strain(fine), reduced_strain(vertices, factor)
                gradients = [torch.autograd.grad(value, vertices, retain_graph=True)[0]
                             for value in (full_shape, small_shape, full_strain, small_strain)]
                ga, gb, gc, gd = gradients
                production = strain_penalty(fine)
                gp = torch.autograd.grad(production, vertices, retain_graph=True)[0]
                tangent = torch.randn_like(vertices)
                epsilon = 1e-8 if name == "near_margin" else 1e-6
                plus, minus = vertices.detach()+epsilon*tangent, vertices.detach()-epsilon*tangent
                fd = ((reduced_shape(plus, factor, diagonal)+reduced_strain(plus, factor)) -
                      (reduced_shape(minus, factor, diagonal)+reduced_strain(minus, factor))) / (2*epsilon)
                ad = ((gb+gd)*tangent).sum()
                rows.append(dict(
                    case=name, diag=diagonal, m=factor,
                    shape_value_abs=float(abs(full_shape-small_shape)),
                    shape_gradient_max_abs=float((ga-gb).abs().max()),
                    shape_gradient_max_relative=float((ga-gb).abs().max()/ga.abs().max().clamp_min(1)),
                    strain_value_abs=float(abs(full_strain-small_strain)),
                    strain_gradient_max_abs=float((gc-gd).abs().max()),
                    production_strain_value_abs=float(abs(production-small_strain)),
                    production_strain_gradient_max_abs=float((gp-gd).abs().max()),
                    fd_relative=float(abs(fd-ad)/abs(ad).clamp_min(1))))
    return dict(
        question="Section20 coarse quadrature versus explicit fine prior and full-Y derivatives",
        seed=327, device="cpu", dtype="float64", torch_version=torch.__version__,
        checks=len(rows), refinement="independent integer-cell barycentric formula",
        derivative_scope="all mapped vertices, including boundary entries",
        production_reference_caveat=(
            "Production strain builds identity_vertices in float32 then casts. "
            "Non-dyadic factor3 differs from the ideal double reference. "
            "No reference or production behavior was modified."),
        interpretation=(
            "Real-arithmetic identity supported; no bitwise-equivalence, performance, "
            "registration-accuracy or new-general-FEM novelty claim. Near-small determinants "
            "amplify absolute numerical discrepancies; inspect relative gradients too."),
        results=rows)


if __name__ == "__main__":
    output = Path(__file__).with_suffix(".json")
    if output.exists():
        raise FileExistsError(output)
    payload = run()
    output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), checks=payload["checks"])))
