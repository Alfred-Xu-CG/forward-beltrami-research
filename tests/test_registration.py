import torch

from qcopt.registration import i_field, registration_loss, s_field, soft_dice


def test_i_and_s_fields_are_bounded_and_geometrically_distinct():
    axis = torch.linspace(0.0, 1.0, 65, dtype=torch.double)
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    image_i = i_field(points)
    image_s = s_field(points)

    assert torch.all((0.0 <= image_i) & (image_i <= 1.0))
    assert torch.all((0.0 <= image_s) & (image_s <= 1.0))
    assert torch.mean(torch.abs(image_i - image_s)) > 0.15
    assert image_i.max() > 0.95
    assert image_s.max() > 0.90


def test_s_field_and_registration_loss_are_differentiable_in_map_positions():
    points = torch.tensor(
        [[0.25, 0.8], [0.5, 0.5], [0.75, 0.2]],
        dtype=torch.double,
        requires_grad=True,
    )
    source = i_field(points.detach())
    target = s_field(points)
    loss = registration_loss(source, target)
    loss.backward()

    assert points.grad is not None
    assert torch.all(torch.isfinite(points.grad))
    assert torch.linalg.vector_norm(points.grad) > 0.0


def test_soft_dice_is_one_for_identical_nonempty_fields():
    values = torch.tensor([0.0, 0.2, 0.7, 1.0], dtype=torch.double)
    assert torch.allclose(soft_dice(values, values), torch.tensor(1.0, dtype=torch.double))
