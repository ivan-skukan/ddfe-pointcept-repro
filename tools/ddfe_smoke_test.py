import argparse
import torch

from pointcept.models import build_model


DDFE_CFG = dict(
    out_channels=32,
    spherical_channels=16,
    displacement_channels=16,
    attention_hidden_channels=16,
    projection_shape=(512, 5120),
    projection_fov=(-30.0, 15.0),
    gaussian_sigmas=(10, 30, 50, 70),
    gaussian_truncate=3.0,
    normalize_gaussian=True,
    range_eps=1e-3,
    clip=dict(
        enabled=True,
        percentile_low=10.0,
        percentile_high=90.0,
        reservoir_size=1000,
        min_samples=16,
        eps=1e-6,
    ),
    sensors=dict(
        semantic_kitti=dict(id=0, hb=2048, vb=64, fov=(-24.8, 2.0)),
        nuscenes=dict(id=1, hb=1080, vb=32, fov=(-30.0, 10.0)),
        waymo=dict(id=2, hb=2560, vb=64, fov=(-17.6, 2.4)),
    ),
)


def make_batch(num_points, num_classes, device):
    coord = torch.empty(num_points, 3, device=device).uniform_(-20.0, 20.0)
    coord[:, 2].uniform_(-2.0, 2.0)
    grid_coord = torch.floor((coord - coord.min(dim=0).values) / 0.2).long()
    return dict(
        coord=coord,
        grid_coord=grid_coord,
        displacement=torch.empty(num_points, 3, device=device).uniform_(-0.5, 0.5),
        offset=torch.tensor([num_points], dtype=torch.long, device=device),
        segment=torch.randint(0, num_classes, (num_points,), device=device),
        lidar_sensor=torch.tensor([0], dtype=torch.long, device=device),
    )


def build_smoke_model(num_classes, freeze_backbone):
    return build_model(
        dict(
            type="DDFESegmentor",
            freeze_backbone=freeze_backbone,
            ddfe=DDFE_CFG,
            backbone=dict(
                type="SpUNet-v1m1",
                in_channels=32,
                num_classes=num_classes,
                base_channels=8,
                channels=(8, 16, 32, 64, 64, 32, 24, 24),
                layers=(1, 1, 1, 1, 1, 1, 1, 1),
            ),
            criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
        )
    )


def assert_ddfe_grad(model):
    grad_norm = 0.0
    for name, param in model.named_parameters():
        if name.startswith("ddfe.") and param.grad is not None:
            grad_norm += float(param.grad.detach().abs().sum().cpu())
    assert grad_norm > 0, "DDFE parameters did not receive gradients"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-classes", type=int, default=4)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "DDFE SpUNet smoke test requires CUDA"
    device = torch.device("cuda")
    for freeze_backbone in [False, True]:
        torch.manual_seed(2026)
        model = build_smoke_model(args.num_classes, freeze_backbone).to(device)
        model.train()
        batch = make_batch(args.num_points, args.num_classes, device)
        output = model(batch)
        output["loss"].backward()
        assert_ddfe_grad(model)
        trainable_backbone = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("backbone.") and p.requires_grad
        )
        if freeze_backbone:
            assert trainable_backbone == 0, "Frozen backbone still has trainable params"
        print(
            f"freeze_backbone={freeze_backbone} loss={float(output['loss'].detach().cpu()):.4f}"
        )
    print("DDFE smoke test passed")


if __name__ == "__main__":
    main()

