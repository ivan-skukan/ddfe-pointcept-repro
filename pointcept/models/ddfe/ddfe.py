import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria
from pointcept.models.utils import offset2batch


def _make_mlp(in_channels, hidden_channels, out_channels, use_bn=True):
    layers = [nn.Linear(in_channels, hidden_channels)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden_channels, out_channels))
    if use_bn:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _make_attention(in_channels, hidden_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        nn.BatchNorm1d(hidden_channels),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_channels, out_channels),
        nn.Sigmoid(),
    )


class BeamDensityEstimator(nn.Module):
    def __init__(
        self,
        sensors,
        projection_shape=(512, 5120),
        projection_fov=(-30.0, 15.0),
        gaussian_sigmas=(10, 30, 50, 70),
        gaussian_truncate=3.0,
        normalize_gaussian=True,
        range_eps=1e-3,
    ):
        super().__init__()
        self.sensor_names = sorted(sensors, key=lambda name: sensors[name]["id"])
        self.sensor_ids = [int(sensors[name]["id"]) for name in self.sensor_names]
        self.sensor_id_to_index = {
            sensor_id: idx for idx, sensor_id in enumerate(self.sensor_ids)
        }
        self.height, self.width = int(projection_shape[0]), int(projection_shape[1])
        self.projection_fov = projection_fov
        self.gaussian_sigmas = tuple(float(sigma) for sigma in gaussian_sigmas)
        self.gaussian_truncate = float(gaussian_truncate)
        self.normalize_gaussian = bool(normalize_gaussian)
        self.range_eps = float(range_eps)
        beam_h, beam_v = self._build_beam_tables(sensors)
        self.register_buffer("beam_h", beam_h, persistent=False)
        self.register_buffer("beam_v", beam_v, persistent=False)

    @property
    def out_channels(self):
        return len(self.gaussian_sigmas)

    def _smooth_1d(self, vector, sigma, circular=False):
        radius = max(1, int(math.ceil(self.gaussian_truncate * sigma)))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-(x**2) / (2 * sigma**2))
        if self.normalize_gaussian:
            kernel = kernel / kernel.sum().clamp_min(1e-12)
        vector = vector.view(1, 1, -1)
        pad_mode = "circular" if circular else "constant"
        padded = F.pad(vector, (radius, radius), mode=pad_mode)
        return F.conv1d(padded, kernel.view(1, 1, -1)).view(-1)

    def _build_beam_tables(self, sensors):
        all_h = []
        all_v = []
        proj_min, proj_max = self.projection_fov
        for name in self.sensor_names:
            cfg = sensors[name]
            hb, vb = int(cfg["hb"]), int(cfg["vb"])
            fov_min, fov_max = float(cfg["fov"][0]), float(cfg["fov"][1])
            endpoint = bool(cfg.get("vertical_endpoint", True))

            h_binary = torch.zeros(self.width, dtype=torch.float32)
            h_idx = torch.floor(torch.arange(hb, dtype=torch.float32) * self.width / hb)
            h_binary[h_idx.long().clamp(0, self.width - 1)] = 1.0

            v_binary = torch.zeros(self.height, dtype=torch.float32)
            v_angles = torch.linspace(fov_min, fov_max, vb)
            if not endpoint:
                step = (fov_max - fov_min) / max(vb, 1)
                v_angles = torch.arange(vb, dtype=torch.float32) * step + fov_min
            v_idx = torch.floor(
                (v_angles - proj_min) / (proj_max - proj_min) * self.height
            )
            v_binary[v_idx.long().clamp(0, self.height - 1)] = 1.0

            all_h.append(
                torch.stack(
                    [
                        self._smooth_1d(h_binary, sigma, circular=True)
                        for sigma in self.gaussian_sigmas
                    ]
                )
            )
            all_v.append(
                torch.stack(
                    [
                        self._smooth_1d(v_binary, sigma, circular=False)
                        for sigma in self.gaussian_sigmas
                    ]
                )
            )
        return torch.stack(all_h), torch.stack(all_v)

    def forward(self, coord, offset, lidar_sensor=None, density_origin=None):
        if density_origin is None:
            density_coord = coord
        else:
            density_coord = coord - density_origin.to(coord.device, coord.dtype)

        r_xy = torch.linalg.norm(density_coord[:, :2], dim=1)
        radius = torch.linalg.norm(density_coord, dim=1).clamp_min(self.range_eps)
        theta = torch.remainder(torch.atan2(density_coord[:, 1], density_coord[:, 0]), 2 * math.pi)
        phi = torch.atan2(density_coord[:, 2], r_xy) * (180.0 / math.pi)

        theta_idx = torch.floor(theta / (2 * math.pi) * self.width).long()
        theta_idx = theta_idx.clamp(0, self.width - 1)
        proj_min, proj_max = self.projection_fov
        phi_idx = torch.floor((phi - proj_min) / (proj_max - proj_min) * self.height)
        phi_idx = phi_idx.long().clamp(0, self.height - 1)

        batch = offset2batch(offset.long())
        batch_size = int(offset.numel())
        if lidar_sensor is None:
            sensor_ids = torch.full(
                (batch_size,),
                self.sensor_ids[0],
                dtype=torch.long,
                device=coord.device,
            )
        else:
            sensor_ids = lidar_sensor.long().view(-1).to(coord.device)
            if sensor_ids.numel() == 1 and batch_size > 1:
                sensor_ids = sensor_ids.expand(batch_size)

        density = coord.new_zeros((coord.shape[0], self.out_channels))
        for sample_idx in range(batch_size):
            mask = batch == sample_idx
            if not bool(mask.any()):
                continue
            sensor_id = int(sensor_ids[sample_idx].item())
            table_idx = self.sensor_id_to_index.get(sensor_id, 0)
            h = self.beam_h[table_idx, :, theta_idx[mask]].transpose(0, 1)
            v = self.beam_v[table_idx, :, phi_idx[mask]].transpose(0, 1)
            density[mask] = torch.sqrt((h * v).clamp_min(0) / (radius[mask, None] ** 2))
        return density


class ReservoirPercentileTracker(nn.Module):
    def __init__(
        self,
        channels,
        reservoir_size=1000,
        percentile_low=10.0,
        percentile_high=90.0,
        min_samples=16,
        eps=1e-6,
    ):
        super().__init__()
        self.channels = int(channels)
        self.reservoir_size = int(reservoir_size)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.min_samples = int(min_samples)
        self.eps = float(eps)
        self.register_buffer("reservoir", torch.zeros(self.reservoir_size, self.channels))
        self.register_buffer("filled", torch.zeros((), dtype=torch.long))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("p_low", torch.zeros(self.channels))
        self.register_buffer("p_high", torch.ones(self.channels))

    @torch.no_grad()
    def _valid_reservoir(self):
        filled = int(self.filled.item())
        if filled <= 0:
            return None
        return self.reservoir[:filled]

    @torch.no_grad()
    def _update(self, density):
        if density.numel() == 0:
            return
        density = density.detach().float()
        count = density.shape[0]
        sample_count = min(count, self.reservoir_size)
        sample_idx = torch.randperm(count, device=density.device)[:sample_count]
        sample = density[sample_idx].to(self.reservoir.device)

        filled = int(self.filled.item())
        if filled < self.reservoir_size:
            add_count = min(sample_count, self.reservoir_size - filled)
            self.reservoir[filled : filled + add_count] = sample[:add_count]
            self.filled += add_count
            sample = sample[add_count:]
            sample_count = sample.shape[0]

        if sample_count > 0:
            replace_idx = torch.randint(
                0, self.reservoir_size, (sample_count,), device=self.reservoir.device
            )
            self.reservoir[replace_idx] = sample.to(self.reservoir.device)

        valid = self._valid_reservoir()
        if valid is None or valid.shape[0] < self.min_samples:
            return

        q = torch.tensor(
            [self.percentile_low / 100.0, self.percentile_high / 100.0],
            device=valid.device,
            dtype=valid.dtype,
        )
        current = torch.quantile(valid, q, dim=0)
        if int(self.updates.item()) == 0:
            self.p_low.copy_(current[0])
            self.p_high.copy_(current[1])
        else:
            weight = 1.0 / float(int(self.updates.item()) + 1)
            self.p_low.add_((current[0] - self.p_low) * weight)
            self.p_high.add_((current[1] - self.p_high) * weight)
        self.updates += 1

    def forward(self, density, training=False):
        if training:
            self._update(density)
        if int(self.updates.item()) == 0:
            q = torch.tensor(
                [self.percentile_low / 100.0, self.percentile_high / 100.0],
                device=density.device,
                dtype=density.dtype,
            )
            current = torch.quantile(density.detach().float(), q, dim=0)
            return current[0].to(density.dtype), current[1].to(density.dtype)
        return self.p_low.to(density.device, density.dtype), self.p_high.to(
            density.device, density.dtype
        )


class DDFEFeatureEncoder(nn.Module):
    def __init__(
        self,
        sensors,
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
        clip=None,
        return_debug=False,
    ):
        super().__init__()
        if clip is None:
            clip = dict(enabled=True)
        self.return_debug = bool(return_debug)
        self.density = BeamDensityEstimator(
            sensors=sensors,
            projection_shape=projection_shape,
            projection_fov=projection_fov,
            gaussian_sigmas=gaussian_sigmas,
            gaussian_truncate=gaussian_truncate,
            normalize_gaussian=normalize_gaussian,
            range_eps=range_eps,
        )
        self.clip_enabled = bool(clip.get("enabled", True))
        self.clip_eps = float(clip.get("eps", 1e-6))
        self.percentiles = ReservoirPercentileTracker(
            channels=self.density.out_channels,
            reservoir_size=clip.get("reservoir_size", 1000),
            percentile_low=clip.get("percentile_low", 10.0),
            percentile_high=clip.get("percentile_high", 90.0),
            min_samples=clip.get("min_samples", 16),
            eps=self.clip_eps,
        )
        self.spherical_mlp = _make_mlp(4, spherical_channels, spherical_channels)
        self.displacement_mlp = _make_mlp(3, displacement_channels, displacement_channels)
        self.voxel_attention = _make_attention(
            self.density.out_channels, attention_hidden_channels, spherical_channels
        )
        self.point_attention = _make_attention(
            self.density.out_channels, attention_hidden_channels, displacement_channels
        )
        self.proj = nn.Sequential(
            nn.Linear(spherical_channels + displacement_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _spherical_features(coord):
        r_xy = torch.linalg.norm(coord[:, :2], dim=1)
        radius = torch.linalg.norm(coord, dim=1).clamp_min(1e-6)
        theta = torch.atan2(coord[:, 1], coord[:, 0])
        phi = torch.atan2(coord[:, 2], r_xy)
        return torch.stack(
            [torch.cos(theta), torch.sin(theta), phi, radius], dim=1
        )

    def _clip_density(self, density):
        if not self.clip_enabled:
            return density
        p_low, p_high = self.percentiles(density, training=self.training)
        center = (p_high + p_low) * 0.5
        half_range = ((p_high - p_low) * 0.5).clamp_min(self.clip_eps)
        return torch.tanh((density - center) / half_range) * half_range + center

    def forward(self, input_dict):
        coord = input_dict["coord"].float()
        offset = input_dict["offset"].long()
        density_origin = input_dict.get("density_origin", None)
        density = self.density(
            coord=coord,
            offset=offset,
            lidar_sensor=input_dict.get("lidar_sensor", None),
            density_origin=density_origin,
        )
        clipped_density = self._clip_density(density)

        displacement = input_dict.get("displacement", None)
        if displacement is None:
            displacement = torch.zeros_like(coord)
        displacement = displacement.float()
        if displacement.shape[1] == 1:
            displacement = displacement.expand(-1, 3)

        voxel_feat = self.spherical_mlp(self._spherical_features(coord))
        point_feat = self.displacement_mlp(displacement[:, :3])
        voxel_feat = voxel_feat * self.voxel_attention(clipped_density)
        point_feat = point_feat * self.point_attention(clipped_density)
        feat = self.proj(torch.cat([voxel_feat, point_feat], dim=1))
        if self.return_debug:
            input_dict["ddfe_density"] = density
            input_dict["ddfe_clipped_density"] = clipped_density
        return feat


@MODELS.register_module()
class DDFESegmentor(nn.Module):
    def __init__(
        self,
        backbone=None,
        ddfe=None,
        criteria=None,
        freeze_backbone=False,
        unfreeze_backbone_keywords=(),
    ):
        super().__init__()
        assert ddfe is not None, "DDFESegmentor requires a ddfe config."
        self.ddfe = DDFEFeatureEncoder(**deepcopy(ddfe))
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_backbone_keywords = tuple(unfreeze_backbone_keywords)
        if self.freeze_backbone:
            self._set_backbone_requires_grad()

    def _set_backbone_requires_grad(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = any(
                keyword in name for keyword in self.unfreeze_backbone_keywords
            )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            input_dict["condition"] = input_dict["condition"][0]
        model_input = dict(input_dict)
        model_input["feat"] = self.ddfe(model_input)
        seg_logits = self.backbone(model_input)
        if self.training:
            loss = self.criteria(seg_logits, model_input["segment"])
            return dict(loss=loss)
        elif "segment" in model_input.keys():
            loss = self.criteria(seg_logits, model_input["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        else:
            return dict(seg_logits=seg_logits)
