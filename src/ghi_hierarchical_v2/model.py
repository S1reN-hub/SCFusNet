"""Compact two-expert model for CPU training on the 799-case dataset."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value):
    return math.log(math.exp(value) - 1.0)


class SatelliteExpert(nn.Module):
    def __init__(self, source_dim=8, station_dim=8, hidden=64):
        super().__init__()
        self.source_embedding = nn.Embedding(3, source_dim)
        self.station_embedding = nn.Embedding(17, station_dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(14, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(33, hidden, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + source_dim + station_dim + 4 + 8, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.lead_embedding = nn.Embedding(6, 8)

    def forward(self, satellite, satellite_valid, age_hours, source_id, station_index, solar):
        batch, history = satellite.shape[:2]
        image = torch.cat([satellite, satellite_valid.float()], dim=2)
        encoded = self.spatial(image.reshape(batch * history, 14, 32, 32)).reshape(
            batch, history, 32
        )
        encoded = torch.cat([encoded, age_hours.unsqueeze(-1) / 5.0], dim=-1)
        _, state = self.temporal(encoded)
        context = state[-1].unsqueeze(1).expand(-1, 6, -1)
        source = self.source_embedding(source_id).unsqueeze(1).expand(-1, 6, -1)
        station = self.station_embedding(station_index).unsqueeze(1).expand(-1, 6, -1)
        lead = self.lead_embedding(torch.arange(6, device=satellite.device)).unsqueeze(0).expand(
            batch, -1, -1
        )
        raw = self.decoder(torch.cat([context, source, station, solar, lead], dim=-1)).squeeze(-1)
        upper = solar[..., 1] * 1.25 + 80.0
        return torch.sigmoid(raw) * upper


class WRFExpert(nn.Module):
    def __init__(self, static_dim=7, station_dim=8):
        super().__init__()
        self.volume = nn.Sequential(
            nn.Conv3d(6, 12, 3, padding=1),
            nn.GroupNorm(3, 12),
            nn.SiLU(),
            nn.Conv3d(12, 16, 3, stride=(2, 1, 1), padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.surface = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 12, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.lead_embedding = nn.Embedding(6, 8)
        self.station_embedding = nn.Embedding(17, station_dim)
        self.decoder = nn.Sequential(
            nn.Linear(254, 96),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        # Start exactly at the physically meaningful WRF anchor. Training can
        # improve from that baseline instead of first undoing a large random
        # residual.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, nwp_3d, nwp_3d_valid, nwp_surface, wrf_anchor, solar, static, station_index):
        batch, leads = nwp_3d.shape[:2]
        volume = self.volume(nwp_3d.reshape(batch * leads, 6, 10, 5, 5)).reshape(
            batch, leads, 16
        )
        surface = self.surface(nwp_surface.reshape(batch * leads, 4, 5, 5)).reshape(
            batch, leads, 12
        )
        valid = nwp_3d_valid[:, :, None, :, :, :].expand_as(nwp_3d)
        count = valid.sum(dim=(-1, -2)).clamp_min(1).float()
        safe = torch.where(valid, nwp_3d, torch.zeros_like(nwp_3d))
        spatial_mean = safe.sum(dim=(-1, -2)) / count
        centered = torch.where(valid, nwp_3d - spatial_mean[..., None, None], torch.zeros_like(nwp_3d))
        spatial_std = torch.sqrt((centered.square().sum(dim=(-1, -2)) / count).clamp_min(1e-8))
        center = torch.where(valid[..., 2, 2], nwp_3d[..., 2, 2], torch.zeros_like(nwp_3d[..., 2, 2]))
        vertical_spatial = torch.cat(
            [
                center.flatten(2),
                spatial_mean.flatten(2),
                spatial_std.flatten(2),
            ],
            dim=-1,
        )
        surface_statistics = torch.cat(
            [
                nwp_surface[..., 2, 2],
                nwp_surface.mean(dim=(-1, -2)),
                nwp_surface.std(dim=(-1, -2), unbiased=False),
            ],
            dim=-1,
        )
        anchor_context = (wrf_anchor / 1000.0).unsqueeze(1).expand(-1, 6, -1)
        lead = self.lead_embedding(torch.arange(6, device=nwp_3d.device)).unsqueeze(0).expand(
            batch, -1, -1
        )
        static = static.unsqueeze(1).expand(-1, 6, -1)
        station = self.station_embedding(station_index).unsqueeze(1).expand(-1, 6, -1)
        anchor_scaled = wrf_anchor.unsqueeze(-1) / 1000.0
        raw = self.decoder(
            torch.cat(
                [
                    volume,
                    surface,
                    vertical_spatial,
                    surface_statistics,
                    anchor_context,
                    solar,
                    static,
                    station,
                    lead,
                    anchor_scaled,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        radius = 0.55 * (solar[..., 1] + 200.0)
        return wrf_anchor + torch.tanh(raw) * radius


class HierarchicalGHIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.satellite_expert = SatelliteExpert()
        self.wrf_expert = WRFExpert()
        self.source_embedding = nn.Embedding(3, 4)
        self.station_embedding = nn.Embedding(17, 4)
        self.router = nn.Sequential(
            nn.Linear(2 + 1 + 4 + 4 + 4, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )
        self.transition = nn.Sequential(
            nn.Linear(4 + 2, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )
        self.aod_gate = nn.Sequential(
            nn.Linear(4 + 3, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.copy_(torch.tensor([-2.0, 2.0]))
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)
        nn.init.zeros_(self.aod_gate[-1].weight)
        nn.init.zeros_(self.aod_gate[-1].bias)
        self.aod_strength_raw = nn.Parameter(torch.tensor(_inverse_softplus(0.01)))

    @staticmethod
    def project(prediction, solar):
        daylight = solar[..., 2] > 0.0
        upper = solar[..., 1] * 1.35 + 100.0
        bounded = torch.minimum(torch.clamp_min(prediction, 0.0), upper)
        return torch.where(daylight, bounded, torch.zeros_like(bounded))

    def forward(self, batch):
        sat = self.satellite_expert(
            batch["satellite"],
            batch["satellite_valid"],
            batch["satellite_age_hours"],
            batch["satellite_source_id"],
            batch["station_index"],
            batch["solar"],
        )
        wrf = self.wrf_expert(
            batch["nwp_3d"],
            batch["nwp_3d_valid"],
            batch["nwp_surface"],
            batch["wrf_anchor"],
            batch["solar"],
            batch["static"],
            batch["station_index"],
        )
        upper = batch["solar"][..., 1] + 100.0
        source = self.source_embedding(batch["satellite_source_id"]).unsqueeze(1).expand(-1, 6, -1)
        station = self.station_embedding(batch["station_index"]).unsqueeze(1).expand(-1, 6, -1)
        router_input = torch.cat(
            [
                (sat / upper).unsqueeze(-1),
                (wrf / upper).unsqueeze(-1),
                (torch.abs(sat - wrf) / upper).unsqueeze(-1),
                batch["solar"],
                source,
                station,
            ],
            dim=-1,
        )
        weights = torch.softmax(self.router(router_input), dim=-1)
        base = weights[..., 0] * sat + weights[..., 1] * wrf
        transition_input = torch.cat(
            [batch["solar"], (base / upper).unsqueeze(-1), (wrf / upper).unsqueeze(-1)], dim=-1
        )
        transition_delta = (
            torch.tanh(self.transition(transition_input).squeeze(-1))
            * batch["solar"][..., 3]
            * 0.15
            * upper
        )
        transitioned = base + transition_delta
        aod_input = torch.cat(
            [
                batch["solar"],
                (transitioned / upper).unsqueeze(-1),
                (wrf / upper).unsqueeze(-1),
                weights[..., :1],
            ],
            dim=-1,
        )
        aod_gate = torch.sigmoid(self.aod_gate(aod_input).squeeze(-1))
        positive_anomaly = F.relu(batch["aod550"] - batch["aod_reference"])
        aod_delta = (
            -F.softplus(self.aod_strength_raw)
            * positive_anomaly
            * aod_gate
            * batch["aod_mask"].float()
            * upper
        )
        final = transitioned + aod_delta
        return {
            "satellite_prediction": self.project(sat, batch["solar"]),
            "wrf_prediction": self.project(wrf, batch["solar"]),
            "base_prediction": self.project(base, batch["solar"]),
            "transition_prediction": self.project(transitioned, batch["solar"]),
            "final_prediction": self.project(final, batch["solar"]),
            "fusion_weights": weights,
            "transition_delta": transition_delta,
            "aod_delta": aod_delta,
            "aod_gate": aod_gate,
        }
