"""V4 dual-primary-expert architecture for 1--6 h GHI forecasting.

The implementation keeps the manuscript's two independently supervised primary
experts.  Satellite and WRF information meet only in a reliability-aware
router; solar-transition and aerosol terms remain bounded conditional
corrections.  The model accepts the complete v4 tensor contract and has an
explicit legacy adapter so that structural tests can run before new WRF fields
are available.  Legacy adaptation is not a substitute for the v4 experiment.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

NWP3D_CHANNELS = 12
NWP_SURFACE_CHANNELS = 15
NWP_LEVELS = 10
LEGACY_3D_TO_V4 = (0, 1, 2, 4, 7, 8)
LEGACY_SURFACE_TO_V4 = (0, 2, 9, 10)


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def bounded_update(q: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply an endpoint-preserving update on a normalized physical interval."""

    q = q.clamp(0.0, 1.0)
    lower = q * torch.exp(delta.clamp(max=0.0))
    upper = 1.0 - (1.0 - q) * torch.exp((-delta).clamp(max=0.0))
    return torch.where(delta <= 0.0, lower, upper).clamp(0.0, 1.0)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, values: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(self.gates(torch.cat([values, hidden], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([values, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class SpatialSummary(nn.Module):
    """Summarize global, station-centred and signed positional information."""

    def __init__(self, channels: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(4 * channels, 2 * output_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2 * output_dim, output_dim), nn.LayerNorm(output_dim),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature.shape
        global_context = feature.mean(dim=(-1, -2))
        cy, cx = height // 2, width // 2
        centre = feature[:, :, max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].mean(dim=(-1, -2))
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        y_moment = (feature * y.view(1, 1, height, 1)).mean(dim=(-1, -2))
        x_moment = (feature * x.view(1, 1, 1, width)).mean(dim=(-1, -2))
        return self.projection(torch.cat([global_context, centre, y_moment, x_moment], dim=-1))


class MonotonicLeadPrior(nn.Module):
    def __init__(self, probabilities=(0.78, 0.64, 0.48, 0.34, 0.24, 0.18)):
        super().__init__()
        logits = [_logit(float(value)) for value in probabilities]
        decrements = [max(logits[index - 1] - logits[index], 1e-3) for index in range(1, 6)]
        self.first = nn.Parameter(torch.tensor(logits[0], dtype=torch.float32))
        self.decrement_raw = nn.Parameter(
            torch.tensor([_inverse_softplus(value) for value in decrements], dtype=torch.float32)
        )

    def logits(self) -> torch.Tensor:
        decrements = F.softplus(self.decrement_raw)
        return torch.cat([self.first.view(1), self.first - torch.cumsum(decrements, dim=0)])

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logits())


def _batch_value(batch: dict[str, torch.Tensor], names: Iterable[str]) -> torch.Tensor | None:
    for name in names:
        if name in batch:
            return batch[name]
    return None


def _pad_or_map_channels(values: torch.Tensor, channels: int, legacy_mapping: tuple[int, ...]) -> torch.Tensor:
    """Map legacy channels to their v4 semantic positions or pad new partial data."""

    if values.shape[2] == channels:
        return values
    result = values.new_zeros(*values.shape[:2], channels, *values.shape[3:])
    if values.shape[2] == len(legacy_mapping):
        for source, target in enumerate(legacy_mapping):
            result[:, :, target] = values[:, :, source]
    else:
        count = min(values.shape[2], channels)
        result[:, :, :count] = values[:, :, :count]
    return result


def _availability_mask(
    batch: dict[str, torch.Tensor], names: tuple[str, ...], original_channels: int,
    channels: int, legacy_mapping: tuple[int, ...], reference: torch.Tensor,
) -> torch.Tensor:
    available = _batch_value(batch, names)
    if available is not None:
        return available.to(reference).bool()
    result = torch.zeros(reference.shape[0], channels, device=reference.device, dtype=torch.bool)
    if original_channels == len(legacy_mapping):
        result[:, list(legacy_mapping)] = True
    else:
        result[:, :min(original_channels, channels)] = True
    return result


def _ensure_seven_states(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] == 7:
        return values
    if values.shape[1] != 6:
        raise ValueError(f"WRF time dimension must contain 6 or 7 states, got {values.shape[1]}")
    return torch.cat([values[:, :1], values], dim=1)


def _resize_volume(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if values.shape[-2:] == size:
        return values
    batch, time, channels, levels, height, width = values.shape
    flattened = values.reshape(batch * time, channels * levels, height, width)
    resized = F.interpolate(flattened, size=size, mode="bilinear", align_corners=False)
    return resized.reshape(batch, time, channels, levels, *size)


def _resize_surface(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if values.shape[-2:] == size:
        return values
    batch, time, channels, height, width = values.shape
    resized = F.interpolate(values.reshape(batch * time, channels, height, width), size=size, mode="bilinear", align_corners=False)
    return resized.reshape(batch, time, channels, *size)


def _normal_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def warp_2d(values: torch.Tensor, flow_xy_cells: torch.Tensor) -> torch.Tensor:
    """Semi-Lagrangian backward sampling for a flow expressed in grid cells."""

    batch, _, height, width = values.shape
    grid = _normal_grid(batch, height, width, values.device, values.dtype).clone()
    grid[..., 0] -= 2.0 * flow_xy_cells[:, 0] / max(width - 1, 1)
    grid[..., 1] -= 2.0 * flow_xy_cells[:, 1] / max(height - 1, 1)
    return F.grid_sample(values, grid, mode="bilinear", padding_mode="border", align_corners=True)


class SatelliteSourceAdapter(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels, hidden // 2, 5, stride=2, padding=2),
            nn.GroupNorm(_groups(hidden // 2), hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([values, valid.to(values)], dim=1))


class SatelliteStationReadout(nn.Module):
    """Lead-conditioned station query over positioned image and rollout tokens."""

    def __init__(self, hidden, dimension):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(14, hidden, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.GELU())
        self.position = nn.Linear(2, hidden)
        self.query = nn.Linear(32 + 8 + 16, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.output = nn.Sequential(nn.Linear(3 * hidden, dimension), nn.GELU(), nn.LayerNorm(dimension))

    def forward(self, fine, latent, context):
        memories = []
        for feature in (fine, latent):
            grid = _normal_grid(feature.shape[0], *feature.shape[-2:], feature.device, feature.dtype)
            memories.append(feature.flatten(2).transpose(1, 2) + self.position(grid.flatten(1, 2)))
        memory = torch.cat(memories, dim=1)
        query = self.query(context).unsqueeze(1)
        attended, _ = self.attention(query, memory, memory, need_weights=False)
        local = fine[:, :, fine.shape[-2] // 2, fine.shape[-1] // 2]
        return self.output(torch.cat([attended[:, 0], local, query[:, 0]], dim=-1))


class TimeAwareSatelliteExpert(nn.Module):
    """Causal satellite expert with explicit time gaps, motion and uncertainty."""

    def __init__(self, d_model: int = 128, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.hidden = hidden
        self.adapters = nn.ModuleList(
            [SatelliteSourceAdapter(5, hidden), SatelliteSourceAdapter(1, hidden), SatelliteSourceAdapter(1, hidden)]
        )
        self.history_cell = ConvGRUCell(hidden, hidden)
        self.future_cell = ConvGRUCell(hidden, hidden)
        self.time_condition = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.motion_head = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 3, 3, padding=1)
        )
        self.lead_embedding = nn.Embedding(6, 32)
        self.source_embedding = nn.Embedding(3, 16)
        self.rollout_condition = nn.Sequential(nn.Linear(32 + 8 + 16, hidden), nn.GELU())
        self.summary = SpatialSummary(hidden, d_model)
        self.station_readout = SatelliteStationReadout(hidden, d_model)
        self.direct_context = nn.Sequential(nn.Linear(32 + 8 + 16 + 7 + 2, d_model),
                                            nn.GELU(), nn.LayerNorm(d_model))
        self.direct_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.path_gate = nn.Sequential(nn.Linear(2 * d_model + 2, d_model // 2), nn.GELU(),
                                       nn.Linear(d_model // 2, 1))
        self.token = nn.Sequential(
            nn.Linear(d_model + 32 + 8 + 16 + 5, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.trajectory = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.clear_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.scale_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.future_cloud_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.constant_(self.scale_head[-1].bias, -1.8)

    def _adapt(self, values: torch.Tensor, valid: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        source_inputs = (
            (values[:, :5], valid[:, :5]),
            (values[:, 5:6], valid[:, 5:6]),
            (values[:, 6:7], valid[:, 6:7]),
        )
        candidates = torch.stack(
            [adapter(x, m) for adapter, (x, m) in zip(self.adapters, source_inputs)], dim=1
        )
        gather = source_id.view(-1, 1, 1, 1, 1).expand(-1, 1, candidates.shape[2], candidates.shape[3], candidates.shape[4])
        return candidates.gather(1, gather).squeeze(1)

    @staticmethod
    def _scan_offsets(batch: dict[str, torch.Tensor], frame_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = _batch_value(batch, ("satellite_scan_time_offset_hours", "satellite_time_offset_hours"))
        if offsets is None:
            offsets = -batch["satellite_age_hours"]
        explicit_unique = batch.get("satellite_unique_frame_mask")
        if explicit_unique is not None:
            return offsets, explicit_unique.bool() & frame_mask
        difference = torch.ones_like(offsets, dtype=torch.bool)
        difference[:, 1:] = (offsets[:, 1:] - offsets[:, :-1]).abs() > 1e-4
        return offsets, difference & frame_mask

    @staticmethod
    @torch.no_grad()
    def _history_transport_score(previous, previous_valid, target, target_valid,
                                 prior_flow, dt, eligible, source_id):
        """Score a previous-pair velocity against the next pre-issue scan.

        Both MAEs use the same finite, source-active, valid and in-bounds pixels.
        No target scan is used to infer ``prior_flow``. Values are the adapter's
        normalized source channels, not physical cloud-displacement ground truth.
        """
        b, c, h, w = previous.shape
        displacement = F.interpolate(prior_flow.detach() * dt[:, None, None, None],
                                     size=(h, w), mode="bilinear", align_corners=True)
        latent_h, latent_w = prior_flow.shape[-2:]
        # Convert latent displacement directly into align_corners grid units.
        grid = _normal_grid(b, h, w, previous.device, previous.dtype)
        grid = grid - (displacement * displacement.new_tensor(
            [2.0 / (latent_w - 1), 2.0 / (latent_h - 1)])[None, :, None, None]).permute(0, 2, 3, 1)
        aligned = F.grid_sample(previous, grid, align_corners=True, padding_mode="border")
        moved_valid = F.grid_sample(previous_valid.to(previous), grid, align_corners=True,
                                    padding_mode="zeros") > .999
        channel = torch.arange(c, device=previous.device)[None, :, None, None]
        sid = source_id[:, None, None, None]
        selected = ((sid == 0) & (channel < 5)) | ((sid == 1) & (channel == 5)) | ((sid == 2) & (channel == 6))
        support = (moved_valid & previous_valid & target_valid & selected
                   & (grid.abs() <= 1).all(-1)[:, None]
                   & torch.isfinite(aligned) & torch.isfinite(previous) & torch.isfinite(target)
                   & eligible[:, None, None, None])
        count = support.flatten(1).sum(1)
        transport = torch.where(support, (aligned-target).abs(), 0).flatten(1).sum(1) / count.clamp_min(1)
        persistence = torch.where(support, (previous-target).abs(), 0).flatten(1).sum(1) / count.clamp_min(1)
        usable = (count > 0) & ((persistence + transport) > 1e-8)
        skill = (persistence - transport) / (persistence + transport).clamp_min(1e-8)
        overlap = count.to(previous) / (selected.flatten(1).sum(1) * h * w).clamp_min(1)
        return torch.where(usable, skill, 0), overlap, usable

    def forward(self, batch: dict[str, torch.Tensor], disable_motion: bool = False,
                disable_station_readout: bool = False, disable_history_alignment: bool = False,
                prediction_path: str = "dual") -> dict[str, torch.Tensor]:
        satellite = batch["satellite"]
        valid = batch["satellite_valid"]
        frame_mask = batch["satellite_frame_mask"].bool()
        source_id = batch["satellite_source_id"]
        batch_size, history = satellite.shape[:2]
        offsets, unique_mask = self._scan_offsets(batch, frame_mask)
        hidden = satellite.new_zeros(batch_size, self.hidden, 8, 8)
        previous_offset = offsets[:, 0]
        seen = torch.zeros(batch_size, device=satellite.device, dtype=torch.bool)
        latest_flow = hidden.new_zeros(batch_size, 2, 8, 8)
        latest_confidence = hidden.new_zeros(batch_size)
        valid_pairs = hidden.new_zeros(batch_size)
        latest_image = torch.zeros_like(satellite[:, 0])
        latest_valid = torch.zeros_like(valid[:, 0])
        alignment_sum = hidden.new_zeros(batch_size)
        alignment_count = hidden.new_zeros(batch_size)
        transport_skill_sum = hidden.new_zeros(batch_size)
        transport_overlap_sum = hidden.new_zeros(batch_size)
        transport_dt_sum = hidden.new_zeros(batch_size)
        transport_pairs = hidden.new_zeros(batch_size)
        decay_rate = F.softplus(torch.tensor(-1.0, device=hidden.device, dtype=hidden.dtype))
        for step in range(history):
            active_frame = unique_mask[:, step]
            dt = torch.where(active_frame & seen, offsets[:, step] - previous_offset,
                             torch.zeros_like(previous_offset)).clamp_min(0.0)
            hidden = hidden * torch.exp(-decay_rate * dt).view(batch_size, 1, 1, 1)
            frame = self._adapt(satellite[:, step], valid[:, step], source_id)
            time_features = torch.stack(
                [dt / 3.0, offsets[:, step] / 6.0, torch.sin(offsets[:, step]), torch.cos(offsets[:, step])], dim=-1
            )
            frame = frame + self.time_condition(time_features).unsqueeze(-1).unsqueeze(-1)
            old_hidden = hidden
            candidate = self.history_cell(frame, hidden)
            active = unique_mask[:, step].view(batch_size, 1, 1, 1).to(frame)
            hidden = active * candidate + (1.0 - active) * hidden
            motion_raw = self.motion_head(torch.cat([old_hidden, hidden], dim=1))
            pair_active = (active_frame & seen & (dt > 1e-4)).to(frame)
            flow = 2.0 * torch.tanh(motion_raw[:, :2]) / dt.clamp_min(0.25).view(batch_size, 1, 1, 1)
            confidence = torch.sigmoid(motion_raw[:, 2]).mean(dim=(-1, -2))
            # latest_flow still belongs to the preceding pair. The current
            # target-derived flow must not enter this held-forward diagnostic.
            with torch.no_grad():
                eligible = (pair_active.bool() & valid_pairs.bool() & (offsets[:, step] <= 0)
                            & (previous_offset <= 0) & (not disable_motion))
                skill, overlap, usable = self._history_transport_score(
                    latest_image.detach(), latest_valid, satellite[:, step].detach(), valid[:, step],
                    latest_flow.detach(), dt.detach(), eligible, source_id)
                transport_skill_sum += skill
                transport_overlap_sum += torch.where(usable, overlap, 0)
                transport_dt_sum += torch.where(usable, dt.detach(), 0)
                transport_pairs += usable.to(satellite)
            # Historical image alignment supplies dense supervision without
            # accessing any post-issue image. Use all valid source channels.
            if self.training and not disable_motion and not disable_history_alignment:
                displacement = F.interpolate(flow * dt.view(-1, 1, 1, 1),
                                             size=satellite.shape[-2:], mode="bilinear", align_corners=True)
                scale = displacement.new_tensor([(satellite.shape[-1] - 1) / 7.0,
                                                  (satellite.shape[-2] - 1) / 7.0]).view(1, 2, 1, 1)
                displacement = displacement * scale
                aligned = warp_2d(latest_image, displacement)
                support = warp_2d(latest_valid.to(satellite), displacement) > 0.999
                grid = _normal_grid(batch_size, *satellite.shape[-2:], satellite.device, satellite.dtype)
                sample_grid = grid - (displacement * displacement.new_tensor(
                    [2.0 / (satellite.shape[-1] - 1), 2.0 / (satellite.shape[-2] - 1)]).view(1, 2, 1, 1)).permute(0, 2, 3, 1)
                support = support & (sample_grid.abs() <= 1).all(-1).unsqueeze(1)
                support = support & valid[:, step] & pair_active.bool().view(-1, 1, 1, 1)
                error = F.smooth_l1_loss(aligned, satellite[:, step], reduction="none")
                count = support.flatten(1).sum(1)
                alignment_sum = alignment_sum + (error * support).flatten(1).sum(1) / count.clamp_min(1)
                alignment_count = alignment_count + (count > 0).to(satellite)
            image_active = active_frame.view(-1, 1, 1, 1)
            latest_image = torch.where(image_active, satellite[:, step], latest_image)
            latest_valid = torch.where(image_active, valid[:, step], latest_valid)
            latest_flow = pair_active.view(batch_size, 1, 1, 1) * flow + (1.0 - pair_active.view(batch_size, 1, 1, 1)) * latest_flow
            latest_confidence = pair_active * confidence + (1.0 - pair_active) * latest_confidence
            valid_pairs = torch.maximum(valid_pairs, pair_active)
            previous_offset = torch.where(unique_mask[:, step], offsets[:, step], previous_offset)
            seen = seen | active_frame

        valid_fraction = unique_mask.float().mean(dim=1)
        latest_age = (-torch.where(frame_mask, offsets, torch.full_like(offsets, -24.0)).amax(dim=1)).clamp(0.0, 24.0)
        available = frame_mask.any(dim=1)
        if disable_motion:
            latest_flow = torch.zeros_like(latest_flow)
            latest_confidence = torch.zeros_like(latest_confidence)
        source = self.source_embedding(source_id)
        lead = self.lead_embedding(torch.arange(6, device=satellite.device)).unsqueeze(0).expand(batch_size, -1, -1)
        tokens, cloud_logits, direct_states = [], [], []
        fine = None if disable_station_readout else self.station_readout.image(
            torch.cat([latest_image, latest_valid.to(latest_image)], dim=1))
        latent = hidden
        history_summary = self.summary(hidden)
        for step in range(6):
            query_context = torch.cat([lead[:, step], batch["solar_context"][:, step], source], dim=-1)
            age_quality = torch.stack([latest_age / 24.0, valid_fraction], dim=-1)
            direct_state = history_summary + self.direct_context(torch.cat(
                [query_context, batch["static"], age_quality], dim=-1))
            if not disable_station_readout:
                direct_state = direct_state + self.station_readout(fine, hidden, query_context)
            direct_states.append(direct_state)
            # The first target is age+1 hours after the last actual scan.
            # Confidence describes reliability, not a reduction in cloud speed.
            elapsed = latest_age + 1.0 if step == 0 else torch.ones_like(latest_age)
            latent = warp_2d(latent, latest_flow * elapsed.view(batch_size, 1, 1, 1))
            condition = self.rollout_condition(
                torch.cat([lead[:, step], batch["solar_context"][:, step], source], dim=-1)
            ).unsqueeze(-1).unsqueeze(-1)
            latent = self.future_cell(condition.expand(-1, -1, 8, 8), latent)
            cloud_logits.append(self.future_cloud_head(latent))
            quality = torch.stack(
                [valid_fraction, (1.0 - latest_age / 24.0).clamp(0.0, 1.0), latest_confidence,
                 valid_pairs, (latest_flow.square().mean(dim=(1, 2, 3)) + 1e-8).sqrt()], dim=-1
            )
            tokens.append(self.token(torch.cat([
                self.summary(latent) + (0.0 if disable_station_readout else self.station_readout(fine, latent,
                    torch.cat([lead[:, step], batch["solar_context"][:, step], source], dim=-1))),
                lead[:, step], batch["solar_context"][:, step], source, quality
            ], dim=-1)))
        state = self.trajectory(torch.stack(tokens, dim=1))
        dynamic_q = torch.sigmoid(self.output_head(state).squeeze(-1))
        direct_state = torch.stack(direct_states, dim=1)
        direct_q = torch.sigmoid(self.direct_head(direct_state).squeeze(-1))
        quality_by_lead = age_quality.unsqueeze(1).expand(-1, 6, -1)
        direct_weight = torch.sigmoid(self.path_gate(torch.cat([direct_state, state, quality_by_lead], dim=-1)).squeeze(-1))
        if prediction_path == "direct":
            direct_weight = torch.ones_like(direct_weight)
        elif prediction_path == "dynamic":
            direct_weight = torch.zeros_like(direct_weight)
        elif prediction_path != "dual":
            raise ValueError(f"unknown satellite prediction path: {prediction_path}")
        q = direct_weight * direct_q + (1.0 - direct_weight) * dynamic_q
        state = direct_weight.unsqueeze(-1) * direct_state + (1.0 - direct_weight.unsqueeze(-1)) * state
        scale_q = 0.01 + 0.49 * torch.sigmoid(self.scale_head(state).squeeze(-1))
        motion_vector = latest_flow.mean(dim=(-1, -2)).unsqueeze(1).expand(-1, 6, -1)
        return {
            "state": state,
            "direct_prediction": direct_q * batch["physical_upper"],
            "dynamic_prediction": dynamic_q * batch["physical_upper"],
            "direct_weight": direct_weight,
            "history_alignment_loss": (alignment_sum / alignment_count.clamp_min(1)).sum() / (alignment_count > 0).sum().clamp_min(1),
            "history_transport_skill": (transport_skill_sum / transport_pairs.clamp_min(1)).unsqueeze(1).expand(-1, 6),
            "history_transport_available": (transport_pairs > 0).unsqueeze(1).expand(-1, 6),
            "history_transport_overlap": (transport_overlap_sum / transport_pairs.clamp_min(1)).unsqueeze(1).expand(-1, 6),
            "history_transport_dt_hours": (transport_dt_sum / transport_pairs.clamp_min(1)).unsqueeze(1).expand(-1, 6),
            "history_transport_pairs": transport_pairs.unsqueeze(1).expand(-1, 6),
            "q": q,
            "prediction": q * batch["physical_upper"],
            "scale_q": scale_q,
            "scale_wm2": scale_q * batch["physical_upper"].clamp_min(1.0),
            "clear_logit": self.clear_head(state).squeeze(-1),
            "available": available.unsqueeze(1).expand(-1, 6),
            "valid_fraction": valid_fraction.unsqueeze(1).expand(-1, 6),
            "freshness": (1.0 - latest_age / 24.0).clamp(0.0, 1.0).unsqueeze(1).expand(-1, 6),
            "motion_confidence": latest_confidence.unsqueeze(1).expand(-1, 6),
            "motion_vector": motion_vector,
            "dynamics": (latest_flow.square().mean(dim=(1, 2, 3)) + 1e-8).sqrt().unsqueeze(1).expand(-1, 6),
            "future_cloud_logits": torch.stack(cloud_logits, dim=1),
            "unique_frame_fraction": valid_fraction.unsqueeze(1).expand(-1, 6),
        }


class VolumeBranchEncoder(nn.Module):
    def __init__(self, channels: int, output_dim: int):
        super().__init__()
        hidden = 32
        self.net = nn.Sequential(
            nn.Conv3d(channels + 1, hidden, 1), nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
            nn.Conv3d(hidden, 48, 3, padding=1), nn.GroupNorm(_groups(48), 48), nn.GELU(),
            nn.Conv3d(48, 48, 3, padding=1), nn.GELU(),
        )
        self.projection = nn.Sequential(nn.Linear(5 * 48, 2 * output_dim), nn.GELU(), nn.Linear(2 * output_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid[:, None].to(values)
        feature = self.net(torch.cat([values, mask], dim=1))
        _, _, levels, height, width = feature.shape
        global_context = feature.mean(dim=(2, 3, 4))
        centre = feature[:, :, :, height // 2, width // 2].mean(dim=2)
        z = torch.linspace(-1.0, 1.0, levels, device=feature.device, dtype=feature.dtype)
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        moments = [
            (feature * z.view(1, 1, levels, 1, 1)).mean(dim=(2, 3, 4)),
            (feature * y.view(1, 1, 1, height, 1)).mean(dim=(2, 3, 4)),
            (feature * x.view(1, 1, 1, 1, width)).mean(dim=(2, 3, 4)),
        ]
        return self.projection(torch.cat([global_context, centre, *moments], dim=-1))


class SurfaceBranchEncoder(nn.Module):
    def __init__(self, channels: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
        )
        self.summary = SpatialSummary(48, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.summary(self.net(values))


class StaticGridEncoder(nn.Module):
    """Encode continuous terrain fields and categorical LU_INDEX separately."""

    def __init__(self, output_dim: int, landuse_classes: int = 64, embedding_dim: int = 8):
        super().__init__()
        self.landuse_classes = landuse_classes
        self.landuse = nn.Embedding(landuse_classes, embedding_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Conv2d(7 + embedding_dim, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
        )
        self.summary = SpatialSummary(48, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        landuse_index = values[:, 2].round().long().clamp(0, self.landuse_classes - 1)
        landuse = self.landuse(landuse_index).permute(0, 3, 1, 2)
        continuous = torch.cat([values[:, :2], values[:, 3:]], dim=1)
        return self.summary(self.net(torch.cat([continuous, landuse], dim=1)))


class TransportEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 24, 3, padding=1), nn.GELU(),
            nn.Conv2d(24, 32, 3, padding=1), nn.GELU(),
        )
        self.summary = SpatialSummary(32, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.summary(self.net(values))


class DomainPhysicalEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.thermo = VolumeBranchEncoder(5, d_model)
        self.hydrometeor = VolumeBranchEncoder(6, d_model)
        self.dynamics = VolumeBranchEncoder(8, d_model)
        self.surface = SurfaceBranchEncoder(NWP_SURFACE_CHANNELS, d_model)
        self.transport = TransportEncoder(d_model)
        self.static_grid = StaticGridEncoder(d_model)
        self.availability = nn.Sequential(nn.Linear(NWP3D_CHANNELS + NWP_SURFACE_CHANNELS, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.physics_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, volume: torch.Tensor, valid: torch.Tensor, surface: torch.Tensor,
        transport: torch.Tensor, static_grid: torch.Tensor, availability: torch.Tensor,
        derived: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_time = volume.shape[0]
        thermo = self.thermo(torch.cat([volume[:, (0, 1, 8, 7)], derived["relative_humidity"]], dim=1), valid)
        hydro = self.hydrometeor(volume[:, (2, 3, 4, 5, 6, 7)], valid)
        dynamics = self.dynamics(torch.cat([
            volume[:, (9, 10, 11)], derived["wind_speed"], derived["divergence"],
            derived["vorticity"], derived["vertical_shear"], derived["moisture_flux_convergence"],
        ], dim=1), valid)
        tokens = torch.stack([thermo, hydro, dynamics], dim=1)
        attended, _ = self.physics_attention(tokens, tokens, tokens, need_weights=False)
        physical = self.norm(tokens + attended).mean(dim=1)
        auxiliary = self.surface(surface) + self.transport(transport) + self.static_grid(static_grid) + self.availability(availability.to(volume))
        return physical, auxiliary.reshape(batch_time, -1)


def _domain_inputs(batch: dict[str, torch.Tensor], domain: str) -> tuple[torch.Tensor, ...]:
    volume = _batch_value(batch, (f"nwp_{domain}_3d", f"nwp_{domain}_3d_raw"))
    surface = _batch_value(batch, (f"nwp_{domain}_surface", f"nwp_{domain}_surface_raw"))
    valid = _batch_value(batch, (f"nwp_{domain}_3d_valid", f"nwp_{domain}_level_valid"))
    raw = _batch_value(batch, (f"nwp_{domain}_3d_physical", f"nwp_{domain}_3d_raw"))
    if volume is None:
        volume = batch["nwp_3d"]
    if surface is None:
        surface = batch["nwp_surface"]
    original_volume_channels = volume.shape[2]
    original_surface_channels = surface.shape[2]
    channel_available = _availability_mask(
        batch, (f"nwp_{domain}_channel_available", "nwp_3d_channel_available"),
        original_volume_channels, NWP3D_CHANNELS, LEGACY_3D_TO_V4, volume,
    )
    surface_available = _availability_mask(
        batch, (f"nwp_{domain}_surface_available", "nwp_surface_available"),
        original_surface_channels, NWP_SURFACE_CHANNELS, LEGACY_SURFACE_TO_V4, surface,
    )
    volume = _pad_or_map_channels(volume, NWP3D_CHANNELS, LEGACY_3D_TO_V4)
    surface = _pad_or_map_channels(surface, NWP_SURFACE_CHANNELS, LEGACY_SURFACE_TO_V4)
    volume = _ensure_seven_states(volume)
    surface = _ensure_seven_states(surface)
    target_size = volume.shape[-2:]
    surface = _resize_surface(surface, target_size)
    if valid is None:
        valid = batch["nwp_3d_valid"]
    valid = _ensure_seven_states(valid)
    if valid.shape[-2:] != target_size:
        batch_size, time, levels, height, width = valid.shape
        valid = F.interpolate(valid.to(volume).reshape(batch_size * time, levels, height, width), size=target_size, mode="nearest").bool().reshape(batch_size, time, levels, *target_size)
    if raw is None or raw.shape[2] != NWP3D_CHANNELS:
        raw = volume
    else:
        raw = _ensure_seven_states(raw)
        raw = _resize_volume(raw, target_size)
    anchor = _batch_value(batch, ("wrf_ghi_anchor_d02", "wrf_ghi_anchor", "wrf_anchor"))
    if anchor is None:
        raise KeyError("missing formal wrf_ghi_anchor_d02 (legacy aliases: wrf_ghi_anchor/wrf_anchor)")
    return volume, valid, surface, raw, anchor, channel_available, surface_available


def cloud_transport(raw: torch.Tensor, valid: torch.Tensor, grid_km: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return six advected-cloud/growth maps and cloud-layer mean wind."""

    cloud = raw[:, :, 7].clamp(0.0, 1.0)
    u = raw[:, :, 9]
    v = raw[:, :, 10]
    level_mask = valid.to(raw)
    cloud_weight = cloud * level_mask
    weight_sum = cloud_weight.sum(dim=2).clamp_min(1e-4)
    cloud_column = cloud_weight.sum(dim=2) / level_mask.sum(dim=2).clamp_min(1.0)
    u_cloud = (u * cloud_weight).sum(dim=2) / weight_sum
    v_cloud = (v * cloud_weight).sum(dim=2) / weight_sum
    no_cloud = weight_sum <= 1e-3
    u_mean = (u * level_mask).sum(dim=2) / level_mask.sum(dim=2).clamp_min(1.0)
    v_mean = (v * level_mask).sum(dim=2) / level_mask.sum(dim=2).clamp_min(1.0)
    u_cloud = torch.where(no_cloud, u_mean, u_cloud)
    v_cloud = torch.where(no_cloud, v_mean, v_cloud)
    maps, winds = [], []
    for lead in range(6):
        flow = torch.stack([u_cloud[:, lead], v_cloud[:, lead]], dim=1) * (3.6 / grid_km)
        advected = warp_2d(cloud_column[:, lead:lead + 1], flow)
        growth = cloud_column[:, lead + 1:lead + 2] - advected
        maps.append(torch.cat([advected, growth], dim=1))
        winds.append(torch.stack([u_cloud[:, lead + 1].mean(dim=(-1, -2)), v_cloud[:, lead + 1].mean(dim=(-1, -2))], dim=-1))
    return torch.stack(maps, dim=1), torch.stack(winds, dim=1)


def _central_difference(values: torch.Tensor, spacing_m: float, axis: int) -> torch.Tensor:
    forward = torch.roll(values, shifts=-1, dims=axis)
    backward = torch.roll(values, shifts=1, dims=axis)
    derivative = (forward - backward) / (2.0 * spacing_m)
    if axis == -1:
        derivative[..., 0] = (values[..., 1] - values[..., 0]) / spacing_m
        derivative[..., -1] = (values[..., -1] - values[..., -2]) / spacing_m
    elif axis == -2:
        derivative[..., 0, :] = (values[..., 1, :] - values[..., 0, :]) / spacing_m
        derivative[..., -1, :] = (values[..., -1, :] - values[..., -2, :]) / spacing_m
    return derivative


def derive_physical_fields(raw: torch.Tensor, grid_km: float) -> dict[str, torch.Tensor]:
    """Build bounded online thermodynamic and dynamic diagnostics."""

    temperature = raw[:, :, 0].clamp(180.0, 340.0)
    specific_humidity = raw[:, :, 1].clamp(0.0, 0.05)
    pressure_hpa = raw.new_tensor([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]).view(1, 1, 10, 1, 1)
    mixing_ratio = specific_humidity / (1.0 - specific_humidity).clamp_min(1e-4)
    vapour_pressure = mixing_ratio * pressure_hpa / (0.622 + mixing_ratio)
    celsius = temperature - 273.15
    saturation = 6.112 * torch.exp((17.67 * celsius / (celsius + 243.5).clamp_min(20.0)).clamp(-20.0, 20.0))
    relative_humidity = (vapour_pressure / saturation.clamp_min(1e-4)).clamp(0.0, 2.0)
    u, v = raw[:, :, 9], raw[:, :, 10]
    speed = torch.sqrt(u.square() + v.square() + 1e-6)
    spacing = grid_km * 1000.0
    du_dx = _central_difference(u, spacing, -1)
    du_dy = _central_difference(u, spacing, -2)
    dv_dx = _central_difference(v, spacing, -1)
    dv_dy = _central_difference(v, spacing, -2)
    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy
    shear = torch.zeros_like(speed)
    shear[:, :, 1:] = speed[:, :, 1:] - speed[:, :, :-1]
    moisture_flux_convergence = -(
        _central_difference(specific_humidity * u, spacing, -1)
        + _central_difference(specific_humidity * v, spacing, -2)
    )
    return {
        "relative_humidity": relative_humidity.unsqueeze(2),
        "wind_speed": (speed / 30.0).clamp(0.0, 4.0).unsqueeze(2),
        "divergence": (divergence * 1e4).clamp(-5.0, 5.0).unsqueeze(2),
        "vorticity": (vorticity * 1e4).clamp(-5.0, 5.0).unsqueeze(2),
        "vertical_shear": (shear / 20.0).clamp(-4.0, 4.0).unsqueeze(2),
        "moisture_flux_convergence": (moisture_flux_convergence * 1e5).clamp(-5.0, 5.0).unsqueeze(2),
    }


class WindCloudWRFExpert(nn.Module):
    """Dual-scale thermodynamic/hydrometeor/dynamic WRF expert."""

    def __init__(self, d_model: int = 128, dropout: float = 0.15):
        super().__init__()
        self.d02 = DomainPhysicalEncoder(d_model)
        self.d01 = DomainPhysicalEncoder(d_model)
        self.cross_scale = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.lead_embedding = nn.Embedding(6, 32)
        self.token = nn.Sequential(
            nn.Linear(4 * d_model + 32 + 8 + 7 + 2, 2 * d_model),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.trajectory = nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)
        self.residual_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.anchor_trust_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.clear_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.scale_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.constant_(self.scale_head[-1].bias, -1.8)

    @staticmethod
    def _anchor_at_station(anchor: torch.Tensor) -> torch.Tensor:
        if anchor.ndim == 2:
            return anchor
        if anchor.ndim != 4:
            raise ValueError(f"WRF anchor must be [B,6] or [B,6,H,W], got {tuple(anchor.shape)}")
        return anchor[:, :, anchor.shape[-2] // 2, anchor.shape[-1] // 2]

    @staticmethod
    def _static_grid(batch: dict[str, torch.Tensor], domain: str, volume: torch.Tensor) -> torch.Tensor:
        values = _batch_value(batch, (f"nwp_{domain}_static", f"nwp_{domain}_static_raw"))
        if values is None:
            return volume.new_zeros(volume.shape[0], 8, *volume.shape[-2:])
        if values.shape[1] != 8:
            resized_channels = values.new_zeros(values.shape[0], 8, *values.shape[-2:])
            count = min(values.shape[1], 8)
            resized_channels[:, :count] = values[:, :count]
            values = resized_channels
        if values.shape[-2:] != volume.shape[-2:]:
            continuous = torch.cat([values[:, :2], values[:, 3:]], dim=1)
            continuous = F.interpolate(continuous, size=volume.shape[-2:], mode="bilinear", align_corners=False)
            landuse = F.interpolate(values[:, 2:3], size=volume.shape[-2:], mode="nearest")
            values = torch.cat([continuous[:, :2], landuse, continuous[:, 2:]], dim=1)
        return values

    def _encode_domain(
        self, encoder, volume, valid, surface, transport, raw, static_grid,
        channel_available, surface_available, grid_km,
    ):
        batch, time = volume.shape[:2]
        derived = derive_physical_fields(raw[:, 1:], grid_km)
        derived = {name: value.reshape(batch * 6, 1, NWP_LEVELS, *volume.shape[-2:]) for name, value in derived.items()}
        physical, auxiliary = encoder(
            volume[:, 1:].reshape(batch * 6, NWP3D_CHANNELS, NWP_LEVELS, *volume.shape[-2:]),
            valid[:, 1:].reshape(batch * 6, NWP_LEVELS, *volume.shape[-2:]),
            surface[:, 1:].reshape(batch * 6, NWP_SURFACE_CHANNELS, *surface.shape[-2:]),
            transport.reshape(batch * 6, 2, *transport.shape[-2:]),
            static_grid.unsqueeze(1).expand(-1, 6, -1, -1, -1).reshape(batch * 6, 8, *static_grid.shape[-2:]),
            torch.cat([channel_available, surface_available], dim=-1).unsqueeze(1).expand(-1, 6, -1).reshape(batch * 6, -1),
            derived,
        )
        return physical.reshape(batch, 6, -1), auxiliary.reshape(batch, 6, -1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d02_volume, d02_valid, d02_surface, d02_raw, anchor, d02_channels, d02_surface_channels = _domain_inputs(batch, "d02")
        d01_volume, d01_valid, d01_surface, d01_raw, _, d01_channels, d01_surface_channels = _domain_inputs(batch, "d01")
        d02_transport, d02_wind = cloud_transport(d02_raw, d02_valid, 3.0)
        d01_transport, d01_wind = cloud_transport(d01_raw, d01_valid, 9.0)
        d02_physical, d02_auxiliary = self._encode_domain(
            self.d02, d02_volume, d02_valid, d02_surface, d02_transport, d02_raw,
            self._static_grid(batch, "d02", d02_volume), d02_channels, d02_surface_channels, 3.0,
        )
        d01_physical, d01_auxiliary = self._encode_domain(
            self.d01, d01_volume, d01_valid, d01_surface, d01_transport, d01_raw,
            self._static_grid(batch, "d01", d01_volume), d01_channels, d01_surface_channels, 9.0,
        )
        d02_cross, _ = self.cross_scale(d02_physical, d01_physical, d01_physical, need_weights=False)
        lead = self.lead_embedding(torch.arange(6, device=d02_volume.device)).unsqueeze(0).expand(d02_volume.shape[0], -1, -1)
        static = batch["static"].unsqueeze(1).expand(-1, 6, -1)
        anchor_station = self._anchor_at_station(anchor)
        anchor_q = (anchor_station / batch["physical_upper"].clamp_min(1.0)).clamp(0.0, 1.0)
        availability = d02_valid[:, 1:].flatten(2).any(dim=-1).to(d02_volume)
        state = self.token(torch.cat([
            d02_physical + d02_cross, d02_auxiliary, d01_physical, d01_auxiliary,
            lead, batch["solar_context"], static, anchor_q.unsqueeze(-1), availability.unsqueeze(-1),
        ], dim=-1))
        state = self.trajectory(state)
        anchor_trust = torch.sigmoid(self.anchor_trust_head(state).squeeze(-1))
        delta = 2.0 * torch.tanh(self.residual_head(state).squeeze(-1)) * (0.5 + 1.5 * (1.0 - anchor_trust))
        q = bounded_update(anchor_q, delta)
        scale_q = 0.01 + 0.49 * torch.sigmoid(self.scale_head(state).squeeze(-1))
        transport_residual = d02_transport[:, :, 1].abs().mean(dim=(-1, -2))
        return {
            "state": state,
            "q": q,
            "prediction": q * batch["physical_upper"],
            "scale_q": scale_q,
            "scale_wm2": scale_q * batch["physical_upper"].clamp_min(1.0),
            "clear_logit": self.clear_head(state).squeeze(-1),
            "available": availability.bool(),
            "delta": delta,
            "anchor_trust": anchor_trust,
            "transport_residual": transport_residual,
            "cloud_wind_vector": d02_wind,
            "d01_cloud_wind_vector": d01_wind,
            "d02_transport": d02_transport,
            "d01_transport": d01_transport,
        }


def motion_wind_consistency(motion: torch.Tensor, wind: torch.Tensor,
                            latent_to_en_m: torch.Tensor | None = None) -> torch.Tensor:
    # Matrix columns are east/north metres per latent x/y cell. Unknown
    # georeferencing must not produce a fabricated physical comparison.
    if latent_to_en_m is None:
        return torch.zeros_like(motion)
    motion = torch.einsum("bij,btj->bti", latent_to_en_m.to(motion), motion) / 3600.0
    motion_norm = motion.norm(dim=-1)
    wind_cells = wind
    wind_norm = wind_cells.norm(dim=-1)
    cosine = (motion * wind_cells).sum(dim=-1) / (motion_norm * wind_norm).clamp_min(1e-4)
    cosine = torch.where((motion_norm > 1e-3) & (wind_norm > 1e-3), cosine, torch.zeros_like(cosine))
    speed_difference = (motion_norm - wind_norm).abs() / (motion_norm + wind_norm).clamp_min(1.0)
    result = torch.stack([cosine.clamp(-1.0, 1.0), speed_difference.clamp(0.0, 4.0)], dim=-1)
    known = torch.isfinite(latent_to_en_m).all(dim=(-1, -2)) & (torch.linalg.det(latent_to_en_m.float()).abs() > 1e-6)
    return torch.where(known[:, None, None], result, torch.zeros_like(result))


ABLATIONS = (
    "full", "no_motion", "no_uncertainty", "fixed_router",
    "no_transition", "no_aod", "no_conditional", "no_station_readout", "no_history_alignment",
    "satellite_direct_only", "satellite_dynamic_only",
)


class ScenarioDrivenGHIModelV4(nn.Module):
    """Two complete experts with uncertainty-aware, risk-shrunk routing."""

    architecture_revision = "v4_satellite_direct_dynamic_r3"
    routing_supervision_mode = "winner"

    def __init__(self, d_model: int = 128, dropout: float = 0.15, ablation: str = "full", satellite_path: str | None = None, routing_quality: str = "legacy"):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation: {ablation}")
        self.ablation = ablation
        if routing_quality not in {"legacy", "history_features", "history_risk"}:
            raise ValueError(routing_quality)
        self.routing_quality = routing_quality
        self.satellite_path = satellite_path or {"satellite_direct_only": "direct", "satellite_dynamic_only": "dynamic"}.get(ablation, "dual")
        if self.satellite_path not in {"direct", "dynamic", "dual"}:
            raise ValueError(self.satellite_path)
        self.satellite_expert = TimeAwareSatelliteExpert(d_model=d_model, dropout=dropout)
        self.wrf_expert = WindCloudWRFExpert(d_model=d_model, dropout=dropout)
        self.source_embedding = nn.Embedding(3, 16)
        self.lead_embedding = nn.Embedding(6, 16)
        route_dim = 2 * d_model + 3 + 2 + 10 + 8 + 7 + 16 + 16
        self.route_evidence = nn.Sequential(
            nn.Linear(route_dim, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.route_confidence = nn.Sequential(
            nn.Linear(route_dim, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.route_evidence[-1].weight)
        nn.init.zeros_(self.route_evidence[-1].bias)
        nn.init.zeros_(self.route_confidence[-1].weight)
        nn.init.constant_(self.route_confidence[-1].bias, -1.0)
        self.uncertainty_strength_raw = nn.Parameter(torch.tensor(-0.45))
        self.lead_prior = MonotonicLeadPrior()
        correction_dim = 2 * d_model + 3 + 8 + 7
        self.transition_gate = nn.Sequential(nn.Linear(correction_dim, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.transition_head = nn.Sequential(nn.Linear(correction_dim + 1, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.aod_gate = nn.Sequential(nn.Linear(correction_dim + 3, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.aod_strength = nn.Sequential(nn.Linear(correction_dim + 3, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        for module in (self.transition_gate, self.transition_head, self.aod_gate, self.aod_strength):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        nn.init.constant_(self.aod_gate[-1].bias, -12.0)
        # Optional module leaves legacy checkpoint keys and initial expert RNG intact.
        self.route_quality_projection = None
        if routing_quality != "legacy":
            self.route_quality_projection = nn.Linear(7, route_dim, bias=False)
            nn.init.zeros_(self.route_quality_projection.weight)

    @staticmethod
    def project(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bounded = torch.minimum(prediction.clamp_min(0.0), batch["physical_upper"])
        return torch.where(batch["solar_context"][..., 2] > 0.0, bounded, torch.zeros_like(bounded))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        satellite = self.satellite_expert(batch, disable_motion=self.ablation == "no_motion",
            disable_station_readout=self.ablation == "no_station_readout",
            disable_history_alignment=self.ablation == "no_history_alignment",
            prediction_path=self.satellite_path)
        wrf = self.wrf_expert(batch)
        batch_size = batch["physical_upper"].shape[0]
        source = self.source_embedding(batch["satellite_source_id"]).unsqueeze(1).expand(-1, 6, -1)
        lead = self.lead_embedding(torch.arange(6, device=batch["physical_upper"].device)).unsqueeze(0).expand(batch_size, -1, -1)
        prediction_state = torch.stack([satellite["q"], wrf["q"], (satellite["q"] - wrf["q"]).abs()], dim=-1)
        clear_state = torch.stack([torch.sigmoid(satellite["clear_logit"]), torch.sigmoid(wrf["clear_logit"])], dim=-1)
        consistency = motion_wind_consistency(satellite["motion_vector"], wrf["cloud_wind_vector"],
                                              batch.get("satellite_latent_to_en_m"))
        if self.ablation == "no_motion":
            consistency = torch.zeros_like(consistency)
        uncertainty = torch.stack([satellite["scale_q"], wrf["scale_q"]], dim=-1)
        quality = torch.stack([
            satellite["valid_fraction"], satellite["freshness"], satellite["dynamics"],
            satellite["motion_confidence"], satellite["unique_frame_fraction"], wrf["available"].to(wrf["q"]),
            wrf["anchor_trust"], wrf["transport_residual"], consistency[..., 0], consistency[..., 1],
        ], dim=-1)
        static = batch["static"].unsqueeze(1).expand(-1, 6, -1)
        route_input = torch.cat([
            satellite["state"], wrf["state"], prediction_state, clear_state, quality,
            batch["solar_context"], static, source, lead,
        ], dim=-1)
        quality_cap = torch.ones_like(satellite["q"])
        if self.route_quality_projection is not None:
            from .route_quality import history_route_quality
            history_features, quality_cap = history_route_quality(satellite, batch)
            route_input = route_input + self.route_quality_projection(history_features)
        prior_logits = self.lead_prior.logits().view(1, 6).expand(batch_size, -1)
        if self.ablation == "fixed_router":
            evidence = torch.zeros_like(prior_logits)
            confidence = torch.zeros_like(prior_logits)
        else:
            evidence = 2.0 * torch.tanh(self.route_evidence(route_input).squeeze(-1))
            confidence = torch.sigmoid(self.route_confidence(route_input).squeeze(-1))
            if self.routing_quality == "history_risk":
                confidence = confidence * quality_cap
        reliability = torch.zeros_like(prior_logits)
        if self.ablation != "no_uncertainty":
            strength = F.softplus(self.uncertainty_strength_raw)
            reliability = strength * torch.log((wrf["scale_q"] + 1e-4) / (satellite["scale_q"] + 1e-4)).clamp(-2.0, 2.0)
        candidate_logit = prior_logits + evidence + reliability
        satellite_logit = prior_logits + confidence * (candidate_logit - prior_logits)
        logits = torch.stack([satellite_logit, torch.zeros_like(satellite_logit)], dim=-1)
        availability = torch.stack([satellite["available"], wrf["available"]], dim=-1)
        logits = logits.masked_fill(~availability, -1e4)
        weights = torch.softmax(logits, dim=-1) * availability.to(logits)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        base_q = weights[..., 0] * satellite["q"] + weights[..., 1] * wrf["q"]
        base_variance = (
            weights[..., 0].square() * satellite["scale_q"].square()
            + weights[..., 1].square() * wrf["scale_q"].square()
            + weights[..., 0] * weights[..., 1] * (satellite["q"] - wrf["q"]).square()
        )
        base_scale_q = base_variance.clamp_min(1e-6).sqrt()

        correction_context = torch.cat([
            satellite["state"], wrf["state"], prediction_state, batch["solar_context"], static
        ], dim=-1)
        if self.ablation in {"no_transition", "no_conditional"}:
            transition_gate = torch.zeros_like(base_q)
            transition_delta = torch.zeros_like(base_q)
            transition_q = base_q
        else:
            transition_gate = batch["transition_support"].clamp(0.0, 1.0) * torch.sigmoid(
                self.transition_gate(correction_context).squeeze(-1)
            )
            transition_delta = math.log(2.0) * torch.tanh(
                self.transition_head(torch.cat([correction_context, base_q.unsqueeze(-1)], dim=-1)).squeeze(-1)
            )
            transition_q = bounded_update(base_q, transition_gate * transition_delta)

        sat_clear = torch.sigmoid(satellite["clear_logit"])
        wrf_clear = torch.sigmoid(wrf["clear_logit"])
        common_clear = torch.minimum(sat_clear, wrf_clear)
        clear_support = torch.sigmoid((common_clear - 0.70) / 0.08)
        stable_daylight = ((batch["solar_context"][..., 2] - 0.8) / 0.2).clamp(0.0, 1.0)
        aod_anomaly = batch["aod550"] - batch["aod_reference"]
        aerosol_inputs = torch.cat([
            correction_context, common_clear.unsqueeze(-1), aod_anomaly.unsqueeze(-1), batch["air_mass"].unsqueeze(-1)
        ], dim=-1)
        if self.ablation in {"no_aod", "no_conditional"}:
            aerosol_gate = torch.zeros_like(base_q)
            delta_log_t = torch.zeros_like(base_q)
            final_q = transition_q
        else:
            aerosol_gate = (
                batch["aod_mask"].to(base_q) * stable_daylight * clear_support
                * torch.sigmoid(self.aod_gate(aerosol_inputs).squeeze(-1))
            )
            beta = torch.sigmoid(self.aod_strength(aerosol_inputs).squeeze(-1))
            limit = math.log(1.3)
            delta_log_t = limit * torch.tanh((-beta * batch["air_mass"] * aod_anomaly) / limit)
            final_q = bounded_update(transition_q, aerosol_gate * delta_log_t)

        return {
            "satellite_prediction": self.project(satellite["prediction"], batch),
            "satellite_direct_prediction": self.project(satellite["direct_prediction"], batch),
            "satellite_dynamic_prediction": self.project(satellite["dynamic_prediction"], batch),
            "satellite_direct_weight": satellite["direct_weight"],
            "satellite_prediction_path": self.satellite_path,
            "routing_supervision_mode": self.routing_supervision_mode,
            "routing_quality_mode": self.routing_quality,
            "route_quality_cap": quality_cap,
            **{key: satellite[key] for key in (
                "history_transport_skill", "history_transport_available", "history_transport_overlap",
                "history_transport_dt_hours", "history_transport_pairs")},
            "satellite_history_alignment_loss": satellite["history_alignment_loss"],
            "wrf_prediction": self.project(wrf["prediction"], batch),
            "base_prediction": self.project(base_q * batch["physical_upper"], batch),
            "transition_prediction": self.project(transition_q * batch["physical_upper"], batch),
            "final_prediction": self.project(final_q * batch["physical_upper"], batch),
            "satellite_scale_wm2": satellite["scale_wm2"],
            "wrf_scale_wm2": wrf["scale_wm2"],
            "base_scale_wm2": base_scale_q * batch["physical_upper"].clamp_min(1.0),
            "fusion_weights": weights,
            "lead_prior": self.lead_prior().view(1, 6).expand(batch_size, -1),
            "route_evidence": evidence,
            "route_reliability": reliability,
            "route_confidence": confidence,
            "satellite_valid_fraction": satellite["valid_fraction"],
            "satellite_unique_frame_fraction": satellite["unique_frame_fraction"],
            "satellite_freshness": satellite["freshness"],
            "satellite_dynamics": satellite["dynamics"],
            "satellite_motion_confidence": satellite["motion_confidence"],
            "satellite_motion_vector": satellite["motion_vector"],
            "satellite_future_cloud_logits": satellite["future_cloud_logits"],
            "wrf_available": wrf["available"],
            "wrf_anchor_trust": wrf["anchor_trust"],
            "wrf_transport_residual": wrf["transport_residual"],
            "motion_wind_consistency": consistency,
            "satellite_clear_probability": sat_clear,
            "wrf_clear_probability": wrf_clear,
            "satellite_clear_logit": satellite["clear_logit"],
            "wrf_clear_logit": wrf["clear_logit"],
            "common_clear_probability": common_clear,
            "transition_gate": transition_gate,
            "transition_delta": transition_delta,
            "aod_gate": aerosol_gate,
            "aod_delta_log_t": delta_log_t,
            "satellite_state": satellite["state"],
            "wrf_state": wrf["state"],
        }


ScenarioDrivenGHIModel = ScenarioDrivenGHIModelV4
