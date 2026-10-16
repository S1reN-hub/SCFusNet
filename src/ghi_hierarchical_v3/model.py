"""Formal dual-primary-expert architecture aligned with manuscript Section 3.

The satellite and WRF branches are independently supervised complete GHI
forecasters.  Their predictions are combined by a lead- and scene-conditioned
router.  Solar-transition and aerosol effects are bounded conditional
corrections applied only after the two primary experts have been combined.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


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
    """Endpoint-recoverable update on a normalized physical interval."""

    q = q.clamp(0.0, 1.0)
    lower = q * torch.exp(delta.clamp(max=0.0))
    upper = 1.0 - (1.0 - q) * torch.exp((-delta).clamp(max=0.0))
    return torch.where(delta <= 0.0, lower, upper).clamp(0.0, 1.0)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(self.gates(torch.cat([x, hidden], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class SourceAdapter(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        middle = output_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, middle, 5, stride=2, padding=2),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
            nn.Conv2d(middle, output_channels, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([values, valid.to(values)], dim=1))


class SpatialSummary(nn.Module):
    """Retain centre, global context, and signed cloud-position moments."""

    def __init__(self, channels: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(4 * channels, 2 * output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature.shape
        global_context = feature.mean(dim=(-1, -2))
        cy, cx = height // 2, width // 2
        centre = feature[:, :, max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2].mean(dim=(-1, -2))
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        y_moment = (feature * y.view(1, 1, height, 1)).mean(dim=(-1, -2))
        x_moment = (feature * x.view(1, 1, 1, width)).mean(dim=(-1, -2))
        return self.projection(torch.cat([global_context, centre, y_moment, x_moment], dim=-1))


class SatelliteExpert(nn.Module):
    """Past-cloud expert with source adapters and spatial recurrent rollout.

    ``physics_constrained=False`` is deliberately a separate forward path.  It
    does not turn off just the last clamp: the sigmoid/physical-upper recovery
    is removed and the head emits GHI directly in W/m2.  This is used by the
    formal ``no_physics`` ablation so that its output is an actual model
    variant, rather than a post-hoc perturbation of a bounded prediction.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden: int = 64,
        dropout: float = 0.15,
        physics_constrained: bool = True,
    ):
        super().__init__()
        self.physics_constrained = bool(physics_constrained)
        self.adapters = nn.ModuleList(
            [SourceAdapter(10, hidden), SourceAdapter(2, hidden), SourceAdapter(2, hidden)]
        )
        self.history_cell = ConvGRUCell(hidden, hidden)
        self.future_cell = ConvGRUCell(hidden, hidden)
        self.lead_embedding = nn.Embedding(6, 32)
        self.source_embedding = nn.Embedding(3, 16)
        self.condition = nn.Sequential(nn.Linear(32 + 8 + 16, hidden), nn.GELU())
        self.summary = SpatialSummary(hidden, d_model)
        self.token_fusion = nn.Sequential(
            nn.Linear(d_model + 32 + 8 + 16 + 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.clear_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def _adapt(self, values: torch.Tensor, valid: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        source_inputs = (
            (values[:, :5], valid[:, :5]),
            (values[:, 5:6], valid[:, 5:6]),
            (values[:, 6:7], valid[:, 6:7]),
        )
        candidates = torch.stack(
            [adapter(x, m) for adapter, (x, m) in zip(self.adapters, source_inputs)], dim=1
        )
        gather = source_id.view(-1, 1, 1, 1, 1).expand(-1, 1, candidates.shape[2], 8, 8)
        return candidates.gather(1, gather).squeeze(1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        satellite = batch["satellite"]
        valid = batch["satellite_valid"]
        frame_mask = batch["satellite_frame_mask"].bool()
        source_id = batch["satellite_source_id"]
        batch_size, history = satellite.shape[:2]
        hidden = satellite.new_zeros(batch_size, 64, 8, 8)
        previous = hidden
        latest_motion = hidden
        for step in range(history):
            frame = self._adapt(satellite[:, step], valid[:, step], source_id)
            candidate = self.history_cell(frame, hidden)
            active = frame_mask[:, step].view(batch_size, 1, 1, 1).to(frame)
            latest_motion = active * (candidate - hidden) + (1.0 - active) * latest_motion
            previous = hidden
            hidden = active * candidate + (1.0 - active) * hidden
        # The last recurrent increment is an explicit representation of recent
        # cloud motion; the hidden state retains cloud position and structure.
        motion = latest_motion + 0.25 * (hidden - previous)
        valid_fraction = frame_mask.float().mean(dim=1)
        inf = torch.full_like(batch["satellite_age_hours"], float("inf"))
        latest_age = torch.where(frame_mask, batch["satellite_age_hours"], inf).amin(dim=1)
        latest_age = torch.where(torch.isfinite(latest_age), latest_age, torch.full_like(latest_age, 24.0))
        available = frame_mask.any(dim=1)
        source = self.source_embedding(source_id)
        lead_ids = torch.arange(6, device=satellite.device)
        lead = self.lead_embedding(lead_ids).unsqueeze(0).expand(batch_size, -1, -1)
        tokens = []
        latent = hidden
        for step in range(6):
            context_vector = torch.cat([lead[:, step], batch["solar_context"][:, step], source], dim=-1)
            condition = self.condition(context_vector).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
            # Motion is strongest for the first leads and decays as persistence
            # becomes less credible.  Its magnitude is still learned by the cell.
            motion_scale = math.exp(-step / 3.0)
            latent = self.future_cell(condition + motion_scale * motion, latent)
            quality = torch.stack(
                [valid_fraction, (1.0 - latest_age / 24.0).clamp(0.0, 1.0), torch.full_like(valid_fraction, motion_scale)],
                dim=-1,
            )
            tokens.append(
                self.token_fusion(
                    torch.cat([self.summary(latent), lead[:, step], batch["solar_context"][:, step], source, quality], dim=-1)
                )
            )
        state = self.trajectory(torch.stack(tokens, dim=1))
        raw_output = self.output_head(state).squeeze(-1)
        if self.physics_constrained:
            q = torch.sigmoid(raw_output)
            prediction = q * batch["physical_upper"]
        else:
            # Direct, unbounded GHI output.  In particular, do not normalize
            # by physical_upper here and do not apply a night/non-negative/
            # clear-sky projection in this branch.
            q = raw_output
            prediction = raw_output
        dynamics = motion.square().mean(dim=(1, 2, 3)).sqrt().unsqueeze(1).expand(-1, 6)
        return {
            "state": state,
            "q": q,
            "prediction": prediction,
            "clear_logit": self.clear_head(state).squeeze(-1),
            "available": available.unsqueeze(1).expand(-1, 6),
            "valid_fraction": valid_fraction.unsqueeze(1).expand(-1, 6),
            "freshness": (1.0 - latest_age / 24.0).clamp(0.0, 1.0).unsqueeze(1).expand(-1, 6),
            "dynamics": dynamics,
        }


class PositionAwareVolumeEncoder(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(7, 48, 1),
            nn.GroupNorm(_groups(48), 48),
            nn.GELU(),
            nn.Conv3d(48, 96, 3, padding=1),
            nn.GroupNorm(_groups(96), 96),
            nn.GELU(),
            nn.Conv3d(96, 96, 3, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Sequential(nn.Linear(5 * 96, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid[:, None].to(values)
        feature = self.net(torch.cat([values, mask], dim=1))
        _, _, levels, height, width = feature.shape
        global_context = feature.mean(dim=(2, 3, 4))
        centre = feature[:, :, :, height // 2, width // 2].mean(dim=2)
        z = torch.linspace(-1.0, 1.0, levels, device=feature.device, dtype=feature.dtype)
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        z_m = (feature * z.view(1, 1, levels, 1, 1)).mean(dim=(2, 3, 4))
        y_m = (feature * y.view(1, 1, 1, height, 1)).mean(dim=(2, 3, 4))
        x_m = (feature * x.view(1, 1, 1, 1, width)).mean(dim=(2, 3, 4))
        return self.projection(torch.cat([global_context, centre, z_m, y_m, x_m], dim=-1))


class SurfaceEncoder(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.summary = SpatialSummary(64, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.summary(self.net(values))


class WRFExpert(nn.Module):
    """Future-state expert with optional bounded or unbounded output.

    The constrained branch retains the endpoint-recoverable normalized update
    used by the full model.  The no-physics branch instead predicts an
    unconstrained residual in GHI units and adds it to the raw WRF anchor.
    """

    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.15,
        physics_constrained: bool = True,
    ):
        super().__init__()
        self.physics_constrained = bool(physics_constrained)
        self.volume = PositionAwareVolumeEncoder(d_model)
        self.surface = SurfaceEncoder(d_model)
        self.lead_embedding = nn.Embedding(6, 32)
        self.token = nn.Sequential(
            nn.Linear(2 * d_model + 32 + 8 + 7 + 2, 2 * d_model),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.trajectory = nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)
        self.residual_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.clear_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        nwp = batch["nwp_3d"]
        batch_size, leads = nwp.shape[:2]
        volume = self.volume(
            nwp.reshape(batch_size * leads, 6, 10, 5, 5),
            batch["nwp_3d_valid"].reshape(batch_size * leads, 10, 5, 5),
        ).reshape(batch_size, leads, -1)
        surface = self.surface(batch["nwp_surface"].reshape(batch_size * leads, 4, 5, 5)).reshape(batch_size, leads, -1)
        lead = self.lead_embedding(torch.arange(6, device=nwp.device)).unsqueeze(0).expand(batch_size, -1, -1)
        static = batch["static"].unsqueeze(1).expand(-1, 6, -1)
        availability = batch["nwp_3d_valid"].flatten(2).any(dim=-1).to(nwp)
        if self.physics_constrained:
            anchor_feature = (
                batch["wrf_anchor"] / batch["physical_upper"].clamp_min(1.0)
            ).clamp(0.0, 1.0)
        else:
            # Keep the raw anchor as an input feature.  No normalized
            # [0, 1]-interval recovery is used by this variant.
            anchor_feature = batch["wrf_anchor"]
        state = self.token(torch.cat([volume, surface, lead, batch["solar_context"], static, anchor_feature.unsqueeze(-1), availability.unsqueeze(-1)], dim=-1))
        state = self.trajectory(state)
        raw_residual = self.residual_head(state).squeeze(-1)
        if self.physics_constrained:
            anchor_q = (
                batch["wrf_anchor"] / batch["physical_upper"].clamp_min(1.0)
            ).clamp(0.0, 1.0)
            delta = 2.0 * torch.tanh(raw_residual)
            q = bounded_update(anchor_q, delta)
            prediction = q * batch["physical_upper"]
        else:
            # Unbounded residual in physical GHI units around the raw WRF
            # anchor.  This can be negative or exceed the clear-sky upper
            # bound by design; the audit reports those violations.
            delta = raw_residual
            q = batch["wrf_anchor"] + delta
            prediction = q
        return {
            "state": state,
            "q": q,
            "prediction": prediction,
            "clear_logit": self.clear_head(state).squeeze(-1),
            "available": availability.bool(),
            "delta": delta,
        }


class MonotonicLeadPrior(nn.Module):
    def __init__(self, probabilities=(0.78, 0.64, 0.48, 0.34, 0.24, 0.18)):
        super().__init__()
        logits = [_logit(float(value)) for value in probabilities]
        decrements = [max(logits[index - 1] - logits[index], 1e-3) for index in range(1, 6)]
        self.first = nn.Parameter(torch.tensor(logits[0], dtype=torch.float32))
        self.decrement_raw = nn.Parameter(torch.tensor([_inverse_softplus(value) for value in decrements], dtype=torch.float32))

    def logits(self) -> torch.Tensor:
        decrements = F.softplus(self.decrement_raw)
        return torch.cat([self.first.view(1), self.first - torch.cumsum(decrements, dim=0)])

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logits())


ABLATIONS = (
    "full",
    "fixed_router",
    "no_transition",
    "no_aod",
    "no_conditional",
    "no_physics",
)


class ScenarioDrivenGHIModel(nn.Module):
    """Two primary experts, contextual routing, and conditional corrections."""

    def __init__(self, d_model: int = 128, dropout: float = 0.15, ablation: str = "full"):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation: {ablation}")
        self.ablation = ablation
        physics_constrained = ablation != "no_physics"
        self.satellite_expert = SatelliteExpert(
            d_model=d_model, dropout=dropout, physics_constrained=physics_constrained
        )
        self.wrf_expert = WRFExpert(
            d_model=d_model, dropout=dropout, physics_constrained=physics_constrained
        )
        self.source_embedding = nn.Embedding(3, 16)
        self.lead_embedding = nn.Embedding(6, 16)
        route_dim = 2 * d_model + 3 + 2 + 4 + 8 + 7 + 16 + 16
        self.route_evidence = nn.Sequential(
            nn.Linear(route_dim, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        # Epoch zero must be the explicit monotonic lead-time prior.  A neutral
        # last layer makes every scenario-dependent deviation attributable to
        # fusion-stage learning rather than to a random initialization draw.
        nn.init.zeros_(self.route_evidence[-1].weight)
        nn.init.zeros_(self.route_evidence[-1].bias)
        self.lead_prior = MonotonicLeadPrior()
        correction_dim = 2 * d_model + 3 + 8 + 7
        self.transition_gate = nn.Sequential(nn.Linear(correction_dim, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.transition_head = nn.Sequential(nn.Linear(correction_dim + 1, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.aod_gate = nn.Sequential(nn.Linear(correction_dim + 3, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.aod_strength = nn.Sequential(nn.Linear(correction_dim + 3, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        for module in (self.transition_gate, self.transition_head, self.aod_gate, self.aod_strength):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        # The optional aerosol path must be effectively neutral before it has
        # learned evidence.  A zero logit would open a sigmoid gate by 50%.
        # The negative bias keeps the inactive path numerically neutral.  The
        # training driver opens it to a trainable value only inside the AOD
        # stage and restores this neutral checkpoint if no safe gain is found.
        nn.init.constant_(self.aod_gate[-1].bias, -12.0)

    @staticmethod
    def project(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bounded = torch.minimum(prediction.clamp_min(0.0), batch["physical_upper"])
        return torch.where(batch["solar_context"][..., 2] > 0.0, bounded, torch.zeros_like(bounded))

    def _output(self, prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply the full-model projection, or return no-physics output raw."""

        if self.ablation == "no_physics":
            return prediction
        return self.project(prediction, batch)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        satellite = self.satellite_expert(batch)
        wrf = self.wrf_expert(batch)
        batch_size = batch["wrf_anchor"].shape[0]
        source = self.source_embedding(batch["satellite_source_id"]).unsqueeze(1).expand(-1, 6, -1)
        lead = self.lead_embedding(torch.arange(6, device=batch["wrf_anchor"].device)).unsqueeze(0).expand(batch_size, -1, -1)
        quality = torch.stack(
            [satellite["valid_fraction"], satellite["freshness"], satellite["dynamics"], wrf["available"].to(batch["wrf_anchor"])], dim=-1
        )
        if self.ablation == "no_physics":
            satellite_route = satellite["prediction"]
            wrf_route = wrf["prediction"]
        else:
            satellite_route = satellite["q"]
            wrf_route = wrf["q"]
        prediction_state = torch.stack(
            [satellite_route, wrf_route, (satellite_route - wrf_route).abs()], dim=-1
        )
        clear_state = torch.stack(
            [torch.sigmoid(satellite["clear_logit"]), torch.sigmoid(wrf["clear_logit"])], dim=-1
        )
        static = batch["static"].unsqueeze(1).expand(-1, 6, -1)
        route_input = torch.cat(
            [satellite["state"], wrf["state"], prediction_state, clear_state, quality, batch["solar_context"], static, source, lead], dim=-1
        )
        if self.ablation == "fixed_router":
            evidence = torch.zeros_like(prior_logits := self.lead_prior.logits().view(1, 6).expand(batch_size, -1))
        else:
            evidence = 2.0 * torch.tanh(self.route_evidence(route_input).squeeze(-1))
            prior_logits = self.lead_prior.logits().view(1, 6).expand(batch_size, -1)
        satellite_logit = prior_logits + evidence
        logits = torch.stack([satellite_logit, torch.zeros_like(satellite_logit)], dim=-1)
        availability = torch.stack([satellite["available"], wrf["available"]], dim=-1)
        logits = logits.masked_fill(~availability, -1e4)
        weights = torch.softmax(logits, dim=-1) * availability.to(logits)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        base_q = weights[..., 0] * satellite_route + weights[..., 1] * wrf_route

        correction_context = torch.cat(
            [satellite["state"], wrf["state"], prediction_state, batch["solar_context"], static], dim=-1
        )
        if self.ablation == "no_physics":
            transition_support = batch["transition_support"].clamp(0.0, 1.0)
            transition_gate = transition_support * torch.sigmoid(
                self.transition_gate(correction_context).squeeze(-1)
            )
            # The unbounded branch uses an additive correction in GHI units,
            # not a normalized logarithmic update followed by bounded_update.
            transition_delta = self.transition_head(
                torch.cat([correction_context, base_q.unsqueeze(-1)], dim=-1)
            ).squeeze(-1)
            transition_q = base_q + transition_gate * transition_delta
        elif self.ablation in {"no_transition", "no_conditional"}:
            transition_gate = torch.zeros_like(base_q)
            transition_delta = torch.zeros_like(base_q)
            transition_q = base_q
        else:
            transition_support = batch["transition_support"].clamp(0.0, 1.0)
            transition_gate = transition_support * torch.sigmoid(self.transition_gate(correction_context).squeeze(-1))
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
        aerosol_inputs = torch.cat(
            [correction_context, common_clear.unsqueeze(-1), aod_anomaly.unsqueeze(-1), batch["air_mass"].unsqueeze(-1)], dim=-1
        )
        if self.ablation in {"no_aod", "no_conditional", "no_physics"}:
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
            "satellite_prediction": self._output(satellite["prediction"], batch),
            "wrf_prediction": self._output(wrf["prediction"], batch),
            "base_prediction": self._output(
                base_q if self.ablation == "no_physics" else base_q * batch["physical_upper"], batch
            ),
            "transition_prediction": self._output(
                transition_q if self.ablation == "no_physics" else transition_q * batch["physical_upper"], batch
            ),
            "final_prediction": self._output(
                final_q if self.ablation == "no_physics" else final_q * batch["physical_upper"], batch
            ),
            "fusion_weights": weights,
            "lead_prior": self.lead_prior().view(1, 6).expand(batch_size, -1),
            "route_evidence": evidence,
            "satellite_valid_fraction": satellite["valid_fraction"],
            "satellite_freshness": satellite["freshness"],
            "satellite_dynamics": satellite["dynamics"],
            "wrf_available": wrf["available"],
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
