"""Training objectives and staged optimization controls for GHI v4."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import ScenarioDrivenGHIModelV4


STAGES = ("satellite", "nwp", "fusion", "transition", "aod", "joint")
SATELLITE_LEAD_WEIGHT = torch.tensor([1.50, 1.35, 1.15, 0.85, 0.65, 0.50])
NWP_LEAD_WEIGHT = torch.tensor([0.65, 0.80, 1.00, 1.15, 1.30, 1.40])


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool)
    effective = valid.to(values)
    if weights is not None:
        effective = effective * weights.to(values)
    active = valid & (effective != 0)
    if active.any() and (not torch.isfinite(values[active]).all() or not torch.isfinite(effective[active]).all()):
        raise FloatingPointError("non-finite value or weight at an active masked position")
    safe_values = torch.where(active, values, torch.zeros_like(values))
    safe_effective = torch.where(active, effective, torch.zeros_like(effective))
    return (safe_values * safe_effective).sum() / safe_effective.sum().clamp_min(1.0)


def case_balanced_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Give each station case equal mass even when daylight/label counts differ."""

    valid = mask.to(dtype=torch.bool)
    effective = valid.to(values)
    if weights is not None:
        effective = effective * weights.to(values)
    active = valid & (effective != 0)
    if active.any() and (not torch.isfinite(values[active]).all() or not torch.isfinite(effective[active]).all()):
        raise FloatingPointError("non-finite value or weight at an active case position")
    safe_values = torch.where(active, values, torch.zeros_like(values))
    safe_effective = torch.where(active, effective, torch.zeros_like(effective))
    numerator = (safe_values * safe_effective).sum(dim=1)
    denominator = safe_effective.sum(dim=1)
    valid_case = denominator > 0
    return (numerator / denominator.clamp_min(1.0))[valid_case].mean() if valid_case.any() else values.new_zeros(())


def trajectory_loss(
    prediction: torch.Tensor,
    batch: dict,
    lead_weight: torch.Tensor | None = None,
    scene_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    point = F.smooth_l1_loss(prediction, batch["target"], beta=50.0, reduction="none") / 100.0
    weight = torch.ones_like(point)
    if lead_weight is not None:
        weight = weight * lead_weight.to(point.device).view(1, 6)
    if scene_weight is not None:
        weight = weight * scene_weight
    absolute = case_balanced_mean(point, batch["target_mask"], weight)
    pred_ramp = prediction[:, 1:] - prediction[:, :-1]
    target_ramp = batch["target"][:, 1:] - batch["target"][:, :-1]
    ramp_mask = batch["target_mask"][:, 1:] & batch["target_mask"][:, :-1]
    ramp = F.smooth_l1_loss(pred_ramp, target_ramp, beta=30.0, reduction="none") / 100.0
    cloud_change = (target_ramp.abs() / 150.0).clamp(0.0, 2.0)
    return absolute + 0.15 * case_balanced_mean(ramp, ramp_mask, 1.0 + cloud_change)


def uncertainty_loss(prediction: torch.Tensor, scale: torch.Tensor, batch: dict) -> torch.Tensor:
    """Laplace proper score for an interpretable expected absolute-error scale."""

    safe_scale = scale.clamp(10.0, 600.0)
    score = (prediction - batch["target"]).abs() / safe_scale + torch.log(2.0 * safe_scale)
    return case_balanced_mean(score, batch["target_mask"])


def clear_loss(output: dict, batch: dict, expert: str = "both") -> torch.Tensor:
    mask = batch["clear_target_mask"]
    sat = F.binary_cross_entropy_with_logits(output["satellite_clear_logit"], batch["clear_target"], reduction="none")
    nwp = F.binary_cross_entropy_with_logits(output["wrf_clear_logit"], batch["clear_target"], reduction="none")
    if expert == "satellite":
        return case_balanced_mean(sat, mask)
    if expert == "nwp":
        return case_balanced_mean(nwp, mask)
    return case_balanced_mean(sat, mask) + case_balanced_mean(nwp, mask)


def routing_supervision(output: dict, batch: dict) -> torch.Tensor:
    mode = output.get("routing_supervision_mode", "winner")
    if mode == "none":
        return output["fusion_weights"].new_zeros(())
    if mode == "convex":
        satellite = output["satellite_prediction"].detach()
        wrf = output["wrf_prediction"].detach()
        gap = satellite - wrf
        valid = batch["target_mask"] & (gap.abs() >= 20.0)
        target = torch.where(valid, batch["target"], wrf)
        denominator = torch.where(valid, gap, torch.ones_like(gap))
        optimal = ((target - wrf) / denominator).clamp(0.0, 1.0).detach()
        # Near-identical experts have poorly identifiable mixture targets.
        weight = (gap.abs() / 100.0).clamp(0.0, 1.0).detach()
        loss = (output["fusion_weights"][..., 0] - optimal).square()
        return case_balanced_mean(loss, valid, weight)
    if mode != "winner":
        raise ValueError(mode)
    sat_error = (output["satellite_prediction"].detach() - batch["target"]).abs()
    nwp_error = (output["wrf_prediction"].detach() - batch["target"]).abs()
    oracle_satellite = torch.sigmoid((nwp_error - sat_error) / 40.0)
    log_weights = output["fusion_weights"].clamp_min(1e-6).log()
    cross_entropy = -(oracle_satellite * log_weights[..., 0] + (1.0 - oracle_satellite) * log_weights[..., 1])
    confidence_target = (1.0 - torch.exp(-(sat_error - nwp_error).abs() / 60.0)).detach()
    confidence = F.binary_cross_entropy(output["route_confidence"].clamp(1e-5, 1.0 - 1e-5), confidence_target, reduction="none")
    return case_balanced_mean(cross_entropy + 0.25 * confidence, batch["target_mask"])


def negative_transfer_penalty(output: dict, batch: dict) -> torch.Tensor:
    fused_error = (output["base_prediction"] - batch["target"]).abs()
    best_error = torch.minimum(
        (output["satellite_prediction"].detach() - batch["target"]).abs(),
        (output["wrf_prediction"].detach() - batch["target"]).abs(),
    )
    excess = F.relu((fused_error - best_error - 20.0) / 100.0)
    return case_balanced_mean(excess, batch["target_mask"])


def conditional_route_regularization(output: dict) -> torch.Tensor:
    """Smooth routing only when the forecast scene itself is stable."""

    weight_change = output["fusion_weights"][:, 1:, 0] - output["fusion_weights"][:, :-1, 0]
    prediction_change = (
        output["satellite_prediction"][:, 1:] - output["satellite_prediction"][:, :-1]
    ).abs() + (
        output["wrf_prediction"][:, 1:] - output["wrf_prediction"][:, :-1]
    ).abs()
    transport_change = output["wrf_transport_residual"][:, 1:] + output["wrf_transport_residual"][:, :-1]
    stable = torch.exp(-prediction_change.detach() / 150.0) * torch.exp(-transport_change.detach() / 0.2)
    return 0.05 * output["route_evidence"].square().mean() + (stable * weight_change.square()).mean()


def grouped_bias_penalty(prediction: torch.Tensor, batch: dict) -> torch.Tensor:
    error = prediction - batch["target"]
    daylight = batch["solar_context"][..., 2] > 0.0
    valid = batch["target_mask"] & daylight
    terms = []
    for lead in range(6):
        if valid[:, lead].any():
            terms.append((error[:, lead][valid[:, lead]].mean() / 100.0).square())
    for source in range(3):
        source_rows = batch["satellite_source_id"] == source
        source_mask = valid & source_rows.unsqueeze(1)
        if source_mask.any():
            terms.append((error[source_mask].mean() / 100.0).square())
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def future_cloud_loss(output: dict, batch: dict) -> torch.Tensor:
    if "satellite_future_cloud" not in batch:
        return output["final_prediction"].new_zeros(())
    target = batch["satellite_future_cloud"].to(output["satellite_future_cloud_logits"])
    mask = batch.get("satellite_future_cloud_mask", torch.isfinite(target)).to(dtype=torch.bool)
    if target.ndim == 4:
        target = target.unsqueeze(2)
        mask = mask.unsqueeze(2)
    if target.shape[2] > 1:
        target = target[:, :, :1]
        mask = mask[:, :, :1]
    batch_size, leads = target.shape[:2]
    target = F.interpolate(target.reshape(batch_size * leads, 1, *target.shape[-2:]), size=(8, 8), mode="bilinear", align_corners=False).reshape(batch_size, leads, 1, 8, 8)
    mask = F.interpolate(mask.to(target).reshape(batch_size * leads, 1, *mask.shape[-2:]), size=(8, 8), mode="nearest").bool().reshape(batch_size, leads, 1, 8, 8)
    loss = F.binary_cross_entropy_with_logits(output["satellite_future_cloud_logits"], target.clamp(0.0, 1.0), reduction="none")
    return masked_mean(loss, mask)


def anchor_trust_loss(output: dict, batch: dict) -> torch.Tensor:
    safe_target = torch.where(batch["target_mask"], batch["target"], batch["wrf_anchor"])
    raw_error = (batch["wrf_anchor"] - safe_target).abs()
    target = torch.exp(-raw_error / 150.0).detach()
    prediction = output["wrf_anchor_trust"]
    if not torch.isfinite(prediction).all():
        bad_rows = ~torch.isfinite(prediction).all(dim=1)
        sample_ids = batch.get("sample_id", [])
        affected = [sample_ids[index] for index in bad_rows.nonzero(as_tuple=False).flatten().tolist()] if sample_ids else []
        raise FloatingPointError(f"non-finite wrf_anchor_trust for samples: {affected}")
    loss = F.binary_cross_entropy(prediction.clamp(1e-5, 1.0 - 1e-5), target, reduction="none")
    return case_balanced_mean(loss, batch["target_mask"])


def correction_regularization(output: dict) -> torch.Tensor:
    return (
        (output["transition_gate"] * output["transition_delta"]).square().mean()
        + (output["aod_gate"] * output["aod_delta_log_t"]).square().mean()
    )


def loss_for_stage(output: dict, batch: dict, stage: str) -> tuple[torch.Tensor, dict[str, float]]:
    if stage not in STAGES:
        raise ValueError(stage)

    zero = output["final_prediction"].new_zeros(())
    satellite = zero
    nwp = zero
    clear = zero
    satellite_uncertainty = zero
    nwp_uncertainty = zero
    base_uncertainty = zero
    cloud = zero
    route = zero
    safety = zero
    trust = zero
    correction = zero
    ridge = zero

    if stage in {"satellite", "joint"}:
        satellite = trajectory_loss(output["satellite_prediction"], batch, SATELLITE_LEAD_WEIGHT)
        satellite_uncertainty = uncertainty_loss(output["satellite_prediction"], output["satellite_scale_wm2"], batch)
        cloud = future_cloud_loss(output, batch)
    if stage in {"nwp", "joint"}:
        nwp = trajectory_loss(output["wrf_prediction"], batch, NWP_LEAD_WEIGHT)
        nwp_uncertainty = uncertainty_loss(output["wrf_prediction"], output["wrf_scale_wm2"], batch)
        trust = anchor_trust_loss(output, batch)
        if bool(batch["ridge_teacher_mask"].any()):
            distill = F.smooth_l1_loss(output["wrf_prediction"], batch["ridge_teacher"], beta=50.0, reduction="none") / 100.0
            ridge = case_balanced_mean(distill, batch["ridge_teacher_mask"])
    if stage == "satellite":
        clear = clear_loss(output, batch, "satellite")
    elif stage == "nwp":
        clear = clear_loss(output, batch, "nwp")
    elif stage == "joint":
        clear = clear_loss(output, batch, "both")
    if stage in {"fusion", "joint"}:
        base_uncertainty = uncertainty_loss(output["base_prediction"], output["base_scale_wm2"], batch)
        route = conditional_route_regularization(output)
        safety = negative_transfer_penalty(output, batch)
    if stage in {"transition", "aod", "joint"}:
        correction = correction_regularization(output)
    if stage == "satellite":
        loss = satellite + 0.05 * clear + 0.03 * satellite_uncertainty + 0.10 * cloud
    elif stage == "nwp":
        loss = nwp + 0.05 * clear + 0.03 * nwp_uncertainty + 0.10 * trust + 0.15 * ridge
    elif stage == "fusion":
        loss = trajectory_loss(output["base_prediction"], batch) + 0.20 * routing_supervision(output, batch) + 0.05 * safety + 0.03 * base_uncertainty + 0.01 * route
    elif stage == "transition":
        loss = trajectory_loss(output["transition_prediction"], batch, scene_weight=1.0 + 3.0 * batch["transition_support"]) + 0.01 * correction
    elif stage == "aod":
        aod_valid = batch["aod_mask"] & torch.isfinite(batch["aod550"]) & torch.isfinite(batch["aod_reference"])
        anomaly = torch.where(aod_valid, batch["aod550"] - batch["aod_reference"], 0.0)
        scene = batch["clear_target"] * batch["clear_target_mask"].to(anomaly) * aod_valid.to(anomaly) * (anomaly / 0.20).clamp(0.0, 2.0)
        loss = trajectory_loss(output["final_prediction"], batch, scene_weight=1.0 + 2.0 * scene) + 0.01 * correction
    elif stage == "joint":
        base = trajectory_loss(output["base_prediction"], batch)
        final = trajectory_loss(output["final_prediction"], batch)
        loss = (
            final + 0.25 * base + 0.15 * satellite + 0.15 * nwp + 0.05 * clear
            + 0.08 * routing_supervision(output, batch) + 0.05 * safety
            + 0.02 * (satellite_uncertainty + nwp_uncertainty + base_uncertainty)
            + 0.05 * grouped_bias_penalty(output["final_prediction"], batch)
            + 0.05 * cloud + 0.01 * route + 0.01 * correction
        )
    history_alignment = output.get("satellite_history_alignment_loss", zero)
    direct_supervision = zero
    if stage in {"satellite", "joint"}:
        if output.get("satellite_prediction_path", "dual") == "dual":
            direct_supervision = trajectory_loss(output["satellite_direct_prediction"], batch, SATELLITE_LEAD_WEIGHT)
            dynamic_supervision = trajectory_loss(output["satellite_dynamic_prediction"], batch, SATELLITE_LEAD_WEIGHT)
            loss = loss + 0.15 * (direct_supervision + dynamic_supervision)
        loss = loss + 0.05 * history_alignment
    parts = {
        "direct_supervision": float(direct_supervision.detach()),
        "history_alignment": float(history_alignment.detach()),
        "total": float(loss.detach()), "satellite": float(satellite.detach()), "nwp": float(nwp.detach()),
        "clear": float(clear.detach()), "satellite_uncertainty": float(satellite_uncertainty.detach()),
        "nwp_uncertainty": float(nwp_uncertainty.detach()), "base_uncertainty": float(base_uncertainty.detach()),
        "cloud": float(cloud.detach()), "route_reg": float(route.detach()), "safety": float(safety.detach()),
        "anchor_trust": float(trust.detach()), "ridge_distill": float(ridge.detach()),
        "correction_reg": float(correction.detach()),
    }
    return loss, parts


def modules_for_stage(model: ScenarioDrivenGHIModelV4, stage: str) -> list[nn.Module]:
    return {
        "satellite": [model.satellite_expert],
        "nwp": [model.wrf_expert],
        "fusion": [model.route_evidence, model.route_confidence, model.lead_prior, model.source_embedding, model.lead_embedding] + ([model.route_quality_projection] if model.route_quality_projection is not None else []),
        "transition": [model.transition_gate, model.transition_head],
        "aod": [model.aod_gate, model.aod_strength],
        "joint": [model],
    }[stage]


def configure_stage(model: ScenarioDrivenGHIModelV4, stage: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in modules_for_stage(model, stage):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if stage in {"fusion", "joint"}:
        model.uncertainty_strength_raw.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def set_stage_training_mode(model: ScenarioDrivenGHIModelV4, stage: str) -> None:
    model.eval()
    for module in modules_for_stage(model, stage):
        module.train()
