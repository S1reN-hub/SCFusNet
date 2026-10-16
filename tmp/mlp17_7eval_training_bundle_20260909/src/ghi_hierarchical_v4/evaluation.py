"""Standalone training/evaluation utilities for the v4 model."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def predictions_for_stage(output: dict, stage: str) -> torch.Tensor:
    return {
        "satellite": output["satellite_prediction"],
        "nwp": output["wrf_prediction"],
        "fusion": output["base_prediction"],
        "transition": output["transition_prediction"],
        "aod": output["final_prediction"],
        "joint": output["final_prediction"],
    }[stage]


def metric_summary(y: np.ndarray, prediction: np.ndarray, day: np.ndarray, stations: np.ndarray) -> dict:
    error = prediction - y
    station_rmse = []
    for station in sorted(set(stations.tolist())):
        selected = day & (stations == station)
        if selected.any():
            station_rmse.append(float(np.sqrt(np.mean(error[selected] ** 2))))
    daylight_rmse = float(np.sqrt(np.mean(error[day] ** 2))) if day.any() else None
    macro_rmse = float(np.mean(station_rmse)) if station_rmse else None
    return {
        "rmse_wm2": float(np.sqrt(np.mean(error ** 2))),
        "mae_wm2": float(np.mean(np.abs(error))),
        "mbe_wm2": float(np.mean(error)),
        "daylight_rmse_wm2": daylight_rmse,
        "macro_station_daylight_rmse_wm2": macro_rmse,
        "count": int(y.size),
        "daylight_count": int(day.sum()),
    }


def selection_score(metrics: dict, stage: str) -> float:
    overall = float(metrics["macro_station_daylight_rmse_wm2"])
    if stage == "satellite":
        role = np.mean([
            metrics["by_lead"][str(lead)]["macro_station_daylight_rmse_wm2"] for lead in (1, 2, 3)
        ])
        return float(0.70 * role + 0.30 * overall)
    if stage == "nwp":
        role = np.mean([
            metrics["by_lead"][str(lead)]["macro_station_daylight_rmse_wm2"] for lead in (3, 4, 5, 6)
        ])
        return float(0.70 * role + 0.30 * overall)
    if stage == "transition" and metrics.get("transition_rmse_wm2") is not None:
        return float(0.70 * metrics["transition_rmse_wm2"] + 0.30 * overall)
    if stage == "aod" and metrics.get("high_aod_clear_rmse_wm2") is not None:
        return float(0.70 * metrics["high_aod_clear_rmse_wm2"] + 0.30 * overall)
    return overall


def checkpoint_is_better(candidate: dict, incumbent: dict, stage: str, incoming_overall: float) -> bool:
    if stage in {"transition", "aod"}:
        globally_safe = candidate["macro_station_daylight_rmse_wm2"] <= incoming_overall + 0.10
        return globally_safe and selection_score(candidate, stage) < selection_score(incumbent, stage)
    return selection_score(candidate, stage) < selection_score(incumbent, stage)


def evaluate(model, loader: DataLoader, stage: str, device: torch.device) -> dict:
    model.eval()
    observed, predicted, daylight, stations, leads, transition, high_aod = [], [], [], [], [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = to_device(raw, device)
            output = predictions_for_stage(model(batch), stage)
            valid = batch["target_mask"]
            observed.append(batch["target"][valid].cpu().numpy())
            predicted.append(output[valid].cpu().numpy())
            daylight.append((batch["solar_context"][..., 2] > 0.0)[valid].cpu().numpy())
            leads.append(torch.arange(1, 7, device=device).view(1, 6).expand_as(valid)[valid].cpu().numpy())
            transition.append((batch["transition_support"] >= 0.25)[valid].cpu().numpy())
            aod_scene = (
                batch["aod_mask"] & batch["clear_target_mask"] & (batch["clear_target"] >= 0.7)
                & (batch["solar_context"][..., 2] > 0.99)
                & ((batch["aod550"] - batch["aod_reference"]) >= 0.10)
            )
            high_aod.append(aod_scene[valid].cpu().numpy())
            for index, name in enumerate(raw["station_id"]):
                stations.extend([name] * int(raw["target_mask"][index].sum()))
    if not observed or sum(value.size for value in observed) == 0:
        raise ValueError("evaluation partition has no valid GHI labels")
    y, prediction = np.concatenate(observed), np.concatenate(predicted)
    day = np.concatenate(daylight).astype(bool)
    station_array = np.asarray(stations)
    result = metric_summary(y, prediction, day, station_array)
    lead_array = np.concatenate(leads)
    result["by_lead"] = {
        str(lead): metric_summary(
            y[lead_array == lead], prediction[lead_array == lead], day[lead_array == lead],
            station_array[lead_array == lead],
        )
        for lead in range(1, 7)
    }
    for name, mask in (
        ("transition", np.concatenate(transition).astype(bool)),
        ("high_aod_clear", np.concatenate(high_aod).astype(bool)),
    ):
        if mask.any():
            error = prediction[mask] - y[mask]
            result[f"{name}_rmse_wm2"] = float(np.sqrt(np.mean(error ** 2)))
            result[f"{name}_count"] = int(mask.sum())
        else:
            result[f"{name}_rmse_wm2"] = None
            result[f"{name}_count"] = 0
    return result


def atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
