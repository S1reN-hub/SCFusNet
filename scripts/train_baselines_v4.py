#!/usr/bin/env python3
"""Train current-protocol Sat-ConvLSTM, WRF-Transformer and EarlyFusion baselines.

The runner consumes only public tensors from ``MultisourceCaseDataset``.  It
does not import SolFUS-Net, frozen expert predictions, hidden features, AOD,
or test labels during model selection.  ``--smoke`` is a local five-case CPU
forward/backward check and is not a performance run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ghi_hierarchical_v4.data import (  # noqa: E402
    MultisourceCaseDataset,
    compute_normalization,
    cycle_key,
    read_index,
)

MODEL_CHOICES = ("sat_convlstm", "wrf_transformer", "early_fusion")
DAYLIGHT_EPS = 1e-6


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def numpy_safe_collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate preloaded tensors without duplicating host memory in workers."""

    result: dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        first = values[0]
        if isinstance(first, torch.Tensor):
            result[key] = torch.from_numpy(
                np.stack([value.detach().cpu().numpy() for value in values])
            )
        elif isinstance(first, np.ndarray):
            result[key] = torch.from_numpy(np.stack(values))
        elif isinstance(first, (bool, int, float, np.generic)):
            result[key] = torch.as_tensor(values)
        elif isinstance(first, (str, bytes)):
            result[key] = values
        else:
            raise TypeError(f"unsupported batch value for {key}: {type(first).__name__}")
    return result


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def current_protocol_rows(
    rows: list[dict[str, str]], split_config: dict[str, Any]
) -> dict[str, list[dict[str, str]]]:
    """Reproduce the sanctioned station-aware 17/7 split from metadata only."""

    train_cycles = {str(value) for value in split_config["split_cycles"]["train"]}
    validation_cycles = {
        str(value) for value in split_config["split_cycles"]["validation"]
    }
    test_cycles = {str(value) for value in split_config["split_cycles"]["test"]}
    all_cycles = train_cycles | validation_cycles | test_cycles
    evaluation_stations = {
        str(value).upper() for value in split_config["evaluation_stations"]
    }
    training_stations = {
        str(value).upper() for value in split_config["training_stations"]
    }
    result = {"train": [], "validation": [], "test": []}
    for row in rows:
        station = str(row.get("station_id", "")).strip().upper()
        cycle = cycle_key(row)
        if station not in training_stations or cycle not in all_cycles:
            continue
        if station not in evaluation_stations:
            result["train"].append(row)
        elif cycle in train_cycles:
            result["train"].append(row)
        elif cycle in validation_cycles:
            result["validation"].append(row)
        elif cycle in test_cycles:
            result["test"].append(row)
    sample_ids = [
        row.get("sample_id", "") for values in result.values() for row in values
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample overlap detected in the current-protocol split")
    expected = split_config.get("active_case_counts")
    actual = [len(result[name]) for name in ("train", "validation", "test")]
    if expected is not None and actual != [int(value) for value in expected]:
        raise ValueError(
            f"current-protocol case counts differ: expected={expected}, actual={actual}"
        )
    return result


def smoke_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    if len(rows) < 3:
        raise ValueError("--smoke requires at least three case rows")
    ordered = sorted(rows, key=lambda row: str(row.get("sample_id", "")))
    return {"train": ordered[:2], "validation": ordered[2:3], "test": []}


def safe_future_states(values: torch.Tensor) -> torch.Tensor:
    """Return six forecast states and tolerate six-state smoke files."""

    if values.ndim < 2:
        raise ValueError(f"state tensor lacks a time dimension: {tuple(values.shape)}")
    if values.shape[1] >= 7:
        return values[:, 1:7]
    if values.shape[1] == 6:
        return values
    raise ValueError(f"expected six or seven states, got {tuple(values.shape)}")


def safe_future_valid(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] >= 7:
        return values[:, 1:7]
    if values.shape[1] == 6:
        return values
    raise ValueError(f"expected six or seven validity states, got {tuple(values.shape)}")


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1
        )

    def forward(
        self, values: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        ingate, forgetgate, outgate, candidate = self.gates(
            torch.cat([values, hidden], dim=1)
        ).chunk(4, dim=1)
        cell = torch.sigmoid(forgetgate) * cell + torch.sigmoid(ingate) * torch.tanh(candidate)
        hidden = torch.sigmoid(outgate) * torch.tanh(cell)
        return hidden, cell


class SatelliteHistoryEncoder(nn.Module):
    """Past-only six-frame satellite ConvLSTM encoder."""

    def __init__(self, d_model: int, hidden: int = 32) -> None:
        super().__init__()
        self.frame = nn.Sequential(
            nn.Conv2d(7, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.recurrent = ConvLSTMCell(hidden, hidden)
        self.projection = nn.Sequential(
            nn.Linear(hidden, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        satellite = batch["satellite"]
        valid = batch["satellite_valid"].bool()
        frame_mask = batch["satellite_frame_mask"].bool()
        if satellite.ndim != 5 or satellite.shape[2] != 7:
            raise ValueError(f"expected satellite [B,6,7,H,W], got {tuple(satellite.shape)}")
        batch_size, history, _, height, width = satellite.shape
        hidden = satellite.new_zeros(batch_size, self.recurrent.hidden_channels, height, width)
        cell = torch.zeros_like(hidden)
        for step in range(history):
            current = satellite[:, step] * valid[:, step].to(satellite)
            encoded = self.frame(current)
            next_hidden, next_cell = self.recurrent(encoded, (hidden, cell))
            active = frame_mask[:, step].view(batch_size, 1, 1, 1).to(satellite)
            hidden = active * next_hidden + (1.0 - active) * hidden
            cell = active * next_cell + (1.0 - active) * cell
        return self.projection(hidden.mean(dim=(2, 3)))


class WRFSourceEncoder(nn.Module):
    """Encode future WRF d01/d02 fields as two simple spatial scales."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.volume = nn.Sequential(
            nn.Conv3d(12, 32, 3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.surface = nn.Sequential(
            nn.Conv2d(15, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.token = nn.Sequential(
            nn.Linear(4 * 64 + 8, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def _encode_domain(
        self, batch: dict[str, torch.Tensor], domain: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        volume = safe_future_states(batch[f"nwp_{domain}_3d"])
        level_valid = safe_future_valid(batch[f"nwp_{domain}_3d_valid"]).bool()
        surface = safe_future_states(batch[f"nwp_{domain}_surface"])
        volume = volume * level_valid.unsqueeze(2).to(volume)
        batch_size, leads, channels, levels, height, width = volume.shape
        volume_feature = self.volume(
            volume.reshape(batch_size * leads, channels, levels, height, width)
        ).flatten(1)
        if surface.shape[-2:] != (height, width):
            surface = F.interpolate(
                surface.reshape(batch_size * leads, surface.shape[2], *surface.shape[-2:]),
                size=(height, width), mode="bilinear", align_corners=False,
            ).reshape(batch_size, leads, surface.shape[2], height, width)
        surface_feature = self.surface(
            surface.reshape(batch_size * leads, surface.shape[2], height, width)
        ).flatten(1)
        return (
            volume_feature.reshape(batch_size, leads, -1),
            surface_feature.reshape(batch_size, leads, -1),
        )

    def encode_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        d01_volume, d01_surface = self._encode_domain(batch, "d01")
        d02_volume, d02_surface = self._encode_domain(batch, "d02")
        solar = batch["solar_context"]
        return self.token(
            torch.cat(
                [
                    d01_volume,
                    d01_surface,
                    d02_volume,
                    d02_surface,
                    solar,
                ],
                dim=-1,
            )
        )


class TemporalBlock(nn.Module):
    def __init__(self, d_model: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(values))


class GHIHead(nn.Module):
    """Predict a non-negative clear-sky-scaled GHI amplitude.

    The head predicts a dimensionless clear-sky index and converts it back to
    radiance units using the causal solar context.  This keeps the initial
    output in the physical GHI range while leaving the learned index
    unconstrained above zero; the daylight gate still suppresses nighttime
    predictions.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.10), nn.Linear(d_model, 1)
        )
        # softplus(0.1) ~= 0.74, giving a useful initial clear-sky index
        # instead of starting from an approximately 1 W m^-2 radiance.
        nn.init.constant_(self.net[-1].bias, 0.1)

    def forward(self, features: torch.Tensor, solar: torch.Tensor) -> torch.Tensor:
        raw = self.net(features).squeeze(-1)
        daylight_factor = solar[..., 2].clamp(0.0, 1.0)
        clear_scale = solar[..., 1] * 1000.0 + 20.0 * solar[..., 2]
        return F.softplus(raw) * clear_scale * daylight_factor


class SatConvLSTMModel(nn.Module):
    model_name = "Sat-ConvLSTM"

    def __init__(self, d_model: int = 64, layers: int = 1, dropout: float = 0.10) -> None:
        super().__init__()
        self.history = SatelliteHistoryEncoder(d_model)
        self.input = nn.Sequential(
            nn.Linear(d_model + 8, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.temporal = TemporalBlock(d_model, layers, 4, dropout)
        self.head = GHIHead(d_model)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        history = self.history(batch).unsqueeze(1).expand(-1, 6, -1)
        features = self.input(torch.cat([history, batch["solar_context"]], dim=-1))
        return {"prediction": self.head(self.temporal(features), batch["solar_context"])}


class WRFTransformerModel(nn.Module):
    model_name = "WRF-Transformer"

    def __init__(self, d_model: int = 64, layers: int = 2, dropout: float = 0.10) -> None:
        super().__init__()
        self.source = WRFSourceEncoder(d_model)
        self.temporal = TemporalBlock(d_model, layers, 4, dropout)
        self.head = GHIHead(d_model)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.temporal(self.source.encode_tokens(batch))
        return {"prediction": self.head(features, batch["solar_context"])}


class EarlyFusionTransformerModel(nn.Module):
    model_name = "EarlyFusion-Transformer"

    def __init__(self, d_model: int = 64, layers: int = 2, dropout: float = 0.10) -> None:
        super().__init__()
        self.satellite = SatelliteHistoryEncoder(d_model)
        self.wrf = WRFSourceEncoder(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model + 8, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.temporal = TemporalBlock(d_model, layers, 4, dropout)
        self.head = GHIHead(d_model)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        satellite = self.satellite(batch).unsqueeze(1).expand(-1, 6, -1)
        wrf = self.wrf.encode_tokens(batch)
        solar = batch["solar_context"]
        features = self.fusion(torch.cat([satellite, wrf, solar], dim=-1))
        return {"prediction": self.head(self.temporal(features), solar)}


def build_model(name: str, d_model: int, layers: int, dropout: float) -> nn.Module:
    if name == "sat_convlstm":
        return SatConvLSTMModel(d_model, layers, dropout)
    if name == "wrf_transformer":
        return WRFTransformerModel(d_model, layers, dropout)
    if name == "early_fusion":
        return EarlyFusionTransformerModel(d_model, layers, dropout)
    raise ValueError(f"unknown model {name!r}; choices={MODEL_CHOICES}")


def training_loss(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = batch["target_mask"].bool()
    if not bool(mask.any()):
        raise ValueError("batch contains no valid GHI labels")
    return F.smooth_l1_loss(prediction[mask], batch["target"][mask])


def scalar_metrics(
    target: np.ndarray, prediction: np.ndarray, daylight: np.ndarray
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    daylight = np.asarray(daylight, dtype=bool)
    if target.size == 0:
        return {"count": 0}
    error = prediction - target
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    result: dict[str, Any] = {
        "count": int(target.size),
        "mae_wm2": float(np.mean(np.abs(error))),
        "rmse_wm2": float(np.sqrt(np.mean(error**2))),
        "mbe_wm2": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / denominator)
        if denominator > 1e-12 else None,
    }
    if daylight.any():
        day_target = target[daylight]
        day_error = error[daylight]
        day_rmse = float(np.sqrt(np.mean(day_error**2)))
        day_mean = float(np.mean(day_target))
        day_denominator = float(np.sum((day_target - day_mean) ** 2))
        result["daylight"] = {
            "count": int(day_target.size),
            "mean_target_wm2": day_mean,
            "mae_wm2": float(np.mean(np.abs(day_error))),
            "rmse_wm2": day_rmse,
            "mbe_wm2": float(np.mean(day_error)),
            "nrmse_pct": float(100.0 * day_rmse / max(abs(day_mean), DAYLIGHT_EPS)),
            "r2": float(1.0 - np.sum(day_error**2) / day_denominator)
            if day_denominator > 1e-12 else None,
        }
    else:
        result["daylight"] = {
            "count": 0,
            "mean_target_wm2": None,
            "mae_wm2": None,
            "rmse_wm2": None,
            "mbe_wm2": None,
            "nrmse_pct": None,
            "r2": None,
        }
    return result


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device, precision: str
) -> dict[str, Any]:
    model.eval()
    target_values: list[np.ndarray] = []
    prediction_values: list[np.ndarray] = []
    daylight_values: list[np.ndarray] = []
    lead_values: list[np.ndarray] = []
    station_values: list[str] = []
    records: list[dict[str, Any]] = []
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    with torch.no_grad():
        for raw in loader:
            batch = to_device(raw, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                prediction = model(batch)["prediction"].float()
            valid = batch["target_mask"].bool()
            daylight = (batch["solar_context"][..., 2] > 0.0) & valid
            target_np = batch["target"].detach().cpu().numpy()
            prediction_np = prediction.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            daylight_np = daylight.detach().cpu().numpy()
            for index, sample_id in enumerate(raw["sample_id"]):
                station = str(raw["station_id"][index])
                cycle = str(raw["cycle"][index])
                for lead in range(6):
                    if not bool(valid_np[index, lead]):
                        continue
                    records.append({
                        "sample_id": sample_id,
                        "station_id": station,
                        "cycle": cycle,
                        "lead_hour": lead + 1,
                        "target_wm2": float(target_np[index, lead]),
                        "prediction_wm2": float(prediction_np[index, lead]),
                        "daylight": int(daylight_np[index, lead]),
                    })
            target_values.append(target_np[valid_np])
            prediction_values.append(prediction_np[valid_np])
            daylight_values.append(daylight_np[valid_np])
            lead_values.append(
                np.broadcast_to(np.arange(1, 7, dtype=np.int64), valid_np.shape)[valid_np]
            )
            for index, station in enumerate(raw["station_id"]):
                station_values.extend([str(station)] * int(valid_np[index].sum()))
    if not target_values:
        raise ValueError("evaluation partition has no valid labels")
    target = np.concatenate(target_values)
    prediction = np.concatenate(prediction_values)
    daylight = np.concatenate(daylight_values).astype(bool)
    leads = np.concatenate(lead_values)
    stations = np.asarray(station_values)
    result = scalar_metrics(target, prediction, daylight)
    result["by_lead"] = {
        str(lead): scalar_metrics(
            target[leads == lead], prediction[leads == lead], daylight[leads == lead]
        )
        for lead in range(1, 7)
    }
    result["by_station_daylight"] = {}
    for station in sorted(set(stations.tolist())):
        selected = stations == station
        result["by_station_daylight"][station] = scalar_metrics(
            target[selected], prediction[selected], daylight[selected]
        ).get("daylight", {})
    return {
        "metrics": result,
        "records": records,
        "selection_score": result["daylight"]["nrmse_pct"],
    }


def write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id", "station_id", "cycle", "lead_hour", "target_wm2",
        "prediction_wm2", "daylight",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def load_reference() -> dict[str, Any]:
    # The dataset adapter exposes an AOD reference for compatibility, but AOD
    # is not used as an input by any baseline.
    return {"station_month_median": {}, "station_median": {}, "global_median": 0.0}


def make_dataset(
    root: Path, rows: list[dict[str, str]], normalization: dict[str, Any], formal: bool
) -> MultisourceCaseDataset:
    return MultisourceCaseDataset(
        root=root,
        rows=rows,
        normalization=normalization,
        aod_reference=load_reference(),
        include_future_satellite_auxiliary=False,
        require_formal_v4=formal,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
) -> float:
    model.train()
    values: list[float] = []
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    for raw in loader:
        batch = to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            prediction = model(batch)["prediction"]
            loss = training_loss(prediction, batch)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        values.append(float(loss.detach().cpu()))
    if not values:
        raise RuntimeError("training loader yielded no batches")
    return float(np.mean(values))


def run_smoke(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    model.train()
    raw = next(iter(train_loader))
    batch = to_device(raw, device)
    model.zero_grad(set_to_none=True)
    prediction = model(batch)["prediction"]
    loss = training_loss(prediction, batch)
    loss.backward()
    finite_gradients = sum(
        1 for parameter in model.parameters()
        if parameter.grad is not None and torch.isfinite(parameter.grad).all()
    )
    report = {
        "smoke": True,
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "batch_size": len(raw["sample_id"]),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "finite_gradient_parameter_count": finite_gradients,
        "formal_v4_required": False,
        "note": "CPU forward/backward smoke only; no performance training or test access.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split-config", type=Path,
        default=PROJECT_ROOT / "configs" / "dataset_split_17train_7eval_expanded_20260909.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="sat_convlstm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--smoke", action="store_true", help="local five-case CPU smoke only")
    args = parser.parse_args()

    if args.workers != 0:
        raise ValueError("workers must remain 0 because MultisourceCaseDataset is preloaded")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = resolve_device(args.device)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a CUDA device with bfloat16 support")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    rows = read_index(args.dataset_root)
    partitions = smoke_rows(rows) if args.smoke else current_protocol_rows(rows, split_config)
    formal = not args.smoke
    normalization_path = args.output_dir / "normalization.json"
    if normalization_path.is_file() and not args.smoke:
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    else:
        normalization = compute_normalization(
            [args.dataset_root / row["case_file"] for row in partitions["train"]]
        )
        normalization_path.write_text(
            json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
        )
    train_dataset = make_dataset(args.dataset_root, partitions["train"], normalization, formal)
    validation_dataset = make_dataset(
        args.dataset_root, partitions["validation"], normalization, formal
    )
    loader_kwargs = {
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": numpy_safe_collate,
    }
    station_counts = Counter(str(row["station_id"]).upper() for row in partitions["train"])
    sample_weights = torch.as_tensor(
        [1.0 / station_counts[str(row["station_id"]).upper()] for row in partitions["train"]],
        dtype=torch.double,
    )
    train_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(partitions["train"]),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )
    model = build_model(args.model, args.d_model, args.transformer_layers, args.dropout).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.smoke:
        run_smoke(args, model, train_loader, device)
        return

    audit = {
        "protocol": "17train_7eval_expanded_20260909",
        "split_config": str(args.split_config.resolve()),
        "split_config_sha256": sha256_file(args.split_config),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_index_sha256": sha256_file(args.dataset_root / "index.csv"),
        "model": args.model,
        "model_name": model.model_name,
        "seed": args.seed,
        "aod_enabled": False,
        "solfus_features_used": False,
        "wrf_dual_scale_inputs": ["d01", "d02"],
        "test_used_for_selection": False,
        "train_cases": len(partitions["train"]),
        "validation_cases": len(partitions["validation"]),
        "test_cases": len(partitions["test"]),
        "parameter_count": parameter_count,
        "allowed_inputs": {
            "sat_convlstm": ["satellite", "satellite_valid", "satellite_frame_mask", "solar_context"],
            "wrf_transformer": [
                "nwp_d01_3d", "nwp_d01_3d_valid", "nwp_d01_surface",
                "nwp_d02_3d", "nwp_d02_3d_valid", "nwp_d02_surface", "solar_context",
            ],
            "early_fusion": [
                "satellite", "satellite_valid", "satellite_frame_mask",
                "nwp_d01_3d", "nwp_d01_3d_valid", "nwp_d01_surface",
                "nwp_d02_3d", "nwp_d02_3d_valid", "nwp_d02_surface", "solar_context",
            ],
        }[args.model],
        "training_sampling": "station_balanced_weighted_replacement",
        "training_station_case_counts": dict(sorted(station_counts.items())),
        "train_normalization": str(normalization_path.resolve()),
        "normalization_sha256": json_sha256(normalization),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda,
        },
    }
    (args.output_dir / "run_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 3), min_lr=1e-6
    )
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.precision)
        validation = collect_predictions(model, validation_loader, device, args.precision)
        score = float(validation["selection_score"])
        scheduler.step(score)
        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation["metrics"],
            "validation_daylight_nrmse_pct": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - epoch_start,
        }
        history.append(item)
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)
        if math.isfinite(score) and score < best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model.model_name,
                    "model_key": args.model,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_daylight_nrmse_pct": score,
                    "parameter_count": parameter_count,
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    if not (args.output_dir / "best.pt").is_file():
        raise RuntimeError("no finite validation checkpoint was produced")
    checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    validation = collect_predictions(model, validation_loader, device, args.precision)
    write_predictions(args.output_dir / "validation_predictions.csv", validation["records"])

    # Test is loaded only after validation-based checkpoint selection.
    test_dataset = make_dataset(args.dataset_root, partitions["test"], normalization, formal)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )
    test = collect_predictions(model, test_loader, device, args.precision)
    write_predictions(args.output_dir / "test_predictions.csv", test["records"])
    summary = {
        "model": model.model_name,
        "model_key": args.model,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "elapsed_seconds": time.monotonic() - started,
        "selection_metric": "validation daylight nRMSE (%)",
        "validation": validation["metrics"],
        "test": test["metrics"],
        "test_loaded_after_selection": True,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    audit["best_epoch"] = best_epoch
    audit["epochs_completed"] = len(history)
    audit["test_loaded_after_selection"] = True
    (args.output_dir / "run_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
