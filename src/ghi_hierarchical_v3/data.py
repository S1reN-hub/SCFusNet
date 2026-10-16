"""Leakage-safe dataset utilities for the formal 1--6 h model."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_index(root: Path) -> list[dict[str, str]]:
    with (root / "index.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def compute_normalization(paths: list[Path]) -> dict:
    sums = {"nwp_3d": np.zeros(6), "nwp_surface": np.zeros(4), "satellite": np.zeros(7)}
    squares = {key: np.zeros_like(value) for key, value in sums.items()}
    counts = {key: np.zeros_like(value) for key, value in sums.items()}
    for path in paths:
        with np.load(str(path), allow_pickle=False) as case:
            nwp = case["nwp_3d"].astype(np.float64)
            nwp_valid = np.broadcast_to(case["nwp_3d_valid"].astype(bool)[:, None], nwp.shape)
            surface = case["nwp_surface"].astype(np.float64)
            satellite = case["satellite"].astype(np.float64)
            satellite_valid = case["satellite_valid"].astype(bool)
        for key, values, valid, axis in (
            ("nwp_3d", nwp, nwp_valid & np.isfinite(nwp), (0, 2, 3, 4)),
            ("nwp_surface", surface, np.isfinite(surface), (0, 2, 3)),
            ("satellite", satellite, satellite_valid & np.isfinite(satellite), (0, 2, 3)),
        ):
            safe = np.where(valid, values, 0.0)
            sums[key] += safe.sum(axis=axis)
            squares[key] += (safe * safe).sum(axis=axis)
            counts[key] += valid.sum(axis=axis)
    result = {}
    for key in sums:
        if np.any(counts[key] == 0):
            raise ValueError(f"normalization channel has no training data: {key}")
        mean = sums[key] / counts[key]
        variance = np.maximum(squares[key] / counts[key] - mean * mean, 1e-8)
        result[key] = {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": counts[key].tolist(),
        }
    return result


def aod_reference_for(case, reference: dict) -> np.ndarray:
    station = str(case["station_id"].item())
    result = []
    for value in case["valid_time_utc"].astype(str):
        month = value[5:7]
        result.append(
            reference["station_month_median"].get(
                f"{station}|{month}", reference["station_median"].get(station, reference["global_median"])
            )
        )
    return np.asarray(result, dtype=np.float32)


def _parse_utc(value: str) -> datetime:
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def derived_solar(case) -> dict[str, np.ndarray]:
    solar = case["solar"].astype(np.float32)
    subhour_cos = case["subhour_cos_sza"].astype(np.float32)
    valid_times = case["valid_time_utc"].astype(str)
    day_of_year = np.asarray([_parse_utc(value).timetuple().tm_yday for value in valid_times], dtype=np.float32)
    positive = np.maximum(subhour_cos, 0.0)
    air_mass_samples = np.minimum(1.0 / np.maximum(positive, 0.08), 12.5)
    weights = positive
    air_mass = (air_mass_samples * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-6)
    air_mass = np.where(solar[:, 2] > 0.0, air_mass, 0.0).astype(np.float32)
    stable_scale = solar[:, 1] + 20.0 * solar[:, 2]
    extraterrestrial_normal = 1361.0 * (1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0))
    physical_upper = np.where(
        solar[:, 2] > 0.0,
        np.minimum(2.0 * stable_scale, extraterrestrial_normal),
        0.0,
    ).astype(np.float32)
    low_sun = solar[:, 2] * np.clip((0.30 - solar[:, 0]) / 0.30, 0.0, 1.0)
    transition_support = np.maximum(solar[:, 3], low_sun).astype(np.float32)
    context = np.stack(
        [
            solar[:, 0],
            solar[:, 1] / 1000.0,
            solar[:, 2],
            solar[:, 3],
            air_mass / 12.5,
            np.sin(2.0 * np.pi * day_of_year / 365.0),
            np.cos(2.0 * np.pi * day_of_year / 365.0),
            low_sun,
        ],
        axis=-1,
    ).astype(np.float32)
    return {
        "solar_context": context,
        "air_mass": air_mass,
        "stable_scale": stable_scale.astype(np.float32),
        "physical_upper": physical_upper,
        "transition_support": transition_support,
    }


class MultisourceCaseDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, str]],
        normalization: dict,
        aod_reference: dict,
        ridge_teacher: dict[tuple[str, int], float] | None = None,
        clear_labels: dict[tuple[str, int], tuple[float, bool]] | None = None,
    ):
        self.samples = []
        nwp_mean = np.asarray(normalization["nwp_3d"]["mean"], np.float32)[None, :, None, None, None]
        nwp_std = np.asarray(normalization["nwp_3d"]["std"], np.float32)[None, :, None, None, None]
        surface_mean = np.asarray(normalization["nwp_surface"]["mean"], np.float32)[None, :, None, None]
        surface_std = np.asarray(normalization["nwp_surface"]["std"], np.float32)[None, :, None, None]
        sat_mean = np.asarray(normalization["satellite"]["mean"], np.float32)[None, :, None, None]
        sat_std = np.asarray(normalization["satellite"]["std"], np.float32)[None, :, None, None]
        for row in rows:
            with np.load(str(root / row["case_file"]), allow_pickle=False) as loaded:
                case = {name: np.asarray(loaded[name]) for name in loaded.files}
            nwp_valid = case["nwp_3d_valid"].astype(bool)
            nwp = (case["nwp_3d"].astype(np.float32) - nwp_mean) / nwp_std
            nwp = np.where(nwp_valid[:, None] & np.isfinite(nwp), np.clip(nwp, -8.0, 8.0), 0.0)
            surface = (case["nwp_surface"].astype(np.float32) - surface_mean) / surface_std
            surface = np.where(np.isfinite(surface), np.clip(surface, -8.0, 8.0), 0.0)
            sat_valid = case["satellite_valid"].astype(bool)
            satellite = (case["satellite"].astype(np.float32) - sat_mean) / sat_std
            satellite = np.where(sat_valid & np.isfinite(satellite), np.clip(satellite, -8.0, 8.0), 0.0)
            latitude = float(case["latitude"])
            longitude = float(case["longitude"])
            static = np.asarray(
                [
                    math.sin(math.radians(latitude)), math.cos(math.radians(latitude)),
                    math.sin(math.radians(longitude)), math.cos(math.radians(longitude)),
                    float(case["elevation_m"]) / 3000.0,
                    float(case["climate_id"]) / 6.0,
                    float(case["issue_offset_hours"]) / 72.0,
                ],
                dtype=np.float32,
            )
            solar = derived_solar(case)
            target = case["target"].astype(np.float32)
            target_mask = case["target_mask"].astype(bool)
            clear_index = target / np.maximum(solar["stable_scale"], 20.0)
            # Initial soft proxy.  It will be replaced by the minute-resolution
            # clear-state label when that optional data product is built.
            clear_target = np.clip((clear_index - 0.55) / 0.35, 0.0, 1.0).astype(np.float32)
            clear_mask = target_mask & (solar["solar_context"][:, 2] > 0.99) & (solar["stable_scale"] > 100.0)
            sample_id = str(case["sample_id"].item())
            if clear_labels is not None:
                clear_target = np.zeros(6, dtype=np.float32)
                clear_mask = np.zeros(6, dtype=bool)
                for lead in range(1, 7):
                    value = clear_labels.get((sample_id, lead))
                    if value is not None:
                        clear_target[lead - 1], clear_mask[lead - 1] = value
            teacher = np.zeros(6, dtype=np.float32)
            teacher_mask = np.zeros(6, dtype=bool)
            if ridge_teacher is not None:
                for lead in range(1, 7):
                    key = (sample_id, lead)
                    if key in ridge_teacher:
                        teacher[lead - 1] = ridge_teacher[key]
                        teacher_mask[lead - 1] = True
            sample = {
                "nwp_3d": torch.from_numpy(nwp),
                "nwp_3d_valid": torch.from_numpy(nwp_valid),
                "nwp_surface": torch.from_numpy(surface),
                "wrf_anchor": torch.from_numpy(case["wrf_anchor"].astype(np.float32)),
                "ridge_teacher": torch.from_numpy(teacher),
                "ridge_teacher_mask": torch.from_numpy(teacher_mask),
                "satellite": torch.from_numpy(satellite),
                "satellite_valid": torch.from_numpy(sat_valid),
                "satellite_frame_mask": torch.from_numpy(case["satellite_frame_mask"].astype(bool)),
                "satellite_age_hours": torch.from_numpy(case["satellite_age_hours"].astype(np.float32)),
                "satellite_source_id": torch.tensor(int(case["satellite_source_id"]), dtype=torch.long),
                "aod550": torch.from_numpy(case["aod550"].astype(np.float32)),
                "aod_reference": torch.from_numpy(aod_reference_for(case, aod_reference)),
                "aod_mask": torch.from_numpy(case["aod_mask"].astype(bool)),
                "solar_context": torch.from_numpy(solar["solar_context"]),
                "air_mass": torch.from_numpy(solar["air_mass"]),
                "physical_upper": torch.from_numpy(solar["physical_upper"]),
                "transition_support": torch.from_numpy(solar["transition_support"]),
                "static": torch.from_numpy(static),
                "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
                "clear_target": torch.from_numpy(clear_target),
                "clear_target_mask": torch.from_numpy(clear_mask),
                "station_id": str(case["station_id"].item()),
                "sample_id": sample_id,
                "cycle": row["cycle"],
            }
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]
