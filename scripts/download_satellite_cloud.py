#!/usr/bin/env python3
"""Build quality-controlled GOES cloud windows for satellite-privileged KD.

The operational sample is an hourly interval ``[t-1h, t)``.  Every GOES
scan in that interval is aligned on the all-sky CloudMask fixed grid, filtered with its own
quality information, and aggregated before it is paired with the hourly GHI
label.  Full files are cached once per scan and reused for every station; this
script does not claim netCDF/HDF5 window reads are S3 byte-range reads.

Phase A currently supports GOES-East and GOES-West:
  * GOES-16 through 2025-04-07 15:00 UTC
  * GOES-19 from    2025-04-07 15:00 UTC
  * GOES-18 for the current GOES-West sector

Himawari and Meteosat adapters are deliberately left for the later gated
phase described in docs/CloudKD卫星特权信息蒸馏修订方案_20260722.md.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


GOES_H = 35_786_023.0
EARTH_A = 6_378_137.0
EARTH_E2 = 0.00669438002290
GOES_2KM_FULL_DISK = 5424
PIXEL_RAD_2KM = 56e-6
HALF_DISK = GOES_2KM_FULL_DISK / 2
GOES19_TRANSITION_UTC = datetime(2025, 4, 7, 15, 0)


@dataclass(frozen=True)
class GoesPlatform:
    name: str
    bucket: str
    longitude: float


GOES_PLATFORMS = {
    # The operational spacecraft longitude is approximately 75.2 W, while
    # distributed ABI fixed-grid products use a -75.0 degree projection
    # origin.  Geometry must follow the product projection.
    "GOES-16": GoesPlatform("GOES-16", "noaa-goes16", -75.0),
    "GOES-19": GoesPlatform("GOES-19", "noaa-goes19", -75.0),
    "GOES-18": GoesPlatform("GOES-18", "noaa-goes18", -137.0),
}


@dataclass(frozen=True)
class ProductSpec:
    name: str
    prefix: str
    variables: tuple[str, ...]
    required: bool = True


# Clear Sky Mask and Cloud Top Phase are distinct products.  Cloud particle
# size (effective radius) is also a distinct CPS product, even though COD and
# CPS are retrieved by the same physical algorithm.
GOES_PRODUCTS = {
    "CloudMask": ProductSpec("CloudMask", "ABI-L2-ACMF", ("BCM", "ACM")),
    # COD is sunlight-dependent. It enriches daytime frames but must not
    # decide whether an otherwise valid all-sky cloud frame exists.
    "COD": ProductSpec("COD", "ABI-L2-CODF", ("COD",), required=False),
    # ACTP is cloud-top phase and is available throughout the 2021+ pilot.
    # The enterprise 2 km cloud-top pressure archive (ACHP2KMF) starts in
    # 2023, so pressure is optional for the 2021-2022 historical period.
    "Phase": ProductSpec("Phase", "ABI-L2-ACTPF", ("Phase",), required=False),
    "CTP": ProductSpec("CTP", "ABI-L2-ACHP2KMF", ("PRES",), required=False),
    # The older GOES-R files expose the same particle-size retrieval as PSD;
    # newer enterprise files use CPS.
    "CPS": ProductSpec("CPS", "ABI-L2-CPSF", ("CPS", "PSD"), required=False),
}

CHANNEL_NAMES = (
    "COD",
    "cloud_fraction",
    "cloud_top_pressure_hpa",
    "CER_liq_um",
    "CER_ice_um",
)

SCAN_RE = re.compile(r"_s(?P<year>\d{4})(?P<doy>\d{3})(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})")


def platform_for_datetime(value: datetime) -> GoesPlatform:
    """Return the operational GOES-East platform for a UTC datetime."""
    return GOES_PLATFORMS["GOES-19" if value >= GOES19_TRANSITION_UTC else "GOES-16"]


def lonlat_to_goes_scan_angles(lon: float, lat: float, lon0: float = -75.0) -> tuple[float, float]:
    """Convert WGS84 longitude/latitude to GOES fixed-grid scan angles."""
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lon0_r = math.radians(lon0)
    h_sat = GOES_H + EARTH_A

    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    n_curv = EARTH_A / math.sqrt(1.0 - EARTH_E2 * sin_lat**2)
    x_p = n_curv * cos_lat * math.cos(lon_r)
    y_p = n_curv * cos_lat * math.sin(lon_r)
    z_p = n_curv * (1.0 - EARTH_E2) * sin_lat
    x_s = h_sat * math.cos(lon0_r)
    y_s = h_sat * math.sin(lon0_r)

    vx, vy, vz = x_p - x_s, y_p - y_s, z_p
    z_comp = vx * -math.cos(lon0_r) + vy * -math.sin(lon0_r)
    if z_comp <= 0:
        raise ValueError(f"({lat}, {lon}) is not visible from GOES at {lon0} degrees")
    x_comp = vx * -math.sin(lon0_r) + vy * math.cos(lon0_r)
    y_comp = vz
    return math.atan2(x_comp, z_comp), math.atan2(y_comp, z_comp)


def lonlat_to_goes_pixel(lon: float, lat: float, lon0: float = -75.0) -> tuple[float, float]:
    x_scan, y_scan = lonlat_to_goes_scan_angles(lon, lat, lon0)
    return HALF_DISK + x_scan / PIXEL_RAD_2KM, HALF_DISK - y_scan / PIXEL_RAD_2KM


def pixel_on_disk(col: float, row: float, margin: int = 35) -> bool:
    return margin <= col < GOES_2KM_FULL_DISK - margin and margin <= row < GOES_2KM_FULL_DISK - margin


def parse_scan_time(key: str) -> datetime:
    match = SCAN_RE.search(Path(key).name)
    if not match:
        raise ValueError(f"cannot parse GOES scan start time from {key}")
    parts = {name: int(value) for name, value in match.groupdict().items()}
    day = datetime(parts["year"], 1, 1) + timedelta(days=parts["doy"] - 1)
    return day.replace(hour=parts["hour"], minute=parts["minute"], second=parts["second"])


def _hour_starts(start: datetime, end: datetime) -> Iterable[datetime]:
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor < end:
        yield cursor
        cursor += timedelta(hours=1)


def list_product_scans(fs, platform: GoesPlatform, spec: ProductSpec, start: datetime, end: datetime) -> list[tuple[datetime, str]]:
    """List product scans whose scan start lies in ``[start, end)``."""
    found: dict[str, datetime] = {}
    for hour in _hour_starts(start, end):
        doy = int(hour.strftime("%j"))
        prefix = f"s3://{platform.bucket}/{spec.prefix}/{hour.year}/{doy:03d}/{hour.hour:02d}/"
        try:
            keys = fs.ls(prefix)
        except FileNotFoundError:
            continue
        for raw_key in keys:
            key = str(raw_key)
            if not key.endswith(".nc"):
                continue
            try:
                scan_time = parse_scan_time(key)
            except ValueError:
                continue
            if start <= scan_time < end:
                found[key] = scan_time
    return sorted(((scan_time, key) for key, scan_time in found.items()), key=lambda item: item[0])


def synchronize_scans(
    scans: Mapping[str, Sequence[tuple[datetime, str]]],
    required: Sequence[str] | None = None,
    tolerance: timedelta = timedelta(minutes=4),
) -> list[dict[str, object]]:
    """Match products to an all-sky CloudMask scan without mixing scan cycles."""
    if required is None:
        required = tuple(name for name, spec in GOES_PRODUCTS.items() if spec.required)
    anchor_product = "CloudMask" if scans.get("CloudMask") else next(
        (name for name in required if scans.get(name)), None
    )
    if anchor_product is None:
        return []
    bundles: list[dict[str, object]] = []
    for anchor_time, anchor_key in scans.get(anchor_product, ()):
        files: dict[str, str] = {anchor_product: anchor_key}
        times: dict[str, datetime] = {anchor_product: anchor_time}
        missing: list[str] = []
        for product, candidates in scans.items():
            if product == anchor_product or not candidates:
                continue
            match_time, match_key = min(candidates, key=lambda item: abs(item[0] - anchor_time))
            if abs(match_time - anchor_time) <= tolerance:
                files[product] = match_key
                times[product] = match_time
        for product in required:
            if product not in files:
                missing.append(product)
        if not missing:
            bundles.append({"anchor_time": anchor_time, "files": files, "times": times})
    return bundles


def goes_product_specs(
    sector: str = "full", products: Sequence[str] | None = None
) -> dict[str, ProductSpec]:
    """Return the requested Full Disk or CONUS products with stable channel names."""
    selected = set(products or GOES_PRODUCTS)
    unknown = selected.difference(GOES_PRODUCTS)
    if unknown:
        raise ValueError(f"unknown GOES products: {sorted(unknown)}")
    if sector not in {"full", "conus"}:
        raise ValueError(f"unsupported GOES sector: {sector}")
    result: dict[str, ProductSpec] = {}
    for name, spec in GOES_PRODUCTS.items():
        if name not in selected:
            continue
        prefix = spec.prefix if sector == "full" else spec.prefix[:-1] + "C"
        result[name] = ProductSpec(spec.name, prefix, spec.variables, spec.required)
    if "CloudMask" not in result:
        raise ValueError("CloudMask must be selected as the all-sky frame anchor")
    return result


def _resolve_variable(dataset, aliases: Sequence[str]):
    for name in aliases:
        if name in dataset.variables:
            return dataset[name]
    raise KeyError(f"none of variables {tuple(aliases)} found; available={tuple(dataset.data_vars)}")


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(values, dtype=np.float64) - target)))


def _reference_coordinates(dataset, x_scan: float, y_scan: float, half_win: int) -> tuple[np.ndarray, np.ndarray]:
    if "x" not in dataset.coords or "y" not in dataset.coords:
        raise KeyError("GOES dataset is missing fixed-grid x/y coordinates")
    x_values = np.asarray(dataset["x"].values, dtype=np.float64)
    y_values = np.asarray(dataset["y"].values, dtype=np.float64)
    cx, cy = _nearest_index(x_values, x_scan), _nearest_index(y_values, y_scan)
    if cx - half_win < 0 or cy - half_win < 0 or cx + half_win >= len(x_values) or cy + half_win >= len(y_values):
        raise ValueError("station window crosses the GOES fixed-grid boundary")
    return x_values[cx - half_win : cx + half_win + 1], y_values[cy - half_win : cy + half_win + 1]


def _align_to_reference(data_array, x_ref: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Nearest-neighbour align a native GOES grid to a reference x/y window."""
    if "x" not in data_array.coords or "y" not in data_array.coords:
        raise KeyError(f"{data_array.name} has no x/y coordinates")
    import xarray as xr

    selected = data_array.sel(
        x=xr.DataArray(x_ref, dims="x"),
        y=xr.DataArray(y_ref, dims="y"),
        method="nearest",
    )
    values = np.asarray(selected.values)
    values = np.squeeze(values)
    expected = (len(y_ref), len(x_ref))
    if values.shape != expected:
        raise ValueError(f"aligned {data_array.name} shape {values.shape} != {expected}")
    return values


def quality_mask_from_dqf(dqf_array, values: np.ndarray) -> np.ndarray:
    """Decode a GOES DQF using CF flag masks, falling back to DQF==0.

    Many cloud optical products use a packed bit field rather than a simple
    ordinal flag.  The CF attributes identify the bit pattern named
    ``good_quality_qf``; using ``DQF == 0`` unconditionally is incorrect for
    those products.
    """
    raw = np.asarray(values)
    finite = np.isfinite(raw)
    masks = np.atleast_1d(dqf_array.attrs.get("flag_masks", []))
    flag_values = np.atleast_1d(dqf_array.attrs.get("flag_values", []))
    meanings = str(dqf_array.attrs.get("flag_meanings", "")).split()
    if len(masks) == len(flag_values) == len(meanings):
        for index, meaning in enumerate(meanings):
            if meaning == "good_quality_qf":
                integer = np.where(finite, raw, 0).astype(np.int64)
                return finite & ((integer & int(masks[index])) == int(flag_values[index]))
    return finite & (raw == 0)


def _data_and_validity(
    dataset,
    aliases: Sequence[str],
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    valid_range: tuple[float, float] | None = None,
    require_good_dqf: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    data_array = _resolve_variable(dataset, aliases)
    values = _align_to_reference(data_array, x_ref, y_ref).astype(np.float32)
    valid = np.isfinite(values)
    if valid_range is not None:
        valid &= (values >= valid_range[0]) & (values <= valid_range[1])
    if require_good_dqf and "DQF" in dataset.variables:
        dqf_array = dataset["DQF"]
        dqf = _align_to_reference(dqf_array, x_ref, y_ref)
        valid &= quality_mask_from_dqf(dqf_array, dqf)
    return values, valid


def _dqf_good_fraction(dataset, x_ref: np.ndarray, y_ref: np.ndarray, available: np.ndarray) -> float:
    """Return strict-good DQF coverage among pixels with a retrieved value."""
    if not available.any() or "DQF" not in dataset.variables:
        return 0.0
    dqf_array = dataset["DQF"]
    dqf = _align_to_reference(dqf_array, x_ref, y_ref)
    good = quality_mask_from_dqf(dqf_array, dqf)
    return float(good[available].mean())


def extract_window_channels(
    datasets: Mapping[str, object],
    x_scan: float,
    y_scan: float,
    half_win: int = 31,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Extract the five canonical channels plus an independent validity mask."""
    reference_name = "COD" if "COD" in datasets else "CloudMask"
    if reference_name not in datasets:
        raise KeyError("CloudMask or COD is required as the 2 km reference grid")
    x_ref, y_ref = _reference_coordinates(datasets[reference_name], x_scan, y_scan, half_win)
    shape = (len(y_ref), len(x_ref))
    window = np.full((*shape, len(CHANNEL_NAMES)), np.nan, dtype=np.float32)
    validity = np.zeros((*shape, len(CHANNEL_NAMES)), dtype=bool)

    # COD/CPS publish physically usable retrieved values even when the packed
    # DQF marks them degraded (for example, an ice-phase penalty).  Keep those
    # finite retrievals and report strict-good coverage separately; otherwise
    # cloudy pixels can all disappear while only clear-sky zeros survive.
    quality_metadata: dict[str, float | list[str]] = {
        "available_products": sorted(datasets)
    }
    if "COD" in datasets:
        cod, cod_valid = _data_and_validity(
            datasets["COD"], ("COD",), x_ref, y_ref, (0.0, 160.0), require_good_dqf=False
        )
        window[:, :, 0], validity[:, :, 0] = cod, cod_valid
        quality_metadata["COD_dqf_good_fraction"] = _dqf_good_fraction(
            datasets["COD"], x_ref, y_ref, cod_valid
        )

    mask_data = datasets["CloudMask"]
    mask_array = _resolve_variable(mask_data, ("BCM", "ACM"))
    raw_mask = _align_to_reference(mask_array, x_ref, y_ref).astype(np.float32)
    if mask_array.name == "BCM":
        cloud = (raw_mask >= 1).astype(np.float32)
        mask_valid = np.isfinite(raw_mask) & np.isin(raw_mask, (0, 1))
    else:
        cloud = (raw_mask >= 2).astype(np.float32)
        mask_valid = np.isfinite(raw_mask) & np.isin(raw_mask, (0, 1, 2, 3))
    if "DQF" in mask_data.variables:
        dqf_array = mask_data["DQF"]
        dqf = _align_to_reference(dqf_array, x_ref, y_ref)
        mask_valid &= quality_mask_from_dqf(dqf_array, dqf)
    window[:, :, 1], validity[:, :, 1] = cloud, mask_valid

    if "CTP" in datasets:
        ctp, ctp_valid = _data_and_validity(datasets["CTP"], ("PRES",), x_ref, y_ref, (0.0, 110000.0))
        units = str(_resolve_variable(datasets["CTP"], ("PRES",)).attrs.get("units", "")).lower()
        if ("pa" in units and "hpa" not in units) or (np.nanmedian(ctp[ctp_valid]) if ctp_valid.any() else 0) > 2000:
            ctp = ctp / 100.0
        ctp_valid &= (ctp >= 0.0) & (ctp <= 1100.0)
        window[:, :, 2], validity[:, :, 2] = ctp, ctp_valid

    if "Phase" in datasets and "CPS" in datasets:
        phase, phase_valid = _data_and_validity(datasets["Phase"], ("Phase",), x_ref, y_ref, (0.0, 5.0))
        cps, cps_valid = _data_and_validity(
            datasets["CPS"], ("CPS", "PSD"), x_ref, y_ref, (0.0, 160.0), require_good_dqf=False
        )
        quality_metadata["CPS_dqf_good_fraction"] = _dqf_good_fraction(
            datasets["CPS"], x_ref, y_ref, cps_valid
        )
        liquid = phase_valid & np.isin(phase, (1, 2)) & cps_valid
        ice = phase_valid & (phase == 4) & cps_valid
        window[:, :, 3] = cps
        window[:, :, 4] = cps
        validity[:, :, 3] = liquid
        validity[:, :, 4] = ice

    window[~validity] = np.nan
    metadata = {
        "x_min_rad": float(np.min(x_ref)),
        "x_max_rad": float(np.max(x_ref)),
        "y_min_rad": float(np.min(y_ref)),
        "y_max_rad": float(np.max(y_ref)),
        **quality_metadata,
        "channel_valid_fraction": validity.mean(axis=(0, 1)).tolist(),
    }
    return window, validity, metadata


def _masked_stats(values: np.ndarray) -> tuple[float, float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 0.0, 0.0, 0.0
    mean, std = float(clean.mean()), float(clean.std())
    if clean.size < 3 or std <= 1e-12:
        skew = 0.0
    else:
        skew = float(np.mean(((clean - mean) / std) ** 3))
    return mean, std, skew


def compute_spatial_features(
    window: np.ndarray,
    validity: np.ndarray,
    solar_azimuth_deg: float = 180.0,
) -> tuple[np.ndarray, list[str]]:
    """Compute quality-aware, solar-directional spatial features."""
    if window.shape != validity.shape or window.ndim != 3:
        raise ValueError("window and validity must have identical (H,W,C) shapes")
    height, width, channels = window.shape
    if channels != len(CHANNEL_NAMES):
        raise ValueError(f"expected {len(CHANNEL_NAMES)} channels, got {channels}")

    features: list[float] = []
    names: list[str] = []
    centre = (height // 2, width // 2)
    for channel, name in enumerate(CHANNEL_NAMES):
        values = np.where(validity[:, :, channel], window[:, :, channel], np.nan)
        mean, std, skew = _masked_stats(values)
        centre_value = float(values[centre]) if np.isfinite(values[centre]) else 0.0
        valid_fraction = float(validity[:, :, channel].mean())
        filled = np.where(np.isfinite(values), values, mean)
        gy, gx = np.gradient(filled)
        gradient = np.hypot(gx, gy)
        gradient_mean = float(gradient[validity[:, :, channel]].mean()) if validity[:, :, channel].any() else 0.0
        features.extend((mean, std, skew, centre_value, gradient_mean, valid_fraction))
        names.extend(f"{name}_{suffix}" for suffix in ("mean", "std", "skew", "centre", "gradient", "valid_fraction"))

    rows, cols = np.indices((height, width))
    dx = cols - centre[1]
    dy_north = centre[0] - rows
    azimuth = math.radians(solar_azimuth_deg)
    along = dx * math.sin(azimuth) + dy_north * math.cos(azimuth)
    cross = dx * math.cos(azimuth) - dy_north * math.sin(azimuth)
    directional_masks = {
        "up_sun": along >= 0,
        "down_sun": along < 0,
        "cross_left": cross >= 0,
        "cross_right": cross < 0,
    }
    cod = np.where(validity[:, :, 0], window[:, :, 0], np.nan)
    for label, direction_mask in directional_masks.items():
        mean, _, _ = _masked_stats(cod[direction_mask])
        features.append(mean)
        names.append(f"COD_{label}_mean")
    return np.asarray(features, dtype=np.float32), names


def aggregate_hourly_scans(
    scan_results: Sequence[Mapping[str, object]],
    solar_azimuth_deg: float,
) -> dict[str, object]:
    """Aggregate all quality-controlled scans in one GHI label interval."""
    if not scan_results:
        raise ValueError("cannot aggregate an hour without valid satellite scans")
    windows = np.stack([np.asarray(item["window"], dtype=np.float32) for item in scan_results])
    validity = np.stack([np.asarray(item["validity"], dtype=bool) for item in scan_results])
    count = validity.sum(axis=0)
    total = np.where(validity, windows, 0.0).sum(axis=0)
    mean_window = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    squared = np.where(validity, (windows - mean_window) ** 2, 0.0).sum(axis=0)
    std_window = np.sqrt(np.divide(squared, count, out=np.full_like(total, np.nan), where=count > 0))
    hourly_validity = count > 0
    features, names = compute_spatial_features(mean_window, hourly_validity, solar_azimuth_deg)

    temporal_features: list[float] = []
    temporal_names: list[str] = []
    for channel, name in enumerate(CHANNEL_NAMES):
        per_scan_means = []
        for scan_index in range(len(scan_results)):
            values = windows[scan_index, :, :, channel][validity[scan_index, :, :, channel]]
            if values.size:
                per_scan_means.append(float(values.mean()))
        temporal_features.extend((float(np.mean(per_scan_means)) if per_scan_means else 0.0,
                                  float(np.std(per_scan_means)) if per_scan_means else 0.0))
        temporal_names.extend((f"{name}_scan_mean", f"{name}_temporal_std"))
    features = np.concatenate([features, np.asarray(temporal_features, dtype=np.float32)])
    names.extend(temporal_names)
    return {
        "window_mean": mean_window.astype(np.float32),
        "window_std": std_window.astype(np.float32),
        "validity": hourly_validity,
        "valid_count": count.astype(np.uint8),
        "features": features,
        "feature_names": names,
        "scan_count": len(scan_results),
        "scan_times": [str(item["scan_time"]) for item in scan_results],
        "scan_quality": [dict(item.get("grid", {})) for item in scan_results],
    }


def approximate_solar_position(value: datetime, latitude: float, longitude: float) -> tuple[float, float]:
    """Return approximate solar zenith and azimuth (degrees) for UTC time."""
    fractional_hour = value.hour + value.minute / 60.0 + value.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (value.timetuple().tm_yday - 1 + (fractional_hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                         - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    true_solar_minutes = (fractional_hour * 60.0 + eqtime + 4.0 * longitude) % 1440.0
    hour_angle = math.radians(true_solar_minutes / 4.0 - 180.0)
    lat = math.radians(latitude)
    cos_zenith = max(-1.0, min(1.0, math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)))
    zenith = math.degrees(math.acos(cos_zenith))
    azimuth = (math.degrees(math.atan2(math.sin(hour_angle), math.cos(hour_angle) * math.sin(lat) - math.tan(decl) * math.cos(lat))) + 180.0) % 360.0
    return zenith, azimuth


def _normalise_s3_key(key: str, bucket: str) -> str:
    if key.startswith("s3://"):
        return key
    if key.startswith(f"{bucket}/"):
        return f"s3://{key}"
    return f"s3://{bucket}/{key.lstrip('/')}"


def _public_s3_url(key: str, bucket: str) -> str:
    value = key.removeprefix("s3://")
    if value.startswith(f"{bucket}/"):
        value = value[len(bucket) + 1 :]
    mirror = os.environ.get("GOES_PUBLIC_MIRROR", "").lower()
    if mirror == "gcp":
        match = re.fullmatch(r"noaa-goes(\d+)", bucket)
        if match:
            return (
                f"https://storage.googleapis.com/gcp-public-data-goes-{match.group(1)}/"
                f"{value.lstrip('/')}"
            )
    if mirror in {"gcp-json", "google-json"}:
        match = re.fullmatch(r"noaa-goes(\d+)", bucket)
        if match:
            object_name = urllib.parse.quote(value.lstrip("/"), safe="")
            return (
                "https://storage.googleapis.com/download/storage/v1/b/"
                f"gcp-public-data-goes-{match.group(1)}/o/{object_name}?alt=media"
            )
    if mirror in {"aws-path", "s3-path"}:
        # Some campus routes handle the path-style AWS endpoint far better
        # than the virtual-hosted bucket endpoint.  Objects and checksums are
        # identical; only the HTTPS route changes.
        return f"https://s3.amazonaws.com/{bucket}/{value.lstrip('/')}"
    return f"https://{bucket}.s3.amazonaws.com/{value.lstrip('/')}"


def _open_public_request(request: urllib.request.Request, timeout: int):
    """Open bulk data directly when the selected mirror outperforms the proxy.

    Archive discovery still uses the configured S3 client/proxy.  This switch only
    affects the large immutable netCDF object transfer and is opt-in via the runner.
    """
    if os.environ.get("GOES_DOWNLOAD_NO_PROXY", "").lower() in {"1", "true", "yes"}:
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=timeout
        )
    return urllib.request.urlopen(request, timeout=timeout)


def cached_download(fs, key: str, bucket: str, cache_dir: Path) -> Path:
    """Download one public netCDF with bounded retries and atomic promotion."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / Path(key).name
    if target.exists() and target.stat().st_size > 0:
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    url = _public_s3_url(key, bucket)
    if os.environ.get("GOES_USE_CURL", "").lower() in {"1", "true", "yes"}:
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if curl is None:
            raise RuntimeError("GOES_USE_CURL is set but curl was not found")
        try:
            info = fs.info(_s3_path(str(key), bucket))
            expected = int(info.get("size", info.get("Size", 0)))
        except Exception:
            expected = 0
        if temporary.exists() and expected > 0 and temporary.stat().st_size > expected:
            temporary.unlink()
        command = [
            curl,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "10",
            "--retry-delay",
            "2",
            "--continue-at",
            "-",
            "--connect-timeout",
            "30",
            "--max-time",
            "900",
            "--output",
            str(temporary),
            url,
        ]
        if os.environ.get("GOES_DOWNLOAD_NO_PROXY", "").lower() in {"1", "true", "yes"}:
            command[1:1] = ["--noproxy", "*"]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"curl failed with status {completed.returncode} for {url}"
            )
        actual = temporary.stat().st_size
        if actual <= 0 or (expected > 0 and actual != expected):
            raise IOError(
                f"incomplete curl object: expected={expected}, actual={actual}"
            )
        os.replace(temporary, target)
        return target
    try:
        info = fs.info(_s3_path(str(key), bucket))
        object_size = int(info.get("size", info.get("Size", 0)))
    except Exception:
        object_size = 0
    if temporary.exists() and object_size > 0:
        partial_size = temporary.stat().st_size
        if partial_size == object_size:
            os.replace(temporary, target)
            return target
        if partial_size > object_size:
            temporary.unlink()
    last_error: Exception | None = None
    maximum_attempts = 10
    for attempt in range(1, maximum_attempts + 1):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"User-Agent": "SWRadNet/1.0"}
            if offset > 0:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with _open_public_request(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                base_size = offset if append else 0
                content_range = response.headers.get("Content-Range", "")
                expected = object_size
                if expected <= 0 and "/" in content_range:
                    tail = content_range.rsplit("/", 1)[-1]
                    if tail.isdigit():
                        expected = int(tail)
                if expected <= 0:
                    content_length = int(response.headers.get("Content-Length", "0"))
                    if content_length > 0:
                        expected = base_size + content_length
                with temporary.open(mode) as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            actual = temporary.stat().st_size
            if actual <= 0 or (expected > 0 and actual != expected):
                raise IOError(f"incomplete S3 object: expected={expected}, actual={actual}")
            os.replace(temporary, target)
            return target
        except urllib.error.HTTPError as error:
            if error.code == 416 and temporary.exists():
                content_range = error.headers.get("Content-Range", "")
                match = re.search(r"\*/(\d+)$", content_range)
                expected = int(match.group(1)) if match else object_size
                actual = temporary.stat().st_size
                if expected > 0 and actual == expected:
                    os.replace(temporary, target)
                    return target
                if expected > 0 and actual > expected:
                    temporary.unlink()
            last_error = error
            if attempt < maximum_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            last_error = error
            if attempt < maximum_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"failed to download {url} after {maximum_attempts} attempts: {last_error}"
    )


@contextmanager
def load_scan_bundle(fs, platform: GoesPlatform, bundle: Mapping[str, object], cache_dir: Path):
    import xarray as xr

    datasets = {}
    try:
        for product, key in bundle["files"].items():
            path = cached_download(fs, str(key), platform.bucket, cache_dir)
            # GOES science variables are indexed by scan coordinates; decoded
            # time coordinates are not needed here.  Keeping times numeric also
            # avoids a hard cftime dependency for a small set of legacy files.
            datasets[product] = xr.open_dataset(
                path,
                engine="h5netcdf",
                mask_and_scale=True,
                decode_times=False,
            )
        yield datasets
    finally:
        for dataset in datasets.values():
            dataset.close()


def build_hourly_sample(
    fs,
    platform: GoesPlatform,
    station: Mapping[str, object],
    valid_time: datetime,
    cache_dir: Path,
    half_win: int = 31,
    dry_run: bool = False,
) -> dict[str, object]:
    """Build one hourly satellite sample matching GHI interval ``[t-1h,t)``."""
    interval_start, interval_end = valid_time - timedelta(hours=1), valid_time
    scans = {
        name: list_product_scans(fs, platform, spec, interval_start, interval_end)
        for name, spec in GOES_PRODUCTS.items()
    }
    bundles = synchronize_scans(scans)
    if dry_run:
        return {"status": "DRY_RUN", "bundle_count": len(bundles), "scans": {k: len(v) for k, v in scans.items()}}
    if not bundles:
        return {"status": "NO_SYNCHRONIZED_SCANS", "bundle_count": 0, "scans": {k: len(v) for k, v in scans.items()}}

    x_scan, y_scan = lonlat_to_goes_scan_angles(float(station["longitude"]), float(station["latitude"]), platform.longitude)
    scan_results = []
    errors = []
    for bundle in bundles:
        try:
            with load_scan_bundle(fs, platform, bundle, cache_dir) as datasets:
                window, validity, grid_metadata = extract_window_channels(datasets, x_scan, y_scan, half_win)
            scan_results.append({"window": window, "validity": validity,
                                 "scan_time": bundle["anchor_time"].isoformat(), "grid": grid_metadata})
        except Exception as exc:
            errors.append(f"{bundle['anchor_time'].isoformat()}: {type(exc).__name__}: {exc}")
    if not scan_results:
        return {"status": "EXTRACTION_FAILED", "errors": errors, "bundle_count": len(bundles)}

    solar_zenith, solar_azimuth = approximate_solar_position(
        interval_start + timedelta(minutes=30), float(station["latitude"]), float(station["longitude"])
    )
    hourly = aggregate_hourly_scans(scan_results, solar_azimuth)
    hourly.update({
        "status": "SUCCESS",
        "platform": platform.name,
        "valid_time": valid_time.isoformat(),
        "interval_start": interval_start.isoformat(),
        "interval_end": interval_end.isoformat(),
        "solar_zenith_deg": solar_zenith,
        "solar_azimuth_deg": solar_azimuth,
        "errors": errors,
        "bundle_count": len(bundles),
    })
    return hourly


def save_hourly_sample(result: Mapping[str, object], station_id: str, output_dir: Path) -> Path:
    valid_time = datetime.fromisoformat(str(result["valid_time"]))
    target_dir = output_dir / station_id / f"{valid_time.year:04d}" / f"{valid_time.month:02d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{valid_time:%Y%m%d_%H%M}.npz"
    array_keys = {
        "window_mean", "window_std", "validity", "valid_count", "features",
        "feature_names", "scan_windows", "scan_validity",
    }
    metadata = {key: value for key, value in result.items() if key not in array_keys}
    arrays = {
        "window_mean": result["window_mean"],
        "window_std": result["window_std"],
        "validity": result["validity"],
        "valid_count": result["valid_count"],
        "features": result["features"],
        "feature_names": np.asarray(result["feature_names"], dtype="U64"),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
    }
    if "scan_windows" in result:
        arrays["scan_windows"] = np.asarray(result["scan_windows"], dtype=np.float16)
    if "scan_validity" in result:
        arrays["scan_validity"] = np.asarray(result["scan_validity"], dtype=bool)
    np.savez_compressed(target, **arrays)
    return target


def load_stations(path: Path) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(station["id"]): station for station in payload["stations"]}


def parse_hours(value: str) -> list[int]:
    start, end = (int(part) for part in value.split("-", 1))
    if not (0 <= start <= 24 and 0 <= end <= 24 and start <= end):
        raise ValueError("hours must be END_HOUR-END_HOUR within 0..24")
    return list(range(start, end + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build quality-controlled hourly GOES-East cloud windows")
    parser.add_argument("--station", default="DRA", help="Comma-separated station IDs")
    parser.add_argument("--dates", default="2021-03-23", help="Comma-separated UTC dates")
    parser.add_argument("--hours", default="13-23", help="Hourly label end times, inclusive")
    parser.add_argument("--output-dir", default="data/satellite_cloud/hourly_features")
    parser.add_argument("--cache-dir", default="data/satellite_cloud/cache/goes")
    parser.add_argument("--config", default="configs/global_radiation_truth_stations.json")
    parser.add_argument("--half-window", type=int, default=31)
    parser.add_argument(
        "--platform",
        choices=("auto", "GOES-16", "GOES-18", "GOES-19"),
        default="auto",
    )
    parser.add_argument("--audit", action="store_true", help="List synchronized scans without downloading files")
    parser.add_argument("--dry-run", action="store_true", help="Alias of --audit")
    args = parser.parse_args()

    try:
        import s3fs
    except ImportError as exc:
        raise SystemExit("s3fs is required for GOES access: pip install s3fs") from exc

    stations = load_stations(Path(args.config))
    selected = []
    for station_id in (item.strip() for item in args.station.split(",")):
        if station_id not in stations:
            raise SystemExit(f"unknown station {station_id}; available={sorted(stations)}")
        selected.append(stations[station_id])
    dates = [datetime.strptime(item.strip(), "%Y-%m-%d") for item in args.dates.split(",")]
    hours = parse_hours(args.hours)
    fs = s3fs.S3FileSystem(anon=True)

    success, failure = 0, 0
    for date in dates:
        for end_hour in hours:
            valid_time = date + timedelta(hours=end_hour)
            platform = platform_for_datetime(valid_time) if args.platform == "auto" else GOES_PLATFORMS[args.platform]
            for station in selected:
                result = build_hourly_sample(
                    fs, platform, station, valid_time, Path(args.cache_dir),
                    half_win=args.half_window, dry_run=args.audit or args.dry_run,
                )
                station_id = str(station["id"])
                if result["status"] == "SUCCESS":
                    path = save_hourly_sample(result, station_id, Path(args.output_dir))
                    success += 1
                    print(f"OK {station_id} {valid_time.isoformat()} {platform.name} scans={result['scan_count']} -> {path}")
                elif result["status"] == "DRY_RUN":
                    print(f"AUDIT {station_id} {valid_time.isoformat()} {platform.name} bundles={result['bundle_count']} scans={result['scans']}")
                else:
                    failure += 1
                    print(f"FAIL {station_id} {valid_time.isoformat()} {platform.name} {result['status']} {result.get('errors', '')}")
    print(f"complete: success={success}, failure={failure}, audit={args.audit or args.dry_run}")
    if failure:
        sys.exit(2)


if __name__ == "__main__":
    main()
