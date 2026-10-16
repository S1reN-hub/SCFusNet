#!/usr/bin/env python3
"""Extract one real WRF domain into the frozen v4 wind/cloud data contract."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, chartostring


PRESSURE_HPA = np.asarray([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200], np.float32)
VOLUME_NAMES = np.asarray([
    "air_temperature", "specific_humidity", "cloud_liquid_water", "rain_water",
    "cloud_ice", "snow", "graupel", "cloud_fraction", "geopotential_height",
    "eastward_wind", "northward_wind", "vertical_velocity",
])
SURFACE_NAMES = np.asarray([
    "T2", "Q2", "PSFC", "TSK", "U10", "V10", "PBLH", "HFX", "LH",
    "ALBEDO", "SWDOWN", "hourly_precipitation", "OLR", "CAPE", "CIN",
])
STATIC_NAMES = np.asarray([
    "HGT", "LANDMASK", "LU_INDEX", "slope_radian", "aspect_sin",
    "aspect_cos", "terrain_std_3x3_m", "terrain_std_9x9_m",
])
CAPE_CIN_ALGORITHM = "wrf_python_cape_2d_rip_max_theta_e_0_3km_500m_parcel_ter_follow_v1"


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_variable(dataset: Dataset, name: str, selection=...) -> np.ndarray:
    return np.asarray(dataset.variables[name][selection], dtype=np.float32)


def center_window(latitude: np.ndarray, longitude: np.ndarray, target_lat: float, target_lon: float) -> tuple[slice, slice, int, int]:
    distance = (latitude - target_lat) ** 2 + ((longitude - target_lon) * np.cos(np.deg2rad(target_lat))) ** 2
    center_y, center_x = np.unravel_index(np.nanargmin(distance), distance.shape)
    y0, x0 = int(center_y) - 16, int(center_x) - 16
    if y0 < 0 or x0 < 0 or y0 + 32 > latitude.shape[0] or x0 + 32 > latitude.shape[1]:
        raise ValueError("32x32 station window crosses the WRF domain boundary")
    return slice(y0, y0 + 32), slice(x0, x0 + 32), int(center_y), int(center_x)


def moving_std(field: np.ndarray, size: int) -> np.ndarray:
    half = size // 2
    padded = np.pad(field, half, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return windows.std(axis=(-2, -1), dtype=np.float64).astype(np.float32)


def interpolate_columns(pressure_pa: np.ndarray, fields: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    times, _, ny, nx = pressure_pa.shape
    outputs = [np.zeros((times, len(PRESSURE_HPA), ny, nx), np.float32) for _ in fields]
    valid = np.zeros((times, len(PRESSURE_HPA), ny, nx), bool)
    targets_pa = PRESSURE_HPA.astype(np.float64) * 100.0
    for time_index in range(times):
        for y_index in range(ny):
            for x_index in range(nx):
                pressure = pressure_pa[time_index, :, y_index, x_index].astype(np.float64)
                finite_pressure = np.isfinite(pressure)
                if finite_pressure.sum() < 2:
                    continue
                pressure = pressure[finite_pressure]
                order = np.argsort(pressure)
                pressure = pressure[order]
                unique_pressure, unique_index = np.unique(pressure, return_index=True)
                if unique_pressure.size < 2:
                    continue
                inside = (targets_pa >= unique_pressure[0]) & (targets_pa <= unique_pressure[-1])
                valid[time_index, :, y_index, x_index] = inside
                for field_index, field in enumerate(fields):
                    values = field[time_index, :, y_index, x_index][finite_pressure].astype(np.float64)[order][unique_index]
                    finite = np.isfinite(values)
                    if finite.sum() < 2:
                        valid[time_index, :, y_index, x_index] = False
                        continue
                    interpolated = np.interp(targets_pa, unique_pressure[finite], values[finite])
                    outputs[field_index][time_index, :, y_index, x_index] = interpolated.astype(np.float32)
    return outputs, valid


def extract_cape_cin(
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    qvapor: np.ndarray,
    height_m: np.ndarray,
    terrain_m: np.ndarray,
    psfc_pa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    zeros = np.zeros((pressure_pa.shape[0], *pressure_pa.shape[-2:]), np.float32)
    try:
        import wrf
    except Exception:
        return zeros.copy(), zeros.copy(), False, "unavailable"
    try:
        cape, cin = zeros.copy(), zeros.copy()
        for time_index in range(pressure_pa.shape[0]):
            diagnosed_raw = wrf.cape_2d(
                    pressure_pa[time_index] / 100.0,
                    temperature_k[time_index],
                    qvapor[time_index],
                    height_m[time_index],
                    terrain_m,
                    psfc_pa[time_index] / 100.0,
                    ter_follow=True,
                    meta=False,
                )
            diagnosed_masked = np.ma.asarray(diagnosed_raw)
            diagnosed = np.asarray(diagnosed_masked.filled(np.nan), dtype=np.float32)
            cape[time_index] = diagnosed[0]
            cin[time_index] = diagnosed[1]
        available = bool(
            np.all(np.isfinite(cape))
            and np.all(np.isfinite(cin))
            and np.all((cape >= -1e-3) & (cape <= 20000.0))
            and np.all(np.abs(cin) <= 10000.0)
        )
        if not available:
            cape.fill(0.0)
            cin.fill(0.0)
        return cape, cin, available, str(wrf.__version__)
    except Exception as exc:
        return zeros.copy(), zeros.copy(), False, f"failed:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrfout", type=Path, required=True)
    parser.add_argument("--domain", choices=("d01", "d02"), required=True)
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    required_volume = [
        "T", "P", "PB", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW",
        "QGRAUP", "CLDFRA", "PH", "PHB", "U", "V", "W", "COSALPHA", "SINALPHA",
    ]
    required_surface = [
        "T2", "Q2", "PSFC", "TSK", "U10", "V10", "PBLH", "HFX", "LH",
        "ALBEDO", "SWDOWN", "RAINC", "RAINNC", "OLR", "ACSWDNB",
    ]
    with Dataset(args.wrfout) as dataset:
        times = chartostring(dataset.variables["Times"][:]).astype(str)
        if times.shape != (7,):
            raise ValueError(f"v4 requires exactly seven WRF states, found {times.shape}")
        parsed_times = [datetime.strptime(value, "%Y-%m-%d_%H:%M:%S") for value in times]
        if any((right - left).total_seconds() != 3600 for left, right in zip(parsed_times, parsed_times[1:])):
            raise ValueError("WRF states are not consecutive hourly endpoints")
        missing = [name for name in required_volume + required_surface if name not in dataset.variables]
        if missing:
            raise KeyError(f"WRF output is missing required variables: {missing}")

        latitude = read_variable(dataset, "XLAT", 0)
        longitude = read_variable(dataset, "XLONG", 0)
        ys, xs, center_y, center_x = center_window(latitude, longitude, args.latitude, args.longitude)
        cosalpha = read_variable(dataset, "COSALPHA")[:, ys, xs]
        sinalpha = read_variable(dataset, "SINALPHA")[:, ys, xs]

        pressure = (read_variable(dataset, "P")[:, :, ys, xs] + read_variable(dataset, "PB")[:, :, ys, xs])
        theta = read_variable(dataset, "T")[:, :, ys, xs] + 300.0
        temperature = theta * np.power(pressure / 100000.0, 0.2854)
        qvapor = read_variable(dataset, "QVAPOR")[:, :, ys, xs]
        specific_humidity = qvapor / (1.0 + qvapor)
        height_stag = (read_variable(dataset, "PH")[:, :, ys, xs] + read_variable(dataset, "PHB")[:, :, ys, xs]) / 9.80665
        height = 0.5 * (height_stag[:, :-1] + height_stag[:, 1:])
        u_stag = read_variable(dataset, "U")[:, :, ys, slice(xs.start, xs.stop + 1)]
        v_stag = read_variable(dataset, "V")[:, :, slice(ys.start, ys.stop + 1), xs]
        w_stag = read_variable(dataset, "W")[:, :, ys, xs]
        u_grid = 0.5 * (u_stag[:, :, :, :-1] + u_stag[:, :, :, 1:])
        v_grid = 0.5 * (v_stag[:, :, :-1, :] + v_stag[:, :, 1:, :])
        vertical_velocity = 0.5 * (w_stag[:, :-1] + w_stag[:, 1:])
        eastward = u_grid * cosalpha[:, None] + v_grid * sinalpha[:, None]
        northward = v_grid * cosalpha[:, None] - u_grid * sinalpha[:, None]

        native_fields = [temperature, specific_humidity]
        channel_available = np.ones(12, bool)
        clipped_negative_counts: dict[str, int] = {}
        for variable_name in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
            values = read_variable(dataset, variable_name)[:, :, ys, xs]
            clipped_negative_counts[variable_name] = int(np.sum(values < 0.0))
            values = np.maximum(values, 0.0)
            native_fields.append(values)
        native_fields.extend([
            np.clip(read_variable(dataset, "CLDFRA")[:, :, ys, xs], 0.0, 1.0),
            height, eastward, northward, vertical_velocity,
        ])
        interpolated, level_valid = interpolate_columns(pressure, native_fields)
        volume = np.stack(interpolated, axis=1).astype(np.float32)
        volume = np.where(level_valid[:, None], volume, 0.0).astype(np.float32)

        surface = np.zeros((7, 15, 32, 32), np.float32)
        surface_available = np.ones(15, bool)
        for index, variable_name in enumerate((
            "T2", "Q2", "PSFC", "TSK", "U10", "V10", "PBLH", "HFX", "LH",
            "ALBEDO", "SWDOWN",
        )):
            surface[:, index] = read_variable(dataset, variable_name)[:, ys, xs]
        u10_grid, v10_grid = surface[:, 4].copy(), surface[:, 5].copy()
        surface[:, 4] = u10_grid * cosalpha + v10_grid * sinalpha
        surface[:, 5] = v10_grid * cosalpha - u10_grid * sinalpha
        accumulated_precip = (
            read_variable(dataset, "RAINC")[:, ys, xs] + read_variable(dataset, "RAINNC")[:, ys, xs]
        )
        precip_difference = np.diff(accumulated_precip, axis=0, prepend=accumulated_precip[:1])
        if float(np.nanmin(precip_difference)) < -1e-3:
            raise ValueError("accumulated precipitation has a material negative increment")
        surface[:, 11] = np.maximum(precip_difference, 0.0)
        surface[:, 12] = read_variable(dataset, "OLR")[:, ys, xs]

        hgt_full = read_variable(dataset, "HGT", 0)
        hgt = hgt_full[ys, xs]
        landmask = read_variable(dataset, "LANDMASK", 0)[ys, xs]
        landuse = read_variable(dataset, "LU_INDEX", 0)[ys, xs]
        if not np.all(np.abs(landuse - np.rint(landuse)) < 1e-5):
            raise ValueError("LU_INDEX is not categorical integer data")
        dx = float(getattr(dataset, "DX"))
        dy = float(getattr(dataset, "DY"))
        dz_dy, dz_dx = np.gradient(hgt_full.astype(np.float64), dy, dx)
        slope = np.arctan(np.hypot(dz_dx, dz_dy)).astype(np.float32)
        aspect = np.arctan2(-dz_dx, -dz_dy).astype(np.float32)
        static = np.stack([
            hgt, landmask, landuse, slope[ys, xs], np.sin(aspect[ys, xs]),
            np.cos(aspect[ys, xs]), moving_std(hgt_full, 3)[ys, xs],
            moving_std(hgt_full, 9)[ys, xs],
        ]).astype(np.float32)

        cape, cin, cape_available, wrf_python_version = extract_cape_cin(
            pressure, temperature, qvapor, height, hgt, surface[:, 2]
        )
        surface[:, 13], surface[:, 14] = cape, cin
        surface_available[13:15] = cape_available

        anchor = np.diff(read_variable(dataset, "ACSWDNB")[:, ys, xs], axis=0) / 3600.0
        anchor_min_before_clip = float(np.nanmin(anchor))
        if anchor_min_before_clip < -1e-3:
            raise ValueError("d02 ACSWDNB has a material negative hourly endpoint difference")
        anchor = np.maximum(anchor, 0.0).astype(np.float32)

        metadata = {
            "schema_version": "wrf_windcloud_v4_domain_v1",
            "sample_id": args.sample_id,
            "station_id": args.station_id,
            "domain": args.domain,
            "wrfout": str(args.wrfout),
            "times": times.tolist(),
            "station_target_latitude": args.latitude,
            "station_target_longitude": args.longitude,
            "nearest_grid_y": center_y,
            "nearest_grid_x": center_x,
            "nearest_grid_latitude": float(latitude[center_y, center_x]),
            "nearest_grid_longitude": float(longitude[center_y, center_x]),
            "window_y": [ys.start, ys.stop],
            "window_x": [xs.start, xs.stop],
            "dx_m": dx,
            "dy_m": dy,
            "pressure_interpolation": "linear_in_pressure",
            "wind_rotation": "u_e=u_grid*cosalpha+v_grid*sinalpha; v_n=v_grid*cosalpha-u_grid*sinalpha",
            "surface_wind_rotation": "same COSALPHA/SINALPHA earth-relative rotation as 3-D wind",
            "hydrometeor_negative_handling": "clip physical mixing ratios below zero to zero and record counts",
            "hydrometeor_negative_counts": clipped_negative_counts,
            "terrain_slope_unit": "radian",
            "terrain_aspect": "downslope azimuth clockwise from north, stored as sine/cosine",
            "cape_cin_available": cape_available,
            "wrf_python_version": wrf_python_version,
            "anchor_min_before_clip_wm2": anchor_min_before_clip,
        }

    arrays = {
        "sample_id": np.asarray(args.sample_id),
        "station_id": np.asarray(args.station_id),
        f"nwp_{args.domain}_3d_raw": volume,
        f"nwp_{args.domain}_level_valid": level_valid,
        f"nwp_{args.domain}_channel_available": channel_available,
        f"nwp_{args.domain}_surface_raw": surface,
        f"nwp_{args.domain}_surface_available": surface_available,
        f"nwp_{args.domain}_static_raw": static,
        "nwp_3d_names": VOLUME_NAMES,
        "nwp_surface_names": SURFACE_NAMES,
        "nwp_static_names": STATIC_NAMES,
        "pressure_hpa": PRESSURE_HPA,
        "cape_cin_algorithm": np.asarray(CAPE_CIN_ALGORITHM),
        "wrf_python_version": np.asarray(wrf_python_version),
        "wrf_extraction_metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    if args.domain == "d02":
        arrays["wrf_ghi_anchor_d02"] = anchor
        arrays["wrf_ghi_anchor_source"] = np.asarray("d02_ACSWDNB_endpoint_difference_3600s")
    atomic_npz(args.output, **arrays)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
