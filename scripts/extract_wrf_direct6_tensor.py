#!/usr/bin/env python3
"""Extract one clean WRF-only tensor from a frozen direct-6 case.

Each WRF file contains the issue time and six future hourly endpoints.  The
output follows ``model_1_6h_hierarchical_v2``: six future steps, six 3-D
variables on ten pressure levels over a 5x5 station window, four surface
variables, and the hourly-mean WRF GHI anchor derived from ACSWDNB endpoint
differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, chartostring


PRESSURE_HPA = np.asarray(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200], dtype=np.float32
)
THREE_D_NAMES = (
    "temperature_k",
    "specific_humidity_kgkg",
    "cloud_water_kgkg",
    "cloud_ice_kgkg",
    "cloud_fraction",
    "geopotential_height_m",
)
SURFACE_NAMES = ("t2_k", "psfc_pa", "albedo", "wrf_ghi_wm2")
G = np.float32(9.80665)
KAPPA = np.float32(287.05 / 1004.0)


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def decode_times(dataset: Dataset) -> list[datetime]:
    values = chartostring(dataset.variables["Times"][:])
    return [
        datetime.strptime(str(value).strip(), "%Y-%m-%d_%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        for value in values
    ]


def station_window(
    latitude: np.ndarray,
    longitude: np.ndarray,
    target_latitude: float,
    target_longitude: float,
) -> tuple[slice, slice, int, int]:
    distance = (latitude - target_latitude) ** 2 + (
        (longitude - target_longitude) * np.cos(np.deg2rad(target_latitude))
    ) ** 2
    center_y, center_x = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
    y0, x0 = center_y - 2, center_x - 2
    if y0 < 0 or x0 < 0 or y0 + 5 > latitude.shape[0] or x0 + 5 > latitude.shape[1]:
        raise ValueError("station is too close to the d02 boundary for a 5x5 window")
    return slice(y0, y0 + 5), slice(x0, x0 + 5), center_y, center_x


def read_float(variable: object, selection: tuple[object, ...]) -> np.ndarray:
    return np.asarray(np.ma.filled(variable[selection], np.nan), dtype=np.float32)


def interpolate_log_pressure(
    values: np.ndarray,
    pressure_pa: np.ndarray,
    surface_pressure_pa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape != pressure_pa.shape:
        raise ValueError("pressure and 3-D source fields must have identical shapes")
    target_pa = PRESSURE_HPA * np.float32(100.0)
    output = np.full(
        (values.shape[0], target_pa.size, values.shape[2], values.shape[3]),
        np.nan,
        dtype=np.float32,
    )
    valid_output = np.zeros(output.shape, dtype=bool)
    for target_index, target in enumerate(target_pa):
        lower_index = np.sum(pressure_pa >= target, axis=1) - 1
        valid = (
            (lower_index >= 0)
            & (lower_index < pressure_pa.shape[1] - 1)
            & (target <= surface_pressure_pa)
        )
        clipped = np.clip(lower_index, 0, pressure_pa.shape[1] - 2)
        selector0 = clipped[:, None, :, :]
        selector1 = (clipped + 1)[:, None, :, :]
        p0 = np.take_along_axis(pressure_pa, selector0, axis=1)[:, 0]
        p1 = np.take_along_axis(pressure_pa, selector1, axis=1)[:, 0]
        v0 = np.take_along_axis(values, selector0, axis=1)[:, 0]
        v1 = np.take_along_axis(values, selector1, axis=1)[:, 0]
        denominator = np.log(p1) - np.log(p0)
        weight = np.divide(
            np.log(target) - np.log(p0),
            denominator,
            out=np.zeros_like(p0),
            where=np.abs(denominator) > 1.0e-12,
        )
        interpolated = v0 + weight * (v1 - v0)
        valid &= np.isfinite(interpolated)
        output[:, target_index] = np.where(valid, interpolated, np.nan)
        valid_output[:, target_index] = valid
    return output, valid_output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output(path: Path, expected_sample_id: str | None = None) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:
        sample_id = str(payload["sample_id"])
        if expected_sample_id is not None and sample_id != expected_sample_id:
            raise ValueError(f"sample_id={sample_id!r}, expected={expected_sample_id!r}")
        three_d = np.asarray(payload["nwp_3d"], dtype=np.float32)
        valid = np.asarray(payload["nwp_3d_valid"], dtype=bool)
        surface = np.asarray(payload["nwp_surface"], dtype=np.float32)
        anchor = np.asarray(payload["wrf_anchor"], dtype=np.float32)
        if three_d.shape != (6, 6, 10, 5, 5):
            raise ValueError(f"nwp_3d shape={three_d.shape}")
        if valid.shape != (6, 10, 5, 5):
            raise ValueError(f"nwp_3d_valid shape={valid.shape}")
        if surface.shape != (6, 4, 5, 5):
            raise ValueError(f"nwp_surface shape={surface.shape}")
        if anchor.shape != (6,):
            raise ValueError(f"wrf_anchor shape={anchor.shape}")
        valid_all_channels = np.broadcast_to(valid[:, None, :, :, :], three_d.shape)
        if not np.isfinite(three_d[valid_all_channels]).all():
            raise ValueError("valid 3-D locations contain non-finite values")
        if not np.isfinite(surface).all() or not np.isfinite(anchor).all():
            raise ValueError("surface tensor or WRF anchor contains non-finite values")
        if np.nanmin(three_d[:, 0]) < 150.0 or np.nanmax(three_d[:, 0]) > 360.0:
            raise ValueError("pressure-level temperature is outside physical limits")
        if np.min(surface[:, 0]) < 150.0 or np.max(surface[:, 0]) > 360.0:
            raise ValueError("T2 is outside physical limits")
        if np.min(surface[:, 1]) < 30000.0 or np.max(surface[:, 1]) > 110000.0:
            raise ValueError("PSFC is outside physical limits")
        if np.min(surface[:, 2]) < 0.0 or np.max(surface[:, 2]) > 1.0:
            raise ValueError("ALBEDO is outside [0, 1]")
        if np.min(anchor) < -0.05 or np.max(anchor) > 1600.0:
            raise ValueError("hourly WRF GHI anchor is outside physical limits")
        if not np.allclose(anchor, surface[:, 3, 2, 2], rtol=0.0, atol=1.0e-5):
            raise ValueError("WRF anchor does not equal the central surface GHI channel")
        return {
            "sample_id": sample_id,
            "valid_fraction": float(valid.mean()),
            "anchor_min_wm2": float(anchor.min()),
            "anchor_max_wm2": float(anchor.max()),
            "bytes": path.stat().st_size,
        }


def extract_case(row: dict[str, str], runs_root: Path, output_root: Path) -> dict[str, object]:
    sample_id = row["sample_id"]
    output = output_root / "cases" / f"{sample_id}.npz"
    if output.exists():
        try:
            result = validate_output(output, sample_id)
            return {**result, "status": "RESUMED", "output": str(output)}
        except Exception:
            output.unlink()

    wrf_files = sorted((runs_root / sample_id / "direct6" / "wrf").glob("wrfout_d02_*"))
    if len(wrf_files) != 1:
        raise ValueError(f"{sample_id}: expected one d02 file, found {len(wrf_files)}")
    wrf_file = wrf_files[0]
    issue_time = parse_utc(row["issue_time_utc"])
    expected_times = [issue_time + timedelta(hours=hour) for hour in range(7)]

    required = (
        "P", "PB", "T", "QVAPOR", "QCLOUD", "QICE", "CLDFRA", "PH", "PHB",
        "T2", "PSFC", "ALBEDO", "ACSWDNB", "XLAT", "XLONG",
    )
    with Dataset(wrf_file) as dataset:
        absent = [name for name in required if name not in dataset.variables]
        if absent:
            raise KeyError(f"{sample_id}: missing WRF variables {absent}")
        actual_times = decode_times(dataset)
        if actual_times != expected_times:
            raise ValueError(
                f"{sample_id}: WRF times do not equal issue..issue+6; "
                f"actual={[item.isoformat() for item in actual_times]}"
            )
        latitude_full = read_float(dataset.variables["XLAT"], (0, slice(None), slice(None)))
        longitude_full = read_float(dataset.variables["XLONG"], (0, slice(None), slice(None)))
        ys, xs, center_y, center_x = station_window(
            latitude_full, longitude_full, float(row["latitude"]), float(row["longitude"])
        )
        future = slice(1, 7)
        p = read_float(dataset.variables["P"], (future, slice(None), ys, xs))
        pb = read_float(dataset.variables["PB"], (future, slice(None), ys, xs))
        pressure = p + pb
        if not np.isfinite(pressure).all() or np.any(np.diff(pressure, axis=1) >= 0.0):
            raise ValueError(f"{sample_id}: mass-level pressure is invalid")
        theta = read_float(dataset.variables["T"], (future, slice(None), ys, xs)) + 300.0
        qvapor = read_float(dataset.variables["QVAPOR"], (future, slice(None), ys, xs))
        ph = read_float(dataset.variables["PH"], (future, slice(None), ys, xs))
        phb = read_float(dataset.variables["PHB"], (future, slice(None), ys, xs))
        source_fields = (
            theta * np.power(pressure / 100000.0, KAPPA),
            qvapor / (1.0 + qvapor),
            read_float(dataset.variables["QCLOUD"], (future, slice(None), ys, xs)),
            read_float(dataset.variables["QICE"], (future, slice(None), ys, xs)),
            read_float(dataset.variables["CLDFRA"], (future, slice(None), ys, xs)),
            0.5 * ((ph + phb)[:, :-1] + (ph + phb)[:, 1:]) / G,
        )
        psfc = read_float(dataset.variables["PSFC"], (future, ys, xs))
        interpolated = []
        shared_valid = None
        for values in source_fields:
            field, field_valid = interpolate_log_pressure(values, pressure, psfc)
            interpolated.append(field)
            shared_valid = field_valid if shared_valid is None else shared_valid & field_valid
        assert shared_valid is not None
        three_d = np.stack(
            [np.where(shared_valid, field, np.nan) for field in interpolated], axis=1
        ).astype(np.float32)

        accumulated = read_float(dataset.variables["ACSWDNB"], (slice(0, 7), ys, xs))
        hourly_ghi = np.diff(accumulated, axis=0) / np.float32(3600.0)
        if float(hourly_ghi.min()) < -0.05:
            raise ValueError(f"{sample_id}: ACSWDNB decreases between hourly endpoints")
        hourly_ghi = np.maximum(hourly_ghi, 0.0).astype(np.float32)
        surface = np.stack(
            [
                read_float(dataset.variables["T2"], (future, ys, xs)),
                psfc,
                read_float(dataset.variables["ALBEDO"], (future, ys, xs)),
                hourly_ghi,
            ],
            axis=1,
        ).astype(np.float32)
        latitude = latitude_full[ys, xs].astype(np.float32)
        longitude = longitude_full[ys, xs].astype(np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.part")
    metadata = {
        "schema_version": "wrf_direct6_tensor_v2",
        "sample_id": sample_id,
        "partition": row.get("partition", row.get("split", "")),
        "legacy_manifest_split": row.get("split", ""),
        "station_id": row["station_id"],
        "wrf_init_time_utc": row["wrf_init_time_utc"],
        "issue_time_utc": row["issue_time_utc"],
        "issue_offset_hours": int(row["issue_offset_hours"]),
        "source_wrfout": str(wrf_file),
        "source_time_count": 7,
        "future_time_count": 6,
        "window": [5, 5],
        "station_window_index": [2, 2],
        "station_grid_latitude": float(latitude[2, 2]),
        "station_grid_longitude": float(longitude[2, 2]),
        "wrf_anchor_definition": "diff(ACSWDNB endpoints)/3600",
    }
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            sample_id=np.asarray(sample_id),
            partition=np.asarray(row.get("partition", row.get("split", ""))),
            legacy_manifest_split=np.asarray(row.get("split", "")),
            station_id=np.asarray(row["station_id"]),
            wrf_init_time_utc=np.asarray(row["wrf_init_time_utc"]),
            issue_time_utc=np.asarray(row["issue_time_utc"]),
            issue_offset_hours=np.asarray(int(row["issue_offset_hours"]), dtype=np.int16),
            valid_time_utc=np.asarray(
                [item.strftime("%Y-%m-%dT%H:%M:%SZ") for item in expected_times[1:]]
            ),
            pressure_hpa=PRESSURE_HPA,
            nwp_3d_names=np.asarray(THREE_D_NAMES),
            nwp_surface_names=np.asarray(SURFACE_NAMES),
            nwp_3d=three_d,
            nwp_3d_valid=shared_valid,
            nwp_surface=surface,
            wrf_anchor=surface[:, 3, 2, 2],
            latitude=latitude,
            longitude=longitude,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
    validation = validate_output(temporary, sample_id)
    temporary.replace(output)
    report = {
        **metadata,
        **validation,
        "status": "SUCCESS",
        "output": str(output),
        "sha256": sha256(output),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True, help="zero-based manifest row")
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not 0 <= args.case_index < len(rows):
        raise IndexError(f"case-index {args.case_index} outside 0..{len(rows) - 1}")
    result = extract_case(rows[args.case_index], args.runs_root, args.output_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
