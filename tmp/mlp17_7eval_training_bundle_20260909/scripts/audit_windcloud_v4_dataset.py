#!/usr/bin/env python3
"""Audit the v4 dual-domain wind/cloud tensor contract without opening test labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


VOLUME_NAMES = np.asarray([
    "air_temperature", "specific_humidity", "cloud_liquid_water", "rain_water",
    "cloud_ice", "snow", "graupel", "cloud_fraction", "geopotential_height",
    "eastward_wind", "northward_wind", "vertical_velocity",
])
SURFACE_NAMES = np.asarray([
    "T2", "Q2", "PSFC", "TSK", "U10", "V10", "PBLH", "HFX", "LH",
    "ALBEDO", "SWDOWN", "hourly_precipitation", "OLR", "CAPE", "CIN",
])
PRESSURE = np.asarray([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200])
CAPE_CIN_ALGORITHM = "wrf_python_cape_2d_rip_max_theta_e_0_3km_500m_parcel_ter_follow_v1"
ANCHOR_SOURCE = "d02_ACSWDNB_endpoint_difference_3600s"


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def parse_utc(value: object) -> datetime:
    text = str(np.asarray(value).item()).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cycle_key(row: dict[str, str]) -> str:
    for name in ("cycle", "init_time_utc", "wrf_init_time_utc"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def data_role(row: dict[str, str]) -> str:
    for name in ("role", "current_role", "data_role", "dataset_role", "partition"):
        value = str(row.get(name, "")).strip().lower()
        if value:
            return value
    return ""


def audit_case(path: Path, inspect_labels: bool) -> tuple[list[str], dict]:
    failures: list[str] = []
    with np.load(str(path), allow_pickle=False) as loaded:
        case = {name: np.asarray(loaded[name]) for name in loaded.files}
    required_common = [
        "sample_id", "station_id", "satellite", "satellite_valid", "satellite_frame_mask",
        "satellite_scan_time_offset_hours", "satellite_unique_frame_mask", "satellite_source_id",
        "wrf_ghi_anchor_d02", "wrf_ghi_anchor_source", "cape_cin_algorithm",
        "wrf_init_time_utc", "issue_offset_hours", "issue_time_utc", "valid_time_utc",
        "solar", "subhour_cos_sza", "aod550", "aod_mask",
    ]
    for name in required_common:
        require(name in case, f"missing {name}", failures)
    for domain in ("d01", "d02"):
        required = [
            f"nwp_{domain}_3d_raw", f"nwp_{domain}_level_valid", f"nwp_{domain}_channel_available",
            f"nwp_{domain}_surface_raw", f"nwp_{domain}_surface_available", f"nwp_{domain}_static_raw",
        ]
        for name in required:
            require(name in case, f"missing {name}", failures)
        if all(name in case for name in required):
            volume = case[f"nwp_{domain}_3d_raw"]
            valid = case[f"nwp_{domain}_level_valid"]
            surface = case[f"nwp_{domain}_surface_raw"]
            volume_available = case[f"nwp_{domain}_channel_available"]
            surface_available = case[f"nwp_{domain}_surface_available"]
            require(volume.shape == (7, 12, 10, 32, 32), f"{domain} 3d shape {volume.shape}", failures)
            require(valid.shape == (7, 10, 32, 32), f"{domain} valid shape {valid.shape}", failures)
            require(surface.shape == (7, 15, 32, 32), f"{domain} surface shape {surface.shape}", failures)
            require(volume_available.shape == (12,), f"{domain} channel availability shape {volume_available.shape}", failures)
            require(surface_available.shape == (15,), f"{domain} surface availability shape {surface_available.shape}", failures)
            require(case[f"nwp_{domain}_static_raw"].shape == (8, 32, 32), f"{domain} static shape", failures)
            static = case[f"nwp_{domain}_static_raw"]
            if static.shape == (8, 32, 32):
                landuse = static[2]
                require(bool(np.all(np.isfinite(landuse))), f"{domain} LU_INDEX non-finite", failures)
                require(bool(np.all(np.abs(landuse - np.rint(landuse)) < 1e-5)), f"{domain} LU_INDEX is not categorical integer", failures)
                require(float(np.nanmin(landuse)) >= 0.0 and float(np.nanmax(landuse)) < 64.0, f"{domain} LU_INDEX outside embedding range", failures)
            if volume.shape == (7, 12, 10, 32, 32):
                finite_wind = np.isfinite(volume[:, 9:12])
                require(bool(finite_wind.any()), f"{domain} has no finite U/V/W", failures)
                if finite_wind.any():
                    require(float(np.nanmax(np.abs(volume[:, 9:11]))) < 150.0, f"{domain} horizontal wind implausible", failures)
                    require(float(np.nanmax(np.abs(volume[:, 11]))) < 80.0, f"{domain} vertical velocity implausible", failures)
                cloud = volume[:, 7]
                require(float(np.nanmin(cloud)) >= -1e-4 and float(np.nanmax(cloud)) <= 1.0001, f"{domain} cloud fraction outside [0,1]", failures)
            if surface.shape == (7, 15, 32, 32) and surface_available.shape == (15,):
                cape_cin_available = surface_available[13:15].astype(bool)
                require(bool(cape_cin_available[0] == cape_cin_available[1]), f"{domain} CAPE/CIN availability must change together", failures)
                cape_cin = surface[:, 13:15]
                if cape_cin_available.all():
                    require(bool(np.all(np.isfinite(cape_cin))), f"{domain} CAPE/CIN marked available but contains missing values", failures)
                    require(float(np.nanmin(cape_cin[:, 0])) >= -1e-3, f"{domain} MCAPE is negative", failures)
                    require(float(np.nanmax(cape_cin[:, 0])) <= 20000.0, f"{domain} MCAPE implausible", failures)
                    require(float(np.nanmax(np.abs(cape_cin[:, 1]))) <= 10000.0, f"{domain} MCIN implausible", failures)
                else:
                    require(bool(np.all(cape_cin == 0.0)), f"{domain} unavailable CAPE/CIN channels must use neutral zeros", failures)
    if "satellite" in case:
        require(case["satellite"].shape == (6, 7, 32, 32), f"satellite shape {case['satellite'].shape}", failures)
        if "satellite_valid" in case:
            require(case["satellite_valid"].shape == case["satellite"].shape, "satellite_valid shape mismatch", failures)
    if "satellite_scan_time_offset_hours" in case:
        offsets = case["satellite_scan_time_offset_hours"].astype(float)
        require(offsets.shape == (6,), "satellite scan offsets shape", failures)
        require(bool(np.all(offsets <= 1e-6)), "future satellite scan detected", failures)
        require(bool(np.all(np.diff(offsets) >= -1e-6)), "satellite scan offsets not monotonic", failures)
        if "satellite_unique_frame_mask" in case:
            unique = case["satellite_unique_frame_mask"].astype(bool)
            frame_mask = case["satellite_frame_mask"].astype(bool)
            require(unique.shape == (6,), "satellite unique frame mask shape", failures)
            require(frame_mask.shape == (6,), "satellite frame mask shape", failures)
            expected = np.zeros(6, bool)
            if "satellite_frame_id" in case:
                frame_id = case["satellite_frame_id"].astype(str)
                require(frame_id.shape == (6,), "satellite frame id shape", failures)
                seen: dict[str, float] = {}
                for index, identifier in enumerate(frame_id):
                    if not frame_mask[index]:
                        continue
                    if identifier not in seen:
                        expected[index] = True
                        seen[identifier] = float(offsets[index])
                    else:
                        require(abs(float(offsets[index]) - seen[identifier]) <= 1e-4, "same satellite frame id has different scan times", failures)
            else:
                seen_offsets: list[float] = []
                for index, offset in enumerate(offsets):
                    if frame_mask[index] and not any(abs(float(offset) - previous) <= 1e-4 for previous in seen_offsets):
                        expected[index] = True
                        seen_offsets.append(float(offset))
            require(bool(np.array_equal(unique, expected)), "unique frame mask disagrees with scan time/frame id", failures)
    if "wrf_ghi_anchor_d02" in case:
        require(case["wrf_ghi_anchor_d02"].shape == (6, 32, 32), f"anchor shape {case['wrf_ghi_anchor_d02'].shape}", failures)
        require(float(np.nanmin(case["wrf_ghi_anchor_d02"])) >= -1e-3, "negative d02 WRF GHI anchor", failures)
    if "wrf_ghi_anchor_source" in case:
        source = str(np.asarray(case["wrf_ghi_anchor_source"]).item())
        require(source == ANCHOR_SOURCE, f"anchor source must be {ANCHOR_SOURCE}", failures)
    if "cape_cin_algorithm" in case:
        algorithm = str(np.asarray(case["cape_cin_algorithm"]).item())
        require(algorithm == CAPE_CIN_ALGORITHM, f"CAPE/CIN algorithm must be {CAPE_CIN_ALGORITHM}", failures)
    if all(name in case for name in ("wrf_init_time_utc", "issue_offset_hours", "issue_time_utc", "valid_time_utc")):
        try:
            initialization = parse_utc(case["wrf_init_time_utc"])
            issue = parse_utc(case["issue_time_utc"])
            offset = float(np.asarray(case["issue_offset_hours"]).item())
            require(any(abs(offset - allowed) <= 1e-6 for allowed in range(3, 67, 3)), "issue offset is not one of 3,6,...,66 h", failures)
            require(abs((issue - initialization).total_seconds() / 3600.0 - offset) <= 1e-6, "issue time does not match WRF initialization plus offset", failures)
            valid_times = np.asarray(case["valid_time_utc"])
            require(valid_times.shape == (6,), f"valid_time_utc shape {valid_times.shape}", failures)
            if valid_times.shape == (6,):
                parsed_valid = [parse_utc(value) for value in valid_times]
                require(all(value == issue + timedelta(hours=lead) for lead, value in enumerate(parsed_valid, 1)), "valid times are not issue+1h through issue+6h", failures)
            if "sample_end_time_utc" in case:
                require(parse_utc(case["sample_end_time_utc"]) == issue + timedelta(hours=6), "sample end time is not issue+6h", failures)
        except (TypeError, ValueError, OverflowError) as error:
            failures.append(f"invalid traceability time metadata: {error}")
    if inspect_labels:
        require("target" in case and case["target"].shape == (6,), "target shape/missing", failures)
        require("target_mask" in case and case["target_mask"].shape == (6,), "target mask shape/missing", failures)
    result = {
        "sample_id": str(case.get("sample_id", path.stem).item()) if np.asarray(case.get("sample_id", path.stem)).ndim == 0 else path.stem,
        "station_id": str(case.get("station_id", "UNKNOWN").item()) if np.asarray(case.get("station_id", "UNKNOWN")).ndim == 0 else "UNKNOWN",
        "wrf_init_time_utc": str(case["wrf_init_time_utc"].item()) if "wrf_init_time_utc" in case and np.asarray(case["wrf_init_time_utc"]).ndim == 0 else "",
        "failures": failures,
    }
    return failures, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-labels", action="store_true", help="Use only on development partitions")
    args = parser.parse_args()
    with (args.dataset_root / "index.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records, all_failures = [], []
    stations, cycles, roles = Counter(), Counter(), Counter()
    cycle_roles: dict[str, set[str]] = {}
    for row in rows:
        path = args.dataset_root / row["case_file"]
        if not path.is_file():
            failures, record = (["case file missing"], {"sample_id": row.get("sample_id", path.stem), "station_id": row.get("station_id", "UNKNOWN"), "failures": ["case file missing"]})
        else:
            failures, record = audit_case(path, args.inspect_labels)
        records.append(record)
        if row.get("sample_id") and row["sample_id"] != record["sample_id"]:
            failures.append("sample_id differs between index and case")
        row_init = str(row.get("wrf_init_time_utc", row.get("init_time_utc", ""))).strip()
        if row_init and record.get("wrf_init_time_utc") and row_init != record["wrf_init_time_utc"]:
            failures.append("WRF initialization differs between index and case")
        role = data_role(row)
        key = cycle_key(row)
        if not key:
            failures.append("index row has no cycle/init_time_utc grouping key")
        if not role:
            failures.append("index row has no development/confirmation role")
        if args.inspect_labels and role.startswith("confirmation"):
            failures.append("confirmation row cannot be audited with labels")
        cycle_roles.setdefault(key, set()).add(role)
        all_failures.extend(f"{record['sample_id']}: {message}" for message in failures)
        stations[record["station_id"]] += 1
        cycles[key] += 1
        roles[role or "MISSING"] += 1
    for key, assigned in cycle_roles.items():
        effective = {role for role in assigned if role}
        if len(effective) > 1:
            all_failures.append(f"cycle {key}: multiple data roles {sorted(effective)}")
    report = {
        "schema_version": "windcloud_v4_dataset_audit_v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "case_count": len(rows), "station_counts": dict(sorted(stations.items())),
        "cycle_counts": dict(sorted(cycles.items())), "role_counts": dict(sorted(roles.items())),
        "labels_inspected": args.inspect_labels,
        "passed": not all_failures, "failure_count": len(all_failures),
        "failures": all_failures, "contracts": {
            "nwp_3d_names": VOLUME_NAMES.tolist(), "nwp_surface_names": SURFACE_NAMES.tolist(),
            "pressure_hpa": PRESSURE.tolist(), "wrf_ghi_anchor_source": ANCHOR_SOURCE,
            "cape_cin_algorithm": CAPE_CIN_ALGORITHM,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "passed", "failure_count", "labels_inspected")}, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
