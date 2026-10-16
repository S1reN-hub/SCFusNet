#!/usr/bin/env python3
"""Extract compact, common-mask GHI comparisons for the 10-cycle WRF study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, chartostring


ROOT = Path("/share/home/xiajingming/xjm_shaoyongjin/guizhou_wrf")
SEL = Path(os.environ.get("WRF_PBL_SEL", str(ROOT / "manifests/cycle_level_10case_draft_20260921/selected_cycle_station_expansion_draft.csv")))
MYNN_AUDIT = Path(os.environ.get("WRF_PBL_MYNN_AUDIT", str(ROOT / "manifests/cycle_level_10case_draft_20260921/pbl_compare_10case_audit.csv")))
PBL_AUDITS = {
    "ysu": Path(os.environ.get("WRF_PBL_YSU_AUDIT", str(ROOT / "manifests/cycle_level_10case_draft_20260921/pbl_compare_10case_ysu_audit.csv"))),
    "myj": Path(os.environ.get("WRF_PBL_MYJ_AUDIT", str(ROOT / "manifests/cycle_level_10case_draft_20260921/pbl_compare_10case_myj_audit.csv"))),
}
LABEL_FILES = {
    "expansion_p46": ROOT / "data/derived/p46_random6h_auxiliary_663/ground_radiation_p46_random6h_leads.npz",
    "formal_0_6": ROOT / "data/model_1_6h_v2/ghi_1_6h_core_799_v2/labels/ground_radiation_799_leads.npz",
}
SOURCE_MANIFESTS = {
    "data/expansion_p46_v4/wrf_random_6h_cases.csv": ROOT / "data/expansion_p46_v4/wrf_random_6h_cases.csv",
    "data/formal_0_6_v1/manifests/wrf_random_6h_cases.csv": ROOT / "data/formal_0_6_v1/manifests/wrf_random_6h_cases.csv",
}
RUNS_ROOT = Path(os.environ.get("WRF_PBL_RUNS_ROOT", str(ROOT / "runs/pbl_compare_10case_draft_20260921")))
OUT = Path(os.environ.get("WRF_PBL_OUT", str(ROOT / "results/wrf_pbl_10cycle_20260921")))
EXPECTED_CASE_COUNT = int(os.environ.get("WRF_PBL_CASE_COUNT", "70"))
RESULT_TAG = os.environ.get("WRF_PBL_RESULT_TAG", "10cycle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def solar_cosine(value: str, latitude: float, longitude: float) -> float:
    dt = parse_utc(value)
    day = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    true_solar_minutes = (dt.hour * 60.0 + dt.minute + equation_of_time + 4.0 * longitude) % 1440.0
    hour_angle = math.radians(true_solar_minutes / 4.0 - 180.0)
    latitude_rad = math.radians(latitude)
    return float(
        math.sin(latitude_rad) * math.sin(declination)
        + math.cos(latitude_rad) * math.cos(declination) * math.cos(hour_angle)
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(a)))


def decode_times(dataset: Dataset) -> list[str]:
    values = chartostring(dataset.variables["Times"][:])
    result = []
    for value in values:
        text = str(value).strip().replace("_", "T")
        if not text.endswith("Z"):
            text += "Z"
        result.append(text)
    return result


def nearest_grid(dataset: Dataset, station_lat: float, station_lon: float) -> dict[str, object]:
    latitude = np.asarray(np.ma.filled(dataset.variables["XLAT"][0], np.nan), dtype=np.float64)
    longitude = np.asarray(np.ma.filled(dataset.variables["XLONG"][0], np.nan), dtype=np.float64)
    distance = (latitude - station_lat) ** 2 + (
        (longitude - station_lon) * math.cos(math.radians(station_lat))
    ) ** 2
    grid_y, grid_x = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
    return {
        "grid_y": int(grid_y),
        "grid_x": int(grid_x),
        "grid_lat": float(latitude[grid_y, grid_x]),
        "grid_lon": float(longitude[grid_y, grid_x]),
        "shape": latitude.shape,
    }


def read_wrf(path: Path, station_lat: float, station_lon: float, mapping: dict[str, object] | None) -> dict[str, object]:
    with Dataset(path) as dataset:
        times = decode_times(dataset)
        if mapping is None:
            grid = nearest_grid(dataset, station_lat, station_lon)
        else:
            grid_y, grid_x = int(mapping["grid_y"]), int(mapping["grid_x"])
            latitude = np.asarray(np.ma.filled(dataset.variables["XLAT"][0], np.nan), dtype=np.float64)
            longitude = np.asarray(np.ma.filled(dataset.variables["XLONG"][0], np.nan), dtype=np.float64)
            if grid_y >= latitude.shape[0] or grid_x >= latitude.shape[1]:
                raise ValueError(f"{path}: common grid index outside shape {latitude.shape}")
            grid = {
                "grid_y": grid_y,
                "grid_x": grid_x,
                "grid_lat": float(latitude[grid_y, grid_x]),
                "grid_lon": float(longitude[grid_y, grid_x]),
                "shape": latitude.shape,
            }
        accumulated = np.asarray(
            np.ma.filled(dataset.variables["ACSWDNB"][:7, grid["grid_y"], grid["grid_x"]], np.nan),
            dtype=np.float64,
        )
        if accumulated.shape != (7,):
            raise ValueError(f"{path}: ACSWDNB shape {accumulated.shape}, expected 7")
        grid["times"] = times
        grid["pred"] = np.diff(accumulated) / 3600.0
        return grid


def metric(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    count = int(observed.size)
    if count == 0:
        return {
            "count": 0,
            "mean_observed_wm2": None,
            "rmse_wm2": None,
            "nrmse_pct": None,
            "mae_wm2": None,
            "mbe_wm2": None,
            "nmbe_pct": None,
            "r2": None,
        }
    error = predicted - observed
    mean_observed = float(np.mean(observed))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum((observed - mean_observed) ** 2))
    return {
        "count": count,
        "mean_observed_wm2": mean_observed,
        "rmse_wm2": rmse,
        "nrmse_pct": float(100.0 * rmse / mean_observed) if mean_observed else None,
        "mae_wm2": float(np.mean(np.abs(error))),
        "mbe_wm2": float(np.mean(error)),
        "nmbe_pct": float(100.0 * np.mean(error) / mean_observed) if mean_observed else None,
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else None,
    }


def find_wrf_file(case_root: Path) -> Path:
    files = sorted((case_root / "wrf").glob("wrfout_d02_*"))
    if len(files) != 1:
        raise ValueError(f"{case_root}: expected exactly one d02 wrfout, found {len(files)}")
    return files[0]


def write_metric_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with SEL.open(encoding="utf-8-sig", newline="") as stream:
        selected = list(csv.DictReader(stream))
    if len(selected) != EXPECTED_CASE_COUNT:
        raise ValueError(f"selected rows={len(selected)}, expected {EXPECTED_CASE_COUNT}")

    source_rows: dict[str, dict[str, str]] = {}
    for relative, path in SOURCE_MANIFESTS.items():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["sample_id"] in source_rows and source_rows[row["sample_id"]]["_source_manifest"] != relative:
                    raise ValueError(f"duplicate source sample_id {row['sample_id']}")
                source_rows[row["sample_id"]] = {**row, "_source_manifest": relative}

    labels: dict[str, dict[str, object]] = {}
    label_source: dict[str, str] = {}
    for label_name, path in LABEL_FILES.items():
        with np.load(path, allow_pickle=False) as loaded:
            for index, sample_id in enumerate(loaded["sample_id"].astype(str)):
                if sample_id in labels:
                    raise ValueError(f"duplicate label sample_id {sample_id}")
                labels[sample_id] = {
                    "station_id": str(loaded["station_id"][index]),
                    "issue_time_utc": str(loaded["issue_time_utc"][index]),
                    "valid_time_utc": np.asarray(loaded["valid_time_utc"][index]).astype(str).tolist(),
                    "target": np.asarray(loaded["target"][index, :, 0], dtype=np.float64),
                    "target_mask": np.asarray(loaded["target_mask"][index, :, 0], dtype=bool),
                }
                label_source[sample_id] = label_name

    with MYNN_AUDIT.open(encoding="utf-8-sig", newline="") as stream:
        mynn_audit = {row["sample_id"]: row for row in csv.DictReader(stream)}
    for scheme, audit_path in PBL_AUDITS.items():
        with audit_path.open(encoding="utf-8-sig", newline="") as stream:
            audit_rows = list(csv.DictReader(stream))
        failures = [row["sample_id"] for row in audit_rows if row.get("final_pass", "True").lower() == "false"]
        if failures:
            raise ValueError(f"{scheme} audit has failed cases: {failures[:5]}")

    model_roots: dict[str, dict[str, Path]] = {}
    for row in selected:
        sample_id = row["source_sample_id"]
        if sample_id not in source_rows or sample_id not in labels or sample_id not in mynn_audit:
            raise ValueError(f"missing source/label/audit mapping for {sample_id}")
        model_roots[sample_id] = {
            "mynn": Path(mynn_audit[sample_id]["case_root"]),
            "ysu": RUNS_ROOT / sample_id / "direct6_ysu",
            "myj": RUNS_ROOT / sample_id / "direct6_myj",
        }

    model_names = ("mynn", "ysu", "myj")
    point_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for case_index, row in enumerate(selected):
        sample_id = row["source_sample_id"]
        source = source_rows[sample_id]
        station_id = row["station_id"]
        station_lat = float(source["latitude"])
        station_lon = float(source["longitude"])
        label = labels[sample_id]
        if label["station_id"] != station_id or label["issue_time_utc"] != row["issue_time_utc"]:
            raise ValueError(f"{sample_id}: station or issue time mismatch")
        expected_times = label["valid_time_utc"]
        outputs: dict[str, dict[str, object]] = {}
        mapping = None
        for scheme in model_names:
            outputs[scheme] = read_wrf(find_wrf_file(model_roots[sample_id][scheme]), station_lat, station_lon, mapping)
            if mapping is None:
                mapping = outputs[scheme]
        if mapping is None:
            raise AssertionError("mapping was not initialized")
        for scheme in model_names:
            if outputs[scheme]["times"][1:7] != expected_times:
                raise ValueError(f"{sample_id}: {scheme} times do not match frozen labels")
        mapping_consistent = True
        for scheme in model_names:
            item = outputs[scheme]
            if item["shape"] != mapping["shape"] or item["grid_y"] != mapping["grid_y"] or item["grid_x"] != mapping["grid_x"]:
                mapping_consistent = False
            if abs(item["grid_lat"] - mapping["grid_lat"]) > 1e-5 or abs(item["grid_lon"] - mapping["grid_lon"]) > 1e-5:
                mapping_consistent = False
        distance_km = haversine_km(station_lat, station_lon, mapping["grid_lat"], mapping["grid_lon"])
        observed = np.asarray(label["target"], dtype=np.float64)
        observed_mask = np.asarray(label["target_mask"], dtype=bool) & np.isfinite(observed)
        cos_zenith = np.asarray([solar_cosine(value, station_lat, station_lon) for value in expected_times])
        daylight = cos_zenith > 0.0
        predictions = {scheme: np.asarray(outputs[scheme]["pred"], dtype=np.float64) for scheme in model_names}
        finite_all = np.ones(6, dtype=bool)
        for scheme in model_names:
            finite_all &= np.isfinite(predictions[scheme])
        common_valid = observed_mask & finite_all
        common_daylight = common_valid & daylight
        for lead in range(1, 7):
            index = lead - 1
            point_rows.append({
                "case_index": case_index,
                "case_id": row["case_id"],
                "sample_id": sample_id,
                "station_id": station_id,
                "cycle": row["cycle"],
                "issue_time_utc": row["issue_time_utc"],
                "valid_time_utc": expected_times[index],
                "lead_hour": lead,
                "station_latitude": f"{station_lat:.8f}",
                "station_longitude": f"{station_lon:.8f}",
                "grid_y": mapping["grid_y"],
                "grid_x": mapping["grid_x"],
                "grid_latitude": f"{mapping['grid_lat']:.8f}",
                "grid_longitude": f"{mapping['grid_lon']:.8f}",
                "station_grid_distance_km": f"{distance_km:.6f}",
                "mapping_consistent_all_schemes": mapping_consistent,
                "daylight_cos_zenith": f"{cos_zenith[index]:.8f}",
                "daylight": bool(daylight[index]),
                "label_valid": bool(observed_mask[index]),
                "mynn_valid": bool(np.isfinite(predictions["mynn"][index])),
                "ysu_valid": bool(np.isfinite(predictions["ysu"][index])),
                "myj_valid": bool(np.isfinite(predictions["myj"][index])),
                "common_valid": bool(common_valid[index]),
                "common_daylight_valid": bool(common_daylight[index]),
                "observed_ghi_wm2": f"{observed[index]:.8f}" if np.isfinite(observed[index]) else "",
                "mynn_ghi_wm2": f"{predictions['mynn'][index]:.8f}" if np.isfinite(predictions["mynn"][index]) else "",
                "ysu_ghi_wm2": f"{predictions['ysu'][index]:.8f}" if np.isfinite(predictions["ysu"][index]) else "",
                "myj_ghi_wm2": f"{predictions['myj'][index]:.8f}" if np.isfinite(predictions["myj"][index]) else "",
                "prediction_unit": "W m-2; diff(ACSWDNB J m-2)/3600 s",
                "label_source": label_source[sample_id],
            })
        case_rows.append({
            "case_index": case_index,
            "case_id": row["case_id"],
            "sample_id": sample_id,
            "station_id": station_id,
            "cycle": row["cycle"],
            "source_manifest": source["_source_manifest"],
            "mynn_source_tree": mynn_audit[sample_id]["source_tree"],
            "mynn_case_root": mynn_audit[sample_id]["case_root"],
            "label_source": label_source[sample_id],
            "grid_y": mapping["grid_y"],
            "grid_x": mapping["grid_x"],
            "station_latitude": station_lat,
            "station_longitude": station_lon,
            "grid_latitude": mapping["grid_lat"],
            "grid_longitude": mapping["grid_lon"],
            "station_grid_distance_km": distance_km,
            "mapping_consistent_all_schemes": mapping_consistent,
            "time_alignment_all_schemes": True,
            "label_valid_count": int(observed_mask.sum()),
            "common_valid_count": int(common_valid.sum()),
            "common_daylight_valid_count": int(common_daylight.sum()),
            "daylight_count": int(daylight.sum()),
        })
        print(f"processed {case_index + 1}/{len(selected)} {sample_id}", flush=True)

    prediction_path = OUT / f"predictions_{RESULT_TAG}_6lead.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    case_path = OUT / "case_alignment_audit.csv"
    with case_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)

    values = {
        "observed": np.asarray([float(row["observed_ghi_wm2"]) if row["observed_ghi_wm2"] else np.nan for row in point_rows]),
        "mynn": np.asarray([float(row["mynn_ghi_wm2"]) if row["mynn_ghi_wm2"] else np.nan for row in point_rows]),
        "ysu": np.asarray([float(row["ysu_ghi_wm2"]) if row["ysu_ghi_wm2"] else np.nan for row in point_rows]),
        "myj": np.asarray([float(row["myj_ghi_wm2"]) if row["myj_ghi_wm2"] else np.nan for row in point_rows]),
    }
    common_valid = np.asarray([row["common_valid"] for row in point_rows], dtype=bool)
    common_daylight = np.asarray([row["common_daylight_valid"] for row in point_rows], dtype=bool)
    leads = np.asarray([int(row["lead_hour"]) for row in point_rows])
    cycles = np.asarray([row["cycle"] for row in point_rows])
    metrics_overall = {}
    metrics_all_common = {}
    metrics_by_lead = []
    metrics_by_cycle = []
    for scheme in model_names:
        metrics_overall[scheme] = metric(values["observed"][common_daylight], values[scheme][common_daylight])
        metrics_all_common[scheme] = metric(values["observed"][common_valid], values[scheme][common_valid])
        for lead in range(1, 7):
            mask = common_daylight & (leads == lead)
            metrics_by_lead.append({"scheme": scheme, "lead_hour": lead, **metric(values["observed"][mask], values[scheme][mask])})
        for cycle in sorted(set(cycles.tolist())):
            mask = common_daylight & (cycles == cycle)
            metrics_by_cycle.append({"scheme": scheme, "cycle": cycle, **metric(values["observed"][mask], values[scheme][mask])})
    lead_path = OUT / "metrics_by_lead.csv"
    cycle_path = OUT / "metrics_by_cycle.csv"
    write_metric_csv(lead_path, metrics_by_lead)
    write_metric_csv(cycle_path, metrics_by_cycle)
    metrics_path = OUT / "metrics.json"
    metrics_path.write_text(json.dumps({
        "schema_version": f"wrf_pbl_{RESULT_TAG}_metrics_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection": str(SEL.relative_to(ROOT)),
        "sample_count": len(case_rows),
        "point_count": len(point_rows),
        "common_valid_count": int(common_valid.sum()),
        "common_daylight_valid_count": int(common_daylight.sum()),
        "evaluation_mask": "common_daylight_valid = frozen GHI label valid AND finite MYNN/YSU/MYJ predictions AND solar cosine > 0",
        "normalization": "nRMSE and nMBE are percentages normalized by mean observed GHI over the same common daylight mask",
        "prediction_definition": "hourly GHI = diff(ACSWDNB endpoints) / 3600; ACSWDNB in J m-2 and predictions in W m-2",
        "overall_common_daylight": metrics_overall,
        "overall_common_all_valid": metrics_all_common,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    input_hashes = {
        "selection_manifest": sha256(SEL),
        "mynn_audit": sha256(MYNN_AUDIT),
        "ysu_audit": sha256(PBL_AUDITS["ysu"]),
        "myj_audit": sha256(PBL_AUDITS["myj"]),
    }
    input_hashes.update({f"labels_{name}": sha256(path) for name, path in LABEL_FILES.items()})
    input_hashes.update({f"source_manifest_{name}": sha256(path) for name, path in SOURCE_MANIFESTS.items()})
    output_hashes = {path.name: sha256(path) for path in [prediction_path, case_path, lead_path, cycle_path, metrics_path]}
    daylight_excluded_cases = [
        row["sample_id"] for row in case_rows if row["common_daylight_valid_count"] == 0
    ]
    audit_payload = {
        "schema_version": f"wrf_pbl_{RESULT_TAG}_extraction_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(row["mapping_consistent_all_schemes"] and row["time_alignment_all_schemes"] for row in case_rows) else "failed",
        "selection": str(SEL.relative_to(ROOT)) if SEL.is_relative_to(ROOT) else str(SEL),
        "case_count": len(case_rows),
        "point_count": len(point_rows),
        "scheme_case_counts": {scheme: len(case_rows) for scheme in model_names},
        "label_source_counts": dict(Counter(label_source[row["sample_id"]] for row in case_rows)),
        "mynn_source_tree_counts": dict(Counter(row["mynn_source_tree"] for row in case_rows)),
        "validity_counts": {
            "label_valid_points": int(sum(row["label_valid_count"] for row in case_rows)),
            "common_valid_points": int(sum(row["common_valid_count"] for row in case_rows)),
            "common_daylight_valid_points": int(sum(row["common_daylight_valid_count"] for row in case_rows)),
            "cases_with_6_common_daylight_points": sum(row["common_daylight_valid_count"] == 6 for row in case_rows),
            "cases_with_no_common_daylight_points": len(daylight_excluded_cases),
        },
        "daylight_excluded_cases": daylight_excluded_cases,
        "mapping": {
            "reference": "YSU d02 nearest grid to station; same grid index applied to MYJ and MYNN",
            "all_scheme_coordinate_consistency": all(row["mapping_consistent_all_schemes"] for row in case_rows),
            "distance_min_km": min(row["station_grid_distance_km"] for row in case_rows),
            "distance_max_km": max(row["station_grid_distance_km"] for row in case_rows),
            "distance_mean_km": float(np.mean([row["station_grid_distance_km"] for row in case_rows])),
        },
        "time_alignment": {
            "all_cases_all_schemes_match_frozen_label_valid_times": all(row["time_alignment_all_schemes"] for row in case_rows),
            "lead_hours": [1, 2, 3, 4, 5, 6],
            "valid_time_basis": "frozen label valid_time_utc",
        },
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "outputs": {
            "predictions": prediction_path.name,
            "case_alignment_audit": case_path.name,
            "metrics": metrics_path.name,
            "metrics_by_lead": lead_path.name,
            "metrics_by_cycle": cycle_path.name,
        },
    }
    audit_path = OUT / "audit.json"
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit_payload["status"],
        "case_count": len(case_rows),
        "point_count": len(point_rows),
        "common_valid": int(common_valid.sum()),
        "common_daylight_valid": int(common_daylight.sum()),
        "distance_km": [audit_payload["mapping"]["distance_min_km"], audit_payload["mapping"]["distance_max_km"]],
        "outputs": {key: str(OUT / value) for key, value in audit_payload["outputs"].items()},
        "overall_common_daylight": metrics_overall,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
