#!/usr/bin/env python3
"""Audit the 49 new WRF-PBL cases for the 119-case test17 comparison."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/share/home/xiajingming/xjm_shaoyongjin/guizhou_wrf")
FULL_MANIFEST = ROOT / "manifests/wrf_pbl_fulltest_20260921/pbl_fulltest_119_jobs.csv"
REMAIN_MANIFEST = ROOT / "manifests/wrf_pbl_fulltest_20260921/pbl_fulltest_remaining49_jobs.csv"
SOURCE_MANIFESTS = {
    "data/expansion_p46_v4/wrf_random_6h_cases.csv": ROOT / "data/expansion_p46_v4/wrf_random_6h_cases.csv",
    "data/formal_0_6_v1/manifests/wrf_random_6h_cases.csv": ROOT / "data/formal_0_6_v1/manifests/wrf_random_6h_cases.csv",
}
OUT = ROOT / "manifests/wrf_pbl_fulltest_20260921"
WRF_ROOT = ROOT.parent / "Model/WRFV4.5.2"
WPS_ROOT = ROOT.parent / "Model/WPSV4.2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def source_rows() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for relative, path in SOURCE_MANIFESTS.items():
        for row in read_csv(path):
            sample_id = row["sample_id"]
            if sample_id in result:
                raise ValueError(f"duplicate source sample_id: {sample_id}")
            result[sample_id] = {**row, "_source_manifest": relative}
    return result


def mynn_root(row: dict[str, str]) -> tuple[str, Path]:
    if row["manifest"].startswith("data/expansion_p46_v4/"):
        return "v4_expansion_1431/direct6", ROOT / "runs/v4_expansion_1431" / row["sample_id"] / "direct6"
    return "v4_hourly_799", ROOT / "runs/v4_hourly_799" / row["sample_id"]


def audit_case(row: dict[str, str], source: dict[str, str]) -> dict[str, object]:
    tree, case_root = mynn_root(row)
    qc_path = case_root / "qc.json"
    qc_status = "missing"
    qc_passed = False
    d01_hours = d02_hours = 0
    d01_files = d02_files = 0
    wrf_success = False
    nan_inf_status = "unknown"
    if qc_path.is_file():
        try:
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            qc_status = str(qc.get("status", "missing"))
            d01_hours = len(qc.get("domains", {}).get("d01", {}).get("times", []))
            d02_hours = len(qc.get("domains", {}).get("d02", {}).get("times", []))
            qc_passed = qc_status == "passed" and d01_hours == 7 and d02_hours == 7
            d01_files = len(list((case_root / "wrf").glob("wrfout_d01_*")))
            d02_files = len(list((case_root / "wrf").glob("wrfout_d02_*")))
            final_log = case_root / "wrf" / "rsl.error.0000"
            if final_log.is_file():
                log_text = final_log.read_text(errors="ignore")
                wrf_success = "SUCCESS COMPLETE WRF" in log_text
                nan_inf_status = "nan_or_inf" if "nan" in log_text.lower() or "inf" in log_text.lower() else "not_reported"
        except (OSError, json.JSONDecodeError, TypeError):
            qc_status = "invalid"

    if row["manifest"].startswith("data/expansion_p46_v4/"):
        input_root = ROOT / "data/raw/gfs_expansion87"
    else:
        input_root = ROOT / "data/raw/gfs"
    cycle = row["cycle"]
    input_dir = input_root / source["station_id"] / cycle
    hours = [int(value) for value in source["gfs_hours_direct"].split(",") if value]
    expected = [input_dir / f"f{hour:03d}.nc" for hour in hours]
    missing = [str(path) for path in expected if not path.is_file()]
    station = source["station_id"]
    geog_dir = ROOT / "cache/geogrid" / station
    geog_complete = all((geog_dir / name).is_file() and (geog_dir / name).stat().st_size > 0 for name in ("geo_em.d01.nc", "geo_em.d02.nc"))
    install_complete = all(path.is_file() for path in (WPS_ROOT / "geogrid.exe", WPS_ROOT / "metgrid.exe", WRF_ROOT / "main/real.exe", WRF_ROOT / "main/wrf.exe"))
    required = (
        qc_passed and d01_files >= 1 and d02_files >= 1 and
        len(missing) == 0 and geog_complete and install_complete
    )
    return {
        "sample_id": row["sample_id"],
        "station_id": source["station_id"],
        "cycle": row["cycle"],
        "source_manifest": row["manifest"],
        "source_tree": tree,
        "mynn_case_root": str(case_root),
        "mynn_qc_path": str(qc_path),
        "mynn_qc_status": qc_status,
        "mynn_qc_passed": qc_passed,
        "mynn_d01_hours": d01_hours,
        "mynn_d02_hours": d02_hours,
        "mynn_d01_files": d01_files,
        "mynn_d02_files": d02_files,
        "mynn_wrf_success": wrf_success,
        "mynn_nan_inf_status": nan_inf_status,
        "gfs_input_dir": str(input_dir),
        "gfs_expected": len(expected),
        "gfs_present": len(expected) - len(missing),
        "gfs_missing": ";".join(missing),
        "geog_cache_complete": geog_complete,
        "wrf_wps_install_complete": install_complete,
        "all_required_inputs_complete": required,
    }


def main() -> int:
    full = read_csv(FULL_MANIFEST)
    remaining = read_csv(REMAIN_MANIFEST)
    source = source_rows()
    errors: list[str] = []
    if len(full) != 119:
        errors.append(f"full manifest rows={len(full)} expected=119")
    if len(remaining) != 49:
        errors.append(f"remaining manifest rows={len(remaining)} expected=49")
    full_keys = {f'{row["station_id"]}__{row["cycle"]}__{row["sample_id"].split("__")[-1]}' for row in full}
    if len(full_keys) != len(full):
        errors.append("full manifest contains duplicate case keys")
    selected = read_csv(ROOT / "manifests/cycle_level_10case_draft_20260921/selected_cycle_station_expansion_draft.csv")
    selected_keys = {row["case_key"] for row in selected}
    remaining_keys = {f'{row["station_id"]}__{row["cycle"]}__{row["sample_id"].split("__")[-1]}' for row in remaining}
    if len(full_keys & selected_keys) != 70:
        errors.append(f"overlap with existing 70={len(full_keys & selected_keys)} expected=70")
    if len(full_keys - selected_keys) != 49 or remaining_keys != full_keys - selected_keys:
        errors.append("remaining manifest does not equal full-minus-existing set")

    rows: list[dict[str, object]] = []
    for row in remaining:
        sample_id = row["sample_id"]
        if sample_id not in source:
            errors.append(f"missing source row: {sample_id}")
            continue
        rows.append(audit_case(row, source[sample_id]))
    fieldnames = list(rows[0]) if rows else []
    with (OUT / "pbl_fulltest_remaining49_input_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    passed = not errors and all(bool(row["all_required_inputs_complete"]) for row in rows)
    payload = {
        "schema_version": "wrf_pbl_fulltest_input_audit_v1",
        "status": "passed" if passed else "failed",
        "full_case_count": len(full),
        "existing_case_count": len(full_keys & selected_keys),
        "remaining_case_count": len(rows),
        "remaining_source_tree_counts": dict(Counter(str(row["source_tree"]) for row in rows)),
        "mynn_qc_passed_count": sum(bool(row["mynn_qc_passed"]) for row in rows),
        "gfs_complete_count": sum(row["gfs_missing"] == "" for row in rows),
        "geog_cache_complete_count": sum(bool(row["geog_cache_complete"]) for row in rows),
        "wrf_wps_install_complete_count": sum(bool(row["wrf_wps_install_complete"]) for row in rows),
        "all_required_inputs_complete_count": sum(bool(row["all_required_inputs_complete"]) for row in rows),
        "errors": errors,
        "manifest_sha256": {
            "full": sha256(FULL_MANIFEST),
            "remaining": sha256(REMAIN_MANIFEST),
        },
    }
    (OUT / "pbl_fulltest_remaining49_input_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
