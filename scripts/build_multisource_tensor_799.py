#!/usr/bin/env python3
"""Build canonical multimodal tensors for the frozen 799 random-issue cases."""

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ghi_hierarchical_v2.data_contract import (  # noqa: E402
    CANONICAL_SATELLITE_CHANNELS,
    EXPECTED_CASES,
    EXPECTED_LEADS,
    iso_utc,
    parse_utc,
    canonicalize_satellite_frame,
    solar_hour_features,
    training_only_aod_reference,
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(str(temporary), **arrays)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_station_table(path):
    rows = read_csv(path)
    climate_names = sorted(set(row["climate_group"] for row in rows))
    climate_index = {name: index for index, name in enumerate(climate_names)}
    table = {}
    for row in rows:
        table[row["station_id"]] = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "elevation_m": float(row["elevation_m"]),
            "climate_group": row["climate_group"],
            "climate_id": climate_index[row["climate_group"]],
        }
    if len(table) != 17:
        raise ValueError("station table must contain exactly 17 stations")
    return table, climate_names


def build(args):
    labels_path = args.labels.resolve()
    with np.load(str(labels_path), allow_pickle=False) as labels:
        label_data = {name: np.asarray(labels[name]) for name in labels.files}
    sample_ids = label_data["sample_id"].astype(str)
    if len(sample_ids) != EXPECTED_CASES or len(set(sample_ids.tolist())) != EXPECTED_CASES:
        raise ValueError("labels do not contain 799 unique sample IDs")
    if tuple(label_data["target"].shape) != (799, 6, 3):
        raise ValueError("unexpected target shape")

    aod_rows = read_csv(args.aod)
    aod_index = {(row["sample_id"], int(row["lead_hour"])): row for row in aod_rows}
    if len(aod_index) != EXPECTED_CASES * len(EXPECTED_LEADS):
        raise ValueError("AOD must contain one unique row for every case and lead")
    inventory_rows = read_csv(args.satellite_inventory)
    sat_index = {
        (row["sample_id"], int(row["history_offset_hour"])): row
        for row in inventory_rows
    }
    if len(sat_index) != EXPECTED_CASES * 6:
        raise ValueError("satellite inventory must contain six unique history slots per case")
    station_table, climate_names = load_station_table(args.stations)
    cv = json.loads(args.cv.read_text(encoding="utf-8"))

    output_root = args.output_root.resolve()
    cases_root = output_root / "cases"
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    source_counts = Counter()
    for case_index, sample_id in enumerate(sample_ids):
        station_id = str(label_data["station_id"][case_index])
        station = station_table[station_id]
        wrf_path = args.wrf_root / "cases" / (sample_id + ".npz")
        if not wrf_path.is_file():
            raise FileNotFoundError(str(wrf_path))
        with np.load(str(wrf_path), allow_pickle=False) as wrf:
            wrf_data = {name: np.asarray(wrf[name]) for name in wrf.files}
        wrf_sample = str(wrf_data["sample_id"].item())
        if wrf_sample != sample_id:
            raise ValueError("WRF sample mismatch for %s" % sample_id)
        if tuple(wrf_data["nwp_3d"].shape) != (6, 6, 10, 5, 5):
            raise ValueError("invalid WRF 3-D shape for %s" % sample_id)
        if tuple(wrf_data["nwp_surface"].shape) != (6, 4, 5, 5):
            raise ValueError("invalid WRF surface shape for %s" % sample_id)

        satellite = np.zeros((6, 7, 32, 32), dtype=np.float32)
        satellite_valid = np.zeros_like(satellite, dtype=bool)
        satellite_channel_mask = np.zeros((6, 7), dtype=bool)
        satellite_frame_mask = np.zeros(6, dtype=bool)
        satellite_age_hours = np.zeros(6, dtype=np.float32)
        source_ids = []
        for history_index, offset in enumerate(range(-5, 1)):
            row = sat_index[(sample_id, offset)]
            satellite_path = args.satellite_root / row["relative_path"]
            with np.load(str(satellite_path), allow_pickle=False) as frame:
                values = np.asarray(frame["window_mean"])
                valid = np.asarray(frame["validity"])
                metadata = json.loads(str(frame["metadata_json"].item()))
            expected_time = parse_utc(row["history_time_utc"])
            frame_time = metadata.get("valid_time") or metadata.get("frame_time")
            if frame_time is None or parse_utc(frame_time) != expected_time:
                raise ValueError("satellite frame time mismatch: %s" % satellite_path)
            scan_times = metadata.get("scan_times") or [metadata.get("scan_time")]
            if not scan_times or any(value is None or parse_utc(value) > expected_time for value in scan_times):
                raise ValueError("non-causal satellite frame: %s" % satellite_path)
            canonical, valid_canonical, channel_mask, source_id = canonicalize_satellite_frame(values, valid)
            satellite[history_index] = canonical
            satellite_valid[history_index] = valid_canonical
            satellite_channel_mask[history_index] = channel_mask
            satellite_frame_mask[history_index] = bool(channel_mask.any())
            satellite_age_hours[history_index] = float(-offset)
            source_ids.append(source_id)
        if len(set(source_ids)) != 1:
            raise ValueError("satellite source changes within a case: %s" % sample_id)
        satellite_source_id = source_ids[0]
        source_counts[str(satellite_source_id)] += 1

        valid_times = label_data["valid_time_utc"][case_index].astype(str)
        aod = np.zeros(6, dtype=np.float32)
        aod_mask = np.zeros(6, dtype=bool)
        solar = np.zeros((6, 4), dtype=np.float32)
        subhour_cos = np.zeros((6, 12), dtype=np.float32)
        for lead_index, lead in enumerate(EXPECTED_LEADS):
            row = aod_index[(sample_id, lead)]
            if parse_utc(row["valid_time_utc"]) != parse_utc(valid_times[lead_index]):
                raise ValueError("AOD valid time mismatch: %s lead %d" % (sample_id, lead))
            if row.get("aod550"):
                aod[lead_index] = float(row["aod550"])
                aod_mask[lead_index] = True
            features = solar_hour_features(
                parse_utc(valid_times[lead_index]), station["latitude"], station["longitude"]
            )
            subhour_cos[lead_index] = features["subhour_cos_sza"]
            solar[lead_index] = np.asarray(
                [
                    features["cos_sza_mean"],
                    features["clear_ghi_wm2"],
                    features["daylight_fraction"],
                    features["transition_geometry"],
                ],
                dtype=np.float32,
            )

        cycle = parse_utc(str(label_data["wrf_init_time_utc"][case_index])).strftime("%Y%m%d%H")
        output_path = cases_root / (sample_id + ".npz")
        atomic_npz(
            output_path,
            sample_id=np.asarray(sample_id),
            partition=np.asarray(str(label_data["partition"][case_index])),
            legacy_manifest_split=np.asarray(str(label_data["legacy_manifest_split"][case_index])),
            station_id=np.asarray(station_id),
            wrf_init_time_utc=np.asarray(str(label_data["wrf_init_time_utc"][case_index])),
            issue_time_utc=np.asarray(str(label_data["issue_time_utc"][case_index])),
            issue_offset_hours=np.asarray(label_data["issue_offset_hours"][case_index], dtype=np.int16),
            valid_time_utc=valid_times,
            lead_hours=np.asarray(EXPECTED_LEADS, dtype=np.int8),
            target=label_data["target"][case_index, :, 0].astype(np.float32),
            target_mask=label_data["target_mask"][case_index, :, 0].astype(bool),
            valid_minutes=label_data["valid_minutes"][case_index, :, 0].astype(np.int16),
            nwp_3d=wrf_data["nwp_3d"].astype(np.float32),
            nwp_3d_valid=wrf_data["nwp_3d_valid"].astype(bool),
            nwp_3d_names=wrf_data["nwp_3d_names"],
            pressure_hpa=wrf_data["pressure_hpa"],
            nwp_surface=wrf_data["nwp_surface"].astype(np.float32),
            nwp_surface_names=wrf_data["nwp_surface_names"],
            wrf_anchor=wrf_data["wrf_anchor"].astype(np.float32),
            satellite=satellite,
            satellite_valid=satellite_valid,
            satellite_channel_mask=satellite_channel_mask,
            satellite_frame_mask=satellite_frame_mask,
            satellite_age_hours=satellite_age_hours,
            satellite_source_id=np.asarray(satellite_source_id, dtype=np.int8),
            aod550=aod,
            aod_mask=aod_mask,
            solar=solar,
            solar_names=np.asarray(
                ["cos_sza_hour_mean", "haurwitz_clear_ghi_wm2", "daylight_fraction", "transition_geometry"]
            ),
            subhour_cos_sza=subhour_cos,
            latitude=np.asarray(station["latitude"], dtype=np.float32),
            longitude=np.asarray(station["longitude"], dtype=np.float32),
            elevation_m=np.asarray(station["elevation_m"], dtype=np.float32),
            climate_id=np.asarray(station["climate_id"], dtype=np.int8),
            climate_group=np.asarray(station["climate_group"]),
        )
        records.append(
            {
                "sample_id": sample_id,
                "cycle": cycle,
                "station_id": station_id,
                "partition": str(label_data["partition"][case_index]),
                "legacy_manifest_split": str(label_data["legacy_manifest_split"][case_index]),
                "case_file": output_path.relative_to(output_root).as_posix(),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
            }
        )
        if (case_index + 1) % 50 == 0 or case_index + 1 == EXPECTED_CASES:
            print("built %d/%d" % (case_index + 1, EXPECTED_CASES), flush=True)

    index_path = output_root / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    statistics_root = output_root / "statistics"
    for fold in cv["folds"]:
        reference = training_only_aod_reference(aod_rows, fold["train"])
        reference["fold"] = int(fold["fold"])
        atomic_json(statistics_root / ("fold_%d_aod_reference.json" % int(fold["fold"])), reference)
    final_reference = training_only_aod_reference(aod_rows, cv["development"])
    final_reference["fold"] = "all_development"
    atomic_json(statistics_root / "development_aod_reference.json", final_reference)
    summary = {
        "schema_version": "multisource_tensor_799_v1",
        "case_count": len(records),
        "target_positions": int(label_data["target_mask"][:, :, 0].sum()),
        "missing_target_positions": int((~label_data["target_mask"][:, :, 0]).sum()),
        "satellite_frames": len(sat_index),
        "aod_positions": int(sum(bool(row.get("aod550")) for row in aod_rows)),
        "source_case_counts": dict(source_counts),
        "canonical_satellite_channels": list(CANONICAL_SATELLITE_CHANNELS),
        "climate_groups": climate_names,
        "index_sha256": sha256(index_path),
    }
    atomic_json(output_root / "build_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-root", type=Path, required=True)
    parser.add_argument("--satellite-root", type=Path, required=True)
    parser.add_argument("--satellite-inventory", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--aod", type=Path, required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--cv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
