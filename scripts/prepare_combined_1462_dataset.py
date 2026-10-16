#!/usr/bin/env python3
"""Build a read-only combined 1462-case view for the 17-train/7-evaluation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_index(root: Path) -> list[dict[str, str]]:
    with (root / "index.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_role(row: dict[str, str]) -> str:
    for name in ("role", "current_role", "data_role", "dataset_role", "partition"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def aod_reference(root: Path, train_rows: list[dict[str, str]]) -> dict:
    station_month: dict[str, list[float]] = defaultdict(list)
    station: dict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for row in train_rows:
        with np.load(root / row["case_file"], allow_pickle=False) as case:
            station_id = str(case["station_id"].item())
            times = case["valid_time_utc"].astype(str)
            values = case["aod550"].astype(float)
            masks = case["aod_mask"].astype(bool)
        for valid_time, value, valid in zip(times, values, masks):
            if not valid or not np.isfinite(value):
                continue
            station_month[f"{station_id}|{valid_time[5:7]}"].append(float(value))
            station[station_id].append(float(value))
            all_values.append(float(value))
    if not all_values:
        raise ValueError("training rows contain no valid AOD values")
    return {
        "schema_version": "station_aware_training_aod_reference_v1",
        "station_month_median": {key: float(np.median(value)) for key, value in sorted(station_month.items())},
        "station_median": {key: float(np.median(value)) for key, value in sorted(station.items())},
        "global_median": float(np.median(all_values)),
        "fallback": "station_month_then_station_then_global",
        "training_case_count": len(train_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    development = args.development_root.resolve()
    confirmation = args.confirmation_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing combined root: {output}")
    for root in (development, confirmation):
        if not (root / "index.csv").is_file():
            raise FileNotFoundError(root / "index.csv")

    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    evaluation_stations = {value.upper() for value in config["evaluation_stations"]}
    training_stations = {value.upper() for value in config["training_stations"]}
    cycle_split = {
        cycle: split_name
        for split_name, cycles in config["split_cycles"].items()
        for cycle in cycles
    }
    if len(cycle_split) != 86:
        raise ValueError(f"split config must cover 86 cycles, found {len(cycle_split)}")

    aliases = {"development": development, "confirmation": confirmation}
    output.mkdir(parents=True)

    combined: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for source_name, source_root in aliases.items():
        for source_row in read_index(source_root):
            row = dict(source_row)
            original_case = Path(row["case_file"])
            if original_case.is_absolute():
                source_case = original_case
            else:
                source_case = (source_root / original_case).resolve()
            if not source_case.is_file():
                raise FileNotFoundError(source_case)
            cycle = str(row.get("cycle", row.get("wrf_init_time_utc", ""))).strip()
            station_id = str(row.get("station_id", "")).upper()
            if cycle not in cycle_split or station_id not in training_stations:
                raise ValueError(f"unrecognized station/cycle: {station_id}/{cycle}")
            experiment_split = cycle_split[cycle] if station_id in evaluation_stations else "train"
            row["case_file"] = str(source_case)
            row["source_dataset"] = source_name
            row["source_data_role"] = source_role(source_row)
            row["experiment_split"] = experiment_split
            row["role"] = "development"
            combined.append(row)
            for field in row:
                if field not in seen_fields:
                    seen_fields.add(field)
                    fieldnames.append(field)

    if len(combined) != 1462 or len({row["sample_id"] for row in combined}) != 1462:
        raise ValueError("combined index must contain 1462 unique sample IDs")
    counts = Counter(row["experiment_split"] for row in combined)
    if counts != Counter({"train": 1224, "validation": 119, "test": 119}):
        raise ValueError(f"unexpected split counts: {dict(counts)}")
    station_counts = Counter(row["station_id"] for row in combined)
    if set(station_counts.values()) != {86} or len(station_counts) != 17:
        raise ValueError(f"unexpected station coverage: {dict(station_counts)}")

    with (output / "index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    (output / "split_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    cv_source = args.split_config.parent.parent / config["cv_file"]
    if not cv_source.is_file():
        cv_source = Path(config["cv_file"])
    if not cv_source.is_file():
        raise FileNotFoundError(f"CV file from split config is unavailable: {config['cv_file']}")
    (output / "cv_17train_7eval.json").write_bytes(cv_source.read_bytes())

    train_rows = [row for row in combined if row["experiment_split"] == "train"]
    reference = aod_reference(output, train_rows)
    statistics = output / "statistics"
    statistics.mkdir()
    reference_path = statistics / "fold_0_aod_reference.json"
    reference_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "combined_1462_station_aware_view_v1",
        "source_roots": {key: str(value) for key, value in aliases.items()},
        "source_index_sha256": {key: sha256(value / "index.csv") for key, value in aliases.items()},
        "combined_index_sha256": sha256(output / "index.csv"),
        "counts": dict(counts),
        "station_counts": dict(sorted(station_counts.items())),
        "evaluation_stations": sorted(evaluation_stations),
        "training_stations": sorted(training_stations),
        "aod_reference": str(reference_path),
    }
    (output / "combined_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
