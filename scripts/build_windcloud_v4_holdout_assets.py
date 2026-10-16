#!/usr/bin/env python3
"""Create one cycle-grouped train/validation split and training-only AOD reference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ghi_hierarchical_v2.data_contract import training_only_aod_reference  # noqa: E402


def read_index(root: Path) -> list[dict[str, str]]:
    with (root / "index.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def aod_rows(root: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in rows:
        with np.load(root / row["case_file"], allow_pickle=False) as case:
            station = str(case["station_id"].item())
            initialization = str(case["wrf_init_time_utc"].item())
            times = case["valid_time_utc"].astype(str)
            values = case["aod550"].astype(float)
            masks = case["aod_mask"].astype(bool)
        for index in range(6):
            result.append({
                "station_id": station,
                "wrf_init_time_utc": initialization,
                "valid_time_utc": str(times[index]),
                "aod550": f"{values[index]:.9g}" if masks[index] else "",
            })
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--validation-cycles", type=int, default=11)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--aod-output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_index(args.development_root)
    confirmation_rows = read_index(args.confirmation_root)
    cycles = sorted({row["cycle"] for row in rows})
    confirmation = sorted({row["cycle"] for row in confirmation_rows})
    if len(cycles) != 73 or len(confirmation) != 13:
        raise ValueError("expected 73 development and 13 confirmation cycles")
    if set(cycles) & set(confirmation):
        raise ValueError("development and confirmation cycles overlap")
    if not 1 <= args.validation_cycles < len(cycles):
        raise ValueError("validation-cycles must be between 1 and 72")

    # Pick evenly spaced chronological positions, preserving complete cycles.
    positions = np.linspace(0, len(cycles) - 1, args.validation_cycles + 2)[1:-1]
    indices = sorted({int(round(value)) for value in positions})
    if len(indices) != args.validation_cycles:
        raise RuntimeError("evenly spaced validation selection produced duplicate indices")
    validation = [cycles[index] for index in indices]
    validation_set = set(validation)
    training = [cycle for cycle in cycles if cycle not in validation_set]
    split = {
        "schema_version": "windcloud_v4_cycle_holdout_85_15_v1",
        "group_unit": "gfs_init_cycle",
        "assignment": "evenly_spaced_chronological_holdout",
        "development": cycles,
        "confirmation": confirmation,
        "test": confirmation,
        "folds": [{"fold": 0, "train": training, "validation": validation}],
    }
    write_json(args.split_output, split)

    reference = training_only_aod_reference(aod_rows(args.development_root, rows), training)
    reference["fold"] = 0
    reference["schema_version"] = "windcloud_v4_holdout_training_aod_reference_v1"
    write_json(args.aod_output, reference)

    train_cases = sum(row["cycle"] in set(training) for row in rows)
    validation_cases = sum(row["cycle"] in validation_set for row in rows)
    print(json.dumps({
        "training_cycles": len(training),
        "validation_cycles": len(validation),
        "training_cases": train_cases,
        "validation_cases": validation_cases,
        "confirmation_cycles": len(confirmation),
        "confirmation_cases": len(confirmation_rows),
        "split_output": str(args.split_output),
        "aod_output": str(args.aod_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
