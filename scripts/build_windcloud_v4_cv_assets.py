#!/usr/bin/env python3
"""Create grouped five-fold splits and fold-only AOD references for v4."""

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


def case_aod_rows(root: Path, index: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in index:
        with np.load(root / row["case_file"], allow_pickle=False) as case:
            station = str(case["station_id"].item())
            initialization = str(case["wrf_init_time_utc"].item())
            times = case["valid_time_utc"].astype(str)
            values = case["aod550"].astype(float)
            mask = case["aod_mask"].astype(bool)
        for lead in range(6):
            result.append({
                "station_id": station,
                "wrf_init_time_utc": initialization,
                "valid_time_utc": str(times[lead]),
                "aod550": f"{values[lead]:.9g}" if mask[lead] else "",
            })
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    development_rows = read_index(args.development_root)
    confirmation_rows = read_index(args.confirmation_root)
    development_cycles = sorted({row["cycle"] for row in development_rows})
    confirmation_cycles = sorted({row["cycle"] for row in confirmation_rows})
    if set(development_cycles) & set(confirmation_cycles):
        raise ValueError("development and confirmation cycles overlap")
    if len(development_cycles) != 73 or len(confirmation_cycles) != 13:
        raise ValueError(
            f"expected 73 development and 13 confirmation cycles, found "
            f"{len(development_cycles)} and {len(confirmation_cycles)}"
        )

    # Chronological round-robin gives every fold broad seasonal/year coverage
    # while keeping the indivisible grouping unit as the complete GFS cycle.
    validation_groups = [development_cycles[index::args.folds] for index in range(args.folds)]
    fold_rows = []
    for fold, validation in enumerate(validation_groups):
        validation_set = set(validation)
        fold_rows.append({
            "fold": fold,
            "train": [cycle for cycle in development_cycles if cycle not in validation_set],
            "validation": validation,
        })
    cv = {
        "schema_version": "windcloud_v4_grouped_cv_73_v1",
        "group_unit": "gfs_init_cycle",
        "assignment": "chronological_round_robin_five_fold",
        "development": development_cycles,
        "confirmation": confirmation_cycles,
        "test": confirmation_cycles,
        "folds": fold_rows,
    }
    write_json(args.output, cv)

    aod_rows = case_aod_rows(args.development_root, development_rows)
    statistics = args.development_root / "statistics"
    for fold in fold_rows:
        reference = training_only_aod_reference(aod_rows, fold["train"])
        reference["fold"] = fold["fold"]
        reference["schema_version"] = "windcloud_v4_fold_aod_reference_v1"
        write_json(statistics / f"fold_{fold['fold']}_aod_reference.json", reference)
    development_reference = training_only_aod_reference(aod_rows, development_cycles)
    development_reference["fold"] = "all_development"
    development_reference["schema_version"] = "windcloud_v4_development_aod_reference_v1"
    write_json(statistics / "development_aod_reference.json", development_reference)

    validation_union = set().union(*(set(fold["validation"]) for fold in fold_rows))
    if validation_union != set(development_cycles):
        raise RuntimeError("five-fold validation union does not equal the development cycles")
    validation_total = sum(len(fold["validation"]) for fold in fold_rows)
    if validation_total != len(validation_union):
        raise RuntimeError("a development cycle appears in multiple validation folds")
    print(json.dumps({
        "development_cases": len(development_rows),
        "development_cycles": len(development_cycles),
        "confirmation_cases": len(confirmation_rows),
        "confirmation_cycles": len(confirmation_cycles),
        "fold_validation_cycle_counts": [len(group) for group in validation_groups],
        "aod_rows": len(aod_rows),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
