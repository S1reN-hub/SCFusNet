#!/usr/bin/env python3
"""Build minute-resolution clear-sky auxiliary labels for the 799 cases.

Future observations are used only to supervise the two clear-state heads.  At
inference the label is absent and the heads operate on their own expert state.
The score combines normalized-GHI smoothness, irradiance level and direct-beam
evidence so that a smooth high-AOD clear hour is not confused with overcast.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_gfs_swradnet_dataset import (  # noqa: E402
    load_observation_minutes,
    load_stations,
    parse_utc,
)
from scripts.radiation_utils import haurwitz_clear_sky_ghi, solar_position  # noqa: E402


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def clear_score_for_hour(rows, valid_time, station, minimum_valid_minutes: int) -> dict:
    start = valid_time - timedelta(hours=1)
    times = [start + timedelta(minutes=index) for index in range(60)]
    values = [rows.get(value) for value in times]
    ghi = np.asarray([np.nan if row is None or row[0] is None else row[0] for row in values], dtype=np.float64)
    dni = np.asarray([np.nan if row is None or row[1] is None else row[1] for row in values], dtype=np.float64)
    dhi = np.asarray([np.nan if row is None or row[2] is None else row[2] for row in values], dtype=np.float64)
    centres = np.asarray([np.datetime64(value.replace(tzinfo=None)) + np.timedelta64(30, "s") for value in times])
    mu, distance = solar_position(centres, station.latitude, station.longitude, clip_below_horizon=False)
    clear = haurwitz_clear_sky_ghi(mu, distance).astype(np.float64)
    valid_ghi = np.isfinite(ghi)
    stable_day = (mu >= 0.15) & (clear >= 100.0)
    used = valid_ghi & stable_day
    result = {
        "clear_score": 0.0,
        "clear_valid": 0,
        "valid_ghi_minutes": int(valid_ghi.sum()),
        "stable_day_minutes": int(stable_day.sum()),
        "used_minutes": int(used.sum()),
        "median_clear_index": "",
        "median_abs_clear_index_increment": "",
        "median_direct_fraction": "",
        "median_diffuse_fraction": "",
        "direct_minutes": 0,
    }
    # The hourly target already requires 54 valid minutes.  Clear supervision
    # additionally requires at least 20 minutes away from the solar horizon.
    if valid_ghi.sum() < minimum_valid_minutes or used.sum() < 20:
        return result

    clear_index = np.clip(ghi[used] / np.maximum(clear[used], 20.0), 0.0, 1.6)
    median_index = float(np.median(clear_index))
    increments = np.abs(np.diff(clear_index))
    increment_noise = float(np.median(increments)) if increments.size else 1.0
    smooth_score = math.exp(-((increment_noise / 0.035) ** 2))
    level_score = sigmoid((median_index - 0.52) / 0.08)

    direct_valid = used & np.isfinite(dni) & (ghi >= 30.0)
    direct_fraction = np.clip(dni[direct_valid] * np.maximum(mu[direct_valid], 0.0) / np.maximum(ghi[direct_valid], 30.0), 0.0, 1.2)
    diffuse_valid = used & np.isfinite(dhi) & (ghi >= 30.0)
    diffuse_fraction = np.clip(dhi[diffuse_valid] / np.maximum(ghi[diffuse_valid], 30.0), 0.0, 1.2)
    if direct_valid.sum() >= 20:
        median_direct = float(np.median(direct_fraction))
        direct_score = sigmoid((median_direct - 0.48) / 0.08)
    elif diffuse_valid.sum() >= 20:
        median_direct = float(1.0 - np.median(diffuse_fraction))
        direct_score = sigmoid((median_direct - 0.48) / 0.08)
    else:
        median_direct = math.nan
        direct_score = level_score

    # Geometric mean requires all three independent indicators to agree while
    # retaining a continuous target for borderline conditions.
    score = float(np.clip((smooth_score * level_score * direct_score) ** (1.0 / 3.0), 0.0, 1.0))
    result.update(
        clear_score=score,
        clear_valid=1,
        median_clear_index=median_index,
        median_abs_clear_index_increment=increment_noise,
        median_direct_fraction="" if not math.isfinite(median_direct) else median_direct,
        median_diffuse_fraction="" if not diffuse_valid.any() else float(np.median(diffuse_fraction)),
        direct_minutes=int(direct_valid.sum()),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-labels", type=Path, required=True)
    parser.add_argument("--stations", type=Path, default=Path("configs/gfs_swradnet_stations.csv"))
    parser.add_argument("--observations", type=Path, default=Path("data/radiation_observations/raw"))
    parser.add_argument("--minimum-valid-minutes", type=int, default=54)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.hourly_labels.open(encoding="utf-8-sig", newline="") as stream:
        input_rows = list(csv.DictReader(stream))
    if len(input_rows) != 799 * 6:
        raise ValueError(f"expected 4794 case-lead rows, found {len(input_rows)}")
    stations = load_stations(args.stations)
    by_station = defaultdict(list)
    for row in input_rows:
        by_station[row["station_id"]].append(row)

    output_rows = []
    for station_id, station_rows in sorted(by_station.items()):
        station = stations[station_id]
        valid_times = [parse_utc(row["valid_time_utc"]) for row in station_rows]
        observations = load_observation_minutes(station, valid_times, args.observations)
        for row, valid_time in zip(station_rows, valid_times):
            diagnostics = clear_score_for_hour(observations, valid_time, station, args.minimum_valid_minutes)
            output_rows.append({
                "sample_id": row["sample_id"],
                "station_id": station_id,
                "cycle": parse_utc(row["wrf_init_time_utc"]).strftime("%Y%m%d%H"),
                "lead_hour": int(row["lead_hour"]),
                "valid_time_utc": row["valid_time_utc"],
                **diagnostics,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    keys = {(row["sample_id"], row["lead_hour"]) for row in output_rows}
    if len(keys) != len(output_rows):
        raise ValueError("duplicate sample_id/lead_hour keys in clear labels")
    fields = list(output_rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    valid = [row for row in output_rows if row["clear_valid"]]
    bins = Counter(
        "clear" if row["clear_score"] >= 0.7 else "cloudy" if row["clear_score"] <= 0.3 else "mixed"
        for row in valid
    )
    print({"rows": len(output_rows), "valid": len(valid), "classes": dict(bins), "output": str(args.output)})


if __name__ == "__main__":
    main()
