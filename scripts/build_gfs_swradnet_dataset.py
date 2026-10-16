#!/usr/bin/env python
"""Build UTC-aligned, multi-station SWRadNet datasets from GFS windows.

Every label is the arithmetic mean of quality-controlled station minutes in
the left-closed/right-open UTC interval ``[valid_time-1h, valid_time)``.
GFS samples are grouped by initialization time before temporal splitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.radiation_utils import haurwitz_clear_sky_ghi, solar_position  # noqa: E402
from scripts.ssrd_schema import load_gfs_schema  # noqa: E402


UTC = timezone.utc
RUN_RE = re.compile(r"^(?P<date>\d{8})_(?P<cycle>\d{2})z$")
LEAD_RE = re.compile(r"^f(?P<lead>\d{3})\.npz$")


@dataclass(frozen=True)
class Station:
    station_id: str
    latitude: float
    longitude: float
    elevation_m: float
    network: str


@dataclass(frozen=True)
class GfsRecord:
    path: Path
    station: Station
    init_time: datetime
    forecast_hour: int

    @property
    def valid_time(self) -> datetime:
        return self.init_time + timedelta(hours=self.forecast_hour)

    @property
    def init_key(self) -> str:
        return self.init_time.strftime("%Y%m%d_%H")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp lacks an explicit UTC offset: {value}")
    return parsed.astimezone(UTC)


def utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("cannot serialize a naive datetime as UTC")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= -990.0:
        return None
    return number


def load_stations(path: Path) -> dict[str, Station]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["station_id"]: Station(
            station_id=row["station_id"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            elevation_m=float(row["elevation_m"]),
            network=row["network"],
        )
        for row in rows
    }


def discover_records(root: Path, stations: dict[str, Station]) -> list[GfsRecord]:
    records: list[GfsRecord] = []
    for station_id, station in stations.items():
        station_root = root / station_id
        if not station_root.exists():
            continue
        for run_dir in sorted(path for path in station_root.iterdir() if path.is_dir()):
            match = RUN_RE.match(run_dir.name)
            if not match:
                continue
            init_time = datetime.strptime(
                match.group("date") + match.group("cycle"), "%Y%m%d%H"
            ).replace(tzinfo=UTC)
            for path in sorted(run_dir.glob("f???.npz")):
                lead_match = LEAD_RE.match(path.name)
                if lead_match:
                    records.append(
                        GfsRecord(path, station, init_time, int(lead_match.group("lead")))
                    )
    return records


def _bsrn_month_path(root: Path, station_id: str, timestamp: datetime) -> Path:
    return root / "BSRN_PANGAEA" / station_id / f"{station_id}_{timestamp:%Y-%m}.txt"


def _surfrad_day_path(root: Path, station_id: str, timestamp: datetime) -> Path:
    doy = timestamp.timetuple().tm_yday
    return root / "NOAA_SURFRAD" / station_id / (
        f"{station_id.lower()}{timestamp.year % 100:02d}{doy:03d}.dat"
    )


def read_bsrn_file(path: Path) -> dict[datetime, tuple[float | None, float | None, float | None]]:
    result: dict[datetime, tuple[float | None, float | None, float | None]] = {}
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        header = None
        for line in handle:
            if line.startswith("Date/Time\t"):
                header = line.rstrip("\r\n").split("\t")
                break
        if header is None:
            raise ValueError(f"BSRN header not found: {path}")
        indices = {name: index for index, name in enumerate(header)}
        ghi_index = next((indices[x] for x in indices if x.startswith("SWD [")), None)
        dni_index = next((indices[x] for x in indices if x.startswith("DIR [")), None)
        dhi_index = next((indices[x] for x in indices if x.startswith("DIF [")), None)
        if ghi_index is None:
            raise ValueError(f"BSRN SWD column missing: {path}")
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if not fields:
                continue
            try:
                timestamp = datetime.fromisoformat(fields[0]).replace(tzinfo=UTC)
            except ValueError:
                continue

            def value(index: int | None) -> float | None:
                return safe_float(fields[index]) if index is not None and index < len(fields) else None

            result[timestamp] = (value(ghi_index), value(dni_index), value(dhi_index))
    return result


def read_surfrad_file(path: Path) -> dict[datetime, tuple[float | None, float | None, float | None]]:
    result: dict[datetime, tuple[float | None, float | None, float | None]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        next(handle, None)
        for line in handle:
            fields = line.split()
            if len(fields) < 16:
                continue
            try:
                timestamp = datetime(
                    int(fields[0]), int(fields[2]), int(fields[3]),
                    int(fields[4]), int(fields[5]), tzinfo=UTC,
                )
                ghi = safe_float(fields[8]) if int(fields[9]) == 0 else None
                dni = safe_float(fields[12]) if int(fields[13]) == 0 else None
                dhi = safe_float(fields[14]) if int(fields[15]) == 0 else None
            except (ValueError, IndexError):
                continue
            result[timestamp] = (ghi, dni, dhi)
    return result


def load_observation_minutes(
    station: Station,
    valid_times: list[datetime],
    observation_root: Path,
) -> dict[datetime, tuple[float | None, float | None, float | None]]:
    paths: set[Path] = set()
    for valid_time in valid_times:
        start = valid_time - timedelta(hours=1)
        for timestamp in (start, valid_time - timedelta(minutes=1)):
            if station.network == "BSRN_PANGAEA":
                paths.add(_bsrn_month_path(observation_root, station.station_id, timestamp))
            elif station.network == "NOAA_SURFRAD":
                paths.add(_surfrad_day_path(observation_root, station.station_id, timestamp))
            else:
                raise ValueError(f"unsupported observation network: {station.network}")
    result: dict[datetime, tuple[float | None, float | None, float | None]] = {}
    for path in sorted(paths):
        if not path.exists():
            continue
        rows = read_bsrn_file(path) if station.network == "BSRN_PANGAEA" else read_surfrad_file(path)
        overlap = set(result).intersection(rows)
        if overlap:
            raise ValueError(f"duplicate UTC observation minutes in {path}: {len(overlap)}")
        result.update(rows)
    return result


def aggregate_hour(
    rows: dict[datetime, tuple[float | None, float | None, float | None]],
    valid_time: datetime,
    minimum_valid_minutes: int,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    start = valid_time - timedelta(hours=1)
    values = [rows.get(start + timedelta(minutes=offset)) for offset in range(60)]
    ghi = [row[0] for row in values if row is not None and row[0] is not None]
    dni = [row[1] for row in values if row is not None and row[1] is not None]
    dhi = [row[2] for row in values if row is not None and row[2] is not None]
    if len(ghi) < minimum_valid_minutes:
        return None
    ghi_mean = max(float(np.mean(ghi)), 0.0)
    target = np.asarray([ghi_mean, 0.0, 0.0], dtype=np.float32)
    target_mask = np.asarray([True, False, False])
    if len(dni) >= minimum_valid_minutes:
        target[1] = max(float(np.mean(dni)), 0.0)
        target_mask[1] = True
    if len(dhi) >= minimum_valid_minutes:
        dhi_mean = float(np.clip(np.mean(dhi), 0.0, ghi_mean))
        target[2] = dhi_mean
        target_mask[2] = True
    return target, target_mask, {
        "ghi_minutes": len(ghi),
        "dni_minutes": len(dni),
        "dhi_minutes": len(dhi),
        "dni_mean_wm2": float(np.mean(dni)) if dni else None,
    }


def solar_features(
    valid_time: datetime,
    station: Station,
    lead: int,
    doy_mode: str,
    aod550: float | None,
    quadrature_substeps: int = 12,
) -> tuple[np.ndarray, list[str], bool]:
    end = np.datetime64(valid_time.replace(tzinfo=None), "ns")
    start = end - np.timedelta64(1, "h")
    offsets = (
        (np.arange(quadrature_substeps) + 0.5)
        * 3600.0 * 1e9 / quadrature_substeps
    ).astype("timedelta64[ns]")
    times = start + offsets
    mu_raw, distance = solar_position(
        times, station.latitude, station.longitude, clip_below_horizon=False
    )
    mu = np.clip(mu_raw, 0.0, None)
    clear_ghi = float(haurwitz_clear_sky_ghi(mu, distance).mean())
    day = bool((mu_raw > 0.0).any())
    doy = valid_time.timetuple().tm_yday
    angle = 2.0 * np.pi * (doy - 1) / 365.25
    values = [float(mu.mean()), clear_ghi]
    names = ["cos_sza", "clear_ghi_wm2"]
    if doy_mode != "none":
        sign = -1.0 if doy_mode == "hemisphere" and station.latitude < 0.0 else 1.0
        values.extend([sign * math.sin(angle), sign * math.cos(angle)])
        suffix = "season" if doy_mode == "hemisphere" else "doy"
        names.extend([f"sin_{suffix}", f"cos_{suffix}"])
    lat_radians = math.radians(station.latitude)
    lon_radians = math.radians(station.longitude)
    values.extend([
        math.sin(lat_radians), math.cos(lat_radians),
        math.sin(lon_radians), math.cos(lon_radians),
        station.elevation_m / 1000.0, lead / 72.0,
    ])
    names.extend([
        "sin_latitude", "cos_latitude", "sin_longitude", "cos_longitude",
        "elevation_km", "lead_fraction_72h",
    ])
    if aod550 is not None:
        values.append(float(aod550))
        names.append("aod550")
    return np.asarray(values, dtype=np.float32), names, day


def load_aod(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    result: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = utc_text(parse_utc(row["valid_time_utc"]))
            value = float(row["aod550"])
            if not 0.0 <= value <= 5.0:
                raise ValueError(f"AOD550 outside [0,5]: {row}")
            result[(row["station_id"], timestamp)] = value
    return result


def load_label_cache(
    path: Path | None,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None] | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "station_id", "valid_time_utc", "target", "target_mask",
            "observation_available",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"label cache is missing arrays: {sorted(missing)}")
        stations = archive["station_id"].astype(str)
        valid_times = archive["valid_time_utc"].astype(str)
        targets = archive["target"].astype(np.float32)
        masks = archive["target_mask"].astype(bool)
        available = archive["observation_available"].astype(bool)
    result: dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None] = {}
    for station, valid_time, target, mask, is_available in zip(
        stations, valid_times, targets, masks, available
    ):
        key = (str(station), str(valid_time))
        if key in result:
            raise ValueError(f"duplicate label-cache key: {key}")
        result[key] = (target, mask) if bool(is_available) else None
    return result


def split_by_init(records: list[GfsRecord], split_map_path: Path | None) -> dict[str, str]:
    init_times = sorted({record.init_time for record in records})
    if split_map_path is not None:
        payload = json.loads(split_map_path.read_text(encoding="utf-8"))
        mapping = {
            datetime.strptime(value, "%Y%m%d%H").replace(tzinfo=UTC): split
            for split in ("train", "val", "test")
            for value in payload.get(split, [])
        }
        missing = set(init_times) - set(mapping)
        if missing:
            raise ValueError(f"split map omits init times: {[utc_text(x) for x in sorted(missing)]}")
        return {utc_text(key): value for key, value in mapping.items()}

    mapping: dict[datetime, str] = {}
    test_dates = {"20210323", "20230715"}
    by_month: dict[int, list[datetime]] = defaultdict(list)
    for init_time in init_times:
        if init_time.strftime("%Y%m%d") in test_dates:
            mapping[init_time] = "test"
        else:
            by_month[init_time.month].append(init_time)
    for values in by_month.values():
        values.sort()
        for index, init_time in enumerate(values):
            mapping[init_time] = "val" if len(values) > 1 and index == len(values) - 1 else "train"
    return {utc_text(key): value for key, value in mapping.items()}


def _normalization(arrays: dict[str, list[np.ndarray]], pressure_levels_hpa: list[int]) -> dict[str, np.ndarray]:
    atmos = np.stack(arrays["atmos"]).astype(np.float64)
    surface = np.stack(arrays["surface"]).astype(np.float64)
    surface_grid = np.stack(arrays["surface_grid"]).astype(np.float64)
    sp_grid = surface_grid[:, 1]
    layer_mask = (
        np.asarray(pressure_levels_hpa).reshape(1, 1, 1, -1) * 100.0
        <= sp_grid[..., None]
    )
    return {
        "atmos_mean": np.asarray(
            [atmos[:, channel][layer_mask].mean() for channel in range(atmos.shape[1])],
            dtype=np.float32,
        ),
        "atmos_std": np.asarray(
            [max(atmos[:, channel][layer_mask].std(), 1e-8) for channel in range(atmos.shape[1])],
            dtype=np.float32,
        ),
        "surface_mean": surface.mean(axis=0).astype(np.float32),
        "surface_std": np.maximum(surface.std(axis=0), 1e-8).astype(np.float32),
    }


def build(args) -> dict:
    schema, schema_digest = load_gfs_schema(args.schema)
    stations = load_stations(args.stations)
    records = discover_records(args.gfs_root, stations)
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise ValueError(f"no GFS records found below {args.gfs_root}")
    split_map = split_by_init(records, args.split_map)
    aod = load_aod(args.aod_csv)
    label_cache = load_label_cache(args.label_cache)
    grouped: dict[str, list[GfsRecord]] = defaultdict(list)
    for record in records:
        grouped[record.station.station_id].append(record)

    keys = [
        "atmos", "surface", "surface_grid", "solar", "target", "target_mask",
        "day_mask", "time", "station_id", "init_time_utc", "forecast_hour",
        "gfs_dswrf",
    ]
    buckets = {split: {key: [] for key in keys} for split in ("train", "val", "test")}
    dropped = defaultdict(int)
    solar_names: list[str] | None = None
    include_aod = args.aod_csv is not None

    for station_id, station_records in sorted(grouped.items()):
        station = stations[station_id]
        observations = (
            load_observation_minutes(
                station, [record.valid_time for record in station_records], args.observations
            )
            if label_cache is None
            else None
        )
        for record in station_records:
            valid_text = utc_text(record.valid_time)
            if label_cache is None:
                aggregate = aggregate_hour(
                    observations, record.valid_time, args.minimum_valid_minutes
                )
            else:
                cache_key = (station_id, valid_text)
                if cache_key not in label_cache:
                    raise ValueError(f"label cache omits {cache_key}")
                cached = label_cache[cache_key]
                aggregate = None if cached is None else (cached[0], cached[1], {})
            aod_value = aod.get((station_id, valid_text))
            if include_aod and aod_value is None:
                dropped["missing_aod"] += 1
                continue
            solar, names, day = solar_features(
                record.valid_time, station, record.forecast_hour, args.doy_mode, aod_value
            )
            if solar_names is None:
                solar_names = names
            elif solar_names != names:
                raise ValueError(f"inconsistent solar features: {names} != {solar_names}")
            if aggregate is None:
                # Missing station observations must not remove the corresponding
                # NWP input from a 72-hour business trajectory.  Retain the
                # meteorological sample and let target_mask exclude the hour
                # from supervised losses.
                target = np.zeros(3, dtype=np.float32)
                target_mask = np.zeros(3, dtype=bool)
                dropped["insufficient_ghi_label_minutes"] += 1
            else:
                target, target_mask, _ = aggregate
            if not day:
                if target_mask[0] and target[0] > args.night_ghi_tolerance:
                    raise ValueError(
                        f"{station_id} {valid_text}: geometric night GHI={target[0]:.2f} "
                        "W m-2; probable UTC or interval-alignment error"
                    )
                target[:] = 0.0
                target_mask[:] = True
            with np.load(record.path, allow_pickle=False) as sample:
                metadata = json.loads(str(sample["metadata"].item()))
                expected_init = utc_text(record.init_time)
                if metadata.get("init_time_utc") != expected_init:
                    raise ValueError(f"{record.path}: init_time_utc mismatch")
                if int(metadata.get("forecast_hour", -1)) != record.forecast_hour:
                    raise ValueError(f"{record.path}: forecast_hour mismatch")
                stored_valid = metadata.get("valid_time_utc")
                if stored_valid is None and not args.allow_legacy_missing_valid_time:
                    raise ValueError(
                        f"{record.path}: missing valid_time_utc; rebuild or migrate the GFS sample"
                    )
                if stored_valid is not None and utc_text(parse_utc(stored_valid)) != valid_text:
                    raise ValueError(f"{record.path}: stored valid_time_utc mismatch")
                atmos = sample["atmos"].astype(np.float32)
                surface_grid = sample["surface_grid"].astype(np.float32)
                gfs_dswrf = float(sample["gfs_dswrf"][2, 2])
            split = split_map[expected_init]
            bucket = buckets[split]
            bucket["atmos"].append(atmos)
            bucket["surface_grid"].append(surface_grid)
            bucket["surface"].append(surface_grid[:, 2, 2])
            bucket["solar"].append(solar)
            bucket["target"].append(target)
            bucket["target_mask"].append(target_mask)
            bucket["day_mask"].append(day)
            bucket["time"].append(valid_text)
            bucket["station_id"].append(station_id)
            bucket["init_time_utc"].append(expected_init)
            bucket["forecast_hour"].append(record.forecast_hour)
            bucket["gfs_dswrf"].append(gfs_dswrf)

    if not buckets["train"]["target"]:
        raise ValueError("training split is empty after label/AOD quality control")
    args.output.mkdir(parents=True, exist_ok=True)
    for split, arrays in buckets.items():
        if not arrays["target"]:
            continue
        order = np.argsort(np.asarray(arrays["time"]))
        payload = {
            "atmos": np.stack(arrays["atmos"])[order],
            "surface": np.stack(arrays["surface"])[order],
            "surface_grid": np.stack(arrays["surface_grid"])[order],
            "solar": np.stack(arrays["solar"])[order],
            "target": np.stack(arrays["target"])[order],
            "target_mask": np.stack(arrays["target_mask"])[order],
            "day_mask": np.asarray(arrays["day_mask"], dtype=bool)[order],
            "time": np.asarray(arrays["time"])[order],
            "station_id": np.asarray(arrays["station_id"])[order],
            "init_time_utc": np.asarray(arrays["init_time_utc"])[order],
            "forecast_hour": np.asarray(arrays["forecast_hour"], dtype=np.int16)[order],
            "gfs_dswrf": np.asarray(arrays["gfs_dswrf"], dtype=np.float32)[order],
        }
        metadata = {
            "schema_version": schema["schema_version"],
            "schema_sha256": schema_digest,
            "atmos_names": schema["atmosphere"]["canonical_names"],
            "surface_names": schema["surface"]["canonical_names"],
            "solar_names": solar_names,
            "target_names": ["ghi", "dni", "dhi"],
            "pressure_levels_hpa": schema["atmosphere"]["level_profiles_hpa"][
                schema["atmosphere"]["active_level_profile"]
            ],
            "time_basis": "UTC",
            "observation_window": "[valid_time_utc-1h, valid_time_utc)",
            "doy_mode": args.doy_mode,
            "aod_policy": "explicit_aod550" if include_aod else "not_available_cloud_driven_claim_only",
        }
        payload["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        np.savez_compressed(args.output / f"{split}.npz", **payload)

    levels = schema["atmosphere"]["level_profiles_hpa"][schema["atmosphere"]["active_level_profile"]]
    stats = _normalization(buckets["train"], levels)
    np.savez_compressed(args.output / "norm_stats.npz", **stats)
    manifest = {
        "schema_version": schema["schema_version"],
        "canonical_schema_sha256": schema_digest,
        "schema": {
            "atmos_names": schema["atmosphere"]["canonical_names"],
            "surface_names": schema["surface"]["canonical_names"],
            "solar_names": solar_names,
            "target_names": ["ghi", "dni", "dhi"],
            "pressure_levels_hpa": levels,
        },
        "split_counts": {
            split: len(arrays["target"]) for split, arrays in buckets.items()
        },
        "split_by_init_time_utc": split_map,
        "dropped": dict(dropped),
        "time_contract": {
            "timezone": "UTC",
            "valid_time": "init_time_utc + forecast_hour",
            "observation_window": "[valid_time_utc-1h, valid_time_utc)",
            "minimum_valid_minutes": args.minimum_valid_minutes,
        },
        "doy_mode": args.doy_mode,
        "aod_policy": "explicit_aod550" if include_aod else "not_available_cloud_driven_claim_only",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build UTC-aligned global GFS SWRadNet data")
    parser.add_argument("--schema", type=Path, default=Path("configs/gfs_swradnet_schema.json"))
    parser.add_argument("--stations", type=Path, default=Path("configs/gfs_swradnet_stations.csv"))
    parser.add_argument("--gfs-root", type=Path, default=Path("data/gfs_swradnet/station_windows"))
    parser.add_argument(
        "--observations", type=Path, default=Path("data/radiation_observations/raw")
    )
    parser.add_argument(
        "--label-cache",
        type=Path,
        help="Compact pre-aggregated station labels; avoids uploading raw observations",
    )
    parser.add_argument("--output", type=Path, default=Path("data/gfs_swradnet/training"))
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--doy-mode", choices=["hemisphere", "raw", "none"], default="hemisphere")
    parser.add_argument(
        "--aod-csv", type=Path,
        help="Optional CSV with station_id,valid_time_utc,aod550; missing matches are dropped",
    )
    parser.add_argument("--minimum-valid-minutes", type=int, default=54)
    parser.add_argument("--night-ghi-tolerance", type=float, default=20.0)
    parser.add_argument("--max-records", type=int, default=0, help="Deterministic smoke-test limit")
    parser.add_argument(
        "--allow-legacy-missing-valid-time", action="store_true",
        help="Temporary migration aid; formal datasets must not use this flag",
    )
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
