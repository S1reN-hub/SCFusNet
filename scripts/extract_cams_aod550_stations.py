#!/usr/bin/env python3
"""Extract station-interpolated AOD550 from downloaded CAMS forecast archives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr


ARCHIVE_PATTERN = re.compile(
    r"^cams_aod550_(?P<date>\d{8})_12z_f013-f084(?:_n72)?\.nc\.zip$"
)
EXPECTED_LEADS = np.arange(13, 85, dtype=np.int64)
OUTPUT_FIELDS = [
    "station_id",
    "gfs_init_time_utc",
    "forecast_hour",
    "valid_time_utc",
    "aod_init_time_utc",
    "aod_lead_hours",
    "aod550",
    "aod_source",
]


def utc_text(value: datetime | np.datetime64) -> str:
    if isinstance(value, np.datetime64):
        nanoseconds = value.astype("datetime64[ns]").astype(np.int64)
        value = datetime.fromtimestamp(nanoseconds / 1e9, tz=UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def load_stations(path: Path) -> list[dict[str, object]]:
    stations: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            station_id = row["station_id"].strip()
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            if not station_id:
                raise ValueError("Empty station_id")
            if not -90.0 <= latitude <= 90.0:
                raise ValueError(f"Invalid latitude for {station_id}")
            if not -180.0 <= longitude <= 180.0:
                raise ValueError(f"Invalid longitude for {station_id}")
            stations.append(
                {
                    "station_id": station_id,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
    if not stations:
        raise ValueError("No stations found")
    if len({item["station_id"] for item in stations}) != len(stations):
        raise ValueError("Duplicate station IDs")
    return stations


def archives_below(
    raw_root: Path,
    gfs_dates: set[str] | None = None,
) -> list[Path]:
    archives = sorted(raw_root.glob("*_12z/cams_aod550_*_12z_f013-f084.nc.zip"))
    if not archives:
        raise ValueError(f"No CAMS AOD550 archives found below {raw_root}")
    selected: list[Path] = []
    selected_dates: set[str] = set()
    for archive in archives:
        match = ARCHIVE_PATTERN.match(archive.name)
        if not match:
            raise ValueError(f"Unexpected CAMS archive name: {archive}")
        cams_date = datetime.strptime(match.group("date"), "%Y%m%d")
        gfs_date = (cams_date + timedelta(days=1)).strftime("%Y%m%d")
        if gfs_dates is None or gfs_date in gfs_dates:
            selected.append(archive)
            selected_dates.add(gfs_date)
    if gfs_dates is not None:
        missing = sorted(gfs_dates - selected_dates)
        if missing:
            raise ValueError(
                "Missing CAMS archives for GFS dates: " + ", ".join(missing)
            )
    if not selected:
        raise ValueError("No CAMS AOD550 archives matched the requested GFS dates")
    return selected


def open_archive(archive: Path, temporary_root: Path) -> tuple[xr.Dataset, Path]:
    temporary_root.mkdir(parents=True, exist_ok=True)
    extraction_dir = Path(
        tempfile.mkdtemp(prefix=archive.stem + ".", dir=temporary_root)
    )
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError(f"Corrupt ZIP member in {archive}")
        members = [item for item in zipped.infolist() if not item.is_dir()]
        if len(members) != 1 or not members[0].filename.lower().endswith(".nc"):
            raise ValueError(f"Expected one NetCDF member in {archive}")
        zipped.extract(members[0], extraction_dir)
        netcdf_path = extraction_dir / members[0].filename
    return xr.open_dataset(netcdf_path), extraction_dir


def remove_tree(path: Path) -> None:
    # All targets are created by tempfile.mkdtemp directly below temporary_root.
    for child in path.iterdir():
        if child.is_dir():
            remove_tree(child)
            child.rmdir()
        else:
            child.unlink()
    path.rmdir()


def validate_dataset(ds: xr.Dataset, archive: Path) -> tuple[datetime, np.ndarray]:
    if "aod550" not in ds:
        raise ValueError(f"aod550 missing from {archive}")
    required_coords = {
        "forecast_period",
        "forecast_reference_time",
        "latitude",
        "longitude",
    }
    missing = required_coords - set(ds.coords)
    if missing:
        raise ValueError(f"Missing coordinates in {archive}: {sorted(missing)}")
    reference_values = ds["forecast_reference_time"].values.reshape(-1)
    if reference_values.size != 1:
        raise ValueError(f"Expected one forecast reference time in {archive}")
    reference_ns = reference_values[0].astype("datetime64[ns]").astype(np.int64)
    reference = datetime.fromtimestamp(reference_ns / 1e9, tz=UTC)
    if reference.hour != 12:
        raise ValueError(f"Expected CAMS 12Z reference time in {archive}")

    periods = (
        ds["forecast_period"].values.astype("timedelta64[h]").astype(np.int64)
    )
    if not np.array_equal(periods, EXPECTED_LEADS):
        raise ValueError(f"Expected CAMS leads 13..84 in {archive}")
    expected_valid = np.asarray(
        [
            np.datetime64(reference.replace(tzinfo=None) + timedelta(hours=int(lead)))
            for lead in periods
        ],
        dtype="datetime64[ns]",
    )
    # Some ADS NetCDF encodings omit the derived ``valid_time`` coordinate.
    # The authoritative forecast axis is still fully specified by the single
    # reference time plus forecast_period. Validate valid_time when present,
    # otherwise use the derived values below.
    if "valid_time" in ds:
        actual_valid = ds["valid_time"].values.reshape(-1).astype("datetime64[ns]")
        if not np.array_equal(actual_valid, expected_valid):
            raise ValueError(f"valid_time mismatch in {archive}")
    return reference, periods


def extract_archive(
    archive: Path,
    stations: list[dict[str, object]],
    temporary_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    ds, extraction_dir = open_archive(archive, temporary_root)
    try:
        reference, leads = validate_dataset(ds, archive)
        station_ids = [str(item["station_id"]) for item in stations]
        station_lats = xr.DataArray(
            [float(item["latitude"]) for item in stations],
            dims="station",
            coords={"station": station_ids},
        )
        longitude_values = np.asarray(ds["longitude"].values, dtype=np.float64)
        if float(np.nanmin(longitude_values)) >= 0.0:
            station_longitudes = [float(item["longitude"]) % 360.0 for item in stations]
        else:
            station_longitudes = [
                ((float(item["longitude"]) + 180.0) % 360.0) - 180.0
                for item in stations
            ]
        station_lons = xr.DataArray(
            station_longitudes,
            dims="station",
            coords={"station": station_ids},
        )
        values = (
            ds["aod550"]
            .squeeze("forecast_reference_time", drop=True)
            .interp(latitude=station_lats, longitude=station_lons, method="linear")
            .transpose("forecast_period", "station")
            .load()
            .values
        )
        if values.shape != (72, len(stations)):
            raise ValueError(f"Unexpected interpolated shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite station AOD550 values in {archive}")
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum < 0.0 or maximum > 5.0:
            raise ValueError(
                f"AOD550 outside [0,5] in {archive}: min={minimum}, max={maximum}"
            )

        gfs_init = reference + timedelta(hours=12)
        rows: list[dict[str, object]] = []
        for lead_index, aod_lead in enumerate(leads):
            forecast_hour = int(aod_lead) - 12
            valid_time = reference + timedelta(hours=int(aod_lead))
            for station_index, station_id in enumerate(station_ids):
                rows.append(
                    {
                        "station_id": station_id,
                        "gfs_init_time_utc": utc_text(gfs_init),
                        "forecast_hour": forecast_hour,
                        "valid_time_utc": utc_text(valid_time),
                        "aod_init_time_utc": utc_text(reference),
                        "aod_lead_hours": int(aod_lead),
                        "aod550": f"{float(values[lead_index, station_index]):.8f}",
                        "aod_source": "CAMS_GLOBAL_FORECAST",
                    }
                )
        audit = {
            "archive": str(archive),
            "cams_init_time_utc": utc_text(reference),
            "gfs_init_time_utc": utc_text(gfs_init),
            "lead_count": len(leads),
            "station_count": len(stations),
            "row_count": len(rows),
            "aod550_min": minimum,
            "aod550_max": maximum,
        }
        return rows, audit
    finally:
        ds.close()
        remove_tree(extraction_dir)


def atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv",
        prefix=path.stem + ".",
        dir=path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=path.stem + ".",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate downloaded CAMS AOD550 forecasts to SWRadNet stations"
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/aod/cams/raw")
    )
    parser.add_argument(
        "--stations",
        type=Path,
        default=Path("configs/gfs_swradnet_stations.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/aod/cams/processed/cams_aod550_station_hour.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/aod/cams/reports/extraction_audit.json"),
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        default=Path("data/aod/cams/tmp"),
    )
    parser.add_argument(
        "--gfs-dates",
        nargs="+",
        metavar="YYYYMMDD",
        help="Only extract archives aligned to these GFS 00Z init dates",
    )
    args = parser.parse_args()

    selected_dates: set[str] | None = None
    if args.gfs_dates:
        selected_dates = set(args.gfs_dates)
        for value in selected_dates:
            try:
                parsed = datetime.strptime(value, "%Y%m%d")
            except ValueError as exc:
                raise ValueError(f"Invalid --gfs-dates value: {value}") from exc
            if parsed.strftime("%Y%m%d") != value:
                raise ValueError(f"Invalid --gfs-dates value: {value}")

    stations = load_stations(args.stations)
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for archive in archives_below(args.raw_root, selected_dates):
        archive_rows, audit = extract_archive(
            archive, stations, args.temporary_root
        )
        rows.extend(archive_rows)
        audits.append(audit)
        print(
            f"{archive.name}: {audit['row_count']} rows, "
            f"AOD550 {audit['aod550_min']:.4f}..{audit['aod550_max']:.4f}"
        )

    keys = [
        (
            str(row["station_id"]),
            str(row["gfs_init_time_utc"]),
            int(row["forecast_hour"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate station/GFS-init/forecast-hour rows")
    rows.sort(
        key=lambda row: (
            str(row["gfs_init_time_utc"]),
            str(row["station_id"]),
            int(row["forecast_hour"]),
        )
    )
    atomic_write_csv(args.output, rows)
    atomic_write_json(
        args.report,
        {
            "source": "CAMS global atmospheric composition forecasts",
            "variable": "total_aerosol_optical_depth_550nm",
            "cycle_policy": "previous_day_12z",
            "archive_count": len(audits),
            "station_count": len(stations),
            "row_count": len(rows),
            "archives": audits,
        },
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
