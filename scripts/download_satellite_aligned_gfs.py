#!/usr/bin/env python3
"""Download GOES station windows aligned to GFS forecasts or pre-init history.

The GFS sample identity is ``init_time + forecast_hour``.  This program
discovers the actual init dates already present under the GFS station-window
tree, expands them to unique valid times or causal pre-initialization history,
downloads each GOES full-disk scan once, extracts every configured GOES-East
or GOES-West station, writes resumable station-hour NPZ files, then removes
only that scan's files from a dedicated cache.

Himawari and Meteosat stations are included in the alignment manifest but are
not downloaded by this GOES-specific implementation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from scripts.download_satellite_cloud import (
        GOES_PRODUCTS,
        GOES_PLATFORMS,
        aggregate_hourly_scans,
        approximate_solar_position,
        extract_window_channels,
        goes_product_specs,
        list_product_scans,
        load_scan_bundle,
        load_stations,
        lonlat_to_goes_scan_angles,
        platform_for_datetime,
        save_hourly_sample,
        synchronize_scans,
    )
except ImportError:  # Direct execution from the scripts directory.
    from download_satellite_cloud import (
        GOES_PRODUCTS,
        GOES_PLATFORMS,
        aggregate_hourly_scans,
        approximate_solar_position,
        extract_window_channels,
        goes_product_specs,
        list_product_scans,
        load_scan_bundle,
        load_stations,
        lonlat_to_goes_scan_angles,
        platform_for_datetime,
        save_hourly_sample,
        synchronize_scans,
    )


GFS_RUN_RE = re.compile(r"^(?P<date>\d{8})_(?P<cycle>\d{2})z$")


def parse_int_range(value: str) -> list[int]:
    values: set[int] = set()
    for token in (part.strip() for part in value.split(",")):
        if not token:
            continue
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    if not values or min(values) < 1:
        raise ValueError("forecast leads must be positive integers")
    return sorted(values)


def discover_gfs_runs(gfs_root: Path) -> list[tuple[str, str]]:
    """Discover init runs from the actual per-station GFS output tree."""
    runs: set[tuple[str, str]] = set()
    if not gfs_root.exists():
        return []
    for station_dir in gfs_root.iterdir():
        if not station_dir.is_dir():
            continue
        for run_dir in station_dir.iterdir():
            if not run_dir.is_dir():
                continue
            match = GFS_RUN_RE.match(run_dir.name)
            if match:
                runs.add((match.group("date"), match.group("cycle")))
    return sorted(runs)


def explicit_runs(dates: str, cycle: str) -> list[tuple[str, str]]:
    runs = []
    for token in (item.strip() for item in dates.split(",")):
        if token:
            datetime.strptime(token, "%Y%m%d")
            runs.append((token, cycle.zfill(2)))
    return sorted(set(runs))


def build_valid_time_links(
    runs: Sequence[tuple[str, str]], leads: Sequence[int]
) -> dict[datetime, list[dict[str, object]]]:
    links: dict[datetime, list[dict[str, object]]] = defaultdict(list)
    for date, cycle in runs:
        init_time = datetime.strptime(f"{date}{cycle}", "%Y%m%d%H")
        for lead in leads:
            links[init_time + timedelta(hours=lead)].append({
                "init_time_utc": init_time.isoformat(),
                "forecast_hour": int(lead),
            })
    return dict(sorted(links.items()))


def build_history_time_links(
    runs: Sequence[tuple[str, str]], history_ages: Sequence[int]
) -> dict[datetime, list[dict[str, object]]]:
    """Return only information available no later than each GFS initialization."""
    links: dict[datetime, list[dict[str, object]]] = defaultdict(list)
    for date, cycle in runs:
        init_time = datetime.strptime(f"{date}{cycle}", "%Y%m%d%H")
        for age in sorted(set(history_ages), reverse=True):
            links[init_time - timedelta(hours=age)].append({
                "init_time_utc": init_time.isoformat(),
                "satellite_history_age_hours": int(age),
                "role": "causal_pre_init_history",
            })
    return dict(sorted(links.items()))


def merge_time_links(
    *tables: Mapping[datetime, Sequence[Mapping[str, object]]],
) -> dict[datetime, list[dict[str, object]]]:
    merged: dict[datetime, list[dict[str, object]]] = defaultdict(list)
    for table in tables:
        for valid_time, links in table.items():
            merged[valid_time].extend(dict(link) for link in links)
    return dict(sorted(merged.items()))


def load_satellite_groups(
    station_config: Path, satellite_schema: Path
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    stations = load_stations(station_config)
    with satellite_schema.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    groups: dict[str, list[str]] = defaultdict(list)
    for station_id, mapping in schema["station_satellite_mapping"].items():
        if station_id in stations:
            groups[str(mapping["satellite"])].append(station_id)
    return stations, {name: sorted(ids) for name, ids in groups.items()}


def hourly_output_path(output_root: Path, station_id: str, valid_time: datetime) -> Path:
    return output_root / station_id / f"{valid_time.year:04d}" / f"{valid_time.month:02d}" / f"{valid_time:%Y%m%d_%H%M}.npz"


def output_is_complete(path: Path, valid_time: datetime) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            return (
                metadata.get("status") == "SUCCESS"
                and metadata.get("valid_time") == valid_time.isoformat()
                and "scan_windows" in payload
                and "scan_validity" in payload
            )
    except Exception:
        return False


def active_daylight_stations(
    stations: Sequence[Mapping[str, object]], valid_time: datetime, max_solar_zenith: float
) -> list[Mapping[str, object]]:
    midpoint = valid_time - timedelta(minutes=30)
    active = []
    for station in stations:
        zenith, _ = approximate_solar_position(
            midpoint, float(station["latitude"]), float(station["longitude"])
        )
        if zenith <= max_solar_zenith:
            active.append(station)
    return active


def _safe_remove_bundle_cache(cache_dir: Path, bundle: Mapping[str, object]) -> int:
    """Remove only files named by one bundle from the dedicated cache root."""
    root = cache_dir.resolve()
    removed = 0
    for key in bundle["files"].values():
        target = (cache_dir / Path(str(key)).name).resolve()
        target.relative_to(root)
        for candidate in (target, target.with_suffix(target.suffix + ".part")):
            if candidate.exists() and candidate.is_file():
                candidate.unlink()
                removed += 1
    return removed


def process_goes_hour(
    fs,
    stations: Sequence[Mapping[str, object]],
    valid_time: datetime,
    gfs_links: Sequence[Mapping[str, object]],
    output_root: Path,
    cache_dir: Path,
    half_win: int,
    keep_cache: bool,
    max_scans_per_hour: int = 0,
    platform_name: str | None = None,
    sector: str = "full",
    products: Sequence[str] | None = None,
    maximum_lookback_hours: int = 0,
) -> dict[str, object]:
    platform = (
        GOES_PLATFORMS[platform_name]
        if platform_name is not None
        else platform_for_datetime(valid_time)
    )
    pending = [
        station for station in stations
        if not output_is_complete(hourly_output_path(output_root, str(station["id"]), valid_time), valid_time)
    ]
    if not pending:
        return {"status": "SKIP", "saved": 0, "failed": 0, "bundles": 0}

    start, end = valid_time - timedelta(hours=1), valid_time
    product_specs = goes_product_specs(sector, products)
    source_start, source_end = start, end
    bundles = []
    for lookback_hours in range(maximum_lookback_hours + 1):
        source_start = start - timedelta(hours=lookback_hours)
        source_end = end - timedelta(hours=lookback_hours)
        scans = {
            name: list_product_scans(fs, platform, spec, source_start, source_end)
            for name, spec in product_specs.items()
        }
        bundles = synchronize_scans(scans)
        if bundles:
            break
    if not bundles:
        return {"status": "NO_SCANS", "saved": 0, "failed": len(pending), "bundles": 0}
    if max_scans_per_hour > 0:
        # The latest complete scan is closest to the hourly interval end and
        # therefore minimizes the satellite age at forecast initialization.
        bundles = bundles[-max_scans_per_hour:]

    per_station: dict[str, list[dict[str, object]]] = {str(station["id"]): [] for station in pending}
    station_errors: dict[str, list[str]] = {str(station["id"]): [] for station in pending}
    scan_angles = {
        str(station["id"]): lonlat_to_goes_scan_angles(
            float(station["longitude"]), float(station["latitude"]), platform.longitude
        )
        for station in pending
    }
    removed = 0
    for bundle in bundles:
        try:
            with load_scan_bundle(fs, platform, bundle, cache_dir) as datasets:
                for station in pending:
                    station_id = str(station["id"])
                    try:
                        x_scan, y_scan = scan_angles[station_id]
                        window, validity, grid = extract_window_channels(
                            datasets, x_scan, y_scan, half_win
                        )
                        per_station[station_id].append({
                            "window": window,
                            "validity": validity,
                            "scan_time": bundle["anchor_time"].isoformat(),
                            "grid": grid,
                        })
                    except Exception as exc:
                        station_errors[station_id].append(
                            f"{bundle['anchor_time'].isoformat()}: {type(exc).__name__}: {exc}"
                        )
        except Exception as exc:
            message = f"{bundle['anchor_time'].isoformat()}: {type(exc).__name__}: {exc}"
            for station in pending:
                station_errors[str(station["id"])].append(message)
        finally:
            if not keep_cache:
                removed += _safe_remove_bundle_cache(cache_dir, bundle)

    saved = 0
    for station in pending:
        station_id = str(station["id"])
        results = per_station[station_id]
        if not results:
            continue
        zenith, azimuth = approximate_solar_position(
            start + timedelta(minutes=30), float(station["latitude"]), float(station["longitude"])
        )
        hourly = aggregate_hourly_scans(results, azimuth)
        hourly.update({
            "status": "SUCCESS",
            "platform": platform.name,
            "station_id": station_id,
            "valid_time": valid_time.isoformat(),
            "interval_start": start.isoformat(),
            "interval_end": end.isoformat(),
            "source_interval_start": source_start.isoformat(),
            "source_interval_end": source_end.isoformat(),
            "causal_staleness_minutes": int((end - source_end).total_seconds() // 60),
            "solar_zenith_deg": zenith,
            "solar_azimuth_deg": azimuth,
            "gfs_links": list(gfs_links),
            "errors": station_errors[station_id],
            "bundle_count": len(bundles),
            "scan_windows": np.stack([item["window"] for item in results]),
            "scan_validity": np.stack([item["validity"] for item in results]),
        })
        save_hourly_sample(hourly, station_id, output_root)
        saved += 1
    return {
        "status": "SUCCESS" if saved else "EXTRACTION_FAILED",
        "saved": saved,
        "failed": len(pending) - saved,
        "bundles": len(bundles),
        "cache_files_removed": removed,
        "errors": {
            station_id: messages[:3]
            for station_id, messages in station_errors.items()
            if messages
        },
    }


def process_goes_hour_job(job: Mapping[str, object]) -> tuple[datetime, dict[str, object]]:
    """Process one hour, holding transient S3/TLS failures instead of flooding retries.

    A short proxy or network outage used to make every already-submitted process-pool
    job fail immediately.  Retrying inside the worker keeps the queue bounded and
    recreates the S3 client for each attempt, which is important after a broken TLS
    connection.  Completed station-hour files remain resumable and are skipped by
    ``process_goes_hour``.
    """
    import s3fs

    valid_time = job["valid_time"]
    attempts = int(job.get("worker_attempts", 5))
    last_result: dict[str, object] | None = None
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            fs = s3fs.S3FileSystem(anon=True)
            result = process_goes_hour(
                fs,
                job["stations"],
                valid_time,
                job["gfs_links"],
                Path(job["output_root"]),
                Path(job["cache_dir"]),
                int(job["half_window"]),
                bool(job["keep_cache"]),
                int(job.get("max_scans_per_hour", 0)),
                str(job["platform_name"]) if job.get("platform_name") else None,
                str(job.get("sector", "full")),
                list(job.get("products", GOES_PRODUCTS)),
                int(job.get("maximum_lookback_hours", 0)),
            )
            last_result = result
            if int(result.get("failed", 0)) == 0:
                return valid_time, result
            # Repeating the same archive listings cannot repair a genuine
            # no-scan interval.  Return it immediately so one outage hour does
            # not hold a worker through the exponential network retry schedule.
            if result.get("status") == "NO_SCANS":
                return valid_time, result
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            # Deliberately wait longer than a brief network/proxy fluctuation.  The
            # cap prevents one bad archive hour from blocking the queue forever.
            time.sleep(min(15 * (2 ** (attempt - 1)), 120))
    if last_result is not None:
        return valid_time, last_result
    assert last_error is not None
    raise last_error


def write_alignment_manifest(
    path: Path,
    runs: Sequence[tuple[str, str]],
    leads: Sequence[int],
    valid_links: Mapping[datetime, Sequence[Mapping[str, object]]],
    groups: Mapping[str, Sequence[str]],
    *,
    history_hours: int = 0,
    history_ages: Sequence[int] = (),
    history_only: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "satellite_gfs_alignment_v1",
        "gfs_runs": [{"date": date, "cycle": cycle} for date, cycle in runs],
        "forecast_hours": list(leads),
        "satellite_history_hours": int(history_hours),
        "satellite_history_ages": [int(value) for value in history_ages],
        "history_only": bool(history_only),
        "valid_time_count": len(valid_links),
        "station_satellite_groups": groups,
        "download_support": {
            "GOES-East": "implemented_GOES19_current",
            "GOES-West": "implemented_GOES18_current",
            "Himawari-8": "manifest_only_requires_provider_access",
            "Meteosat-11": "manifest_only_requires_provider_access",
        },
        "records": [
            {"valid_time_utc": value.isoformat(), "gfs_links": list(links)}
            for value, links in valid_links.items()
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfs-root", type=Path, default=Path("data/gfs_swradnet/station_windows"))
    parser.add_argument("--init-dates", help="Optional comma-separated YYYYMMDD override")
    parser.add_argument(
        "--cycles-json",
        type=Path,
        help="Optional JSON containing init_times; avoids discovering or downloading new GFS runs",
    )
    parser.add_argument("--cycle", default="00")
    parser.add_argument("--leads", default="1-72")
    parser.add_argument(
        "--history-hours",
        type=int,
        default=0,
        help="Also download causal hourly frames ending at or before each GFS initialization",
    )
    parser.add_argument(
        "--history-ages",
        help="Optional comma-separated ages in hours, for example 3 or 3,2; overrides --history-hours",
    )
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Download only --history-hours frames and omit forecast-valid satellite times",
    )
    parser.add_argument("--station-config", type=Path, default=Path("configs/global_radiation_truth_stations.json"))
    parser.add_argument("--satellite-schema", type=Path, default=Path("configs/satellite_cloud_schema.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/satellite_cloud/gfs_aligned"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/satellite_cloud/cache/gfs_aligned_goes"))
    parser.add_argument("--manifest", type=Path, default=Path("data/satellite_cloud/gfs_alignment_manifest.json"))
    parser.add_argument("--half-window", type=int, default=31)
    parser.add_argument("--max-solar-zenith", type=float, default=90.0)
    parser.add_argument(
        "--sector",
        choices=("full", "conus"),
        default="full",
        help="Use compact CONUS products for US stations to reduce transfer volume.",
    )
    parser.add_argument(
        "--products",
        default=",".join(GOES_PRODUCTS),
        help="Comma-separated products; CloudMask is mandatory.",
    )
    parser.add_argument(
        "--include-night",
        action="store_true",
        help=(
            "Retain night-time all-sky cloud-mask/cloud-top frames. "
            "Sunlight-dependent COD/CPS channels remain explicitly masked."
        ),
    )
    parser.add_argument("--only-valid-time", help="Run one UTC hour, e.g. 20250715T1800")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument(
        "--max-scans-per-hour",
        type=int,
        default=0,
        help="Keep only the latest N synchronized scans in each hour; zero keeps all scans",
    )
    parser.add_argument("--workers", type=int, default=1, help="Independent valid-time processes")
    parser.add_argument(
        "--maximum-failed",
        type=int,
        default=0,
        help="Allow at most this many missing station-hours after retries; masks retain missing frames",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    products = [value.strip() for value in args.products.split(",") if value.strip()]
    try:
        goes_product_specs(args.sector, products)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    leads = parse_int_range(args.leads)
    if args.history_hours < 0:
        raise SystemExit("--history-hours must be non-negative")
    if args.max_scans_per_hour < 0:
        raise SystemExit("--max-scans-per-hour must be non-negative")
    if args.maximum_failed < 0:
        raise SystemExit("--maximum-failed must be non-negative")
    if args.history_ages:
        try:
            history_ages = sorted(
                {int(value.strip()) for value in args.history_ages.split(",") if value.strip()},
                reverse=True,
            )
        except ValueError as error:
            raise SystemExit("--history-ages must contain non-negative integers") from error
        if not history_ages or min(history_ages) < 0:
            raise SystemExit("--history-ages must contain non-negative integers")
    else:
        history_ages = list(range(args.history_hours - 1, -1, -1))
    history_hours = max(history_ages, default=-1) + 1
    if args.history_only and not history_ages:
        raise SystemExit("--history-only requires --history-hours > 0")
    if args.init_dates and args.cycles_json:
        raise SystemExit("use only one of --init-dates and --cycles-json")
    init_dates = args.init_dates
    if args.cycles_json:
        payload = json.loads(args.cycles_json.read_text(encoding="utf-8"))
        values = payload.get("init_times", ())
        if not values:
            raise SystemExit("--cycles-json does not contain init_times")
        init_dates = ",".join(sorted({str(value)[:10].replace("-", "") for value in values}))
    runs = explicit_runs(init_dates, args.cycle) if init_dates else discover_gfs_runs(args.gfs_root)
    if not runs:
        raise SystemExit("no GFS init runs discovered; pass --init-dates or check --gfs-root")
    forecast_links = {} if args.history_only else build_valid_time_links(runs, leads)
    history_links = build_history_time_links(runs, history_ages)
    valid_links = merge_time_links(forecast_links, history_links)
    stations, groups = load_satellite_groups(args.station_config, args.satellite_schema)
    write_alignment_manifest(
        args.manifest,
        runs,
        leads,
        valid_links,
        groups,
        history_hours=history_hours,
        history_ages=history_ages,
        history_only=args.history_only,
    )
    print(f"manifest={args.manifest} runs={len(runs)} valid_times={len(valid_links)}")
    print("groups=" + ", ".join(f"{name}:{len(ids)}" for name, ids in sorted(groups.items())))
    if args.dry_run:
        return 0

    if args.only_valid_time:
        selected_time = datetime.strptime(args.only_valid_time, "%Y%m%dT%H%M")
        valid_links = {selected_time: valid_links.get(selected_time, [])}
        if not valid_links[selected_time]:
            raise SystemExit(f"{selected_time.isoformat()} is not a GFS valid time in the manifest")

    total_saved = total_failed = 0
    jobs = []
    for index, (valid_time, links) in enumerate(valid_links.items(), 1):
        # GOES-East must follow the historical operational transition
        # (GOES-16 before 2025-04-07, GOES-19 afterward). GOES-West is fixed
        # to GOES-18 for the supported archive interval.
        for group_name, platform_name in (("GOES-East", None), ("GOES-West", "GOES-18")):
            group_stations = [stations[station_id] for station_id in groups.get(group_name, [])]
            active = (
                group_stations
                if args.include_night or args.history_only
                else active_daylight_stations(group_stations, valid_time, args.max_solar_zenith)
            )
            if not active:
                continue
            jobs.append({
                "valid_time": valid_time,
                "gfs_links": links,
                "stations": active,
                "platform_name": platform_name,
                "output_root": str(args.output_root),
                "cache_dir": str(args.cache_dir) + "/" + (
                    platform_name.lower() if platform_name else "goes-east-auto"
                ),
                "half_window": args.half_window,
                "keep_cache": args.keep_cache,
                "max_scans_per_hour": args.max_scans_per_hour,
                "sector": args.sector,
                "products": products,
                "worker_attempts": 5,
            })

    def report(position: int, valid_time: datetime, result: Mapping[str, object]) -> None:
        nonlocal total_saved, total_failed
        total_saved += int(result["saved"])
        total_failed += int(result["failed"])
        print(
            f"[{position}/{len(jobs)}] {valid_time.isoformat()} {result['status']} "
            f"stations={result['saved']} failed={result['failed']} bundles={result['bundles']} "
            f"cache_removed={result.get('cache_files_removed', 0)}",
            flush=True,
        )
        if result.get("errors"):
            print("  errors=" + json.dumps(result["errors"], ensure_ascii=False), flush=True)

    if args.workers <= 1:
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        for position, job in enumerate(jobs, 1):
            valid_time = job["valid_time"]
            result = process_goes_hour(
                fs, job["stations"], valid_time, job["gfs_links"],
                Path(job["output_root"]), Path(job["cache_dir"]),
                args.half_window, args.keep_cache,
                args.max_scans_per_hour,
                str(job["platform_name"]) if job.get("platform_name") else None,
                str(job["sector"]),
                list(job["products"]),
            )
            report(position, valid_time, result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(process_goes_hour_job, job): job["valid_time"] for job in jobs}
            for position, future in enumerate(concurrent.futures.as_completed(future_map), 1):
                valid_time = future_map[future]
                try:
                    completed_time, result = future.result()
                except Exception as exc:
                    result = {
                        "status": "WORKER_FAILED", "saved": 0,
                        "failed": len(next(job["stations"] for job in jobs if job["valid_time"] == valid_time)),
                        "bundles": 0, "errors": {"worker": [f"{type(exc).__name__}: {exc}"]},
                    }
                    completed_time = valid_time
                report(position, completed_time, result)
    print(f"complete saved={total_saved} failed={total_failed} output={args.output_root}")
    return 0 if total_failed <= args.maximum_failed else 2


if __name__ == "__main__":
    sys.exit(main())
