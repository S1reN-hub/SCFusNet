#!/usr/bin/env python
"""Fill exact GridSat gaps for the 799-case causal satellite inventory.

The NOAA Azure mirror stores one global NetCDF file per three-hour scan.  This
downloader fetches each required scan once, extracts every required station and
hour from it, and then removes the global cache file unless ``--keep-cache`` is
set.  This avoids both the currently unstable THREDDS subset service and
duplicate downloads across stations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import requests


DEFAULT_STATIONS = {"CAB", "CNR", "PAY", "IZA", "GOB", "TAM", "RUN"}
CHANNELS = 7
CHANNEL_NAMES = (
    "COD", "cloud_fraction", "cloud_top_pressure_hpa", "CER_liq_um",
    "CER_ice_um", "ahi_band11_brightness", "gridsat_irwin_bt_k",
)
SOURCE_ROOTS = (
    # On the two project download hosts the Azure NODD route is consistently
    # faster than the AWS public bucket.  Keep AWS as a byte-identical
    # fallback, but do not pay its connection penalty on every normal scan.
    "https://noaacdr.blob.core.windows.net/gridsat-b1/data",
    "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data",
)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def causal_scan(frame_time: datetime) -> datetime:
    return frame_time.replace(
        hour=(frame_time.hour // 3) * 3, minute=0, second=0, microsecond=0
    )


def completed_output(path: Path, grid: int) -> bool:
    if not path.exists() or path.stat().st_size <= 128:
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            validity = payload["validity"]
            if metadata.get("status") != "SUCCESS":
                return False
            if validity.shape == (grid, grid, CHANNELS):
                return bool(validity[:, :, 6].any())
            # Preserve completed six-channel Himawari files when this script is
            # used only to fill AHI archive outages with GridSat.
            if validity.shape == (grid, grid, 6):
                return bool(validity[:, :, 5].any())
            return False
    except Exception:
        return False


def download_scan(scan: datetime, cache: Path) -> Path:
    name = scan.strftime("GRIDSAT-B1.%Y.%m.%d.%H.v02r01.nc")
    output = cache / f"{scan:%Y}" / f"{scan:%m}" / name
    if output.exists() and output.stat().st_size > 1_000_000:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    errors: list[str] = []
    completed = False
    for root in SOURCE_ROOTS:
        url = f"{root}/{scan.year}/{name}"
        for attempt in range(1, 6):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
            try:
                session = requests.Session()
                session.trust_env = os.environ.get(
                    "GRIDSAT_DOWNLOAD_NO_PROXY", ""
                ).lower() not in {"1", "true", "yes"}
                with session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(30, 300),
                ) as response:
                    response.raise_for_status()
                    # A compliant range response can be appended.  If a mirror
                    # ignores Range and returns 200, restart this attempt from
                    # byte zero rather than corrupting the object.
                    append = offset > 0 and response.status_code == 206
                    mode = "ab" if append else "wb"
                    base_size = offset if append else 0
                    content_range = response.headers.get("Content-Range", "")
                    total_size = 0
                    if "/" in content_range:
                        tail = content_range.rsplit("/", 1)[-1]
                        if tail.isdigit():
                            total_size = int(tail)
                    if total_size <= 0:
                        content_length = int(response.headers.get("Content-Length", "0"))
                        if content_length > 0:
                            total_size = base_size + content_length
                    with partial.open(mode) as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                session.close()
                actual = partial.stat().st_size if partial.exists() else 0
                if actual > 1_000_000 and (total_size <= 0 or actual == total_size):
                    completed = True
                    break
                errors.append(
                    f"empty or truncated response: {url} attempt={attempt} "
                    f"expected={total_size} actual={actual}"
                )
            except requests.RequestException as exc:
                errors.append(
                    f"requests {type(exc).__name__}: {url} attempt={attempt}: {exc}"
                )
        if completed:
            break
    if not completed:
        raise RuntimeError("all GridSat mirrors failed: " + "; ".join(errors))
    partial.replace(output)
    return output


def window_from_global(
    raw: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    latitude: float,
    longitude: float,
    grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    cy = int(np.abs(latitudes - latitude).argmin())
    cx = int(np.abs(longitudes - longitude).argmin())
    half = grid // 2
    y0, x0 = cy - half, cx - half
    y1, x1 = y0 + grid, x0 + grid
    if y0 < 0 or x0 < 0 or y1 > raw.shape[0] or x1 > raw.shape[1]:
        raise ValueError("requested GridSat window crosses the global grid edge")
    subset = np.asarray(raw[y0:y1, x0:x1], dtype=np.int32)
    if subset.shape != (grid, grid):
        raise ValueError(f"unexpected GridSat window shape: {subset.shape}")
    valid = subset != -31999
    values = subset.astype(np.float32) * 0.01 + 200.0
    valid &= (values >= 140.0) & (values <= 375.0)
    values[~valid] = np.nan
    return values, valid


def save_output(
    path: Path,
    station: dict[str, str],
    frame_time: datetime,
    scan: datetime,
    window: np.ndarray,
    valid: np.ndarray,
    links: list[dict[str, object]],
) -> None:
    channels = np.full((window.shape[0], window.shape[1], CHANNELS), np.nan, dtype=np.float32)
    validity = np.zeros_like(channels, dtype=bool)
    channels[:, :, 6] = window
    validity[:, :, 6] = valid
    metadata = {
        "station_id": station["station_id"],
        "frame_time": frame_time.isoformat(),
        "scan_time": scan.isoformat(),
        "status": "SUCCESS",
        "source": "NOAA GridSat-B1 v02r01 via NOAA Azure NODD mirror",
        "source_channel": "irwin_cdr",
        "target_channel": CHANNEL_NAMES[6],
        "scan_times": [scan.isoformat()],
        "channel_names": list(CHANNEL_NAMES),
        "channel_valid_fraction": [0.0] * 6 + [float(valid.mean())],
        "gfs_links": links,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            window_mean=channels,
            validity=validity,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def process_scan(job: dict[str, object]) -> dict[str, object]:
    scan = parse_utc(str(job["scan_time"]))
    cache = Path(str(job["cache"]))
    grid = int(job["grid"])
    keep_cache = bool(job["keep_cache"])
    items = list(job["items"])
    record: dict[str, object] = {
        "scan_time": scan.isoformat(),
        "requested_outputs": len(items),
        "saved": 0,
        "failed": 0,
        "errors": {},
    }
    source = None
    try:
        source = download_scan(scan, cache)
        with h5py.File(source, "r") as payload:
            latitudes = np.asarray(payload["lat"][:], dtype=np.float32)
            longitudes = np.asarray(payload["lon"][:], dtype=np.float32)
            raw = np.asarray(payload["irwin_cdr"][0, :, :], dtype=np.int16)
        station_windows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for item in items:
            station = dict(item["station"])
            station_id = station["station_id"]
            try:
                if station_id not in station_windows:
                    station_windows[station_id] = window_from_global(
                        raw, latitudes, longitudes,
                        float(station["latitude"]), float(station["longitude"]), grid,
                    )
                window, valid = station_windows[station_id]
                if not valid.any():
                    raise ValueError("GridSat station window has no valid pixels")
                save_output(
                    Path(str(item["output"])), station,
                    parse_utc(str(item["frame_time"])), scan, window, valid,
                    list(item["links"]),
                )
                record["saved"] = int(record["saved"]) + 1
            except Exception as exc:
                record["failed"] = int(record["failed"]) + 1
                record["errors"][str(item["output"])] = str(exc)
        record["status"] = "SUCCESS" if not record["failed"] else "PARTIAL"
    except Exception as exc:
        record["failed"] = len(items)
        record["status"] = "FAILED"
        record["error"] = str(exc)
    finally:
        if source is not None and source.exists() and not keep_cache:
            source.unlink()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument(
        "--station-ids",
        default=",".join(sorted(DEFAULT_STATIONS)),
        help="Comma-separated stations; may include Himawari stations for outage fallback",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--limit-scans", type=int)
    args = parser.parse_args()

    selected_stations = {
        value.strip() for value in args.station_ids.split(",") if value.strip()
    }
    with args.stations.open(encoding="utf-8-sig", newline="") as stream:
        stations = {
            row["station_id"]: row for row in csv.DictReader(stream)
            if row["station_id"] in selected_stations
        }
    if set(stations) != selected_stations:
        raise ValueError(f"unknown station IDs: {sorted(selected_stations - set(stations))}")
    with args.inventory.open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["station_id"] in selected_stations
        ]

    grouped_links: dict[str, list[dict[str, object]]] = defaultdict(list)
    rows_by_path: dict[str, dict[str, str]] = {}
    for row in rows:
        path = args.output_root / Path(row["relative_path"])
        if completed_output(path, args.grid):
            continue
        key = str(path.resolve())
        rows_by_path.setdefault(key, row)
        grouped_links[key].append(
            {
                "sample_id": row["sample_id"],
                "station_id": row["station_id"],
                "satellite_history_offset_hour": int(row["history_offset_hour"]),
                "role": "causal_random_issue_history",
            }
        )

    scan_items: dict[datetime, list[dict[str, object]]] = defaultdict(list)
    for key, row in rows_by_path.items():
        frame_time = parse_utc(row["history_time_utc"])
        scan_items[causal_scan(frame_time)].append(
            {
                "station": stations[row["station_id"]],
                "frame_time": frame_time.isoformat(),
                "output": key,
                "links": grouped_links[key],
            }
        )
    scans = sorted(scan_items)
    if args.limit_scans is not None:
        scans = scans[: args.limit_scans]
    jobs = [
        {
            "scan_time": scan.isoformat(),
            "items": scan_items[scan],
            "cache": str(args.cache),
            "grid": args.grid,
            "keep_cache": args.keep_cache,
        }
        for scan in scans
    ]
    print(json.dumps({
        "pending_scans": len(jobs),
        "pending_outputs": sum(len(job["items"]) for job in jobs),
        "workers": args.workers,
    }), flush=True)
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_scan, job): job for job in jobs}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(jobs)}] {result['scan_time']} {result['status']} "
                f"saved={result['saved']} failed={result['failed']}",
                flush=True,
            )
    manifest = {
        "schema_version": "gridsat_random_issue_799_v2",
        "source": "NOAA Azure NODD GridSat-B1 mirror",
        "scan_count": len(results),
        "saved": sum(int(item["saved"]) for item in results),
        "failed": sum(int(item["failed"]) for item in results),
        "results": sorted(results, key=lambda item: str(item["scan_time"])),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "schema_version", "scan_count", "saved", "failed"
    )}), flush=True)


if __name__ == "__main__":
    main()
