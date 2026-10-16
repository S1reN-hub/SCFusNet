#!/usr/bin/env python3
"""Download CAMS forecast AOD550 aligned with archived GFS SWRadNet samples.

Security properties:
- Credentials are read from an external text file and are never written to the
  repository, command output, manifests, or reports.
- The CDS URL stored beside each credential is deliberately ignored. CAMS
  requests always use the ADS endpoint.
- Account reports contain only a short SHA-256 fingerprint.

Operational alignment:
- GFS cycle: D 00 UTC, forecast hours 1..72.
- CAMS cycle: D-1 12 UTC, forecast lead hours 13..84.
This guarantees that the CAMS forecast was available before the GFS 00 UTC
forecast cycle under the documented CAMS delivery schedule.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import cdsapi
import requests


ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
DATASET = "cams-global-atmospheric-composition-forecasts"
VARIABLE = "total_aerosol_optical_depth_550nm"
GFS_INIT_PATTERN = re.compile(r"^(?P<date>\d{8})_00z$")
AUTH_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "invalid credentials",
    "authentication",
    "not agreed",
    "licence",
    "license",
    "terms and conditions",
)
REQUEST_MARKERS = (
    "400 client error",
    "404 client error",
    "invalid request",
    "dataset not found",
    "process not found",
    "no data is available",
)


@dataclass(frozen=True)
class Credential:
    index: int
    key: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:12]


def parse_credentials(path: Path) -> list[Credential]:
    """Parse repeated url/key YAML-like blocks without retaining source URLs."""
    if not path.is_file():
        raise FileNotFoundError(path)
    credentials: list[Credential] = []
    pending_url = False
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("url:"):
            pending_url = True
            continue
        if line.lower().startswith("key:"):
            if not pending_url:
                raise ValueError("Found key without a preceding url entry")
            key = line.split(":", 1)[1].strip()
            if not key:
                raise ValueError("Found an empty API key")
            credentials.append(Credential(index=len(credentials) + 1, key=key))
            pending_url = False
            continue
        raise ValueError("Credential file contains an unsupported non-empty line")
    if pending_url:
        raise ValueError("Credential file ends with a url entry without a key")
    if not credentials:
        raise ValueError("No credentials found")
    if len({item.key for item in credentials}) != len(credentials):
        raise ValueError("Credential file contains duplicate API keys")
    return credentials


def discover_gfs_init_times(gfs_root: Path) -> list[datetime]:
    """Discover GFS 00 UTC init directories that contain at least one NPZ."""
    if not gfs_root.is_dir():
        raise FileNotFoundError(gfs_root)
    found: set[datetime] = set()
    for init_dir in gfs_root.glob("*/*_00z"):
        match = GFS_INIT_PATTERN.match(init_dir.name)
        if match and any(init_dir.glob("f[0-9][0-9][0-9].npz")):
            found.add(
                datetime.strptime(match.group("date"), "%Y%m%d").replace(tzinfo=UTC)
            )
    if not found:
        raise ValueError(f"No GFS 00Z NPZ samples found below {gfs_root}")
    return sorted(found)


def parse_gfs_dates(values: Iterable[str]) -> list[datetime]:
    result: set[datetime] = set()
    for value in values:
        parsed = datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
        result.add(parsed)
    return sorted(result)


def cams_init_for_gfs(gfs_init: datetime) -> datetime:
    if gfs_init.tzinfo is None or gfs_init.utcoffset() is None:
        raise ValueError("GFS init time must be timezone-aware UTC")
    normalized = gfs_init.astimezone(UTC)
    if normalized.hour != 0 or normalized.minute != 0:
        raise ValueError("Expected a GFS 00 UTC init time")
    return normalized - timedelta(hours=12)


def build_request(
    gfs_init: datetime,
    *,
    probe: bool = False,
    area: list[float] | None = None,
    lead_hours: list[int] | None = None,
) -> dict[str, object]:
    cams_init = cams_init_for_gfs(gfs_init)
    leads = ["13"] if probe else [str(hour) for hour in (lead_hours or range(13, 85))]
    request: dict[str, object] = {
        "variable": [VARIABLE],
        "date": [cams_init.strftime("%Y-%m-%d")],
        "time": ["12:00"],
        "leadtime_hour": leads,
        "type": ["forecast"],
        "data_format": "netcdf_zip",
    }
    if probe:
        # A tiny regular-grid subset. The probe checks authentication, licence,
        # request syntax, and the ability to retrieve the selected variable.
        request["area"] = [1.0, 0.0, 0.0, 1.0]  # North, West, South, East
    elif area is not None:
        request["area"] = area
    return request


def target_path(raw_root: Path, gfs_init: datetime, lead_hours: list[int] | None = None) -> Path:
    cams_init = cams_init_for_gfs(gfs_init)
    directory = raw_root / cams_init.strftime("%Y%m%d_12z")
    leads = lead_hours or list(range(13, 85))
    suffix = f"f{min(leads):03d}-f{max(leads):03d}"
    if len(leads) != max(leads) - min(leads) + 1:
        suffix += f"_n{len(leads):02d}"
    return directory / f"cams_aod550_{cams_init:%Y%m%d}_12z_{suffix}.nc.zip"


def classify_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in AUTH_MARKERS):
        return "credential_or_licence"
    if any(marker in text for marker in REQUEST_MARKERS):
        return "request"
    return "transient"


def safe_error_summary(exc: BaseException) -> str:
    """Return a non-secret, bounded error summary for local reports."""
    text = " ".join(str(exc).split())
    text = re.sub(r"https?://\S+", "<url>", text)
    text = text[:300]
    return f"{type(exc).__name__}: {text}"


def is_valid_zip(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            return bool(members) and archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def make_client(
    credential: Credential,
    *,
    wait_until_complete: bool = True,
    retry_max: int = 2,
) -> cdsapi.Client:
    return cdsapi.Client(
        url=ADS_URL,
        key=credential.key,
        quiet=True,
        progress=False,
        retry_max=max(int(retry_max), 0),
        sleep_max=15,
        timeout=120,
        session=requests.Session(),
        wait_until_complete=wait_until_complete,
    )


def retrieve(
    credential: Credential,
    request: dict[str, object],
    target: Path,
    *,
    retry_max: int = 2,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    client = make_client(credential, retry_max=retry_max)
    try:
        result = client.retrieve(DATASET, request)
        result.download(str(partial))
    finally:
        client.session.close()
    if not is_valid_zip(partial):
        partial.unlink(missing_ok=True)
        raise ValueError("ADS response is not a valid non-empty ZIP archive")
    partial.replace(target)


def submit_probe(
    credential: Credential,
    request: dict[str, object],
) -> str:
    """Submit a tiny request and return its initial ADS state without waiting."""
    client = make_client(credential, wait_until_complete=False)
    try:
        result = client.retrieve(DATASET, request)
        state = str(result.reply.get("state", "accepted"))
    finally:
        client.session.close()
    return state


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


def filter_active_from_report(
    credentials: list[Credential],
    report_path: Path,
) -> list[Credential]:
    """Select credentials previously accepted by ADS, matching by fingerprint."""
    if not report_path.is_file():
        return credentials
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    active_fingerprints = {
        str(item["fingerprint"])
        for item in payload.get("accounts", [])
        if item.get("status") == "active"
    }
    if not active_fingerprints:
        raise RuntimeError(f"Account report contains no active accounts: {report_path}")
    selected = [
        credential
        for credential in credentials
        if credential.fingerprint in active_fingerprints
    ]
    if len(selected) != len(active_fingerprints):
        raise RuntimeError(
            "Credential file no longer matches the saved account-status report"
        )
    return selected


def validate_credentials(
    credentials: list[Credential],
    gfs_init: datetime,
    report_path: Path,
) -> tuple[list[Credential], list[dict[str, object]]]:
    request = build_request(gfs_init, probe=True)
    valid: list[Credential] = []
    records: list[dict[str, object]] = []
    for credential in credentials:
        print(
            f"[credential {credential.index:02d}/{len(credentials):02d}] "
            f"checking {credential.fingerprint}",
            flush=True,
        )
        try:
            state = submit_probe(credential, request)
        except Exception as exc:  # noqa: BLE001 - external API boundary
            category = classify_error(exc)
            records.append(
                {
                    "account_index": credential.index,
                    "fingerprint": credential.fingerprint,
                    "status": "inactive" if category == "credential_or_licence" else "error",
                    "category": category,
                    "error": safe_error_summary(exc),
                }
            )
            print(f"  -> {category}", flush=True)
            if category == "request":
                atomic_write_json(report_path, {"accounts": records})
                raise RuntimeError(
                    "CAMS request syntax/data error; account validation stopped"
                ) from exc
        else:
            valid.append(credential)
            records.append(
                {
                    "account_index": credential.index,
                    "fingerprint": credential.fingerprint,
                    "status": "active",
                    "submission_state": state,
                    "category": None,
                    "error": None,
                }
            )
            print(f"  -> accepted ({state})", flush=True)
        atomic_write_json(report_path, {"accounts": records})
    return valid, records


def download_dates(
    credentials: list[Credential],
    gfs_init_times: list[datetime],
    raw_root: Path,
    report_path: Path,
    area: list[float] | None = None,
    lead_hours: list[int] | None = None,
) -> dict[str, object]:
    if not credentials:
        raise RuntimeError("No active ADS credentials are available")
    account_cursor = 0
    inactive: set[int] = set()
    jobs: list[dict[str, object]] = []
    for job_index, gfs_init in enumerate(gfs_init_times, start=1):
        target = target_path(raw_root, gfs_init, lead_hours)
        cams_init = cams_init_for_gfs(gfs_init)
        base_record: dict[str, object] = {
            "gfs_init_time_utc": gfs_init.isoformat(),
            "cams_init_time_utc": cams_init.isoformat(),
            "cams_lead_hours": lead_hours or [13, 84],
            "target": str(target),
            "area": area,
        }
        if is_valid_zip(target):
            jobs.append({**base_record, "status": "existing"})
            print(f"[{job_index}/{len(gfs_init_times)}] existing {target.name}")
            continue
        attempted = 0
        transient_errors = 0
        completed = False
        while attempted < len(credentials):
            credential = credentials[account_cursor % len(credentials)]
            account_cursor += 1
            if credential.index in inactive:
                continue
            attempted += 1
            print(
                f"[{job_index}/{len(gfs_init_times)}] "
                f"GFS {gfs_init:%Y-%m-%d} <- CAMS {cams_init:%Y-%m-%d} 12Z "
                f"using account {credential.index:02d}",
                flush=True,
            )
            try:
                retrieve(
                    credential,
                    build_request(gfs_init, area=area, lead_hours=lead_hours),
                    target,
                )
            except Exception as exc:  # noqa: BLE001 - external API boundary
                category = classify_error(exc)
                if category == "credential_or_licence":
                    inactive.add(credential.index)
                elif category == "request":
                    jobs.append(
                        {
                            **base_record,
                            "status": "failed",
                            "category": category,
                            "error": safe_error_summary(exc),
                        }
                    )
                    atomic_write_json(
                        report_path,
                        {
                            "dataset": DATASET,
                            "variable": VARIABLE,
                            "ads_url": ADS_URL,
                            "jobs": jobs,
                            "inactive_account_indices": sorted(inactive),
                        },
                    )
                    raise RuntimeError("CAMS request syntax/data error") from exc
                else:
                    transient_errors += 1
                print(f"  -> {category}; trying another active account", flush=True)
                continue
            jobs.append(
                {
                    **base_record,
                    "status": "downloaded",
                    "account_index": credential.index,
                    "account_fingerprint": credential.fingerprint,
                    "bytes": target.stat().st_size,
                }
            )
            completed = True
            break
        if not completed:
            jobs.append(
                {
                    **base_record,
                    "status": "failed",
                    "category": "credentials_exhausted",
                    "transient_errors": transient_errors,
                }
            )
            atomic_write_json(
                report_path,
                {
                    "dataset": DATASET,
                    "variable": VARIABLE,
                    "ads_url": ADS_URL,
                    "jobs": jobs,
                    "inactive_account_indices": sorted(inactive),
                },
            )
            raise RuntimeError(f"Unable to download CAMS data for {gfs_init:%Y-%m-%d}")
        atomic_write_json(
            report_path,
            {
                "dataset": DATASET,
                "variable": VARIABLE,
                "ads_url": ADS_URL,
                "cycle_policy": "previous_day_12z",
                "gfs_forecast_hours": [1, 72],
                "cams_lead_hours": lead_hours or [13, 84],
                "area": area,
                "jobs": jobs,
                "inactive_account_indices": sorted(inactive),
            },
        )
    return {
        "dataset": DATASET,
        "variable": VARIABLE,
        "ads_url": ADS_URL,
        "cycle_policy": "previous_day_12z",
        "gfs_forecast_hours": [1, 72],
        "cams_lead_hours": lead_hours or [13, 84],
        "area": area,
        "jobs": jobs,
        "inactive_account_indices": sorted(inactive),
    }


def download_dates_parallel(
    credentials: list[Credential],
    gfs_init_times: list[datetime],
    raw_root: Path,
    report_path: Path,
    *,
    workers: int,
    area: list[float] | None = None,
    lead_hours: list[int] | None = None,
) -> dict[str, object]:
    """Download independent dates concurrently, using at most one slot per account.

    Each job starts with its assigned account and falls back to the remaining
    active accounts only after an error. Targets are unique per GFS date, while
    the shared manifest is updated under a lock with an atomic file replacement.
    Re-running the same command skips every valid ZIP already on disk.
    """
    if not credentials:
        raise RuntimeError("No active ADS credentials are available")
    if workers < 1:
        raise ValueError("workers must be positive")
    worker_count = min(workers, len(credentials), max(len(gfs_init_times), 1))
    lock = threading.Lock()
    inactive: set[int] = set()
    records: dict[str, dict[str, object]] = {}

    def base_record(gfs_init: datetime) -> dict[str, object]:
        cams_init = cams_init_for_gfs(gfs_init)
        return {
            "gfs_init_time_utc": gfs_init.isoformat(),
            "cams_init_time_utc": cams_init.isoformat(),
            "cams_lead_hours": lead_hours or [13, 84],
            "target": str(target_path(raw_root, gfs_init, lead_hours)),
            "area": area,
        }

    def payload_locked() -> dict[str, object]:
        return {
            "dataset": DATASET,
            "variable": VARIABLE,
            "ads_url": ADS_URL,
            "cycle_policy": "previous_day_12z",
            "gfs_forecast_hours": [1, 72],
            "cams_lead_hours": lead_hours or [13, 84],
            "area": area,
            "workers": worker_count,
            "jobs": sorted(records.values(), key=lambda item: str(item["gfs_init_time_utc"])),
            "inactive_account_indices": sorted(inactive),
        }

    def save(record: dict[str, object]) -> None:
        with lock:
            records[str(record["gfs_init_time_utc"])] = record
            atomic_write_json(report_path, payload_locked())

    pending: list[tuple[int, datetime]] = []
    for job_index, gfs_init in enumerate(gfs_init_times, start=1):
        target = target_path(raw_root, gfs_init, lead_hours)
        record = base_record(gfs_init)
        if is_valid_zip(target):
            records[gfs_init.isoformat()] = {**record, "status": "existing"}
            print(f"[{job_index}/{len(gfs_init_times)}] existing {target.name}", flush=True)
        else:
            pending.append((job_index, gfs_init))
    with lock:
        atomic_write_json(report_path, payload_locked())

    def run_one(job_index: int, gfs_init: datetime, preferred: int) -> None:
        target = target_path(raw_root, gfs_init, lead_hours)
        cams_init = cams_init_for_gfs(gfs_init)
        record = base_record(gfs_init)
        transient_errors = 0
        attempted_accounts: list[int] = []
        for offset in range(len(credentials)):
            credential = credentials[(preferred + offset) % len(credentials)]
            with lock:
                if credential.index in inactive:
                    continue
            attempted_accounts.append(credential.index)
            print(
                f"[{job_index}/{len(gfs_init_times)}] "
                f"GFS {gfs_init:%Y-%m-%d} <- CAMS {cams_init:%Y-%m-%d} 12Z "
                f"using account {credential.index:02d}",
                flush=True,
            )
            try:
                retrieve(
                    credential,
                    build_request(gfs_init, area=area, lead_hours=lead_hours),
                    target,
                )
            except Exception as exc:  # noqa: BLE001 - external API boundary
                category = classify_error(exc)
                if category == "credential_or_licence":
                    with lock:
                        inactive.add(credential.index)
                elif category == "request":
                    save({
                        **record,
                        "status": "failed",
                        "category": category,
                        "error": safe_error_summary(exc),
                        "attempted_account_indices": attempted_accounts,
                    })
                    return
                else:
                    transient_errors += 1
                print(f"  -> {category}; trying another active account", flush=True)
                continue
            save({
                **record,
                "status": "downloaded",
                "account_index": credential.index,
                "account_fingerprint": credential.fingerprint,
                "bytes": target.stat().st_size,
            })
            return
        save({
            **record,
            "status": "failed",
            "category": "credentials_exhausted",
            "transient_errors": transient_errors,
            "attempted_account_indices": attempted_accounts,
        })

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_one, job_index, gfs_init, index % worker_count)
                for index, (job_index, gfs_init) in enumerate(pending)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    with lock:
        report = payload_locked()
        atomic_write_json(report_path, report)
    failed = [job for job in report["jobs"] if job["status"] == "failed"]
    if failed:
        raise RuntimeError(f"Unable to download CAMS data for {len(failed)} date(s)")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download historical CAMS AOD550 forecasts aligned to GFS SWRadNet"
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path(r"C:\Users\12041\Desktop\CDS.txt"),
        help="External repeated url/key credential file; source URLs are ignored",
    )
    parser.add_argument(
        "--gfs-root",
        type=Path,
        default=Path("data/gfs_swradnet/station_windows"),
    )
    parser.add_argument(
        "--gfs-dates",
        nargs="*",
        metavar="YYYYMMDD",
        help="Optional explicit GFS init dates; otherwise discover existing NPZ dates",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/aod/cams/raw"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("data/aod/cams/reports"),
    )
    parser.add_argument(
        "--manifest-name",
        default="download_manifest.json",
        help="Manifest filename written below --report-root",
    )
    parser.add_argument(
        "--validate-accounts",
        action="store_true",
        help="Probe all accounts and exclude authentication/licence failures",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Stop after account validation",
    )
    parser.add_argument(
        "--max-accounts",
        type=int,
        default=0,
        help="Limit credentials used for a probe/test run; 0 uses all accounts",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Concurrent date downloads; 0 uses one worker per active account. "
            "The effective value never exceeds the number of active accounts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a non-secret download plan without network requests",
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help=(
            "Optional geographic subset in degrees. Use a station-covering "
            "crop to avoid downloading the global CAMS grid."
        ),
    )
    parser.add_argument(
        "--lead-hours",
        nargs="+",
        type=int,
        help="Optional CAMS lead subset in 13..84; default is every hourly lead.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lead_hours = sorted(set(args.lead_hours or range(13, 85)))
    if not lead_hours or min(lead_hours) < 13 or max(lead_hours) > 84:
        raise ValueError("--lead-hours must lie in 13..84")
    area = list(args.area) if args.area is not None else None
    if area is not None:
        north, west, south, east = area
        if not (-90.0 <= south < north <= 90.0):
            raise ValueError("--area requires -90 <= SOUTH < NORTH <= 90")
        if not (-180.0 <= west < east <= 180.0):
            raise ValueError("--area requires -180 <= WEST < EAST <= 180")
    credentials = parse_credentials(args.credentials)
    if args.max_accounts < 0:
        raise ValueError("--max-accounts must be non-negative")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if Path(args.manifest_name).name != args.manifest_name:
        raise ValueError("--manifest-name must be a filename, not a path")
    if args.max_accounts:
        credentials = credentials[: args.max_accounts]
    gfs_init_times = (
        parse_gfs_dates(args.gfs_dates)
        if args.gfs_dates
        else discover_gfs_init_times(args.gfs_root)
    )
    print(
        f"Loaded {len(credentials)} credentials; ADS endpoint forced to {ADS_URL}",
        flush=True,
    )
    if area is not None:
        print(f"Geographic crop [N,W,S,E]: {area}", flush=True)
    print(
        f"Planned GFS init dates: {len(gfs_init_times)} "
        f"({gfs_init_times[0]:%Y-%m-%d} .. {gfs_init_times[-1]:%Y-%m-%d})",
        flush=True,
    )
    if args.dry_run:
        for gfs_init in gfs_init_times:
            cams_init = cams_init_for_gfs(gfs_init)
            print(
                f"GFS {gfs_init:%Y-%m-%d} 00Z -> "
                f"CAMS {cams_init:%Y-%m-%d} 12Z leads {lead_hours}"
            )
        return 0

    active = credentials
    if args.validate_accounts:
        active, records = validate_credentials(
            credentials,
            gfs_init_times[0],
            args.report_root / "account_status.json",
        )
        inactive_count = sum(item["status"] == "inactive" for item in records)
        error_count = sum(item["status"] == "error" for item in records)
        print(
            f"Credential audit: {len(active)} active, "
            f"{inactive_count} inactive, {error_count} transient/error",
            flush=True,
        )
    else:
        status_path = args.report_root / "account_status.json"
        active = filter_active_from_report(credentials, status_path)
        if len(active) != len(credentials):
            print(
                f"Using {len(active)} accounts accepted by the saved ADS audit; "
                f"excluded {len(credentials) - len(active)} inactive accounts",
                flush=True,
            )
    if args.validate_only:
        return 0 if active else 2

    report_path = args.report_root / args.manifest_name
    worker_count = min(args.workers or len(active), len(active))
    if worker_count > 1:
        print(
            f"Parallel CAMS downloads: {worker_count} workers for "
            f"{len(active)} active accounts",
            flush=True,
        )
        report = download_dates_parallel(
            active,
            gfs_init_times,
            args.raw_root,
            report_path,
            workers=worker_count,
            area=area,
            lead_hours=lead_hours,
        )
    else:
        report = download_dates(
            active,
            gfs_init_times,
            args.raw_root,
            report_path,
            area=area,
            lead_hours=lead_hours,
        )
    downloaded = sum(job["status"] == "downloaded" for job in report["jobs"])
    existing = sum(job["status"] == "existing" for job in report["jobs"])
    print(f"Completed: {downloaded} downloaded, {existing} existing", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
