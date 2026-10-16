#!/usr/bin/env python3
"""Download station-domain GFS subsets for GPU-WRF through NCAR GDEX OPeNDAP.

The downloader reads only the requested pressure levels, variables, and the
station-specific bounding box. Each lead is stored in a normalized NetCDF4
file with an atomic sidecar check record, so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
import time
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from netCDF4 import Dataset, num2date
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ``datetime.UTC`` is only available in Python 3.11+.  The HPC runtime is
# Python 3.10, so keep one explicit alias that behaves identically on both.
UTC = timezone.utc


TDS_ROOT = "https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d084001"
PRESSURE_HPA = np.asarray(
    [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550,
     500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40],
    dtype=np.float32,
)
FORECAST_HOURS = tuple(range(0, 73, 3))
SOIL_TOP_M = np.asarray([0.0, 0.1, 0.4, 1.0], dtype=np.float32)
SOIL_BOTTOM_M = np.asarray([0.1, 0.4, 1.0, 2.0], dtype=np.float32)

UPPER_AIR = {
    "temperature": ("Temperature_isobaric", "K"),
    "u_wind": ("u-component_of_wind_isobaric", "m s-1"),
    "v_wind": ("v-component_of_wind_isobaric", "m s-1"),
    "relative_humidity": ("Relative_humidity_isobaric", "%"),
    "geopotential_height": ("Geopotential_height_isobaric", "m"),
}

SURFACE = {
    "surface_pressure": ("Pressure_surface", "Pa", None),
    "msl_pressure": ("MSLP_Eta_model_reduction_msl", "Pa", None),
    "terrain_height": ("Geopotential_height_surface", "m", None),
    "skin_temperature": ("Temperature_surface", "K", None),
    "temperature_2m": ("Temperature_height_above_ground", "K", 2.0),
    "relative_humidity_2m": ("Relative_humidity_height_above_ground", "%", 2.0),
    "u_wind_10m": ("u-component_of_wind_height_above_ground", "m s-1", 10.0),
    "v_wind_10m": ("v-component_of_wind_height_above_ground", "m s-1", 10.0),
    "land_mask": ("Land_cover_0__sea_1__land_surface", "1", None),
    "sea_ice_fraction": ("Ice_cover_surface", "1", None),
    "snow_water_equivalent": (
        "Water_equivalent_of_accumulated_snow_depth_surface", "kg m-2", None
    ),
    "snow_depth": ("Snow_depth_surface", "m", None),
}

SOIL = {
    "soil_temperature": ("Soil_temperature_depth_below_surface_layer", "K"),
    "soil_moisture": (
        "Volumetric_Soil_Moisture_Content_depth_below_surface_layer", "m3 m-3"
    ),
}

REQUIRED_LOCAL_VARIABLES = tuple(UPPER_AIR) + tuple(SURFACE) + tuple(SOIL)


@dataclass(frozen=True)
class Task:
    station_id: str
    cycle: str
    forecast_hour: int
    south: float
    north: float
    west: float
    east: float
    output_dir: str
    retries: int
    download_layout: str

    @property
    def source_url(self) -> str:
        year = self.cycle[:4]
        day = self.cycle[:8]
        return (
            f"{TDS_ROOT}/{year}/{day}/"
            f"gfs.0p25.{self.cycle}.f{self.forecast_hour:03d}.grib2"
        )

    @property
    def output_path(self) -> Path:
        return (
            Path(self.output_dir)
            / "raw"
            / self.station_id
            / self.cycle
            / f"f{self.forecast_hour:03d}.nc"
        )

    @property
    def check_path(self) -> Path:
        return (
            Path(self.output_dir)
            / "checks"
            / self.station_id
            / self.cycle
            / f"f{self.forecast_hour:03d}.json"
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, dir=path.parent
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_index(values: np.ndarray, target: float, tolerance: float = 1e-3) -> int:
    index = int(np.argmin(np.abs(values - target)))
    if abs(float(values[index]) - target) > tolerance:
        raise ValueError(f"Coordinate {target} not found; nearest={values[index]}")
    return index


def _remote_variable_groups(layout: str) -> dict[str, list[str]]:
    """Build either one NCSS request or the conservative three-request layout."""
    upper = [value[0] for value in UPPER_AIR.values()]
    surface = [value[0] for value in SURFACE.values()]
    soil = [value[0] for value in SOIL.values()]
    if layout == "combined":
        return {"all": upper + surface + soil}
    if layout == "split":
        return {"upper": upper, "surface": surface, "soil": soil}
    raise ValueError(f"Unknown download layout: {layout}")


def _source_time_matches_task(dataset: Dataset, task: Task) -> bool:
    expected = datetime.strptime(task.cycle, "%Y%m%d%H").replace(tzinfo=UTC)
    expected += timedelta(hours=task.forecast_hour)
    found = False
    for name, variable in dataset.variables.items():
        if not name.lower().startswith("time") or variable.ndim == 0:
            continue
        units = getattr(variable, "units", "")
        if " since " not in units:
            continue
        try:
            values = np.asarray(np.ma.filled(variable[:], np.nan)).ravel()
            # NCSS requests one valid time. Reject multi-valued or masked time
            # coordinates instead of silently selecting index zero later.
            if values.size != 1 or not np.isfinite(values).all():
                return False
            dates = num2date(values, units, calendar=getattr(variable, "calendar", "standard"))
        except Exception:
            return False
        found = True
        for actual in np.atleast_1d(dates):
            fields = (
                int(actual.year), int(actual.month), int(actual.day),
                int(actual.hour), int(actual.minute), int(actual.second),
                int(actual.microsecond),
            )
            expected_fields = (
                expected.year, expected.month, expected.day,
                expected.hour, expected.minute, expected.second,
                expected.microsecond,
            )
            if fields != expected_fields:
                return False
    return found


def _source_group_valid(path: Path, variable_names: list[str], task: Task) -> bool:
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    try:
        with Dataset(path) as dataset:
            return (
                all(name in dataset.variables for name in variable_names)
                and _source_time_matches_task(dataset, task)
            )
    except Exception:
        return False


def _download_ncss(task: Task, destination: Path, variable_names: list[str]) -> None:
    if _source_group_valid(destination, variable_names, task):
        return
    params: list[tuple[str, str]] = [("var", name) for name in variable_names]
    params.extend(
        [
            ("north", f"{task.north:.2f}"),
            ("south", f"{task.south:.2f}"),
            ("west", f"{task.west:.2f}"),
            ("east", f"{task.east:.2f}"),
            ("horizStride", "1"),
            ("accept", "netcdf3"),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    transfer = destination.with_suffix(destination.suffix + ".http")
    transfer.unlink(missing_ok=True)
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    with requests.Session() as session:
        session.mount(
            "https://",
            HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1),
        )
        try:
            with session.get(
                task.source_url,
                params=params,
                stream=True,
                timeout=(30, 600),
                headers={"Connection": "close", "User-Agent": "wrf-gfs-subset/1.0"},
            ) as response:
                response.raise_for_status()
                with transfer.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            if transfer.stat().st_size < 10_000:
                raise ValueError(
                    f"NCSS response is too small: {transfer.stat().st_size} bytes"
                )
            if not _source_group_valid(transfer, variable_names, task):
                raise ValueError(
                    "NCSS response has missing variables or a valid time that does not "
                    f"match f{task.forecast_hour:03d}"
                )
            transfer.replace(destination)
        except Exception:
            transfer.unlink(missing_ok=True)
            raise


def _read_local_window(
    ds: Dataset,
    variable_name: str,
    lat_order: np.ndarray,
    lon_order: np.ndarray,
    dimension_selectors: dict[str, int | slice | np.ndarray] | None = None,
) -> np.ndarray:
    variable = ds.variables[variable_name]
    dimension_selectors = dimension_selectors or {}
    index: list[int | slice] = []
    for dimension in variable.dimensions:
        if dimension.lower().startswith("time"):
            index.append(0)
        elif dimension in dimension_selectors:
            index.append(dimension_selectors[dimension])
        else:
            index.append(slice(None))
    output = np.asarray(np.ma.filled(variable[tuple(index)], np.nan), dtype=np.float32)
    output = np.take(output, lat_order, axis=-2)
    output = np.take(output, lon_order, axis=-1)
    return output.copy()


def _pressure_coordinate(ds: Dataset, variable_name: str) -> tuple[str, np.ndarray]:
    """Return the pressure dimension and its values in Pa.

    NCAR's NCSS responses do not use one stable coordinate name across the
    archive.  Newer files commonly expose ``isobaric`` while older files use
    names such as ``isobaric1`` or ``isobaric3``.  Resolve the coordinate from
    the requested variable's dimensions instead of assuming one spelling.
    """
    variable = ds.variables[variable_name]
    vertical_dimensions = [
        dimension
        for dimension in variable.dimensions
        if not dimension.lower().startswith("time")
        and dimension.lower() not in {"lat", "lon", "latitude", "longitude"}
    ]
    if len(vertical_dimensions) != 1:
        raise ValueError(
            f"Unexpected pressure dimensions for {variable_name}: "
            f"{vertical_dimensions}"
        )
    coordinate_name = vertical_dimensions[0]
    if coordinate_name not in ds.variables:
        raise KeyError(
            f"Pressure coordinate {coordinate_name} for {variable_name} is missing"
        )
    pressure = np.asarray(ds.variables[coordinate_name][:], dtype=np.float64)
    pressure_pa = pressure * 100.0 if float(np.nanmax(pressure)) < 2_000.0 else pressure
    if pressure_pa.ndim != 1 or not np.all(np.isfinite(pressure_pa)):
        raise ValueError(f"Invalid pressure coordinate for {variable_name}")
    if len(np.unique(pressure_pa)) != pressure_pa.size:
        raise ValueError(f"Duplicate pressure coordinate for {variable_name}")
    return coordinate_name, pressure_pa


def _read_pressure_levels(
    ds: Dataset,
    variable_name: str,
    lat_order: np.ndarray,
    lon_order: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Read the frozen 24 pressure levels, interpolating only when bracketed.

    The 2020 d084001 archive omits 40 hPa in some files while retaining 30 and
    50 hPa.  Linear interpolation in pressure preserves the frozen 40 hPa
    output coordinate without relabelling a neighbouring source level.
    """
    pressure_name, source_pa = _pressure_coordinate(ds, variable_name)
    values = _read_local_window(ds, variable_name, lat_order, lon_order)
    remaining_dimensions = [
        dimension
        for dimension in ds.variables[variable_name].dimensions
        if not dimension.lower().startswith("time")
    ]
    pressure_axis = remaining_dimensions.index(pressure_name)
    values = np.moveaxis(values, pressure_axis, 0)
    if values.shape[0] != source_pa.size:
        raise ValueError(f"Pressure dimension mismatch for {variable_name}")

    outputs: list[np.ndarray] = []
    provenance: list[dict[str, float]] = []
    for target_hpa in PRESSURE_HPA.astype(np.float64):
        target_pa = float(target_hpa * 100.0)
        exact = np.where(np.isclose(source_pa, target_pa, rtol=0.0, atol=1e-3))[0]
        if exact.size == 1:
            outputs.append(values[int(exact[0])])
            continue
        below = np.where(source_pa < target_pa)[0]
        above = np.where(source_pa > target_pa)[0]
        if below.size == 0 or above.size == 0:
            nearest = float(source_pa[np.argmin(np.abs(source_pa - target_pa))])
            raise ValueError(
                f"Pressure {target_pa} Pa is absent and not bracketed; nearest={nearest}"
            )
        lower_index = int(below[np.argmax(source_pa[below])])
        upper_index = int(above[np.argmin(source_pa[above])])
        lower_pa = float(source_pa[lower_index])
        upper_pa = float(source_pa[upper_index])
        weight = np.float32((target_pa - lower_pa) / (upper_pa - lower_pa))
        outputs.append(
            (values[lower_index] * (np.float32(1.0) - weight)
             + values[upper_index] * weight).astype(np.float32)
        )
        provenance.append({
            "target_hpa": float(target_hpa),
            "lower_hpa": lower_pa / 100.0,
            "upper_hpa": upper_pa / 100.0,
            "upper_weight": float(weight),
        })
    return np.stack(outputs).astype(np.float32), provenance


def _read_subset(source_paths: dict[str, Path], task: Task) -> dict[str, Any]:
    with ExitStack() as stack:
        datasets = {
            name: stack.enter_context(Dataset(path))
            for name, path in source_paths.items()
        }
        def dataset_for(variable_name: str) -> Dataset:
            for candidate in datasets.values():
                if variable_name in candidate.variables:
                    return candidate
            raise KeyError(f"Downloaded groups do not contain {variable_name}")

        ds = dataset_for("Temperature_isobaric")
        latitude_name = "latitude" if "latitude" in ds.variables else "lat"
        longitude_name = "longitude" if "longitude" in ds.variables else "lon"
        source_lat = np.asarray(ds.variables[latitude_name][:], dtype=np.float64)
        source_lon = np.asarray(ds.variables[longitude_name][:], dtype=np.float64)
        normalized_lon = source_lon.copy()
        normalized_lon[normalized_lon > 180.0] -= 360.0
        lat_order = np.argsort(source_lat)
        lon_order = np.argsort(normalized_lon)
        lat = source_lat[lat_order].astype(np.float32)
        lon = normalized_lon[lon_order].astype(np.float32)
        if not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
            raise ValueError("NCSS subset coordinates are not unique and increasing")
        if not (
            math.isclose(float(lat[0]), task.south)
            and math.isclose(float(lat[-1]), task.north)
            and math.isclose(float(lon[0]), task.west)
            and math.isclose(float(lon[-1]), task.east)
        ):
            raise ValueError(
                f"NCSS bounds differ from request: lat={lat[0]},{lat[-1]} "
                f"lon={lon[0]},{lon[-1]}"
            )
        payload: dict[str, Any] = {"lat": lat, "lon": lon}
        pressure_level_handling: dict[str, list[dict[str, float]]] = {}
        for local_name, (remote_name, units) in UPPER_AIR.items():
            upper_ds = dataset_for(remote_name)
            values, provenance = _read_pressure_levels(
                upper_ds, remote_name, lat_order, lon_order
            )
            payload[local_name] = values
            payload[f"{local_name}_units"] = units
            if provenance:
                pressure_level_handling[local_name] = provenance
        payload["pressure_level_handling"] = pressure_level_handling

        for local_name, (remote_name, units, level_value) in SURFACE.items():
            surface_ds = dataset_for(remote_name)
            selectors: dict[str, int | slice] = {}
            if level_value is not None:
                remote = surface_ds.variables[remote_name]
                vertical_dimensions = [
                    dim for dim in remote.dimensions
                    if not dim.lower().startswith("time")
                    and dim not in {"lat", "lon", "latitude", "longitude"}
                ]
                if len(vertical_dimensions) != 1:
                    raise ValueError(f"Unexpected dimensions for {remote_name}")
                coordinate_name = vertical_dimensions[0]
                coordinate = np.asarray(
                    surface_ds.variables[coordinate_name][:], dtype=np.float64
                )
                selectors[coordinate_name] = _nearest_index(coordinate, level_value)
            payload[local_name] = _read_local_window(
                surface_ds, remote_name, lat_order, lon_order, selectors
            )
            payload[f"{local_name}_units"] = units

        for local_name, (remote_name, units) in SOIL.items():
            soil_ds = dataset_for(remote_name)
            payload[local_name] = _read_local_window(
                soil_ds, remote_name, lat_order, lon_order
            )
            payload[f"{local_name}_units"] = units
        return payload


def _write_normalized(path: Path, task: Task, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    init = datetime.strptime(task.cycle, "%Y%m%d%H").replace(tzinfo=UTC)
    valid = init + timedelta(hours=task.forecast_hour)
    lat = payload["lat"]
    lon = payload["lon"]

    with Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("pressure", len(PRESSURE_HPA))
        output.createDimension("soil_layer", 4)
        output.createDimension("lat", len(lat))
        output.createDimension("lon", len(lon))
        output.station_id = task.station_id
        output.init_time_utc = init.isoformat().replace("+00:00", "Z")
        output.valid_time_utc = valid.isoformat().replace("+00:00", "Z")
        output.forecast_hour = task.forecast_hour
        output.source_url = task.source_url
        output.source_dataset = "NCAR GDEX d084001"
        output.spatial_subset = "d01 envelope plus 1.5 degree buffer"
        output.source_group_times_verified = 1
        output.pressure_level_handling_json = json.dumps(
            payload.get("pressure_level_handling", {}), sort_keys=True
        )

        def coordinate(name: str, dimensions: tuple[str, ...], values: np.ndarray, units: str):
            variable = output.createVariable(name, "f4", dimensions)
            variable.units = units
            variable[:] = values

        coordinate("pressure", ("pressure",), PRESSURE_HPA, "hPa")
        coordinate("soil_layer_top", ("soil_layer",), SOIL_TOP_M, "m")
        coordinate("soil_layer_bottom", ("soil_layer",), SOIL_BOTTOM_M, "m")
        coordinate("lat", ("lat",), lat, "degrees_north")
        coordinate("lon", ("lon",), lon, "degrees_east")

        for local_name in UPPER_AIR:
            variable = output.createVariable(
                local_name, "f4", ("pressure", "lat", "lon"),
                zlib=True, complevel=4, shuffle=True, fill_value=np.float32(np.nan),
            )
            variable.units = payload[f"{local_name}_units"]
            variable[:] = payload[local_name]

        for local_name in SURFACE:
            variable = output.createVariable(
                local_name, "f4", ("lat", "lon"),
                zlib=True, complevel=4, shuffle=True, fill_value=np.float32(np.nan),
            )
            variable.units = payload[f"{local_name}_units"]
            variable[:] = payload[local_name]

        for local_name in SOIL:
            variable = output.createVariable(
                local_name, "f4", ("soil_layer", "lat", "lon"),
                zlib=True, complevel=4, shuffle=True, fill_value=np.float32(np.nan),
            )
            variable.units = payload[f"{local_name}_units"]
            variable[:] = payload[local_name]
    temporary.replace(path)


def validate_file(path: Path, task: Task) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size < 100_000:
        raise ValueError(f"Missing or implausibly small file: {path}")
    with Dataset(path) as ds:
        missing = sorted(set(REQUIRED_LOCAL_VARIABLES) - set(ds.variables))
        if missing:
            raise ValueError(f"Missing local variables: {missing}")
        if len(ds.dimensions["pressure"]) != 24 or len(ds.dimensions["soil_layer"]) != 4:
            raise ValueError("Unexpected pressure or soil dimension")
        if not np.allclose(ds.variables["pressure"][:], PRESSURE_HPA):
            raise ValueError("Pressure coordinate mismatch")
        lat = np.asarray(ds.variables["lat"][:])
        lon = np.asarray(ds.variables["lon"][:])
        if not (math.isclose(float(lat[0]), task.south) and math.isclose(float(lat[-1]), task.north)):
            raise ValueError("Latitude bounds mismatch")
        if not (math.isclose(float(lon[0]), task.west) and math.isclose(float(lon[-1]), task.east)):
            raise ValueError("Longitude bounds mismatch")
        if not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
            raise ValueError("Coordinates are not increasing")
        finite_report: dict[str, float] = {}
        land = np.asarray(np.ma.filled(ds.variables["land_mask"][:], np.nan)) > 0.5
        land_only = {
            "soil_temperature", "soil_moisture",
            "snow_water_equivalent", "snow_depth",
        }
        for name in REQUIRED_LOCAL_VARIABLES:
            values = np.asarray(np.ma.filled(ds.variables[name][:], np.nan))
            fraction = float(np.isfinite(values).mean())
            finite_report[name] = fraction
            if name in land_only:
                land_values = values[..., land]
                land_fraction = float(np.isfinite(land_values).mean())
                finite_report[f"{name}_over_land"] = land_fraction
                if land_fraction < 0.999:
                    raise ValueError(
                        f"Too many non-finite land values in {name}: {land_fraction:.3f}"
                    )
            elif fraction < 0.999:
                raise ValueError(f"Too many non-finite values in {name}: {fraction:.3f}")
        if min(finite_report[name] for name in UPPER_AIR) < 0.999:
            raise ValueError("Upper-air completeness check failed")
        source_group_time_verified = bool(
            getattr(ds, "source_group_times_verified", 0) == 1
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "pressure_level_count": 24,
        "surface_field_count": 20,
        "field_slab_count": 140,
        "source_group_time_verified": source_group_time_verified,
        "finite_fraction": finite_report,
        "validated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def run_task(task: Task) -> dict[str, Any]:
    if task.output_path.exists():
        try:
            report = validate_file(task.output_path, task)
            report.update({"status": "already_complete", "task": asdict(task)})
            _atomic_json(task.check_path, report)
            return report
        except Exception:
            task.output_path.unlink(missing_ok=True)

    last_error = ""
    layouts = (
        ("combined", "split")
        if task.download_layout == "auto"
        else (task.download_layout,)
    )
    for attempt in range(1, task.retries + 1):
        for layout in layouts:
            groups = _remote_variable_groups(layout)
            source_temporaries = {
                group: task.output_path.with_suffix(f".{group}.source.nc.part")
                for group in groups
            }
            try:
                for group, variable_names in groups.items():
                    try:
                        _download_ncss(task, source_temporaries[group], variable_names)
                    except Exception as exc:
                        raise RuntimeError(f"layout={layout} group={group}: {exc}") from exc
                payload = _read_subset(source_temporaries, task)
                _write_normalized(task.output_path, task, payload)
                report = validate_file(task.output_path, task)
                report.update(
                    {
                        "status": "downloaded",
                        "attempt": attempt,
                        "download_layout": layout,
                        "task": asdict(task),
                    }
                )
                _atomic_json(task.check_path, report)
                for source_temporary in source_temporaries.values():
                    source_temporary.unlink(missing_ok=True)
                return report
            except Exception as exc:  # noqa: BLE001 - persisted for retry diagnostics
                last_error = f"{type(exc).__name__}: {exc}"
                task.output_path.with_suffix(task.output_path.suffix + ".part").unlink(missing_ok=True)
                # Never reuse a partial or semantically invalid NCSS response.
                for source_temporary in source_temporaries.values():
                    source_temporary.unlink(missing_ok=True)
                # In auto mode, immediately try the conservative split layout.
                if layout == "combined" and task.download_layout == "auto":
                    continue
                break
        if attempt < task.retries:
            time.sleep(min(60.0, 2.0 ** (attempt - 1)))
    error = {"status": "failed", "error": last_error, "task": asdict(task)}
    _atomic_json(task.check_path, error)
    return error


def _read_domains(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["station_id"]: row for row in rows}


def _read_cycles(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "development" in payload:
        cycles = payload["development"] + payload["test"]
    else:
        cycles = payload["train"] + payload["val"] + payload["test"]
    return sorted(str(value) for value in cycles)


def _write_run_summary(output_dir: Path, reports: list[dict[str, Any]]) -> Path:
    counts: dict[str, int] = {}
    for report in reports:
        status = str(report["status"])
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "task_count": len(reports),
        "status_counts": counts,
        "total_bytes": sum(int(report.get("bytes", 0)) for report in reports),
        "reports": reports,
    }
    path = output_dir / "manifests" / "last_run.json"
    _atomic_json(path, summary)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains", type=Path, default=Path("wrf_gpu/manifests/station_domains.csv")
    )
    parser.add_argument(
        "--cycles", type=Path, default=Path("configs/gfs_temporal_grouped_cv_47.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/wrf_gfs_subset"))
    parser.add_argument(
        "--case-manifest",
        type=Path,
        help="Download only station/cycle/forecast-hour triples required by a random-6h case manifest",
    )
    parser.add_argument("--station", action="append", help="Station ID; repeatable")
    parser.add_argument("--cycle", action="append", help="YYYYMMDDHH; repeatable")
    parser.add_argument("--forecast-hours", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument(
        "--download-layout",
        choices=("auto", "combined", "split"),
        default="auto",
        help="NCSS request layout; auto tries one combined request then falls back to split",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Diagnostic limit applied after deterministic source-file ordering",
    )
    args = parser.parse_args()

    domains = _read_domains(args.domains)
    selected_stations = args.station or sorted(domains)
    unknown = sorted(set(selected_stations) - set(domains))
    if unknown:
        raise SystemExit(f"Unknown stations: {unknown}")
    # A case manifest is self-contained: each row provides its WRF/GFS cycle.
    # Do not require the legacy cycle JSON when this explicit manifest is used.
    selected_cycles = args.cycle or ([] if args.case_manifest else _read_cycles(args.cycles))
    for cycle in selected_cycles:
        datetime.strptime(cycle, "%Y%m%d%H")
    hours = tuple(args.forecast_hours or FORECAST_HOURS)
    invalid_hours = [hour for hour in hours if hour not in FORECAST_HOURS]
    if invalid_hours:
        raise SystemExit(f"Invalid forecast hours: {invalid_hours}")

    requested: list[tuple[str, str, int]] = []
    if args.case_manifest:
        with args.case_manifest.open(encoding="utf-8-sig", newline="") as stream:
            case_rows = list(csv.DictReader(stream))
        for row in case_rows:
            # ``--station`` also applies when a case manifest is supplied.  This
            # lets multiple download hosts work on disjoint station partitions
            # while sharing one authoritative case manifest.
            if args.station and row["station_id"] not in selected_stations:
                continue
            init_time = datetime.fromisoformat(row["wrf_init_time_utc"].replace("Z", "+00:00"))
            cycle = init_time.strftime("%Y%m%d%H")
            for forecast_hour in row["gfs_hours_direct"].split(","):
                requested.append((row["station_id"], cycle, int(forecast_hour)))
        requested_stations = {item[0] for item in requested}
        unknown = sorted(requested_stations - set(domains))
        if unknown:
            raise SystemExit(f"Unknown stations in case manifest: {unknown}")
    else:
        requested = [
            (station_id, cycle, forecast_hour)
            for station_id in selected_stations
            for cycle in selected_cycles
            for forecast_hour in hours
        ]

    # Adjacent station subsets from the same GRIB2 source can reuse server-side
    # file metadata/cache.  The previous station-first order repeatedly jumped
    # between unrelated archive files.
    requested = sorted(set(requested), key=lambda item: (item[1], item[2], item[0]))
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise SystemExit("--max-tasks must be positive")
        requested = requested[:args.max_tasks]

    tasks: list[Task] = []
    for station_id, cycle, forecast_hour in requested:
        domain = domains[station_id]
        tasks.append(
            Task(
                station_id=station_id,
                cycle=cycle,
                forecast_hour=forecast_hour,
                south=float(domain["gfs_south"]),
                north=float(domain["gfs_north"]),
                west=float(domain["gfs_west"]),
                east=float(domain["gfs_east"]),
                output_dir=str(args.output_dir),
                retries=args.retries,
                download_layout=args.download_layout,
            )
        )

    print(
        f"GFS subset tasks={len(tasks)} stations={len({item.station_id for item in tasks})} "
        f"cycles={len({item.cycle for item in tasks})} workers={args.workers}",
        flush=True,
    )
    reports: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                report = future.result()
            except Exception as exc:  # child crash outside run_task
                report = {"status": "failed", "error": str(exc), "task": asdict(task)}
            reports.append(report)
            print(
                f"[{index}/{len(tasks)}] {task.station_id} {task.cycle} "
                f"f{task.forecast_hour:03d} {report['status']}",
                flush=True,
            )

    summary_path = _write_run_summary(args.output_dir, reports)
    failures = [report for report in reports if report["status"] == "failed"]
    print(f"summary={summary_path} failures={len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
