"""Leakage-safe data adapter for the v4 wind/cloud-aware GHI model."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DOMAIN_KEYS = {
    "d02": {
        "volume": ("nwp_d02_3d_raw", "nwp_d02_3d", "nwp_3d"),
        "valid": ("nwp_d02_level_valid", "nwp_d02_3d_valid", "nwp_3d_valid"),
        "surface": ("nwp_d02_surface_raw", "nwp_d02_surface", "nwp_surface"),
        "channel": ("nwp_d02_channel_available", "nwp_3d_channel_available"),
        "surface_channel": ("nwp_d02_surface_available", "nwp_surface_available"),
    },
    "d01": {
        "volume": ("nwp_d01_3d_raw", "nwp_d01_3d", "nwp_3d"),
        "valid": ("nwp_d01_level_valid", "nwp_d01_3d_valid", "nwp_3d_valid"),
        "surface": ("nwp_d01_surface_raw", "nwp_d01_surface", "nwp_surface"),
        "channel": ("nwp_d01_channel_available", "nwp_3d_channel_available"),
        "surface_channel": ("nwp_d01_surface_available", "nwp_surface_available"),
    },
}
CAPE_CIN_ALGORITHM = "wrf_python_cape_2d_rip_max_theta_e_0_3km_500m_parcel_ter_follow_v1"
ANCHOR_SOURCE = "d02_ACSWDNB_endpoint_difference_3600s"
EXPECTED_LATENT_TO_EN_CONVENTION = "columns=east_north_metres_per_latent_x_y_cell"


def cycle_key(row: dict[str, str]) -> str:
    """Return the canonical GFS initialization-cycle grouping key."""

    for name in ("cycle", "init_time_utc", "wrf_init_time_utc"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    raise KeyError("index row must contain cycle, init_time_utc, or wrf_init_time_utc")


def data_role(row: dict[str, str]) -> str:
    for name in ("role", "current_role", "data_role", "dataset_role", "partition"):
        value = str(row.get(name, "")).strip().lower()
        if value:
            return value
    return ""


def read_index(root: Path) -> list[dict[str, str]]:
    with (root / "index.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def aod_reference_for(case: dict[str, np.ndarray], reference: dict) -> np.ndarray:
    station = str(case["station_id"].item())
    result = []
    for value in case["valid_time_utc"].astype(str):
        month = value[5:7]
        result.append(
            reference["station_month_median"].get(
                f"{station}|{month}", reference["station_median"].get(station, reference["global_median"])
            )
        )
    return np.asarray(result, dtype=np.float32)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def derived_solar(case: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct causal solar geometry, physical bounds and transition support."""

    solar = case["solar"].astype(np.float32)
    subhour_cos = case["subhour_cos_sza"].astype(np.float32)
    day_of_year = np.asarray(
        [_parse_utc(value).timetuple().tm_yday for value in case["valid_time_utc"].astype(str)],
        dtype=np.float32,
    )
    positive = np.maximum(subhour_cos, 0.0)
    samples = np.minimum(1.0 / np.maximum(positive, 0.08), 12.5)
    air_mass = (samples * positive).sum(axis=1) / np.maximum(positive.sum(axis=1), 1e-6)
    air_mass = np.where(solar[:, 2] > 0.0, air_mass, 0.0).astype(np.float32)
    stable_scale = solar[:, 1] + 20.0 * solar[:, 2]
    extraterrestrial = 1361.0 * (1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0))
    physical_upper = np.where(
        solar[:, 2] > 0.0, np.minimum(2.0 * stable_scale, extraterrestrial), 0.0
    ).astype(np.float32)
    low_sun = solar[:, 2] * np.clip((0.30 - solar[:, 0]) / 0.30, 0.0, 1.0)
    transition_support = np.maximum(solar[:, 3], low_sun).astype(np.float32)
    solar_context = np.stack(
        [
            solar[:, 0], solar[:, 1] / 1000.0, solar[:, 2], solar[:, 3], air_mass / 12.5,
            np.sin(2.0 * np.pi * day_of_year / 365.0),
            np.cos(2.0 * np.pi * day_of_year / 365.0), low_sun,
        ],
        axis=-1,
    ).astype(np.float32)
    return {
        "solar_context": solar_context,
        "air_mass": air_mass,
        "stable_scale": stable_scale.astype(np.float32),
        "physical_upper": physical_upper,
        "transition_support": transition_support,
    }


def _first(case: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in case:
            return case[name]
    raise KeyError(f"none of the fields are present: {names}")


def _optional(case: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in case:
            return case[name]
    return None


def _stats(values: np.ndarray, valid: np.ndarray, channel_axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduction = tuple(index for index in range(values.ndim) if index != channel_axis)
    safe = np.where(valid, values, 0.0).astype(np.float64)
    return safe.sum(axis=reduction), (safe * safe).sum(axis=reduction), valid.sum(axis=reduction).astype(np.float64)


def compute_normalization(paths: list[Path]) -> dict:
    """Compute fold-training-only statistics for every available v4 source."""

    accumulators: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for path in paths:
        with np.load(str(path), allow_pickle=False) as loaded:
            case = {name: np.asarray(loaded[name]) for name in loaded.files}
        sources: list[tuple[str, np.ndarray, np.ndarray, int]] = []
        satellite = case["satellite"].astype(np.float64)
        sources.append(("satellite", satellite, case["satellite_valid"].astype(bool) & np.isfinite(satellite), 1))
        for domain, keys in DOMAIN_KEYS.items():
            volume = _first(case, keys["volume"]).astype(np.float64)
            level_valid = _first(case, keys["valid"]).astype(bool)
            valid = np.broadcast_to(level_valid[:, None], volume.shape) & np.isfinite(volume)
            channel_available = _optional(case, keys["channel"])
            if channel_available is not None:
                valid &= np.asarray(channel_available, bool).reshape(1, -1, 1, 1, 1)
            surface = _first(case, keys["surface"]).astype(np.float64)
            surface_valid = np.isfinite(surface)
            surface_available = _optional(case, keys["surface_channel"])
            if surface_available is not None:
                surface_valid &= np.asarray(surface_available, bool).reshape(1, -1, 1, 1)
            sources.extend([
                (f"nwp_{domain}_3d", volume, valid, 1),
                (f"nwp_{domain}_surface", surface, surface_valid, 1),
            ])
        for name, values, valid, channel_axis in sources:
            sums, squares, counts = _stats(values, valid, channel_axis)
            if name not in accumulators:
                accumulators[name] = (np.zeros_like(sums), np.zeros_like(squares), np.zeros_like(counts))
            total, total_square, total_count = accumulators[name]
            total += sums
            total_square += squares
            total_count += counts
    result = {}
    for name, (sums, squares, counts) in accumulators.items():
        if np.any(counts == 0):
            missing = np.flatnonzero(counts == 0).tolist()
            # Structurally unavailable microphysics channels are represented by
            # availability masks and use neutral normalization.
            counts = counts.copy()
            counts[counts == 0] = 1.0
            means = sums / counts
            variance = np.maximum(squares / counts - means * means, 1e-8)
            std = np.sqrt(variance)
            means[missing] = 0.0
            std[missing] = 1.0
        else:
            means = sums / counts
            std = np.sqrt(np.maximum(squares / counts - means * means, 1e-8))
            missing = []
        result[name] = {"mean": means.tolist(), "std": std.tolist(), "count": counts.tolist(), "unavailable_channels": missing}
    return result


def _normalize(values: np.ndarray, valid: np.ndarray, statistics: dict) -> np.ndarray:
    mean = np.asarray(statistics["mean"], np.float32)
    std = np.asarray(statistics["std"], np.float32)
    shape = [1] * values.ndim
    shape[1] = mean.size
    normalized = (values.astype(np.float32) - mean.reshape(shape)) / std.reshape(shape)
    return np.where(valid & np.isfinite(normalized), np.clip(normalized, -8.0, 8.0), 0.0).astype(np.float32)


def _static_vector(case: dict[str, np.ndarray]) -> np.ndarray:
    latitude = float(case["latitude"])
    longitude = float(case["longitude"])
    return np.asarray([
        math.sin(math.radians(latitude)), math.cos(math.radians(latitude)),
        math.sin(math.radians(longitude)), math.cos(math.radians(longitude)),
        float(case["elevation_m"]) / 3000.0,
        float(case["climate_id"]) / 6.0,
        float(case["issue_offset_hours"]) / 72.0,
    ], dtype=np.float32)


def _satellite_times(case: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "satellite_scan_time_offset_hours" in case:
        offsets = case["satellite_scan_time_offset_hours"].astype(np.float32)
    elif "satellite_time_offset_hours" in case:
        offsets = case["satellite_time_offset_hours"].astype(np.float32)
    else:
        offsets = -case["satellite_age_hours"].astype(np.float32)
    if "satellite_unique_frame_mask" in case:
        unique = case["satellite_unique_frame_mask"].astype(bool)
    else:
        # Keep only the first occurrence of each actual scan time.  This is a
        # compatibility fallback; formal v4 data must provide a mask derived
        # from scan timestamps or stable frame identifiers during extraction.
        unique = np.zeros_like(offsets, dtype=bool)
        seen: list[float] = []
        for index, offset in enumerate(offsets):
            if not any(abs(float(offset) - previous) <= 1e-4 for previous in seen):
                unique[index] = True
                seen.append(float(offset))
        unique &= case["satellite_frame_mask"].astype(bool)
    return offsets, unique


def _optional_text_vector(case: dict[str, np.ndarray], key: str) -> list[str] | None:
    if key not in case:
        return None
    values = np.asarray(case[key])
    if values.shape != (6,):
        raise ValueError(f"{key} must contain exactly six historical-frame entries")
    result = []
    for value in values.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value))
    return result


def _optional_scalar_text(case: dict[str, np.ndarray], key: str) -> str | None:
    if key not in case:
        return None
    value = np.asarray(case[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar string")
    scalar = value.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return str(scalar)


def _grid_fingerprints(case: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    keys = (
        "satellite_scan_x_rad", "satellite_scan_y_rad",
        "satellite_scan_x_edges_rad", "satellite_scan_y_edges_rad",
        "satellite_scan_longitude", "satellite_scan_latitude",
    )
    result: dict[str, dict[str, object]] = {}
    for key in keys:
        if key not in case:
            continue
        values = np.ascontiguousarray(case[key])
        result[key] = {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    return result


def _grid_fingerprints_sha256(fingerprints: dict[str, dict[str, object]]) -> str | None:
    if not fingerprints:
        return None
    payload = json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_optional_bool(case: dict[str, np.ndarray], key: str) -> tuple[bool, bool]:
    """Read an explicit scalar Boolean without truth-coercing text or numbers."""
    if key not in case:
        return False, False
    value = np.asarray(case[key])
    if value.shape != () or value.dtype.kind != "b":
        raise ValueError(f"{key} must be a scalar Boolean")
    return True, bool(value.item())


def _certified_latent_geometry(case: dict[str, np.ndarray]) -> dict[str, object]:
    """Return geometry only when the case carries a matching explicit certificate.

    The flag is necessary but not sufficient: the source-declared convention,
    audit reference, common-grid identifier, and actual coordinate fingerprints
    must agree. Frame identity/time metadata must cover every active history slot.
    """
    matrix_present = "satellite_latent_to_en_m" in case
    if matrix_present:
        transform = np.asarray(case["satellite_latent_to_en_m"], dtype=np.float32)
        if transform.shape != (2, 2) or not np.isfinite(transform).all():
            raise ValueError("satellite_latent_to_en_m must be a finite 2x2 matrix")
        matrix_nonsingular = abs(float(np.linalg.det(transform.astype(np.float64)))) > 1e-6
    else:
        transform = np.zeros((2, 2), dtype=np.float32)
        matrix_nonsingular = False

    flag_present, flag_value = _strict_optional_bool(case, "satellite_latent_to_en_available")
    convention = _optional_scalar_text(case, "satellite_latent_to_en_convention")
    provenance_ref = _optional_scalar_text(case, "satellite_latent_to_en_provenance_ref")
    geometry_source = _optional_scalar_text(case, "satellite_geometry_source")
    resampling_grid_id = _optional_scalar_text(case, "satellite_resampling_grid_id")
    declared_grid_sha256 = _optional_scalar_text(case, "satellite_latent_to_en_grid_sha256")
    grid_fingerprints = _grid_fingerprints(case)
    computed_grid_sha256 = _grid_fingerprints_sha256(grid_fingerprints)
    has_grid_pair = any(
        {x_key, y_key}.issubset(grid_fingerprints)
        for x_key, y_key in (
            ("satellite_scan_x_rad", "satellite_scan_y_rad"),
            ("satellite_scan_x_edges_rad", "satellite_scan_y_edges_rad"),
            ("satellite_scan_longitude", "satellite_scan_latitude"),
        )
    )

    frame_ids = _optional_text_vector(case, "satellite_frame_id")
    scan_times = _optional_text_vector(case, "satellite_scan_time_utc")
    frame_times = _optional_text_vector(case, "satellite_frame_time_utc")
    frame_mask = np.asarray(case.get("satellite_frame_mask", np.zeros(6, dtype=bool)), dtype=bool)
    if frame_mask.shape != (6,):
        raise ValueError("satellite_frame_mask must contain exactly six historical-frame entries")
    frame_metadata_complete = all(values is not None for values in (frame_ids, scan_times, frame_times))
    if frame_metadata_complete:
        frame_metadata_complete = all(
            values is not None and bool(values[index].strip())
            for values in (frame_ids, scan_times, frame_times)
            for index in np.flatnonzero(frame_mask)
        )

    available = bool(
        matrix_present and matrix_nonsingular and flag_present and flag_value
        and convention == EXPECTED_LATENT_TO_EN_CONVENTION
        and bool(provenance_ref) and bool(geometry_source) and bool(resampling_grid_id)
        and has_grid_pair and computed_grid_sha256 is not None
        and declared_grid_sha256 == computed_grid_sha256
        and frame_metadata_complete
    )
    return {
        "transform": transform,
        "available": available,
        "matrix_present": matrix_present,
        "matrix_finite_nonsingular": bool(matrix_present and matrix_nonsingular),
        "availability_flag_present": flag_present,
        "availability_flag_value": flag_value,
        "matrix_convention": convention,
        "expected_matrix_convention": EXPECTED_LATENT_TO_EN_CONVENTION,
        "provenance_ref": provenance_ref,
        "geometry_source": geometry_source,
        "resampling_grid_id": resampling_grid_id,
        "declared_grid_sha256": declared_grid_sha256,
        "computed_grid_sha256": computed_grid_sha256,
        "grid_coordinate_fingerprints": grid_fingerprints,
        "frame_ids": frame_ids,
        "scan_times_utc": scan_times,
        "frame_times_utc": frame_times,
        "frame_metadata_complete": bool(frame_metadata_complete),
        "has_coordinate_grid_pair": bool(has_grid_pair),
    }


def _scaled_static_grid(case: dict[str, np.ndarray], domain: str, size: tuple[int, int]) -> np.ndarray:
    values = _optional(case, (f"nwp_{domain}_static", f"nwp_{domain}_static_raw"))
    if values is None:
        return np.zeros((8, *size), dtype=np.float32)
    values = np.asarray(values, np.float32).copy()
    if values.shape[0] != 8:
        raise ValueError(f"nwp_{domain}_static must have 8 channels")
    if values.shape[-2:] != size:
        raise ValueError(f"nwp_{domain}_static grid must match its dynamic domain")
    landuse = values[2]
    if not np.all(np.isfinite(landuse)) or not np.all(np.abs(landuse - np.rint(landuse)) < 1e-5):
        raise ValueError(f"nwp_{domain}_static LU_INDEX must contain finite integer categories")
    if float(np.min(landuse)) < 0.0 or float(np.max(landuse)) >= 64.0:
        raise ValueError(f"nwp_{domain}_static LU_INDEX outside [0, 63]")
    # HGT, LANDMASK, LU_INDEX, slope, sin(aspect), cos(aspect), std3, std9.
    values[0] /= 3000.0
    # LU_INDEX (channel 2) deliberately remains an integer-valued category for
    # StaticGridEncoder; it is never z-score normalized or scaled.
    values[3] /= 2.0
    values[6:] /= 1500.0
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


class MultisourceCaseDataset(Dataset):
    """Load v4 tensors while retaining physical wind/cloud fields for transport."""

    def __init__(
        self,
        root: Path,
        rows: list[dict[str, str]],
        normalization: dict,
        aod_reference: dict,
        ridge_teacher: dict[tuple[str, int], float] | None = None,
        clear_labels: dict[tuple[str, int], tuple[float, bool]] | None = None,
        include_future_satellite_auxiliary: bool = False,
        require_formal_v4: bool = False,
    ):
        self.samples = []
        self.provenance: list[dict[str, object]] = []
        for row in rows:
            with np.load(str(root / row["case_file"]), allow_pickle=False) as loaded:
                case = {name: np.asarray(loaded[name]) for name in loaded.files}
            formal_v4 = "wrf_ghi_anchor_d02" in case
            if require_formal_v4 and not formal_v4:
                raise ValueError(f"formal v4 training cannot load legacy case {row['case_file']}")
            if formal_v4:
                required = (
                    "wrf_ghi_anchor_source", "cape_cin_algorithm",
                    "satellite_scan_time_offset_hours", "satellite_unique_frame_mask",
                    "wrf_init_time_utc", "issue_offset_hours", "issue_time_utc", "valid_time_utc",
                )
                missing = [name for name in required if name not in case]
                if missing:
                    raise KeyError(f"formal v4 case {row['case_file']} is missing metadata: {missing}")
                if str(case["wrf_ghi_anchor_source"].item()) != ANCHOR_SOURCE:
                    raise ValueError(f"formal v4 case {row['case_file']} has a non-d02 WRF GHI anchor")
                if str(case["cape_cin_algorithm"].item()) != CAPE_CIN_ALGORITHM:
                    raise ValueError(f"formal v4 case {row['case_file']} uses a different CAPE/CIN diagnostic")
                case_sample_id = str(case["sample_id"].item())
                if row.get("sample_id") and case_sample_id != str(row["sample_id"]):
                    raise ValueError(f"sample_id mismatch between index and {row['case_file']}")
                row_init = str(row.get("wrf_init_time_utc", row.get("init_time_utc", ""))).strip()
                case_init = str(case["wrf_init_time_utc"].item()).strip()
                if row_init and row_init != case_init:
                    raise ValueError(f"WRF initialization mismatch between index and {row['case_file']}")
                initialization = _parse_utc(case_init)
                issue = _parse_utc(str(case["issue_time_utc"].item()))
                offset = float(case["issue_offset_hours"].item())
                if not any(abs(offset - allowed) <= 1e-6 for allowed in range(3, 67, 3)):
                    raise ValueError(f"formal v4 case {row['case_file']} has an invalid issue offset")
                if abs((issue - initialization).total_seconds() / 3600.0 - offset) > 1e-6:
                    raise ValueError(f"formal v4 case {row['case_file']} has inconsistent issue time")
                valid_times = case["valid_time_utc"].astype(str)
                if valid_times.shape != (6,) or any(
                    _parse_utc(value) != issue + timedelta(hours=lead)
                    for lead, value in enumerate(valid_times, 1)
                ):
                    raise ValueError(f"formal v4 case {row['case_file']} has inconsistent valid times")
            sample: dict[str, object] = {}
            for domain, keys in DOMAIN_KEYS.items():
                volume = _first(case, keys["volume"]).astype(np.float32)
                level_valid = _first(case, keys["valid"]).astype(bool)
                volume_valid = np.broadcast_to(level_valid[:, None], volume.shape) & np.isfinite(volume)
                available = _optional(case, keys["channel"])
                if available is not None:
                    volume_valid &= np.asarray(available, bool).reshape(1, -1, 1, 1, 1)
                surface = _first(case, keys["surface"]).astype(np.float32)
                surface_valid = np.isfinite(surface)
                surface_available = _optional(case, keys["surface_channel"])
                if surface_available is not None:
                    surface_valid &= np.asarray(surface_available, bool).reshape(1, -1, 1, 1)
                sample[f"nwp_{domain}_3d"] = torch.from_numpy(_normalize(volume, volume_valid, normalization[f"nwp_{domain}_3d"]))
                sample[f"nwp_{domain}_3d_physical"] = torch.from_numpy(np.where(volume_valid, volume, 0.0).astype(np.float32))
                sample[f"nwp_{domain}_3d_valid"] = torch.from_numpy(level_valid)
                sample[f"nwp_{domain}_surface"] = torch.from_numpy(_normalize(surface, surface_valid, normalization[f"nwp_{domain}_surface"]))
                sample[f"nwp_{domain}_static"] = torch.from_numpy(_scaled_static_grid(case, domain, volume.shape[-2:]))
                if available is None:
                    available = np.ones(volume.shape[1], dtype=bool)
                if surface_available is None:
                    surface_available = np.ones(surface.shape[1], dtype=bool)
                sample[f"nwp_{domain}_channel_available"] = torch.from_numpy(np.asarray(available, bool))
                sample[f"nwp_{domain}_surface_available"] = torch.from_numpy(np.asarray(surface_available, bool))

            sat_valid = case["satellite_valid"].astype(bool)
            satellite = _normalize(case["satellite"], sat_valid & np.isfinite(case["satellite"]), normalization["satellite"])
            scan_offsets, unique_frames = _satellite_times(case)
            solar = derived_solar(case)
            target = case["target"].astype(np.float32)
            target_mask = case["target_mask"].astype(bool)
            clear_index = target / np.maximum(solar["stable_scale"], 20.0)
            clear_target = np.clip((clear_index - 0.55) / 0.35, 0.0, 1.0).astype(np.float32)
            clear_mask = target_mask & (solar["solar_context"][:, 2] > 0.99) & (solar["stable_scale"] > 100.0)
            sample_id = str(case["sample_id"].item())
            if clear_labels is not None:
                clear_target = np.zeros(6, dtype=np.float32)
                clear_mask = np.zeros(6, dtype=bool)
                for lead in range(1, 7):
                    value = clear_labels.get((sample_id, lead))
                    if value is not None:
                        clear_target[lead - 1], clear_mask[lead - 1] = value
            teacher = np.zeros(6, dtype=np.float32)
            teacher_mask = np.zeros(6, dtype=bool)
            if ridge_teacher is not None:
                for lead in range(1, 7):
                    value = ridge_teacher.get((sample_id, lead))
                    if value is not None:
                        teacher[lead - 1] = value
                        teacher_mask[lead - 1] = True
            anchor = _first(case, ("wrf_ghi_anchor_d02", "wrf_ghi_anchor", "wrf_anchor")).astype(np.float32)
            sample.update({
                "wrf_ghi_anchor_d02": torch.from_numpy(anchor),
                "wrf_ghi_anchor": torch.from_numpy(anchor),
                "wrf_anchor": torch.from_numpy(anchor if anchor.ndim == 1 else anchor[:, anchor.shape[-2] // 2, anchor.shape[-1] // 2]),
                "ridge_teacher": torch.from_numpy(teacher),
                "ridge_teacher_mask": torch.from_numpy(teacher_mask),
                "satellite": torch.from_numpy(satellite),
                "satellite_valid": torch.from_numpy(sat_valid),
                "satellite_frame_mask": torch.from_numpy(case["satellite_frame_mask"].astype(bool)),
                "satellite_age_hours": torch.from_numpy(case["satellite_age_hours"].astype(np.float32)),
                "satellite_scan_time_offset_hours": torch.from_numpy(scan_offsets),
                "satellite_unique_frame_mask": torch.from_numpy(unique_frames),
                "satellite_source_id": torch.tensor(int(case["satellite_source_id"]), dtype=torch.long),
                "aod550": torch.from_numpy(case["aod550"].astype(np.float32)),
                "aod_reference": torch.from_numpy(aod_reference_for(case, aod_reference)),
                "aod_mask": torch.from_numpy(case["aod_mask"].astype(bool)),
                "solar_context": torch.from_numpy(solar["solar_context"]),
                "air_mass": torch.from_numpy(solar["air_mass"]),
                "physical_upper": torch.from_numpy(solar["physical_upper"]),
                "transition_support": torch.from_numpy(solar["transition_support"]),
                "static": torch.from_numpy(_static_vector(case)),
                "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
                "clear_target": torch.from_numpy(clear_target),
                "clear_target_mask": torch.from_numpy(clear_mask),
                "station_id": str(case["station_id"].item()),
                "sample_id": sample_id,
                "cycle": cycle_key(row),
                "data_role": data_role(row),
            })
            geometry = _certified_latent_geometry(case)
            transform = geometry["transform"]
            matrix_available = bool(geometry["available"])
            sample["satellite_latent_to_en_m"] = torch.from_numpy(transform)
            sample["satellite_latent_to_en_available"] = torch.tensor(matrix_available, dtype=torch.bool)
            # Always expose the optional auxiliary target with an explicit
            # validity mask.  This permits batches to mix cases with and
            # without future satellite coverage without making the first
            # dual-scale training run depend on those labels.
            if include_future_satellite_auxiliary and "satellite_future_cloud" in case:
                future_cloud = case["satellite_future_cloud"].astype(np.float32)
                future_mask = case.get("satellite_future_cloud_mask", np.isfinite(future_cloud))
            else:
                future_cloud = np.zeros((6, 1, *case["satellite"].shape[-2:]), dtype=np.float32)
                future_mask = np.zeros_like(future_cloud, dtype=bool)
            sample["satellite_future_cloud"] = torch.from_numpy(future_cloud)
            sample["satellite_future_cloud_mask"] = torch.from_numpy(np.asarray(future_mask, bool))
            self.samples.append(sample)
            self.provenance.append({
                "sample_id": sample_id,
                "case_file": str(row["case_file"]),
                "station_id": str(case["station_id"].item()),
                "cycle": cycle_key(row),
                "issue_time_utc": str(np.asarray(case["issue_time_utc"]).item()) if "issue_time_utc" in case else None,
                "source_id": int(case["satellite_source_id"].item()),
                "frame_mask": case["satellite_frame_mask"].astype(bool).tolist(),
                "unique_frame_mask": unique_frames.astype(bool).tolist(),
                "scan_time_offset_hours": scan_offsets.astype(float).tolist(),
                "frame_ids": geometry["frame_ids"],
                "scan_times_utc": geometry["scan_times_utc"],
                "frame_times_utc": geometry["frame_times_utc"],
                "latent_to_en_available": bool(matrix_available),
                "latent_to_en_m": transform.astype(float).tolist() if geometry["matrix_present"] else None,
                "matrix_convention": geometry["matrix_convention"],
                "expected_matrix_convention": geometry["expected_matrix_convention"],
                "latent_to_en_provenance_ref": geometry["provenance_ref"],
                "availability_flag_present": geometry["availability_flag_present"],
                "availability_flag_value": geometry["availability_flag_value"],
                "matrix_finite_nonsingular": geometry["matrix_finite_nonsingular"],
                "declared_grid_sha256": geometry["declared_grid_sha256"],
                "computed_grid_sha256": geometry["computed_grid_sha256"],
                "grid_coordinate_fingerprints": geometry["grid_coordinate_fingerprints"],
                "has_coordinate_grid_pair": geometry["has_coordinate_grid_pair"],
                "frame_metadata_complete": geometry["frame_metadata_complete"],
                "geometry_source": geometry["geometry_source"],
                "resampling_grid_id": geometry["resampling_grid_id"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]
