"""Load and validate the canonical SWRadNet SSRD data contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEFAULT_SCHEMA_PATH = Path(__file__).parent.parent / "configs" / "era5_ssrd_schema.json"
DEFAULT_GFS_SCHEMA_PATH = Path(__file__).parent.parent / "configs" / "gfs_swradnet_schema.json"
REQUIRED_TOP_LEVEL = {
    "schema_version", "site", "time", "atmosphere", "surface", "solar",
    "radiation", "splits", "paths", "legacy",
}


def schema_hash(schema: dict) -> str:
    canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_schema_dict(schema: dict) -> None:
    missing = REQUIRED_TOP_LEVEL - set(schema)
    if missing:
        raise ValueError(f"schema missing top-level keys: {sorted(missing)}")
    atmos = schema["atmosphere"]
    surface = schema["surface"]
    solar = schema["solar"]
    radiation = schema["radiation"]
    if len(atmos["names"]) != len(atmos["units"]):
        raise ValueError("atmosphere names/units length mismatch")
    if len(surface["names"]) != len(surface["units"]):
        raise ValueError("surface names/units length mismatch")
    if len(solar["names"]) != len(solar["units"]):
        raise ValueError("solar names/units length mismatch")
    levels = atmos["pressure_levels_hpa"]
    if len(levels) < 2 or any(levels[i] <= levels[i + 1] for i in range(len(levels) - 1)):
        raise ValueError("pressure levels must be unique and strictly descending")
    if radiation["target_names"] != ["ssrd", "fdir", "dhi"]:
        raise ValueError("target order must be ssrd,fdir,dhi")
    splits = schema["splits"]
    groups = [set(splits[k]) for k in ("train_years", "val_years", "test_years")]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("train/val/test years overlap")
    if schema["time"]["solar_geometry_time"] != "interval_mean_midpoint_quadrature":
        raise ValueError("solar geometry must be averaged over the radiation interval")
    if int(schema["time"].get("solar_quadrature_substeps", 0)) < 2:
        raise ValueError("solar quadrature requires at least two substeps")
    if "grid_semantics" not in surface:
        raise ValueError("surface_grid semantics must be explicit")


def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> tuple[dict, str]:
    path = Path(path)
    schema = json.loads(path.read_text(encoding="utf-8"))
    validate_schema_dict(schema)
    return schema, schema_hash(schema)


def load_gfs_schema(path: str | Path = DEFAULT_GFS_SCHEMA_PATH) -> tuple[dict, str]:
    """Load the operational multi-station GFS contract.

    The former ``configs/ssrd_schema.json`` is only a compatibility pointer.
    Keeping the GFS and legacy single-site ERA5 contracts separate prevents a
    pointer file from being mistaken for a complete training schema.
    """
    path = Path(path)
    schema = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "input_policy", "archive", "atmosphere", "surface",
        "nwp_radiation_baseline", "deployment_available_features",
        "sample_identity", "splits", "stations", "paths",
    }
    missing = required - set(schema)
    if missing:
        raise ValueError(f"GFS schema missing top-level keys: {sorted(missing)}")
    if schema["atmosphere"]["tensor_order"] != "C,H,W,L":
        raise ValueError("GFS atmosphere tensor order must be C,H,W,L")
    if schema["surface"]["tensor_order"] != "C,H,W":
        raise ValueError("GFS surface tensor order must be C,H,W")
    if schema["sample_identity"]["valid_time"] != "init_time_utc + forecast_hour":
        raise ValueError("GFS valid-time rule must be explicit and UTC")
    return schema, schema_hash(schema)


def load_training_contract(path: str | Path) -> tuple[dict, str]:
    """Load either a GFS operational contract or an ERA5 pretraining contract."""
    path = Path(path)
    schema = json.loads(path.read_text(encoding="utf-8"))
    if "input_policy" in schema and "sample_identity" in schema:
        return load_gfs_schema(path)
    validate_schema_dict(schema)
    return schema, schema_hash(schema)


def assert_metadata_matches_schema(metadata: dict, schema: dict, allow_legacy: bool = False) -> None:
    version = metadata.get("schema_version")
    if version == schema["legacy"]["three_level_schema_version"]:
        if allow_legacy:
            return
        raise ValueError("legacy_3level sample is forbidden in the ten-level main pipeline")
    if version != schema["schema_version"]:
        raise ValueError(f"schema version mismatch: sample={version!r}, expected={schema['schema_version']!r}")
    checks = {
        "atmos_names": schema["atmosphere"]["names"],
        "surface_names": schema["surface"]["names"],
        "solar_names": schema["solar"]["names"],
        "target_names": schema["radiation"]["target_names"],
        "pressure_levels_hpa": schema["atmosphere"]["pressure_levels_hpa"],
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata {key} mismatch: {metadata.get(key)!r} != {expected!r}")
