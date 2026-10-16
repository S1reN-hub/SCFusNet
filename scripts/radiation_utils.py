"""Shared radiation and solar-geometry utilities for the SSRD pipeline.

The functions in this module deliberately keep radiation *energy* (J m-2)
and radiation *flux* (W m-2) distinct.  ERA5 hourly radiation downloaded
from CDS is normally an accumulation over the preceding hour, while GFS
DSWRF is commonly an average flux over a forecast interval.  Callers must
therefore pass the intended accumulation interval explicitly when metadata
does not provide it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np


SOLAR_CONSTANT_WM2 = 1361.0


def _normalise_units(units: str | None) -> str:
    return (units or "").lower().replace(" ", "").replace("**", "^")


def radiation_to_wm2(values, units: str | None, accumulation_seconds: float = 3600.0):
    """Convert a radiation array to mean W m-2.

    Parameters
    ----------
    values:
        NumPy-compatible radiation values.
    units:
        Source units from the NetCDF/GRIB variable attributes.
    accumulation_seconds:
        Length of the accumulation interval for energy units.  ERA5 hourly
        reanalysis radiation normally uses 3600 seconds.

    Raises
    ------
    ValueError
        If units are missing/unknown or the interval is non-positive.  A
        silent unit guess is intentionally forbidden.
    """
    arr = np.asarray(values, dtype=np.float64)
    unit = _normalise_units(units)
    if "w" in unit and ("m-2" in unit or "m^-2" in unit):
        return arr.astype(np.float32)
    if "j" in unit and ("m-2" in unit or "m^-2" in unit):
        if accumulation_seconds <= 0:
            raise ValueError("accumulation_seconds must be positive for J m-2 data")
        return (arr / float(accumulation_seconds)).astype(np.float32)
    raise ValueError(
        f"Unsupported radiation units {units!r}; pass explicit CF/GRIB units "
        "instead of guessing whether values are energy or flux."
    )


def find_coord_name(obj, candidates: Sequence[str]) -> str:
    """Return the first coordinate/dimension name found in an xarray object."""
    names = set(getattr(obj, "coords", {})) | set(getattr(obj, "dims", {}))
    for name in candidates:
        if name in names:
            return name
    raise KeyError(f"None of the coordinates {list(candidates)} exist; found {sorted(names)}")


def find_data_var(dataset, preferred: Sequence[str]) -> str:
    """Find a data variable by short name, then by GRIB shortName attribute."""
    for name in preferred:
        if name in dataset.data_vars:
            return name
    wanted = {x.lower() for x in preferred}
    for name, var in dataset.data_vars.items():
        aliases = {
            name.lower(),
            str(var.attrs.get("GRIB_shortName", "")).lower(),
            str(var.attrs.get("short_name", "")).lower(),
        }
        if aliases & wanted:
            return name
    raise KeyError(f"Could not find {list(preferred)} in {list(dataset.data_vars)}")


def _to_datetime64_ns(times: Iterable) -> np.ndarray:
    return np.asarray(times).astype("datetime64[ns]")


def interval_midpoint(times: Iterable, accumulation_seconds: float) -> np.ndarray:
    """Return centres of intervals ending at the supplied valid timestamps."""
    if accumulation_seconds <= 0:
        raise ValueError("accumulation_seconds must be positive")
    ts = _to_datetime64_ns(times)
    half_ns = int(round(accumulation_seconds * 0.5 * 1e9))
    return ts - np.timedelta64(half_ns, "ns")


def solar_position(
    times: Iterable,
    lat: float,
    lon: float,
    clip_below_horizon: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate cosine zenith angle and Earth-Sun distance correction.

    Uses the NOAA fractional-year approximation, including equation of time.
    Input timestamps are interpreted as UTC.
    """
    ts = _to_datetime64_ns(times)
    seconds = (ts - ts.astype("datetime64[Y]")).astype("timedelta64[s]").astype(np.float64)
    day = np.floor(seconds / 86400.0) + 1.0
    hour = (seconds % 86400.0) / 3600.0

    gamma = 2.0 * np.pi / 365.0 * (day - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    true_solar_minutes = (hour * 60.0 + eqtime + 4.0 * lon) % 1440.0
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    lat_rad = np.deg2rad(lat)
    mu0 = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    if clip_below_horizon:
        mu0 = np.clip(mu0, 0.0, 1.0)
    else:
        mu0 = np.clip(mu0, -1.0, 1.0)
    mu0 = mu0.astype(np.float32)

    distance_factor = (
        1.000110
        + 0.034221 * np.cos(gamma)
        + 0.001280 * np.sin(gamma)
        + 0.000719 * np.cos(2 * gamma)
        + 0.000077 * np.sin(2 * gamma)
    ).astype(np.float32)
    return mu0, distance_factor


def haurwitz_clear_sky_ghi(mu0, distance_factor=None) -> np.ndarray:
    """Return a deterministic clear-sky GHI proxy in W m-2.

    This lightweight Haurwitz-style proxy is used to make the first pipeline
    self-contained.  It is not a replacement for a turbidity-aware clear-sky
    model; the dataset metadata records this method so it cannot be confused
    with ERA5 SSRDC or pvlib/Ineichen values.
    """
    mu = np.asarray(mu0, dtype=np.float64)
    safe = np.maximum(mu, 1e-4)
    ghi = 1098.0 * mu * np.exp(-0.059 / safe)
    if distance_factor is not None:
        ghi = ghi * np.asarray(distance_factor, dtype=np.float64)
    return np.where(mu > 0, ghi, 0.0).astype(np.float32)


def derive_radiation_components(ssrd_wm2, fdir_wm2, negative_tolerance: float = 2.0):
    """Build non-negative SSRD/FDIR/DHI targets and return a QC mask.

    Values below ``-negative_tolerance`` in raw SSRD-FDIR are marked invalid;
    small negative residuals are treated as rounding/alignment noise and
    clipped to zero.
    """
    ssrd = np.asarray(ssrd_wm2, dtype=np.float32)
    fdir = np.asarray(fdir_wm2, dtype=np.float32)
    raw_dhi = ssrd - fdir
    finite = np.isfinite(ssrd) & np.isfinite(fdir)
    valid = finite & (ssrd >= -negative_tolerance) & (fdir >= -negative_tolerance)
    valid &= raw_dhi >= -negative_tolerance
    ssrd = np.maximum(ssrd, 0.0)
    fdir = np.clip(fdir, 0.0, ssrd)
    dhi = np.maximum(ssrd - fdir, 0.0)
    return ssrd, fdir, dhi, valid


@dataclass(frozen=True)
class RadiationSemantics:
    source: str
    variable: str
    source_units: str
    accumulation_seconds: float
    output_units: str = "W m-2"
