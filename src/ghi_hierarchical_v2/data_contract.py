"""Data-contract helpers for the direct 1--6 h multimodal experiment.

The functions in this module deliberately have no dependency on legacy model
code.  They normalize the heterogeneous satellite products and derive solar
geometry for the exact hourly averaging interval used by WRF ACSWDNB and the
ground-radiation labels: ``[valid_time - 1 h, valid_time)``.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Tuple

import numpy as np


EXPECTED_CASES = 799
EXPECTED_LEADS = (1, 2, 3, 4, 5, 6)
CANONICAL_SATELLITE_CHANNELS = (
    "COD",
    "cloud_fraction",
    "cloud_top_pressure_hpa",
    "CER_liq_um",
    "CER_ice_um",
    "ahi_band11_brightness",
    "gridsat_irwin_bt_k",
)
SATELLITE_SOURCE_IDS = {"GOES": 0, "Himawari": 1, "GridSat": 2}
UTC = timezone.utc


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if len(text) >= 6 and text[-6] in "+-" and text[-3] == ":":
        parsed = datetime.strptime(text[:-6], "%Y-%m-%dT%H:%M:%S")
        sign = 1 if text[-6] == "+" else -1
        offset = timedelta(hours=int(text[-5:-3]), minutes=int(text[-2:]))
        parsed = parsed.replace(tzinfo=timezone(sign * offset))
    else:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _area_pool_63_to_32(values: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validity-aware area pooling without scipy or image-library dependencies."""

    if values.shape[:2] != (63, 63) or valid.shape != values.shape:
        raise ValueError("GOES pooling expects matching [63,63,C] arrays")
    channels = values.shape[2]
    output = np.zeros((32, 32, channels), dtype=np.float32)
    output_valid = np.zeros((32, 32, channels), dtype=bool)
    edges = np.rint(np.linspace(0, 63, 33)).astype(int)
    edges[0], edges[-1] = 0, 63
    for y in range(32):
        for x in range(32):
            block_valid = valid[edges[y] : edges[y + 1], edges[x] : edges[x + 1], :]
            block = values[edges[y] : edges[y + 1], edges[x] : edges[x + 1], :]
            count = block_valid.sum(axis=(0, 1))
            numerator = np.where(block_valid, block, 0.0).sum(axis=(0, 1), dtype=np.float64)
            active = count > 0
            output[y, x, active] = (numerator[active] / count[active]).astype(np.float32)
            output_valid[y, x, active] = True
    return output, output_valid


def canonicalize_satellite_frame(
    window_mean: np.ndarray,
    validity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Convert one native satellite frame to ``[7,32,32]`` plus masks.

    Native channel positions already follow the canonical schema: GOES uses
    channels 0--4, Himawari channel 5, and GridSat channel 6.  Inactive
    channels remain zero but are always marked invalid.
    """

    values = np.asarray(window_mean, dtype=np.float32)
    valid = np.asarray(validity, dtype=bool)
    if values.shape != valid.shape or values.ndim != 3:
        raise ValueError("satellite window_mean and validity must be matching HWC arrays")
    if values.shape == (63, 63, 5):
        values, valid = _area_pool_63_to_32(values, valid)
        source = "GOES"
    elif values.shape == (32, 32, 6):
        source = "Himawari"
    elif values.shape == (32, 32, 7):
        source = "GridSat"
    else:
        raise ValueError("unsupported satellite native shape: %r" % (values.shape,))

    output = np.zeros((7, 32, 32), dtype=np.float32)
    output_valid = np.zeros((7, 32, 32), dtype=bool)
    channels = values.shape[2]
    transposed = np.moveaxis(values, -1, 0)
    transposed_valid = np.moveaxis(valid, -1, 0) & np.isfinite(transposed)
    output[:channels] = np.where(transposed_valid, transposed, 0.0)
    output_valid[:channels] = transposed_valid
    channel_mask = output_valid.reshape(7, -1).any(axis=1)

    expected = {
        "GOES": np.asarray([1, 1, 1, 1, 1, 0, 0], dtype=bool),
        "Himawari": np.asarray([0, 0, 0, 0, 0, 1, 0], dtype=bool),
        "GridSat": np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=bool),
    }[source]
    if np.any(channel_mask & ~expected):
        raise ValueError("active satellite channels contradict the source contract")
    if not np.any(channel_mask & expected):
        raise ValueError("satellite frame has no valid source channel")
    return output, output_valid, channel_mask, SATELLITE_SOURCE_IDS[source]


def solar_cosine(moment: datetime, latitude: float, longitude: float) -> float:
    """NOAA-style solar-zenith cosine for an aware UTC datetime."""

    moment = moment.astimezone(UTC)
    day = moment.timetuple().tm_yday
    hour = moment.hour + moment.minute / 60.0 + moment.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    solar_minutes = (
        moment.hour * 60.0
        + moment.minute
        + moment.second / 60.0
        + equation_of_time
        + 4.0 * longitude
    ) % 1440.0
    hour_angle = math.radians(solar_minutes / 4.0 - 180.0)
    latitude_radians = math.radians(latitude)
    return float(
        math.sin(latitude_radians) * math.sin(declination)
        + math.cos(latitude_radians) * math.cos(declination) * math.cos(hour_angle)
    )


def _haurwitz(clear_cosine: float) -> float:
    if clear_cosine <= 0.0:
        return 0.0
    return float(1098.0 * clear_cosine * math.exp(-0.059 / clear_cosine))


def solar_hour_features(
    valid_time: datetime,
    latitude: float,
    longitude: float,
    substeps: int = 12,
) -> Dict[str, object]:
    """Derive geometry over the same hour represented by an hourly GHI target."""

    if substeps <= 0:
        raise ValueError("substeps must be positive")
    interval_start = valid_time.astimezone(UTC) - timedelta(hours=1)
    centers = [
        interval_start + timedelta(seconds=(index + 0.5) * 3600.0 / substeps)
        for index in range(substeps)
    ]
    raw = np.asarray(
        [solar_cosine(moment, latitude, longitude) for moment in centers], dtype=np.float32
    )
    positive = np.maximum(raw, 0.0)
    daylight = raw > 0.0
    clear = np.asarray([_haurwitz(float(value)) for value in positive], dtype=np.float32)
    fraction = float(daylight.mean())
    # Peaks at a half-lit hour and is zero for wholly day/night hours.
    transition = float(4.0 * fraction * (1.0 - fraction))
    return {
        "subhour_cos_sza": raw,
        "cos_sza_mean": float(positive.mean()),
        "daylight_fraction": fraction,
        "transition_geometry": transition,
        "clear_ghi_wm2": float(clear.mean()),
    }


def training_only_aod_reference(
    rows: Iterable[dict],
    training_cycles: Iterable[str],
) -> dict:
    """Build a station-month AOD reference with a station fallback.

    The returned object records the exact training-cycle provenance so an audit
    can prove that no held-out or test value influenced the background.
    """

    cycle_set = set(training_cycles)
    station_month: Dict[Tuple[str, str], list] = {}
    station: Dict[str, list] = {}
    all_values = []
    for row in rows:
        cycle = parse_utc(row["wrf_init_time_utc"]).strftime("%Y%m%d%H")
        if cycle not in cycle_set or not row.get("aod550"):
            continue
        value = float(row["aod550"])
        month = parse_utc(row["valid_time_utc"]).strftime("%m")
        station_id = row["station_id"]
        station_month.setdefault((station_id, month), []).append(value)
        station.setdefault(station_id, []).append(value)
        all_values.append(value)
    if not all_values:
        raise ValueError("no AOD rows belong to the supplied training cycles")
    return {
        "training_cycles": sorted(cycle_set),
        "station_month_median": {
            "%s|%s" % key: float(np.median(values))
            for key, values in sorted(station_month.items())
        },
        "station_median": {
            key: float(np.median(values)) for key, values in sorted(station.items())
        },
        "global_median": float(np.median(all_values)),
        "fallback": "station_month_then_station_then_global",
    }
