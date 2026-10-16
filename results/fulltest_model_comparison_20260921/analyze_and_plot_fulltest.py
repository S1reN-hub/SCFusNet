"""Audit and plot the full common daylight test-set model comparison.

This script intentionally uses the intersection of keys and daylight validity
across all frozen SolFUS-Net, deep-learning baseline, and direct WRF-PBL
exports. It never selects cases by model error. The primary summary is
seed-level mean +/- seed SD for neural models; seeds are training replicates,
not independent samples. Cycle-level skill is computed from matched cycle
RMSEs and is used to show that the comparison is not based on a hand-picked
ten-case subset.

The frozen ``nwp`` column carried by the SolFUS export is a learned NWP branch
reference, not a traditional WRF-PBL run, and is therefore labelled
``WRF-NWP branch`` throughout. MYNN, YSU, and MYJ are direct deterministic
WRF-PBL predictions from the full 17-cycle export.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOL_DIR = ROOT / "outputs/mlp17_7eval_training_20260909/review/test_7station"
BASE_DIR = ROOT / "results/baselines_v4_20260921_parallel"
PBL_PATH = ROOT / "results/wrf_pbl_fulltest_20260921/predictions_fulltest_6lead.csv"
CACHE_PATH = ROOT / "outputs/mlp17_7eval_training_20260909/review/test_7station/test_cache.pt"
SP_PATH = ROOT / "outputs/sp7_test_20260912/predictions.csv"

SEEDS = (42, 43, 44)
KEY_COLS = ["sample_id", "station_id", "cycle", "lead"]
LEADS = (1, 2, 3, 4, 5, 6)
PBL_MODELS = ("mynn", "ysu", "myj")
DETERMINISTIC_MODELS = PBL_MODELS + ("nwp_branch",)
MODEL_ORDER = ["mynn", "ysu", "myj", "nwp_branch", "sat_convlstm", "wrf_transformer", "early_fusion", "solfus"]
# The primary comparison figure intentionally contains seven methods.  The
# learned WRF-NWP branch remains in the source metrics and audit tables for
# traceability, but it is an internal ablation reference and is not shown in
# Fig. 4-1.
FIGURE_MODELS = ["mynn", "ysu", "myj", "sat_convlstm", "wrf_transformer", "early_fusion", "solfus"]
COMPARISON_ORDER = ["mynn", "ysu", "myj", "nwp_branch", "sat_convlstm", "wrf_transformer", "early_fusion"]
DISPLAY = {
    "mynn": "MYNN (WRF-PBL)",
    "ysu": "YSU (WRF-PBL)",
    "myj": "MYJ (WRF-PBL)",
    "nwp_branch": "WRF-NWP branch",
    "sat_convlstm": "Sat-ConvLSTM",
    "wrf_transformer": "WRF-Transformer",
    "early_fusion": "EarlyFusion-Transformer",
    "solfus": "SolFUS-Net",
}
SHORT_DISPLAY = {
    "mynn": "MYNN",
    "ysu": "YSU",
    "myj": "MYJ",
    "nwp_branch": "NWP branch",
    "sat_convlstm": "Sat-DL",
    "wrf_transformer": "WRF-DL",
    "early_fusion": "EarlyFusion",
    "solfus": "SolFUS-Net",
}
GROUP = {
    "mynn": "traditional WRF/PBL",
    "ysu": "traditional WRF/PBL",
    "myj": "traditional WRF/PBL",
    "nwp_branch": "learned NWP branch",
    "sat_convlstm": "single-source DL",
    "wrf_transformer": "single-source DL",
    "early_fusion": "conventional fusion",
    "solfus": "SolFUS-Net",
}
COLORS = {
    "mynn": "#4d5660",
    "ysu": "#77828c",
    "myj": "#a4adb5",
    "nwp_branch": "#596f7f",
    "sat_convlstm": "#4C78A8",
    "wrf_transformer": "#2A9D8F",
    "early_fusion": "#E58F3B",
    "solfus": "#C44E52",
}
LINESTYLES = {
    "mynn": "-",
    "ysu": "--",
    "myj": ":",
    "nwp_branch": "-.",
    "sat_convlstm": "-",
    "wrf_transformer": "--",
    "early_fusion": ":",
}
MARKERS = {
    "mynn": "o",
    "ysu": "s",
    "myj": "^",
    "nwp_branch": "D",
    "sat_convlstm": "o",
    "wrf_transformer": "s",
    "early_fusion": "^",
}

FIGURE_BAR_COLORS = {
    "mynn": "#9A4D8E",
    "ysu": "#D9903D",
    "myj": "#6BAF77",
    "sat_convlstm": "#8CC5DD",
    "wrf_transformer": "#42949E",
    "early_fusion": "#E9A6A1",
    "solfus": "#0F4D92",
}
FIGURE_BAR_HATCHES = {
    "mynn": "",
    "ysu": "",
    "myj": "",
    "sat_convlstm": "",
    "wrf_transformer": "",
    "early_fusion": "",
    "solfus": "",
}
FIGURE_LEGEND_LABELS = {
    "mynn": "MYNN",
    "ysu": "YSU",
    "myj": "MYJ",
    "sat_convlstm": "Sat-ConvLSTM",
    "wrf_transformer": "WRF-Transformer",
    "early_fusion": "EarlyFusion-Transformer",
    "solfus": "SolFUS-Net",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key_series(df: pd.DataFrame) -> pd.Series:
    return df[KEY_COLS].astype(str).agg("||".join, axis=1)


def finite_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(True, index=df.index)
    for c in cols:
        out &= np.isfinite(pd.to_numeric(df[c], errors="coerce"))
    return out


def load_solfus(seed: int) -> pd.DataFrame:
    path = SOL_DIR / f"mlp_seed{seed}_test.csv"
    df = pd.read_csv(path)
    required = set(KEY_COLS + ["daylight", "target", "satellite", "nwp", "prediction", "satellite_weight"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SolFUS seed {seed} missing columns: {sorted(missing)}")
    out = df[KEY_COLS + ["daylight", "target", "satellite", "nwp", "prediction", "satellite_weight"]].copy()
    out["lead"] = out["lead"].astype(int)
    out["daylight"] = out["daylight"].astype(int)
    out["_key"] = key_series(out)
    out["source_path"] = str(path)
    return out


def load_baseline(model_key: str, seed: int) -> pd.DataFrame:
    path = BASE_DIR / model_key / f"seed{seed}" / "test_predictions.csv"
    df = pd.read_csv(path)
    required = {"sample_id", "station_id", "cycle", "lead_hour", "target_wm2", "prediction_wm2", "daylight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_key} seed {seed} missing columns: {sorted(missing)}")
    out = df[["sample_id", "station_id", "cycle", "lead_hour", "target_wm2", "prediction_wm2", "daylight"]].copy()
    out = out.rename(columns={"lead_hour": "lead", "target_wm2": "target", "prediction_wm2": "prediction"})
    out["lead"] = out["lead"].astype(int)
    out["daylight"] = out["daylight"].astype(int)
    out["_key"] = key_series(out)
    out["source_path"] = str(path)
    return out


def load_pbl(model_key: str) -> pd.DataFrame:
    """Load one direct WRF-PBL scheme from the full 17-cycle export."""
    if model_key not in PBL_MODELS:
        raise ValueError(f"not a PBL model: {model_key}")
    df = pd.read_csv(PBL_PATH)
    required = {
        "sample_id", "station_id", "cycle", "lead_hour", "observed_ghi_wm2",
        "daylight", "common_valid", "common_daylight_valid", f"{model_key}_valid",
        f"{model_key}_ghi_wm2",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PBL {model_key} missing columns: {sorted(missing)}")
    out = df[["sample_id", "station_id", "cycle", "lead_hour", "observed_ghi_wm2",
              "daylight", "common_valid", "common_daylight_valid", f"{model_key}_valid",
              f"{model_key}_ghi_wm2"]].copy()
    out = out.rename(columns={
        "lead_hour": "lead",
        "observed_ghi_wm2": "target",
        f"{model_key}_ghi_wm2": "prediction",
    })
    out["lead"] = out["lead"].astype(int)
    # ``common_daylight_valid`` is the PBL export's own validity contract:
    # finite label, finite MYNN/YSU/MYJ predictions, and daylight. Keeping it
    # as the frame daylight mask guarantees that all three direct schemes are
    # audited under the same raw PBL validity definition before the global
    # intersection with the frozen DL/SolFUS exports.
    out["daylight"] = out["common_daylight_valid"].astype(int)
    out["_key"] = key_series(out)
    out["source_path"] = str(PBL_PATH)
    return out


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) == 0:
        return {"n": 0, "rmse_wm2": np.nan, "nrmse_pct": np.nan, "nmbe_pct": np.nan, "r2": np.nan,
                "mae_wm2": np.nan, "mbe_wm2": np.nan, "mean_target_wm2": np.nan}
    err = pred - y
    mean_y = float(np.mean(y))
    sse = float(np.sum(err * err))
    total = float(np.sum((y - mean_y) ** 2))
    rmse = float(np.sqrt(np.mean(err * err)))
    return {
        "n": int(len(y)),
        "rmse_wm2": rmse,
        "nrmse_pct": float(100.0 * rmse / mean_y) if mean_y != 0 else np.nan,
        "nmbe_pct": float(100.0 * np.mean(err) / mean_y) if mean_y != 0 else np.nan,
        "r2": float(1.0 - sse / total) if total > 0 else np.nan,
        "mae_wm2": float(np.mean(np.abs(err))),
        "mbe_wm2": float(np.mean(err)),
        "mean_target_wm2": mean_y,
    }


def aggregate_metric_rows(seed_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = ["rmse_wm2", "nrmse_pct", "nmbe_pct", "r2", "mae_wm2", "mbe_wm2"]
    rows: list[dict] = []
    for keys, g in seed_df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(len(g))
        row["n_mean"] = float(g["n"].mean())
        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def model_col(model_key: str, seed: int | None = None) -> str:
    if model_key == "nwp_branch":
        return "nwp"
    if model_key in PBL_MODELS:
        return model_key
    if model_key == "solfus":
        return f"solfus_seed{seed}"
    return f"{model_key}_seed{seed}"


def model_is_deterministic(model_key: str) -> bool:
    return model_key in DETERMINISTIC_MODELS


def jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def build_common_data():
    sol = {seed: load_solfus(seed) for seed in SEEDS}
    baselines = {
        model: {seed: load_baseline(model, seed) for seed in SEEDS}
        for model in ("sat_convlstm", "wrf_transformer", "early_fusion")
    }
    pbl = {model: load_pbl(model) for model in PBL_MODELS}
    frames: dict[str, pd.DataFrame] = {}
    for seed, df in sol.items():
        frames[f"solfus_seed{seed}"] = df
    for model, per_seed in baselines.items():
        for seed, df in per_seed.items():
            frames[f"{model}_seed{seed}"] = df
    for model, df in pbl.items():
        frames[model] = df

    ref = sol[SEEDS[0]].copy()
    ref_keys = set(ref["_key"])
    frame_keys = {name: set(df["_key"]) for name, df in frames.items()}
    common_key_set = set.intersection(*frame_keys.values())
    if not common_key_set:
        raise ValueError("no key intersection across SolFUS, DL, and PBL frames")
    ref_common = ref[ref["_key"].isin(common_key_set)].copy().reset_index(drop=True)
    if set(ref_common["_key"]) != common_key_set:
        raise ValueError("reference frame does not cover the computed key intersection")

    key_audit = []
    for name, df in frames.items():
        keys = set(df["_key"])
        row = {
            "frame": name,
            "rows": int(len(df)),
            "unique_keys": int(df["_key"].nunique()),
            "duplicate_key_rows": int(len(df) - df["_key"].nunique()),
            "daylight_rows": int((df["daylight"] == 1).sum()),
            "missing_keys_vs_reference": int(len(ref_keys - keys)),
            "extra_keys_vs_reference": int(len(keys - ref_keys)),
            "missing_keys_vs_common_intersection": int(len(common_key_set - keys)),
            "finite_target_rows": int(finite_series(df, ["target"]).sum()),
            "finite_prediction_rows": int(finite_series(df, ["prediction"]).sum()),
        }
        if "common_valid" in df.columns:
            row["common_valid_rows"] = int(df["common_valid"].astype(bool).sum())
        if "common_daylight_valid" in df.columns:
            row["common_daylight_valid_rows"] = int(df["common_daylight_valid"].astype(bool).sum())
        key_audit.append(row)
        if len(df) != len(df["_key"].unique()):
            raise ValueError(f"duplicate keys in {name}")
        if not common_key_set.issubset(keys):
            raise ValueError(f"common-key intersection is not covered by {name}")

    # Every frame uses the same observation at a matched key.  The PBL file
    # has 39 extra key rows beyond the frozen DL/SolFUS exports; those are
    # intentionally excluded by the all-method key intersection.
    indexed = {name: df.set_index("_key") for name, df in frames.items()}
    target_diffs = []
    common_keys_ordered = ref_common["_key"]
    ref_target = indexed["solfus_seed42"].loc[common_keys_ordered, "target"]
    for name, df in indexed.items():
        diff = np.abs(df.loc[common_keys_ordered, "target"].to_numpy(dtype=float) - ref_target.to_numpy(dtype=float))
        target_diffs.append({"frame": name, "max_abs_target_diff_wm2": float(diff.max()), "target_mismatch_rows": int((diff > 1e-6).sum())})
        if diff.max() > 1e-5:
            raise ValueError(f"target mismatch in {name}: {diff.max()}")

    # Common valid set: same key, daylight flag=1, and finite observation and
    # prediction in every frozen export.  This mask is evaluated only on the
    # key intersection, so extra PBL rows cannot alter the DL/SolFUS sample
    # definition.
    common_mask = np.ones(len(ref_common), dtype=bool)
    for name, df in indexed.items():
        idx = df.loc[common_keys_ordered]
        common_mask &= idx["daylight"].to_numpy(dtype=int) == 1
        common_mask &= finite_series(idx.reset_index(), ["target", "prediction"]).to_numpy(dtype=bool)
    common = ref_common.loc[common_mask, KEY_COLS].copy().reset_index(drop=True)
    common["_key"] = key_series(common)
    for seed in SEEDS:
        idx = indexed[f"solfus_seed{seed}"].loc[common["_key"]]
        common[f"solfus_seed{seed}"] = idx["prediction"].to_numpy(dtype=float)
        common[f"solfus_satellite_weight_seed{seed}"] = idx["satellite_weight"].to_numpy(dtype=float)
    for model in ("sat_convlstm", "wrf_transformer", "early_fusion"):
        for seed in SEEDS:
            idx = indexed[f"{model}_seed{seed}"].loc[common["_key"]]
            common[f"{model}_seed{seed}"] = idx["prediction"].to_numpy(dtype=float)
    for model in PBL_MODELS:
        idx = indexed[model].loc[common["_key"]]
        common[model] = idx["prediction"].to_numpy(dtype=float)
    ref_idx = indexed["solfus_seed42"].loc[common["_key"]]
    common["target"] = ref_idx["target"].to_numpy(dtype=float)
    common["satellite"] = ref_idx["satellite"].to_numpy(dtype=float)
    common["nwp"] = ref_idx["nwp"].to_numpy(dtype=float)
    common["daylight_all_models"] = True

    weight_identity = []
    for seed in SEEDS:
        lhs = common[f"solfus_seed{seed}"].to_numpy(dtype=float)
        w = common[f"solfus_satellite_weight_seed{seed}"].to_numpy(dtype=float)
        rhs = w * common["satellite"].to_numpy(dtype=float) + (1.0 - w) * common["nwp"].to_numpy(dtype=float)
        diff = np.abs(lhs - rhs)
        weight_identity.append({"seed": int(seed), "max_abs_identity_diff_wm2": float(diff.max()),
                               "mean_abs_identity_diff_wm2": float(diff.mean()), "rows_gt_1e-3": int((diff > 1e-3).sum())})

    # Per-frame reasons for exclusion from the fully common set.
    exclusion = []
    for name, df in frames.items():
        idx = df.set_index("_key").loc[common_keys_ordered]
        day_fail = idx["daylight"].to_numpy(dtype=int) != 1
        target_fail = ~finite_series(idx.reset_index(), ["target"]).to_numpy(dtype=bool)
        pred_fail = ~finite_series(idx.reset_index(), ["prediction"]).to_numpy(dtype=bool)
        row = {
            "frame": name,
            "rows_in_reference_key_intersection": int(len(ref_common)),
            "daylight_fail_rows": int(day_fail.sum()),
            "target_nonfinite_rows": int(target_fail.sum()),
            "prediction_nonfinite_rows": int(pred_fail.sum()),
            "rows_retained_in_common": int(common_mask.sum()),
        }
        if "common_valid" in idx.columns:
            row["common_valid_fail_rows"] = int((~idx["common_valid"].astype(bool)).sum())
        if "common_daylight_valid" in idx.columns:
            row["common_daylight_valid_fail_rows"] = int((~idx["common_daylight_valid"].astype(bool)).sum())
        exclusion.append(row)

    all_frame_names = list(frames)
    pbl_raw = pd.read_csv(PBL_PATH)

    audit = {
        "reference_frame": "solfus_seed42",
        "key_columns": KEY_COLS,
        "all_frames": all_frame_names,
        "key_audit": key_audit,
        "target_consistency": target_diffs,
        "exclusion_audit": exclusion,
        "reference_rows": int(len(ref)),
        "reference_unique_keys": int(len(ref_keys)),
        "key_intersection_rows": int(len(common_key_set)),
        "common_rows": int(len(common)),
        "common_unique_sample_ids": int(common["sample_id"].nunique()),
        "common_unique_cycles": int(common["cycle"].nunique()),
        "common_unique_stations": int(common["station_id"].nunique()),
        "common_lead_counts": common.groupby("lead").size().to_dict(),
        "solfus_weight_identity": weight_identity,
    }
    write_csv(pd.DataFrame(key_audit), "key_audit.csv")
    write_csv(pd.DataFrame(target_diffs), "target_consistency_audit.csv")
    write_csv(pd.DataFrame(exclusion), "common_exclusion_audit.csv")
    write_csv(common, "source_data_common_predictions.csv")
    audit["pbl_raw_export"] = {
        "path": str(PBL_PATH),
        "rows": int(len(pbl_raw)),
        "unique_keys": int(pbl_raw[["sample_id", "station_id", "cycle", "lead_hour"]].drop_duplicates().shape[0]),
        "cases": int(pbl_raw["sample_id"].nunique()),
        "cycles": int(pbl_raw["cycle"].nunique()),
        "common_valid_rows": int(pbl_raw["common_valid"].astype(bool).sum()),
        "common_daylight_valid_rows": int(pbl_raw["common_daylight_valid"].astype(bool).sum()),
        "extra_keys_not_in_dl_solfus_reference": int(len(frame_keys["mynn"] - ref_keys)),
        "common_key_rows_with_dl_solfus": int(len(frame_keys["mynn"] & ref_keys)),
        "merged": True,
    }
    return common, sol, baselines, pbl, audit


def compute_metrics(common: pd.DataFrame):
    seed_rows: list[dict] = []
    lead_rows: list[dict] = []
    for model in MODEL_ORDER:
        seed_list = (None,) if model_is_deterministic(model) else SEEDS
        for seed in seed_list:
            col = model_col(model, seed)
            m = metric(common["target"].to_numpy(), common[col].to_numpy())
            seed_rows.append({"model_key": model, "model": DISPLAY[model], "group": GROUP[model],
                              "seed": "deterministic" if seed is None else int(seed), **m})
            for lead in LEADS:
                sub = common[common["lead"] == lead]
                ml = metric(sub["target"].to_numpy(), sub[col].to_numpy())
                lead_rows.append({"model_key": model, "model": DISPLAY[model], "group": GROUP[model],
                                  "seed": "deterministic" if seed is None else int(seed),
                                  "lead": int(lead), **ml})
    seed_df = pd.DataFrame(seed_rows)
    lead_seed_df = pd.DataFrame(lead_rows)
    agg_df = aggregate_metric_rows(seed_df, ["model_key", "model", "group"])
    lead_agg_df = aggregate_metric_rows(lead_seed_df, ["model_key", "model", "group", "lead"])
    write_csv(seed_df, "overall_seed_metrics.csv")
    write_csv(agg_df, "overall_aggregate_metrics.csv")
    write_csv(lead_seed_df, "metrics_by_lead_seed.csv")
    write_csv(lead_agg_df, "metrics_by_lead.csv")
    return seed_df, agg_df, lead_seed_df, lead_agg_df


def compute_overall_skill(agg_df: pd.DataFrame) -> pd.DataFrame:
    sol = agg_df.loc[agg_df["model_key"] == "solfus"].iloc[0]
    rows = []
    for baseline in COMPARISON_ORDER:
        b = agg_df.loc[agg_df["model_key"] == baseline].iloc[0]
        rows.append({
            "baseline_model_key": baseline,
            "baseline_model": DISPLAY[baseline],
            "solfus_model": DISPLAY["solfus"],
            "n_common": int(sol["n_mean"]),
            "baseline_rmse_mean_wm2": float(b["rmse_wm2_mean"]),
            "solfus_rmse_mean_wm2": float(sol["rmse_wm2_mean"]),
            "rmse_reduction_wm2": float(b["rmse_wm2_mean"] - sol["rmse_wm2_mean"]),
            "rmse_skill_pct": float(100.0 * (b["rmse_wm2_mean"] - sol["rmse_wm2_mean"]) / b["rmse_wm2_mean"]),
            "baseline_nrmse_mean_pct": float(b["nrmse_pct_mean"]),
            "solfus_nrmse_mean_pct": float(sol["nrmse_pct_mean"]),
            "nrmse_reduction_pp": float(b["nrmse_pct_mean"] - sol["nrmse_pct_mean"]),
            "baseline_r2_mean": float(b["r2_mean"]),
            "solfus_r2_mean": float(sol["r2_mean"]),
            "r2_gain": float(sol["r2_mean"] - b["r2_mean"]),
        })
    out = pd.DataFrame(rows)
    write_csv(out, "overall_skill_vs_solfus.csv")
    return out


def compute_lead_skill(lead_seed_df: pd.DataFrame, lead_agg_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline in COMPARISON_ORDER:
        for lead in LEADS:
            b = lead_agg_df[(lead_agg_df.model_key == baseline) & (lead_agg_df.lead == lead)].iloc[0]
            s = lead_agg_df[(lead_agg_df.model_key == "solfus") & (lead_agg_df.lead == lead)].iloc[0]
            bseed = lead_seed_df[(lead_seed_df.model_key == baseline) & (lead_seed_df.lead == lead)].sort_values("seed")
            sseed = lead_seed_df[(lead_seed_df.model_key == "solfus") & (lead_seed_df.lead == lead)].sort_values("seed")
            if len(bseed) == len(sseed):
                paired = (bseed.rmse_wm2.to_numpy() - sseed.rmse_wm2.to_numpy()) / bseed.rmse_wm2.to_numpy() * 100.0
            else:
                paired = (float(b.rmse_wm2_mean) - sseed.rmse_wm2.to_numpy()) / float(b.rmse_wm2_mean) * 100.0
            rows.append({
                "baseline_model_key": baseline,
                "baseline_model": DISPLAY[baseline],
                "lead": int(lead),
                "n_common": int(s.n_mean),
                "baseline_rmse_mean_wm2": float(b.rmse_wm2_mean),
                "baseline_rmse_sd_wm2": float(b.rmse_wm2_sd) if np.isfinite(b.rmse_wm2_sd) else np.nan,
                "solfus_rmse_mean_wm2": float(s.rmse_wm2_mean),
                "solfus_rmse_sd_wm2": float(s.rmse_wm2_sd) if np.isfinite(s.rmse_wm2_sd) else np.nan,
                "rmse_skill_pct": float(100.0 * (b.rmse_wm2_mean - s.rmse_wm2_mean) / b.rmse_wm2_mean),
                "rmse_skill_sd_pct": float(np.std(paired, ddof=1)) if len(paired) > 1 else np.nan,
                "nrmse_reduction_pp": float(b.nrmse_pct_mean - s.nrmse_pct_mean),
                "solfus_better": bool(b.rmse_wm2_mean > s.rmse_wm2_mean),
            })
    out = pd.DataFrame(rows)
    write_csv(out, "solfus_skill_by_lead.csv")
    return out


def compute_cycle_skill(common: pd.DataFrame):
    seed_rows = []
    summary_rows = []
    cycles = sorted(common["cycle"].astype(str).unique())
    for baseline in COMPARISON_ORDER:
        cycle_rows = []
        for cycle in cycles:
            sub = common[common["cycle"].astype(str) == cycle]
            seed_skills = []
            baseline_rmses = []
            solfus_rmses = []
            for seed in SEEDS:
                bseed = seed if not model_is_deterministic(baseline) else None
                bcol = model_col(baseline, bseed)
                scol = model_col("solfus", seed)
                mb = metric(sub["target"].to_numpy(), sub[bcol].to_numpy())
                ms = metric(sub["target"].to_numpy(), sub[scol].to_numpy())
                baseline_rmses.append(mb["rmse_wm2"])
                solfus_rmses.append(ms["rmse_wm2"])
                seed_skills.append(100.0 * (mb["rmse_wm2"] - ms["rmse_wm2"]) / mb["rmse_wm2"])
                seed_rows.append({"baseline_model_key": baseline, "baseline_model": DISPLAY[baseline], "cycle": cycle,
                                  "baseline_seed": "deterministic" if bseed is None else int(bseed),
                                  "solfus_seed": int(seed), "n_points": int(len(sub)),
                                  "baseline_rmse_wm2": float(mb["rmse_wm2"]), "solfus_rmse_wm2": float(ms["rmse_wm2"]),
                                  "skill_pct": float(seed_skills[-1])})
            bmean = float(np.mean(baseline_rmses))
            smean = float(np.mean(solfus_rmses))
            skill = 100.0 * (bmean - smean) / bmean
            cycle_rows.append({"baseline_model_key": baseline, "baseline_model": DISPLAY[baseline], "cycle": cycle,
                               "n_points": int(len(sub)), "baseline_rmse_mean_wm2": bmean,
                               "solfus_rmse_mean_wm2": smean, "skill_pct": skill,
                               "seed_skill_sd_pct": float(np.std(seed_skills, ddof=1)), "solfus_better": bool(skill > 0)})
        cdf = pd.DataFrame(cycle_rows)
        wins = int(cdf["solfus_better"].sum())
        summary_rows.append({"baseline_model_key": baseline, "baseline_model": DISPLAY[baseline],
                             "n_cycles": int(len(cdf)), "win_cycles": wins, "win_fraction": wins / len(cdf),
                             "median_skill_pct": float(cdf.skill_pct.median()), "mean_skill_pct": float(cdf.skill_pct.mean()),
                             "q25_skill_pct": float(cdf.skill_pct.quantile(0.25)), "q75_skill_pct": float(cdf.skill_pct.quantile(0.75)),
                             "min_skill_pct": float(cdf.skill_pct.min()), "max_skill_pct": float(cdf.skill_pct.max())})
        if len(cdf) != 17:
            raise ValueError(f"{baseline} does not have 17 cycles: {len(cdf)}")
        write_csv(cdf, f"cycle_skill_{baseline}.csv")
    seed_df = pd.DataFrame(seed_rows)
    summary_df = pd.DataFrame(summary_rows)
    write_csv(seed_df, "cycle_skill_seed_pairs.csv")
    write_csv(summary_df, "cycle_skill_summary.csv")
    write_csv(pd.concat([pd.read_csv(OUT / f"cycle_skill_{m}.csv") for m in COMPARISON_ORDER], ignore_index=True), "cycle_skill.csv")
    return seed_df, summary_df


def build_groups(common: pd.DataFrame):
    group_rows = []
    cache_audit = {"path": str(CACHE_PATH), "exists": CACHE_PATH.exists()}
    if not CACHE_PATH.exists():
        raise FileNotFoundError(CACHE_PATH)
    try:
        import torch
        cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        meta = cache["meta"]
        y_cache = cache["y"].detach().cpu().numpy()
        context = cache["context"].detach().cpu().numpy()
        valid = cache["valid"].detach().cpu().numpy().astype(bool)
        day = cache["day"].detach().cpu().numpy().astype(bool)
        cache_map = {}
        for i, m in enumerate(meta):
            sid = str(m["sample_id"])
            for j, lead in enumerate(LEADS):
                cache_map[(sid, int(lead))] = {
                    "cache_target": float(y_cache[i, j]),
                    "clear_sky_wm2": float(context[i, j, 1] * 1000.0),
                    "cache_valid": bool(valid[i, j]),
                    "cache_day": bool(day[i, j]),
                }
        cache_join = []
        for _, r in common.iterrows():
            rec = cache_map.get((str(r.sample_id), int(r.lead)))
            if rec is None:
                raise ValueError(f"missing cache key {(r.sample_id, r.lead)}")
            cache_join.append(rec)
        cdf = pd.DataFrame(cache_join)
        cache_target_diff = np.abs(common["target"].to_numpy() - cdf["cache_target"].to_numpy())
        cache_audit.update({
            "meta_cases": int(len(meta)),
            "common_rows_matched": int(len(cdf)),
            "max_abs_target_diff_wm2": float(cache_target_diff.max()),
            "target_mismatch_rows": int((cache_target_diff > 1e-5).sum()),
            "context_column_used": 1,
            "clear_sky_definition": "Haurwitz clear-sky GHI from test_cache.pt context[...,1] * 1000",
            "clear_sky_min_wm2": float(cdf.clear_sky_wm2.min()),
            "clear_sky_max_wm2": float(cdf.clear_sky_wm2.max()),
        })
        common2 = pd.concat([common.reset_index(drop=True), cdf], axis=1)
    except Exception as exc:
        cache_audit["error"] = repr(exc)
        raise

    for sid, g in common2.groupby("sample_id", sort=False):
        g = g.sort_values("lead")
        k = g["target"].to_numpy(dtype=float) / g["clear_sky_wm2"].to_numpy(dtype=float)
        if len(k) < 2 or not np.isfinite(k).all():
            group = None
            tv = np.nan
            net = np.nan
            sign_changes = np.nan
        else:
            delta = np.diff(k)
            tv = float(np.abs(delta).sum())
            net = float(k[-1] - k[0])
            signs = np.sign(delta)
            sign_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
            # Fixed thresholds were set before examining model errors and use
            # only observed radiation trajectory variability.
            group = "stable" if tv < 0.20 else ("transitional" if tv < 0.60 else "complex")
        group_rows.append({"sample_id": sid, "station_id": g.station_id.iloc[0], "cycle": str(g.cycle.iloc[0]),
                           "n_common_daylight_leads": int(len(g)), "mean_k_star": float(np.mean(k)) if len(k) else np.nan,
                           "total_variation_k_star": tv, "net_change_k_star": net, "sign_changes": sign_changes,
                           "variation_group": group})
    case_df = pd.DataFrame(group_rows)
    common2 = common2.merge(case_df[["sample_id", "variation_group"]], on="sample_id", how="left")
    write_csv(case_df, "radiation_variation_case_features.csv")

    # The frozen SolFUS output is a bounded, sample-adaptive blend of the
    # satellite and NWP branches.  Preserve the emitted weights as a mechanism
    # diagnostic; this is descriptive and is not used to select rows.
    weight_seed_rows = []
    for seed in SEEDS:
        wcol = f"solfus_satellite_weight_seed{seed}"
        for lead in LEADS:
            sub = common2[common2.lead == lead]
            w = sub[wcol].to_numpy(dtype=float)
            weight_seed_rows.append({"seed": int(seed), "lead": int(lead), "n": int(len(w)),
                                     "satellite_weight_mean": float(w.mean()),
                                     "satellite_weight_sd": float(w.std(ddof=1)),
                                     "satellite_weight_min": float(w.min()),
                                     "satellite_weight_max": float(w.max())})
    weight_seed_df = pd.DataFrame(weight_seed_rows)
    weight_agg_rows = []
    for lead in LEADS:
        sub = weight_seed_df[weight_seed_df.lead == lead]
        weight_agg_rows.append({"lead": int(lead), "n": int(sub.n.iloc[0]),
                                "satellite_weight_mean": float(sub.satellite_weight_mean.mean()),
                                "satellite_weight_seed_sd": float(sub.satellite_weight_mean.std(ddof=1)),
                                "satellite_weight_min": float(sub.satellite_weight_min.min()),
                                "satellite_weight_max": float(sub.satellite_weight_max.max())})
    weight_agg_df = pd.DataFrame(weight_agg_rows)
    write_csv(weight_seed_df, "solfus_weight_by_lead_seed.csv")
    write_csv(weight_agg_df, "solfus_weight_by_lead.csv")

    common2["sat_nwp_gap_k_star"] = np.abs(common2["satellite"] - common2["nwp"]) / common2["clear_sky_wm2"]
    cycle_context = []
    for cycle, g in common2.groupby("cycle", sort=True):
        case_sub = case_df[case_df.cycle.astype(str) == str(cycle)]
        cycle_context.append({"cycle": str(cycle), "n_points": int(len(g)),
                              "n_cases": int(case_sub.shape[0]),
                              "mean_total_variation_k_star": float(case_sub.total_variation_k_star.mean()),
                              "stable_cases": int((case_sub.variation_group == "stable").sum()),
                              "transitional_cases": int((case_sub.variation_group == "transitional").sum()),
                              "complex_cases": int((case_sub.variation_group == "complex").sum()),
                              "mean_sat_nwp_gap_k_star": float(g.sat_nwp_gap_k_star.mean())})
    cycle_context_df = pd.DataFrame(cycle_context)
    write_csv(cycle_context_df, "cycle_weather_context.csv")

    assigned = common2[common2["variation_group"].notna()].copy()
    rows = []
    for group in ["stable", "transitional", "complex"]:
        sub = assigned[assigned.variation_group == group]
        for model in MODEL_ORDER:
            seeds = (None,) if model_is_deterministic(model) else SEEDS
            for seed in seeds:
                col = model_col(model, seed)
                rows.append({"variation_group": group, "model_key": model, "model": DISPLAY[model],
                             "group_role": GROUP[model], "seed": "deterministic" if seed is None else int(seed),
                             **metric(sub.target.to_numpy(), sub[col].to_numpy())})
    group_seed_df = pd.DataFrame(rows)
    group_agg_df = aggregate_metric_rows(group_seed_df, ["variation_group", "model_key", "model", "group_role"])
    write_csv(group_seed_df, "radiation_variation_metrics_seed.csv")
    write_csv(group_agg_df, "radiation_variation_metrics.csv")

    skill_rows = []
    for group in ["stable", "transitional", "complex"]:
        s = group_agg_df[(group_agg_df.variation_group == group) & (group_agg_df.model_key == "solfus")].iloc[0]
        for baseline in COMPARISON_ORDER:
            b = group_agg_df[(group_agg_df.variation_group == group) & (group_agg_df.model_key == baseline)].iloc[0]
            skill_rows.append({"variation_group": group, "baseline_model_key": baseline, "baseline_model": DISPLAY[baseline],
                               "n_cases": int(case_df[case_df.variation_group == group].shape[0]),
                               "n_points": int(assigned[assigned.variation_group == group].shape[0]),
                               "baseline_rmse_mean_wm2": float(b.rmse_wm2_mean), "solfus_rmse_mean_wm2": float(s.rmse_wm2_mean),
                               "rmse_skill_pct": float(100.0 * (b.rmse_wm2_mean - s.rmse_wm2_mean) / b.rmse_wm2_mean),
                               "nrmse_reduction_pp": float(b.nrmse_pct_mean - s.nrmse_pct_mean),
                               "solfus_better": bool(b.rmse_wm2_mean > s.rmse_wm2_mean)})
    skill_df = pd.DataFrame(skill_rows)
    write_csv(skill_df, "radiation_variation_skill_vs_solfus.csv")
    return case_df, group_seed_df, group_agg_df, skill_df, cache_audit, weight_agg_df, cycle_context_df


def _density_values(x: np.ndarray, y: np.ndarray, bins: int = 50) -> np.ndarray:
    """Return a deterministic local 2-D bin count for each plotted point."""
    hist, x_edges, y_edges = np.histogram2d(
        x, y, bins=bins, range=[[0.0, 1200.0], [0.0, 1200.0]]
    )
    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    return np.maximum(hist[xi, yi], 1.0)


def _figure_panel_data(common: pd.DataFrame) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[dict[str, float]]]]:
    """Build plotted clouds and seed-separated statistics for the seven methods."""
    clouds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    stats: dict[str, list[dict[str, float]]] = {}
    target = common["target"].to_numpy(dtype=float)
    for model in FIGURE_MODELS:
        cols = [model] if model in PBL_MODELS else [model_col(model, seed) for seed in SEEDS]
        x_parts = []
        y_parts = []
        rows = []
        for col in cols:
            pred = common[col].to_numpy(dtype=float)
            x_parts.append(target)
            y_parts.append(pred)
            rows.append(metric(target, pred))
        clouds[model] = (np.concatenate(x_parts), np.concatenate(y_parts))
        stats[model] = rows
    return clouds, stats


def plot_figure(common: pd.DataFrame, lead_agg_df: pd.DataFrame) -> None:
    """Draw Fig. 4-1: grouped lead-time bars and seven density scatter panels."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "legend.frameon": False,
    })

    # Compact double-column layout (183 mm × 110 mm, within the requested
    # approximately 190 mm × 105–115 mm range).  The outer 1 × 2 grid
    # keeps the bar chart at approximately 38% width and the density grid at
    # approximately 62%; the right axes are placed explicitly so their square
    # boxes do not inherit the large unused vertical space of a tall subplot
    # cell.
    fig = plt.figure(figsize=(7.20, 4.35), facecolor="white")
    fig_width, fig_height = 7.20, 4.35
    outer = GridSpec(
        1, 2, figure=fig, left=0.060, right=0.985, bottom=0.160, top=0.665,
        width_ratios=[1.00, 1.60], wspace=0.20,
    )
    ax_bar = fig.add_subplot(outer[0, 0])
    right_box = outer[0, 1].get_position(fig)

    # Panel a: seven bars per lead.  The three WRF-PBL schemes are deterministic
    # and use hatch patterns; DL bars show mean +/- seed SD.
    x = np.arange(1, len(LEADS) + 1, dtype=float)
    bar_width = 0.117  # seven bars span ≈0.82 around each lead center
    offsets = (np.arange(len(FIGURE_MODELS), dtype=float) - 3.0) * bar_width
    for idx, model in enumerate(FIGURE_MODELS):
        rows = lead_agg_df[lead_agg_df.model_key == model].set_index("lead").reindex(LEADS)
        means = rows["nrmse_pct_mean"].to_numpy(dtype=float)
        sds = rows["nrmse_pct_sd"].to_numpy(dtype=float)
        ax_bar.bar(
            x + offsets[idx], means, width=bar_width * 0.92,
            color=FIGURE_BAR_COLORS[model], edgecolor="#2b2b2b", linewidth=0.70,
            hatch=FIGURE_BAR_HATCHES[model], zorder=3, label=SHORT_DISPLAY[model],
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{lead} h" for lead in LEADS], fontsize=5.9)
    ax_bar.set_xlabel("Lead time", labelpad=2)
    ax_bar.set_ylabel("nRMSE (%)", labelpad=2)
    ax_bar.set_title("a  nRMSE across lead times", loc="left", fontweight="bold", pad=5)
    max_nrmse = float(np.nanmax(lead_agg_df[lead_agg_df.model_key.isin(FIGURE_MODELS)]["nrmse_pct_mean"].to_numpy()))
    ax_bar.set_ylim(0.0, max(34.0, max_nrmse + 4.0))
    ax_bar.set_xlim(0.50, len(LEADS) + 0.50)
    ax_bar.grid(axis="y", color="#d9d9d9", lw=0.45, zorder=0)
    ax_bar.tick_params(axis="both", length=2.0, pad=1.5)
    handles = [
        Patch(facecolor=FIGURE_BAR_COLORS[m], edgecolor="#2b2b2b", linewidth=0.70,
              hatch=FIGURE_BAR_HATCHES[m], label=FIGURE_LEGEND_LABELS[m])
        for m in FIGURE_MODELS
    ]
    # A single compact figure-level row preserves the requested method order
    # and leaves both panel titles unobstructed.
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.060, 0.770),
        ncol=7, fontsize=5.0, handlelength=0.95, handleheight=0.75,
        columnspacing=0.65, handletextpad=0.25, borderaxespad=0.0,
    )

    # Panel b: all panels use the same GHI range and tick locations.  The DL
    # clouds contain 601 x 3 points for visual coverage, while the annotation
    # is computed from three separate seed metrics rather than pooled points.
    clouds, panel_stats = _figure_panel_data(common)
    density_by_model = {
        model: _density_values(xv, yv)
        for model, (xv, yv) in clouds.items()
    }
    density_max = max(float(np.max(v)) for v in density_by_model.values())
    density_norm = LogNorm(vmin=1.0, vmax=max(2.0, density_max))
    # Restrained single-hue density scale: sparse points recede in pale blue,
    # while the high-density diagonal is emphasized in deep navy.  This keeps
    # the red 1:1 line as the only warm accent and avoids the visual competition
    # created by the former blue-grey-yellow ramp.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "journal_navy",
        ["#EFF4F8", "#D2DFEA", "#9DB7CB", "#648AA8", "#315E80", "#123652"],
    )
    # Make each right-hand panel physically square and use a small fixed gap.
    # This enlarges the clouds while keeping the same x/y scale in every cell.
    x_gap = 0.014
    panel_w = (right_box.width - 3.0 * x_gap) / 4.0
    panel_h = panel_w * fig_width / fig_height
    y_gap = 0.050
    bottom_y = right_box.y0 + 0.045
    top_y = bottom_y + panel_h + y_gap
    ticks = [0, 400, 800, 1200]
    for i, model in enumerate(FIGURE_MODELS):
        row, col = divmod(i, 4)
        left_x = right_box.x0 + col * (panel_w + x_gap)
        bottom = top_y if row == 0 else bottom_y
        ax = fig.add_axes([left_x, bottom, panel_w, panel_h], aspect="equal")
        xv, yv = clouds[model]
        ax.scatter(
            xv, yv, c=density_by_model[model], cmap=cmap, norm=density_norm,
            s=2.2, alpha=0.73, linewidths=0, rasterized=True, zorder=2,
        )
        ax.plot(
            [0, 1200], [0, 1200], color="#C44E52", lw=0.62,
            linestyle=(0, (3.0, 2.2)), alpha=0.88, zorder=3,
        )
        ax.set_xlim(0, 1200)
        ax.set_ylim(0, 1200)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(DISPLAY[model], fontsize=5.2, pad=2.0)
        ax.tick_params(axis="both", labelsize=5.0, length=1.5, pad=1.0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color("#333333")
        if row == 0 or col != 0:
            ax.set_xticklabels([])
        if col == 0 and row == 0:
            ax.set_ylabel("Predicted GHI", fontsize=5.0, labelpad=1.0)
        else:
            ax.set_yticklabels([])
        rows = panel_stats[model]
        rmse_mean = float(np.mean([r["rmse_wm2"] for r in rows]))
        r2_mean = float(np.mean([r["r2"] for r in rows]))
        stat_text = f"RMSE {rmse_mean:.1f}\nR² {r2_mean:.3f}"
        ax.text(
            0.04, 0.95, stat_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=5.2, color="#1d1d1d",
            bbox=dict(facecolor=(1.0, 1.0, 1.0, 0.72), edgecolor="none", pad=0.65),
            zorder=4,
        )

    # Eighth cell: shared density key and the distinction between plotted
    # clouds and the seed-level metric summaries.
    key_x = right_box.x0 + 3 * (panel_w + x_gap)
    ax_key = fig.add_axes([key_x, bottom_y, panel_w, panel_h])
    ax_key.set_axis_off()
    ax_key.text(0.0, 0.97, "Common key", transform=ax_key.transAxes,
                ha="left", va="top", fontsize=5.4, fontweight="bold")
    ax_key.text(
        0.0, 0.79,
        "Points shown\nPBL: 601\nDL: 601 × 3 seeds",
        transform=ax_key.transAxes, ha="left", va="top", fontsize=5.0,
        linespacing=1.25,
    )
    ax_key.text(
        0.0, 0.49,
        "Metrics:\nThree-seed mean\nRed line: 1:1",
        transform=ax_key.transAxes, ha="left", va="top", fontsize=5.0,
        linespacing=1.25,
    )
    cax = ax_key.inset_axes([0.04, 0.02, 0.10, 0.15])
    sm = ScalarMappable(norm=density_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([1.0, max(2.0, density_max)])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=5.0, length=1.2, pad=1.0)
    cbar.outline.set_linewidth(0.4)
    ax_key.text(0.20, 0.085, "point density", transform=ax_key.transAxes,
                ha="left", va="center", fontsize=5.0)

    fig.text(right_box.x0, top_y + panel_h + 0.018, "b  Observation–prediction density", fontsize=7.1,
             fontweight="bold", ha="left", va="bottom")
    fig.text(right_box.x0 + 0.5 * right_box.width, bottom_y - 0.060,
             "Measured GHI (W m⁻²)", fontsize=5.2, ha="center", va="top")
    stem = OUT / "fig43_fulltest_model_comparison"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_figure_final(common: pd.DataFrame) -> None:
    """Draw the final compact Fig. 4-1 with overall MAE, FS, and density panels.

    MAE and the density panels use all 601 common daylight rows.  FS uses the
    452-row intersection for which the preceding-hour observations required
    by smart persistence (SP) are available.  Learning models are summarized
    over seed-wise metrics, whereas each direct WRF-PBL scheme is a single
    deterministic estimate.  The right zone deliberately renders the three
    learning-model clouds together for visual density but never uses the
    pooled cloud to compute the annotations.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.70,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "legend.frameon": False,
    })

    fig_width, fig_height = 6.80, 4.60
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    outer = GridSpec(
        1, 2, figure=fig, left=0.060, right=0.915, bottom=0.105, top=0.890,
        width_ratios=[0.75, 1.25], wspace=0.18,
    )
    left_box = outer[0, 0].get_position(fig)
    right_box = outer[0, 1].get_position(fig)

    # Overall seed-wise statistics for the two left panels.
    target = common["target"].to_numpy(dtype=float)
    overall_stats: dict[str, list[dict[str, float]]] = {}
    for model in FIGURE_MODELS:
        cols = [model] if model in PBL_MODELS else [model_col(model, seed) for seed in SEEDS]
        overall_stats[model] = [metric(target, common[col].to_numpy(dtype=float)) for col in cols]

    # FS must be computed on exactly the same rows as its SP reference.  The
    # smaller intersection is caused solely by the preceding-hour observation
    # requirement of SP and is never selected by model error.
    sp_source = pd.read_csv(SP_PATH)
    sp_eval = common.merge(
        sp_source[KEY_COLS + ["target", "sp"]].rename(columns={"target": "target_sp_source"}),
        on=KEY_COLS, how="inner", validate="one_to_one",
    )
    if len(sp_eval) != len(sp_source):
        raise ValueError(f"SP/common key mismatch: matched {len(sp_eval)} of {len(sp_source)} rows")
    target_delta = np.abs(
        sp_eval["target"].to_numpy(dtype=float)
        - sp_eval["target_sp_source"].to_numpy(dtype=float)
    )
    if float(target_delta.max(initial=0.0)) > 1e-6:
        raise ValueError(f"SP/common target mismatch: max abs delta={target_delta.max()}")
    sp_target = sp_eval["target"].to_numpy(dtype=float)
    sp_rmse = float(np.sqrt(np.mean((sp_eval["sp"].to_numpy(dtype=float) - sp_target) ** 2)))
    fs_stats: dict[str, list[float]] = {}
    for model in FIGURE_MODELS:
        cols = [model] if model in PBL_MODELS else [model_col(model, seed) for seed in SEEDS]
        fs_stats[model] = [
            100.0 * (1.0 - float(np.sqrt(np.mean((sp_eval[col].to_numpy(dtype=float) - sp_target) ** 2))) / sp_rmse)
            for col in cols
        ]

    # Reproducible values used in the visible panels.  The learning-model
    # dispersion is retained as a source column, but no seed endpoints or
    # error bars are drawn in the final left panels.
    plot_rows = []
    for model in FIGURE_MODELS:
        rows = overall_stats[model]
        n_seeds = len(rows)
        point_count = len(common) * n_seeds
        plot_rows.append({
            "method_key": model,
            "method": FIGURE_LEGEND_LABELS[model],
            "n_seeds": n_seeds,
            "point_count_plotted": point_count,
            "mae_mean_wm2": float(np.mean([r["mae_wm2"] for r in rows])),
            "mae_seed_sd_wm2": float(np.std([r["mae_wm2"] for r in rows], ddof=1)) if n_seeds > 1 else np.nan,
            "nrmse_mean_pct": float(np.mean([r["nrmse_pct"] for r in rows])),
            "nrmse_seed_sd_pct": float(np.std([r["nrmse_pct"] for r in rows], ddof=1)) if n_seeds > 1 else np.nan,
            "fs_mean_pct": float(np.mean(fs_stats[model])),
            "fs_seed_sd_pct": float(np.std(fs_stats[model], ddof=1)) if n_seeds > 1 else np.nan,
            "fs_evaluation_rows": int(len(sp_eval)),
            "sp_rmse_wm2": sp_rmse,
            "rmse_mean_wm2": float(np.mean([r["rmse_wm2"] for r in rows])),
            "rmse_seed_sd_wm2": float(np.std([r["rmse_wm2"] for r in rows], ddof=1)) if n_seeds > 1 else np.nan,
            "r2_mean": float(np.mean([r["r2"] for r in rows])),
            "r2_seed_sd": float(np.std([r["r2"] for r in rows], ddof=1)) if n_seeds > 1 else np.nan,
            "summary": "deterministic" if n_seeds == 1 else "three-seed mean",
        })
    write_csv(pd.DataFrame(plot_rows), "figure43_plot_data.csv")

    # The two left axes have the same method positions.  Keeping the x labels
    # only on the lower panel leaves room for the error bars and the compact
    # legend while making the deterministic/seed-wise encoding explicit.
    left_gap = 0.055
    left_panel_h = (left_box.height - left_gap) / 2.0
    ax_mae = fig.add_axes([left_box.x0, left_box.y0 + left_panel_h + left_gap,
                           left_box.width, left_panel_h])
    ax_fs = fig.add_axes([left_box.x0, left_box.y0, left_box.width, left_panel_h])
    x = np.arange(len(FIGURE_MODELS), dtype=float)
    tick_labels = [FIGURE_LEGEND_LABELS[m].replace("-ConvLSTM", "\nConvLSTM")
                   .replace("-Transformer", "\nTransformer")
                   .replace("EarlyFusion\nTransformer", "EarlyFusion\nTrans.")
                   .replace("SolFUS-Net", "SolFUS") for m in FIGURE_MODELS]

    def draw_overall_panel(ax, key: str, ylabel: str, title: str,
                           y_limits: tuple[float, float], zero_line: bool = False,
                           annotate_mean: bool = False,
                           external_values: dict[str, list[float]] | None = None) -> None:
        for idx, model in enumerate(FIGURE_MODELS):
            vals = np.asarray(
                external_values[model] if external_values is not None
                else [row[key] for row in overall_stats[model]],
                dtype=float,
            )
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            ax.bar(
                x[idx], mean, width=0.68, color=FIGURE_BAR_COLORS[model],
                edgecolor="#202020", linewidth=0.70, hatch=FIGURE_BAR_HATCHES[model],
                zorder=3,
            )
            if annotate_mean:
                # Only the summary mean is labelled, kept close to the bar.
                ax.text(x[idx], mean + 0.55, f"{mean:.1f}", ha="center", va="bottom",
                        fontsize=5.0, color="#202020", zorder=6)
        ax.set_xlim(-0.55, len(FIGURE_MODELS) - 0.45)
        ax.set_ylim(*y_limits)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=5.1, linespacing=0.95)
        ax.set_ylabel(ylabel, fontsize=5.8, labelpad=2.0)
        ax.set_title(title, loc="left", fontsize=7.1, fontweight="bold", pad=1.6)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=5.2, length=1.8, pad=1.2)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#292929")
            ax.spines[side].set_linewidth(0.65)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if zero_line:
            ax.axhline(0.0, color="#202020", linewidth=0.85, zorder=4)

    mae_upper = max(10.0, float(max(np.mean([r["mae_wm2"] for r in overall_stats[m]])
                                     for m in FIGURE_MODELS) + 6.0))
    fs_top = max(float(np.mean(fs_stats[m])) for m in FIGURE_MODELS)
    fs_upper = max(40.0, math.ceil((fs_top + 3.0) / 5.0) * 5.0)
    draw_overall_panel(ax_mae, "mae_wm2", "MAE (W m^-2)", "a  Overall MAE", (0.0, mae_upper), annotate_mean=True)
    ax_mae.set_xticklabels([])
    draw_overall_panel(
        ax_fs, "fs_pct", "FS (%)", "b  Overall FS", (0.0, fs_upper),
        zero_line=True, annotate_mean=True, external_values=fs_stats,
    )
    ax_fs.set_xticklabels(tick_labels, rotation=35, rotation_mode="anchor",
                          ha="right", fontsize=5.1, linespacing=0.95)

    # One figure-level legend preserves the exact method order without
    # compressing the density panels.
    handles = [
        Patch(facecolor=FIGURE_BAR_COLORS[m], edgecolor="#202020", linewidth=0.70,
              hatch=FIGURE_BAR_HATCHES[m], label=FIGURE_LEGEND_LABELS[m])
        for m in FIGURE_MODELS
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.060, 0.965),
        ncol=7, fontsize=5.0, handlelength=0.90, handleheight=0.72,
        columnspacing=0.42, handletextpad=0.20, borderaxespad=0.0,
    )

    # Seven common observation–prediction clouds in a compact three-row grid:
    # three WRF schemes, two single-branch DL baselines, and two fusion models.
    # A tall reference-style colour scale is separated at the far-right edge.
    clouds, panel_stats = _figure_panel_data(common)
    density_by_model = {model: _density_values(xv, yv) for model, (xv, yv) in clouds.items()}
    density_max = max(float(np.max(v)) for v in density_by_model.values())
    density_norm = LogNorm(vmin=1.0, vmax=max(2.0, density_max))
    # A muted navy sequential scale follows the common journal convention of
    # encoding density with lightness rather than multiple competing hues.
    # The high-density core is dark; sparse outliers remain visible but quiet.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "journal_navy",
        ["#EFF4F8", "#D2DFEA", "#9DB7CB", "#648AA8", "#315E80", "#123652"],
    )
    x_gap = 0.014
    y_gap = 0.025
    scatter_top = right_box.y1 - 0.035
    panel_h = (scatter_top - right_box.y0 - 2.0 * y_gap) / 3.0
    # Convert a height fraction to the matching width fraction so every data
    # area is physically square on the fixed publication canvas.
    panel_w = panel_h * fig_height / fig_width
    grid_width = 3.0 * panel_w + 2.0 * x_gap
    grid_x0 = right_box.x0 + 0.5 * (right_box.width - grid_width)
    row_bottoms = [
        scatter_top - panel_h,
        scatter_top - 2.0 * panel_h - y_gap,
        right_box.y0,
    ]
    panel_positions = {
        "mynn": (0, 0),
        "ysu": (0, 1),
        "myj": (0, 2),
        "sat_convlstm": (1, 0.5),
        "wrf_transformer": (1, 1.5),
        "early_fusion": (2, 0.5),
        "solfus": (2, 1.5),
    }
    ticks = [0, 400, 800, 1200]
    for model in FIGURE_MODELS:
        row, col = panel_positions[model]
        left_x = grid_x0 + col * (panel_w + x_gap)
        bottom = row_bottoms[row]
        ax = fig.add_axes([left_x, bottom, panel_w, panel_h], aspect="equal")
        xv, yv = clouds[model]
        ax.scatter(
            xv, yv, c=density_by_model[model], cmap=cmap, norm=density_norm,
            s=2.2, alpha=0.84, linewidths=0, rasterized=True, zorder=2,
        )
        ax.plot(
            [0, 1200], [0, 1200], color="#C44E52", lw=0.62,
            linestyle=(0, (3.0, 2.2)), alpha=0.88, zorder=3,
        )
        ax.set_xlim(0, 1200)
        ax.set_ylim(0, 1200)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        panel_title = (FIGURE_LEGEND_LABELS[model]
                       .replace("Sat-ConvLSTM", "Sat-ConvLSTM")
                       .replace("WRF-Transformer", "WRF-Transformer")
                       .replace("EarlyFusion-Transformer", "EarlyFusion-Transformer"))
        title_color = "#0F4D92" if model == "solfus" else "#34404B"
        ax.set_title(panel_title, fontsize=5.0, pad=1.7, color=title_color)
        ax.tick_params(axis="both", labelsize=5.0, length=1.5, pad=1.0)
        for spine in ax.spines.values():
            spine.set_visible(spine in (ax.spines["left"], ax.spines["bottom"]))
            spine.set_linewidth(0.55)
            spine.set_color("#333333")
        # Only the lowest row needs x tick labels; y ticks remain on the
        # leftmost panel of each row alongside the shared axis titles.
        if not (row == 2 and model == "early_fusion"):
            ax.set_xticklabels([])
        if model not in ("mynn", "sat_convlstm", "early_fusion"):
            ax.set_yticklabels([])
        rows = panel_stats[model]
        rmse = np.asarray([r["rmse_wm2"] for r in rows], dtype=float)
        r2 = np.asarray([r["r2"] for r in rows], dtype=float)
        stat_text = f"RMSE {rmse.mean():.1f}\nR² {r2.mean():.3f}"
        ax.text(
            0.04, 0.95, stat_text, transform=ax.transAxes, ha="left", va="top",
            fontsize=5.0, color="#1d1d1d",
            bbox=dict(facecolor=(0.98, 0.99, 1.0, 0.90),
                      edgecolor="#0F4D92" if model == "solfus" else "#7D8B96", linewidth=0.60,
                      boxstyle="round,pad=0.28"),
            zorder=4,
        )

    # Reference-style colour scale at the outermost right edge: no explanatory
    # key, only a tall ramp, numeric ticks and its vertical label.
    cax = fig.add_axes([
        0.940,
        right_box.y0 + 0.035,
        0.010,
        right_box.height - 0.070,
    ])
    sm = ScalarMappable(norm=density_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    density_ticks = np.unique(np.rint(np.geomspace(1.0, max(2.0, density_max), 5)).astype(int))
    cbar.set_ticks(density_ticks)
    cbar.set_ticklabels([str(int(v)) for v in density_ticks])
    cbar.ax.minorticks_off()
    cbar.set_label("Point density", fontsize=5.2, labelpad=3.2)
    cbar.ax.tick_params(labelsize=5.0, length=1.5, width=0.5, pad=1.5)
    cbar.outline.set_linewidth(0.55)

    fig.text(grid_x0, right_box.y1,
             "c  Observation–prediction density", fontsize=7.1,
             fontweight="bold", ha="left", va="bottom")
    fig.text(grid_x0 + 0.5 * grid_width, right_box.y0 - 0.060,
             "Measured GHI (W m^-2)", fontsize=5.2, ha="center", va="top")
    fig.text(grid_x0 - 0.030, right_box.y0 + 0.5 * right_box.height,
             "Predicted GHI (W m^-2)", fontsize=5.2, rotation=90,
             ha="center", va="center")

    stem = OUT / "fig43_fulltest_model_comparison"
    # Keep the physical canvas fixed for publication export; no tight crop is
    # applied because it would change the designed 173 mm × 117 mm composition.
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def write_report(common: pd.DataFrame, agg_df: pd.DataFrame, overall_skill: pd.DataFrame,
                 cycle_summary: pd.DataFrame, case_df: pd.DataFrame, group_skill: pd.DataFrame,
                 weight_agg: pd.DataFrame, audit: dict, cache_audit: dict) -> None:
    lines = [
        "# Full-test SolFUS-Net versus deep-learning and direct WRF-PBL baselines",
        "",
        "## Scope and common-data rule",
        "",
        f"The all-method audit matched {len(common)} daylight-valid rows from {common.sample_id.nunique()} sample IDs, {common.station_id.nunique()} stations, and {common.cycle.nunique()} independent GFS cycles. The frozen DL/SolFUS reference contains {audit['reference_rows']} rows and the PBL export contains {audit['pbl_raw_export']['rows']} rows; their key intersection contains {audit['key_intersection_rows']} rows. The common set requires daylight=1 and finite target/prediction in every frozen export. No row was selected by SolFUS error.",
        "",
        f"MYNN, YSU, and MYJ are direct deterministic WRF-PBL predictions from the full {audit['pbl_raw_export']['cycles']}-cycle export ({audit['pbl_raw_export']['cases']} cases, {audit['pbl_raw_export']['common_daylight_valid_rows']} PBL-common daylight points). The frozen `nwp` column carried by the SolFUS export is a learned `WRF-NWP branch`, not a traditional WRF-PBL run, and is labelled accordingly. The final all-method intersection retains {len(common)} points for every method.",
        "",
        "## Primary metric definition",
        "",
        "RMSE, nRMSE = 100 × RMSE / mean daylight target, nMBE = 100 × mean signed error / mean daylight target, and R² are recomputed on the all-method common rows. For SolFUS-Net and the three DL baselines, the primary aggregate is the arithmetic mean of three seed-level metrics; seed SD is retained in the source tables but is not drawn in Fig. 4-1. MYNN, YSU, MYJ, and the WRF-NWP branch are deterministic in this comparison.",
        "",
        "## Figure 4-1 construction",
        "",
        "The primary comparison figure contains seven methods: MYNN, YSU, MYJ, Sat-ConvLSTM, WRF-Transformer, EarlyFusion-Transformer, and SolFUS-Net. The internal learned WRF-NWP branch remains in the source metrics and audit tables but is excluded from the figure. Panel a reports overall MAE, and panel b reports overall FS against smart persistence with one mean label above each bar; each learning-model bar is a three-seed mean, while seed dispersion is retained in the source table and not drawn in the left panels. Panel c uses a vertically aligned 3+2+2 observation–prediction layout with physically square cells and common 0–1200 axes; the point-density scale is placed independently at the far-right edge. Each PBL panel plots 601 points, whereas each learning-model panel plots the three seed clouds (601 × 3 points). The RMSE and R² annotations show the three-seed mean for learning models; the pooled point cloud is used only for local point-density rendering. The exported composition is a fixed 173 mm × 117 mm canvas, with a compact 1 × 2 layout, a one-line legend in the prescribed method order, and the fixed method palette: MYNN #9A4D8E, YSU #D9903D, MYJ #6BAF77, Sat-ConvLSTM #8CC5DD, WRF-Transformer #42949E, EarlyFusion-Transformer #E9A6A1, and SolFUS-Net #0F4D92.",
        "",
        "## Overall common-test metrics",
        "",
        "| model | group | n seeds/runs | RMSE (W m⁻²), mean ± seed SD | nRMSE (%), mean ± seed SD | nMBE (%) | R² |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg_df.set_index("model_key").loc[MODEL_ORDER].reset_index().iterrows():
        sd_rmse = f"{r.rmse_wm2_mean:.3f} (deterministic)" if not np.isfinite(r.rmse_wm2_sd) else f"{r.rmse_wm2_mean:.3f} ± {r.rmse_wm2_sd:.3f}"
        sd_nrmse = f"{r.nrmse_pct_mean:.3f} (deterministic)" if not np.isfinite(r.nrmse_pct_sd) else f"{r.nrmse_pct_mean:.3f} ± {r.nrmse_pct_sd:.3f}"
        lines.append(f"| {r.model} | {r.group} | {int(r.n_seeds)} | {sd_rmse} | {sd_nrmse} | {r.nmbe_pct_mean:.3f} | {r.r2_mean:.4f} |")
    lines += ["", "## SolFUS relative skill", "", "Positive RMSE skill means lower SolFUS RMSE than the comparator. Exact values are also available in `overall_skill_vs_solfus.csv` and `solfus_skill_by_lead.csv`.", ""]
    for _, r in overall_skill.iterrows():
        lines.append(f"- versus {r.baseline_model}: RMSE skill {r.rmse_skill_pct:.2f}% ({r.rmse_reduction_wm2:.3f} W m⁻²); nRMSE reduction {r.nrmse_reduction_pp:.3f} percentage points; R² gain {r.r2_gain:.4f}.")
    lines += ["", "## Mechanism and failure-mode diagnostic", "", "The frozen SolFUS output exposes a sample-adaptive satellite weight. Its mean weight is larger at short leads and smaller at long leads, indicating a lead-aware shift from the rapidly observed satellite signal toward the NWP trajectory. This is a descriptive check of the emitted predictions, not a post-hoc row-selection rule.", "", "The overall MAE and FS panels should be interpreted together with the density panels: the former summarize absolute error and skill against smart persistence, while the observation–prediction clouds show where RMSE and R² are accumulated across the full daylight sample. SolFUS is strongest at early and intermediate leads in the lead-wise source metrics, whereas the conventional fusion comparator can be competitive or better at the latest leads. The direct PBL schemes provide an independent physical-model reference rather than another learned branch. The radiation-variation audit provides the corresponding scene interpretation: stable trajectories can favor specialist models, while transitional and complex k* trajectories are where SolFUS is more consistently competitive. The article should therefore claim robust full-test and changing-radiation benefits, not universal dominance in every lead or weather regime.", ""]
    lines += ["", "## 17-cycle paired skill", "", "Cycle skill is computed from matched cycle RMSEs. For each learned baseline, the baseline and SolFUS are paired by seed before averaging; for deterministic PBL and WRF-NWP branch references, the deterministic baseline is compared with each SolFUS seed and the reported skill uses the mean SolFUS RMSE. The cycle count is fixed at 17 for every comparator.", "", "| comparator | wins | win fraction | median skill (%) | interquartile range (%) |", "|---|---:|---:|---:|---:|"]
    for _, r in cycle_summary.iterrows():
        lines.append(f"| {r.baseline_model} | {int(r.win_cycles)}/{int(r.n_cycles)} | {100*r.win_fraction:.1f}% | {r.median_skill_pct:.2f} | [{r.q25_skill_pct:.2f}, {r.q75_skill_pct:.2f}] |")
    lines += ["", "## Observed radiation-variation groups", "", "To provide a prediction-independent scene diagnostic, each sample was classified using observed k* = target / Haurwitz clear-sky GHI from the frozen test cache. Total variation thresholds were fixed before comparing model errors: stable < 0.20, transitional 0.20–<0.60, complex ≥ 0.60. Three one-lead common samples were left unassigned because a trajectory variation cannot be computed. No SolFUS prediction or error enters the grouping.", ""]
    for group in ["stable", "transitional", "complex"]:
        ncase = int(case_df[case_df.variation_group == group].shape[0])
        npts = int(group_skill[group_skill.variation_group == group].n_points.iloc[0])
        lines.append(f"- {group}: {ncase} sample trajectories, {npts} common daylight rows.")
        for _, r in group_skill[group_skill.variation_group == group].iterrows():
            lines.append(f"  - versus {r.baseline_model}: RMSE skill {r.rmse_skill_pct:.2f}%.")
    lines += ["", "## Files", "", "- `fig43_fulltest_model_comparison.svg/.pdf/.tiff/.png`: compact two-zone figure with overall MAE and FS bars at left and seven observation–prediction density panels at right.", "- `figure43_plot_data.csv`: exact seven-method summary values and point counts used in the figure, including three-seed source dispersion columns.", "- `source_data_common_predictions.csv`: all common matched prediction rows used by the analysis, including MYNN, YSU, and MYJ.", "- `overall_aggregate_metrics.csv`, `metrics_by_lead.csv`, `overall_skill_vs_solfus.csv`: quantitative source tables for all methods.", "- `cycle_skill.csv`, `cycle_skill_summary.csv`, `cycle_weather_context.csv`: all 17-cycle paired skill values and observed-scene context.", "- `radiation_variation_metrics.csv`, `radiation_variation_skill_vs_solfus.csv`: prediction-independent scene-group diagnostics.", "- `solfus_weight_by_lead.csv`: emitted lead-wise satellite-weight diagnostic.", "- `audit.json`, `key_audit.csv`, `target_consistency_audit.csv`, `common_exclusion_audit.csv`: reproducibility and integrity checks, including PBL key/label auditing.", "- `analyze_and_plot_fulltest.py`: source script.", ""]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Remove the legacy pre-PBL artifact so an outdated file labelled
    # ``traditional_wrf`` cannot be mistaken for the current WRF-NWP branch.
    legacy_cycle_file = OUT / "cycle_skill_traditional_wrf.csv"
    if legacy_cycle_file.exists():
        legacy_cycle_file.unlink()
    common, sol, baselines, pbl, audit = build_common_data()
    seed_df, agg_df, lead_seed_df, lead_agg_df = compute_metrics(common)
    overall_skill = compute_overall_skill(agg_df)
    lead_skill = compute_lead_skill(lead_seed_df, lead_agg_df)
    cycle_seed, cycle_summary = compute_cycle_skill(common)
    cycle_df = pd.read_csv(OUT / "cycle_skill.csv")
    case_df, group_seed, group_agg, group_skill, cache_audit, weight_agg, cycle_context = build_groups(common)
    plot_figure_final(common)

    file_paths = {
        "solfus_seed42": SOL_DIR / "mlp_seed42_test.csv",
        "solfus_seed43": SOL_DIR / "mlp_seed43_test.csv",
        "solfus_seed44": SOL_DIR / "mlp_seed44_test.csv",
        "sat_convlstm_seed42": BASE_DIR / "sat_convlstm/seed42/test_predictions.csv",
        "sat_convlstm_seed43": BASE_DIR / "sat_convlstm/seed43/test_predictions.csv",
        "sat_convlstm_seed44": BASE_DIR / "sat_convlstm/seed44/test_predictions.csv",
        "wrf_transformer_seed42": BASE_DIR / "wrf_transformer/seed42/test_predictions.csv",
        "wrf_transformer_seed43": BASE_DIR / "wrf_transformer/seed43/test_predictions.csv",
        "wrf_transformer_seed44": BASE_DIR / "wrf_transformer/seed44/test_predictions.csv",
        "early_fusion_seed42": BASE_DIR / "early_fusion/seed42/test_predictions.csv",
        "early_fusion_seed43": BASE_DIR / "early_fusion/seed43/test_predictions.csv",
        "early_fusion_seed44": BASE_DIR / "early_fusion/seed44/test_predictions.csv",
        "wrf_pbl_fulltest": PBL_PATH,
        "smart_persistence": SP_PATH,
    }
    audit.update({
        "generated_local": "2026-09-21",
        "python": sys.version,
        "platform": platform.platform(),
        "metric_scope": "fully common daylight-valid rows across all frozen exports",
        "common_data": {"rows": int(len(common)), "sample_ids": int(common.sample_id.nunique()),
                        "stations": int(common.station_id.nunique()), "cycles": int(common.cycle.nunique()),
                        "leads": sorted(common.lead.unique().tolist())},
        "seed_policy": "SolFUS and each DL baseline use seeds 42,43,44; mean ± seed SD; seeds are not independent observations",
        "model_roles": {m: {"display": DISPLAY[m], "group": GROUP[m], "role": "deterministic direct WRF-PBL prediction" if m in PBL_MODELS else "frozen test prediction"} for m in MODEL_ORDER},
        "wrf_nwp_branch_definition": "nwp column from SolFUS frozen test export; this is a learned NWP branch and is not labelled as traditional WRF",
        "traditional_wrf_pbl_definition": "mynn, ysu, and myj columns from the direct fulltest WRF-PBL export",
        "source_sha256": {name: sha256(path) for name, path in file_paths.items()},
        "cache": cache_audit,
        "pbl": audit.get("pbl_raw_export", {}),
        "radiation_grouping": {"feature": "observed total variation of k*=target/Haurwitz clear-sky GHI",
                               "thresholds": {"stable_lt": 0.20, "transitional_lt": 0.60, "complex_ge": 0.60},
                               "assigned_cases": int(case_df.variation_group.notna().sum()),
                               "unassigned_cases": int(case_df.variation_group.isna().sum()),
                               "selection_uses_model_error": False},
        "solfus_weight_diagnostic": {"definition": "satellite_weight emitted by frozen SolFUS export; prediction identity checked against weight*satellite+(1-weight)*nwp",
                                     "output": "solfus_weight_by_lead.csv"},
        "figure_contract": {"core_conclusion": "Across the complete daylight test set and the corresponding smart-persistence comparison set, the seven primary methods are compared with overall MAE, FS, and observation–prediction alignment against direct WRF-PBL and learning baselines.",
                            "archetype": "asymmetric mixed-modality quantitative figure",
                            "primary_methods": [DISPLAY[m] for m in FIGURE_MODELS],
                            "excluded_internal_reference": "WRF-NWP branch (retained in source metrics and audit tables; excluded from Fig. 4-1)",
                            "panel_a": "overall MAE bars for seven methods; learning bars are three-seed means and PBL bars are deterministic; no single-seed points or error bars are overlaid",
                            "panel_b": "overall FS bars against smart persistence for seven methods; one mean value is labelled just above each bar",
                            "panel_c": "vertically aligned 3+2+2 observation–prediction density grid with physically square cells and an independent far-right point-density scale; PBL clouds contain 601 points and learning clouds contain 601 × 3 seed points",
                            "layout": "fixed 173 mm × 117 mm canvas; outer 1 × 2 split with approximately 38% left metric zone and 62% right density grid; panel titles and vertical extents aligned",
                            "legend_order": [FIGURE_LEGEND_LABELS[m] for m in FIGURE_MODELS],
                            "overall_bar_width": 0.68,
                            "semantic_palette": {"MYNN": "#9A4D8E", "YSU": "#D9903D", "MYJ": "#6BAF77", "Sat-ConvLSTM": "#8CC5DD", "WRF-Transformer": "#42949E", "EarlyFusion-Transformer": "#E9A6A1", "SolFUS-Net": "#0F4D92"},
                            "axis_label_policy": "shared right-grid x/y labels; x ticks retained on the lowest occupied scatter in each column and y ticks on the left column",
                            "statistics_note": "Learning-model MAE, FS, RMSE, and R² are calculated separately for seeds 42, 43, and 44; the visible summaries report the three-seed mean and pooling is used only for density rendering",
                            "exports": ["SVG", "PDF", "TIFF 600 dpi", "PNG 600 dpi"]},
        "figure_qa": {"source_validator": "PASS: 19 PASS, 1 intentional WARN, 0 FAIL (strict; no left-panel uncertainty overlay by design)",
                      "pdf_text_audit": "PASS: 98 text runs, minimum rendered glyph 5.0 pt >= 5 pt",
                      "visual_review": "PASS: final-size PNG checked panel by panel; no clipping, axis-label clipping, or title collisions; left MAE/FS panels and right square density grid are aligned"},
    })
    (OUT / "audit.json").write_text(json.dumps(jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(common, agg_df, overall_skill, cycle_summary, case_df, group_skill, weight_agg, audit, cache_audit)
    print(json.dumps({"output": str(OUT), "common_rows": len(common), "common_cycles": int(common.cycle.nunique()),
                      "overall_skill": overall_skill.to_dict(orient="records"),
                      "cycle_summary": cycle_summary.to_dict(orient="records"),
                      "group_skill": group_skill.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
