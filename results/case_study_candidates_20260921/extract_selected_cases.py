"""Extract the three pre-selected case-study trajectories for section 4.4.

The cases are selected outside this script.  This script only joins the frozen
common-prediction export with the matching cache records, derives the observed
trajectory diagnostics, and writes auditable long and case-level tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "results/fulltest_model_comparison_20260921/source_data_common_predictions.csv"
CACHE_PATH = ROOT / "outputs/mlp17_7eval_training_20260909/review/test_7station/test_cache.pt"
SEEDS = (42, 43, 44)
LEADS = tuple(range(1, 7))

# These IDs and labels are fixed by the section-4.4 case-study plan.  No row
# is selected using model error in this script.
CASES = (
    {
        "case_order": 1,
        "case_category": "cloud_enhancement_radiation_drop",
        "case_label": "云遮蔽增强/辐射骤降",
        "sample_id": "development__TBL__2023090200__r18",
    },
    {
        "case_order": 2,
        "case_category": "cloud_dissipation_transmittance_recovery",
        "case_label": "云消散/透射率快速恢复",
        "sample_id": "val__CAB__2024012000__r57",
    },
    {
        "case_order": 3,
        "case_category": "intermittent_cloud_variability",
        "case_label": "间歇性云变/多次波动",
        "sample_id": "val__SEL__2024012000__r15",
    },
)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)) ** 2)))


def pct_improvement(baseline_rmse: float, solfus_rmse: float) -> float:
    """Positive values mean SolFUS has lower RMSE than the branch."""
    return float(100.0 * (baseline_rmse - solfus_rmse) / baseline_rmse) if baseline_rmse else np.nan


def load_cache() -> dict[str, dict[str, np.ndarray | bool]]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(CACHE_PATH)
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    required = {"context", "y", "valid", "day", "meta"}
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"cache missing keys: {sorted(missing)}")

    context = cache["context"].detach().cpu().numpy()
    y_cache = cache["y"].detach().cpu().numpy()
    valid = cache["valid"].detach().cpu().numpy().astype(bool)
    day = cache["day"].detach().cpu().numpy().astype(bool)
    if context.ndim != 3 or context.shape[2] <= 1:
        raise ValueError(f"unexpected cache context shape: {context.shape}")
    if y_cache.shape[1] < len(LEADS):
        raise ValueError(f"cache has fewer than six leads: {y_cache.shape}")

    out: dict[str, dict[str, np.ndarray | bool]] = {}
    for i, meta in enumerate(cache["meta"]):
        sid = str(meta["sample_id"])
        if sid in out:
            raise ValueError(f"duplicate cache sample_id: {sid}")
        out[sid] = {
            "cache_target": y_cache[i, : len(LEADS)].astype(float),
            # Per the frozen evaluation definition, context[..., 1] is the
            # clear-sky GHI ratio; convert from kW m^-2 to W m^-2.
            "clear_sky": (context[i, : len(LEADS), 1] * 1000.0).astype(float),
            "cache_valid": valid[i, : len(LEADS)],
            "cache_day": day[i, : len(LEADS)],
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    source = pd.read_csv(SOURCE_PATH)
    required = {
        "sample_id", "station_id", "cycle", "lead", "target", "satellite", "nwp",
        *(f"solfus_seed{s}" for s in SEEDS),
        *(f"solfus_satellite_weight_seed{s}" for s in SEEDS),
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"source CSV missing columns: {sorted(missing)}")
    source["lead"] = source["lead"].astype(int)
    source["sample_id"] = source["sample_id"].astype(str)
    source["cycle"] = source["cycle"].astype(str)

    cache = load_cache()
    sequence_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for case in CASES:
        sid = case["sample_id"]
        sub = source[source["sample_id"] == sid].copy().sort_values("lead")
        if len(sub) != len(LEADS) or set(sub["lead"].tolist()) != set(LEADS):
            raise ValueError(f"{sid}: expected exactly leads {LEADS}, got {sub['lead'].tolist()}")
        if sub[["sample_id", "lead"]].duplicated().any():
            raise ValueError(f"{sid}: duplicate sample_id/lead rows")
        if sid not in cache:
            raise ValueError(f"{sid}: missing from test cache")

        rec = cache[sid]
        cache_target = np.asarray(rec["cache_target"], dtype=float)
        clear_sky = np.asarray(rec["clear_sky"], dtype=float)
        cache_valid = np.asarray(rec["cache_valid"], dtype=bool)
        cache_day = np.asarray(rec["cache_day"], dtype=bool)
        target = sub["target"].to_numpy(dtype=float)
        if np.max(np.abs(target - cache_target)) > 1e-5:
            raise ValueError(f"{sid}: source target disagrees with cache by >1e-5 W m^-2")
        if not cache_valid.all() or not cache_day.all():
            raise ValueError(f"{sid}: selected sequence contains invalid/non-daylight cache lead")
        if not np.isfinite(clear_sky).all() or (clear_sky <= 0).any():
            raise ValueError(f"{sid}: clear-sky values must be finite and positive")

        k_star = target / clear_sky
        delta = np.diff(k_star)
        signs = np.sign(delta)
        tv = float(np.abs(delta).sum())
        net = float(k_star[-1] - k_star[0])
        sign_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

        sat = sub["satellite"].to_numpy(dtype=float)
        nwp = sub["nwp"].to_numpy(dtype=float)
        solfus = {seed: sub[f"solfus_seed{seed}"].to_numpy(dtype=float) for seed in SEEDS}
        weights = {
            seed: sub[f"solfus_satellite_weight_seed{seed}"].to_numpy(dtype=float)
            for seed in SEEDS
        }
        if any((~np.isfinite(v)).any() for v in (*solfus.values(), *weights.values())):
            raise ValueError(f"{sid}: non-finite SolFUS prediction or weight")
        for seed in SEEDS:
            if ((weights[seed] < -1e-8) | (weights[seed] > 1 + 1e-8)).any():
                raise ValueError(f"{sid}: out-of-range satellite weight for seed {seed}")

        # Verify the emitted SolFUS prediction is the reported weighted blend
        # of the same two branch values before writing any output.
        identity_error = {
            seed: solfus[seed] - (weights[seed] * sat + (1.0 - weights[seed]) * nwp)
            for seed in SEEDS
        }
        max_identity_error = max(float(np.max(np.abs(v))) for v in identity_error.values())
        if max_identity_error > 1e-4:
            raise ValueError(f"{sid}: SolFUS blend identity exceeds 1e-4 W m^-2 ({max_identity_error})")

        solfus_mean = np.mean(np.vstack([solfus[s] for s in SEEDS]), axis=0)
        solfus_sd = np.std(np.vstack([solfus[s] for s in SEEDS]), axis=0, ddof=1)
        weight_mean = np.mean(np.vstack([weights[s] for s in SEEDS]), axis=0)
        weight_sd = np.std(np.vstack([weights[s] for s in SEEDS]), axis=0, ddof=1)

        for j, (_, row) in enumerate(sub.reset_index(drop=True).iterrows()):
            out: dict[str, object] = {
                **case,
                "station_id": str(row["station_id"]),
                "cycle": str(row["cycle"]),
                "lead": int(row["lead"]),
                "target": float(target[j]),
                "cache_target": float(cache_target[j]),
                "cache_target_abs_diff": float(abs(target[j] - cache_target[j])),
                "cache_valid": bool(cache_valid[j]),
                "cache_day": bool(cache_day[j]),
                "clear_sky": float(clear_sky[j]),
                "k_star": float(k_star[j]),
                "satellite": float(sat[j]),
                "nwp": float(nwp[j]),
                "satellite_abs_error": float(abs(sat[j] - target[j])),
                "nwp_abs_error": float(abs(nwp[j] - target[j])),
                "solfus_mean": float(solfus_mean[j]),
                "solfus_sd": float(solfus_sd[j]),
                "solfus_abs_error": float(abs(solfus_mean[j] - target[j])),
                "satellite_weight_mean": float(weight_mean[j]),
                "satellite_weight_sd": float(weight_sd[j]),
                "nwp_weight": float(1.0 - weight_mean[j]),
                "nwp_weight_mean": float(1.0 - weight_mean[j]),
                "solfus_blend_identity_max_abs_diff": float(max(abs(identity_error[s][j]) for s in SEEDS)),
            }
            for seed in SEEDS:
                out[f"solfus_seed{seed}"] = float(solfus[seed][j])
                out[f"solfus_seed{seed}_abs_error"] = float(abs(solfus[seed][j] - target[j]))
                out[f"solfus_satellite_weight_seed{seed}"] = float(weights[seed][j])
                out[f"nwp_weight_seed{seed}"] = float(1.0 - weights[seed][j])
                out[f"solfus_blend_identity_diff_seed{seed}"] = float(identity_error[seed][j])
            sequence_rows.append(out)

        sat_rmse = rmse(target, sat)
        nwp_rmse = rmse(target, nwp)
        solfus_seed_rmses = {seed: rmse(target, solfus[seed]) for seed in SEEDS}
        solfus_rmse = rmse(target, solfus_mean)
        summary_rows.append(
            {
                **case,
                "station_id": str(sub["station_id"].iloc[0]),
                "cycle": str(sub["cycle"].iloc[0]),
                "n_leads": int(len(sub)),
                "mean_k_star": float(np.mean(k_star)),
                "min_k_star": float(np.min(k_star)),
                "max_k_star": float(np.max(k_star)),
                "total_variation_k_star": tv,
                "net_change_k_star": net,
                "sign_changes": sign_changes,
                "satellite_rmse_wm2": sat_rmse,
                "nwp_rmse_wm2": nwp_rmse,
                "solfus_rmse_wm2": solfus_rmse,
                "solfus_rmse_seed42_wm2": solfus_seed_rmses[42],
                "solfus_rmse_seed43_wm2": solfus_seed_rmses[43],
                "solfus_rmse_seed44_wm2": solfus_seed_rmses[44],
                "solfus_rmse_seed_mean_wm2": float(np.mean(list(solfus_seed_rmses.values()))),
                "solfus_rmse_seed_sd_wm2": float(np.std(list(solfus_seed_rmses.values()), ddof=1)),
                "solfus_vs_satellite_rmse_improvement_wm2": float(sat_rmse - solfus_rmse),
                "solfus_vs_satellite_rmse_improvement_pct": pct_improvement(sat_rmse, solfus_rmse),
                "solfus_vs_nwp_rmse_improvement_wm2": float(nwp_rmse - solfus_rmse),
                "solfus_vs_nwp_rmse_improvement_pct": pct_improvement(nwp_rmse, solfus_rmse),
                "max_solfus_blend_identity_abs_diff_wm2": max_identity_error,
                "max_cache_target_abs_diff_wm2": float(np.max(np.abs(target - cache_target))),
            }
        )

    sequence = pd.DataFrame(sequence_rows).sort_values(["case_order", "lead"])
    summary = pd.DataFrame(summary_rows).sort_values("case_order")
    sequence.to_csv(OUT / "selected_case_sequences.csv", index=False, float_format="%.10g")
    summary.to_csv(OUT / "selected_case_summary.csv", index=False, float_format="%.10g")

    readme = f"""# 4.4 selected case-study candidates

This directory contains the three cases fixed by the section-4.4 plan. The
script only extracts the named IDs; it does not select cases using model error
and it does not create the final figure.

## Inputs

- Predictions: `{SOURCE_PATH.relative_to(ROOT).as_posix()}`.
- Clear-sky and target audit: `{CACHE_PATH.relative_to(ROOT).as_posix()}`.
- `clear_sky = test_cache.pt['context'][i, lead-1, 1] * 1000`, in W m^-2.
- The cache `y` is the reference target. Every selected source target matches
  it within `1e-5 W m^-2`; all six selected leads are valid and daylight.

## Cases

1. `development__TBL__2023090200__r18`: cloud enhancement / abrupt radiation drop.
2. `val__CAB__2024012000__r57`: cloud dissipation / rapid transmittance recovery.
3. `val__SEL__2024012000__r15`: intermittent cloud change / repeated fluctuation.

## Definitions and checks

- `k_star = target / clear_sky`.
- `total_variation_k_star (TV) = sum(abs(diff(k_star)))` across the six leads.
- `net_change_k_star = k_star[lead 6] - k_star[lead 1]`.
- `sign_changes` counts adjacent sign changes in `diff(k_star)` (zero is kept
  as its own sign, matching the frozen full-test audit convention).
- SolFUS seed means and SDs use seeds 42, 43, and 44; SD is sample SD
  (`ddof=1`). `nwp_weight = 1 - satellite_weight_mean`, with per-seed NWP
  weights also retained.
- Trajectory RMSEs are over the six leads in W m^-2. Relative improvement is
  `100 * (branch_RMSE - SolFUS_RMSE) / branch_RMSE`; positive means SolFUS is
  lower error.
- For each seed, the script verifies
  `SolFUS = satellite_weight * satellite + (1 - satellite_weight) * nwp`
  within `1e-4 W m^-2`.

Outputs:

- `selected_case_sequences.csv`: one row per case and lead, including targets,
  clear-sky values, `k_star`, all branch/seed values, weights, absolute errors,
  and cache/blend audit fields.
- `selected_case_summary.csv`: one row per case with trajectory variability,
  branch/SolFUS RMSEs, and SolFUS relative improvement versus satellite and NWP.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {len(sequence)} sequence rows and {len(summary)} summary rows to {OUT}")
    print(summary.to_string(index=False))
    print(f"max target/cache abs diff: {sequence['cache_target_abs_diff'].max():.3g} W m^-2")
    print(f"max SolFUS blend identity abs diff: {sequence['solfus_blend_identity_max_abs_diff'].max():.3g} W m^-2")


if __name__ == "__main__":
    main()
