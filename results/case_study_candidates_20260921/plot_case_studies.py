"""Plot three mechanism-focused SolFUS-Net case studies for Section 4.4.

The selected cases illustrate cloud enhancement, cloud dissipation, and
intermittent cloud variability.  Each column aligns the branch/fused GHI
trajectories with the corresponding dynamic fusion weights.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/fulltest_model_comparison_20260921/source_data_common_predictions.csv"
CACHE = ROOT / "outputs/mlp17_7eval_training_20260909/review/test_7station/test_cache.pt"
SEEDS = (42, 43, 44)

CASES = [
    {
        "sample_id": "development__TBL__2023090200__r18",
        "panel": "a",
        "title": "Cloud thickening",
        "station_cycle": "TBL | 2023-09-02",
    },
    {
        "sample_id": "val__CAB__2024012000__r57",
        "panel": "b",
        "title": "Cloud dissipation",
        "station_cycle": "CAB | 2024-01-20",
    },
    {
        "sample_id": "val__SEL__2024012000__r15",
        "panel": "c",
        "title": "Intermittent cloud",
        "station_cycle": "SEL | 2024-01-20",
    },
]

COLORS = {
    "observation": "#272727",
    "satellite": "#D9903D",
    "nwp": "#2A9D8F",
    "solfus": "#0F4D92",
}


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def load_plot_data() -> pd.DataFrame:
    source = pd.read_csv(SOURCE)
    selected_ids = [case["sample_id"] for case in CASES]
    data = source[source["sample_id"].isin(selected_ids)].copy()
    if data.groupby("sample_id").size().to_dict() != {sid: 6 for sid in selected_ids}:
        raise ValueError("Each selected case must contain exactly six forecast leads")

    data["solfus_mean"] = data[[f"solfus_seed{s}" for s in SEEDS]].mean(axis=1)
    data["solfus_sd"] = data[[f"solfus_seed{s}" for s in SEEDS]].std(axis=1, ddof=1)
    data["satellite_weight_mean"] = data[
        [f"solfus_satellite_weight_seed{s}" for s in SEEDS]
    ].mean(axis=1)
    data["satellite_weight_sd"] = data[
        [f"solfus_satellite_weight_seed{s}" for s in SEEDS]
    ].std(axis=1, ddof=1)
    data["nwp_weight_mean"] = 1.0 - data["satellite_weight_mean"]

    import torch

    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    meta = cache["meta"]
    target = cache["y"].detach().cpu().numpy()
    clear_sky = cache["context"].detach().cpu().numpy()[:, :, 1] * 1000.0
    cache_map: dict[tuple[str, int], tuple[float, float]] = {}
    for idx, item in enumerate(meta):
        sid = str(item["sample_id"])
        if sid not in selected_ids:
            continue
        for lead in range(1, 7):
            cache_map[(sid, lead)] = (float(target[idx, lead - 1]), float(clear_sky[idx, lead - 1]))

    cache_target = []
    clear = []
    for row in data.itertuples(index=False):
        rec = cache_map[(str(row.sample_id), int(row.lead))]
        cache_target.append(rec[0])
        clear.append(rec[1])
    data["cache_target"] = cache_target
    data["clear_sky_wm2"] = clear
    if float(np.max(np.abs(data["target"] - data["cache_target"]))) > 1e-5:
        raise ValueError("Prediction export and cache targets are inconsistent")
    data["k_star"] = data["target"] / data["clear_sky_wm2"]

    reconstruction = (
        data["satellite_weight_mean"] * data["satellite"]
        + data["nwp_weight_mean"] * data["nwp"]
    )
    data["fusion_reconstruction_delta"] = data["solfus_mean"] - reconstruction
    if float(np.max(np.abs(data["fusion_reconstruction_delta"]))) > 1e-3:
        raise ValueError("Mean SolFUS trajectory does not match the emitted dynamic weights")

    order = {case["sample_id"]: idx for idx, case in enumerate(CASES)}
    data["case_order"] = data["sample_id"].map(order)
    data = data.sort_values(["case_order", "lead"]).reset_index(drop=True)
    data.to_csv(OUT / "figure44_plot_data.csv", index=False, encoding="utf-8-sig")
    return data


def plot(data: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.10, 4.55),
        gridspec_kw={"height_ratios": [2.35, 1.0], "hspace": 0.13, "wspace": 0.22},
        sharex="col",
    )
    leads = np.arange(1, 7)
    for col, case in enumerate(CASES):
        subset = data[data["sample_id"] == case["sample_id"]].sort_values("lead")
        obs = subset["target"].to_numpy(dtype=float)
        sat = subset["satellite"].to_numpy(dtype=float)
        nwp = subset["nwp"].to_numpy(dtype=float)
        fused = subset["solfus_mean"].to_numpy(dtype=float)
        sat_w = subset["satellite_weight_mean"].to_numpy(dtype=float)
        nwp_w = subset["nwp_weight_mean"].to_numpy(dtype=float)

        ax = axes[0, col]
        ax.plot(leads, obs, color=COLORS["observation"], marker="o", ms=3.5, lw=1.8, zorder=5)
        ax.plot(leads, sat, color=COLORS["satellite"], marker="s", ms=3.0, lw=1.35,
                ls="--", zorder=3)
        ax.plot(leads, nwp, color=COLORS["nwp"], marker="^", ms=3.2, lw=1.35,
                ls="-.", zorder=3)
        ax.plot(leads, fused, color=COLORS["solfus"], marker="D", ms=3.1, lw=2.0, zorder=6)
        ax.set_ylim(0, 1100)
        ax.set_yticks([0, 250, 500, 750, 1000])
        ax.grid(axis="y", color="#dedede", lw=0.45, zorder=0)
        ax.set_title(
            f"{case['panel']}  {case['title']}\n{case['station_cycle']}",
            loc="left",
            fontsize=8.2,
            fontweight="bold",
            pad=4,
        )
        if col == 0:
            ax.set_ylabel("GHI (W m$^{-2}$)")
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(axis="both", labelsize=7, length=2.2, pad=1.5)

        solfus_seed_rmse = np.mean(
            [rmse(obs, subset[f"solfus_seed{seed}"].to_numpy(dtype=float)) for seed in SEEDS]
        )
        metric_text = f"RMSE  {rmse(obs, sat):.0f} / {rmse(obs, nwp):.0f} / {solfus_seed_rmse:.0f}"
        ax.text(
            0.04,
            0.95,
            metric_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="#303030",
            bbox={"facecolor": (1, 1, 1, 0.84), "edgecolor": "#b7b7b7", "linewidth": 0.55,
                  "boxstyle": "round,pad=0.25"},
            zorder=8,
        )

        wax = axes[1, col]
        wax.fill_between(leads, 0.0, nwp_w, color=COLORS["nwp"], alpha=0.12, lw=0)
        wax.fill_between(leads, nwp_w, 1.0, color=COLORS["satellite"], alpha=0.12, lw=0)
        wax.plot(leads, sat_w, color=COLORS["satellite"], marker="s", ms=2.8, lw=1.45)
        wax.plot(leads, nwp_w, color=COLORS["nwp"], marker="^", ms=3.0, lw=1.45)
        wax.set_xlim(0.8, 6.2)
        wax.set_ylim(0.0, 1.0)
        wax.set_xticks(leads)
        wax.set_xticklabels([f"{lead} h" for lead in leads])
        wax.set_yticks([0.0, 0.5, 1.0])
        wax.grid(axis="y", color="#dedede", lw=0.45, zorder=0)
        wax.set_xlabel("Lead time", labelpad=2)
        if col == 0:
            wax.set_ylabel("Fusion weight")
        else:
            wax.tick_params(labelleft=False)
        wax.tick_params(axis="both", labelsize=7, length=2.2, pad=1.5)

    handles = [
        Line2D([], [], color=COLORS["observation"], marker="o", lw=1.8, ms=3.5, label="Observation"),
        Line2D([], [], color=COLORS["satellite"], marker="s", lw=1.35, ls="--", ms=3.0,
               label="Satellite branch / weight"),
        Line2D([], [], color=COLORS["nwp"], marker="^", lw=1.35, ls="-.", ms=3.2,
               label="WRF-NWP branch / weight"),
        Line2D([], [], color=COLORS["solfus"], marker="D", lw=2.0, ms=3.1, label="SolFUS-Net"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=7.2,
        handlelength=2.1,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.115, top=0.87)

    stem = OUT / "fig44_dynamic_fusion_cases"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot(load_plot_data())
