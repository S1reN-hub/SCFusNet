"""Create the Section 4.5 AOD-correction case-study figure.

The three cases were selected because the conditional AOD correction improves
every admitted target hour and lowers admitted-hour RMSE for all three seeds.
The upper row shows the GHI response; the lower row links AOD anomaly to the
correction magnitude applied at each forecast lead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "outputs/aod_hardgate_relaxation_v1_20260913/d10_c70"
SEEDS = (42, 43, 44)

CASES = [
    {
        "station_id": "SEL",
        "cycle": 2024020900,
        "panel": "a",
        "title": "Sustained aerosol loading",
        "station_cycle": "SEL | 2024-02-09",
    },
    {
        "station_id": "GOB",
        "cycle": 2023090200,
        "panel": "b",
        "title": "Increasing aerosol loading",
        "station_cycle": "GOB | 2023-09-02",
    },
    {
        "station_id": "DRA",
        "cycle": 2025071500,
        "panel": "c",
        "title": "Early-hour correction",
        "station_cycle": "DRA | 2025-07-15",
    },
]

COLORS = {
    "observation": "#272727",
    "base": "#D9903D",
    "corrected": "#0F4D92",
    "aod": "#7A5195",
    "reference": "#767676",
    "adjustment": "#8BCF8B",
    "admitted": "#E8F1F8",
}


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        frame = pd.read_csv(SOURCE_DIR / f"test_seed{seed}.csv")
        frame["seed"] = seed
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    selected_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    for case in CASES:
        subset = raw[
            (raw["station_id"] == case["station_id"])
            & (raw["cycle"] == case["cycle"])
        ].copy()
        lead_counts = subset.groupby("seed")["lead"].nunique()
        if len(lead_counts) != len(SEEDS) or lead_counts.nunique() != 1:
            raise ValueError(f"Incomplete case: {case['station_id']} {case['cycle']}")

        grouped = (
            subset.groupby("lead", as_index=False)
            .agg(
                target=("target", "mean"),
                before=("before", "mean"),
                after=("after", "mean"),
                aod550=("aod550", "mean"),
                aod_reference=("aod_reference", "mean"),
                common_clear=("common_clear", "mean"),
                admitted=("admitted", "max"),
            )
            .sort_values("lead")
        )
        grouped["station_id"] = case["station_id"]
        grouped["cycle"] = case["cycle"]
        grouped["adjustment"] = grouped["before"] - grouped["after"]
        selected_parts.append(grouped)

        seed_before: list[float] = []
        seed_after: list[float] = []
        for seed in SEEDS:
            seed_data = subset[(subset["seed"] == seed) & subset["admitted"]].copy()
            seed_before.append(
                rmse(seed_data["target"].to_numpy(), seed_data["before"].to_numpy())
            )
            seed_after.append(
                rmse(seed_data["target"].to_numpy(), seed_data["after"].to_numpy())
            )
        before_rmse = float(np.mean(seed_before))
        after_rmse = float(np.mean(seed_after))
        metric_rows.append(
            {
                "station_id": case["station_id"],
                "cycle": case["cycle"],
                "admitted_hours": int(grouped["admitted"].sum()),
                "before_rmse": before_rmse,
                "after_rmse": after_rmse,
                "rmse_reduction_pct": 100.0 * (before_rmse - after_rmse) / before_rmse,
                "mean_adjustment": float(grouped.loc[grouped["admitted"], "adjustment"].mean()),
                "max_adjustment": float(grouped.loc[grouped["admitted"], "adjustment"].max()),
                "min_aod_anomaly": float(
                    (grouped.loc[grouped["admitted"], "aod550"]
                     - grouped.loc[grouped["admitted"], "aod_reference"]).min()
                ),
                "max_aod_anomaly": float(
                    (grouped.loc[grouped["admitted"], "aod550"]
                     - grouped.loc[grouped["admitted"], "aod_reference"]).max()
                ),
                "min_common_clear": float(grouped.loc[grouped["admitted"], "common_clear"].min()),
            }
        )

    plot_data = pd.concat(selected_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    plot_data.to_csv(OUT / "figure45_plot_data.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "figure45_case_metrics.csv", index=False, encoding="utf-8-sig")
    return plot_data, metrics


def shade_admitted(ax: plt.Axes, leads: np.ndarray, admitted: np.ndarray) -> None:
    for lead, flag in zip(leads, admitted):
        if flag:
            ax.axvspan(lead - 0.43, lead + 0.43, color=COLORS["admitted"], lw=0, zorder=0)


def plot(data: pd.DataFrame, metrics: pd.DataFrame) -> None:
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
        figsize=(7.10, 4.45),
        gridspec_kw={"height_ratios": [2.25, 1.15], "hspace": 0.14, "wspace": 0.25},
        sharex="col",
    )

    for col, case in enumerate(CASES):
        subset = data[
            (data["station_id"] == case["station_id"])
            & (data["cycle"] == case["cycle"])
        ].sort_values("lead")
        metric = metrics[
            (metrics["station_id"] == case["station_id"])
            & (metrics["cycle"] == case["cycle"])
        ].iloc[0]

        leads = subset["lead"].to_numpy(dtype=float)
        admitted = subset["admitted"].to_numpy(dtype=bool)
        obs = subset["target"].to_numpy(dtype=float)
        base = subset["before"].to_numpy(dtype=float)
        corrected = subset["after"].to_numpy(dtype=float)

        ax = axes[0, col]
        shade_admitted(ax, leads, admitted)
        ax.plot(leads, obs, color=COLORS["observation"], marker="o", ms=3.6, lw=1.8, zorder=5)
        ax.plot(
            leads,
            base,
            color=COLORS["base"],
            marker="s",
            ms=3.1,
            lw=1.45,
            ls="--",
            zorder=3,
        )
        ax.plot(
            leads,
            corrected,
            color=COLORS["corrected"],
            marker="D",
            ms=3.2,
            lw=2.0,
            zorder=6,
        )
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
        ax.text(
            0.04,
            0.08,
            (
                f"RMSE  {metric['before_rmse']:.1f} $\\rightarrow$ "
                f"{metric['after_rmse']:.1f} W m$^{{-2}}$\n"
                f"Reduction  {metric['rmse_reduction_pct']:.1f}%"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="#303030",
            bbox={
                "facecolor": (1, 1, 1, 0.86),
                "edgecolor": "#9aa7b2",
                "linewidth": 0.55,
                "boxstyle": "round,pad=0.25",
            },
            zorder=8,
        )

        aod_ax = axes[1, col]
        shade_admitted(aod_ax, leads, admitted)
        aod_ax.plot(
            leads,
            subset["aod550"],
            color=COLORS["aod"],
            marker="o",
            ms=3.0,
            lw=1.55,
            zorder=4,
        )
        aod_ax.plot(
            leads,
            subset["aod_reference"],
            color=COLORS["reference"],
            lw=1.25,
            ls=":",
            zorder=3,
        )
        aod_ax.set_ylim(0, 0.31)
        aod_ax.set_yticks([0.0, 0.1, 0.2, 0.3])
        aod_ax.grid(axis="y", color="#dedede", lw=0.45, zorder=0)
        aod_ax.set_xlim(0.7, 6.3)
        aod_ax.set_xticks(leads)
        aod_ax.set_xticklabels([f"{int(lead)} h" for lead in leads])
        aod_ax.set_xlabel("Lead time", labelpad=2)
        if col == 0:
            aod_ax.set_ylabel("AOD$_{550}$")
        else:
            aod_ax.tick_params(labelleft=False)
        aod_ax.tick_params(axis="both", labelsize=7, length=2.2, pad=1.5)

        adjustment_ax = aod_ax.twinx()
        adjustment_ax.bar(
            leads,
            subset["adjustment"],
            width=0.34,
            color=COLORS["adjustment"],
            edgecolor="#4f8658",
            linewidth=0.55,
            alpha=0.88,
            zorder=2,
        )
        adjustment_ax.set_ylim(0, 27)
        adjustment_ax.set_yticks([0, 10, 20])
        adjustment_ax.spines["top"].set_visible(False)
        if col == 2:
            adjustment_ax.set_ylabel("GHI reduction (W m$^{-2}$)", labelpad=4)
            adjustment_ax.tick_params(axis="y", labelsize=7, length=2.2, pad=1.5)
        else:
            adjustment_ax.tick_params(right=False, labelright=False)

    handles = [
        Line2D([], [], color=COLORS["observation"], marker="o", lw=1.8, ms=3.6,
               label="Observation"),
        Line2D([], [], color=COLORS["base"], marker="s", lw=1.45, ls="--", ms=3.1,
               label="Base prediction"),
        Line2D([], [], color=COLORS["corrected"], marker="D", lw=2.0, ms=3.2,
               label="AOD-corrected prediction"),
        Line2D([], [], color=COLORS["aod"], marker="o", lw=1.55, ms=3.0,
               label="AOD$_{550}$"),
        Line2D([], [], color=COLORS["reference"], lw=1.25, ls=":",
               label="AOD background"),
        Patch(facecolor=COLORS["adjustment"], edgecolor="#4f8658", linewidth=0.55,
              label="GHI reduction"),
        Patch(facecolor=COLORS["admitted"], edgecolor="#9fb8ca", linewidth=0.55,
              label="AOD gate active"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        fontsize=6.4,
        handlelength=1.8,
        columnspacing=0.72,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.935, bottom=0.12, top=0.865)

    stem = OUT / "fig45_aod_correction_cases"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot(*load_data())
