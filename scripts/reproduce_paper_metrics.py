#!/usr/bin/env python3
"""Independently reproduce the published comparison and AOD metrics from CSVs.

Uses only the Python standard library. Metrics are calculated for each seed
before averaging; predictions are never averaged before scoring. Historical
``solfus`` column names are retained as source identifiers for SCFusNet.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
COMPARISON = Path("results/fulltest_model_comparison_20260921/source_data_common_predictions.csv")
PERSISTENCE = Path("outputs/sp7_test_20260912/predictions.csv")
AOD = Path("outputs/aod_hardgate_relaxation_v1_20260913/d10_c70")
VERTICAL = Path("work/wrf_vertical_profile_ablation_20260925_remote/bundle/"
                "wrf_vertical_profile_ablation_20260924_retry1_post_20260925/test_7station")
MODELS = {
    "mynn": "MYNN", "ysu": "YSU", "myj": "MYJ",
    "sat_convlstm": "Sat-ConvLSTM", "wrf_transformer": "WRF-Transformer",
    "early_fusion": "EarlyFusion-Transformer", "solfus": "SCFusNet",
}
# Frozen reference values, in the order MAE, RMSE, nRMSE (%), R2, FS (%).
EXPECTED = {
    "mynn": (77.5659494024, 128.7643481149, 24.3524242149, .82658686596, 20.1877694013),
    "ysu": (77.3865381366, 128.2568407739, 24.2564424137, .82795114114, 19.8057313365),
    "myj": (83.1235373950, 135.8042051684, 25.6838299021, .80710669413, 12.3602296717),
    "sat_convlstm": (79.2803749448, 125.6525323591, 23.7639052773, .83485114673, 21.4197385940),
    "wrf_transformer": (70.5056448390, 123.3709866526, 23.3324103043, .84078165111, 19.9160146648),
    "early_fusion": (69.0958792855, 116.2771074355, 21.9907877313, .85854634104, 26.9247398023),
    "solfus": (62.3010207391, 109.4912140110, 20.7074126530, .87461140606, 34.2341111682),
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(row[name] for name in ("sample_id", "station_id", "cycle", "lead"))


def index(rows):
    result = {key(row): row for row in rows}
    assert len(result) == len(rows), "Duplicate forecast keys"
    return result


def metric(rows, column):
    y = [float(row["target"]) for row in rows]
    p = [float(row[column]) for row in rows]
    assert y and all(math.isfinite(v) for v in y + p), "Missing/nonfinite data"
    error = [b - a for a, b in zip(y, p)]
    mean = statistics.fmean(y)
    sse = math.fsum(e * e for e in error)
    rmse = math.sqrt(sse / len(y))
    return {
        "mae_wm2": statistics.fmean(abs(e) for e in error),
        "rmse_wm2": rmse, "nrmse_pct": 100 * rmse / mean,
        "r2": 1 - sse / math.fsum((v - mean) ** 2 for v in y),
    }


def aggregate(rows):
    return {
        name: {"mean": statistics.fmean(row[name] for row in rows),
               "seed_sd": statistics.stdev(row[name] for row in rows) if len(rows) > 1 else None}
        for name in rows[0]
    }


def reproduce(root):
    rows = read_csv(root / COMPARISON)
    common = index(rows)
    assert len(rows) == 601 and len({r["sample_id"] for r in rows}) == 110
    assert len({r["station_id"] for r in rows}) == 7
    assert len({r["cycle"] for r in rows}) == 17
    assert all(r["daylight_all_models"].lower() == "true" for r in rows)
    sp = index(read_csv(root / PERSISTENCE))
    assert len(sp) == 452 and set(sp) <= set(common)
    paired = []
    for forecast_key, reference in sp.items():
        row = dict(common[forecast_key], sp=reference["sp"])
        assert abs(float(row["target"]) - float(reference["target"])) < 1e-6
        paired.append(row)
    sp_rmse = metric(paired, "sp")["rmse_wm2"]
    result = {"protocol": {"comparison_rows": 601, "fs_rows": 452,
        "seeds": list(SEEDS), "aggregation": "score each seed, then arithmetic mean",
        "nrmse": "100 * RMSE / mean observed daylight GHI",
        "fs": "100 * (1 - model RMSE / smart-persistence RMSE) on paired 452 rows",
        "sp_rmse_wm2": sp_rmse}, "table_iii": {}}
    for model, label in MODELS.items():
        columns = [("deterministic", model)] if model in ("mynn", "ysu", "myj") else [
            (str(seed), f"{model}_seed{seed}") for seed in SEEDS]
        by_seed = {}
        for seed, column in columns:
            values = metric(rows, column)
            values["fs_pct"] = 100 * (1 - metric(paired, column)["rmse_wm2"] / sp_rmse)
            by_seed[seed] = values
        summary = aggregate(list(by_seed.values()))
        for name, expected in zip(("mae_wm2", "rmse_wm2", "nrmse_pct", "r2", "fs_pct"), EXPECTED[model]):
            assert abs(summary[name]["mean"] - expected) < 1e-7, (model, name, summary[name], expected)
        result["table_iii"][label] = {"source_model_key": model, "by_seed": by_seed, "summary": summary}
    # Table II uses the same 601-hour comparison and 452-hour persistence sets.
    # A fixed blend combines source predictions directly; it is not a fitted head.
    for row in rows:
        row["fixed_blend"] = .2 * float(row["satellite"]) + .8 * float(row["nwp"])
    for row in paired:
        row["fixed_blend"] = .2 * float(row["satellite"]) + .8 * float(row["nwp"])
    result["table_ii"] = {}
    ablations = {
        "Satellite expert": ("satellite", (79.46, 124.88, 23.62, 20.97, .837)),
        "NWP expert": ("nwp", (69.50, 126.31, 23.89, 21.34, .833)),
        "Fixed fusion (0.2 satellite + 0.8 NWP)": ("fixed_blend", (63.97, 115.36, 21.82, 29.69, .861)),
        "Vertical-profile ablation": (None, (62.77, 112.93, 21.36, 30.04, .867)),
    }
    for label, (column, expected) in ablations.items():
        by_seed = {}
        for seed in SEEDS if column is None else ("deterministic",):
            if column is None:
                vertical = index(read_csv(root / VERTICAL / f"mlp_seed{seed}_test.csv"))
                assert set(common) <= set(vertical), "Vertical-profile comparison keys missing"
                for forecast_key in common:
                    assert abs(float(common[forecast_key]["target"]) - float(vertical[forecast_key]["target"])) < 1e-6
                    assert str(vertical[forecast_key]["daylight"]).lower() in ("1", "true")
                evaluation_rows = [vertical[k] for k in common]
                fs_rows = [vertical[k] for k in sp]
                source_column = "prediction"
            else:
                evaluation_rows, fs_rows, source_column = rows, paired, column
            values = metric(evaluation_rows, source_column)
            values["fs_pct"] = 100 * (1 - metric(fs_rows, source_column)["rmse_wm2"] / sp_rmse)
            by_seed[str(seed)] = values
        summary = aggregate(list(by_seed.values()))
        for name, target in zip(("mae_wm2", "rmse_wm2", "nrmse_pct", "fs_pct", "r2"), expected):
            digits = 3 if name == "r2" else 2
            assert round(summary[name]["mean"], digits) == target, (label, name, summary[name], target)
        result["table_ii"][label] = {"by_seed": by_seed, "summary": summary}
    result["table_ii"]["SCFusNet"] = result["table_iii"]["SCFusNet"]
    aod_seeds, admitted_keys = {}, None
    for seed in SEEDS:
        aod_rows = read_csv(root / AOD / f"test_seed{seed}.csv")
        assert len(index(aod_rows)) == 601
        admitted = [r for r in aod_rows if r["admitted"].lower() == "true"]
        assert len(admitted) == 38 and len({r["sample_id"] for r in admitted}) == 10
        selected_keys = set(index(admitted))
        if admitted_keys is None:
            admitted_keys = selected_keys
        assert selected_keys == admitted_keys, "AOD seed admission sets differ"
        assert all(float(r["before"]) == float(r["after"]) for r in aod_rows if r["admitted"].lower() != "true")
        assert all(float(r["after"]) <= float(r["before"]) + 1e-5 for r in admitted)
        aod_seeds[str(seed)] = {stage: metric(admitted, stage) for stage in ("before", "after")}
    aod_summary = {stage: aggregate([v[stage] for v in aod_seeds.values()]) for stage in ("before", "after")}
    assert round(aod_summary["before"]["rmse_wm2"]["mean"], 2) == 51.96
    assert round(aod_summary["after"]["rmse_wm2"]["mean"], 2) == 46.04
    result["aod"] = {"admitted_hours": 38, "admitted_cases": 10,
                     "by_seed": aod_seeds, "summary": aod_summary}
    result["assertions_passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("reproduction"))
    args = parser.parse_args()
    result = reproduce(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    (args.output_dir / "paper_metrics.json").write_text(encoded + "\n", encoding="utf-8")
    for table in ("table_ii", "table_iii"):
        with (args.output_dir / f"{table}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["model", "mae_wm2", "rmse_wm2", "nrmse_pct", "r2", "fs_pct"])
            writer.writeheader()
            for model, values in result[table].items():
                writer.writerow(dict(model=model, **{name: v["mean"] for name, v in values["summary"].items()}))
    print(encoded)


if __name__ == "__main__":
    main()
