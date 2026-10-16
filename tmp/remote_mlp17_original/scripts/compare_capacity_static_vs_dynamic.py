#!/usr/bin/env python3
"""Compare dynamic and capacity-matched static-gate test predictions.

The script treats the three seed CSVs as repeated model fits, while treating
the 17 forecast cycles as the paired resampling unit.  It is intentionally
read-only with respect to the input test directories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 43, 44)
REQUIRED_COLUMNS = {
    "sample_id", "station_id", "cycle", "lead", "daylight", "target", "prediction",
}
IDENTITY_COLUMNS = ("sample_id", "station_id", "cycle", "lead")
METRIC_NAMES = ("rmse", "nrmse_pct", "nmbe_pct", "r2")


def _parse_daylight(value: str, path: Path, row_number: int) -> int:
    value = value.strip().lower()
    if value in {"1", "true", "yes"}:
        return 1
    if value in {"0", "false", "no"}:
        return 0
    raise ValueError(f"invalid daylight value in {path} row {row_number}: {value!r}")


def _read_prediction_csv(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"prediction CSV is empty: {path}")

    keys = []
    target = []
    daylight = []
    prediction = []
    for row_number, row in enumerate(rows, start=2):
        keys.append(tuple(row[column] for column in IDENTITY_COLUMNS))
        try:
            target.append(float(row["target"]))
            prediction.append(float(row["prediction"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric target/prediction in {path} row {row_number}") from exc
        daylight.append(_parse_daylight(row["daylight"], path, row_number))
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate identity key in {path}")
    return {
        "path": str(path),
        "keys": tuple(keys),
        "target": np.asarray(target, dtype=np.float64),
        "daylight": np.asarray(daylight, dtype=bool),
        "prediction": np.asarray(prediction, dtype=np.float64),
        "cycles": tuple(sorted({key[2] for key in keys})),
    }


def _check_compatible(reference: dict, candidate: dict, label: str) -> None:
    if candidate["keys"] != reference["keys"]:
        first_difference = next(
            (index for index, (left, right) in enumerate(zip(reference["keys"], candidate["keys"])) if left != right),
            min(len(reference["keys"]), len(candidate["keys"])),
        )
        raise ValueError(
            f"{label} identity keys/order differ at row {first_difference}: "
            f"reference={reference['keys'][first_difference] if first_difference < len(reference['keys']) else None}, "
            f"candidate={candidate['keys'][first_difference] if first_difference < len(candidate['keys']) else None}"
        )
    if not np.array_equal(candidate["daylight"], reference["daylight"]):
        raise ValueError(f"{label} daylight mask differs from the reference CSV")
    if not np.array_equal(candidate["target"], reference["target"]):
        difference = np.max(np.abs(candidate["target"] - reference["target"]))
        raise ValueError(f"{label} target values differ from the reference CSV; max_abs_difference={difference}")


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if target.size == 0:
        raise ValueError("metric mask contains no daylight points")
    error = prediction - target
    mean_target = float(target.mean())
    rmse = float(np.sqrt(np.mean(error * error)))
    denominator = float(np.sum((target - mean_target) ** 2))
    if mean_target <= 0 or denominator <= 0:
        raise ValueError("daylight target has invalid mean or variance")
    return {
        "count": int(target.size),
        "rmse": rmse,
        "nrmse_pct": float(100.0 * rmse / mean_target),
        "nmbe_pct": float(100.0 * np.mean(error) / mean_target),
        "r2": float(1.0 - np.sum(error * error) / denominator),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_seed_metrics(seed_metrics: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    rows = []
    summary = {}
    for configuration, metrics_by_seed in seed_metrics.items():
        for seed, metric in zip(SEEDS, metrics_by_seed):
            rows.append({
                "configuration": configuration, "model": "seed", "seed": seed,
                **{name: metric[name] for name in METRIC_NAMES}, "count": metric["count"],
            })
        values = {name: np.asarray([metric[name] for metric in metrics_by_seed], dtype=float) for name in METRIC_NAMES}
        summary[configuration] = {
            "seed_mean": {name: float(values[name].mean()) for name in METRIC_NAMES},
            "seed_sd": {name: float(values[name].std(ddof=1)) for name in METRIC_NAMES},
        }
        rows.append({
            "configuration": configuration, "model": "seed_mean", "seed": "mean",
            **{name: float(values[name].mean()) for name in METRIC_NAMES},
            "count": metrics_by_seed[0]["count"],
        })
        rows.append({
            "configuration": configuration, "model": "seed_sd", "seed": "sd",
            **{name: float(values[name].std(ddof=1)) for name in METRIC_NAMES},
            "count": metrics_by_seed[0]["count"],
        })
    return rows, summary


def _paired_bootstrap(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    cycles: tuple[str, ...],
    cycle_indices: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    deltas = np.empty((repetitions, len(METRIC_NAMES)), dtype=np.float64)
    predictions_daylight = {
        configuration: values[:, :] for configuration, values in predictions.items()
    }
    for replicate in range(repetitions):
        sampled_cycle_indices = rng.integers(0, len(cycles), size=len(cycles))
        row_indices = np.concatenate([cycle_indices[cycles[index]] for index in sampled_cycle_indices])
        target_sample = target[row_indices]
        metrics_by_configuration = {}
        for configuration, values in predictions_daylight.items():
            seed_values = []
            for seed_index in range(len(SEEDS)):
                seed_values.append(_metrics(target_sample, values[seed_index, row_indices]))
            metrics_by_configuration[configuration] = np.asarray(
                [[metric[name] for name in METRIC_NAMES] for metric in seed_values], dtype=np.float64,
            ).mean(axis=0)
        deltas[replicate] = metrics_by_configuration["static_mean"] - metrics_by_configuration["dynamic"]

    result = {}
    for index, name in enumerate(METRIC_NAMES):
        values = deltas[:, index]
        result[name] = {
            "bootstrap_mean_delta": float(values.mean()),
            "bootstrap_median": float(np.median(values)),
            "bootstrap_sd": float(values.std(ddof=1)),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-test-dir", type=Path, required=True)
    parser.add_argument("--static-test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output directory: {args.output_dir}")

    configurations = {"dynamic": args.full_test_dir, "static_mean": args.static_test_dir}
    reference = None
    loaded = {}
    for configuration, directory in configurations.items():
        seed_data = []
        for seed in SEEDS:
            current = _read_prediction_csv(directory / f"mlp_seed{seed}_test.csv")
            if reference is None:
                reference = current
            else:
                _check_compatible(reference, current, f"{configuration} seed {seed}")
            seed_data.append(current)
        ensemble_path = directory / "mlp_ensemble_test.csv"
        ensemble = None
        if ensemble_path.is_file():
            ensemble = _read_prediction_csv(ensemble_path)
            _check_compatible(reference, ensemble, f"{configuration} ensemble")
        loaded[configuration] = {"seeds": seed_data, "ensemble": ensemble}

    if reference is None:
        raise RuntimeError("no test predictions were loaded")
    if len(reference["cycles"]) != 17:
        raise ValueError(f"expected 17 cycles, found {len(reference['cycles'])}")
    daylight = reference["daylight"]
    daylight_count = int(daylight.sum())
    if daylight_count != 601:
        raise ValueError(f"expected 601 daylight points, found {daylight_count}")

    target_daylight = reference["target"][daylight]
    seed_metrics = {"dynamic": [], "static_mean": []}
    metric_rows = []
    ensemble_metrics = {}
    prediction_arrays = {}
    for configuration, bundle in loaded.items():
        prediction_arrays[configuration] = np.stack([
            item["prediction"][daylight] for item in bundle["seeds"]
        ])
        for item in bundle["seeds"]:
            seed_metrics[configuration].append(_metrics(target_daylight, item["prediction"][daylight]))
        ensemble = bundle["ensemble"]
        if ensemble is not None:
            ensemble_prediction = ensemble["prediction"][daylight]
            ensemble_source = "mlp_ensemble_test.csv"
        else:
            ensemble_prediction = prediction_arrays[configuration].mean(axis=0)
            ensemble_source = "mean_of_seed_predictions"
        ensemble_metrics[configuration] = _metrics(target_daylight, ensemble_prediction)
        metric_rows.append({
            "configuration": configuration, "model": "ensemble", "seed": "ensemble",
            **{name: ensemble_metrics[configuration][name] for name in METRIC_NAMES},
            "count": daylight_count,
        })

    seed_rows, seed_summary = _summarize_seed_metrics(seed_metrics)
    metric_rows.extend(seed_rows)
    cycles = reference["cycles"]
    daylight_cycle_labels = np.asarray(
        [key[2] for key in reference["keys"]], dtype=object,
    )[daylight]
    cycle_indices = {
        cycle: np.flatnonzero(daylight_cycle_labels == cycle)
        for cycle in cycles
    }
    if any(len(indices) == 0 for indices in cycle_indices.values()):
        raise ValueError("every cycle must contain at least one daylight point")
    bootstrap = _paired_bootstrap(
        target_daylight, prediction_arrays, cycles,
        cycle_indices,
        args.bootstrap_reps, args.bootstrap_seed,
    )
    for name in METRIC_NAMES:
        bootstrap[name]["point_delta_static_minus_dynamic"] = float(
            seed_summary["static_mean"]["seed_mean"][name]
            - seed_summary["dynamic"]["seed_mean"][name]
        )

    args.output_dir.mkdir(parents=True)
    metric_fields = ["configuration", "model", "seed", "count", *METRIC_NAMES]
    _write_csv(args.output_dir / "metrics.csv", metric_fields, metric_rows)
    bootstrap_rows = [
        {
            "metric": name, **values,
            "bootstrap_reps": args.bootstrap_reps, "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "17 GFS cycles", "seed_aggregation": "mean of three seed metrics",
        }
        for name, values in bootstrap.items()
    ]
    _write_csv(
        args.output_dir / "paired_bootstrap.csv",
        ["metric", "point_delta_static_minus_dynamic", "bootstrap_mean_delta", "bootstrap_median", "bootstrap_sd", "ci_low", "ci_high",
         "bootstrap_reps", "bootstrap_seed", "bootstrap_unit", "seed_aggregation"],
        bootstrap_rows,
    )
    summary = {
        "schema_version": "capacity_static_vs_dynamic_v1",
        "input_directories": {configuration: str(directory.resolve()) for configuration, directory in configurations.items()},
        "seed_ids": list(SEEDS), "daylight_points": daylight_count, "cycles": list(cycles),
        "cycle_count": len(cycles), "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed, "bootstrap_unit": "17 GFS cycles",
        "seed_aggregation": "within each replicate, compute each seed metric then average three seeds",
        "metric_definitions": {
            "rmse": "sqrt(mean((prediction-target)^2))",
            "nrmse_pct": "100*RMSE/mean(target)",
            "nmbe_pct": "100*mean(prediction-target)/mean(target)",
            "r2": "1-SSE/SST",
        },
        "seed_metrics": seed_summary,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_sources": {
            configuration: (
                "mlp_ensemble_test.csv" if loaded[configuration]["ensemble"] is not None
                else "mean_of_seed_predictions"
            ) for configuration in configurations
        },
        "paired_bootstrap": bootstrap,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "daylight_points": daylight_count,
        "cycles": len(cycles),
        "seed_mean": seed_summary,
        "ensemble": ensemble_metrics,
        "paired_bootstrap": bootstrap,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
