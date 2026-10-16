#!/usr/bin/env python3
"""Train the established dynamic MLP on frozen, newly repartitioned expert states."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("dynamic_mlp_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_weights(data: dict) -> torch.Tensor:
    counts = Counter(row["station_id"] for row in data["meta"])
    weights = torch.tensor([1.0 / counts[row["station_id"]] for row in data["meta"]], dtype=torch.double)
    return weights / weights.mean()


def fit_time_weights(data: dict, weights: torch.Tensor) -> torch.Tensor:
    y = data["y"].double()
    prediction = data["prediction"].double()
    mask = data["available"].all(-1) & data["valid"] & data["day"]
    gap = prediction[..., 0] - prediction[..., 1]
    weighted = weights[:, None] * mask
    numerator = (weighted * (y - prediction[..., 1]) * gap).sum(0)
    denominator = (weighted * gap.square()).sum(0)
    result = torch.where(denominator > 1e-12, numerator / denominator.clamp_min(1e-12), 0.5).clamp(0, 1)
    if not ((result > 1e-6) & (result < 1 - 1e-6)).all():
        raise ValueError("analytic temporal weights reached a boundary optimum")
    return result.float()


def weighted_normalization(data: dict, weights: torch.Tensor, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    values = data[key]
    weight = weights.float()[:, None, None]
    denominator = weight.sum() * values.shape[1]
    mean = (values * weight).sum((0, 1), keepdim=True) / denominator
    variance = ((values - mean).square() * weight).sum((0, 1), keepdim=True) / denominator
    return mean, variance.sqrt().clamp_min(0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    reference = load_reference(args.reference_implementation.resolve())
    device = torch.device(args.device)
    train = torch.load(args.cache_dir / "train_cache.pt", weights_only=False)
    validation = torch.load(args.cache_dir / "validation_cache.pt", weights_only=False)
    if (len(train["meta"]), len(validation["meta"])) != (1224, 119):
        raise ValueError("cache counts must be 1224 training and 119 validation cases")
    if {row["station_id"] for row in validation["meta"]} != {"GOB", "DRA", "IZA", "TBL", "SEL", "GWN", "CAB"}:
        raise ValueError("validation cache is not restricted to the seven evaluation stations")

    weights = case_weights(train)
    initial = fit_time_weights(train, weights)
    normalization = {}
    for key in ("s", "n", "context"):
        mean, standard_deviation = weighted_normalization(train, weights, key)
        normalization[key] = {
            "mean": mean.flatten().tolist(), "std": standard_deviation.flatten().tolist(),
        }
        train[key] = (train[key] - mean) / standard_deviation
        validation[key] = (validation[key] - mean) / standard_deviation
    for data in (train, validation):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
    weights = weights.to(device)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    architecture_sha = hashlib.sha256(args.reference_implementation.read_bytes()).hexdigest()
    protocol = {
        "architecture": "established_dynamic_mlp_20260908", "architecture_sha256": architecture_sha,
        "seeds": [int(value) for value in args.seeds.split(",")], "train_cases": 1224,
        "validation_cases": 119, "evaluation_stations": ["GOB", "DRA", "IZA", "TBL", "SEL", "GWN", "CAB"],
        "training_sampling": "station_balanced_with_replacement", "normalization": "station_balanced_train_only",
        "temporal_baseline": "station-balanced analytic train SSE optimum", "test_accessed": False,
        "max_epochs": args.epochs, "patience": args.patience, "batch_size": 32,
    }
    reference.save(args.output_dir / "protocol.json", protocol)
    reference.save(args.output_dir / "normalization.json", normalization)

    dimensions = [train[key].shape[-1] for key in ("s", "n", "context")]
    base_logits = torch.stack([torch.logit(initial), torch.zeros(6)], dim=-1).to(device)
    baseline_weights = reference.final_weights(
        base_logits.unsqueeze(0).expand(len(validation["y"]), -1, -1), validation["available"],
    )
    baseline_prediction = (baseline_weights * validation["prediction"]).sum(-1)
    baseline_metrics = reference.metrics(
        validation["y"], baseline_prediction, validation["valid"], validation["day"], validation["meta"],
    )
    reference.save(args.output_dir / "temporal_baseline_validation.json", {
        "weights": initial.tolist(), "metrics": baseline_metrics,
    })
    reference.export(
        args.output_dir / "temporal_baseline_validation.csv", validation,
        baseline_prediction, baseline_weights,
    )

    results = {}
    for seed in protocol["seeds"]:
        destination = args.output_dir / f"mlp_seed{seed}"
        destination.mkdir()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = reference.DynamicFusion(dimensions, initial, "mlp").to(device)
        model.eval()
        with torch.inference_mode():
            initial_prediction, initial_weight = model(validation)
            if not torch.equal(initial_prediction, baseline_prediction) or not torch.equal(initial_weight, baseline_weights):
                raise AssertionError("dynamic MLP does not initialize exactly at the temporal baseline")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        best_score = baseline_metrics["daylight"]["nrmse_pct"]
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        stale = 0
        valid_points = train["valid"] & train["day"]
        points_per_case = float(valid_points.sum()) / len(valid_points)
        started = time.monotonic()
        generator = torch.Generator(device=device).manual_seed(seed)
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.multinomial(weights, len(weights), replacement=True, generator=generator)
            losses = []
            for ids in order.split(32):
                batch = reference.subset(train, ids)
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(batch)
                use = batch["valid"] & batch["day"]
                loss = (((prediction - batch["y"]) / 100).square() * use).sum() / (len(ids) * points_per_case)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite dynamic MLP loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                prediction, _ = model(validation)
            metrics = reference.metrics(
                validation["y"], prediction, validation["valid"], validation["day"], validation["meta"],
            )
            score = metrics["daylight"]["nrmse_pct"]
            record = {
                "epoch": epoch, "train_loss": float(np.mean(losses)), "daylight_nrmse_pct": score,
                "daylight_rmse": metrics["daylight"]["rmse"], "elapsed_seconds": time.monotonic() - started,
            }
            with (destination / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"mlp_seed{seed} {json.dumps(record)}", flush=True)
            if score < best_score:
                best_score, best_epoch, best_state, stale = score, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            prediction, dynamic_weights = model(validation)
        metrics = reference.metrics(
            validation["y"], prediction, validation["valid"], validation["day"], validation["meta"],
        )
        metrics["best_epoch"] = best_epoch
        reference.save(destination / "metrics.json", metrics)
        reference.export(destination / "validation.csv", validation, prediction, dynamic_weights)
        torch.save({
            "model": best_state, "kind": "mlp", "dims": dimensions, "initial_weights": initial,
            "normalization": normalization, "best_epoch": best_epoch, "seed": seed,
        }, destination / "best.pt")
        results[f"mlp_seed{seed}"] = metrics
    reference.save(args.output_dir / "summary.json", {
        "temporal_baseline": baseline_metrics, "runs": results, "test_accessed": False,
    })
    reference.save(args.output_dir / "status.json", {"state": "completed", "test_accessed": False})
    print("MLP_HEADS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
