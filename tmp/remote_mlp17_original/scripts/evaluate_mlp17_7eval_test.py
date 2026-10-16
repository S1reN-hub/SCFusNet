#!/usr/bin/env python3
"""One-time seven-station test evaluation for the frozen dynamic MLP experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from ghi_hierarchical_v4.data import MultisourceCaseDataset, read_index  # noqa: E402
from ghi_hierarchical_v4.model import ScenarioDrivenGHIModelV4  # noqa: E402
from scripts.cache_expert_states_17train_7eval import extract  # noqa: E402
from scripts.train_hierarchical_799_v4 import select_station_aware_partitions  # noqa: E402


GATE_INPUT_KEYS = ("s", "n", "context")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("dynamic_mlp_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_audit(data: dict, prediction: torch.Tensor) -> dict:
    valid = data["valid"].bool().cpu()
    day = data["day"].bool().cpu()
    prediction = prediction.detach().cpu()
    upper = data["physical_upper"].detach().cpu() if "physical_upper" in data else None
    negative = prediction < 0
    above = prediction > upper if upper is not None else None
    night = valid & ~day
    night_nonzero = night & (prediction.abs() > 1e-6)

    def rate(mask: torch.Tensor, denominator: torch.Tensor = valid) -> float | None:
        count = int(denominator.sum())
        return float(mask[denominator].float().mean()) if count else None

    return {
        "valid_points": int(valid.sum()),
        "negative_ghi_rate": rate(negative),
        "physical_upper_evaluable": upper is not None,
        "physical_upper_exceedance_rate": rate(above) if above is not None else None,
        "night_nonzero_evaluable": bool(night.any()),
        "night_valid_points": int(night.sum()),
        "night_nonzero_rate": rate(night_nonzero, night) if night.any() else None,
        "negative_ghi_count": int((negative & valid).sum()),
        "physical_upper_exceedance_count": int((above & valid).sum()) if above is not None else None,
        "night_nonzero_count": int(night_nonzero.sum()),
    }


def write_physical_audit_csv(path: Path, data: dict, prediction: torch.Tensor) -> None:
    prediction = prediction.detach().cpu()
    upper = data["physical_upper"].detach().cpu() if "physical_upper" in data else None
    valid = data["valid"].bool().cpu()
    day = data["day"].bool().cpu()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sample_id", "station_id", "cycle", "lead", "daylight", "prediction",
            "physical_upper", "negative_ghi", "above_physical_upper", "night_nonzero",
        ])
        for i, row in enumerate(data["meta"]):
            for h in range(6):
                if not valid[i, h]:
                    continue
                value = float(prediction[i, h])
                bound = float(upper[i, h]) if upper is not None else ""
                above = int(value > bound) if upper is not None else ""
                writer.writerow([
                    row["sample_id"], row["station_id"], row["cycle"], h + 1,
                    int(day[i, h]), value, bound, int(value < 0), above,
                    int((not bool(day[i, h])) and abs(value) > 1e-6),
                ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-normalization", type=Path, required=True)
    parser.add_argument("--aod-reference", type=Path, required=True)
    parser.add_argument("--expert-ablation", default="fixed_router")
    parser.add_argument("--satellite-path", choices=("direct", "dynamic", "dual"), default="dual")
    parser.add_argument("--routing-quality", choices=("legacy", "history_features", "history_risk"), default="legacy")
    parser.add_argument("--heads-dir", type=Path, required=True)
    parser.add_argument("--reference-implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gate-input-mode", choices=("dynamic", "static_mean"), default=None,
        help=("override the gate-input mode recorded by the trained heads; "
              "static_mean replaces normalized s/n/context with zero"),
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace test output: {args.output_dir}")
    heads_protocol = args.heads_dir / "protocol.json"
    protocol_gate_input_mode = "dynamic"
    if heads_protocol.is_file():
        protocol = json.loads(heads_protocol.read_text(encoding="utf-8"))
        if protocol.get("expert_ablation") not in (None, args.expert_ablation):
            raise ValueError("dynamic MLP heads ablation does not match the requested expert ablation")
        protocol_gate_input_mode = protocol.get("gate_input_mode", "dynamic")
    gate_input_mode = args.gate_input_mode or protocol_gate_input_mode
    if gate_input_mode != protocol_gate_input_mode:
        raise ValueError(
            f"gate-input mode mismatch: heads={protocol_gate_input_mode!r}, "
            f"requested={gate_input_mode!r}"
        )
    device = torch.device(args.device)
    reference = load_module(args.reference_implementation)
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    cycles = config["split_cycles"]
    rows = read_index(args.dataset_root)
    _train_rows, _validation_rows, test_rows = select_station_aware_partitions(
        rows, set(cycles["train"]), set(cycles["validation"]), set(cycles["test"]),
        {value.upper() for value in config["evaluation_stations"]},
        {value.upper() for value in config["training_stations"]},
    )
    if len(test_rows) != 119:
        raise ValueError(f"expected 119 seven-station test cases, found {len(test_rows)}")

    expert_normalization = json.loads(args.expert_normalization.read_text(encoding="utf-8"))
    aod_reference = json.loads(args.aod_reference.read_text(encoding="utf-8"))
    dataset = MultisourceCaseDataset(
        args.dataset_root, test_rows, expert_normalization, aod_reference,
        include_future_satellite_auxiliary=False, require_formal_v4=True,
    )
    checkpoint = torch.load(args.expert_checkpoint, map_location=device, weights_only=False)
    checkpoint_ablation = checkpoint.get("ablation")
    if checkpoint_ablation is not None and checkpoint_ablation != args.expert_ablation:
        raise ValueError("expert checkpoint ablation does not match the requested expert ablation")
    expert = ScenarioDrivenGHIModelV4(
        ablation=args.expert_ablation, satellite_path=args.satellite_path,
        routing_quality=args.routing_quality,
    ).to(device)
    expert.load_state_dict(checkpoint["model"], strict=True)
    expert.eval()
    for parameter in expert.parameters():
        parameter.requires_grad_(False)

    args.output_dir.mkdir(parents=True)
    cache_path = args.output_dir / "test_cache.pt"
    test_cycles = extract(expert, dataset, cache_path, device)
    if len(test_cycles) != 17:
        raise ValueError(f"expected 17 test cycles, found {len(test_cycles)}")
    test = torch.load(cache_path, weights_only=False)
    if {row["station_id"] for row in test["meta"]} != set(config["test_stations"]):
        raise ValueError("test cache contains a station outside the seven-station cohort")

    head_paths = [args.heads_dir / f"mlp_seed{seed}" / "best.pt" for seed in (42, 43, 44)]
    head_hashes = {str(path): sha256(path) for path in head_paths}
    predictions, weights, runs, audits = [], [], {}, {}
    temporal_prediction = temporal_weight = None
    for seed, path in zip((42, 43, 44), head_paths):
        head = torch.load(path, map_location="cpu", weights_only=False)
        normalized = {}
        for key, value in test.items():
            normalized[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        for key, parameters in head["normalization"].items():
            mean = torch.tensor(parameters["mean"], device=device).reshape(1, 1, -1)
            std = torch.tensor(parameters["std"], device=device).reshape(1, 1, -1)
            normalized[key] = (normalized[key] - mean) / std
        if gate_input_mode == "static_mean":
            for key in GATE_INPUT_KEYS:
                normalized[key] = torch.zeros_like(normalized[key])
        model = reference.DynamicFusion(head["dims"], head["initial_weights"], "mlp").to(device)
        model.load_state_dict(head["model"], strict=True)
        model.eval()
        with torch.inference_mode():
            prediction, dynamic_weight = model(normalized)
        prediction, dynamic_weight = prediction.cpu(), dynamic_weight.cpu()
        predictions.append(prediction)
        weights.append(dynamic_weight)
        metric = reference.metrics(test["y"], prediction, test["valid"], test["day"], test["meta"])
        metric["validation_best_epoch"] = int(head["best_epoch"])
        runs[f"mlp_seed{seed}"] = metric
        reference.export(args.output_dir / f"mlp_seed{seed}_test.csv", test, prediction, dynamic_weight)
        audits[f"mlp_seed{seed}"] = physical_audit(test, prediction)
        write_physical_audit_csv(
            args.output_dir / f"mlp_seed{seed}_test_physical_audit.csv", test, prediction,
        )
        if temporal_prediction is None:
            logits = torch.stack([torch.logit(head["initial_weights"]), torch.zeros(6)], dim=-1)
            temporal_weight = reference.final_weights(
                logits.unsqueeze(0).expand(len(test["y"]), -1, -1), test["available"],
            )
            temporal_prediction = (temporal_weight * test["prediction"]).sum(-1)

    baseline = reference.metrics(
        test["y"], temporal_prediction, test["valid"], test["day"], test["meta"],
    )
    audits["temporal_baseline"] = physical_audit(test, temporal_prediction)
    ensemble_prediction = torch.stack(predictions).mean(0)
    ensemble_weight = torch.stack(weights).mean(0)
    ensemble = reference.metrics(
        test["y"], ensemble_prediction, test["valid"], test["day"], test["meta"],
    )
    reference.export(args.output_dir / "temporal_baseline_test.csv", test, temporal_prediction, temporal_weight)
    reference.export(args.output_dir / "mlp_ensemble_test.csv", test, ensemble_prediction, ensemble_weight)
    audits["mlp_ensemble"] = physical_audit(test, ensemble_prediction)
    write_physical_audit_csv(
        args.output_dir / "temporal_baseline_test_physical_audit.csv", test, temporal_prediction,
    )
    write_physical_audit_csv(
        args.output_dir / "mlp_ensemble_test_physical_audit.csv", test, ensemble_prediction,
    )
    base_rmse = baseline["daylight"]["rmse"]
    selected_rmse = runs["mlp_seed44"]["daylight"]["rmse"]
    summary = {
        "schema_version": "mlp17_7eval_test_v1", "test_cases": 119, "test_cycles": 17,
        "test_stations": sorted(config["test_stations"]), "selection": "seed44 selected by validation nRMSE",
        "temporal_baseline": baseline, "runs": runs, "ensemble": ensemble,
        "expert_ablation": args.expert_ablation,
        "gate_input_mode": gate_input_mode,
        "gate_input_keys": list(GATE_INPUT_KEYS),
        "gate_input_replacement": (
            "station-balanced train-only normalized mean (zero vector)"
            if gate_input_mode == "static_mean" else "sample-specific normalized inputs"
        ),
        "physical_output_policy": (
            "unbounded direct-GHI/unbounded WRF-anchor residual; no final "
            "night/nonnegative/upper projection"
            if args.expert_ablation == "no_physics" else
            "sigmoid/physical-upper recovery with final night/nonnegative/upper projection"
        ),
        "aod_enabled": args.expert_ablation not in {"no_aod", "no_conditional", "no_physics"},
        "prediction_source": (
            "external dynamic MLP over cached satellite_prediction and wrf_prediction; "
            "V4 internal final_prediction is not used"
        ),
        "physical_audit": audits,
        "selected_seed44_skill_vs_temporal_pct": 100.0 * (1.0 - selected_rmse / base_rmse),
        "expert_checkpoint_sha256": sha256(args.expert_checkpoint), "head_sha256": head_hashes,
        "test_accessed": True,
    }
    reference.save(args.output_dir / "test_summary.json", summary)
    reference.save(args.output_dir / "status.json", {"state": "completed", "test_accessed": True})
    print(json.dumps({
        "temporal_nrmse_pct": baseline["daylight"]["nrmse_pct"],
        "seed_nrmse_pct": {key: value["daylight"]["nrmse_pct"] for key, value in runs.items()},
        "ensemble_nrmse_pct": ensemble["daylight"]["nrmse_pct"],
        "selected_seed44_skill_vs_temporal_pct": summary["selected_seed44_skill_vs_temporal_pct"],
    }, indent=2))


if __name__ == "__main__":
    main()
