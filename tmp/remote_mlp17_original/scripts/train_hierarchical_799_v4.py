#!/usr/bin/env python3
"""Train one grouped-CV fold of the v4 wind/cloud-aware GHI model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from ghi_hierarchical_v4.data import (  # noqa: E402
    MultisourceCaseDataset, compute_normalization, cycle_key, data_role, read_index,
)
from ghi_hierarchical_v4.evaluation import (  # noqa: E402
    atomic_save, checkpoint_is_better, evaluate, seed_everything, selection_score, to_device,
)
from ghi_hierarchical_v4.model import ABLATIONS, ScenarioDrivenGHIModelV4  # noqa: E402
from ghi_hierarchical_v4.training import (  # noqa: E402
    STAGES, configure_stage, loss_for_stage, modules_for_stage, set_stage_training_mode,
)


def numpy_safe_collate(rows: list[dict]) -> dict:
    """Collate preloaded CPU tensors into contiguous arrays, preserving dtype."""

    collated: dict[str, object] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        first = values[0]
        if isinstance(first, torch.Tensor):
            collated[key] = torch.from_numpy(np.stack([value.detach().cpu().numpy() for value in values]))
        elif isinstance(first, np.ndarray):
            collated[key] = torch.from_numpy(np.stack(values))
        elif isinstance(first, (bool, int, float, np.generic)):
            collated[key] = torch.as_tensor(values)
        elif isinstance(first, (str, bytes)):
            collated[key] = values
        else:
            raise TypeError(f"unsupported batch value for {key}: {type(first).__name__}")
    return collated


def validate_cycle_partition(
    rows: list[dict[str, str]], selected_cycles: set[str], split_name: str,
    enforce_development_roles: bool = True,
) -> list[dict[str, str]]:
    """Select every available case in each requested cycle without fixed station counts."""

    selected_rows = [row for row in rows if cycle_key(row) in selected_cycles]
    present_cycles = {cycle_key(row) for row in selected_rows}
    if present_cycles != selected_cycles:
        missing = sorted(selected_cycles - present_cycles)
        raise ValueError(f"{split_name} cycles absent from index: {missing}")
    sample_ids = [row["sample_id"] for row in selected_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"duplicate sample_id in {split_name} split")
    if enforce_development_roles:
        roles = {data_role(row) for row in selected_rows}
        if "" in roles:
            raise ValueError(f"{split_name} index rows must declare a development/confirmation role")
        allowed = {"development", "develop", "dev", "train", "training", "cv"}
        disallowed = sorted(role for role in roles if role not in allowed)
        if disallowed:
            raise ValueError(f"held-out roles cannot enter {split_name}: {disallowed}")
    return selected_rows


def restrict_stations(
    rows: list[dict[str, str]], selected_cycles: set[str], station_ids: set[str], split_name: str,
) -> list[dict[str, str]]:
    """Apply an explicit station cohort and require every cycle-station pair."""

    selected = [row for row in rows if row.get("station_id", "").upper() in station_ids]
    present = {(cycle_key(row), row.get("station_id", "").upper()) for row in selected}
    expected = {(cycle, station_id) for cycle in selected_cycles for station_id in station_ids}
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ValueError(
            f"{split_name} station policy mismatch; missing={missing[:12]}, extra={extra[:12]}"
        )
    return selected


def select_station_aware_partitions(
    rows: list[dict[str, str]], train_cycles: set[str], validation_cycles: set[str],
    test_cycles: set[str], evaluation_stations: set[str], training_stations: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Keep all non-evaluation stations in training and split only the evaluation cohort."""

    all_cycles = train_cycles | validation_cycles | test_cycles
    if train_cycles & validation_cycles or train_cycles & test_cycles or validation_cycles & test_cycles:
        raise ValueError("evaluation-station cycle partitions overlap")
    selected = {"train": [], "validation": [], "test": []}
    for row in rows:
        cycle = cycle_key(row)
        station_id = row.get("station_id", "").upper()
        if cycle not in all_cycles or station_id not in training_stations:
            continue
        if station_id not in evaluation_stations:
            selected["train"].append(row)
        elif cycle in train_cycles:
            selected["train"].append(row)
        elif cycle in validation_cycles:
            selected["validation"].append(row)
        elif cycle in test_cycles:
            selected["test"].append(row)
    sample_ids = [row["sample_id"] for values in selected.values() for row in values]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample overlap detected in station-aware split")
    expected = {
        "train": len(all_cycles) * (len(training_stations) - len(evaluation_stations)) + len(train_cycles) * len(evaluation_stations),
        "validation": len(validation_cycles) * len(evaluation_stations),
        "test": len(test_cycles) * len(evaluation_stations),
    }
    actual = {name: len(values) for name, values in selected.items()}
    if actual != expected:
        raise ValueError(f"station-aware split count mismatch: expected={expected}, actual={actual}")
    return selected["train"], selected["validation"], selected["test"]


def train_stage(model, loader, validation_loader, stage, epochs, learning_rate, patience, device, output_dir,
                precision="fp32"):
    stage_started = time.monotonic()
    parameters = configure_stage(model, stage)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    warmup = max(1, int(0.05 * epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup else 0.5 * (
            1.0 + math.cos(math.pi * (step - warmup) / max(epochs - warmup, 1))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_path = output_dir / f"best_{stage}.pt"
    if stage == "aod":
        with torch.no_grad():
            model.aod_gate[-1].weight.zero_()
            model.aod_gate[-1].bias.fill_(-12.0)
            model.aod_strength[-1].weight.zero_()
            model.aod_strength[-1].bias.zero_()
        initial = evaluate(model, validation_loader, "transition", device)
    else:
        initial = evaluate(model, validation_loader, stage, device)
    incoming_overall = float(initial["macro_station_daylight_rmse_wm2"])
    best_validation = initial
    best = selection_score(initial, stage)
    history = [{"stage": stage, "epoch": 0, "validation": initial, "selection_score": best, "role": "incoming_stage_checkpoint"}]
    atomic_save(best_path, {"model": model.state_dict(), "stage": stage, "epoch": 0, "metric": best,
                           "ablation": model.ablation, "physics_constrained": model.physics_constrained,
                           "routing_quality": model.routing_quality, "satellite_path": model.satellite_path})
    if stage == "aod":
        with torch.no_grad():
            model.aod_gate[-1].bias.fill_(-4.0)
    stale = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        set_stage_training_mode(model, stage)
        records = []
        for step, raw in enumerate(loader, 1):
            batch = to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and precision == "bf16"):
                output = model(batch)
            # Probability BCE and physical losses are evaluated in float32.
            output = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                      for key, value in output.items()}
            loss, record = loss_for_stage(output, batch, stage)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite {stage} loss for samples: {list(raw['sample_id'])}"
                )
            if not loss.requires_grad:
                if loss.item() != 0:
                    raise RuntimeError("nonzero training loss has no gradient graph")
                print(json.dumps({"event": "empty_supervision_batch", "stage": stage,
                                  "epoch": epoch, "step": step,
                                  "sample_ids": list(raw["sample_id"])}), flush=True)
                continue  # No optimizer step or weight decay without supervision.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            bad_gradients = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if bad_gradients:
                raise FloatingPointError(
                    f"non-finite {stage} gradients for samples {list(raw['sample_id'])}: {bad_gradients[:8]}"
                )
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            records.append(record)
            if step == 1 or step % 50 == 0:
                print(json.dumps({"event": "batch_progress", "stage": stage, "epoch": epoch,
                                  "step": step, "steps": len(loader), "loss": record["total"]}), flush=True)
        if not records:
            raise RuntimeError(f"no supervised training batches in stage {stage}")
        scheduler.step()
        validation = evaluate(model, validation_loader, stage, device)
        score = selection_score(validation, stage)
        item = {
            "stage": stage, "epoch": epoch,
            "train": {key: float(np.mean([record[key] for record in records])) for key in records[0]},
            "validation": validation, "selection_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - epoch_started,
            "stage_elapsed_seconds": time.monotonic() - stage_started,
        }
        history.append(item)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)
        if checkpoint_is_better(validation, best_validation, stage, incoming_overall):
            best, best_validation, stale = score, validation, 0
            atomic_save(best_path, {"model": model.state_dict(), "stage": stage, "epoch": epoch, "metric": best,
                                   "ablation": model.ablation, "physics_constrained": model.physics_constrained,
                                   "routing_quality": model.routing_quality, "satellite_path": model.satellite_path})
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    return history


def write_predictions(model, loader, path, device):
    fields = [
        "sample_id", "station_id", "cycle", "lead_hour", "target", "raw_wrf_anchor",
        "satellite_prediction", "wrf_prediction", "base_prediction", "transition_prediction", "final_prediction",
        "satellite_weight", "wrf_weight", "satellite_prior", "route_evidence", "route_reliability", "route_confidence",
        "satellite_scale_wm2", "wrf_scale_wm2", "base_scale_wm2",
        "satellite_motion_confidence", "motion_wind_cosine", "motion_wind_speed_difference",
        "history_transport_skill", "history_transport_available", "history_transport_overlap",
        "history_transport_dt_hours", "history_transport_pairs", "route_quality_cap",
        "wrf_anchor_trust", "wrf_transport_residual", "satellite_source_id",
        "satellite_clear_probability", "wrf_clear_probability", "transition_gate", "aod_gate",
    ]
    model.eval()
    with path.open("w", encoding="utf-8", newline="") as stream, torch.no_grad():
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for raw in loader:
            batch = to_device(raw, device)
            output = model(batch)
            for i, sample_id in enumerate(raw["sample_id"]):
                for h in range(6):
                    if not bool(raw["target_mask"][i, h]):
                        continue
                    writer.writerow({
                        "sample_id": sample_id, "station_id": raw["station_id"][i], "cycle": raw["cycle"][i],
                        "lead_hour": h + 1, "target": float(raw["target"][i, h]), "raw_wrf_anchor": float(raw["wrf_anchor"][i, h]),
                        "satellite_prediction": float(output["satellite_prediction"][i, h]),
                        "wrf_prediction": float(output["wrf_prediction"][i, h]),
                        "base_prediction": float(output["base_prediction"][i, h]),
                        "transition_prediction": float(output["transition_prediction"][i, h]),
                        "final_prediction": float(output["final_prediction"][i, h]),
                        "satellite_weight": float(output["fusion_weights"][i, h, 0]),
                        "wrf_weight": float(output["fusion_weights"][i, h, 1]),
                        "satellite_prior": float(output["lead_prior"][i, h]),
                        "route_evidence": float(output["route_evidence"][i, h]),
                        "route_reliability": float(output["route_reliability"][i, h]),
                        "route_confidence": float(output["route_confidence"][i, h]),
                        "satellite_scale_wm2": float(output["satellite_scale_wm2"][i, h]),
                        "wrf_scale_wm2": float(output["wrf_scale_wm2"][i, h]),
                        "base_scale_wm2": float(output["base_scale_wm2"][i, h]),
                        "satellite_motion_confidence": float(output["satellite_motion_confidence"][i, h]),
                        **{key: float(output[key][i, h]) for key in (
                            "history_transport_skill", "history_transport_available", "history_transport_overlap",
                            "history_transport_dt_hours", "history_transport_pairs", "route_quality_cap")},
                        "motion_wind_cosine": float(output["motion_wind_consistency"][i, h, 0]),
                        "motion_wind_speed_difference": float(output["motion_wind_consistency"][i, h, 1]),
                        "wrf_anchor_trust": float(output["wrf_anchor_trust"][i, h]),
                        "wrf_transport_residual": float(output["wrf_transport_residual"][i, h]),
                        "satellite_source_id": int(raw["satellite_source_id"][i]),
                        "satellite_clear_probability": float(output["satellite_clear_probability"][i, h]),
                        "wrf_clear_probability": float(output["wrf_clear_probability"][i, h]),
                        "transition_gate": float(output["transition_gate"][i, h]),
                        "aod_gate": float(output["aod_gate"][i, h]),
                    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--cv", type=Path, required=True)
    parser.add_argument(
        "--split-config", type=Path,
        help="Optional explicit train/validation station policy for a sanctioned repartition.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-predictions", type=Path)
    parser.add_argument("--clear-labels", type=Path)
    parser.add_argument(
        "--era5-encoder-checkpoint", type=Path,
        help="Optional self-supervised ERA5 initialization for v4 thermo/hydrometeor branches.",
    )
    parser.add_argument(
        "--aod-reference", type=Path,
        help="Training-split-only AOD reference; defaults to dataset statistics/fold_N.",
    )
    parser.add_argument(
        "--normalization-cache", type=Path,
        help="Load cached training-only normalization, or create it when absent.",
    )
    parser.add_argument(
        "--collate-audit-only", action="store_true",
        help="Load and collate every split item, then exit before model construction.",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--satellite-path", choices=("direct", "dynamic", "dual"))
    parser.add_argument("--routing-supervision", choices=("winner", "none", "convex"), default="winner")
    parser.add_argument("--routing-quality", choices=("legacy", "history_features", "history_risk"), default="legacy")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run one real batch forward/backward for every stage, then exit.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--joint-learning-rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", default="satellite:80,nwp:80,fusion:50,transition:25,aod:25,joint:50")
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the driver and PyTorch CUDA wheel")
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a CUDA GPU with bfloat16 support")
    if args.workers != 0:
        raise ValueError("This preloaded dataset uses workers=0 to avoid duplicating host memory")
    print(json.dumps({"event": "runtime", "torch": torch.__version__, "device": str(device),
                      "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      "precision": args.precision}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv = json.loads(args.cv.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if not bool(audit.get("passed")):
        raise ValueError("formal v4 training requires a passed dataset audit")
    if not bool(audit.get("labels_inspected")):
        raise ValueError("development audit must include label checks")
    audited_root = Path(str(audit.get("dataset_root", ""))).resolve()
    if audited_root != args.dataset_root.resolve():
        raise ValueError("audit report belongs to a different dataset root")
    fold = next(value for value in cv["folds"] if int(value["fold"]) == args.fold)
    train_cycles, validation_cycles = set(fold["train"]), set(fold["validation"])
    closed_test = set(cv.get("test", cv.get("closed_test", [])))
    if train_cycles & validation_cycles or (train_cycles | validation_cycles) & closed_test:
        raise ValueError("cycle leakage detected in grouped split")
    station_policy = None
    if args.split_config:
        station_policy = json.loads(args.split_config.read_text(encoding="utf-8"))
        configured_cycles = station_policy.get("split_cycles", {})
        if set(configured_cycles.get("train", [])) != train_cycles:
            raise ValueError("split config train cycles differ from the CV file")
        if set(configured_cycles.get("validation", [])) != validation_cycles:
            raise ValueError("split config validation cycles differ from the CV file")
        if set(configured_cycles.get("test", [])) != closed_test:
            raise ValueError("split config test cycles differ from the CV file")
    rows = read_index(args.dataset_root)
    if int(audit.get("case_count", -1)) != len(rows):
        raise ValueError("dataset index changed after the audit report was generated")
    if station_policy:
        evaluation_stations = {
            value.upper() for value in station_policy.get(
                "evaluation_stations", station_policy["validation_stations"],
            )
        }
        train_rows, validation_rows, test_rows = select_station_aware_partitions(
            rows, train_cycles, validation_cycles, closed_test, evaluation_stations,
            {value.upper() for value in station_policy["training_stations"]},
        )
        configured_counts = station_policy.get("active_case_counts")
        if configured_counts and configured_counts != [len(train_rows), len(validation_rows), len(test_rows)]:
            raise ValueError("station-aware split counts differ from split config")
    else:
        train_rows = validate_cycle_partition(rows, train_cycles, "train")
        validation_rows = validate_cycle_partition(rows, validation_cycles, "validation")
    if args.normalization_cache and args.normalization_cache.is_file():
        normalization = json.loads(args.normalization_cache.read_text(encoding="utf-8"))
    else:
        normalization = compute_normalization(
            [args.dataset_root / row["case_file"] for row in train_rows]
        )
        if args.normalization_cache:
            args.normalization_cache.parent.mkdir(parents=True, exist_ok=True)
            args.normalization_cache.write_text(
                json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
            )
    reference_path = args.aod_reference or (
        args.dataset_root / "statistics" / f"fold_{args.fold}_aod_reference.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    ridge_teacher = None
    if args.ridge_predictions:
        with args.ridge_predictions.open(encoding="utf-8-sig", newline="") as stream:
            ridge_teacher = {(row["sample_id"], int(row["lead_hour"])): float(row["ridge_prediction"]) for row in csv.DictReader(stream)}
    clear_labels = None
    if args.clear_labels:
        with args.clear_labels.open(encoding="utf-8-sig", newline="") as stream:
            clear_labels = {
                (row["sample_id"], int(row["lead_hour"])): (float(row["clear_score"]), bool(int(row["clear_valid"])))
                for row in csv.DictReader(stream)
            }
    train = MultisourceCaseDataset(
        args.dataset_root, train_rows, normalization, reference, ridge_teacher, clear_labels,
        include_future_satellite_auxiliary=True, require_formal_v4=True,
    )
    validation = MultisourceCaseDataset(
        args.dataset_root, validation_rows, normalization, reference, ridge_teacher, clear_labels,
        include_future_satellite_auxiliary=False, require_formal_v4=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    training_sampling = station_policy.get("training_sampling") if station_policy else None
    train_sampler = None
    if training_sampling == "station_balanced":
        station_counts = Counter(row["station_id"] for row in train_rows)
        sample_weights = torch.as_tensor(
            [1.0 / station_counts[row["station_id"]] for row in train_rows], dtype=torch.double,
        )
        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_rows), replacement=True, generator=generator,
        )
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, generator=generator,
        num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=numpy_safe_collate,
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=numpy_safe_collate,
    )
    if args.collate_audit_only:
        counts = {}
        checksums = {}
        for split_name, loader in (("train", train_loader), ("validation", validation_loader)):
            count = 0
            checksum = 0.0
            for batch in loader:
                count += len(batch["sample_id"])
                checksum += float(batch["target"].double().sum())
            counts[split_name] = count
            checksums[split_name] = checksum
        print(json.dumps({
            "collate_audit_passed": True,
            "counts": counts,
            "target_checksums": checksums,
        }), flush=True)
        return
    model = ScenarioDrivenGHIModelV4(ablation=args.ablation, satellite_path=args.satellite_path, routing_quality=args.routing_quality).to(device)
    model.routing_supervision_mode = args.routing_supervision
    if args.era5_encoder_checkpoint:
        pretraining = torch.load(
            args.era5_encoder_checkpoint, map_location=device, weights_only=False
        )
        if bool(pretraining.get("target_or_satellite_accessed", True)):
            raise ValueError("ERA5 encoder checkpoint does not certify input-only pretraining")
        for domain in (model.wrf_expert.d01, model.wrf_expert.d02):
            domain.thermo.load_state_dict(pretraining["thermo"], strict=True)
            domain.hydrometeor.load_state_dict(pretraining["hydrometeor"], strict=True)
        print(json.dumps({
            "era5_encoder_checkpoint_loaded": str(args.era5_encoder_checkpoint.resolve()),
            "transferred_modules": [
                "wrf_expert.d01.thermo", "wrf_expert.d01.hydrometeor",
                "wrf_expert.d02.thermo", "wrf_expert.d02.hydrometeor",
            ],
        }), flush=True)
    if args.resume_checkpoint:
        model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device, weights_only=False)["model"])
    if args.preflight_only:
        batch = to_device(next(iter(train_loader)), device)
        reports = []
        disabled_preflight = {"no_transition": {"transition"}, "no_aod": {"aod"},
                              "no_conditional": {"transition", "aod"},
                              "no_physics": {"aod"}}.get(args.ablation, set())
        preflight_stages = [stage.strip() for stage in args.stages.split(",")
                            if stage.strip() not in disabled_preflight]
        if not preflight_stages or any(stage not in STAGES for stage in preflight_stages):
            raise ValueError(f"invalid preflight stages: {preflight_stages}")
        for stage in preflight_stages:
            model.zero_grad(set_to_none=True)
            configure_stage(model, stage)
            set_stage_training_mode(model, stage)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and args.precision == "bf16"):
                output = model(batch)
            output = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                      for k, v in output.items()}
            loss, _ = loss_for_stage(output, batch, stage)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{stage}: non-finite preflight loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"{stage}: non-finite gradient in {name}")
            reports.append({"stage": stage, "loss": float(loss.detach()),
                            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        (args.output_dir / "preflight.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(json.dumps({"preflight_passed": True, "stages": reports}), flush=True)
        return
    epochs = {item.split(":")[0]: int(item.split(":")[1]) for item in args.epochs.split(",")}
    requested = tuple(value.strip() for value in args.stages.split(",") if value.strip())
    if not requested or any(value not in STAGES for value in requested):
        raise ValueError(f"invalid stages: {requested}")
    disabled = {"no_transition": {"transition"}, "no_aod": {"aod"},
                "no_conditional": {"transition", "aod"}, "no_physics": {"aod"}}.get(args.ablation, set())
    stages = tuple(stage for stage in requested if stage not in disabled)
    history = []
    for stage in stages:
        history.extend(train_stage(
            model, train_loader, validation_loader, stage, epochs[stage],
            args.joint_learning_rate if stage == "joint" else args.learning_rate,
            args.patience, device, args.output_dir, args.precision,
        ))
    components = {stage: evaluate(model, validation_loader, stage, device) for stage in STAGES}
    summary = {
        "schema_version": "scenario_driven_ghi_v4_fold_v1", "fold": args.fold, "seed": args.seed,
        "device": str(device), "train_cases": len(train), "validation_cases": len(validation),
        "precision": args.precision,
        "split_config": str(args.split_config.resolve()) if args.split_config else None,
        "training_sampling": training_sampling or "uniform_cases",
        "train_stations": sorted({row["station_id"] for row in train_rows}),
        "validation_stations": sorted({row["station_id"] for row in validation_rows}),
        "train_cycle_case_counts": {cycle: sum(cycle_key(row) == cycle for row in train_rows) for cycle in sorted({cycle_key(row) for row in train_rows})},
        "validation_cycle_case_counts": {cycle: sum(cycle_key(row) == cycle for row in validation_rows) for cycle in sorted(validation_cycles)},
        "parameter_count": sum(value.numel() for value in model.parameters()), "test_accessed": False,
        "architecture_revision": model.architecture_revision,
        "satellite_path": model.satellite_path,
        "routing_supervision": args.routing_supervision,
        "routing_quality": args.routing_quality,
        "final_validation": components[stages[-1]], "component_validation": components,
        "primary_evaluation_stage": stages[-1],
        "lead_prior": model.lead_prior().detach().cpu().tolist(), "history": history,
        "ablation": args.ablation, "requested_stages": list(requested), "stages_run": list(stages),
        "data_contract": "wind_cloud_v4_dual_domain_7_state",
        "physical_output_policy": (
            "unbounded direct-GHI/unbounded WRF-anchor residual; no final "
            "night/nonnegative/upper projection"
            if args.ablation == "no_physics" else
            "sigmoid/physical-upper recovery with final night/nonnegative/upper projection"
        ),
        "aod_enabled": "aod" in stages,
        "clear_label_policy": "minute_resolution_v1" if args.clear_labels else "hourly_soft_proxy",
        "aod_reference": str(reference_path.resolve()),
        "normalization_cache": (
            str(args.normalization_cache.resolve()) if args.normalization_cache else None
        ),
        "era5_encoder_checkpoint": (
            str(args.era5_encoder_checkpoint.resolve())
            if args.era5_encoder_checkpoint else None
        ),
    }
    (args.output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_predictions(model, validation_loader, args.output_dir / "validation_predictions.csv", device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
