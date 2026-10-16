#!/usr/bin/env python3
"""Freeze the retrained experts and cache train/validation states for the dynamic MLP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from ghi_hierarchical_v4.data import MultisourceCaseDataset, read_index  # noqa: E402
from ghi_hierarchical_v4.model import ScenarioDrivenGHIModelV4  # noqa: E402
from scripts import train_hierarchical_799_v4 as trainer  # noqa: E402


def extract(model, dataset, output: Path, device: torch.device) -> set[str]:
    snapshots: dict[str, dict] = {}
    handles = [
        model.satellite_expert.register_forward_hook(lambda _m, _i, value: snapshots.update(s=value)),
        model.wrf_expert.register_forward_hook(lambda _m, _i, value: snapshots.update(n=value)),
    ]
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=trainer.numpy_safe_collate, pin_memory=device.type == "cuda",
    )
    arrays = {key: [] for key in ("s", "n", "context", "prediction", "available", "y", "valid", "day")}
    meta: list[dict[str, str]] = []
    with torch.inference_mode():
        for raw in loader:
            batch = trainer.to_device(raw, device)
            result = model(batch)
            satellite, nwp = snapshots["s"], snapshots["n"]

            def feature(expert: dict, names: list[str]) -> torch.Tensor:
                return torch.cat(
                    [expert["state"], *[expert[name].float().unsqueeze(-1) for name in names]], dim=-1,
                )

            arrays["s"].append(feature(satellite, [
                "q", "scale_q", "clear_logit", "valid_fraction", "freshness", "dynamics",
                "motion_confidence", "unique_frame_fraction",
            ]).cpu())
            arrays["n"].append(feature(nwp, [
                "q", "scale_q", "clear_logit", "anchor_trust", "transport_residual",
            ]).cpu())
            arrays["context"].append(torch.cat([
                batch["solar_context"], result["motion_wind_consistency"],
                (satellite["q"] - nwp["q"]).unsqueeze(-1),
            ], dim=-1).cpu())
            arrays["prediction"].append(torch.stack([
                result["satellite_prediction"], result["wrf_prediction"],
            ], dim=-1).cpu())
            arrays["available"].append(torch.stack([
                satellite["available"], nwp["available"],
            ], dim=-1).bool().cpu())
            arrays["y"].append(batch["target"].cpu())
            arrays["valid"].append(batch["target_mask"].bool().cpu())
            arrays["day"].append((batch["solar_context"][..., 0] >= 0.0871557427).cpu())
            meta.extend([
                {"sample_id": sample_id, "station_id": raw["station_id"][index], "cycle": raw["cycle"][index]}
                for index, sample_id in enumerate(raw["sample_id"])
            ])
    for handle in handles:
        handle.remove()
    cache = {key: torch.cat(value) for key, value in arrays.items()}
    cache["meta"] = meta
    cache["y"] = torch.where(cache["valid"], cache["y"], torch.zeros_like(cache["y"]))
    if len(meta) != len({row["sample_id"] for row in meta}):
        raise ValueError("duplicate sample IDs in expert cache")
    for key in ("s", "n", "context", "prediction"):
        if not torch.isfinite(cache[key]).all():
            raise FloatingPointError(f"non-finite cached tensor: {key}")
    torch.save(cache, output)
    return {row["cycle"] for row in meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--aod-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    rows = read_index(args.dataset_root)
    cycles = config["split_cycles"]
    train_rows, validation_rows, _test_rows = trainer.select_station_aware_partitions(
        rows, set(cycles["train"]), set(cycles["validation"]), set(cycles["test"]),
        {value.upper() for value in config["evaluation_stations"]},
        {value.upper() for value in config["training_stations"]},
    )
    if (len(train_rows), len(validation_rows)) != (1224, 119):
        raise ValueError("unexpected train/validation case counts")
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    reference = json.loads(args.aod_reference.read_text(encoding="utf-8"))
    train_dataset = MultisourceCaseDataset(
        args.dataset_root, train_rows, normalization, reference,
        include_future_satellite_auxiliary=True, require_formal_v4=True,
    )
    validation_dataset = MultisourceCaseDataset(
        args.dataset_root, validation_rows, normalization, reference,
        include_future_satellite_auxiliary=False, require_formal_v4=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ScenarioDrivenGHIModelV4(ablation="fixed_router", satellite_path="dual").to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_cycles = extract(model, train_dataset, args.output_dir / "train_cache.pt", device)
    validation_cycles = extract(model, validation_dataset, args.output_dir / "validation_cache.pt", device)
    if len(train_cycles) != 86 or len(validation_cycles) != 17:
        raise ValueError("cache cycle coverage differs from station-aware split")
    report = {
        "passed": True, "train_cases": len(train_dataset), "validation_cases": len(validation_dataset),
        "train_cycles": len(train_cycles), "validation_cycles": len(validation_cycles),
        "test_accessed": False, "checkpoint": str(args.checkpoint.resolve()),
    }
    (args.output_dir / "cache_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
