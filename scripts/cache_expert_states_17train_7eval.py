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

# The external dynamic MLP input tensors remain compatible with the original
# schema.  This revision adds an auxiliary physical_upper tensor and explicit
# provenance for the expert output policy used by a cache.
FEATURE_SCHEMA_VERSION = "fusion_inputs_20260920_v3"
SATELLITE_QUALITY_FIELDS = (
    "q", "scale_q", "clear_logit", "history_unique_frame_fraction",
    "freshness", "dynamics", "motion_confidence", "motion_available",
)
NWP_QUALITY_FIELDS = ("q", "scale_q", "clear_logit", "anchor_trust", "transport_residual")
BASE_CONTEXT_FIELDS = ("solar_context[8]", "satellite_q_minus_nwp_q")

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
    arrays = {key: [] for key in (
        "s", "n", "context", "motion_wind_consistency", "motion_wind_consistency_available",
        "history_transport", "history_transport_available", "prediction", "physical_upper",
        "available", "y", "valid", "day",
    )}
    meta: list[dict[str, object]] = []
    with torch.inference_mode():
        for raw in loader:
            batch = trainer.to_device(raw, device)
            result = model(batch)
            satellite, nwp = snapshots["s"], snapshots["n"]

            def feature(expert: dict, names: list[str]) -> torch.Tensor:
                return torch.cat(
                    [expert["state"], *[expert[name].float().unsqueeze(-1) for name in names]], dim=-1,
                )

            arrays["s"].append(feature(satellite, list(SATELLITE_QUALITY_FIELDS)).cpu())
            arrays["n"].append(feature(nwp, list(NWP_QUALITY_FIELDS)).cpu())
            arrays["context"].append(torch.cat([
                batch["solar_context"], (satellite["q"] - nwp["q"]).unsqueeze(-1),
            ], dim=-1).cpu())
            consistency = result["motion_wind_consistency"]
            consistency_available = result["motion_wind_consistency_available"].bool()
            if consistency.shape != (len(raw["sample_id"]), 6, 2):
                raise ValueError("motion_wind_consistency must have shape [case, 6, 2]")
            if consistency_available.shape != (len(raw["sample_id"]), 6):
                raise ValueError("motion_wind_consistency_available must have shape [case, 6]")
            if not torch.isfinite(consistency).all():
                raise FloatingPointError("non-finite motion_wind_consistency")
            if torch.any(consistency[~consistency_available] != 0):
                raise ValueError("unavailable motion_wind_consistency values must be zero")
            arrays["motion_wind_consistency"].append(consistency.cpu())
            arrays["motion_wind_consistency_available"].append(consistency_available.cpu())
            history_transport = torch.stack([
                satellite["history_transport_skill"], satellite["history_transport_overlap"],
                satellite["history_transport_dt_hours"], satellite["history_transport_pairs"],
            ], dim=-1)
            history_transport_available = satellite["history_transport_available"].bool()
            if history_transport.shape != (len(raw["sample_id"]), 6, 4):
                raise ValueError("history_transport must have shape [case, 6, 4]")
            if history_transport_available.shape != (len(raw["sample_id"]), 6):
                raise ValueError("history_transport_available must have shape [case, 6]")
            if not torch.isfinite(history_transport).all():
                raise FloatingPointError("non-finite history_transport")
            if torch.any(history_transport[~history_transport_available] != 0):
                raise ValueError("unavailable history_transport values must be zero")
            arrays["history_transport"].append(history_transport.cpu())
            arrays["history_transport_available"].append(history_transport_available.cpu())
            arrays["prediction"].append(torch.stack([
                result["satellite_prediction"], result["wrf_prediction"],
            ], dim=-1).cpu())
            arrays["physical_upper"].append(batch["physical_upper"].cpu())
            arrays["available"].append(torch.stack([
                satellite["available"], nwp["available"],
            ], dim=-1).bool().cpu())
            arrays["y"].append(batch["target"].cpu())
            arrays["valid"].append(batch["target_mask"].bool().cpu())
            arrays["day"].append((batch["solar_context"][..., 0] >= 0.0871557427).cpu())
            start = len(meta)
            for index, sample_id in enumerate(raw["sample_id"]):
                provenance = dict(dataset.provenance[start + index])
                if provenance["sample_id"] != sample_id:
                    raise ValueError("dataset provenance order differs from deterministic cache order")
                if provenance["station_id"] != raw["station_id"][index]:
                    raise ValueError("dataset provenance station does not match the batch")
                meta.append(provenance)
    for handle in handles:
        handle.remove()
    cache = {key: torch.cat(value) for key, value in arrays.items()}
    cache["meta"] = meta
    cache["schema_version"] = FEATURE_SCHEMA_VERSION
    cache["model_ablation"] = getattr(model, "ablation", "unknown")
    cache["physical_output_policy"] = (
        "unbounded direct-GHI/unbounded WRF-anchor residual; no final "
        "night/nonnegative/upper projection"
        if cache["model_ablation"] == "no_physics" else
        "sigmoid/physical-upper recovery with final night/nonnegative/upper projection"
    )
    cache["aod_enabled"] = cache["model_ablation"] not in {"no_aod", "no_conditional", "no_physics"}
    cache["prediction_source"] = (
        "frozen satellite_prediction and wrf_prediction component outputs; "
        "external dynamic MLP computes the final prediction"
    )
    cache["feature_schema"] = {
        "s": ["state[128]", *SATELLITE_QUALITY_FIELDS],
        "n": ["state[128]", *NWP_QUALITY_FIELDS],
        "context": list(BASE_CONTEXT_FIELDS),
        "optional_consistency": {
            "values_key": "motion_wind_consistency", "availability_key": "motion_wind_consistency_available",
            "values_shape": "case x 6 leads x 2 components", "availability_shape": "case x 6 leads",
            "matrix_convention": "case-level matrix; columns are east/north metres per latent x/y cell",
            "motion_unit": "latent x/y cells per hour", "wind_unit": "metres per second",
            "conversion": "matrix @ motion / 3600 before comparison",
        },
        "history_transport": {
            "values_key": "history_transport", "availability_key": "history_transport_available",
            "components": ["skill", "overlap", "dt_hours", "usable_pair_count"],
            "included_in_fusion_input": False,
            "reason": "input inclusion requires observed cross-case variation and provenance review",
        },
        "provenance_keys": sorted(meta[0]) if meta else [],
    }
    cache["y"] = torch.where(cache["valid"], cache["y"], torch.zeros_like(cache["y"]))
    if len(meta) != len({row["sample_id"] for row in meta}):
        raise ValueError("duplicate sample IDs in expert cache")
    for key in ("s", "n", "context", "prediction", "physical_upper"):
        if not torch.isfinite(cache[key]).all():
            raise FloatingPointError(f"non-finite cached tensor: {key}")
    for key in ("motion_wind_consistency", "history_transport"):
        if not torch.isfinite(cache[key]).all():
            raise FloatingPointError(f"non-finite cached tensor: {key}")
    torch.save(cache, output)
    consistency = cache["motion_wind_consistency"]
    consistency_available = cache["motion_wind_consistency_available"]
    zero_summary = {}
    for available in (False, True):
        values = consistency[consistency_available == available]
        zero_summary[str(available).lower()] = {
            "case_leads": int((consistency_available == available).sum()),
            "scalar_values": int(values.numel()),
            "zero_scalar_values": int((values == 0).sum()),
            "zero_fraction": float((values == 0).float().mean()) if values.numel() else None,
        }
    audit_path = output.with_name(output.stem + "_schema_audit.json")
    audit_path.write_text(json.dumps({
        "schema_version": FEATURE_SCHEMA_VERSION,
        "samples": len(meta),
        "cache_construction_schema_passed": True,
        "real_geometry_provenance": "fields captured; sample-level sidecar certification requires separate audit",
        "geometry_available_cases": sum(bool(row["latent_to_en_available"]) for row in meta),
        "geometry_unavailable_cases": sum(not bool(row["latent_to_en_available"]) for row in meta),
        "consistency_available_case_leads": int(consistency_available.sum()),
        "consistency_unavailable_case_leads": int((~consistency_available).sum()),
        "consistency_zero_summary_by_availability": zero_summary,
        "feature_schema": cache["feature_schema"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {row["cycle"] for row in meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--aod-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ablation", default="fixed_router",
        help="Expert ablation used for the frozen checkpoint; use no_physics for the key-component control.",
    )
    parser.add_argument("--satellite-path", choices=("direct", "dynamic", "dual"), default="dual")
    parser.add_argument("--routing-quality", choices=("legacy", "history_features", "history_risk"), default="legacy")
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
    checkpoint_ablation = checkpoint.get("ablation")
    if checkpoint_ablation is not None and checkpoint_ablation != args.ablation:
        raise ValueError(
            f"checkpoint ablation mismatch: checkpoint={checkpoint_ablation!r}, requested={args.ablation!r}"
        )
    model = ScenarioDrivenGHIModelV4(
        ablation=args.ablation, satellite_path=args.satellite_path,
        routing_quality=args.routing_quality,
    ).to(device)
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
        "ablation": args.ablation,
        "physical_output_policy": (
            "unbounded direct-GHI/unbounded WRF-anchor residual; no final "
            "night/nonnegative/upper projection"
            if args.ablation == "no_physics" else
            "sigmoid/physical-upper recovery with final night/nonnegative/upper projection"
        ),
        "aod_enabled": args.ablation not in {"no_aod", "no_conditional", "no_physics"},
        "prediction_source": (
            "frozen satellite_prediction and wrf_prediction component outputs; "
            "external dynamic MLP computes the final prediction"
        ),
    }
    (args.output_dir / "cache_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
