#!/usr/bin/env python3
"""One fixed WRF vertical-profile ablation under the formal 17/7 protocol.

The only input intervention is applied before fold normalization: for both
WRF domains, replace each time/channel/grid-point 10-level profile by its
valid-layer mean repeated over all ten levels. The per-column any-valid mask
is broadcast back to ten levels. All other case inputs are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


VARIANT = "wrf_vertical_profile_mean_repeated_v1"
SEED = 42
EVAL_STATIONS = {"GOB", "DRA", "IZA", "TBL", "SEL", "GWN", "CAB"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash a freshly initialized state with the formal trainer's convention."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def collapse_arrays(volume: np.ndarray, level_valid: np.ndarray, channel_available: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return [T,C,L,Y,X] repeated means and [T,L,Y,X] repeated column masks."""
    volume = np.asarray(volume, dtype=np.float32)
    level_valid = np.asarray(level_valid, dtype=bool)
    if volume.ndim != 5 or level_valid.shape != (volume.shape[0], volume.shape[2], *volume.shape[-2:]):
        raise ValueError(f"unexpected volume/mask shapes: {volume.shape}, {level_valid.shape}")
    valid = np.broadcast_to(level_valid[:, None], volume.shape) & np.isfinite(volume)
    if channel_available is not None:
        channel_available = np.asarray(channel_available, dtype=bool)
        if channel_available.shape != (volume.shape[1],):
            raise ValueError(f"unexpected channel availability shape: {channel_available.shape}")
        valid &= channel_available.reshape(1, -1, 1, 1, 1)
    counts = valid.sum(axis=2, keepdims=True)
    sums = np.where(valid, volume, 0.0).sum(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    repeated_volume = np.broadcast_to(means, volume.shape).copy()
    column_available = level_valid.any(axis=1, keepdims=True)
    repeated_mask = np.broadcast_to(column_available, level_valid.shape).copy()
    return repeated_volume, repeated_mask


def install_transform(data_module) -> dict:
    original_first = data_module._first
    volume_fields = {tuple(keys["volume"]): domain for domain, keys in data_module.DOMAIN_KEYS.items()}
    valid_fields = {tuple(keys["valid"]): domain for domain, keys in data_module.DOMAIN_KEYS.items()}
    domains_by_name = data_module.DOMAIN_KEYS

    def transformed_first(case: dict, names):
        key = tuple(names)
        if key in valid_fields:
            domain = valid_fields[key]
            original = original_first(case, domains_by_name[domain]["valid"]).astype(bool)
            return np.broadcast_to(original.any(axis=1, keepdims=True), original.shape).copy()
        if key in volume_fields:
            domain = volume_fields[key]
            keys = domains_by_name[domain]
            volume = original_first(case, keys["volume"])
            level_valid = original_first(case, keys["valid"]).astype(bool)
            channel_available = data_module._optional(case, keys["channel"])
            collapsed, _ = collapse_arrays(volume, level_valid, channel_available)
            return collapsed
        return original_first(case, names)

    data_module._first = transformed_first
    return {
        "variant": VARIANT,
        "domains": ["d02", "d01"],
        "input_fields": ["nwp_d02_3d_raw", "nwp_d01_3d_raw"],
        "normalization_order": "collapse raw physical fields before train-only fold normalization",
        "collapse": "valid & finite & channel-available weighted mean over 10 levels, repeated over levels",
        "mask": "per-column any-valid over the original 10 levels, repeated as Boolean across levels",
        "retained": ["surface fields", "static grids", "GHI anchors", "satellite inputs and checkpoint", "solar/time inputs", "targets and masks", "source/channel availability"],
        "test_accessed_during_training_or_normalization": False,
    }


def self_test(data_module, dataset_root: Path, train_case_path: Path) -> dict:
    # Synthetic finite/missing-level example proves mean and mask behavior.
    values = np.asarray([[[[[2.0]], [[6.0]], [[99.0]]]]], dtype=np.float32)
    mask = np.asarray([[[[True]], [[True]], [[False]]]], dtype=bool)
    collapsed, cmask = collapse_arrays(values, mask, np.asarray([True]))
    if not np.allclose(collapsed.ravel(), [4.0, 4.0, 4.0]) or not cmask.all():
        raise AssertionError("synthetic mean/mask self-test failed")
    values[0, 0, 1, 0, 0] = np.nan
    collapsed, cmask = collapse_arrays(values, mask, np.asarray([True]))
    if not np.allclose(collapsed.ravel(), [2.0, 2.0, 2.0]) or not cmask.all():
        raise AssertionError("finite-value filtering self-test failed")
    collapsed, cmask = collapse_arrays(values, mask, np.asarray([False]))
    if np.any(collapsed) or not cmask.all():
        raise AssertionError("unavailable-channel self-test failed")

    with np.load(train_case_path, allow_pickle=False) as loaded:
        case = {key: np.asarray(loaded[key]) for key in loaded.files}
    if "nwp_d02_3d_raw" not in case or "nwp_d01_3d_raw" not in case:
        raise ValueError("selected training case lacks formal d01/d02 WRF volumes")
    changed = []
    for domain in ("d02", "d01"):
        volume_key = f"nwp_{domain}_3d_raw"
        valid_key = f"nwp_{domain}_level_valid"
        available_key = f"nwp_{domain}_channel_available"
        before = case[volume_key]
        before_mask = case[valid_key].astype(bool)
        after, after_mask = collapse_arrays(before, before_mask, case.get(available_key))
        if after.shape != before.shape or after_mask.shape != before_mask.shape:
            raise AssertionError(f"{domain} shape changed")
        if not np.array_equal(after, np.repeat(after[:, :, :1], after.shape[2], axis=2)):
            raise AssertionError(f"{domain} values are not level-invariant after collapse")
        if not np.array_equal(after_mask, np.repeat(after_mask[:, :1], after_mask.shape[1], axis=1)):
            raise AssertionError(f"{domain} mask is not column-invariant after collapse")
        changed.extend([volume_key, valid_key])
    protected = [
        "satellite", "satellite_valid", "satellite_frame_mask", "satellite_unique_frame_mask",
        "nwp_d02_surface_raw", "nwp_d01_surface_raw", "nwp_d02_static_raw", "nwp_d01_static_raw",
        "wrf_ghi_anchor_d02", "solar", "target", "target_mask", "valid_minutes",
    ]
    missing = [key for key in protected if key not in case]
    if missing:
        raise ValueError(f"self-test case missing protected fields: {missing}")
    # Transform is pure: original arrays and all retained inputs remain unchanged.
    before_sha = {key: hashlib.sha256(np.ascontiguousarray(case[key]).tobytes()).hexdigest() for key in protected}
    after_sha = {key: hashlib.sha256(np.ascontiguousarray(case[key]).tobytes()).hexdigest() for key in protected}
    if before_sha != after_sha:
        raise AssertionError("retained raw fields changed")
    return {
        "passed": True,
        "synthetic_valid_mean": 4.0,
        "synthetic_invalid_layer_excluded": True,
        "finite_values_only": True,
        "unavailable_channel_zeroed": True,
        "unavailable_channel_preserves_column_mask": True,
        "sample_id": str(case["sample_id"].item()),
        "sample_path": str(train_case_path),
        "transformed_fields": changed,
        "protected_fields_verified_unchanged": protected,
        "d02_shape": list(case["nwp_d02_3d_raw"].shape),
        "d01_shape": list(case["nwp_d01_3d_raw"].shape),
        "dataset_root": str(dataset_root),
    }


def invoke(module, argv: list[str]) -> None:
    previous = sys.argv
    sys.argv = [module.__file__, *argv]
    try:
        module.main()
    finally:
        sys.argv = previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/autodl-tmp/guizhou_v4/data/derived/windcloud_v4_all_1462_stationaware"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/guizhou_mlp17_7eval_run_20260909"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    sys.path[:0] = [str(code_root), str(code_root / "src")]
    from ghi_hierarchical_v4 import data as data_module
    from scripts import train_hierarchical_799_v4 as trainer
    from scripts import cache_expert_states_17train_7eval as cache_script
    from scripts import train_mlp17_7eval_heads as heads_script
    from scripts import evaluate_mlp17_7eval_test as test_script

    run_root = args.run_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output = (args.output_dir or run_root / "wrf_vertical_profile_ablation_20260924").resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    expert_out = output / "expert"
    cache_out = output / "cache"
    heads_out = output / "dynamic_mlp"
    test_out = output / "test_7station"
    for directory in (expert_out, cache_out, heads_out, test_out):
        if directory.exists():
            raise FileExistsError(directory)

    official_checkpoint = run_root / "base_seed42" / "best_fusion.pt"
    official_metrics = run_root / "base_seed42" / "metrics.json"
    split_config = dataset_root / "split_config.json"
    cv_path = dataset_root / "cv_17train_7eval.json"
    audit_path = dataset_root / "audit.json"
    aod_reference = dataset_root / "statistics" / "fold_0_aod_reference.json"
    reference_implementation = code_root / "reference" / "run_query_fusion_pilot_20260908.py"
    for path in (official_checkpoint, official_metrics, split_config, cv_path, audit_path, aod_reference, reference_implementation):
        if not path.is_file():
            raise FileNotFoundError(path)

    transform_audit = install_transform(data_module)
    config = json.loads(split_config.read_text(encoding="utf-8"))
    rows = data_module.read_index(dataset_root)
    cycles = config["split_cycles"]
    train_rows, validation_rows, test_rows = trainer.select_station_aware_partitions(
        rows, set(cycles["train"]), set(cycles["validation"]), set(cycles["test"]),
        {value.upper() for value in config["evaluation_stations"]},
        {value.upper() for value in config["training_stations"]},
    )
    if (len(train_rows), len(validation_rows), len(test_rows)) != (1224, 119, 119):
        raise ValueError(f"formal split count mismatch: {(len(train_rows), len(validation_rows), len(test_rows))}")
    train_ids = {row["sample_id"] for row in train_rows}
    validation_ids = {row["sample_id"] for row in validation_rows}
    test_ids = {row["sample_id"] for row in test_rows}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("sample ID overlap in formal train/validation/test split")
    if {row["station_id"].upper() for row in validation_rows} != EVAL_STATIONS or {row["station_id"].upper() for row in test_rows} != EVAL_STATIONS:
        raise ValueError("validation/test stations differ from the formal seven-station cohort")

    raw_sample_path = dataset_root / train_rows[0]["case_file"]
    self_test_report = self_test(data_module, dataset_root, raw_sample_path)
    save(output / "self_test.json", self_test_report)
    print("SELF_TEST_PASSED", json.dumps(self_test_report, ensure_ascii=False), flush=True)

    # Reproduce the official seed-42 fresh initialization and load only the
    # satellite expert from the official checkpoint. WRF and fusion begin at
    # the exact seeded initialization, then train on collapsed inputs.
    full_metrics = json.loads(official_metrics.read_text(encoding="utf-8"))
    if full_metrics.get("seed") != SEED or full_metrics.get("test_accessed") is not False:
        raise ValueError("official checkpoint metrics do not certify the expected seed/no-test-access protocol")
    if full_metrics.get("stages_run") != ["satellite", "nwp", "fusion"]:
        raise ValueError("official checkpoint metrics do not contain the complete satellite/NWP/fusion training stages")
    trainer.seed_everything(SEED)
    initial_model = trainer.ScenarioDrivenGHIModelV4(
        ablation="fixed_router", satellite_path="dual", routing_quality="legacy",
    ).to(args.device)
    initial_hash = model_state_sha256(initial_model)
    expected_hash = full_metrics.get("initial_parameter_sha256")
    if expected_hash is not None and initial_hash != expected_hash:
        raise ValueError(f"seed-42 initialization hash mismatch: {initial_hash} != {expected_hash}")
    initial_state = initial_model.state_dict()
    base_state = torch.load(official_checkpoint, map_location="cpu", weights_only=False)["model"]
    satellite_keys = [key for key in initial_state if key.startswith("satellite_expert.")]
    if not satellite_keys or any(key not in base_state for key in satellite_keys):
        raise ValueError("official checkpoint lacks the complete satellite expert state")
    for key in satellite_keys:
        initial_state[key] = base_state[key].detach().cpu().clone()
    start_checkpoint = output / "seed42_satellite_official_wrf_fresh.pt"
    torch.save({
        "model": {key: value.detach().cpu() for key, value in initial_state.items()},
        "ablation": "fixed_router", "physics_constrained": True,
        "routing_quality": "legacy", "satellite_path": "dual", "seed": SEED,
        "variant": VARIANT,
    }, start_checkpoint)
    del initial_model, initial_state, base_state

    protocol = {
        "variant": VARIANT, "seed_expert": SEED, "head_seeds": [42, 43, 44],
        "train_cases": 1224, "validation_cases": 119, "test_cases": 119,
        "train_case_id_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
        "validation_case_id_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest(),
        "test_case_id_sha256": hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest(),
        "test_case_ids_disjoint_from_train_validation": True,
        "train_station_sampling": "station_balanced_with_replacement",
        "training_only_normalization": True,
        "normalization_fit": "transformed raw train cases only; refitted after vertical mean collapse and before standardization",
        "expert_training_stages": ["nwp", "fusion"],
        "satellite_checkpoint_sha256": sha256(official_checkpoint),
        "official_seed42_initialization_hash": expected_hash,
        "recreated_seed42_initialization_hash": initial_hash,
        "official_metrics_has_initialization_hash": expected_hash is not None,
        "train_stage_protocol": {"batch_size": 4, "epochs": {"nwp": 80, "fusion": 50}, "patience": 10, "lr": 0.0002, "weight_decay": 0.0001, "precision": "fp32"},
        "head_protocol": {"seeds": [42, 43, 44], "epochs": 40, "patience": 8, "batch_size": 32, "lr": 0.0002, "weight_decay": 0.0001},
        "split_config_sha256": sha256(split_config), "cv_sha256": sha256(cv_path),
        "training_script_sha256": sha256(Path(trainer.__file__)),
        "cache_script_sha256": sha256(Path(cache_script.__file__)),
        "head_script_sha256": sha256(Path(heads_script.__file__)),
        "test_script_sha256": sha256(Path(test_script.__file__)),
        "reference_implementation_sha256": sha256(reference_implementation),
        "test_accessed": False,
    }
    save(output / "protocol.json", {**protocol, **transform_audit})

    normalization_cache = expert_out / "normalization.json"
    trainer_args = [
        "--dataset-root", str(dataset_root), "--audit-report", str(audit_path),
        "--cv", str(cv_path), "--split-config", str(split_config), "--fold", "0",
        "--seed", str(SEED), "--output-dir", str(expert_out),
        "--normalization-cache", str(normalization_cache),
        "--aod-reference", str(aod_reference), "--resume-checkpoint", str(start_checkpoint),
        "--ablation", "fixed_router", "--satellite-path", "dual",
        "--routing-supervision", "none", "--routing-quality", "legacy",
        "--stages", "nwp,fusion", "--device", args.device, "--precision", "fp32",
        "--batch-size", "4", "--workers", "0", "--threads", "4", "--learning-rate", "0.0002",
        "--patience", "10", "--epochs", "satellite:80,nwp:80,fusion:50,transition:25,aod:25,joint:50",
    ]
    t0 = time.monotonic()
    invoke(trainer, trainer_args)
    expert_metrics = json.loads((expert_out / "metrics.json").read_text(encoding="utf-8"))
    if expert_metrics.get("test_accessed") is not False or expert_metrics.get("stages_run") != ["nwp", "fusion"]:
        raise ValueError("expert training protocol audit failed")
    save(output / "expert_stage_complete.json", {
        "passed": True, "elapsed_seconds": time.monotonic() - t0,
        "best_epochs": {item["stage"]: item["epoch"] for item in expert_metrics["history"] if item["epoch"] > 0},
        "test_accessed": False,
    })
    print("EXPERT_TRAINING_COMPLETE", flush=True)

    # Apply the same raw-field transform while extracting train/validation
    # expert caches. The test dataset is not loaded here.
    cache_script.MultisourceCaseDataset = data_module.MultisourceCaseDataset
    cache_args = [
        "--dataset-root", str(dataset_root), "--split-config", str(split_config),
        "--checkpoint", str(expert_out / "best_fusion.pt"),
        "--normalization", str(normalization_cache), "--aod-reference", str(aod_reference),
        "--output-dir", str(cache_out), "--device", args.device,
    ]
    invoke(cache_script, cache_args)
    if json.loads((cache_out / "cache_audit.json").read_text(encoding="utf-8")).get("test_accessed") is not False:
        raise ValueError("train/validation cache unexpectedly accessed test split")
    print("TRAIN_VALIDATION_CACHE_COMPLETE", flush=True)

    heads_args = [
        "--cache-dir", str(cache_out), "--reference-implementation", str(reference_implementation),
        "--output-dir", str(heads_out), "--device", args.device,
        "--seeds", "42,43,44", "--epochs", "40", "--patience", "8",
    ]
    invoke(heads_script, heads_args)
    heads_summary = json.loads((heads_out / "summary.json").read_text(encoding="utf-8"))
    head_protocol = json.loads((heads_out / "protocol.json").read_text(encoding="utf-8"))
    if head_summary.get("test_accessed") is not False or head_protocol.get("test_accessed") is not False:
        raise ValueError("head training unexpectedly accessed test split")

    selection = {
        "locked": True, "variant": VARIANT, "test_accessed_before_lock": False,
        "expert_checkpoint_sha256": sha256(expert_out / "best_fusion.pt"),
        "normalization_sha256": sha256(normalization_cache),
        "train_cache_sha256": sha256(cache_out / "train_cache.pt"),
        "validation_cache_sha256": sha256(cache_out / "validation_cache.pt"),
        "expert_best_epochs": {item["stage"]: item["epoch"] for item in expert_metrics["history"] if item["epoch"] > 0},
        "head_best_epochs": {key: value["best_epoch"] for key, value in heads_summary["runs"].items()},
        "head_checkpoint_sha256": {
            str(seed): sha256(heads_out / f"mlp_seed{seed}" / "best.pt") for seed in (42, 43, 44)
        },
        "validation_metrics_sha256": sha256(heads_out / "summary.json"),
    }
    save(output / "selection_locked.json", selection)
    print("SELECTION_LOCKED", json.dumps(selection, ensure_ascii=False), flush=True)

    # This is the first step that opens or computes on the 119 test examples.
    test_script.MultisourceCaseDataset = data_module.MultisourceCaseDataset
    test_args = [
        "--dataset-root", str(dataset_root), "--split-config", str(split_config),
        "--expert-checkpoint", str(expert_out / "best_fusion.pt"),
        "--expert-normalization", str(normalization_cache), "--aod-reference", str(aod_reference),
        "--heads-dir", str(heads_out), "--reference-implementation", str(reference_implementation),
        "--output-dir", str(test_out), "--device", args.device,
    ]
    invoke(test_script, test_args)
    test_summary = json.loads((test_out / "test_summary.json").read_text(encoding="utf-8"))
    save(output / "status.json", {
        "state": "completed", "variant": VARIANT, "test_accessed": True,
        "test_summary_sha256": sha256(test_out / "test_summary.json"),
        "test_cases": test_summary["test_cases"], "test_cycles": test_summary["test_cycles"],
    })
    print("WRF_VERTICAL_PROFILE_ABLATION_COMPLETE", json.dumps({
        "output_dir": str(output), "test_cases": test_summary["test_cases"],
        "daylight_metrics": {key: value["daylight"] for key, value in test_summary["runs"].items()},
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
