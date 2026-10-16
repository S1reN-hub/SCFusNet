#!/usr/bin/env python3
"""Resume cache/head/test postprocessing from a completed WRF profile run."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    dataset_root = args.dataset_root.resolve()
    expert_run = args.expert_run.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite postprocessing output: {output}")

    sys.path[:0] = [str(code_root), str(code_root / "src")]
    from ghi_hierarchical_v4 import data as data_module
    from scripts import cache_expert_states_17train_7eval as cache_script
    from scripts import evaluate_mlp17_7eval_test as test_script
    from scripts import train_hierarchical_799_v4 as trainer
    from scripts import train_mlp17_7eval_heads as heads_script

    runner = runpy.run_path(str(Path(__file__).with_name("run_wrf_vertical_profile_ablation_20260924.py")))
    install_transform = runner["install_transform"]
    invoke = runner["invoke"]
    save = runner["save"]
    sha256 = runner["sha256"]
    transform_audit = install_transform(data_module)

    expert_out = expert_run / "expert"
    checkpoint = expert_out / "best_fusion.pt"
    normalization = expert_out / "normalization.json"
    expert_metrics_path = expert_out / "metrics.json"
    split_config = dataset_root / "split_config.json"
    cv_path = dataset_root / "cv_17train_7eval.json"
    aod_reference = dataset_root / "statistics" / "fold_0_aod_reference.json"
    reference_implementation = code_root / "reference" / "run_query_fusion_pilot_20260908.py"
    for path in (checkpoint, normalization, expert_metrics_path, split_config, cv_path, aod_reference, reference_implementation):
        if not path.is_file():
            raise FileNotFoundError(path)

    expert_metrics = json.loads(expert_metrics_path.read_text(encoding="utf-8"))
    if expert_metrics.get("test_accessed") is not False or expert_metrics.get("stages_run") != ["nwp", "fusion"]:
        raise ValueError("completed expert output is not certified as NWP+fusion train/validation only")

    config = json.loads(split_config.read_text(encoding="utf-8"))
    rows = data_module.read_index(dataset_root)
    cycles = config["split_cycles"]
    train_rows, validation_rows, test_rows = trainer.select_station_aware_partitions(
        rows, set(cycles["train"]), set(cycles["validation"]), set(cycles["test"]),
        {value.upper() for value in config["evaluation_stations"]},
        {value.upper() for value in config["training_stations"]},
    )
    if (len(train_rows), len(validation_rows), len(test_rows)) != (1224, 119, 119):
        raise ValueError("formal 17/7 split counts differ from 1224/119/119")
    train_ids = {row["sample_id"] for row in train_rows}
    validation_ids = {row["sample_id"] for row in validation_rows}
    test_ids = {row["sample_id"] for row in test_rows}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("formal train/validation/test sample IDs overlap")

    output.mkdir(parents=True)
    cache_out = output / "cache"
    heads_out = output / "dynamic_mlp"
    test_out = output / "test_7station"
    protocol = {
        "variant": "wrf_vertical_profile_mean_repeated_v1",
        "postprocess_only": True,
        "expert_run": str(expert_run),
        "train_cases": 1224,
        "validation_cases": 119,
        "test_cases": 119,
        "split_config_sha256": sha256(split_config),
        "cv_sha256": sha256(cv_path),
        "expert_checkpoint_sha256": sha256(checkpoint),
        "expert_normalization_sha256": sha256(normalization),
        "expert_metrics_sha256": sha256(expert_metrics_path),
        "reference_implementation_sha256": sha256(reference_implementation),
        "test_accessed_before_selection_lock": False,
        **transform_audit,
    }
    save(output / "protocol.json", protocol)

    cache_script.MultisourceCaseDataset = data_module.MultisourceCaseDataset
    cache_args = [
        "--dataset-root", str(dataset_root), "--split-config", str(split_config),
        "--checkpoint", str(checkpoint), "--normalization", str(normalization),
        "--aod-reference", str(aod_reference), "--output-dir", str(cache_out),
        "--device", args.device,
    ]
    invoke(cache_script, cache_args)
    cache_audit = json.loads((cache_out / "cache_audit.json").read_text(encoding="utf-8"))
    if cache_audit.get("test_accessed") is not False or (cache_audit.get("train_cases"), cache_audit.get("validation_cases")) != (1224, 119):
        raise ValueError("train/validation cache audit failed")

    heads_args = [
        "--cache-dir", str(cache_out), "--reference-implementation", str(reference_implementation),
        "--output-dir", str(heads_out), "--device", args.device,
        "--seeds", "42,43,44", "--epochs", "40", "--patience", "8",
    ]
    invoke(heads_script, heads_args)
    heads_summary = json.loads((heads_out / "summary.json").read_text(encoding="utf-8"))
    head_protocol = json.loads((heads_out / "protocol.json").read_text(encoding="utf-8"))
    if heads_summary.get("test_accessed") is not False or head_protocol.get("test_accessed") is not False:
        raise ValueError("MLP head training unexpectedly accessed test split")

    selection = {
        "locked": True,
        "variant": "wrf_vertical_profile_mean_repeated_v1",
        "test_accessed_before_lock": False,
        "expert_checkpoint_sha256": sha256(checkpoint),
        "normalization_sha256": sha256(normalization),
        "train_cache_sha256": sha256(cache_out / "train_cache.pt"),
        "validation_cache_sha256": sha256(cache_out / "validation_cache.pt"),
        "head_best_epochs": {key: value["best_epoch"] for key, value in heads_summary["runs"].items()},
        "head_checkpoint_sha256": {
            str(seed): sha256(heads_out / f"mlp_seed{seed}" / "best.pt") for seed in (42, 43, 44)
        },
        "validation_metrics_sha256": sha256(heads_out / "summary.json"),
    }
    save(output / "selection_locked.json", selection)

    test_script.MultisourceCaseDataset = data_module.MultisourceCaseDataset
    test_args = [
        "--dataset-root", str(dataset_root), "--split-config", str(split_config),
        "--expert-checkpoint", str(checkpoint), "--expert-normalization", str(normalization),
        "--aod-reference", str(aod_reference), "--heads-dir", str(heads_out),
        "--reference-implementation", str(reference_implementation),
        "--output-dir", str(test_out), "--device", args.device,
    ]
    invoke(test_script, test_args)
    test_summary = json.loads((test_out / "test_summary.json").read_text(encoding="utf-8"))
    if test_summary.get("test_accessed") is not True or test_summary.get("test_cases") != 119:
        raise ValueError("one-time test summary failed audit")
    save(output / "status.json", {
        "state": "completed", "variant": "wrf_vertical_profile_mean_repeated_v1",
        "test_accessed": True, "test_cases": test_summary["test_cases"],
        "test_cycles": test_summary["test_cycles"],
        "test_summary_sha256": sha256(test_out / "test_summary.json"),
    })
    print(json.dumps({
        "postprocess_output": str(output),
        "cache_audit": cache_audit,
        "head_best_epochs": selection["head_best_epochs"],
        "test_cases": test_summary["test_cases"],
        "test_cycles": test_summary["test_cycles"],
        "test_metrics": {key: value["daylight"] for key, value in test_summary["runs"].items()},
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
