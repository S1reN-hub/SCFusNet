#!/usr/bin/env python3
"""Run released frozen expert-state caches through the original fusion heads.

Requires PyTorch and NumPy. This verifies fusion inference, not regeneration of
satellite/WRF expert states from raw inputs. Only load trusted published .pt files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Released test_cache.pt")
    parser.add_argument("--heads-dir", type=Path, required=True, help="Directory containing mlp_seed{42,43,44}/best.pt")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Root containing original scripts and src")
    parser.add_argument("--comparison", type=Path, default=ROOT / "results/fulltest_model_comparison_20260921/source_data_common_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("reproduction/cached_inference"))
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    sys.path[:0] = [str(args.source_root), str(args.source_root / "src")]
    torch.set_num_threads(4)
    reference = load_module(args.source_root / "scripts/run_query_fusion_pilot_20260908.py", "released_dynamic_fusion")
    implementation = load_module(args.source_root / "scripts/run_aod_7eval_20260912.py", "released_aod_prediction")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    with args.comparison.open(encoding="utf-8-sig", newline="") as stream:
        expected = list(csv.DictReader(stream))
    lookup = {str(meta["sample_id"]): i for i, meta in enumerate(cache["meta"])}
    assert len(lookup) == len(cache["meta"]), "Duplicate cache sample IDs"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"scope": "frozen expert cache to dynamic MLP fusion; 601 common rows",
              "cache_sha256": sha256(args.cache), "seeds": {}}
    for seed in (42, 43, 44):
        path = args.heads_dir / f"mlp_seed{seed}" / "best.pt"
        head = torch.load(path, map_location="cpu", weights_only=False)
        prediction = implementation.predict(cache, head, reference).cpu()
        errors, rows = [], []
        for row in expected:
            i, h = lookup[row["sample_id"]], int(row["lead"]) - 1
            assert bool(cache["valid"][i, h] & cache["day"][i, h])
            assert abs(float(cache["y"][i, h]) - float(row["target"])) < 1e-4
            value = float(prediction[i, h])
            delta = abs(value - float(row[f"solfus_seed{seed}"]))
            errors.append(delta)
            rows.append({"sample_id": row["sample_id"], "lead": h + 1,
                         "prediction": value, "absolute_difference": delta})
        assert len(errors) == 601 and max(errors) <= args.tolerance, (seed, max(errors))
        with (args.output_dir / f"seed{seed}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result["seeds"][str(seed)] = {"rows": len(errors), "max_absolute_difference": max(errors),
                                      "checkpoint_sha256": sha256(path)}
    result["assertions_passed"] = True
    encoded = json.dumps(result, indent=2, allow_nan=False)
    (args.output_dir / "verification.json").write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
