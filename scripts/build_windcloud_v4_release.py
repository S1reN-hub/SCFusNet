#!/usr/bin/env python3
"""Build leakage-safe development and confirmation views of fused v4 cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


FIELDS = (
    "sample_id", "cycle", "station_id", "role", "partition",
    "source_dataset", "case_file",
)


def read_index(root: Path) -> list[dict[str, str]]:
    with (root / "index.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty release root: {path}")
    (path / "cases").mkdir(parents=True, exist_ok=True)


def write_release(
    output: Path,
    rows: list[tuple[Path, dict[str, str], str, str]],
    release_role: str,
) -> None:
    prepare_root(output)
    index_rows: list[dict[str, str]] = []
    for source_root, row, source_name, role in rows:
        source = (source_root / row["case_file"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "cases" / f"{row['sample_id']}.npz"
        relative_target = os.path.relpath(source, destination.parent)
        destination.symlink_to(relative_target)
        index_rows.append({
            "sample_id": row["sample_id"],
            "cycle": row["cycle"],
            "station_id": row["station_id"],
            "role": role,
            "partition": row.get("partition", ""),
            "source_dataset": source_name,
            "case_file": destination.relative_to(output).as_posix(),
        })
    index_rows.sort(key=lambda item: (item["cycle"], item["station_id"], item["sample_id"]))
    with (output / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)
    summary = {
        "schema_version": "windcloud_v4_release_view_v1",
        "role": release_role,
        "case_count": len(index_rows),
        "cycle_count": len({row["cycle"] for row in index_rows}),
        "station_counts": dict(sorted(Counter(row["station_id"] for row in index_rows).items())),
        "source_counts": dict(sorted(Counter(row["source_dataset"] for row in index_rows).items())),
        "legacy_partition_counts": dict(sorted(Counter(row["partition"] for row in index_rows).items())),
        "storage": "relative symbolic links to immutable fused case files",
    }
    (output / "release.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--expansion-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    args = parser.parse_args()

    core = read_index(args.core_root)
    expansion = read_index(args.expansion_root)
    core_cycles = {row["cycle"] for row in core}
    expansion_cycles = {row["cycle"] for row in expansion}
    overlap = core_cycles & expansion_cycles
    if overlap:
        raise ValueError(f"core and expansion cycle overlap: {sorted(overlap)}")
    sample_ids = [row["sample_id"] for row in core + expansion]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs across source datasets")

    development: list[tuple[Path, dict[str, str], str, str]] = [
        (args.core_root, row, "core799", "development") for row in core
    ]
    confirmation: list[tuple[Path, dict[str, str], str, str]] = []
    for row in expansion:
        role = str(row.get("partition", "")).strip().lower()
        item = (args.expansion_root, row, "p46_expansion663", role)
        if role == "development":
            development.append(item)
        elif role == "confirmation":
            confirmation.append(item)
        else:
            raise ValueError(f"unexpected expansion role for {row['sample_id']}: {role!r}")

    if len(development) != 1241 or len(confirmation) != 221:
        raise ValueError(
            f"expected 1241 development and 221 confirmation cases, got "
            f"{len(development)} and {len(confirmation)}"
        )
    write_release(args.development_root, development, "development")
    write_release(args.confirmation_root, confirmation, "confirmation")


if __name__ == "__main__":
    main()
