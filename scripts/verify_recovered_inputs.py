#!/usr/bin/env python3
"""Verify the recovered 1462-case input snapshot without training any model.

Full verification checks published bytes/SHA256, case identities, and NPZ array
headers. When present, recovered_source_* columns separately record the input
snapshot before public metadata redaction. Neither set of hashes establishes
historical identity because original training-time per-case hashes are absent.
Only NumPy is required for full verification; --metadata-only uses stdlib.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
import re
import struct
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data/recovered_inputs_20261018"
PROVENANCE_SHA256 = "957c605e4f1bd1d39f469581b9ebe505d18b0e75909d4a300bc5003d4ba05cfe"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def safe_path(root, relative):
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in relative,
            f"Unsafe relative path: {relative}")
    target = (root / Path(*path.parts)).resolve()
    require(target.is_relative_to(root.resolve()), f"Path escapes dataset root: {relative}")
    return target


def npy_header(stream):
    """Read shape/dtype without decompressing the large forecast arrays."""
    require(stream.read(6) == b"\x93NUMPY", "Invalid NPY magic")
    version = tuple(stream.read(2))
    require(version in ((1, 0), (2, 0), (3, 0)), f"Unsupported NPY version {version}")
    width = 2 if version == (1, 0) else 4
    raw_length = stream.read(width)
    require(len(raw_length) == width, "Truncated NPY header length")
    length = struct.unpack("<H" if width == 2 else "<I", raw_length)[0]
    require(length <= 1024 * 1024, "NPY header exceeds 1 MiB")
    encoded = stream.read(length)
    require(len(encoded) == length, "Truncated NPY header")
    header = ast.literal_eval(encoded.decode("utf-8" if version == (3, 0) else "latin1"))
    require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"},
            "Invalid NPY header fields")
    return header


def verify_case(path, row, schema, np):
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        require(len(members) == len(set(members)), f"Duplicate NPZ members: {row['sample_id']}")
        require(all(name.endswith(".npy") and "/" not in name and "\\" not in name for name in members),
                f"Unexpected NPZ member path: {row['sample_id']}")
        names = {name[:-4] for name in members}
        required = set(schema["required_shapes"]) | set(schema["required_scalars"])
        require(required <= names, f"Missing NPZ fields in {row['sample_id']}: {sorted(required - names)}")
        for name in names:
            with archive.open(name + ".npy") as stream:
                header = npy_header(stream)
            require(not np.dtype(header["descr"]).hasobject, f"Object/pickle dtype: {row['sample_id']}:{name}")
            if name in schema["required_shapes"]:
                require(list(header["shape"]) == schema["required_shapes"][name],
                        f"Shape mismatch: {row['sample_id']}:{name}: {header['shape']}")
            if name in schema["required_scalars"]:
                require(header["shape"] == (), f"Expected scalar: {row['sample_id']}:{name}")
        contracts = schema["scientific_contracts"]
        expected = {"sample_id": row["sample_id"], "station_id": row["station_id"],
                    "wrf_ghi_anchor_source": contracts["wrf_ghi_anchor_source"],
                    "cape_cin_algorithm": contracts["cape_cin_algorithm"]}
        for name, value in expected.items():
            with archive.open(name + ".npy") as stream:
                actual = str(np.load(stream, allow_pickle=False).item())
            require(actual == value, f"Identity/contract mismatch: {row['sample_id']}:{name}")


def verify_metadata(directory):
    provenance_path = directory / "provenance.json"
    require(sha256(provenance_path) == PROVENANCE_SHA256, "Provenance digest mismatch")
    provenance = load_json(provenance_path)
    for name, expected in provenance["metadata_files"].items():
        path = safe_path(directory, name)
        require(path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"],
                f"Metadata digest/size mismatch: {name}")
    index = read_csv(directory / "index.csv")
    manifest = read_csv(directory / "manifest.csv")
    schema = load_json(directory / "schema.json")
    split = load_json(directory / "split_config.json")
    require(len(index) == len(manifest) == schema["cases"] == 1462, "Expected exactly 1462 cases")
    lookup = {row["sample_id"]: row for row in manifest}
    require(len(lookup) == 1462 and len({r["sample_id"] for r in index}) == 1462, "Duplicate case IDs")
    require({r["sample_id"] for r in index} == set(lookup), "Index/manifest case mapping differs")
    require(len({r["station_id"] for r in index}) == schema["stations"], "Station count differs")
    require(len({r["cycle"] for r in index}) == schema["cycles"], "Cycle count differs")
    require(dict(collections.Counter(r["experiment_split"] for r in index)) == schema["split_counts"],
            "Experiment split counts differ")
    require(dict(collections.Counter(r["source_dataset"] for r in index)) == schema["source_counts"],
            "Source dataset counts differ")
    require(sum(int(r["bytes"]) for r in manifest) == provenance["total_payload_bytes"], "Payload byte total differs")
    source_columns = {"recovered_source_bytes", "recovered_source_sha256"}
    has_source = bool(source_columns & set(manifest[0]))
    if has_source:
        require(source_columns <= set(manifest[0]), "Incomplete recovered-source digest columns")
        require(sum(int(r["recovered_source_bytes"]) for r in manifest) == provenance["source_snapshot_payload_bytes"],
                "Recovered-source byte total differs")
    evaluation = set(split["evaluation_stations"])
    for row in index:
        sample = row["sample_id"]
        require(re.fullmatch(r"[A-Za-z0-9_]+", sample) is not None, f"Unsafe sample ID {sample}")
        require(row["case_file"] == f"cases/{sample}.npz", f"Noncanonical case path: {sample}")
        matched = lookup[sample]
        require(all(row[k] == matched[k] for k in ("station_id", "cycle", "source_dataset", "case_file")),
                f"Index/manifest fields differ: {sample}")
        require(re.fullmatch(r"[0-9a-f]{64}", matched["sha256"]) is not None and int(matched["bytes"]) > 0,
                f"Invalid digest/size: {sample}")
        if has_source:
            require(re.fullmatch(r"[0-9a-f]{64}", matched["recovered_source_sha256"]) is not None
                    and int(matched["recovered_source_bytes"]) > 0,
                    f"Invalid recovered-source digest/size: {sample}")
        assigned = row["experiment_split"]
        if row["station_id"] in evaluation:
            require(row["cycle"] in split["split_cycles"][assigned], f"Evaluation split/cycle differs: {sample}")
        else:
            require(assigned == "train", f"Non-evaluation station outside training: {sample}")
    return manifest, schema, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Unpacked payload root containing cases/")
    parser.add_argument("--metadata-dir", type=Path, default=METADATA)
    parser.add_argument("--metadata-only", action="store_true", help="Verify released mappings/checksums metadata without opening NPZs")
    parser.add_argument("--output", type=Path, help="Optional JSON verification report")
    args = parser.parse_args()
    if not args.metadata_only and args.data_root is None:
        parser.error("--data-root is required unless --metadata-only is used")
    manifest, schema, provenance = verify_metadata(args.metadata_dir)
    checked = 0
    if not args.metadata_only:
        import numpy as np
        root = args.data_root.resolve()
        actual = {p.relative_to(root).as_posix() for p in (root / "cases").glob("*.npz")}
        require(actual == {r["case_file"] for r in manifest}, "Payload case file set differs from manifest")
        for row in manifest:
            path = safe_path(root, row["case_file"])
            require(path.stat().st_size == int(row["bytes"]), f"Payload size mismatch: {row['sample_id']}")
            require(sha256(path) == row["sha256"], f"Payload SHA256 mismatch: {row['sample_id']}")
            verify_case(path, row, schema, np)
            checked += 1
    result = {"metadata_verified": True, "mapped_cases": len(manifest), "payload_cases_verified": checked,
              "payload_checks": "size, SHA256, NPZ shape/dtype, scalar identity and scientific contracts" if checked else "not performed (--metadata-only)",
              "snapshot_payload_bytes": provenance["total_payload_bytes"],
              "recovered_source_payload_bytes": provenance.get("source_snapshot_payload_bytes", provenance["total_payload_bytes"]),
              "historical_raw_payload_identity_verified": False,
              "identity_limit": schema["identity_limit"], "training_performed": False}
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
