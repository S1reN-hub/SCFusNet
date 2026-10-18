#!/usr/bin/env python3
"""Download verified Zenodo transport chunks and reconstruct four input ZIPs.

Python standard library only. Already correct chunks and archives are reused.
To reuse files obtained with Zenodo's download-all feature, extract that download
and pass its chunk directory with --local-chunk-dir. Use --offline to forbid
network access. This restores archives; it does not extract or train on inputs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/recovered_inputs_20261018/archive_chunks.json"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
BLOCK = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def filename(value):
    require(isinstance(value, str) and NAME.fullmatch(value) and ".." not in value,
            f"Unsafe filename: {value!r}")
    return value


def confined(root, name):
    filename(name)
    target = root / name
    require(not target.is_symlink(), f"Refusing symbolic link: {name}")
    require(target.resolve().is_relative_to(root.resolve()), f"Path escapes output directory: {name}")
    return target


def file_matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest() == entry["sha256"]


def load_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    require(str(data.get("record")) == "23445294", "Unexpected Zenodo record")
    require(data.get("chunk_size") == 67108864, "Expected 64 MiB transport chunks")
    archives = data.get("archives")
    require(isinstance(archives, list) and len(archives) == 4, "Expected four archives")
    names = set()
    for archive in archives:
        for entry in [archive, *archive.get("chunks", [])]:
            name = filename(entry["name"])
            require(name not in names, f"Duplicate filename: {name}")
            names.add(name)
            require(type(entry["bytes"]) is int and entry["bytes"] > 0, f"Invalid byte count: {name}")
            require(isinstance(entry["sha256"], str) and DIGEST.fullmatch(entry["sha256"]),
                    f"Invalid SHA256: {name}")
        require(archive["name"].lower().endswith(".zip"), "Archive filename must end in .zip")
        chunks = archive.get("chunks")
        require(isinstance(chunks, list) and chunks, "Archive has no chunks")
        require(sum(c["bytes"] for c in chunks) == archive["bytes"], "Chunk/archive sizes differ")
        require(all(c["bytes"] == data["chunk_size"] for c in chunks[:-1])
                and chunks[-1]["bytes"] <= data["chunk_size"], "Invalid chunk sizes/order")
    return data


def reject_error_body(response, first):
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    require(not content_type.startswith("text/") and content_type not in
            {"application/json", "application/problem+json", "application/xhtml+xml"},
            f"Server returned an error/document content type: {content_type}")
    prefix = first[:512].lstrip().lower()
    require(not prefix.startswith((b"<!doctype html", b"<html", b"<?xml", b'{"error', b'{"message')),
            "Server returned HTML/XML/JSON instead of chunk bytes")


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        require(parsed.scheme == "https" and parsed.hostname in {"zenodo.org", "www.zenodo.org"}
                and parsed.port in (None, 443) and parsed.username is None,
                "Refusing redirect outside HTTPS Zenodo")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_chunk(entry, cache, local, record, retries, timeout, offline):
    name = entry["name"]
    target = confined(cache, name)
    if file_matches(target, entry):
        return name, target, "cached"
    if local is not None:
        supplied = confined(local, name)
        if supplied.exists():
            require(file_matches(supplied, entry), f"Local chunk checksum/size mismatch: {name}")
            return name, supplied, "local"
    require(not offline, f"Verified chunk unavailable in offline mode: {name}")
    partial = confined(cache, name + ".partial")
    url = f"https://zenodo.org/records/{record}/files/{urllib.parse.quote(name, safe='')}?download=1"
    opener = urllib.request.build_opener(SafeRedirect())
    last_error = None
    for attempt in range(retries):
        try:
            if partial.exists() and partial.stat().st_size >= entry["bytes"]:
                if file_matches(partial, entry):
                    os.replace(partial, target)
                    return name, target, "downloaded"
                partial.unlink()
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "SCFusNet-input-restorer/1.0", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                status = response.getcode()
                require(status in (200, 206), f"Unexpected HTTP status: {status}")
                if status == 206:
                    value = response.headers.get("Content-Range", "")
                    require(value == f"bytes {offset}-{entry['bytes'] - 1}/{entry['bytes']}",
                            "Invalid partial response range")
                else:
                    offset = 0
                remaining = entry["bytes"] - offset
                length = response.headers.get("Content-Length")
                if length is not None:
                    require(int(length) == remaining, "Response length differs from manifest")
                first = response.read(min(BLOCK, remaining + 1))
                reject_error_body(response, first)
                with partial.open("ab" if offset else "wb") as stream:
                    written = offset
                    block = first
                    while block:
                        written += len(block)
                        require(written <= entry["bytes"], "Response exceeds expected chunk size")
                        stream.write(block)
                        block = response.read(min(BLOCK, entry["bytes"] - written + 1))
            require(file_matches(partial, entry), "Downloaded chunk checksum/size mismatch")
            os.replace(partial, target)
            return name, target, "downloaded"
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
            last_error = error
            # A complete but incorrect body cannot be resumed safely.
            if partial.exists() and partial.stat().st_size >= entry["bytes"]:
                partial.unlink()
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Could not retrieve verified chunk {name} after {retries} attempts: {last_error}")


def restore(args):
    data = load_manifest(args.manifest)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    require(not cache.is_symlink(), "Refusing symbolic-link cache directory")
    cache.mkdir(exist_ok=True)
    local = args.local_chunk_dir.resolve() if args.local_chunk_dir else None
    if local is not None:
        require(local.is_dir(), "Local chunk directory does not exist")
    archives = data["archives"]
    pending = [a for a in archives if not file_matches(confined(output, a["name"]), a)]
    chunks = [c for a in pending for c in a["chunks"]]
    available = {}
    counts = {"cached": 0, "local": 0, "downloaded": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_chunk, c, cache, local, data["record"],
                               args.retries, args.timeout, args.offline) for c in chunks]
        for future in concurrent.futures.as_completed(futures):
            name, path, mode = future.result()
            available[name] = path
            counts[mode] += 1
            print(f"Verified chunks: {sum(counts.values())}/{len(chunks)} ({name}, {mode})", flush=True)
    for archive in pending:
        target = confined(output, archive["name"])
        temporary = confined(output, archive["name"] + ".assembling")
        with temporary.open("wb") as writer:
            for chunk in archive["chunks"]:
                with available[chunk["name"]].open("rb") as reader:
                    shutil.copyfileobj(reader, writer, BLOCK)
        require(file_matches(temporary, archive), f"Restored archive checksum mismatch: {archive['name']}")
        os.replace(temporary, target)
    result = {"zenodo_record": str(data["record"]), "archives_verified": len(archives),
              "chunks": counts, "archives": [{k: a[k] for k in ("name", "bytes", "sha256")} for a in archives],
              "note": "Verified transport reconstruction; historical raw training-payload identity is not established."}
    report = confined(output, "restore_verification.json")
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("restored_inputs"))
    parser.add_argument("--local-chunk-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be between 1 and 12")
    if not 1 <= args.retries <= 12 or not 0 < args.timeout <= 300:
        parser.error("--retries must be 1..12; --timeout must be positive and <=300 seconds")
    restore(args)


if __name__ == "__main__":
    main()
