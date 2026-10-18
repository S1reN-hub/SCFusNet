# Recovered input data: provenance and release status

Updated **2026-10-18**. This document describes the packaged case-level input supplement for [GitHub release `inputs-20261018`](https://github.com/S1reN-hub/SCFusNet/releases/tag/inputs-20261018), whose page becomes public when the draft release is published. The [original v1 archive](https://doi.org/10.5281/zenodo.23418519) remains preserved. **Version DOI: `10.5281/zenodo.23445294` (registration becomes effective on publication).** Treat it as reserved until its public landing page is accessible; listing it here is not publication confirmation. The [Zenodo concept DOI](https://doi.org/10.5281/zenodo.23418518) identifies the release family.

## Located inputs

| Source dataset | Cases present | Source NPZ payload bytes | Index identity |
| --- | ---: | ---: | --- |
| `windcloud_v4_development_1241` | 1,241 / 1,241 | 4,514,891,544 | Matches the formal preparation log |
| `windcloud_v4_confirmation_221` | 221 / 221 | 792,703,160 | Matches the formal preparation log |
| Total | **1,462 / 1,462** | **5,307,594,704** | Both indices match |

The files cover 17 stations and 86 cycles. Canonical case keys match the published case assignments with no missing or extra cases, preserving the **1,224/119/119** train/validation/test split and 1,462 unique physical payload files.

The totals above describe the source snapshot before publication-path sanitization. All 1,462 source files have now been downloaded and their source checksums verified. The sanitized publication NPZ files total **5,305,765,926 bytes**. The four outer input ZIPs total **5,305,991,774 bytes**; the 82 transport chunks reconstruct exactly those archive bytes.

Historical source-index SHA-256 values:

```text
development:  749940fca701051c7602a77b53d28e56dcb099578f764e2e1e803a9ef92ae965
confirmation: 6a169c57eb6e9a69b0b563bc1ca2afe1fe2904a261d8fd5c6c14f13cf64b1133
```

A newly completed per-file SHA-256 manifest enumerates all 1,462 current NPZ payloads. Source metadata and source-level AOD-reference files were retrieved with hash checks. Sampled NPZ headers expose 54 fields, including satellite arrays shaped `[6, 7, 32, 32]` and target arrays shaped `[6]`; sampled headers alone do not validate every field of every case.

## Historical identity boundary

Exact source-index hashes establish identity of the index bytes and the recorded case mapping. They do **not** hash the NPZ payloads referenced by those indices. No historical per-case NPZ digest or equivalent raw-tensor fingerprint was found in the inspected records.

The supplement must therefore be described as a **source-index-matched current recovered snapshot**. Its new checksums can establish transfer integrity and provide a stable baseline for future work. They cannot prove that every payload is byte-for-byte identical to the original formal-training inputs.

## Packaging and verification

The source datasets reference case files through symbolic links into `windcloud_v4_799` and `windcloud_v4_p46_663`. A self-contained data supplement must dereference those links and contain regular NPZ payload files, the case/split mapping, source metadata, and per-file source and publication digests. Merely copying link objects would produce an incomplete archive.

### Personal-path sanitization and two sets of hashes

Each case contains two textual WRF-output references, `nwp_d01_extraction_metadata_json.wrfout` and `nwp_d02_extraction_metadata_json.wrfout`, whose personal absolute-path prefixes were removed for publication, retaining portable `runs/...` suffixes. No actual personal path is reproduced in the public documentation. Independent full-case checks confirm that the remaining 52 NPY members per case preserve their original compressed payload bytes and that all numerical NPY payloads are bit-exact. The path edit applies to text metadata; all other extraction-metadata values remain unchanged. See the [public sanitization report](../reproduction/recovered_inputs_sanitization.json).

Editing textual metadata changes the containing NPZ bytes, so a sanitized file can have a different SHA-256 and size from its source. Preserve both:

- **Source SHA-256 and size:** identify the current remote input snapshot measured by the recovery audit.
- **Published-file SHA-256 and size:** identify the sanitized NPZ actually distributed and are used for download verification.

A published-file hash differing from its source hash is expected where documented sanitization changed the text. It must not be described as accidental corruption or an unchanged source file. Keep the source-to-publication mapping and sanitization record with the release so users can distinguish this deliberate publication transformation from the separate, unresolved question of historical raw-input byte identity.

Large data are intended for separate release assets, not the Git history. Download all documented parts, follow the release-specific extraction instructions, and verify every extracted payload against the supplied **published-file** SHA-256 manifest. Use the actual release filenames and hashes; do not infer availability or final size from this document or the approximately 5.31 GB source payload size. Text sanitization, archive overhead, and multipart packaging can change the distributed byte counts.

Public recovery evidence must use portable scientific identifiers and relative paths. Credentials, connection details, and absolute personal server locations are outside the public supplement.

## Download layout and verification commands

There are two input-download routes: **GitHub provides four independent ZIP archives directly; the Zenodo distribution uses binary transport chunks that must first be reassembled into those same ZIPs.** Both routes use the code archive `SCFusNet-code-and-data-20261018.zip`, which extracts to `SCFusNet/`. The GitHub download page is tied to tag `inputs-20261018`. Use the Zenodo network route only after its public landing page and files are accessible; GitHub publication does not imply Zenodo publication.

### Route A: four ZIPs from GitHub

Download the code archive and all four input archives from [GitHub release `inputs-20261018`](https://github.com/S1reN-hub/SCFusNet/releases/tag/inputs-20261018) into one working directory:

| File | Contents |
| --- | --- |
| `SCFusNet-code-and-data-20261018.zip` | Code, metadata, cached-head artifacts, and verification scripts under `SCFusNet/` |
| `SCFusNet-inputs-20261018-part01.zip` | First subset of regular files under `cases/` |
| `SCFusNet-inputs-20261018-part02.zip` | Second subset under `cases/` |
| `SCFusNet-inputs-20261018-part03.zip` | Third subset under `cases/` |
| `SCFusNet-inputs-20261018-part04.zip` | Fourth subset under `cases/` |

The four input files are **independent ZIP archives**, each at or below 1.5 GB, totaling 5,305,991,774 bytes. They are not segments of one split ZIP. Extract each archive into the **same** `inputs/` directory; do not concatenate the four ZIPs or extract them into separate per-part data roots.

The following Python commands work without a separate unzip utility. Run them from the directory containing the five downloaded archives:

```sh
python -m zipfile -e SCFusNet-code-and-data-20261018.zip .
python -m zipfile -e SCFusNet-inputs-20261018-part01.zip inputs
python -m zipfile -e SCFusNet-inputs-20261018-part02.zip inputs
python -m zipfile -e SCFusNet-inputs-20261018-part03.zip inputs
python -m zipfile -e SCFusNet-inputs-20261018-part04.zip inputs
cd SCFusNet
python -m pip install numpy
python scripts/verify_recovered_inputs.py --data-root ../inputs
```

### Route B: transport chunks from Zenodo

Zenodo distributes **82 binary chunks of up to 64 MiB** instead of the four large input ZIPs. These chunks are transport pieces, not individually extractable ZIP archives. `data/recovered_inputs_20261018/archive_chunks.json` defines their download locations, order, sizes, checksums, and the four parent ZIPs.

After downloading and extracting the code archive, run from its `SCFusNet/` root:

```sh
python scripts/restore_zenodo_inputs.py --output-dir ../downloaded-input-archives --workers 4
```

The script uses the bundled chunk manifest by default, downloads and verifies chunks, then reassembles and checks the four ZIPs in `../downloaded-input-archives/`. If all Zenodo files have already been downloaded into one directory, use them locally:

```sh
python scripts/restore_zenodo_inputs.py --local-chunk-dir /path/to/download-all --offline --output-dir ../downloaded-input-archives --workers 4
```

Only chunks assigned to the same parent ZIP are concatenated in manifest order. **The four restored ZIPs are independent archives and must never be concatenated together.** Extract all four into one `inputs/` directory, then verify its NPZ files:

```sh
python -m zipfile -e ../downloaded-input-archives/SCFusNet-inputs-20261018-part01.zip ../inputs
python -m zipfile -e ../downloaded-input-archives/SCFusNet-inputs-20261018-part02.zip ../inputs
python -m zipfile -e ../downloaded-input-archives/SCFusNet-inputs-20261018-part03.zip ../inputs
python -m zipfile -e ../downloaded-input-archives/SCFusNet-inputs-20261018-part04.zip ../inputs
python -m pip install numpy
python scripts/verify_recovered_inputs.py --data-root ../inputs
```

Restoring ZIPs validates transport chunks and parent archives. The final `verify_recovered_inputs.py` command separately checks the published NPZ payloads and case contracts. Keep the two verification stages distinct. The Zenodo version identifier must be treated as reserved until its public landing page is accessible; these commands do not establish public availability.

The bundled [offline restoration report](../reproduction/recovered_archive_restoration.json) records reconstruction of all four ZIPs from all 82 local chunks, with archive hashes matching the originals. This was a local transport test; no network download or public availability is implied by that report.

### Final layout and payload verification

Both routes produce the following code/data layout:

```text
working-directory/
  SCFusNet/
    scripts/verify_recovered_inputs.py
    data/recovered_inputs_20261018/
  inputs/
    cases/
      <sample_id>.npz
```

For an existing repository checkout, extract all four input archives into a chosen input root, then run from the repository root:

```sh
python scripts/verify_recovered_inputs.py --data-root /path/to/inputs
```

The metadata directory defaults to `data/recovered_inputs_20261018/` inside the code repository, so it need not be copied into the input root. Full verification checks all 1,462 published-file sizes and SHA-256 values, the case-file set, NPZ array headers and dtypes, scalar identities, and saved scientific contracts. Only NumPy is needed for this data verification; PyTorch is not required. The check does not train a model or establish historical raw-payload identity. To validate just the bundled metadata without downloading inputs, run `python scripts/verify_recovered_inputs.py --metadata-only`; this does not check any NPZ payload.

## Missing artifacts and reproducibility

The formal expert/backbone checkpoint has expected SHA-256:

```text
0c2eadcd625a57686e02d604714e81de864efb19ca92c951a4d8e40e863532f9
```

All 87 examined HPC checkpoint candidates were hashed; none matched. The matching expert normalization also remains missing. Its historical size is 4,712 bytes, but no independently recorded expected normalization hash was located. Fusion-head normalization is not a substitute.

Existing fusion/AOD weights and feature caches remain useful for the released metric and cached-head checks. Input recovery does not supply the missing original experts, establish complete executed-source identity, or make exact original-run raw-input inference available. Preserved source snapshots differ. A replacement training run must be reported as a new experiment, with its own preprocessing, implementation, configuration, weights, and checksums.

## Rights and citation

Source data retain their upstream attribution and access conditions; recovery does not assign a new blanket data license. BSRN notices and dataset DOI information must be preserved where applicable. See [Data availability](DATA_AVAILABILITY.md) and the repository rights statement.

Cite the version-specific archive actually used and identify whether the experiment used the original cached-feature release or the recovered-input supplement. The v1 DOI refers to the preserved original archive; the concept DOI identifies the release family. Version DOI `10.5281/zenodo.23445294` becomes effective on publication and must not be cited as a published Zenodo supplement until that publication is confirmed.
