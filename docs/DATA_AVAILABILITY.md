# Data availability

The original release accompanies the 2026-10-16 manuscript version of SCFusNet. This statement was updated on 2026-10-18 to distinguish the preserved original archive, the recovered-input supplement, and artifacts that remain missing.

## Included material

- Case-level split manifests for 1,462 station–cycle cases: 1,224 training, 119 validation, and 119 testing cases.
- Frozen branch feature caches and targets, fusion-head checkpoints for seeds 42/43/44, and exported predictions.
- AOD-enriched feature caches, candidate-specific AOD correction weights, admission diagnostics, and all nine sensitivity candidates.
- Matched daytime prediction tables, smart-persistence comparison material, summary metrics, data-quality audits, and ablation result exports.
- Historical launch and partial recovered remote snapshots, which differ and are not an exact complete formal-training snapshot.

The original public artifacts support independent recalculation of released metrics and verified inference checks for the cached fusion heads. AOD feature caches, correction weights, and historical scripts are included for inspection. The original v1 archive does not contain the full case-level multimodal input tensors.

## Recovered-input supplement

The 2026-10-18 recovery audit located all **1,462 referenced NPZ case payloads** across two source datasets, totaling **5,307,594,704 source-snapshot bytes**, with 17 stations and 86 cycles. All 1,462 source files have been downloaded and their source checksums verified. Both source-index SHA-256 values match the formal 2026-09-09 preparation log. Canonical case keys match the published assignments exactly, with no missing or extra cases and the same 1,224/119/119 train/validation/test split. The sanitized publication payload totals **5,305,765,926 bytes**; the four outer ZIP archives total **5,305,991,774 bytes**.

The public [GitHub release `inputs-20261018`](https://github.com/S1reN-hub/SCFusNet/releases/tag/inputs-20261018) provides the complete current code archive, four independent input ZIPs, and associated release files; all nine assets passed server-digest checks. Input archives `SCFusNet-inputs-20261018-part01.zip` through `part04.zip` contain actual dereferenced payloads and all extract into the same directory containing `cases/`. The current code archive is `SCFusNet-code-and-data-20261018.zip`, with root directory `SCFusNet/`. See [extraction and verification](DATA_RECOVERY.md#download-layout-and-verification-commands). Large payloads are release assets rather than Git-tracked files.

The new Zenodo deposit preserves the original v1 code ZIP and adds the small `SCFusNet-recovery-supplement-20261018.zip` overlay with updated metadata, scripts, documentation, three recovery-verification reports, rights information, and archive/hash manifests. Apply the overlay to the extracted v1 code root to obtain the current recovery material. The complete current code ZIP and all bulk inputs are hosted on GitHub. Zenodo retains durable provenance and exact-hash references to those bulk files; **it is not a bulk-data mirror and does not contain the previously planned 82 transport chunks**. See [overlay instructions](DATA_RECOVERY.md#zenodo-preservation-and-recovery-overlay). GitHub availability does not establish Zenodo publication.

Personal absolute-path prefixes in `nwp_d01_extraction_metadata_json.wrfout` and `nwp_d02_extraction_metadata_json.wrfout` are removed from publication copies, retaining portable `runs/...` references. Independent checks across all 1,462 cases confirm that numerical array NPY bytes and the 52 unchanged members per case retain identical original compressed payload bytes. Because the text edit changes NPZ file bytes, manifests distinguish the source-snapshot SHA-256 from the published-file SHA-256. The latter is the download-verification digest. Source byte totals must not be presented as the final sanitized payload or archive size. Public [verification reports](REPRODUCIBILITY.md#published-recovery-verification-reports) record full payload, sanitization, and offline archive-restoration checks.

The recovered files are a **source-index-matched current snapshot**, not a historically byte-authenticated copy of every original training NPZ. Newly measured hashes support transfer integrity and future reproducibility; no historical per-case payload digest was located for comparison. See [Data recovery](DATA_RECOVERY.md) for the evidence and its limits.

**Version DOI: `10.5281/zenodo.23445294` (registration becomes effective on publication).** Treat the identifier as reserved until the public landing page is accessible; its inclusion here is not publication confirmation. The [Zenodo concept DOI](https://doi.org/10.5281/zenodo.23418518) resolves the release family. The [original v1 DOI](https://doi.org/10.5281/zenodo.23418519) continues to identify the preserved original archive. Cite the new Zenodo version only after its publication is confirmed.

## Material absent from this snapshot

The formal trained satellite/NWP expert checkpoint and its matching expert normalization have **not been recovered**. All 87 checkpoint candidates examined on HPC were hashed; none matched the historically recorded formal checkpoint digest. The expert normalization has only a historical 4,712-byte size record and no independently recorded expected hash. Fusion-state normalization does not replace expert normalization.

Consequently, publishing recovered inputs will not by itself enable exact original-checkpoint raw-input inference or establish complete original-run end-to-end reproduction. A new training run may use the inputs after the required preprocessing and configuration work, but a newly trained model must not be described as the recovered original checkpoint. Historical source snapshots differ and do not establish the complete executed-source identity.

The full manuscript, correspondence, account credentials, and unrelated project material are outside the release. Manuscript version identification uses bibliographic metadata and a source-file hash.

## Upstream provenance

| Component | Source or access point | Role |
| --- | --- | --- |
| Ground GHI | [BSRN](https://bsrn.awi.de/data/), [SURFRAD](https://gml.noaa.gov/grad/stardata.html) | Hourly supervision and evaluation labels |
| Satellite | GOES, Himawari, [GridSat-B1](https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature) | Recent cloud morphology and motion; source varies by station coverage |
| NWP | GFS-driven WRF-Chem simulations, 9-km and 3-km domains | Multilayer atmospheric fields and surface diagnostics |
| Aerosol | [CAMS Atmosphere Data Store](https://ads.atmosphere.copernicus.eu/) | AOD at 550 nm |
| Solar geometry | Derived from station location and target time | Daylight, clear-sky reference, and air mass |

Use the original providers' access procedures and attribution terms when obtaining source data. BSRN source files retain their BSRN-1.0 notices and dataset DOI information where present. The release does not claim ownership of upstream observations or forecasts, and no blanket data license is imposed on them.

## Interpretation of released subsets

The 601-record common daytime comparison, the 452-record smart-persistence intersection, and the 38-hour AOD-admitted subset answer different evaluation questions. Counts are unique target-hour records, not multiplied by the three model seeds. AOD admission spans 10 unique test cases. Split manifests contain potential geometrically eligible hours before all label and persistence checks; these counts need not equal final metric cohort sizes.

AOD sensitivity analysis was conducted on a previously inspected test split. Its protocol explicitly records this exploratory status. Validation selects checkpoints and the final candidate; all declared candidates remain available to expose sensitivity rather than only the selected outcome.

## Rights

No software or dataset license is newly assigned in this release. Consult the respective rights holders and upstream terms for reuse beyond the permissions otherwise applicable. Public availability is not a statement of unrestricted licensing.
