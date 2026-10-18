# SCFusNet

Research code and derived data for **SCFusNet: Scenario-Conditioned Satellite–NWP Fusion for 1–6-h Solar Irradiance Forecasting**, by Shaocheng Liu and Jingming Xia. The original public snapshot accompanies the manuscript source dated **2026-10-16**. The **2026-10-18 input-recovery update** adds a separately packaged 1,462-case input snapshot. Manuscript metadata and its source hash identify the version; the manuscript itself is not distributed here.

SCFusNet combines independently supervised satellite and numerical weather prediction (NWP) branches, assigns scenario- and lead-dependent fusion weights, and applies conditional aerosol optical depth (AOD) attenuation. Older artifact and code names such as `SolFUS-Net`, `SolSC-Net`, and `solfus` are retained for traceability.

## Quick start

Run from the repository root in a Python environment with the dependencies in `requirements.txt` installed:

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_paper_metrics.py
python scripts/verify_cached_inference.py --cache outputs/mlp17_7eval_training_20260909/review/test_7station/test_cache.pt --heads-dir outputs/mlp17_7eval_training_20260909/review/dynamic_mlp
```

The first command recomputes reported metrics from released prediction tables. The second checks fusion-head inference using released cached branch representations and checkpoints. These commands do not reconstruct satellite imagery or WRF fields.

## What is released

| Path | Contents |
| --- | --- |
| `src/ghi_hierarchical_v2`, `src/ghi_hierarchical_v3`, `src/ghi_hierarchical_v4` | Versioned model, dataset, training, and evaluation implementations |
| `scripts/` | Selected data preparation, training, evaluation, and release verification scripts |
| `outputs/dataset_split_17train_7eval_expanded_20260909/` | Case assignments, train/validation/test lists, and split audits |
| `outputs/mlp17_7eval_training_20260909/review/` | Frozen expert caches, dynamic-fusion checkpoints, and prediction exports |
| `outputs/aod_cloud_cache_20260913/` | Train/validation/test feature caches enriched with AOD and cloud diagnostics |
| `outputs/aod_hardgate_relaxation_v1_20260913/` | All nine AOD sensitivity candidates, selection protocol, checkpoints, and results |
| `results/fulltest_model_comparison_20260921/` | Matched prediction tables, baseline comparisons, metrics, and audits |
| `work/wrf_vertical_profile_ablation_20260925_remote/bundle/` | Retrieved vertical-profile ablation results |
| `tmp/remote_mlp17_original/` | Partial recovered remote implementation snapshot; historical provenance rather than an exact complete training snapshot |

## Recovered input supplement

A **1,462-case, 17-station, 86-cycle source input snapshot** has been located and fully enumerated: **5,307,594,704 source bytes** of NPZ payloads, with a new per-file SHA-256 manifest. Its two source-index hashes match the formal 2026-09-09 preparation log, and its canonical case assignments exactly match the published 1,224/119/119 split. Download the complete current code archive, four input archives, and checksums from the public [GitHub release `inputs-20261018`](https://github.com/S1reN-hub/SCFusNet/releases/tag/inputs-20261018). All nine release assets have passed server-digest checks. Large input tensors are release assets rather than Git-tracked files.

All 1,462 files have now been downloaded with their source checksums verified. Publication copies remove personal absolute-path prefixes from textual extraction-metadata WRF-output references, retaining portable `runs/...` paths. Numeric array NPY bytes and their original compressed payload bytes are preserved. The sanitized publication payload totals **5,305,765,926 bytes**, compared with 5,307,594,704 source bytes; the four input ZIPs together total **5,305,991,774 bytes**. The supplement records **source SHA-256 and published-file SHA-256 separately**; verify downloaded files against the published-file hashes.

The **GitHub** download contains four **independent ZIP archives**, `SCFusNet-inputs-20261018-part01.zip` through `part04.zip`. **Zenodo does not host these bulk inputs or the previously planned 82 transport chunks.** The new Zenodo deposit preserves the original v1 code ZIP and adds `SCFusNet-recovery-supplement-20261018.zip`: a small overlay containing updated metadata, scripts, documentation, verification reports, rights information, and manifests that identify the GitHub bulk files by exact hash. See [the Zenodo overlay instructions](docs/DATA_RECOVERY.md#zenodo-preservation-and-recovery-overlay).

Extract all four input ZIPs into one input directory so it contains `cases/<sample_id>.npz`; **never concatenate the four independent ZIPs together**. The complete current GitHub code archive, `SCFusNet-code-and-data-20261018.zip`, extracts to `SCFusNet/`. From that code root, run:

```sh
python -m pip install numpy
python scripts/verify_recovered_inputs.py --data-root /path/to/inputs
```

Metadata are read from the repository's `data/recovered_inputs_20261018/` by default. The repository includes [full payload verification](reproduction/recovered_inputs_verification.json), [independent numerical-preservation and sanitization checks](reproduction/recovered_inputs_sanitization.json), and [offline restoration of all 82 chunks](reproduction/recovered_archive_restoration.json). See [complete extraction instructions](docs/DATA_RECOVERY.md#download-layout-and-verification-commands). Local validation and a reserved DOI do not establish public Zenodo availability.

The supplement represents the **current recovered snapshot**. No historical per-NPZ digest was located, so index identity and new checksums do not prove byte-for-byte identity with the original training inputs. The **formal satellite/NWP expert checkpoint and its matching normalization remain missing**. No claim is made that they will be included in the data supplement. Fusion and AOD head weights are already included; complete original-run end-to-end reproducibility remains unavailable. See [data recovery](docs/DATA_RECOVERY.md) for evidence, packaging requirements, and limitations.

**Version DOI: `10.5281/zenodo.23445294` (registration becomes effective on publication).** Treat this as a reserved identifier until its public landing page is accessible; listing it here is not proof of publication or downloadable files. The [Zenodo concept DOI](https://doi.org/10.5281/zenodo.23418518) identifies the release family, and the original [v1 archive](https://doi.org/10.5281/zenodo.23418519) is preserved unchanged.

## Evaluation scope

The split comprises 1,224 training, 119 validation, and 119 test station–cycle cases. Seven evaluation stations are GOB, DRA, IZA, TBL, SEL, GWN, and CAB. Other training stations can share weather cycles with the evaluation stations; this is not a globally held-out-cycle experiment.

The primary model comparison uses **601 common valid daytime records**. Forecast skill against smart persistence uses a separate **452-record intersection** with valid persistence inputs. SCFusNet obtains MAE 62.30 W m⁻², RMSE 109.49 W m⁻², nRMSE 20.71%, R² 0.875, and forecast skill 34.23% under these respective cohorts. Metrics are computed separately for seeds 42, 43, and 44 and then averaged; they are not metrics of an averaged forecast.

The selected AOD policy admits **38 unique target hours across 10 test cases** and reports an 11.40% admitted-hour RMSE reduction. The gate study is **exploratory**: the test split had already been inspected. Candidate and checkpoint selection used validation data, and all candidates are retained. This result is not a fresh confirmatory test or a reduction over the entire 601-record cohort.

See [reproducibility](docs/REPRODUCIBILITY.md), [data availability](docs/DATA_AVAILABILITY.md), [data recovery](docs/DATA_RECOVERY.md), and the [data dictionary](docs/DATA_DICTIONARY.md) for definitions and limitations.

## Data sources and reuse

Ground observations originate from [BSRN](https://bsrn.awi.de/data/) and [SURFRAD](https://gml.noaa.gov/grad/stardata.html); satellite inputs include GOES, Himawari, and [GridSat-B1](https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature); aerosol forecasts originate from [CAMS](https://ads.atmosphere.copernicus.eu/). WRF-Chem simulations use GFS forcing. Released derived tables and feature caches do not replace the upstream datasets or their attribution and access conditions.

No blanket open-source or dataset license is assigned by this snapshot. Public visibility alone does not grant unrestricted reuse; upstream data terms and the rights of the respective owners continue to apply. The repository/archive DOIs above identify research artifacts, not a journal publication.
