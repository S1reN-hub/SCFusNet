# Reproducibility

## Complete inference verification update — 2026-10-18

The **reproduction-20261018** version provides the full NPZ-to-experts-to-formal-fusion-to-AOD workflow and original-cohort metric checks. All 119 test cases and three formal fusion seeds were evaluated without retraining or selection. Main-table metrics match the manuscript at its reported precision, and the exploratory 38-hour AOD RMSE reduction remains 11.40%. Unrounded differences are preserved in the reports. See [raw-input reproduction](RAW_INPUT_REPRODUCTION.md) and [submission-ready availability/reproducibility wording](REPRODUCIBILITY_STATEMENT.md).

Archive identifier: **[10.5281/zenodo.23446670](https://doi.org/10.5281/zenodo.23446670)**. This version preserves a complete code/results package and the recovered-expert ZIP together. The earlier input-preservation DOI `10.5281/zenodo.23445294` is unchanged; statements below about the model supplement being absent from that older version remain historical facts. Bulk inputs continue to use the four fixed GitHub ZIPs.

## Supported reproduction levels

| Level | Supported by this snapshot | Limitation |
| --- | --- | --- |
| Recalculate reported metrics | Yes, from released prediction and reference tables | Must retain each metric's original evaluation cohort |
| Re-evaluate fusion heads | Verified from released feature caches and head checkpoints | Starts after pretrained satellite/NWP feature extraction |
| Run the selected AOD correction | Verified from raw inputs with the unchanged selected heads and gate | A separate exploratory 38-hour result; not the main-table forecast |
| Retrain heads on cached representations | Code and cached features are provided | Historical scripts may retain original-host defaults; reproducibility depends on the exact archived implementation and configuration |
| Inspect/reconstruct case-level multimodal inputs | Packaged input supplement; see the fixed GitHub release assets | Current recovered snapshot has matching historical source indices and published split, but no historical per-NPZ digest |
| Infer with recovered formal expert branches | Verified on all 119 test cases; see [model recovery](MODEL_RECOVERY.md) | Recovered from a frozen pilot into a new serialization; original full checkpoint bytes and exact original-run inference remain unverified |
| Retrain a complete replacement model | Not established as a portable workflow | Recovered inputs can support new work once downloaded and configured; historical source versions differ, and a new run does not recover the original model |

## Historical table and cached-head verification commands

From the repository root:

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_paper_metrics.py
python scripts/verify_cached_inference.py --cache outputs/mlp17_7eval_training_20260909/review/test_7station/test_cache.pt --heads-dir outputs/mlp17_7eval_training_20260909/review/dynamic_mlp
```

The metrics script checks released numerical evidence. The inference script checks fusion heads using cached features and head weights. Review the scripts' output for actual checks and numerical differences; successful cached verification does not establish complete raw-input reproducibility.

PyTorch `.pt` artifacts contain research dictionaries and checkpoints. Load only artifacts from a source you trust; historical scripts use `torch.load(..., weights_only=False)` to read metadata as well as tensors.

## Version and implementation provenance

This package targets the manuscript source dated 2026-10-16. Version metadata and its source hash identify the manuscript without distributing its full text. The final paper name is SCFusNet. Earlier `SolFUS-Net`, `SolSC-Net`, `solfus`, and `ghi_hierarchical_v*` names in artifact paths and columns are preserved so that tables remain traceable to their producers.

Historical launch and partial recovered snapshots are retained in `tmp/mlp17_7eval_training_bundle_20260909/` and `tmp/remote_mlp17_original/`. Their model files differ, and neither is claimed to be the exact complete implementation of the formal training run. The current working-tree model also differs from the recovered snapshot. Source trees for versions 2, 3, and 4 document experiment lineage. Successful cached inference checks establish agreement for the tested released heads, not complete provenance of the backbone training implementation.

## Recovered-input verification

The 2026-10-18 audit enumerated and hashed all 1,462 source NPZ payloads, totaling 5,307,594,704 source-snapshot bytes. All files have been downloaded with source checksums verified; sanitized publication payloads total 5,305,765,926 bytes before outer ZIP packaging. Source indices match hashes recorded in the formal data-preparation log; canonical case assignments match the published split exactly. These checks establish the recovered case mapping and the current source snapshot's transfer-verification baseline. No historical per-case NPZ hashes were located, so original-training payload byte equality remains unverified.

The download contains actual payload bytes rather than the source directories' symbolic links. Publication copies remove personal absolute-path prefixes from textual extraction-metadata WRF-output references while preserving numerical array NPY bytes and their original compressed payload bytes. Source and published-file hashes are therefore recorded separately. After obtaining the supplement from [GitHub release `inputs-20261018`](https://github.com/S1reN-hub/SCFusNet/releases/tag/inputs-20261018), compare every extracted case with its **published-file SHA-256** before running new experiments. That comparison establishes agreement with the distributed sanitized snapshot, not retroactive identity to the original training data. See [Data recovery](DATA_RECOVERY.md) for details.

Download the four independent input ZIPs directly from the public GitHub release. Zenodo preserves the original v1 code ZIP plus a small recovery overlay with current metadata, scripts, documentation, verification reports, and exact-hash references to the GitHub files. It does not mirror bulk inputs or host the previously planned 82 transport chunks. See [the preservation and overlay instructions](DATA_RECOVERY.md#zenodo-preservation-and-recovery-overlay). The retained chunk helper and offline restoration report document an earlier local transport test; they are not a working Zenodo bulk-download route.

Extract all four independent input ZIPs into a single input root containing `cases/`; the current GitHub code archive extracts to `SCFusNet/`. From the code root, run `python scripts/verify_recovered_inputs.py --data-root /path/to/inputs`. Only NumPy is required for this payload check, and metadata are read from `data/recovered_inputs_20261018/` by default. See [full extraction commands](DATA_RECOVERY.md#download-layout-and-verification-commands). A v1 code directory updated with the Zenodo overlay can use the same verifier against the GitHub inputs.

The original [v1 archive](https://doi.org/10.5281/zenodo.23418519) is preserved. The published input-recovery version is [10.5281/zenodo.23445294](https://doi.org/10.5281/zenodo.23445294); the [concept DOI](https://doi.org/10.5281/zenodo.23418518) identifies the family. The later GitHub model supplement is not in that Zenodo version.

## Published recovery verification reports

The following reports are included in the repository as public verification evidence. Their inclusion does not establish Zenodo publication.

| Report | Completed checks |
| --- | --- |
| [Recovered inputs](../reproduction/recovered_inputs_verification.json) | All 1,462 published payload sizes and SHA-256 values, metadata mapping, NPZ shape/dtype, identities, and saved scientific contracts |
| [Independent sanitization audit](../reproduction/recovered_inputs_sanitization.json) | All source hashes; all 54 NPY members per case; bit-exact numerical payloads; identical compressed bytes for the 52 unchanged members per case; permitted edits restricted to the two WRF-output path fields; full published-member decompression/CRC checks; personal-path checks across all string members |
| [Archive restoration](../reproduction/recovered_archive_restoration.json) | Historical local test: offline reconstruction from all 82 chunks to the four original ZIP hashes, totaling 5,305,991,774 archive bytes; these chunks were not deposited as a public Zenodo download |

The corresponding source and sanitized NPZ totals are 5,307,594,704 and 5,305,765,926 bytes. These checks support recovery, publication transformation, and transport integrity. These input audits do not establish historical raw-training payload byte identity or represent a new training experiment. The later [model recovery supplement](MODEL_RECOVERY.md) separately recovers the frozen expert branches and the exact paired normalization.

## Split and temporal interpretation

The experimental dataset has 17 stations, 86 initialization cycles, and 1,462 station–cycle cases. Seven evaluation stations use 52/17/17 training/validation/test cycles. The remaining ten stations contribute all their cases to training, yielding 1,224/119/119 cases overall. Overlapping history/target windows are grouped in the split construction.

Validation and test weather cycles may also occur at the other ten training stations. Describe this as a cross-station training setting, not as globally disjoint weather-cycle testing. The current `split` field and `case_key` govern assignments; an old split prefix in `source_sample_id` records provenance and does not override the manifest.

For lead `h` in 1–6, the target is hourly mean GHI over `[initialization + h − 1 hour, initialization + h hours)`. Times explicitly suffixed `_utc` are UTC. Model inputs are restricted to information available at forecast initialization according to the experiment protocol.

## Metric cohorts and aggregation

1. **601 common daytime records:** all compared methods have finite predictions and a valid target. MAE, RMSE, nRMSE, and R² use these matched records.
2. **452 smart-persistence records:** common records additionally satisfying the preceding-observation requirements for the persistence reference. Forecast skill is calculated on this intersection for both model and reference.
3. **38 AOD-admitted hours in 10 cases:** selected policy `d10_c70`, evaluated on its admitted test hours. This is separate from the all-method common cohort.

Let error be forecast minus target. MAE is mean absolute error; RMSE is the square root of mean squared error; nRMSE is `100 × RMSE / mean(target)`; nMBE is `100 × mean(error) / mean(target)`; R² is `1 − sum(error²)/sum((target − mean(target))²)`. Forecast skill is `100 × (1 − RMSE(model)/RMSE(smart persistence))`, using the same 452 records for each RMSE.

Calculate metrics independently for each seed, then average the three metrics for seeds 42, 43, and 44. Do not calculate metrics from an averaged prediction or pool seed rows as independent observations. Deterministic WRF configurations and the frozen learned NWP branch have one prediction per record. The learned NWP branch is distinct from a direct WRF-PBL forecast.

At manuscript rounding, the SCFusNet common-cohort values are MAE 62.30 W m⁻², RMSE 109.49 W m⁻², nRMSE 20.71%, and R² 0.875. Its smart-persistence skill is 34.23% on the persistence intersection.

## AOD sensitivity and selection

The grid contains anomaly thresholds 0.10/0.15/0.20 and joint-clear-sky thresholds 0.70/0.75/0.80. Daylight fraction must exceed 0.99 and mean cosine of solar zenith angle must be at least 0.25; AOD and relevant inputs must be valid. The selected `d10_c70` policy uses anomaly ≥0.10 and joint clear-sky support ≥0.70.

Per-candidate checkpoints use validation admitted-hour MAE, including epoch zero as a no-correction option. Candidate selection minimizes mean per-seed MSE on the same full validation cohort and requires no seed to worsen full-validation MSE, with the recorded tolerance and tie-breaking rule. Consult `protocol.json` and `selection_frozen.json` for the exact saved rules and provenance hashes.

The test split and its gate-feature coverage had already been inspected before this sensitivity grid was defined. Therefore this is an exploratory sensitivity analysis with validation-based selection, not a fresh confirmatory test. All nine candidates, including nonselected outcomes, are retained. The reported 11.40% admitted-hour RMSE reduction does not describe full-test RMSE reduction.

## Historical training scripts

Several scripts preserve absolute Windows or original GPU-host paths. To pursue new training, obtain and verify the recovered-input supplement when available, then adapt dataset roots, split paths, output roots, and device options using each script's arguments. The original serialized full backbone checkpoint remains missing. A later [model supplement](MODEL_RECOVERY.md) recovers the expert branches from a frozen pilot and the exact paired normalization; training any replacement backbone remains a new experiment. Existing audit files preserve historical provenance and are not current instructions to access a remote host.
