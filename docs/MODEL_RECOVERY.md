# Formal expert recovery supplement

## Complete inference verification update — 2026-10-18

The **reproduction-20261018** version provides the full NPZ-to-experts-to-formal-fusion-to-AOD workflow and original-cohort metric checks. All 119 test cases and three formal fusion seeds were evaluated without retraining or selection. Main-table metrics match the manuscript at its reported precision, and the exploratory 38-hour AOD RMSE reduction remains 11.40%. Unrounded differences are preserved in the reports. See [raw-input reproduction](RAW_INPUT_REPRODUCTION.md) and [submission-ready availability/reproducibility wording](REPRODUCIBILITY_STATEMENT.md).

Archive identifier: **[10.5281/zenodo.23446670](https://doi.org/10.5281/zenodo.23446670)**. This version preserves a complete code/results package and the recovered-expert ZIP together. The earlier input-preservation DOI `10.5281/zenodo.23445294` is unchanged; statements below about the model supplement being absent from that older version remain historical facts. Bulk inputs continue to use the four fixed GitHub ZIPs.

Download [the verified model supplement](https://github.com/S1reN-hub/SCFusNet/releases/download/experts-20261018/SCFusNet-recovered-experts-20261018.zip) from [GitHub release `experts-20261018`](https://github.com/S1reN-hub/SCFusNet/releases/tag/experts-20261018). Extract it and run commands from `SCFusNet-recovered-experts/`.

Recovered on **2026-10-18**, without retraining. This supplement restores raw-input inference through the two frozen expert branches and the separately preserved formal dynamic MLP heads.

## Recovery status

| Artifact | Status |
| --- | --- |
| `base_normalization.json` | Exact original paired normalization bytes; 4,712 bytes, SHA-256 `3432ea2177762468cdae240a42edf9f1de8777345eabd327950d8570f7a02474` |
| `recovered_experts.pt` | 383 expert state-dictionary entries, 4,156,909 parameters, extracted from the frozen pilot checkpoint into a **new serialization** |
| `formal_router_seed42.pt`, `formal_router_seed43.pt`, `formal_router_seed44.pt` | Unmodified formal dynamic MLP checkpoints, including their separate feature normalization |
| `source_frozen_pilot.pt` | Unmodified extraction source, SHA-256 `6adf17bb57eedc54d022c7f50fe922b6371c4132448985563b4af2a96a232795`; its trained pilot router is excluded from inference |
| `aod_reference.json` | Semantically preserved input reference from the frozen checkpoint; JSON formatting is newly generated |
| `compat_model.py`, `compat_data.py`, `compat_router.py`, `predict.py` | Compatible historical preprocessing/model code and a portable inference entry point |

The missing historical full checkpoint has SHA-256 `0c2eadcd625a57686e02d604714e81de864efb19ca92c951a4d8e40e863532f9`. This package **does not contain that original serialized file**. Its built-in router and conditional correction modules have not been recovered. The package uses recovered expert branches with the original external dynamic MLP heads; it does not execute an original complete `best_fusion.pt`.

The frozen pilot protocol records that exact historical checkpoint and the paired normalization hash. Its source loads the formal checkpoint strictly, freezes both experts, and trains only the pilot router. The saved first-step gradient audit records zero expert gradient tensors. The recovered normalization also equals the input normalization embedded in the frozen checkpoint.

## Validation

`VALIDATION.json` records inference on **all 119 original test cases**, six leads, and seven evaluation stations, using the recovered NPZ inputs. No parameters were trained or selected using this check.

- All 383 extracted tensors survive a restricted-loader save/load round trip bit for bit. Both expert branches and all three formal heads load strictly.
- Targets and all availability, validity, and daylight masks exactly match the historical test cache.
- Across 1,428 expert predictions, maximum absolute difference from the historical cache is **0.20255 W m-2**, mean absolute difference **0.00454 W m-2**.
- Against the original valid-record prediction CSVs, maximum absolute differences for formal head seeds 42/43/44 are **0.14832 / 0.14429 / 0.16479 W m-2**, respectively, across 675 rows per seed.

These results establish close numerical compatibility. They do not establish bit-exact original raw-input inference. The validation used CPU PyTorch 2.12.0; historical caches came from the original GPU environment. Backend differences may contribute, but the precise cause of every difference has not been isolated. Historical input NPZ byte identity and complete executed-source identity remain unverified. The 675-row model-specific valid cohort is different from the manuscript's 601-record all-method comparison.

## Run inference

Use Python with NumPy and PyTorch installed. Download and extract all four published input archives into one directory containing `cases/<sample_id>.npz`. Keep the files in this model supplement together.

```sh
python predict.py --dataset-root /path/to/inputs --index index.csv --output predictions.csv --seed 42 --device cpu
```

The default partition is `test`. Use `--split validation`, `--split train`, or `--split all` explicitly for other partitions. `--limit 2` provides a quick check. Use seeds 42, 43, and 44 separately when reproducing seed-level comparisons. This command returns the base dynamic-fusion forecast; the separately released exploratory AOD correction is not applied.

`predict.py` and the validation use `torch.load(..., weights_only=True)`. The runtime intentionally preserves the original 136/133/11 feature schema and September 2026 preprocessing rather than substituting later revisions.

## Provenance and rights

See `PROVENANCE.json`, `evidence/`, `VALIDATION.json`, and `CHECKSUMS.sha256`. The original evidence files and source frozen checkpoint are included so that the extraction and freeze policy can be inspected. This supplement is hosted on GitHub and was produced **after** Zenodo version DOI `10.5281/zenodo.23445294`; it is not contained in that immutable Zenodo snapshot. No new DOI is assigned to this supplement.

Public visibility does not assign a blanket license. Existing code rights and upstream data attribution/access conditions continue to apply.
