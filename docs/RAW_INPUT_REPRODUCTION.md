# Raw-input inference and paper-metric reproduction

Version **reproduction-20261018**, archive identifier **[10.5281/zenodo.23446670](https://doi.org/10.5281/zenodo.23446670)**. This package adds recovered frozen expert branches, their exact paired normalization, the original formal fusion and AOD heads, a portable full inference entry point, and numerical verification. It does not restore the original monolithic checkpoint or claim an exact reconstruction of its training execution.

## Download layout

The Zenodo version contains a complete current code/results ZIP and the unchanged recovered-expert ZIP. Extract the code ZIP to `SCFusNet/` and the model ZIP to `SCFusNet-recovered-experts/`. No older overlay is needed for this version. The four large input archives remain at the fixed [GitHub input release](https://github.com/S1reN-hub/SCFusNet/releases/tag/inputs-20261018); their sizes and hashes are archived with the code. Zenodo is not a mirror of the 5.31-GB bulk inputs.

From the extracted `SCFusNet/` directory:

```sh
python scripts/restore_zenodo_inputs.py --source github --output-dir ../input-archives
python scripts/fetch_recovered_models.py --output-dir ../models
```

Extract all four independent input ZIPs into the same `../inputs` directory, which must then contain `cases/<sample_id>.npz`. Do not concatenate the ZIPs. The model fetcher verifies and extracts its ZIP automatically. If the model ZIP was downloaded from Zenodo manually, extract it and use that directory in the inference command instead.

## Environment and execution

The verified environment is **Windows 11, CPython 3.14.3, NumPy 2.4.3, PyTorch 2.12.0+cu126**, using **CPU inference**, four threads and batches of four cases. `requirements-reproduction.txt` pins its Python packages; PyTorch's CUDA-enabled distribution was used even though inference selected CPU. Official wheels for this build are listed at [PyTorch's cu126 index](https://download.pytorch.org/whl/cu126/torch/). The original training environment has not been recovered. Other platforms/builds may produce different small numerical errors and have not been certified by this release.

Install in a dedicated Python environment:

```sh
python -m venv .venv
# Activate the environment with the command appropriate for your shell.
python -m pip install -r requirements-reproduction.txt
```

Run the complete inference and verification workflow:

```sh
python scripts/reproduce_raw_pipeline.py --data-root ../inputs --model-bundle ../models/SCFusNet-recovered-experts --output-dir reproduction/raw_pipeline --device cpu
```

This command verifies all model-manifest entries and the published SHA-256 of all 119 selected test inputs before inference. It computes expert representations from NPZ files, applies each original formal fusion head, applies the already selected AOD correction checkpoints, writes per-seed predictions, compares historical references, and scores the original evaluation cohorts. Historical cached states and predictions are **reference evidence only**; they are not substituted for raw-input inference. No training, tuning, gate selection, or checkpoint selection is performed. All `.pt` loading in the new runtime uses `weights_only=True`.

The first measured full run took approximately 61 seconds. A second complete run from independently extracted code/model archives, launched in an unrelated empty working directory, produced identical metrics (approximately 68 seconds for inference/verification). See [archive portability verification](../reproduction/raw_input_20261018/PORTABILITY_REPORT.json). Timings exclude dependency installation, downloading and extraction. The workflow loads only a small batch of NPZ files at a time. Execution time depends on hardware, storage and software. The installation recipe is supplied with exact pins; verification used an existing installed environment rather than a newly installed machine.

## Scope and verified results

The workflow runs **119 cases x 6 leads**, with formal head seeds 42, 43 and 44. Every saved seed CSV contains 714 forecast rows, including invalid/nighttime records with masks.

| Result | Cohort and aggregation | Recomputed value |
| --- | --- | --- |
| Main comparison MAE | 601 common daylight records; score seeds separately, then average | 62.301081 W m-2 -> 62.30 |
| Main comparison RMSE | Same 601 records | 109.491034 W m-2 -> 109.49 |
| Main comparison nRMSE | Same 601 records; normalize by mean target | 20.707379% -> 20.71% |
| Main comparison R2 | Same 601 records | 0.874612 -> 0.875 |
| Forecast skill | Original 452-record smart-persistence intersection | 34.234043% -> 34.23% |
| AOD admitted-hour RMSE | 38 unique hours in 10 cases; average seed metrics | 51.961531 -> 46.040052 W m-2 |
| AOD admitted-hour RMSE reduction | Ratio of the preceding mean per-seed RMSE values | 11.395891% -> 11.40% |

The main comparison uses the **before-AOD dynamic-fusion forecast**. The after-AOD forecast is a separate exploratory sensitivity result on a previously inspected test split. It must not replace the main-table column or be described as a new confirmatory test. The AOD admission keys match the historical 38-hour set exactly, and forecasts outside admission remain unchanged.

Main-table MAE/RMSE/nRMSE/R2/FS and the three raw-input expert/fixed-fusion rows match the manuscript at its published rounding. Exact unrounded deltas, per-seed metrics, correction-checkpoint hashes and environment details are recorded in [the full report](../reproduction/raw_input_20261018/RAW_PIPELINE_REPORT.json). The maximum individual before/after prediction difference from historical AOD-table references is approximately 0.165 W m-2; the main-table mean RMSE difference is -0.000180 W m-2. These are numerical compatibility results, not bit-exact equality.

## Outputs and remaining limits

- `RAW_PIPELINE_REPORT.json`: all numerical checks, cohort sizes, per-seed metrics, rounding comparisons and measured runtime.
- `raw_seed42.csv`, `raw_seed43.csv`, `raw_seed44.csv`: fresh predictions, targets, validity/daylight masks, branch outputs, fusion weights, AOD inputs and admission flags; GHI is in W m-2.
- `raw_expert_features.pt`: freshly computed branch features, saved as a diagnostic output.
- `ENVIRONMENT.json`: the verification environment, distinct from an unrecovered original training environment.
- `HISTORICAL_TABLE_RECALCULATION.json`: independent metric recalculation from the preserved source tables, including baselines and vertical-profile ablation; these other models are not retrained or rerun from raw inputs here.

The original full `best_fusion.pt` byte stream and its built-in routing/conditional modules remain unavailable. The released inference route instead uses recovered expert branches, the preserved external formal dynamic MLP heads, and separately preserved AOD heads. Exact executed-source identity, original optimizer/RNG state and original training-time per-NPZ byte identity are not established. This release verifies inference and numerical evidence; it does not demonstrate complete training reproducibility or raw-input reproduction of every ablation/baseline.
