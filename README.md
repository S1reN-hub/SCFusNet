# SCFusNet

Research code and derived data for **SCFusNet: Scenario-Conditioned Satellite–NWP Fusion for 1–6-h Solar Irradiance Forecasting**, by Shaocheng Liu and Jingming Xia. This public snapshot is organized against the manuscript source dated **2026-10-16** and packaged on **2026-10-16**. Manuscript metadata and its source hash identify the version; the manuscript itself is not distributed here.

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

The full 1,462-case multimodal input tensors and the trained satellite/NWP backbone checkpoint are retained on the original remote training server and will be uploaded in a subsequent update. They are not included in the current local snapshot. **This is a results-and-cached-head reproducibility release, not a complete end-to-end training release.** Fusion and AOD head weights are included. Historical training scripts may require the original GPU-host layout and external input files; their presence does not establish a portable raw-data training workflow.

## Evaluation scope

The split comprises 1,224 training, 119 validation, and 119 test station–cycle cases. Seven evaluation stations are GOB, DRA, IZA, TBL, SEL, GWN, and CAB. Other training stations can share weather cycles with the evaluation stations; this is not a globally held-out-cycle experiment.

The primary model comparison uses **601 common valid daytime records**. Forecast skill against smart persistence uses a separate **452-record intersection** with valid persistence inputs. SCFusNet obtains MAE 62.30 W m⁻², RMSE 109.49 W m⁻², nRMSE 20.71%, R² 0.875, and forecast skill 34.23% under these respective cohorts. Metrics are computed separately for seeds 42, 43, and 44 and then averaged; they are not metrics of an averaged forecast.

The selected AOD policy admits **38 unique target hours across 10 test cases** and reports an 11.40% admitted-hour RMSE reduction. The gate study is **exploratory**: the test split had already been inspected. Candidate and checkpoint selection used validation data, and all candidates are retained. This result is not a fresh confirmatory test or a reduction over the entire 601-record cohort.

See [reproducibility](docs/REPRODUCIBILITY.md), [data availability](docs/DATA_AVAILABILITY.md), and the [data dictionary](docs/DATA_DICTIONARY.md) for definitions and limitations.

## Data sources and reuse

Ground observations originate from [BSRN](https://bsrn.awi.de/data/) and [SURFRAD](https://gml.noaa.gov/grad/stardata.html); satellite inputs include GOES, Himawari, and [GridSat-B1](https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature); aerosol forecasts originate from [CAMS](https://ads.atmosphere.copernicus.eu/). WRF-Chem simulations use GFS forcing. Released derived tables and feature caches do not replace the upstream datasets or their attribution and access conditions.

No blanket open-source or dataset license is assigned by this snapshot. Public visibility alone does not grant unrestricted reuse; upstream data terms and the rights of the respective owners continue to apply. No publication DOI is asserted.

