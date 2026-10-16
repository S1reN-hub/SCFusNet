# Full-test SolFUS-Net versus deep-learning and direct WRF-PBL baselines

## Scope and common-data rule

The all-method audit matched 601 daylight-valid rows from 110 sample IDs, 7 stations, and 17 independent GFS cycles. The frozen DL/SolFUS reference contains 675 rows and the PBL export contains 714 rows; their key intersection contains 675 rows. The common set requires daylight=1 and finite target/prediction in every frozen export. No row was selected by SolFUS error.

MYNN, YSU, and MYJ are direct deterministic WRF-PBL predictions from the full 17-cycle export (119 cases, 609 PBL-common daylight points). The frozen `nwp` column carried by the SolFUS export is a learned `WRF-NWP branch`, not a traditional WRF-PBL run, and is labelled accordingly. The final all-method intersection retains 601 points for every method.

## Primary metric definition

RMSE, nRMSE = 100 × RMSE / mean daylight target, nMBE = 100 × mean signed error / mean daylight target, and R² are recomputed on the all-method common rows. For SolFUS-Net and the three DL baselines, the primary aggregate is the arithmetic mean of three seed-level metrics; seed SD is retained in the source tables but is not drawn in Fig. 4-1. MYNN, YSU, MYJ, and the WRF-NWP branch are deterministic in this comparison.

## Figure 4-1 construction

The primary comparison figure contains seven methods: MYNN, YSU, MYJ, Sat-ConvLSTM, WRF-Transformer, EarlyFusion-Transformer, and SolFUS-Net. The internal learned WRF-NWP branch remains in the source metrics and audit tables but is excluded from the figure. Panel a reports overall MAE, and panel b reports overall FS against smart persistence with one mean label above each bar; each learning-model bar is a three-seed mean, while seed dispersion is retained in the source table and not drawn in the left panels. Panel c uses a vertically aligned 3+2+2 observation–prediction layout with physically square cells and common 0–1200 axes; the point-density scale is placed independently at the far-right edge. Each PBL panel plots 601 points, whereas each learning-model panel plots the three seed clouds (601 × 3 points). The RMSE and R² annotations show the three-seed mean for learning models; the pooled point cloud is used only for local point-density rendering. The exported composition is a fixed 173 mm × 117 mm canvas, with a compact 1 × 2 layout, a one-line legend in the prescribed method order, and the fixed method palette: MYNN #9A4D8E, YSU #D9903D, MYJ #6BAF77, Sat-ConvLSTM #8CC5DD, WRF-Transformer #42949E, EarlyFusion-Transformer #E9A6A1, and SolFUS-Net #0F4D92.

## Overall common-test metrics

| model | group | n seeds/runs | RMSE (W m⁻²), mean ± seed SD | nRMSE (%), mean ± seed SD | nMBE (%) | R² |
|---|---|---:|---:|---:|---:|---:|
| MYNN (WRF-PBL) | traditional WRF/PBL | 1 | 128.764 (deterministic) | 24.352 (deterministic) | 8.300 | 0.8266 |
| YSU (WRF-PBL) | traditional WRF/PBL | 1 | 128.257 (deterministic) | 24.256 (deterministic) | 6.613 | 0.8280 |
| MYJ (WRF-PBL) | traditional WRF/PBL | 1 | 135.804 (deterministic) | 25.684 (deterministic) | 6.779 | 0.8071 |
| WRF-NWP branch | learned NWP branch | 1 | 126.306 (deterministic) | 23.887 (deterministic) | 4.009 | 0.8331 |
| Sat-ConvLSTM | single-source DL | 3 | 125.653 ± 1.520 | 23.764 ± 0.287 | -0.872 | 0.8349 |
| WRF-Transformer | single-source DL | 3 | 123.371 ± 2.003 | 23.332 ± 0.379 | 2.657 | 0.8408 |
| EarlyFusion-Transformer | conventional fusion | 3 | 116.277 ± 2.511 | 21.991 ± 0.475 | 1.311 | 0.8585 |
| SolFUS-Net | SolFUS-Net | 3 | 109.491 ± 0.603 | 20.707 ± 0.114 | 4.010 | 0.8746 |

## SolFUS relative skill

Positive RMSE skill means lower SolFUS RMSE than the comparator. Exact values are also available in `overall_skill_vs_solfus.csv` and `solfus_skill_by_lead.csv`.

- versus MYNN (WRF-PBL): RMSE skill 14.97% (19.273 W m⁻²); nRMSE reduction 3.645 percentage points; R² gain 0.0480.
- versus YSU (WRF-PBL): RMSE skill 14.63% (18.766 W m⁻²); nRMSE reduction 3.549 percentage points; R² gain 0.0467.
- versus MYJ (WRF-PBL): RMSE skill 19.38% (26.313 W m⁻²); nRMSE reduction 4.976 percentage points; R² gain 0.0675.
- versus WRF-NWP branch: RMSE skill 13.31% (16.815 W m⁻²); nRMSE reduction 3.180 percentage points; R² gain 0.0415.
- versus Sat-ConvLSTM: RMSE skill 12.86% (16.161 W m⁻²); nRMSE reduction 3.056 percentage points; R² gain 0.0398.
- versus WRF-Transformer: RMSE skill 11.25% (13.880 W m⁻²); nRMSE reduction 2.625 percentage points; R² gain 0.0338.
- versus EarlyFusion-Transformer: RMSE skill 5.84% (6.786 W m⁻²); nRMSE reduction 1.283 percentage points; R² gain 0.0161.

## Mechanism and failure-mode diagnostic

The frozen SolFUS output exposes a sample-adaptive satellite weight. Its mean weight is larger at short leads and smaller at long leads, indicating a lead-aware shift from the rapidly observed satellite signal toward the NWP trajectory. This is a descriptive check of the emitted predictions, not a post-hoc row-selection rule.

The overall MAE and FS panels should be interpreted together with the density panels: the former summarize absolute error and skill against smart persistence, while the observation–prediction clouds show where RMSE and R² are accumulated across the full daylight sample. SolFUS is strongest at early and intermediate leads in the lead-wise source metrics, whereas the conventional fusion comparator can be competitive or better at the latest leads. The direct PBL schemes provide an independent physical-model reference rather than another learned branch. The radiation-variation audit provides the corresponding scene interpretation: stable trajectories can favor specialist models, while transitional and complex k* trajectories are where SolFUS is more consistently competitive. The article should therefore claim robust full-test and changing-radiation benefits, not universal dominance in every lead or weather regime.


## 17-cycle paired skill

Cycle skill is computed from matched cycle RMSEs. For each learned baseline, the baseline and SolFUS are paired by seed before averaging; for deterministic PBL and WRF-NWP branch references, the deterministic baseline is compared with each SolFUS seed and the reported skill uses the mean SolFUS RMSE. The cycle count is fixed at 17 for every comparator.

| comparator | wins | win fraction | median skill (%) | interquartile range (%) |
|---|---:|---:|---:|---:|
| MYNN (WRF-PBL) | 14/17 | 82.4% | 16.47 | [8.33, 29.37] |
| YSU (WRF-PBL) | 13/17 | 76.5% | 20.95 | [2.99, 37.32] |
| MYJ (WRF-PBL) | 14/17 | 82.4% | 31.95 | [8.73, 37.59] |
| WRF-NWP branch | 13/17 | 76.5% | 8.93 | [1.88, 28.60] |
| Sat-ConvLSTM | 14/17 | 82.4% | 14.52 | [4.11, 34.61] |
| WRF-Transformer | 11/17 | 64.7% | 13.20 | [-6.39, 27.05] |
| EarlyFusion-Transformer | 12/17 | 70.6% | 7.45 | [-1.43, 15.66] |

## Observed radiation-variation groups

To provide a prediction-independent scene diagnostic, each sample was classified using observed k* = target / Haurwitz clear-sky GHI from the frozen test cache. Total variation thresholds were fixed before comparing model errors: stable < 0.20, transitional 0.20–<0.60, complex ≥ 0.60. Three one-lead common samples were left unassigned because a trajectory variation cannot be computed. No SolFUS prediction or error enters the grouping.

- stable: 46 sample trajectories, 251 common daylight rows.
  - versus MYNN (WRF-PBL): RMSE skill 9.09%.
  - versus YSU (WRF-PBL): RMSE skill 9.63%.
  - versus MYJ (WRF-PBL): RMSE skill 21.39%.
  - versus WRF-NWP branch: RMSE skill 5.16%.
  - versus Sat-ConvLSTM: RMSE skill -16.12%.
  - versus WRF-Transformer: RMSE skill -4.20%.
  - versus EarlyFusion-Transformer: RMSE skill -5.57%.
- transitional: 36 sample trajectories, 200 common daylight rows.
  - versus MYNN (WRF-PBL): RMSE skill 29.86%.
  - versus YSU (WRF-PBL): RMSE skill 31.48%.
  - versus MYJ (WRF-PBL): RMSE skill 32.84%.
  - versus WRF-NWP branch: RMSE skill 20.88%.
  - versus Sat-ConvLSTM: RMSE skill 26.33%.
  - versus WRF-Transformer: RMSE skill 24.63%.
  - versus EarlyFusion-Transformer: RMSE skill 6.95%.
- complex: 25 sample trajectories, 147 common daylight rows.
  - versus MYNN (WRF-PBL): RMSE skill 9.41%.
  - versus YSU (WRF-PBL): RMSE skill 6.76%.
  - versus MYJ (WRF-PBL): RMSE skill 9.57%.
  - versus WRF-NWP branch: RMSE skill 13.78%.
  - versus Sat-ConvLSTM: RMSE skill 15.84%.
  - versus WRF-Transformer: RMSE skill 10.78%.
  - versus EarlyFusion-Transformer: RMSE skill 9.74%.

## Files

- `fig43_fulltest_model_comparison.svg/.pdf/.tiff/.png`: compact two-zone figure with overall MAE and FS bars at left and seven observation–prediction density panels at right.
- `figure43_plot_data.csv`: exact seven-method summary values and point counts used in the figure, including three-seed source dispersion columns.
- `source_data_common_predictions.csv`: all common matched prediction rows used by the analysis, including MYNN, YSU, and MYJ.
- `overall_aggregate_metrics.csv`, `metrics_by_lead.csv`, `overall_skill_vs_solfus.csv`: quantitative source tables for all methods.
- `cycle_skill.csv`, `cycle_skill_summary.csv`, `cycle_weather_context.csv`: all 17-cycle paired skill values and observed-scene context.
- `radiation_variation_metrics.csv`, `radiation_variation_skill_vs_solfus.csv`: prediction-independent scene-group diagnostics.
- `solfus_weight_by_lead.csv`: emitted lead-wise satellite-weight diagnostic.
- `audit.json`, `key_audit.csv`, `target_consistency_audit.csv`, `common_exclusion_audit.csv`: reproducibility and integrity checks, including PBL key/label auditing.
- `analyze_and_plot_fulltest.py`: source script.
