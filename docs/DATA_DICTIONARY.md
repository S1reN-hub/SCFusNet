# Data dictionary

This guide covers the principal released manifests, derived tables, and feature caches. Individual CSV headers and saved cache metadata remain the authoritative field lists. GHI forecasts and observations are in W m⁻² unless a field explicitly denotes a normalized value or percentage.

## Keys and split manifests

Manifests are under `outputs/dataset_split_17train_7eval_expanded_20260909/`.

| Field | Meaning |
| --- | --- |
| `source_sample_id` / `sample_id` | Historical sample identifier; its embedded old split prefix is provenance only |
| `case_key` | Station/cycle/issue-offset case identifier independent of old split naming |
| `station_id` | Station code; seven evaluation stations are GOB, DRA, IZA, TBL, SEL, GWN, CAB |
| `cycle` | Forecast-cycle identifier formatted as `YYYYMMDDHH`; preserve as text |
| `issue_offset_hours` | Case initialization offset from the source forecast cycle |
| `issue_time_utc` | Forecast initialization timestamp in UTC |
| `sample_end_time_utc` | End of the six-hour target window in UTC |
| `split` | Actual train/validation/test assignment for this experiment |
| `evaluation_station` | Whether the station belongs to the seven-station evaluation group |
| `evaluation_cycle_partition` | Cycle partition used for those evaluation stations |
| `geometry_eligible_hours` | Potential daylight/geometry-qualified hours before all ground-observation and persistence checks |
| `source_manifest`, `old_split`, `selection_rule` | Input provenance and assignment rule |

`train_all17.csv`, `validation_7station.csv`, and `test_7station.csv` contain the corresponding subsets. `cycle_assignments.csv`, `split_audit.json`, and related tables document grouping and coverage.

## Common prediction table

`results/fulltest_model_comparison_20260921/source_data_common_predictions.csv` contains 601 rows. Each row is a matched station/case/lead target hour, shared by all compared models.

| Field | Meaning |
| --- | --- |
| `sample_id`, `station_id`, `cycle`, `lead` | Row identity; `lead` is 1–6 hours |
| `_key` | Serialized composite matching key |
| `target` | Observed hourly mean GHI |
| `solfus_seed42/43/44` | SCFusNet dynamic-fusion forecasts; historical prefix retained |
| `solfus_satellite_weight_seed42/43/44` | Satellite contribution weight for each seed and hour |
| `sat_convlstm_seed42/43/44` | Satellite ConvLSTM baseline forecasts |
| `wrf_transformer_seed42/43/44` | Learned WRF Transformer baseline forecasts |
| `early_fusion_seed42/43/44` | Input-level Transformer fusion baseline forecasts (TransFusion in the manuscript) |
| `mynn`, `ysu`, `myj` | Direct WRF forecasts with the corresponding PBL scheme |
| `satellite`, `nwp` | Frozen independently supervised branch forecasts; `nwp` is learned, not a direct PBL forecast |
| `daylight_all_models` | Common daytime-validity flag |

Summary tables retain metric values by seed and their means/standard deviations. A suffix `_pct` denotes percent; `_wm2` denotes W m⁻². Forecast-skill source tables use the separate 452-record persistence intersection, so their counts differ from the common prediction table.

## Frozen branch caches

Caches in the review tree and `outputs/aod_cloud_cache_20260913/` are PyTorch dictionaries. Let `N` be the number of cases, `H=6` the forecast leads, and `D` a saved feature dimension. Inspect actual tensor shapes and any `feature_schema` before reconstructing inputs. Historical recovered and launch snapshots differ; neither is asserted to be an exact complete training snapshot. The released cached verification script checks the tested fusion-head input contract.

| Key | Meaning / usual layout |
| --- | --- |
| `meta` | Length-`N` case metadata, including sample/station identifiers and provenance |
| `s`, `n` | Satellite and NWP branch feature/state tensors, `[N,H,D]`; combine latent state with branch quality and forecast summaries |
| `context` | Target-hour solar/context features and branch disagreement, `[N,H,D]` |
| `prediction` | Frozen satellite/NWP forecasts, `[N,H,2]` |
| `available` | Source-availability masks, `[N,H,2]` |
| `y` | Ground target GHI, `[N,H]` |
| `valid` | Target-validity mask, `[N,H]`; invalid stored target values are not observations |
| `day` | Daytime eligibility mask, `[N,H]` |
| `schema_version`, `feature_schema` | Schema/provenance descriptors where present |
| `history_transport*`, `motion_wind_consistency*` | Additional historical transport/consistency diagnostics and availability masks where present |

Do not assume that every cached diagnostic entered the published fusion head. Source versions differ; use saved checkpoints, feature metadata, and the tested cached verification path to inspect the input contract.

## AOD-enriched caches

The AOD cache adds the following derived fields to the frozen branch data:

| Key | Meaning |
| --- | --- |
| `aod550` | AOD at 550 nm (dimensionless) |
| `aod_reference` | Training-derived station/month background with documented fallbacks |
| `aod_mask` | AOD validity mask |
| `air_mass` | Relative/effective air mass derived from solar geometry |
| `stable_scale` | Saved irradiance scaling/reference used by the original preprocessing |
| `static` | Encoded station attributes |
| `cloud_fraction`, `cloud_valid` | Forecast cloud-overlap proxy and validity from the WRF d02 central 3×3 patch |
| `cloud_column_max` | Maximum sampled cloud fraction in the diagnostic patch/levels |
| `satellite_last_channels` | Central-patch values from the latest satellite input |

The cloud diagnostic uses sampled pressure levels and maximum–random overlap; it is not native full-model-level or subhourly cloud coverage. Its input is a forecast, not a future ground observation.

## AOD sensitivity outputs

`outputs/aod_hardgate_relaxation_v1_20260913/` retains the complete grid.

| Field / file | Meaning |
| --- | --- |
| Candidate `d10_c70` | AOD anomaly ≥0.10 and common clear-sky probability ≥0.70; analogous names encode other grid thresholds |
| `protocol.json` | Grid, fixed gates, validation selection rules, seeds, exploratory status, and hashes |
| `selection_frozen.json` | Selected candidate and validation evidence for all candidates |
| `candidate`, `selected` | Candidate identity and whether it was selected |
| `valid_hours`, `admitted_hours`, `admitted_leads` | Valid and correction-admitted target hours within a case |
| `min/max_aod`, `min/max_anomaly`, `min/max_clear` | Admission-feature ranges |
| `admitted_before_*`, `admitted_after_*` | Metrics restricted to admitted hours before/after correction |
| `whole_case_before_*`, `whole_case_after_*` | Metrics over all valid hours of the case |
| `mean_adjustment`, `max_adjustment` | Reported correction magnitude summaries |
| `test_cases_by_seed.csv` | Per-seed case evidence; unique case/hour counts must not be multiplied by seed count |

Rows in aggregate case tables can contain means of seed-level counts or metrics. Use per-seed tables for exact seed-specific outcomes. The selected policy covers 38 unique test hours in 10 cases; the sensitivity study remains exploratory because the test split had previously been inspected.
