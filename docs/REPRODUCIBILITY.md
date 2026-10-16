# Reproducibility

## Supported reproduction levels

| Level | Supported by this snapshot | Limitation |
| --- | --- | --- |
| Recalculate reported metrics | Yes, from released prediction and reference tables | Must retain each metric's original evaluation cohort |
| Re-evaluate fusion heads | Verified from released feature caches and head checkpoints | Starts after pretrained satellite/NWP feature extraction |
| Inspect or rerun AOD correction | Feature caches, weights, and historical code are included | The quick verification command checks fusion heads; AOD metrics are checked from released tables |
| Retrain heads on cached representations | Code and cached features are provided | Historical scripts may retain original-host defaults; reproducibility depends on the exact archived implementation and configuration |
| Infer from raw satellite/NWP inputs or retrain complete model | No | Complete inputs and the trained backbone checkpoint remain on the remote training server and will be added in a subsequent update |

## Verification commands

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

Several scripts preserve absolute Windows or original GPU-host paths. To pursue new training, first obtain the input tensors and backbone checkpoint pending upload from the remote training server, or train a replacement backbone, then adapt dataset roots, split paths, output roots, and device options using each script's arguments. Such a new run is not the same as reproducing the frozen release. Existing audit files preserve historical provenance and are not current instructions to access a remote host.

