# 4.4 selected case-study candidates

This directory contains the three cases fixed by the section-4.4 plan. The
script only extracts the named IDs; it does not select cases using model error
and it does not create the final figure.

## Inputs

- Predictions: `results/fulltest_model_comparison_20260921/source_data_common_predictions.csv`.
- Clear-sky and target audit: `outputs/mlp17_7eval_training_20260909/review/test_7station/test_cache.pt`.
- `clear_sky = test_cache.pt['context'][i, lead-1, 1] * 1000`, in W m^-2.
- The cache `y` is the reference target. Every selected source target matches
  it within `1e-5 W m^-2`; all six selected leads are valid and daylight.

## Cases

1. `development__TBL__2023090200__r18`: cloud enhancement / abrupt radiation drop.
2. `val__CAB__2024012000__r57`: cloud dissipation / rapid transmittance recovery.
3. `val__SEL__2024012000__r15`: intermittent cloud change / repeated fluctuation.

## Definitions and checks

- `k_star = target / clear_sky`.
- `total_variation_k_star (TV) = sum(abs(diff(k_star)))` across the six leads.
- `net_change_k_star = k_star[lead 6] - k_star[lead 1]`.
- `sign_changes` counts adjacent sign changes in `diff(k_star)` (zero is kept
  as its own sign, matching the frozen full-test audit convention).
- SolFUS seed means and SDs use seeds 42, 43, and 44; SD is sample SD
  (`ddof=1`). `nwp_weight = 1 - satellite_weight_mean`, with per-seed NWP
  weights also retained.
- Trajectory RMSEs are over the six leads in W m^-2. Relative improvement is
  `100 * (branch_RMSE - SolFUS_RMSE) / branch_RMSE`; positive means SolFUS is
  lower error.
- For each seed, the script verifies
  `SolFUS = satellite_weight * satellite + (1 - satellite_weight) * nwp`
  within `1e-4 W m^-2`.

Outputs:

- `selected_case_sequences.csv`: one row per case and lead, including targets,
  clear-sky values, `k_star`, all branch/seed values, weights, absolute errors,
  and cache/blend audit fields.
- `selected_case_summary.csv`: one row per case with trajectory variability,
  branch/SolFUS RMSEs, and SolFUS relative improvement versus satellite and NWP.
