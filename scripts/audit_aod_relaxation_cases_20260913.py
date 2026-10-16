"""Ground-observation audit of all test cases admitted by the fixed gate grid."""
from collections import defaultdict
from datetime import datetime, timedelta
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.run_aod_hardgate_relaxation_20260913 import OUT, write_csv
from scripts.validate_aod_event_windows_20260913 import hour_qc
from scripts.build_gfs_swradnet_dataset import load_stations, load_observation_minutes, aggregate_hour
from scripts.run_aod_7eval_20260912 import save


def main():
    cases = json.loads((OUT / 'test_cases.json').read_text(encoding='utf-8'))
    unique = {r['sample_id']: r for r in cases}
    c = torch.load(ROOT / 'outputs/aod_cloud_cache_20260913/test.pt', weights_only=False, map_location='cpu')
    indices = {r['sample_id']: i for i, r in enumerate(c['meta'])}
    stations = load_stations(ROOT / 'configs/gfs_swradnet_stations.csv')
    existing = list(csv.DictReader((ROOT / 'outputs/aod_continuous_ground_qc_20260913/hourly_qc.csv').open(encoding='utf-8-sig')))
    prior = {(r['sample_id'], int(r['lead'])): r for r in existing if r['split'] == 'test'}
    groups = defaultdict(list)
    for sid, case in unique.items():
        issue = datetime.fromisoformat(case['issue_time_utc'].replace('Z', '+00:00'))
        i = indices[sid]
        for h in (c['valid'][i] & c['day'][i]).nonzero().flatten().tolist():
            end = issue + timedelta(hours=h+1)
            groups[(case['station_id'], end.strftime('%Y-%m'))].append((sid, h, end))
    rows = []
    for (station, month), records in sorted(groups.items()):
        missing = [rec for rec in records if (rec[0], rec[1]+1) not in prior]
        obs = load_observation_minutes(stations[station], [r[2] for r in records], ROOT / 'data/radiation_observations/raw')
        for sid, h, end in records:
            i = indices[sid]
            old = prior.get((sid, h+1))
            if old:
                q = {k: (old[k] == 'True') for k in ('ground_usable', 'observed_near_clear')}
                q.update({k: float(old[k]) if old[k] else None for k in ('p90_relative_ghi_increment', 'direct_beam_fraction', 'closure_relative_error', 'ground_cached_target_abs_difference')})
            else:
                raw = hour_qc(end-timedelta(hours=1), obs, {}, stations[station])
                q = dict(ground_usable=raw['ground_usable'], observed_near_clear=raw['smooth_sun_visible'],
                         p90_relative_ghi_increment=raw['p90_relative_ghi_increment'], direct_beam_fraction=raw['direct_beam_fraction'],
                         closure_relative_error=raw['closure_relative_error'],
                         ground_cached_target_abs_difference=abs(raw['mean_ghi']-float(c['y'][i,h])) if raw['mean_ghi'] is not None else None)
            original_hour = aggregate_hour(obs, end, 1)
            assert original_hour is not None
            raw_difference = abs(float(original_hour[0][0])-float(c['y'][i,h]))
            rows.append(dict(sample_id=sid, station_id=station, lead=h+1, valid_time_utc=end.isoformat(),
                             cached_target=float(c['y'][i,h]), forecast_cloud=float(c['cloud_fraction'][i,h]),
                             original_hour_aggregation_target_difference=raw_difference, **q))
    write_csv(OUT / 'test_case_ground_qc.csv', rows)
    qlookup = {(r['sample_id'], r['lead']): r for r in rows}
    for r in cases:
        qc = [qlookup[(r['sample_id'], int(h))] for h in r['admitted_leads'].split(',')]
        r['admitted_ground_usable_hours'] = sum(q['ground_usable'] for q in qc)
        r['admitted_observed_near_clear_hours'] = sum(q['observed_near_clear'] for q in qc)
        r['admitted_observed_near_clear_leads'] = ','.join(str(q['lead']) for q in qc if q['observed_near_clear'])
        r['admitted_forecast_cloud_max'] = max(q['forecast_cloud'] for q in qc)
        r['admitted_relative_rmse_reduction_pct'] = 100*(1-r['admitted_after_rmse']/r['admitted_before_rmse'])
        r['whole_case_relative_rmse_reduction_pct'] = 100*(1-r['whole_case_after_rmse']/r['whole_case_before_rmse'])
    save(OUT / 'test_cases_with_ground_qc.json', cases)
    write_csv(OUT / 'test_cases_with_ground_qc.csv', cases)

    # Validate numerical reproduction of the existing unmodified hard-gate run.
    old = ROOT / 'outputs/aod_original_hardgate_cases_20260913'
    diffs = []
    for seed in (42,43,44):
        for split in ('validation','test'):
            before = list(csv.DictReader((old / f'{split}_seed{seed}.csv').open(encoding='utf-8-sig')))
            now = list(csv.DictReader((OUT / 'd20_c80' / f'{split}_seed{seed}.csv').open(encoding='utf-8-sig')))
            lookup = {(r['sample_id'],r['lead']): r for r in before}
            assert len(lookup) == len(now)
            for r in now:
                o = lookup[(r['sample_id'],r['lead'])]
                assert r['admitted'] == o['admitted']
                diffs.extend(abs(float(r[k])-float(o[k])) for k in ('target','before','after'))
    assert max(diffs) < .002
    mismatches = sum(r['ground_cached_target_abs_difference'] is not None and r['ground_cached_target_abs_difference'] > 1 for r in rows)
    original_target_differences = [r['original_hour_aggregation_target_difference'] for r in rows]
    assert max(original_target_differences) < .002
    admitted_keys = {(r['sample_id'],int(h)) for r in cases for h in r['admitted_leads'].split(',')}
    admitted_mismatches = sum(r['ground_cached_target_abs_difference'] is not None and r['ground_cached_target_abs_difference'] > 1 for r in rows if (r['sample_id'],r['lead']) in admitted_keys)
    assert admitted_mismatches == 0
    save(OUT / 'audit.json', dict(original_run_max_absolute_difference=max(diffs), ground_target_mismatches_over_1_Wm2=mismatches,
        original_aggregation_max_target_difference=max(original_target_differences), admitted_ground_target_mismatches=admitted_mismatches,
        qc_mean_difference_explanation='Three ungated low-sun hours differ because hour_qc averages only minutes with SZA<=85; the original label averages all accepted minute GHI. Reproducing original aggregation matches every target. No admitted hour has this discrepancy.',
        audited_unique_cases=len(unique), audited_valid_hours=len(rows),
        ground_qc_usage='Post-hoc weather interpretation only; never used as an inference gate or a training/selection feature.',
        ground_qc_definition='At least 54 valid minute GHI and 45 valid component minutes; p90 normalized minute increment <=0.02; DNI>=50 fraction>=0.90; component closure relative error<=0.08.',
        interpretation='Observed smooth direct-sun conditions are a near-clear proxy; no causal attribution to aerosol alone. AOD values are forecast inputs, not verified coincident observed aerosol.'))
    print(json.dumps([{k:r[k] for k in ('candidate','station_id','issue_time_utc','admitted_leads','admitted_observed_near_clear_leads','admitted_forecast_cloud_max','admitted_relative_rmse_reduction_pct')} for r in cases if r['candidate'] in ('d15_c80','d10_c70')], indent=2))


if __name__ == '__main__':
    main()
