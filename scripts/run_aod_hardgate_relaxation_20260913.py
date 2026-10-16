"""Bounded exploratory AOD gate sensitivity; select on validation before test."""
from pathlib import Path
import copy
import csv
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import run_aod_7eval_20260912 as a

OUT = ROOT / 'outputs/aod_hardgate_relaxation_v1_20260913'
SEEDS = (42, 43, 44)
CANDIDATES = [dict(name=f'd{int(round(delta*100)):02d}_c{int(round(clear*100)):02d}',
                   anomaly_min=delta, common_clear_min=clear)
              for delta in (.20, .15, .10) for clear in (.80, .75, .70)]


def prepared(cache, candidate):
    a.POLICY.update({k: candidate[k] for k in ('anomaly_min', 'common_clear_min')})
    dataset = []
    for i, meta in enumerate(cache['meta']):
        row = {k: cache[k][i] for k in ('static', 'aod550', 'aod_reference', 'aod_mask', 'air_mass')}
        row.update(sample_id=meta['sample_id'], solar_context=cache['context'][i, :, :8],
                   target=cache['y'][i], target_mask=cache['valid'][i])
        dataset.append(row)
    return a.enrich(cache, dataset)


def applied(result, d, before):
    with torch.no_grad():
        if result['model'] is None:
            return before.clone()
        model = a.Correction(result['dim'])
        model.load_state_dict(result['model'])
        model.eval()
        after = model(d, before)
        assert torch.equal(after[~d['admit']], before[~d['admit']])
        assert torch.all(after <= before + 1e-5)
        return after


def aggregate_seed_metrics(results, scope='all_valid'):
    return {side: {key: float(np.mean([r[scope][side][key] for r in results]))
                   for key in ('mae', 'rmse', 'nrmse_pct', 'nmbe_pct')}
            for side in ('before', 'after')}


def write_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    heads = {s: ROOT / f'outputs/mlp17_7eval_training_20260909/review/dynamic_mlp/mlp_seed{s}/best.pt' for s in SEEDS}
    cache_paths = {s: ROOT / f'outputs/aod_cloud_cache_20260913/{s}.pt' for s in ('train', 'validation', 'test')}
    paths = [*heads.values(), *cache_paths.values()]
    hashes = {str(p): a.digest(p) for p in paths}
    protocol = dict(
        candidates=CANDIDATES, original_policy=copy.deepcopy(a.POLICY), seeds=SEEDS,
        scope='Exploratory sensitivity on an already inspected test split; not a fresh confirmatory test.',
        feature_coverage_audit='Test gate-feature coverage was inspected before defining this bounded grid; test errors were not used for candidate selection.',
        training='Frozen DynamicMLP; independently refit original AOD Correction for each gate and seed; unchanged optimizer, loss, epochs and patience.',
        checkpoint_selection='Original per-candidate validation admitted-hour MAE, including epoch-zero no correction.',
        candidate_selection='Minimize mean of per-seed MSE on the SAME full validation cohort. Require no seed to worsen full-validation MSE versus no correction (tolerance 1e-8). No-correction fallback. Exact ties prefer higher anomaly threshold, then higher clear threshold.',
        gate='Only anomaly_min and common_clear_min vary. Day fraction, cosSZA, continuous support and correction bounds unchanged.',
        reporting='Every predeclared candidate reported, metrics calculated separately per seed then averaged. Unique cases/hours not multiplied by seed count.',
        sha256=hashes, started_unix=time.time())
    a.save(OUT / 'protocol.json', protocol)
    ref = a.load_module(ROOT / 'scripts/run_query_fusion_pilot_20260908.py')
    ckpts = {s: torch.load(p, map_location='cpu', weights_only=False) for s, p in heads.items()}
    cache = {s: torch.load(cache_paths[s], map_location='cpu', weights_only=False) for s in ('train', 'validation')}
    baseline = {split: {s: a.predict(c, ckpts[s], ref) for s in SEEDS} for split, c in cache.items()}
    fitted = {}
    development = {}
    for candidate in CANDIDATES:
        name = candidate['name']
        out = OUT / name
        out.mkdir()
        data = {split: prepared(c, candidate) for split, c in cache.items()}
        cov = {split: a.coverage(d) for split, d in data.items()}
        fitted[name] = {}
        vals = []
        mse_gain = []
        for seed in SEEDS:
            result = a.fit(seed, data['train'], data['validation'], baseline['train'][seed], baseline['validation'][seed], out)
            result['policy'] = copy.deepcopy(result['policy'])
            fitted[name][seed] = result
            metrics = a.assess(result, data['validation'], baseline['validation'][seed], out / f'validation_seed{seed}.csv')
            vals.append(metrics)
            mse_gain.append(metrics['all_valid']['before']['rmse'] ** 2 - metrics['all_valid']['after']['rmse'] ** 2)
        development[name] = dict(candidate=candidate, coverage=cov, metrics_by_seed=dict(zip(SEEDS, vals)),
                                 aggregate=aggregate_seed_metrics(vals), validation_mse_gain_by_seed=mse_gain,
                                 selection_score=float(np.mean(mse_gain)), eligible=bool(min(mse_gain) >= -1e-8),
                                 best_epochs={s: fitted[name][s]['best_epoch'] for s in SEEDS})
        a.save(out / 'development.json', development[name])
        print('DEVELOPMENT', name, json.dumps({k:development[name][k] for k in ('coverage', 'selection_score', 'eligible', 'best_epochs')}), flush=True)
    eligible = [v for v in development.values() if v['eligible'] and v['selection_score'] > 1e-8]
    chosen = max(eligible, key=lambda v: (v['selection_score'], v['candidate']['anomaly_min'], v['candidate']['common_clear_min']))['candidate']['name'] if eligible else 'no_correction'
    selection = dict(selected=chosen, frozen_unix=time.time(), development=development,
                     test_predictions_not_evaluated_yet=True)
    a.save(OUT / 'selection_frozen.json', selection)
    print('FROZEN_SELECTION', chosen, flush=True)

    # Only now calculate test forecasts/errors for every fixed sensitivity candidate.
    test = torch.load(cache_paths['test'], map_location='cpu', weights_only=False)
    before = {s: a.predict(test, ckpts[s], ref) for s in SEEDS}
    assignments = list(csv.DictReader((ROOT / 'outputs/dataset_split_17train_7eval_expanded_20260909/case_assignments.csv').open(encoding='utf-8-sig')))
    lookup = {(r['station_id'], r['cycle']): r for r in assignments}
    summary = []
    cases = []
    case_seed_rows = []
    for candidate in CANDIDATES:
        name = candidate['name']
        out = OUT / name
        d = prepared(test, candidate)
        cov = a.coverage(d)
        after = {s: applied(fitted[name][s], d, before[s]) for s in SEEDS}
        metrics = [a.assess(fitted[name][s], d, before[s], out / f'test_seed{s}.csv') for s in SEEDS]
        agg = aggregate_seed_metrics(metrics)
        row = dict(candidate=name, selected=name == chosen, **candidate, **cov,
                   validation_mse_gain=development[name]['selection_score'],
                   validation_before_nrmse=development[name]['aggregate']['before']['nrmse_pct'],
                   validation_after_nrmse=development[name]['aggregate']['after']['nrmse_pct'],
                   test_before_nrmse=agg['before']['nrmse_pct'], test_after_nrmse=agg['after']['nrmse_pct'],
                   test_before_mae=agg['before']['mae'], test_after_mae=agg['after']['mae'])
        row.pop('name')
        summary.append(row)
        a.save(out / 'test_summary.json', dict(coverage=cov, metrics_by_seed=dict(zip(SEEDS, metrics)), aggregate=agg))
        for i in (d['valid'] & d['admit']).any(-1).nonzero().flatten().tolist():
            meta = d['meta'][i]
            assn = lookup[(meta['station_id'], meta['cycle'])]
            assert assn['split'] == 'test', assn
            mask = d['valid'][i] & d['admit'][i]
            r = dict(candidate=name, selected=name == chosen, **meta, issue_time_utc=assn['issue_time_utc'],
                     valid_hours=int(d['valid'][i].sum()), admitted_hours=int(mask.sum()),
                     admitted_leads=','.join(str(h+1) for h in mask.nonzero().flatten().tolist()),
                     min_aod=float(d['aod550'][i][mask].min()), max_aod=float(d['aod550'][i][mask].max()),
                     min_anomaly=float(d['anomaly'][i][mask].min()), max_anomaly=float(d['anomaly'][i][mask].max()),
                     min_clear=float(d['common_clear'][i][mask].min()), max_clear=float(d['common_clear'][i][mask].max()))
            seed_rows = []
            for s in SEEDS:
                rs = dict(r, seed=s)
                for scope, m in [('admitted', mask), ('whole_case', d['valid'][i])]:
                    for side, p in [('before', before[s][i]), ('after', after[s][i])]:
                        rs.update({f'{scope}_{side}_{k}': v for k, v in a.metric(d['y'][i], p, m).items() if k != 'n'})
                rs['mean_adjustment'] = float((before[s][i]-after[s][i])[mask].mean())
                rs['max_adjustment'] = float((before[s][i]-after[s][i])[mask].max())
                rs['improved_hours'] = int(((after[s][i]-d['y'][i]).abs()[mask] < (before[s][i]-d['y'][i]).abs()[mask]-1e-6).sum())
                rs['worsened_hours'] = int(((after[s][i]-d['y'][i]).abs()[mask] > (before[s][i]-d['y'][i]).abs()[mask]+1e-6).sum())
                seed_rows.append(rs)
            case_seed_rows.extend(seed_rows)
            for k in seed_rows[0]:
                if k not in r and k != 'seed':
                    r[k] = float(np.mean([rs[k] for rs in seed_rows]))
            r['seeds_improving_admitted_rmse'] = sum(rs['admitted_after_rmse'] < rs['admitted_before_rmse']-1e-6 for rs in seed_rows)
            r['seeds_worsening_admitted_rmse'] = sum(rs['admitted_after_rmse'] > rs['admitted_before_rmse']+1e-6 for rs in seed_rows)
            cases.append(r)
        print('TEST', name, json.dumps(row), flush=True)
    a.save(OUT / 'summary.json', summary)
    a.save(OUT / 'test_cases.json', cases)
    write_csv(OUT / 'summary.csv', summary)
    write_csv(OUT / 'test_cases.csv', cases)
    write_csv(OUT / 'test_cases_by_seed.csv', case_seed_rows)
    assert all(a.digest(Path(p)) == h for p, h in hashes.items())
    a.save(OUT / 'complete.json', dict(complete=True, selected=chosen, sha256_verified=True,
            all_predeclared_candidates_reported=True, test_split_unchanged=True, finished_unix=time.time()))


if __name__ == '__main__':
    main()
