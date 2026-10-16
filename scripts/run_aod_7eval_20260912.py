"""Fit AOD correction on frozen current MLP caches; select only on validation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.evaluate_mlp17_7eval_test import load_module

POLICY = dict(anomaly_min=0.20, common_clear_min=0.80, day_fraction_min=0.99,
              mean_cos_sza_min=0.25, epochs=24, patience=8, learning_rate=0.001,
              selection='validation admitted-hour MAE; epoch zero is no correction',
              batch_cases=4, loss='admitted-hour MAE', comparator='frozen current dynamic MLP',
              seed_reporting='mean of separately calculated seed metrics',
              background='current training rows only; station-month median with fallbacks')


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def metric(y, p, mask):
    y, p = y[mask].double(), p[mask].double()
    if not y.numel():
        return dict(n=0, mae=None, rmse=None, nrmse_pct=None, nmbe_pct=None)
    rmse = (p - y).square().mean().sqrt().item()
    mean = y.mean().item()
    return dict(n=y.numel(), mae=(p-y).abs().mean().item(), rmse=rmse,
                nrmse_pct=100 * rmse / mean if mean > 0 else None,
                nmbe_pct=100 * (p - y).mean().item() / mean if mean > 0 else None)


class Correction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.strength = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.)
        nn.init.zeros_(self.strength[-1].weight)
        nn.init.zeros_(self.strength[-1].bias)

    def forward(self, d, before):
        gate = torch.sigmoid(self.gate(d['x']).squeeze(-1))
        gate = gate * d['support']
        beta = torch.sigmoid(self.strength(d['x']).squeeze(-1))
        limit = math.log(1.3)
        delta = limit * torch.tanh(-beta * d['air_mass'] * d['anomaly'] / limit)
        # Admission requires positive AOD anomaly: bounded_update is q*exp(delta).
        # Applied directly to frozen MLP GHI, with no legacy transition/router output.
        after = before * torch.exp(gate * delta)
        return torch.where(d['admit'], after, before)


def enrich(cache, dataset):
    raw = {item['sample_id']: item for item in dataset}
    if set(raw) != {m['sample_id'] for m in cache['meta']}:
        raise ValueError('dataset/cache sample IDs differ')
    items = [raw[m['sample_id']] for m in cache['meta']]
    keys = ['static', 'solar_context', 'aod550', 'aod_reference', 'aod_mask', 'air_mass']
    b = {k: torch.stack([v[k] for v in items]).float() for k in keys}
    target = torch.stack([v['target'] for v in items])
    valid = torch.stack([v['target_mask'] for v in items]).bool()
    if not torch.equal(valid, cache['valid']) or not torch.allclose(target[valid], cache['y'][valid]):
        raise ValueError('labels/cache differ')
    if not torch.allclose(b['solar_context'], cache['context'][..., :b['solar_context'].shape[-1]]):
        raise ValueError('solar context/cache differ')
    ns, nnwp = cache['s'].shape[-1] - 8, cache['n'].shape[-1] - 5
    sq, nq = cache['s'][..., ns], cache['n'][..., nnwp]
    clear = torch.minimum(cache['s'][..., ns + 2].sigmoid(), cache['n'][..., nnwp + 2].sigmoid())
    anomaly = b['aod550'] - b['aod_reference']
    solar = b['solar_context']
    admit = (b['aod_mask'].bool() & torch.isfinite(anomaly) & torch.isfinite(b['air_mass'])
             & (anomaly >= POLICY['anomaly_min']) & (clear >= POLICY['common_clear_min'])
             & (solar[..., 2] > POLICY['day_fraction_min'])
             & (solar[..., 0] >= POLICY['mean_cos_sza_min']))
    anomaly = torch.nan_to_num(anomaly)
    air_mass = torch.nan_to_num(b['air_mass'])
    x = torch.cat([cache['s'][..., :ns], cache['n'][..., :nnwp],
                   torch.stack([sq, nq, (sq - nq).abs()], -1), solar,
                   b['static'][:, None, :].expand(-1, 6, -1),
                   clear[..., None], anomaly[..., None], air_mass[..., None]], -1)
    if not torch.isfinite(x).all():
        raise ValueError('non-finite AOD input')
    return dict(x=x, admit=admit, anomaly=anomaly, air_mass=air_mass,
                support=((solar[..., 2] - .8) / .2).clamp(0, 1) * torch.sigmoid((clear - .70) / .08),
                aod550=b['aod550'], aod_reference=b['aod_reference'], common_clear=clear,
                y=cache['y'], valid=cache['valid'] & cache['day'], meta=cache['meta'])


def coverage(d):
    mask = d['admit'] & d['valid']
    counts = Counter(d['meta'][i]['station_id'] for i, _h in mask.nonzero().tolist())
    return dict(admitted_hours=int(mask.sum()), admitted_cases=int(mask.any(-1).sum()),
                valid_hours=int(d['valid'].sum()), by_station=dict(sorted(counts.items())))


@torch.no_grad()
def predict(cache, checkpoint, reference):
    model = reference.DynamicFusion(checkpoint['dims'], checkpoint['initial_weights'], 'mlp')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    d = dict(cache)
    for key, value in checkpoint['normalization'].items():
        d[key] = (d[key] - torch.tensor(value['mean']).reshape(1, 1, -1)) / torch.tensor(value['std']).reshape(1, 1, -1)
    return model(d)[0].detach()


def verify_existing_prediction(path, cache, prediction):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    lookup = {m['sample_id']: i for i, m in enumerate(cache['meta'])}
    errors = [abs(float(r['prediction']) - float(prediction[lookup[r['sample_id']], int(r['lead']) - 1])) for r in rows]
    if not errors or max(errors) > 0.002:
        raise ValueError(f'frozen prediction mismatch: {path}, max error {max(errors, default=None)}')
    return max(errors)


def fit(seed, train, val, before_train, before_val, out):
    torch.manual_seed(seed)
    model = Correction(train['x'].shape[-1])
    trmask, vamask = train['admit'] & train['valid'], val['admit'] & val['valid']
    before_metrics = metric(val['y'], before_val, vamask)
    history = [dict(epoch=0, validation=before_metrics)]
    best, best_epoch, state, stale = before_metrics['mae'], 0, None, 0
    if trmask.any() and vamask.any():
        opt = torch.optim.AdamW(model.parameters(), lr=POLICY['learning_rate'], weight_decay=1e-4)
        eligible = trmask.any(-1).nonzero().flatten()
        for epoch in range(1, POLICY['epochs'] + 1):
            model.train()
            order = eligible[torch.randperm(len(eligible))]
            loss_sum, points = 0., 0
            for ids in order.split(POLICY['batch_cases']):
                batch = {k: v[ids] for k, v in train.items() if isinstance(v, torch.Tensor)}
                mask = trmask[ids]
                opt.zero_grad()
                p = model(batch, before_train[ids])
                loss = ((p[mask] - batch['y'][mask]) / 100).abs().mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                loss_sum += float(loss.detach()) * int(mask.sum())
                points += int(mask.sum())
            model.eval()
            with torch.no_grad():
                m = metric(val['y'], model(val, before_val), vamask)
            history.append(dict(epoch=epoch, train_loss=loss_sum / points, validation=m))
            if m['mae'] < best - 1e-5:
                best, best_epoch, state, stale = m['mae'], epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= POLICY['patience']:
                break
    reason = ('trained' if state is not None else 'no_validation_improvement')
    if not trmask.any() or not vamask.any():
        reason = 'insufficient_admitted_training_or_validation_data'
    result = dict(seed=seed, best_epoch=best_epoch, status=reason, policy=POLICY,
                  dim=train['x'].shape[-1], model=state)
    torch.save(result, out / f'aod_seed{seed}.pt')
    save(out / f'history_seed{seed}.json', history)
    return result


@torch.no_grad()
def assess(result, d, before, out):
    after = before.clone()
    if result['model'] is not None:
        model = Correction(result['dim'])
        model.load_state_dict(result['model'])
        model.eval()
        after = model(d, before)
    assert torch.equal(before[~d['admit']], after[~d['admit']])
    assert torch.all(after <= before + 1e-5)
    summary = {}
    for group, mask in [('all_valid', d['valid']), ('admitted', d['valid'] & d['admit'])]:
        summary[group] = dict(before=metric(d['y'], before, mask), after=metric(d['y'], after, mask))
    rows = []
    for i, h in d['valid'].nonzero().tolist():
        rows.append(dict(**d['meta'][i], lead=h + 1, target=float(d['y'][i, h]),
                         before=float(before[i, h]), after=float(after[i, h]),
                         admitted=bool(d['admit'][i, h]), aod550=float(d['aod550'][i, h]),
                         aod_reference=float(d['aod_reference'][i, h]),
                         common_clear=float(d['common_clear'][i, h])))
    if rows:
        with out.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=ROOT / 'reference/run_query_fusion_pilot_20260908.py')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out, run, root = args.output_dir, args.run_root, args.dataset_root
    save(out / 'protocol.json', POLICY)
    from ghi_hierarchical_v4.data import MultisourceCaseDataset, read_index
    from scripts.prepare_combined_1462_dataset import aod_reference
    from scripts.train_hierarchical_799_v4 import select_station_aware_partitions
    config = json.loads((root / 'split_config.json').read_text(encoding='utf-8'))
    cycles = config['split_cycles']
    groups = select_station_aware_partitions(read_index(root), set(cycles['train']), set(cycles['validation']),
        set(cycles['test']), set(config['evaluation_stations']), set(config['training_stations']))
    if tuple(map(len, groups)) != (1224, 119, 119):
        raise ValueError('unexpected current split')
    normalization = json.loads((run / 'base_normalization.json').read_text())
    background = aod_reference(root, groups[0])
    old_background = json.loads((root / 'statistics/fold_0_aod_reference.json').read_text())
    if background != old_background:
        raise ValueError('current training-only AOD background differs from existing reference')
    save(out / 'aod_reference.json', background)
    reference = load_module(args.reference)
    cache_paths = dict(train=run / 'cache/train_cache.pt', validation=run / 'cache/validation_cache.pt',
                       test=run / 'test_7station/test_cache.pt')
    cache, data = {}, {}
    def prepare(split, rows):
        cache[split] = torch.load(cache_paths[split], map_location='cpu', weights_only=False)
        dataset = MultisourceCaseDataset(root, rows, normalization, background,
                    include_future_satellite_auxiliary=False, require_formal_v4=True)
        data[split] = enrich(cache[split], dataset)
        save(out / f'coverage_{split}.json', coverage(data[split]))
        print(split, coverage(data[split]), flush=True)
    for split, rows in zip(('train', 'validation'), groups[:2]):
        prepare(split, rows)
    head_paths = {s: run / f'dynamic_mlp/mlp_seed{s}/best.pt' for s in (42, 43, 44)}
    hashes = {str(p): digest(p) for p in [*head_paths.values(), *cache_paths.values()]}
    fitted = {}
    for seed, path in head_paths.items():
        head = torch.load(path, map_location='cpu', weights_only=False)
        ptr, pva = (predict(cache[k], head, reference) for k in ('train', 'validation'))
        verify_existing_prediction(run / f'dynamic_mlp/mlp_seed{seed}/validation.csv', cache['validation'], pva)
        fitted[seed] = fit(seed, data['train'], data['validation'], ptr, pva, out)
    save(out / 'selection_frozen.json', {s: {k: v for k, v in r.items() if k != 'model'} for s, r in fitted.items()})
    # Test features/labels are loaded only after all checkpoint choices are fixed.
    prepare('test', groups[2])
    summary = {}
    for seed, path in head_paths.items():
        head = torch.load(path, map_location='cpu', weights_only=False)
        before = predict(cache['test'], head, reference)
        verify_existing_prediction(run / f'test_7station/mlp_seed{seed}_test.csv', cache['test'], before)
        summary[seed] = assess(fitted[seed], data['test'], before, out / f'test_seed{seed}.csv')
    save(out / 'test_summary.json', summary)
    for path, checksum in hashes.items():
        if digest(Path(path)) != checksum:
            raise RuntimeError(f'frozen file changed: {path}')
    save(out / 'complete.json', dict(state='complete', frozen_verified=True, sha256=hashes,
         test_coverage=coverage(data['test']), aggregate='calculate metrics per seed, then average'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
