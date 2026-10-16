"""Frozen 17-station expert states: analytic lead baseline, MLP and query fusion.

Run on the GPU host. Only the original development train/validation split is read.
The first prototype attends to six expert forecast-state tokens, not raw images.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

SOURCE = Path('/root/autodl-tmp/guizhou_r3_fusion')
DEFAULT_OUT = Path('/root/autodl-tmp/guizhou_query_fusion_20260908')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_worker(out):
    sys.path.insert(0, str(out / 'code'))
    sys.path.insert(0, str(out / 'code/src'))
    from scripts import train_hierarchical_799_v4 as trainer

    def capture(model, loader, validation_loader, stage, epochs, learning_rate,
                patience, device, output_dir, precision='fp32'):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        snapshots = {}
        handles = [model.satellite_expert.register_forward_hook(lambda m, i, o: snapshots.update(s=o)),
                   model.wrf_expert.register_forward_hook(lambda m, i, o: snapshots.update(n=o))]
        from torch.utils.data import DataLoader
        def extract(original, name):
            deterministic = DataLoader(original.dataset, batch_size=4, shuffle=False,
                                       collate_fn=trainer.numpy_safe_collate, num_workers=0)
            arrays = {k: [] for k in ['s', 'n', 'context', 'prediction', 'available', 'y', 'valid', 'day']}
            meta = []
            with torch.no_grad():
                for raw in deterministic:
                    batch = trainer.to_device(raw, device)
                    output = model(batch)
                    s, n = snapshots['s'], snapshots['n']
                    def feature(expert, names):
                        return torch.cat([expert['state'], *[expert[k].float().unsqueeze(-1) for k in names]], dim=-1)
                    arrays['s'].append(feature(s, ['q', 'scale_q', 'clear_logit', 'valid_fraction',
                        'freshness', 'dynamics', 'motion_confidence', 'unique_frame_fraction']).cpu())
                    arrays['n'].append(feature(n, ['q', 'scale_q', 'clear_logit', 'anchor_trust', 'transport_residual']).cpu())
                    arrays['context'].append(torch.cat([batch['solar_context'], output['motion_wind_consistency'],
                        (s['q']-n['q']).unsqueeze(-1)], dim=-1).cpu())
                    arrays['prediction'].append(torch.stack([output['satellite_prediction'], output['wrf_prediction']], -1).cpu())
                    arrays['available'].append(torch.stack([s['available'], n['available']], -1).bool().cpu())
                    arrays['y'].append(batch['target'].cpu())
                    arrays['valid'].append(batch['target_mask'].bool().cpu())
                    arrays['day'].append((batch['solar_context'][..., 2] > 0).cpu())
                    meta.extend([{'sample_id': sid, 'station_id': raw['station_id'][i], 'cycle': raw['cycle'][i]}
                                 for i, sid in enumerate(raw['sample_id'])])
            result = {k: torch.cat(v) for k, v in arrays.items()}
            result['meta'] = meta
            assert len({r['sample_id'] for r in meta}) == len(meta)
            for key in ['s', 'n', 'context', 'prediction']:
                assert torch.isfinite(result[key]).all(), key
            assert torch.isfinite(result['y'][result['valid']]).all()
            result['y'] = torch.where(result['valid'], result['y'], torch.zeros_like(result['y']))
            torch.save(result, out / f'{name}_cache.pt')
            print('CACHE', name, len(meta), flush=True)
            return {r['cycle'] for r in meta}
        train_cycles = extract(loader, 'train')
        val_cycles = extract(validation_loader, 'validation')
        assert len(train_cycles) == 62 and len(val_cycles) == 11 and not train_cycles & val_cycles
        for h in handles:
            h.remove()
        save(out / 'cache_verified.json', {'passed': True, 'train_cycles': 62, 'validation_cycles': 11,
             'model_eval': True, 'expert_gradients_disabled': True, 'test_accessed': False})
        raise SystemExit(0)
    trainer.train_stage = capture
    command = json.loads((SOURCE/'runs/robust_dual_time_only_seed42/command.json').read_text())
    for flag, value in {'--resume-checkpoint': str(SOURCE/'runs/robust_dual_time_only_seed42/best_fusion.pt'),
                        '--threads': '4'}.items():
        command[command.index(flag)+1] = value
    sys.argv = [__file__] + command[3:] + ['--output-dir', str(out/'cache_runtime')]
    trainer.main()


def fit_time_weights(data):
    y = data['y'].double()
    p = data['prediction'].double()
    both = data['available'].all(-1) & data['valid'] & data['day']
    gap = p[..., 0] - p[..., 1]
    numerator = ((y-p[..., 1])*gap*both).sum(0)
    denominator = (gap.square()*both).sum(0)
    w = torch.where(denominator > 1e-12, numerator/denominator.clamp_min(1e-12), .5).clamp(0, 1)
    # Sigmoid dynamic initialization requires interior weights; stop rather than silently change the baseline.
    assert ((w > 1e-6) & (w < 1-1e-6)).all(), 'Boundary optimum needs explicit handling'
    return w.float()


def final_weights(logits, available):
    weight = torch.softmax(logits.masked_fill(~available, -1e4), -1) * available
    return weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)


class DynamicFusion(nn.Module):
    def __init__(self, dims, initial_weight, kind):
        super().__init__()
        self.kind = kind
        self.s_proj = nn.Sequential(nn.Linear(dims[0], 64), nn.LayerNorm(64), nn.GELU())
        self.n_proj = nn.Sequential(nn.Linear(dims[1], 64), nn.LayerNorm(64), nn.GELU())
        self.query = nn.Sequential(nn.Linear(dims[2], 64), nn.GELU())
        self.position = nn.Parameter(torch.randn(6, 64)*.02)
        self.query_position = nn.Parameter(torch.randn(6, 64)*.02)
        self.lead_logits = nn.Parameter(torch.stack([torch.logit(initial_weight), torch.zeros(6)], -1))
        if kind == 'mlp':
            self.head = nn.Sequential(nn.Linear(12*64+64+12, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 2))
        else:
            self.attention = nn.MultiheadAttention(64, 2, dropout=.1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64*3+12, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, data):
        available = data['available']
        s = (self.s_proj(data['s']) + self.position) * available[..., 0:1]
        n = (self.n_proj(data['n']) + self.position) * available[..., 1:2]
        query = self.query(data['context']) + self.query_position
        mask_features = available.float().reshape(len(s), 12).unsqueeze(1).expand(-1, 6, -1)
        if self.kind == 'mlp':
            features = torch.cat([s, n], 1).flatten(1).unsqueeze(1).expand(-1, 6, -1)
            features = torch.cat([features, query, mask_features], -1)
        else:
            def read(tokens, present):
                missing_all = ~present.any(1)
                safe_mask = ~present.clone()
                safe_mask[missing_all, 0] = False
                value, _ = self.attention(query, tokens, tokens, key_padding_mask=safe_mask, need_weights=False)
                return value.masked_fill(missing_all[:, None, None], 0)
            features = torch.cat([read(s, available[..., 0]), read(n, available[..., 1]), query, mask_features], -1)
        weight = final_weights(self.head(features) + self.lead_logits, available)
        prediction = (weight * data['prediction']).sum(-1)
        return prediction, weight


def subset(data, ids):
    return {k: v[ids] for k, v in data.items() if isinstance(v, torch.Tensor)}


def metrics(y, prediction, valid, day, meta):
    error = prediction.double() - y.double()
    def calc(mask):
        if not mask.any():
            return {'count': 0, 'rmse': None, 'mae': None, 'mean_observed_wm2': None, 'nrmse_pct': None}
        e = error[mask]
        rmse = float(e.square().mean().sqrt())
        mean = float(y.double()[mask].mean())
        return {'count': int(mask.sum()), 'rmse': rmse, 'mae': float(e.abs().mean()),
                'mean_observed_wm2': mean, 'nrmse_pct': 100*rmse/mean if mean > 0 else None}
    daylight = valid & day
    result = {'all': calc(valid), 'daylight': calc(daylight)}
    result['by_lead'] = {}
    for h in range(6):
        mask = torch.zeros_like(valid); mask[:, h] = daylight[:, h]
        result['by_lead'][str(h+1)] = calc(mask)
    for field in ('station_id', 'cycle'):
        result['by_'+field] = {}
        for key in sorted({r[field] for r in meta}):
            take = torch.tensor([r[field] == key for r in meta], device=valid.device)
            result['by_'+field][key] = calc(daylight & take[:, None])
    result['macro_station_daylight_rmse'] = float(np.mean([x['rmse'] for x in result['by_station_id'].values() if x['count']]))
    normalized = [x['nrmse_pct'] for x in result['by_station_id'].values()]
    result['macro_station_daylight_nrmse_pct'] = float(np.mean(normalized)) if normalized and all(x is not None for x in normalized) else None
    return result


def export(path, data, prediction, weight):
    p, w = prediction.detach().cpu(), weight.detach().cpu()
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id','station_id','cycle','lead','daylight','target','satellite','nwp','prediction','satellite_weight'])
        for i, row in enumerate(data['meta']):
            for h in range(6):
                if data['valid'][i,h]:
                    writer.writerow([row['sample_id'],row['station_id'],row['cycle'],h+1,
                        int(data['day'][i,h]),float(data['y'][i,h]),float(data['prediction'][i,h,0]),
                        float(data['prediction'][i,h,1]),float(p[i,h]),float(w[i,h,0])])


def run_training(out):
    torch.set_num_threads(4)
    device = torch.device('cuda')
    train = torch.load(out/'train_cache.pt', weights_only=False)
    val = torch.load(out/'validation_cache.pt', weights_only=False)
    assert len(train['meta']) == 1054 and len(val['meta']) == 187
    # Apply the current evaluation domain before feature standardization.
    for data in (train, val):
        data['day'] = data['context'][..., 0] >= float(np.cos(np.deg2rad(85.0)))
    initial = fit_time_weights(train)
    normalization = {}
    for key in ['s', 'n', 'context']:
        mean = train[key].mean((0,1), keepdim=True)
        std = train[key].std((0,1), keepdim=True).clamp_min(.01)
        normalization[key] = {'mean': mean.flatten().tolist(), 'std': std.flatten().tolist()}
        train[key] = (train[key]-mean)/std
        val[key] = (val[key]-mean)/std
    save(out/'normalization.json', normalization)
    dims = [train[k].shape[-1] for k in ['s','n','context']]
    for data in [train, val]:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
    base_logit = torch.stack([torch.logit(initial), torch.zeros(6)], -1).to(device)
    baseline_weight = final_weights(base_logit.unsqueeze(0).expand(len(val['y']),-1,-1), val['available'])
    baseline = (baseline_weight*val['prediction']).sum(-1)
    base_metrics = metrics(val['y'], baseline, val['valid'], val['day'], val['meta'])
    save(out/'baseline.json', {'weights': initial.tolist(), 'fit': 'exact train pooled daylight SSE optimum', 'validation': base_metrics})
    export(out/'baseline_validation.csv', val, baseline, baseline_weight)
    results = {}
    for kind in ['mlp', 'query']:
        for seed in [42,43,44]:
            name = f'{kind}_seed{seed}'
            dest = out/name; dest.mkdir()
            torch.manual_seed(seed); np.random.seed(seed)
            model = DynamicFusion(dims, initial, kind).to(device)
            model.eval()
            with torch.no_grad():
                p, weight = model(val)
                assert torch.equal(p, baseline) and torch.equal(weight, baseline_weight), 'Initial baseline mismatch'
                # Removing an entire modality must remain finite and route to the other one.
                small = subset(val, slice(0,2))
                small['available'] = small['available'].clone(); small['available'][...,0] = False
                pp, ww = model(small)
                assert torch.isfinite(pp).all() and torch.isfinite(ww).all()
                assert (ww[...,0] == 0).all()
            save(dest/'preflight.json', {'initial_equivalence': True, 'missing_modality_finite': True,
                  'parameters': sum(p.numel() for p in model.parameters())})
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            best = base_metrics['daylight']['nrmse_pct']
            assert best is not None, 'Primary nRMSE requires a positive observed mean'
            best_epoch = 0; best_state = copy.deepcopy(model.state_dict()); stale = 0
            mask = train['valid'] & train['day']
            points_per_case = float(mask.sum())/len(mask)
            history = []
            started = time.monotonic()
            for epoch in range(1,41):
                model.train()
                order = torch.randperm(len(mask), device=device)
                losses = []
                for ids in order.split(32):
                    batch = subset(train, ids)
                    optimizer.zero_grad(set_to_none=True)
                    pred, weight = model(batch)
                    use = batch['valid'] & batch['day']
                    loss = (((pred-batch['y'])/100).square()*use).sum()/(len(ids)*points_per_case)
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss))
                model.eval()
                with torch.no_grad():
                    pred, _ = model(val)
                    metric = metrics(val['y'], pred, val['valid'], val['day'], val['meta'])
                score = metric['daylight']['nrmse_pct']
                record = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'daylight_nrmse_pct': score,
                          'daylight_rmse': metric['daylight']['rmse'],
                          'all_rmse': metric['all']['rmse'], 'elapsed_seconds': time.monotonic()-started}
                history.append(record)
                with (dest/'history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                print(name, json.dumps(record), flush=True)
                if score < best:
                    best, best_epoch, best_state, stale = score, epoch, copy.deepcopy(model.state_dict()), 0
                else:
                    stale += 1
                if stale >= 8: break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                pred, weight = model(val)
                _, tw = model(train)
            metric = metrics(val['y'], pred, val['valid'], val['day'], val['meta'])
            mean_weight = tw.mean(0,keepdim=True)
            # Availability is reapplied so unavailable experts never gain weight.
            mw = mean_weight.expand_as(weight)*val['available']
            mw = mw/mw.sum(-1,keepdim=True).clamp_min(1e-8)
            mean_pred = (mw*val['prediction']).sum(-1)
            metric['train_mean_replacement'] = metrics(val['y'], mean_pred, val['valid'], val['day'], val['meta'])['daylight']
            metric['best_epoch'] = best_epoch
            save(dest/'metrics.json', metric)
            torch.save({'model': best_state, 'kind': kind, 'dims': dims, 'initial_weights': initial,
                        'normalization': normalization, 'best_epoch': best_epoch}, dest/'best.pt')
            export(dest/'validation.csv', val, pred, weight)
            results[name] = metric
            save(out/'progress.json', {'completed': list(results), 'last': name})
    save(out/'summary.json', {'baseline': base_metrics, 'runs': results, 'test_accessed': False})
    lines = ['# 时效查询动态融合首轮结果', '', '原17站、62训练周期、11开发验证周期；未读取测试集。', '',
             '时效基线为训练白昼平方误差下的解析最优权重。动态头均从该基线初始化，使用相同冻结专家、数据、平方误差目标。', '',
             '查询模型读取6个专家预报状态tokens，不是原始图像空间交互模型。', '',
             '|模型|全部有效点nRMSE（%）|白昼nRMSE（%，主指标）|最佳轮次|', '|---|---:|---:|---:|',
             f"|自由时效基线|{base_metrics['all']['nrmse_pct']:.3f}|{base_metrics['daylight']['nrmse_pct']:.3f}|解析解|"]
    for name, m in results.items(): lines.append(f"|{name}|{m['all']['nrmse_pct']:.3f}|{m['daylight']['nrmse_pct']:.3f}|{m['best_epoch']}|")
    lines += ['', '三个种子共享专家，不能视为三次完整模型重训；本轮为开发验证结果，不能作为新封存测试结论。',
              '本轮目标/选模口径不同于原复合损失与宏平均选模；历史原模型只作参照，不用于单独归因新架构。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cache-worker', action='store_true')
    args = parser.parse_args(); out = args.output
    if args.cache_worker:
        cache_worker(out); return
    out.mkdir(exist_ok=False)
    save(out/'status.json', {'state': 'preparing'})
    for folder in ['src', 'scripts']:
        shutil.copytree(SOURCE/folder, out/'code'/folder, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, out/Path(__file__).name)
    source_checkpoint = SOURCE/'runs/robust_dual_time_only_seed42/best_fusion.pt'
    digest = sha(source_checkpoint)
    save(out/'protocol.json', {'arms': ['analytic_time','mlp','query'], 'seeds': [42,43,44],
         'data': 'original17; train62/validation11 cycles; no confirmation access',
         'source_checkpoint': str(source_checkpoint), 'source_sha256': digest,
         'loss': 'pooled daylight squared GHI error /10000', 'selection': 'pooled daylight nRMSE percent',
         'nrmse_definition': '100 * RMSE / mean observed GHI on same evaluation mask',
         'solar_filter': 'hourly mean positive cosine SZA >= cos(85 degrees), before feature normalization',
         'max_epochs':40, 'patience':8, 'batch_size':32, 'lr':.0002, 'weight_decay':.0001,
         'initialization': 'all heads equal analytic time-only train optimum; no dynamic amplitude cap',
         'expert_state_tokens': 'six forecast states per source; history already encoded by original expert',
         'normalization': 'train only', 'static_station_id_input': False,
         'baseline': 'one analytic optimum; no fake independent baseline seeds',
         'script_sha256': sha(Path(__file__)),
         'code_sha256': {str(p.relative_to(out/'code')):sha(p) for p in (out/'code').rglob('*.py')}})
    with (out/'cache.log').open('w') as log:
        subprocess.run([sys.executable,'-u',__file__,'--cache-worker','--output',str(out)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    save(out/'status.json', {'state':'training'})
    run_training(out)
    assert sha(source_checkpoint) == digest
    save(out/'status.json', {'state':'completed','dynamic_runs':6,'analytic_baselines':1,
                            'original_checkpoint_unchanged':True,'test_accessed':False})
    print('ALL_COMPLETE',flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Never obscure the traceback; save failure state only inside this experiment.
        if DEFAULT_OUT.exists():
            save(DEFAULT_OUT/'failure.json', {'type':type(exc).__name__,'message':str(exc)})
        raise
