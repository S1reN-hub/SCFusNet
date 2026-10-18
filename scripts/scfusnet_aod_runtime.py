"""Inference-only definitions copied without changes from run_aod_7eval_20260912.py.
Source SHA-256: 030fea2ad3ab9e015b7d1d1a1a28ca72af0f7f9a8b0f457c76c2f1fb171e67cc
"""
import math
import torch
from torch import nn

POLICY = dict(anomaly_min=0.20, common_clear_min=0.80, day_fraction_min=0.99,
              mean_cos_sza_min=0.25, epochs=24, patience=8, learning_rate=0.001,
              selection='validation admitted-hour MAE; epoch zero is no correction',
              batch_cases=4, loss='admitted-hour MAE', comparator='frozen current dynamic MLP',
              seed_reporting='mean of separately calculated seed metrics',
              background='current training rows only; station-month median with fallbacks')

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
