"""Pre-issue evidence for optional routing; never consumes radiation targets."""
import torch


def history_route_quality(satellite, batch):
    skill = satellite['history_transport_skill'].detach().clamp(-1, 1)
    available = satellite['history_transport_available'].detach().bool()
    overlap = satellite['history_transport_overlap'].detach().clamp(0, 1)
    dt = satellite['history_transport_dt_hours'].detach().clamp(0, 6)
    pairs = satellite['history_transport_pairs'].detach().clamp(0, 4)
    frames = batch['satellite_unique_frame_mask'].bool() & batch['satellite_frame_mask'].bool()
    offsets = batch['satellite_scan_time_offset_hours']
    if not torch.isfinite(offsets[frames]).all() or (offsets[frames] > 0).any():
        raise ValueError('history routing requires finite pre-issue scan offsets')
    latest = offsets.masked_fill(~frames, -torch.inf).argmax(1)
    bi = torch.arange(len(latest), device=latest.device)
    seen = frames.any(1)
    age = (-offsets[bi, latest]).clamp_min(0)
    age = torch.where(seen, age, torch.full_like(age, 6.))[:, None].expand_as(skill)
    # Count coverage only over channels belonging to this source, not padded zeros.
    valid = batch['satellite_valid'][bi, latest].float()
    source = batch['satellite_source_id']
    channel = torch.arange(valid.shape[1], device=valid.device)[None, :]
    active = ((source[:, None] == 0) & (channel < 5)) | ((source[:, None] == 1) & (channel == 5)) | ((source[:, None] == 2) & (channel == 6))
    coverage = (valid.mean((-1, -2)) * active).sum(1) / active.sum(1).clamp_min(1)
    coverage = (coverage * seen)[:, None].expand_as(skill)
    features = torch.stack([skill * available, available.to(skill), overlap * available,
                            dt / 6., pairs / 4., age.clamp_max(6) / 6., coverage], -1)
    if not torch.isfinite(features).all():
        raise ValueError('nonfinite historical route evidence')
    # This fixed cap is an experimental prior, not a calibrated probability.
    # Unknown motion skill has a conservative half-strength fallback, distinct
    # from a measured zero skill through the explicit availability feature.
    motion_support = torch.where(available, overlap * (skill + 1.) / 2., torch.full_like(skill, .5))
    cap = (coverage * torch.exp(-age / 3.) * motion_support).clamp(0, 1)
    return features.detach(), cap.detach()
