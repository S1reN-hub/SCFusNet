"""Publish an immutable development dataset with audited satellite replacements.

All work is written to a new .staging directory; index.csv is published only
after every case and train-only statistic has passed verification.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


PATCH_FIELDS = {
    'satellite', 'satellite_valid', 'satellite_valid_area_fraction',
    'satellite_frame_mask', 'satellite_channel_mask', 'satellite_source_id',
    'satellite_scan_x_rad', 'satellite_scan_y_rad', 'satellite_scan_x_edges_rad',
    'satellite_scan_y_edges_rad', 'satellite_latent_to_en_m',
    'satellite_frame_id', 'satellite_unique_frame_mask',
    'satellite_scan_time_utc', 'satellite_frame_time_utc',
}
CHANGED_FIELDS = PATCH_FIELDS | {'satellite_repair_metadata_json'}


def cycle(row):
    return next(row[k] for k in ('cycle', 'init_time_utc', 'wrf_init_time_utc') if row.get(k))


def check_original(z, reference):
    for field in ('satellite', 'satellite_valid'):
        if array_sha(z[field]) != reference['original_'+field+'_sha256']:
            raise ValueError('original satellite checksum mismatch: '+field)
    if int(z['satellite_source_id']) != reference['source_id']:
        raise ValueError('original source mismatch')


def patch_case(original, patch, source):
    extra = set(patch) - PATCH_FIELDS - {'sample_id', 'metadata_json'}
    if extra:
        raise ValueError('unauthorized replacement fields: '+str(extra))
    result = dict(original)
    result.update({k: v for k, v in patch.items() if k in PATCH_FIELDS})
    result['satellite_repair_metadata_json'] = np.asarray(str(patch['metadata_json'].item()))
    valid = result['satellite_valid']
    if valid.dtype != bool or valid.shape != result['satellite'].shape:
        raise ValueError('invalid replacement mask')
    result['satellite_frame_mask'] = valid.reshape(len(valid), -1).any(1)
    result['satellite_channel_mask'] = valid.reshape(valid.shape[0], valid.shape[1], -1).any(2)
    if source == 0:
        # GOES frames retain their original actual-scan identity and timing.
        keys = original.get('satellite_frame_id')
        if keys is None:
            keys = original['satellite_scan_time_offset_hours']
        seen = set()
        unique = np.zeros(len(keys), dtype=bool)
        for i, key in enumerate(keys.tolist()):
            if result['satellite_frame_mask'][i] and key not in seen:
                unique[i] = True
                seen.add(key)
        result['satellite_unique_frame_mask'] = unique
    else:
        result['satellite_unique_frame_mask'] = patch['satellite_unique_frame_mask'] & result['satellite_frame_mask']
    # Original offset/age and all non-satellite arrays must be bit-for-bit equal.
    for k, value in original.items():
        if k not in CHANGED_FIELDS and array_sha(value) != array_sha(result[k]):
            raise ValueError('protected field changed: '+k)
    return result


def satellite_statistics(paths):
    sums = squares = counts = None
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            values = z['satellite'].astype(np.float64)
            valid = z['satellite_valid'].astype(bool) & np.isfinite(values)
        safe = np.where(valid, values, 0.)
        reduction = tuple(i for i in range(values.ndim) if i != 1)
        s, q, n = safe.sum(axis=reduction), (safe*safe).sum(axis=reduction), valid.sum(axis=reduction).astype(np.float64)
        if sums is None:
            sums, squares, counts = np.zeros_like(s), np.zeros_like(q), np.zeros_like(n)
        sums += s
        squares += q
        counts += n
    if counts is None:
        raise ValueError('empty training partition')
    missing = np.flatnonzero(counts == 0).tolist()
    counts[counts == 0] = 1.  # Same neutral missing-channel convention as data.py.
    means = sums/counts
    std = np.sqrt(np.maximum(squares/counts-means*means, 1e-8))
    means[missing], std[missing] = 0., 1.
    return dict(mean=means.tolist(), std=std.tolist(), count=counts.tolist(), unavailable_channels=missing)


def merge(original_root, goes_root, himawari_root, manifest_path, output_root):
    original_root, goes_root, himawari_root, manifest_path, output_root = map(Path, (original_root, goes_root, himawari_root, manifest_path, output_root))
    stage = output_root.with_name(output_root.name+'.staging')
    if output_root.exists() or stage.exists():
        raise FileExistsError('output or staging already exists; refusing overwrite')
    manifest = read_json(manifest_path)
    with (original_root/'index.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns, rows = reader.fieldnames, list(reader)
    ids = [r['sample_id'] for r in rows]
    digest = hashlib.sha256(('\n'.join(sorted(ids))+'\n').encode()).hexdigest()
    if len(set(ids)) != len(ids) or digest != manifest['dataset_id_sha256'] or set(ids) != set(manifest['cases']):
        raise ValueError('development manifest/index identity mismatch')
    allowed_roles = {'development', 'develop', 'dev', 'train', 'training', 'cv'}
    for row in rows:
        role = next((row[k].strip().lower() for k in ('role', 'current_role', 'data_role', 'dataset_role', 'partition') if row.get(k)), '')
        if role not in allowed_roles:
            raise ValueError('non-development or undeclared role prohibited')
        if Path(row['sample_id']).name != row['sample_id'] or '/' in row['sample_id'] or '\\' in row['sample_id']:
            raise ValueError('unsafe sample identifier')
    split_path = original_root/'holdout_split_73.json'
    split = read_json(split_path)['folds'][0]
    train, validation = set(split['train']), set(split['validation'])
    if train & validation or train | validation != {cycle(r) for r in rows}:
        raise ValueError('invalid development cycle partition')
    roots = {0: goes_root, 1: himawari_root}
    patch_paths, patch_hashes = {}, {}
    for source, root in roots.items():
        audit = read_json(root/'gpu_input_audit.json')
        expected = {s for s, ref in manifest['cases'].items() if ref['source_id'] == source}
        found = {p.stem: p for p in (root/'cases').glob('*.npz')}
        if not audit.get('passed') or audit.get('source_id') != source or audit.get('dataset_id_sha256') != digest or audit.get('cases') != len(expected) or set(found) != expected:
            raise ValueError('patch audit or case inventory mismatch')
        patch_paths.update(found)
    # Preflight all originals and patches before constructing the destination.
    for row in rows:
        sid = row['sample_id']
        ref = manifest['cases'][sid]
        if ref['source_id'] not in (0, 1, 2):
            raise ValueError('unsupported source')
        with np.load(original_root/row['case_file'], allow_pickle=False) as z:
            check_original(z, ref)
            if sid in patch_paths:
                with np.load(patch_paths[sid], allow_pickle=False) as p:
                    if str(p['sample_id'].item()) != sid or int(p['satellite_source_id']) != ref['source_id']:
                        raise ValueError('patch identity mismatch')
                    patch_case(dict(z), dict(p), ref['source_id'])
                patch_hashes[sid] = sha(patch_paths[sid])
    norm_path = original_root/'statistics/holdout_train_normalization.json'
    normalization = read_json(norm_path)
    aod_path = original_root/'statistics/holdout_train_aod_reference.json'
    if not aod_path.is_file():
        raise FileNotFoundError(aod_path)
    (stage/'cases').mkdir(parents=True)
    (stage/'statistics').mkdir()
    report = dict(passed=False, dataset_id_sha256=digest, original_root=str(original_root.resolve()), manifest_sha256=sha(manifest_path), cases=len(rows), patched=len(patch_paths), train_cycles=len(train), validation_cycles=len(validation), confirmation_accessed=False, patch_sha256=patch_hashes, original_index_sha256=sha(original_root/'index.csv'))
    try:
        for row in rows:
            sid = row['sample_id']
            source = (original_root/row['case_file']).resolve()
            dest = stage/'cases'/(sid+'.npz')
            if sid not in patch_paths:
                try:
                    dest.symlink_to(source)
                except OSError:
                    shutil.copy2(source, dest)
            else:
                if sha(patch_paths[sid]) != patch_hashes[sid]:
                    raise ValueError('patch changed after preflight')
                with np.load(source, allow_pickle=False) as z, np.load(patch_paths[sid], allow_pickle=False) as p:
                    original = dict(z)
                    check_original(original, manifest['cases'][sid])
                    merged = patch_case(original, dict(p), manifest['cases'][sid]['source_id'])
                np.savez_compressed(dest, **merged)
                with np.load(dest, allow_pickle=False) as z:
                    if set(z.files) != set(merged) or any(array_sha(z[k]) != array_sha(v) for k, v in merged.items()):
                        raise ValueError('serialized case integrity mismatch')
            row['case_file'] = 'cases/'+sid+'.npz'
        old_normalization = json.loads(json.dumps(normalization))
        normalization['satellite'] = satellite_statistics([stage/r['case_file'] for r in rows if cycle(r) in train])
        assert {k:v for k,v in normalization.items() if k != 'satellite'} == {k:v for k,v in old_normalization.items() if k != 'satellite'}
        write_json(stage/'statistics/holdout_train_normalization.json', normalization)
        shutil.copy2(split_path, stage/split_path.name)
        shutil.copy2(aod_path, stage/'statistics'/aod_path.name)
        write_json(stage/'statistics/satellite_normalization_provenance.json', dict(train_cycles=sorted(train), train_cases=sum(cycle(r) in train for r in rows), original_normalization_sha256=sha(norm_path), unchanged_non_satellite_statistics=True, definition='data.py compute_normalization: channel axis 1, finite & satellite_valid, all historical pixels; no deduplication', dataset_id_sha256=digest))
        report['passed'] = True
        write_json(stage/'merge_report.json', report)
        with (stage/'index.csv').open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.rename(stage, output_root)
    except Exception as exc:
        (stage/'index.csv').unlink(missing_ok=True)
        write_json(stage/'merge_failure.json', dict(error=str(exc), passed=False))
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original-root', 'goes-root', 'himawari-root', 'manifest', 'output-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = merge(args.original_root, args.goes_root, args.himawari_root, args.manifest, args.output_root)
    print(json.dumps({k:v for k,v in report.items() if k != 'patch_sha256'}))


if __name__ == '__main__':
    main()
