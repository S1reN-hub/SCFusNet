#!/usr/bin/env python3
"""Verify public NPZ -> recovered experts -> formal MLP -> fixed AOD -> paper cohorts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import scfusnet_aod_runtime as aod
import reproduce_paper_metrics as scoring

SEEDS=(42,43,44)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True,help='Extracted input directory containing cases/.')
    p.add_argument('--model-bundle',type=Path,required=True,help='Extracted SCFusNet-recovered-experts directory.')
    p.add_argument('--output-dir',type=Path,default=ROOT/'reproduction/raw_pipeline')
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args()
    if args.batch_size<1:p.error('batch-size must be positive')
    started=time.monotonic()
    torch.set_num_threads(4)
    device=torch.device(args.device)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    bundle=args.model_bundle.resolve()
    # Validate every supplied model artifact before importing its compatibility implementation.
    model_manifest=json.loads((bundle/'MANIFEST.json').read_text(encoding='utf-8'))
    for item in model_manifest['files']:
        f=(bundle/item['path']).resolve()
        if not f.is_relative_to(bundle):raise ValueError('Invalid model manifest path')
        if f.stat().st_size!=item['bytes'] or digest(f)!=item['sha256']:
            raise ValueError('Model manifest mismatch: '+item['path'])
    sys.path.insert(0,str(bundle))
    import compat_model as architecture
    import compat_data as dataset_code
    import compat_router as routing
    from predict import RecoveredSCFusNet
    model=RecoveredSCFusNet(bundle,42).to(device).eval()
    normalization=json.loads((bundle/'base_normalization.json').read_text(encoding='utf-8'))
    reference=json.loads((bundle/'aod_reference.json').read_text(encoding='utf-8'))
    index=read(ROOT/'data/recovered_inputs_20261018/index.csv')
    rows=[r for r in index if r['experiment_split']=='test']
    assert len(rows)==119 and len({r['sample_id'] for r in rows})==119
    hashes={r['sample_id']:r for r in read(ROOT/'data/recovered_inputs_20261018/manifest.csv')}
    for row in rows:
        f=args.data_root/row['case_file']; expected=hashes[row['sample_id']]
        assert f.stat().st_size==int(expected['bytes']) and digest(f)==expected['sha256'],row['sample_id']
    arrays={k:[] for k in ('s','n','context','prediction','available','y','valid','day')}
    raw_items=[];meta=[]
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            chunk=rows[start:start+args.batch_size]
            dataset=dataset_code.MultisourceCaseDataset(args.data_root,chunk,normalization,reference,require_formal_v4=True)
            raw=default_collate([dataset[i] for i in range(len(dataset))])
            batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in raw.items()}
            s=model.satellite_expert(batch,prediction_path='dual');n=model.wrf_expert(batch)
            def token(e,names):return torch.cat([e['state'],*[e[k].float().unsqueeze(-1) for k in names]],-1)
            consistency=architecture.motion_wind_consistency(s['motion_vector'],n['cloud_wind_vector'],batch.get('satellite_latent_to_en_m'))
            result={'s':token(s,['q','scale_q','clear_logit','valid_fraction','freshness','dynamics','motion_confidence','unique_frame_fraction']),
                'n':token(n,['q','scale_q','clear_logit','anchor_trust','transport_residual']),
                'context':torch.cat([batch['solar_context'],consistency,(s['q']-n['q']).unsqueeze(-1)],-1),
                'prediction':torch.stack([architecture.ScenarioDrivenGHIModelV4.project(e['prediction'],batch) for e in (s,n)],-1),
                'available':torch.stack([s['available'],n['available']],-1).bool(),
                'y':torch.where(batch['target_mask'].bool(),batch['target'],0.),'valid':batch['target_mask'].bool(),
                'day':batch['solar_context'][...,0]>=.0871557427}
            for key in arrays:arrays[key].append(result[key].cpu())
            for i,sample in enumerate(dataset.samples):
                raw_items.append({k:sample[k] for k in ('static','solar_context','aod550','aod_reference','aod_mask','air_mass','target','target_mask','sample_id')})
                meta.append({k:raw[k][i] for k in ('sample_id','station_id','cycle')})
            print(f'Raw-input experts: {min(start+args.batch_size,len(rows))}/{len(rows)} cases',flush=True)
    cache={k:torch.cat(v) for k,v in arrays.items()};cache['meta']=meta
    # This output is freshly computed from NPZ inputs, never substituted with a historical cache.
    torch.save(cache,args.output_dir/'raw_expert_features.pt')
    aod.POLICY.update(anomaly_min=.10,common_clear_min=.70)
    enriched=aod.enrich(cache,raw_items)
    common=scoring.index(read(ROOT/scoring.COMPARISON))
    persistence=scoring.index(read(ROOT/scoring.PERSISTENCE))
    assert len(common)==601 and len(persistence)==452 and set(persistence)<=set(common)
    actual_keys={(m['sample_id'],m['station_id'],m['cycle'],str(h+1)):(i,h) for i,m in enumerate(meta) for h in range(6)}
    assert set(common)<=set(actual_keys)
    valid_keys={k for k,(i,h) in actual_keys.items() if bool(enriched['valid'][i,h])}
    assert valid_keys==set(common),'Historical AOD/daylight cohort changed'
    for k in common:
        i,h=actual_keys[k]
        assert abs(float(cache['y'][i,h])-float(common[k]['target']))<1e-6
    admission={k for k in common if bool(enriched['admit'][actual_keys[k]])}
    expected_admission=None;seed_results={};predictions={}
    policy_root=ROOT/'outputs/aod_hardgate_relaxation_v1_20260913'
    assert json.loads((policy_root/'selection_frozen.json').read_text(encoding='utf-8'))['selected']=='d10_c70'
    for seed in SEEDS:
        head=torch.load(bundle/f'formal_router_seed{seed}.pt',map_location='cpu',weights_only=True)
        router=routing.DynamicFusion(head['dims'],head['initial_weights'],head['kind']).eval()
        router.load_state_dict(head['model'],strict=True)
        scaled=dict(cache)
        for key,stats in head['normalization'].items():
            scaled[key]=(cache[key]-torch.tensor(stats['mean']).view(1,1,-1))/torch.tensor(stats['std']).view(1,1,-1)
        correction_path=policy_root/f'd10_c70/aod_seed{seed}.pt'
        correction=torch.load(correction_path,map_location='cpu',weights_only=True)
        assert correction['policy']['anomaly_min']==.10 and correction['policy']['common_clear_min']==.70
        with torch.inference_mode():
            before,weights=router(scaled)
            if correction['model'] is None:after=before.clone()
            else:
                corr=aod.Correction(correction['dim']).eval();corr.load_state_dict(correction['model'],strict=True)
                after=corr(enriched,before)
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        assert torch.equal(after[~enriched['admit']],before[~enriched['admit']])
        historical=scoring.index(read(policy_root/f'd10_c70/test_seed{seed}.csv'))
        expected={k for k,v in historical.items() if v['admitted'].lower()=='true'}
        assert set(historical)==set(common) and admission==expected
        if expected_admission is None:expected_admission=expected
        assert expected==expected_admission and len(expected)==38
        out=[]
        for k,(i,h) in actual_keys.items():
            out.append(dict(zip(('sample_id','station_id','cycle','lead'),k),target=float(cache['y'][i,h]),valid=bool(cache['valid'][i,h]),daylight=bool(cache['day'][i,h]),satellite=float(cache['prediction'][i,h,0]),nwp=float(cache['prediction'][i,h,1]),before=float(before[i,h]),after=float(after[i,h]),admitted=bool(enriched['admit'][i,h]),aod550=float(enriched['aod550'][i,h]),aod_reference=float(enriched['aod_reference'][i,h]),common_clear=float(enriched['common_clear'][i,h]),satellite_weight=float(weights[i,h,0])))
        lookup=scoring.index(out)
        with (args.output_dir/f'raw_seed{seed}.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n');writer.writeheader();writer.writerows(out)
        comparisons={}
        for field in ('before','after','aod550','aod_reference','common_clear'):
            differences=[abs(float(lookup[k][field])-float(historical[k][field])) for k in common]
            comparisons[field]={'max_abs':max(differences),'mean_abs':statistics.fmean(differences)}
        for field in ('before','after'):
            # Predeclared compatibility envelope; failures remain failures, not tuned away.
            assert comparisons[field]['max_abs']<.5,(seed,field,comparisons[field])
        current=[lookup[k] for k in common]
        paired=[dict(lookup[k],sp=persistence[k]['sp']) for k in persistence]
        sp_rmse=scoring.metric(paired,'sp')['rmse_wm2']
        metrics={side:scoring.metric(current,side) for side in ('before','after')}
        for side in metrics:metrics[side]['fs_pct']=100*(1-scoring.metric(paired,side)['rmse_wm2']/sp_rmse)
        admitted=[lookup[k] for k in expected]
        seed_results[str(seed)]={'comparison':comparisons,'common_601':metrics,'admitted_38':{side:scoring.metric(admitted,side) for side in ('before','after')},'correction_sha256':digest(correction_path),'correction_epoch':correction['best_epoch']}
        predictions[str(seed)]=lookup
    main_summary=scoring.aggregate([r['common_601']['before'] for r in seed_results.values()])
    aod_summary={side:scoring.aggregate([r['admitted_38'][side] for r in seed_results.values()]) for side in ('before','after')}
    reference_metrics=scoring.reproduce(ROOT)
    paper_main=reference_metrics['table_iii']['SCFusNet']['summary']
    main_deltas={k:main_summary[k]['mean']-paper_main[k]['mean'] for k in main_summary}
    rounding={k:round(main_summary[k]['mean'],3 if k=='r2' else 2)==round(paper_main[k]['mean'],3 if k=='r2' else 2) for k in main_summary}
    # Table II expert/fixed-fusion rows also run from newly inferred branch predictions.
    first=predictions['42'];branch_comparison={}
    for label,field in [('Satellite expert','satellite'),('NWP expert','nwp'),('Fixed fusion (0.2 satellite + 0.8 NWP)','fixed_blend')]:
        mapped={k:dict(v,fixed_blend=.2*v['satellite']+.8*v['nwp']) for k,v in first.items()}
        table=[mapped[k] for k in common];paired=[dict(mapped[k],sp=persistence[k]['sp']) for k in persistence]
        values=scoring.metric(table,field);values['fs_pct']=100*(1-scoring.metric(paired,field)['rmse_wm2']/sp_rmse)
        target=reference_metrics['table_ii'][label]['summary']
        branch_comparison[label]={'raw_metrics':values,'delta_vs_published':{k:values[k]-target[k]['mean'] for k in values},'paper_rounding_matches':{k:round(values[k],3 if k=='r2' else 2)==round(target[k]['mean'],3 if k=='r2' else 2) for k in values}}
    env={'python':platform.python_version(),'implementation':platform.python_implementation(),'os':platform.system(),'release':platform.release(),'machine':platform.machine(),'numpy':np.__version__,'torch':str(torch.__version__),'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'inference_device':str(device),'threads':torch.get_num_threads(),'batch_size':args.batch_size}
    report={'schema':'scfusnet_raw_pipeline_verification_v1','date':'2026-10-18','assertions_passed':True,'training_performed':False,'checkpoint_or_gate_selection_performed':False,'model_artifact_manifest_verified':True,'selected_input_files_sha256_verified':119,'cases':119,'leads':6,'main_comparison_rows':601,'persistence_rows':452,'aod_admitted_hours':38,'aod_admitted_cases':len({k[0] for k in admission}),'admission_keys_identical_to_historical':True,'prediction_source':'Fresh inference from published NPZ files; historical caches/tables are validation references only','model_source_original_whole_checkpoint_recovered':False,'by_seed':seed_results,'main_table_summary':main_summary,'main_table_delta_vs_published':main_deltas,'main_table_paper_rounding_matches':rounding,'table_ii_branch_rows':branch_comparison,'aod_summary':aod_summary,'aod_admitted_rmse_reduction_pct':100*(1-aod_summary['after']['rmse_wm2']['mean']/aod_summary['before']['rmse_wm2']['mean']),'environment':env,'elapsed_seconds':round(time.monotonic()-started,2),'limits':['Numerical compatibility, not bit-exact original inference.','This verifies inference and metric calculation, not retraining or all ablation models from raw inputs.','The main comparison uses BEFORE-AOD predictions; the admitted-hour AOD sensitivity is a separate exploratory result.','The original full checkpoint bytes, complete executed-source/environment identity and historical per-NPZ hashes remain unavailable.']}
    write_json(args.output_dir/'RAW_PIPELINE_REPORT.json',report)
    write_json(args.output_dir/'ENVIRONMENT.json',env)
    write_json(args.output_dir/'HISTORICAL_TABLE_RECALCULATION.json',reference_metrics)
    print(json.dumps({'assertions_passed':True,'rounding':rounding,'main_metric_deltas':main_deltas,'aod_rmse_reduction_pct':report['aod_admitted_rmse_reduction_pct'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    if not all(rounding.values()):
        raise SystemExit('Raw-inference metrics differ at published rounding; full differences were saved in the report.')

if __name__=='__main__':main()
