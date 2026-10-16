"""Station x lead RMSE/MAE from the manuscript's frozen dynamic-route test."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]/'outputs/original_r3_confirmation_20260907'
OUT=P/'station_lead_metrics'


def main():
    meta=json.loads((P/'test_metadata.json').read_text())
    datasets=[];hashes={}
    for seed in (42,43,44):
        p=P/f'test_dynamic_{seed}.csv';hashes[seed]=hashlib.sha256(p.read_bytes()).hexdigest()
        rows=list(csv.DictReader(p.open()))
        indexed={(r['sample_id'],int(r['lead_hour'])):r for r in rows}
        assert len(indexed)==len(rows)==1284
        datasets.append(indexed)
    keys=sorted(datasets[0]);stations=sorted({r['station_id'] for r in datasets[0].values()});assert len(stations)==17
    for d in datasets:
        assert d.keys()==datasets[0].keys()
        assert all(d[k]['station_id']==datasets[0][k]['station_id'] and float(d[k]['target'])==float(datasets[0][k]['target']) for k in keys)
    y=np.array([float(datasets[0][k]['target']) for k in keys])
    e=np.array([[float(d[k]['base_prediction']) for k in keys] for d in datasets])-y
    assert np.isfinite(e).all()
    station=np.array([datasets[0][k]['station_id'] for k in keys]);lead=np.array([k[1] for k in keys])
    day=np.array([meta[k[0]]['day'][k[1]-1] for k in keys],bool)
    assert day.sum()==1102
    overall=float(np.sqrt(np.mean(e**2,axis=1)).mean())
    assert abs(overall-116.6166)<.001
    result=[];perseed=[]
    for scope,scope_mask in [('all',np.ones(len(keys),bool)),('daylight',day)]:
        for s in stations:
            for h in range(1,7):
                mask=scope_mask&(station==s)&(lead==h);n=int(mask.sum())
                rmse=np.sqrt(np.mean(e[:,mask]**2,axis=1)) if n else np.full(3,np.nan)
                mae=np.mean(abs(e[:,mask]),axis=1) if n else np.full(3,np.nan)
                result.append(dict(scope=scope,station=s,lead_hour=h,n=n,rmse=float(rmse.mean()) if n else None,mae=float(mae.mean()) if n else None))
                for i,seed in enumerate((42,43,44)):
                    perseed.append(dict(scope=scope,station=s,lead_hour=h,seed=seed,n=n,rmse=float(rmse[i]) if n else None,mae=float(mae[i]) if n else None))
        assert sum(r['n'] for r in result if r['scope']==scope)==int(scope_mask.sum())
    OUT.mkdir(exist_ok=True)
    for name,rows in [('metrics.csv',result),('per_seed_metrics.csv',perseed)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    lookup={(r['scope'],r['station'],r['lead_hour']):r for r in result}
    lines=['# 17站点1–6 h RMSE与MAE','',
           '模型：第四章4.1–4.3主体动态路由（base_prediction），关闭AOD和昼夜转换修正。完整测试含TAM，221案例中216案例具有有效目标，共1284记录。指标先按42、43、44种子分别计算，再取算术平均；未先平均预测。单位W/m²。', '',
           'RMSE = sqrt(mean((预测−观测)²))；MAE = mean(abs(预测−观测))。n为该站该时效的有效案例数，不乘种子数。缺少有效目标不计入分母；不进行显著性推断。']
    for scope,title in [('all','全部有效小时（含昼夜）'),('daylight','严格白昼（小时日照比例>0.99）')]:
        lines+=['',f'## {title}']
        for metric in ('rmse','mae','n'):
            lines+=['',f'### {metric.upper()}', '', '| 站点 | 1 h | 2 h | 3 h | 4 h | 5 h | 6 h |','|---|---:|---:|---:|---:|---:|---:|']
            for s in stations:
                vals=[lookup[scope,s,h][metric] for h in range(1,7)]
                lines.append('| '+s+' | '+' | '.join('—' if v is None else str(v) if metric=='n' else f'{v:.2f}' for v in vals)+' |')
    lines+=['','两套口径均保留全部17站，白昼口径对应正文分时效表。逐站逐时效RMSE不能直接算术平均来还原总体RMSE。采用统计核验技能检查记录一致性、有效分母和种子汇总；重算总体RMSE与原文116.62一致。未改动论文。']
    (OUT/'17站点逐时效指标.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps(dict(source_hashes=hashes,overall_rmse=overall,valid_records=1284,daylight_records=1102,station_count=17,rows=len(result)),indent=2),encoding='utf-8')
    print('\n'.join(lines[:50]))


if __name__=='__main__':main()
