"""Rank all fixed-hard-gate examples without changing models or dataset roles."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/aod_original_hardgate_cases_20260913'

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    cases=json.loads((OUT/'case_summary.json').read_text())
    hours=read(OUT/'admitted_case_hourly.csv')
    qc={(r['sample_id'],r['lead']):r for r in read(ROOT/'outputs/aod_continuous_ground_qc_20260913/hourly_qc.csv')}
    rows=[]
    for c in cases:
        rr=[r for r in hours if r['sample_id']==c['sample_id'] and r['admitted']=='True']
        bylead=defaultdict(list)
        for r in rr:bylead[int(r['lead'])].append(r)
        changes=[float(np.mean([float(r['before'])-float(r['after']) for r in rs])) for rs in bylead.values()]
        near=[];checked=[]
        for h in bylead:
            q=qc.get((c['sample_id'],str(h)))
            if q and q['ground_usable']=='True':
                checked.append(h)
                if q['observed_near_clear']=='True':near.append(h)
        rows.append(dict(**c,
            admitted_mae_gain_Wm2=c['admitted_before_mae']-c['admitted_after_mae'],
            admitted_nrmse_gain_pp=c['admitted_before_nrmse']-c['admitted_after_nrmse'],
            admitted_rmse_relative_reduction_pct=100*(1-c['admitted_after_nrmse']/c['admitted_before_nrmse']),
            whole_case_nrmse_gain_pp=c['whole_case_before_nrmse']-c['whole_case_after_nrmse'],
            mean_prediction_decrease_Wm2=float(np.mean(changes)),max_hour_seedmean_prediction_decrease_Wm2=max(changes),
            observed_near_clear_admitted_leads=','.join(map(str,sorted(near))),
            ground_checked_admitted_leads=','.join(map(str,sorted(checked)))))
    rows.sort(key=lambda r:r['admitted_mae_gain_Wm2'],reverse=True)
    for rank,r in enumerate(rows,1):r['mae_gain_rank']=rank
    with (OUT/'ranked_case_candidates.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    names=['development__GOB__2022100900__r09','development__IZA__2021072300__r60','development__DRA__2024100200__r63']
    examples=[next(r for r in rows if r['sample_id']==sid) for sid in names]
    (OUT/'stronger_case_candidates.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
    table=[]
    for c in examples:
        table.append(f"| {c['station_id']} | {c['issue_time_utc']} | {c['admitted_leads']} | {c['admitted_before_nrmse']:.3f} → {c['admitted_after_nrmse']:.3f} | {c['admitted_before_mae']:.2f} → {c['admitted_after_mae']:.2f} | {c['admitted_rmse_relative_reduction_pct']:.2f}% |")
    text=f'''# 硬门控下改善较大的案例

在原硬门控、当前主模型、既有 AOD 检查点下复核了全部 28 个触发案例。以下候选均属于当前训练集，三个种子的触发段 MAE 和整段有效小时 nRMSE 均改善。按误差收益事后排序用于寻找图示素材，不能视为代表性或独立测试评估。

| 站点 | 起报时刻 UTC | 触发时效/h | 触发段 nRMSE/% | 触发段 MAE/(W/m²) | RMSE 相对下降 |
|---|---|---|---|---|---:|
{chr(10).join(table)}

## 候选特点

- **GOB，2022-10-09**：六小时真值完整，六小时均触发。nRMSE 下降 0.619 个百分点，MAE 下降 4.53 W/m²。分钟观测中 2–6 h 通过近晴空判据，1 h 未通过。最适合先观察完整预测序列中的修正效果；不能把六小时全部描述为已确认晴空。
- **IZA，2021-07-25**：有效小时为 1、2、5、6 h，3–4 h 真值不可用。有效小时 nRMSE 下降 0.596 个百分点，MAE 下降 4.98 W/m²。只有 1、6 h 通过观测近晴空判据，完整性弱于 GOB。图中必须保留真值缺口。
- **DRA，2024-10-04**：六小时真值完整，3–5 h 触发。触发段 nRMSE 相对下降约 10.52%，对应绝对下降约 0.315 个百分点；MAE 下降 2.71 W/m²。触发段中 3、4 h 通过观测近晴空判据，5 h 未通过。

候选的实际预测下降量、每个种子的指标和近晴空核验明细分别保存在 `ranked_case_candidates.csv`、`case_metrics.csv` 和分钟观测 QC 文件中。预测改变量与误差收益分开统计，避免把更大的下调直接当作更大的效果。

当前验证集仍只有 IZA 2024-03-17 这一较小收益案例，测试集仍没有触发案例。本轮没有调整门槛、重新划分数据或使用旧模型结果代替当前结果。
'''
    (OUT/'较强案例筛选.md').write_text(text,encoding='utf-8')
    print(json.dumps([{k:c[k] for k in ('station_id','issue_time_utc','split','admitted_mae_gain_Wm2',
        'mean_prediction_decrease_Wm2','max_hour_seedmean_prediction_decrease_Wm2','admitted_rmse_relative_reduction_pct')} for c in examples],indent=2))

if __name__=='__main__':main()
