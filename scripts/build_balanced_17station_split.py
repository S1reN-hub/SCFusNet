"""Metadata-only stratified split; fixed seed, complete cycles, overlap groups.
No observations, predictions or model scores are used to choose assignments.
"""
from pathlib import Path
from datetime import datetime,timedelta
from collections import defaultdict,Counter
import csv,json,hashlib,sys
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_gfs_swradnet_dataset import load_stations,solar_features
OUT=ROOT/'outputs/dataset_split_17train_7eval_expanded_20260909';OUT.mkdir(exist_ok=True)
SEED=20260909;SPLITS=['train','validation','test'];TARGET=np.array([52,17,17]);EVAL=['GOB','DRA','IZA','TBL','SEL','GWN','CAB']
FILES=[ROOT/'data/formal_0_6_v1/manifests/wrf_random_6h_cases.csv',ROOT/'data/expansion_p46_v4/wrf_random_6h_cases.csv']
def dt(t):return datetime.fromisoformat(t.replace('Z','+00:00'))
def write(name,rr):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
rows=[];stations=load_stations(ROOT/'configs/gfs_swradnet_stations.csv')
for path in FILES:
    for r in csv.DictReader(path.open(encoding='utf-8-sig')):
        issue=dt(r['issue_time_utc']);cycle=dt(r['wrf_init_time_utc']).strftime('%Y%m%d%H')
        solar_hours=sum(solar_features(issue+timedelta(hours=h),stations[r['station_id']],h,'none',None)[0][0]>=np.cos(np.deg2rad(85)) for h in range(1,7))
        rows.append(dict(source_sample_id=r['sample_id'],case_key=f"{r['station_id']}__{cycle}__r{int(r['issue_offset_hours']):02d}",station_id=r['station_id'],cycle=cycle,issue_offset_hours=int(r['issue_offset_hours']),issue_time_utc=r['issue_time_utc'],sample_end_time_utc=r['sample_end_time_utc'],source_manifest=str(path.relative_to(ROOT)).replace('\\','/'),geometry_eligible_hours=int(solar_hours),evaluation_station=r['station_id'] in EVAL))
assert len(rows)==1462 and len({r['case_key'] for r in rows})==1462
bycycle=defaultdict(list)
for r in rows:bycycle[r['cycle']].append(r)
cycles=sorted(bycycle);assert len(cycles)==86 and all(len(z)==17 for z in bycycle.values())
old=json.loads((ROOT/'outputs/original_r3_confirmation_20260907/locked_protocol.json').read_text())
oldsplit={c:s for s,k in [('train','train_cycles'),('validation','validation_cycles'),('test','cycles')] for c in old[k]}
assert set(oldsplit)==set(cycles)
# Conservative full-cycle envelopes include satellite history and all station targets.
envelopes={c:(min(dt(r['issue_time_utc'])-timedelta(hours=6) for r in z),max(dt(r['sample_end_time_utc']) for r in z)) for c,z in bycycle.items()}
parent={c:c for c in cycles}
def find(c):
    while parent[c]!=c:c=parent[c]
    return c
edges=[]
for i,a in enumerate(cycles):
    for b in cycles[i+1:]:
        if max(envelopes[a][0],envelopes[b][0])<min(envelopes[a][1],envelopes[b][1]):
            parent[find(b)]=find(a);edges.append([a,b])
groups=defaultdict(list)
for c in cycles:groups[find(c)].append(c)
groups=sorted(groups.values(),key=lambda g:g[0]);ng=len(groups);sizes=np.array([len(g) for g in groups])
features={};weights={}
for year in sorted({c[:4] for c in cycles}):features['year_'+year]={c:float(c[:4]==year) for c in cycles};weights['year_'+year]=5
for q in range(1,5):features[f'quarter_{q}']={c:float((int(c[4:6])-1)//3+1==q) for c in cycles};weights[f'quarter_{q}']=5
for year,q in sorted({(c[:4],(int(c[4:6])-1)//3+1) for c in cycles}):
    k=f'stratum_{year}_Q{q}';features[k]={c:float(c[:4]==year and (int(c[4:6])-1)//3+1==q) for c in cycles};weights[k]=1
for sid in EVAL:
    k='solar_hours_'+sid;features[k]={c:float(next(r['geometry_eligible_hours'] for r in bycycle[c] if r['station_id']==sid)) for c in cycles};weights[k]=.7
names=list(features);F=np.array([[sum(features[k][c] for c in g) for g in groups] for k in names]);nf=len(names);nx=ng*3;nv=nx+nf*3
rng=np.random.default_rng(SEED);objective=np.zeros(nv);objective[:nx]=rng.uniform(0,1e-5,nx)
for k,name in enumerate(names):objective[nx+k*3:nx+k*3+3]=weights[name]/max(F[k].sum(),1)
A=[];lb=[];ub=[]
def constraint(a,l,u):A.append(a);lb.append(l);ub.append(u)
for g in range(ng):
    a=np.zeros(nv);a[g*3:g*3+3]=1;constraint(a,1,1)
for s in range(3):
    a=np.zeros(nv);a[np.arange(ng)*3+s]=sizes;constraint(a,TARGET[s],TARGET[s])
for k,name in enumerate(names):
    for s in range(3):
        a=np.zeros(nv);a[np.arange(ng)*3+s]=F[k];target=F[k].sum()*TARGET[s]/86
        if name.startswith('year_'):constraint(a.copy(),1,np.inf)
        if name.startswith('quarter_'):constraint(a.copy(),2,np.inf)
        if name.startswith('stratum_') and s==0:constraint(a.copy(),1,np.inf)
        a[nx+k*3+s]=-1;constraint(a,-np.inf,target)
        b=-a;b[nx+k*3+s]=-1;constraint(b,-np.inf,-target)
result=milp(objective,integrality=np.r_[np.ones(nx),np.zeros(nf*3)],bounds=Bounds(np.zeros(nv),np.r_[np.ones(nx),np.full(nf*3,np.inf)]),constraints=LinearConstraint(np.asarray(A),lb,ub),options={'time_limit':30,'mip_rel_gap':.02})
assert result.x is not None,result.message
chosen=result.x[:nx].reshape(ng,3);assert np.allclose(chosen,np.round(chosen),atol=1e-5)
assignment={c:SPLITS[int(np.argmax(chosen[g]))] for g,cs in enumerate(groups) for c in cs}
assert [sum(s==x for s in assignment.values()) for x in SPLITS]==TARGET.tolist()
for a,b in edges:assert assignment[a]==assignment[b]
for r in rows:
    r['evaluation_cycle_partition']=assignment[r['cycle']]
    r['split']=r['evaluation_cycle_partition'] if r['evaluation_station'] else 'train'
    r['selection_rule']='seven_station_cycle_partition' if r['evaluation_station'] else 'all_cases_to_train'
    r['old_split']=oldsplit[r['cycle']]
# Explicit station-hour overlap audit, including six-hour input history.
keys=defaultdict(set);targetkeys=defaultdict(set)
for r in rows:
    for h in range(-6,6):keys[r['split']].add((r['station_id'],(dt(r['issue_time_utc'])+timedelta(hours=h)).isoformat()))
    for h in range(1,7):targetkeys[r['split']].add((r['station_id'],(dt(r['issue_time_utc'])+timedelta(hours=h)).isoformat()))
for i,s in enumerate(SPLITS):
    for t in SPLITS[i+1:]:assert not keys[s]&keys[t] and not targetkeys[s]&targetkeys[t]
write('case_assignments.csv',rows)
write('train_all17.csv',[r for r in rows if r['split']=='train'])
write('validation_7station.csv',[r for r in rows if r['split']=='validation'])
write('test_7station.csv',[r for r in rows if r['split']=='test'])
cycle_rows=[dict(cycle=c,year=int(c[:4]),quarter=(int(c[4:6])-1)//3+1,evaluation_split=assignment[c],old_split=oldsplit[c],overlap_group=find(c)) for c in cycles];write('cycle_assignments.csv',cycle_rows)
coverage=[dict(year=y,quarter=q,**{s:sum(r['year']==y and r['quarter']==q and r['evaluation_split']==s for r in cycle_rows) for s in SPLITS}) for y,q in sorted({(r['year'],r['quarter']) for r in cycle_rows})];write('year_quarter_coverage.csv',coverage)
summary={
    'train':dict(cycles=len({r['cycle'] for r in rows if r['split']=='train'}),stations=17,cases=sum(r['split']=='train' for r in rows),evaluation_station_cases=sum(r['split']=='train' and r['evaluation_station'] for r in rows),non_evaluation_station_cases=sum(r['split']=='train' and not r['evaluation_station'] for r in rows),geometry_eligible_hours=sum(r['geometry_eligible_hours'] for r in rows if r['split']=='train')),
    'validation':dict(cycles=sum(v=='validation' for v in assignment.values()),stations=7,cases=sum(r['split']=='validation' for r in rows),geometry_eligible_hours=sum(r['geometry_eligible_hours'] for r in rows if r['split']=='validation')),
    'test':dict(cycles=sum(v=='test' for v in assignment.values()),stations=7,cases=sum(r['split']=='test' for r in rows),geometry_eligible_hours=sum(r['geometry_eligible_hours'] for r in rows if r['split']=='test')),
}
split_cycles={s:[c for c in cycles if assignment[c]==s] for s in SPLITS}
cv=dict(schema_version='station_aware_grouped_split_v1',folds=[dict(fold=0,train=split_cycles['train'],validation=split_cycles['validation'])],test=split_cycles['test'],note='Cycle lists partition the seven evaluation stations; all cases from the other ten stations belong to training.')
(OUT/'cv_17train_7eval.json').write_text(json.dumps(cv,ensure_ascii=False,indent=2),encoding='utf-8')
config=dict(version='17train_7eval_expanded_balanced_20260909',seed=SEED,training_stations=sorted(stations),evaluation_stations=EVAL,non_evaluation_training_stations=sorted(set(stations)-set(EVAL)),validation_stations=EVAL,test_stations=EVAL,split_cycles=split_cycles,station_case_policy='All 86 cases from each non-evaluation station are training cases; the seven evaluation stations use the 52/17/17 cycle split.',training_sampling='station_balanced',cv_file=str((OUT/'cv_17train_7eval.json').relative_to(ROOT)).replace('\\','/'),source_manifests=[dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in FILES],case_manifest=str((OUT/'case_assignments.csv').relative_to(ROOT)).replace('\\','/'),status='split_defined_not_retrained',required_retraining='Refit normalization and all label-trained components on the 1224 training cases; perform model selection and final evaluation only on the seven named stations',test_status='Repartition of previously inspected data, not a newly untouched test set',evaluation_station_cycle_ratio=[52/86,17/86,17/86],active_case_counts=[1224,119,119],split_method='Fixed-seed metadata-only year/quarter balancing for the seven-station evaluation cohort; non-evaluation stations are assigned entirely to training',counts=summary)
(ROOT/'configs/dataset_split_17train_7eval_expanded_20260909.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
audit=dict(counts=summary,overlap_edges=edges,overlap_groups=ng,cross_split_station_hour_overlap=0,cross_split_target_overlap=0,validation_cycles_present_in_training_at_other_stations=17,test_cycles_present_in_training_at_other_stations=17,solver_status=int(result.status),solver_message=result.message,metadata_balance_objective=float(result.fun),model_outputs_used=False,labels_used=False,old_to_new_evaluation_cycle=dict(Counter(r['old_split']+'->'+r['evaluation_split'] for r in cycle_rows)),training_launched=False)
(OUT/'split_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
report=['# 17站训练、7站验证与测试：扩展训练划分','', 'GOB、DRA、IZA、TBL、SEL、GWN、CAB按52/17/17个周期划分训练、验证和测试。其余10站的86个案例全部用于训练，因此不再丢弃原验证、测试周期中的260个案例。','', '固定随机种子20260909，以年份、季度以及7个评估站的太阳几何合格小时数平衡三个集合。重叠的历史输入/目标窗口保持同组。划分不读取模型预测、误差或观测标签。','', '|集合|周期数|站点数|案例数|几何筛选潜在小时数|','|---|---:|---:|---:|---:|']
for s in SPLITS:
    v=summary[s];report.append(f"|{s}|{v['cycles']}|{v['stations']}|{v['cases']}|{v['geometry_eligible_hours']}|")
report+=['','几何筛选小时数尚未叠加地面有效性和SP历史观测条件，不能视为最终有效小时数。source_sample_id中的旧split前缀只作来源标识；实际实验集合以新split列为准，跨版本优先用case_key关联。','','## 重叠检查','']+[f'- {a}与{b}绑定分组。' for a,b in edges]+['','通过检查：1462个案例全部使用；训练1224个，验证119个，测试119个。验证和测试均恰含7站，每站分别17个案例，同一站点的案例及其历史/目标时间窗不跨集合。验证和测试的天气周期会在其他10个训练站中出现，这是本方案采用的跨站点训练设置。','','## 使用范围','', '模型归一化、专家与融合模块只用1224个训练案例拟合；早停和超参数选择只看7站验证集；论文最终指标只在7站测试集计算。训练时采用站点均衡采样，避免每站86个训练案例的10站压低每站52个训练案例的7站贡献。','', '配置：configs/dataset_split_17train_7eval_expanded_20260909.json。训练使用train_all17.csv，验证使用validation_7station.csv，最终测试使用test_7station.csv。']
(OUT/'数据划分说明.md').write_text('\n'.join(report)+'\n',encoding='utf-8');print(json.dumps(audit,ensure_ascii=True,indent=2))
if __name__=='__main__':pass
