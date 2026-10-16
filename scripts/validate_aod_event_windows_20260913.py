"""Build six-hour independent event windows from aerosol and radiation observations."""
import csv
from collections import defaultdict
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_aod_event_index_20260913 import rows_aeronet,aero550,noaa_points,excluded_dates,write_csv,PROTOCOL
from scripts.build_gfs_swradnet_dataset import load_stations,load_observation_minutes
from scripts.radiation_utils import solar_position,haurwitz_clear_sky_ghi
DATA=ROOT/'data/aod/event_screen_20260913';OUT=ROOT/'outputs/aod_event_set_20260913';UTC=timezone.utc


def aerosol_points(record):
    day=datetime.fromisoformat(record['date']).replace(tzinfo=UTC);sid=record['station_id'];values={}
    if record['source']=='AERONET_L2':
        path=DATA/f'{sid}_aeronet_points_{day:%Y-%m}.txt'
        if not path.exists():return {}
        for r in rows_aeronet(path):
            v=aero550(r)
            if v is None:continue
            t=datetime.strptime(r['Date(dd:mm:yyyy)']+' '+r['Time(hh:mm:ss)'],'%d:%m:%Y %H:%M:%S').replace(tzinfo=UTC)
            if t.date()==day.date():values[t]=v
    else:
        for name in record['raw_file'].split(';'):
            for t,v in noaa_points(ROOT/name):
                if t.date()==day.date():values[t]=v
    return values


def hour_qc(start,obs,aod,station):
    times=[start+timedelta(minutes=k) for k in range(60)]
    values=np.array([[np.nan if v is None else v for v in obs.get(t,(None,None,None))] for t in times],float)
    ghi,dni,dhi=values.T
    centres=np.array([np.datetime64(t.replace(tzinfo=None))+np.timedelta64(30,'s') for t in times])
    mu,dist=solar_position(centres,station.latitude,station.longitude,clip_below_horizon=False)
    # Use the solar curve only to remove deterministic shape, without a GHI-level threshold.
    reference=haurwitz_clear_sky_ghi(mu,dist).astype(float)
    day=mu>=np.cos(np.deg2rad(85.))
    gv=np.isfinite(ghi)&(ghi>=0)&(ghi<=1500)&day
    cv=gv&np.isfinite(dni)&np.isfinite(dhi)&(dni>=0)&(dni<=1500)&(dhi>=0)&(dhi<=1500)
    pair=gv[:-1]&gv[1:]
    ratio=np.where(gv,ghi/np.maximum(reference,20),np.nan)
    changes=np.abs(np.diff(ratio))[pair]
    scale=np.nanmedian(ratio) if gv.any() else np.nan
    smooth=float(np.quantile(changes,.90)/max(scale,.01)) if len(changes)>=30 else None
    direct=float(np.mean(dni[cv]>=50.)) if cv.any() else None
    closure=float(np.median(np.abs(ghi[cv]-dni[cv]*mu[cv]-dhi[cv])/np.maximum(ghi[cv],100.))) if cv.any() else None
    ap=sorted((t,v) for t,v in aod.items() if start<=t<start+timedelta(hours=1))
    gaps=np.diff([0.,*[(t-start).total_seconds()/60 for t,v in ap],60.])
    med=float(np.median([v for t,v in ap])) if ap else None
    aq=len(ap)>=3 and max(gaps)<=20
    ground=gv.sum()>=54 and cv.sum()>=45
    clear=ground and smooth is not None and smooth<=.02 and direct>=.90 and closure<=.08
    selected=clear and aq and med>=.30
    return dict(valid_time_utc=(start+timedelta(hours=1)).isoformat(),ghi_minutes=int(gv.sum()),component_minutes=int(cv.sum()),
        mean_ghi=float(np.mean(ghi[gv])) if gv.any() else None,aod_points=len(ap),aod_max_gap_minutes=float(max(gaps)),
        median_aod550=med,p90_relative_ghi_increment=smooth,direct_beam_fraction=direct,closure_relative_error=closure,
        ground_usable=bool(ground),smooth_sun_visible=bool(clear),aerosol_usable=bool(aq),high_aod_clear_hour=bool(selected))


def main():
    rows=list(csv.DictReader((OUT/'daily_candidates_new_dates.csv').open()))
    stations=load_stations(ROOT/'configs/gfs_swradnet_stations.csv');bygroup=defaultdict(list)
    for r in rows:bygroup[(r['station_id'],r['date'][:7])].append(r)
    hours=[];cases=[];days=[]
    for (sid,month),records in sorted(bygroup.items()):
        dates=[datetime.fromisoformat(r['date']).replace(tzinfo=UTC) for r in records]
        ends=[d+timedelta(hours=h) for d in dates for h in range(1,25)]
        obs=load_observation_minutes(stations[sid],ends,ROOT/'data/radiation_observations/raw')
        for rec,day in zip(records,dates):
            ap=aerosol_points(rec)
            rr=[hour_qc(day+timedelta(hours=h),obs,ap,stations[sid]) for h in range(24)]
            for h in rr:hours.append(dict(station_id=sid,date=rec['date'],source=rec['source'],**h))
            windows=[]
            for start in range(19):
                group=rr[start:start+6]
                if not all(r['ground_usable'] for r in group):continue
                high=sum(r['high_aod_clear_hour'] for r in group)
                if high<4:continue
                # Coverage first, then closeness to local solar noon; no error or AOD magnitude ranking.
                noon_distance=abs((start+3+stations[sid].longitude/15)-12)
                windows.append((high,noon_distance,start,group))
            days.append(dict(**rec,observed_aod_points=len(ap),qualifying_hours=sum(r['high_aod_clear_hour'] for r in rr),window_available=bool(windows)))
            if not windows:continue
            high,_,start,group=sorted(windows,key=lambda v:(-v[0],v[1],v[2]))[0]
            issue=day+timedelta(hours=start)
            cases.append(dict(case_id=f'AOD_EVENT_{sid}_{issue:%Y%m%dT%H}',station_id=sid,date=rec['date'],
                 issue_time_utc=issue.isoformat(),end_time_utc=(issue+timedelta(hours=6)).isoformat(),source=rec['source'],
                 high_aod_clear_hours=high,high_aod_leads=','.join(str(h+1) for h,r in enumerate(group) if r['high_aod_clear_hour']),
                 median_aod550=float(np.median([r['median_aod550'] for r in group if r['median_aod550'] is not None])),
                 mean_ghi=float(np.mean([r['mean_ghi'] for r in group])),forecast_inputs_ready=False))
        print(sid,month,'candidate days',len(records),'windows so far',len(cases),flush=True)
    # Cluster dates globally; nearby days/stations remain in the same event partition.
    dates=sorted({r['date'] for r in cases});clusters=[]
    for day in dates:
        if not clusters or (datetime.fromisoformat(day)-datetime.fromisoformat(clusters[-1][-1])).days>3:clusters.append([day])
        else:clusters[-1].append(day)
    mapping={d:(f'event_{c[0]}', 'event_development' if int(hashlib.sha256(('aod_event_v1_'+c[0]).encode()).hexdigest(),16)%2==0 else 'event_confirmation') for c in clusters for d in c}
    for r in cases:r['event_id'],r['partition']=mapping[r['date']]
    write_csv(OUT/'hourly_observation_qc.csv',hours);write_csv(OUT/'candidate_day_audit.csv',days)
    write_csv(OUT/'event_cases_6h.csv',cases)
    for part in ('event_development','event_confirmation'):write_csv(OUT/f'{part}.csv',[r for r in cases if r['partition']==part])
    banned=excluded_dates();assert all(datetime.fromisoformat(r['date']).date() not in banned for r in cases)
    parts={p:{r['event_id'] for r in cases if r['partition']==p} for p in ('event_development','event_confirmation')}
    assert not parts['event_development']&parts['event_confirmation']
    summary=dict(candidate_days=len(days),cases=len(cases),independent_event_clusters=len(clusters),
        stations={s:sum(r['station_id']==s for r in cases) for s in PROTOCOL['stations']},
        partitions={p:dict(cases=sum(r['partition']==p for r in cases),event_clusters=len(e)) for p,e in parts.items()},
        forecasting_cases_ready=0,uses_prediction_errors=False,old_weather_exclusion_passed=True,
        scope='observation-qualified near-clear/high-AOD event cohort; forecast inputs pending')
    (OUT/'event_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
