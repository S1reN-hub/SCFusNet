"""Observation-only daily candidate index with exclusion of previously used weather periods."""
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/aod/event_screen_20260913'
OUT=ROOT/'outputs/aod_event_set_20260913'
UTC=timezone.utc
PROTOCOL=dict(version='aod_event_cohort_v1',stations=['CAB','DRA','GOB','GWN','IZA','SEL','TBL'],
    years=[2021,2025],daily_discovery_aod550_min=.30,
    selection_not_based_on_prediction_errors=True,
    old_weather_exclusion='global UTC dates from each old GFS cycle day minus 1 through plus 4 inclusive',
    wavelength='AERONET AOD500 converted with 440-870 Angstrom exponent; NOAA actual wavelengths and Angstrom exponent',
    primary_aerosol_quality='AERONET Level2; NOAA MFRSR cloud-screen flag=0',
    event_hours=6,minimum_high_aod_hours=4,minimum_aod_points_per_hour=3,maximum_aod_gap_minutes=20,
    min_ghi_minutes=54,min_component_minutes=45,max_sza_deg=85.,min_direct_beam_Wm2=50.,
    direct_beam_fraction_min=.90,p90_relative_ghi_increment_max=.02,median_closure_relative_error_max=.08,
    clear_definition='cloud-screened direct-sun aerosol measurements plus smooth GHI and persistent direct beam; not proof of cloud-free entire sky',
    partition='group UTC event days within 72h across stations; deterministic hash into development/confirmation 50:50',
    data_role='ground AOD/cloud evidence define evaluation strata; forecast inputs must be available at issue time',
    inference_ready_requires=['new satellite history','GFS/WRF forecast tensors','CAMS forecast matched to issue time'])


def rows_aeronet(path):
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    index=next((i for i,s in enumerate(lines) if s.startswith('AERONET_Site,Date')),None)
    if index is None:return []
    return list(csv.DictReader(lines[index:]))


def aero550(r):
    try:
        a=float(r['AOD_500nm']);ae=float(r['440-870_Angstrom_Exponent'])
        if a<=0 or ae<=-9 or not math.isfinite(a+ae):return None
        return a*(550./500.)**(-ae)
    except (KeyError,ValueError):return None


def noaa_points(path):
    lines=path.read_text().splitlines()
    if len(lines)<7:return []
    fields=lines[1].split();date=datetime(int(fields[3]),int(fields[2]),int(fields[1]),tzinfo=UTC)
    wavelengths=list(map(float,lines[2].split()[:5]));sid=path.name[:3].upper()
    offset=dict(DRA=8,GWN=6,TBL=7)[sid]
    result=[]
    for line in lines[6:]:
        f=line.split()
        if len(f)<14:continue
        try:
            if int(f[1])!=0:continue
            a,b=float(f[3]),float(f[6]);ae=float(f[13])
            if a<=0 or b<=0 or ae<=-9:continue
            # Spectral slope computed from the actual channels avoids a nominal wavelength mismatch.
            alpha=-math.log(a/b)/math.log(wavelengths[1]/wavelengths[4])
            aod=a*(550/wavelengths[1])**(-alpha)
            hhmm=int(f[0]);t=date+timedelta(hours=hhmm//100+offset,minutes=hhmm%100)
            result.append((t,aod))
        except (ValueError,OverflowError):continue
    return result


def excluded_dates():
    p=ROOT/'outputs/dataset_split_17train_7eval_expanded_20260909/case_assignments.csv'
    rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
    dates=set()
    for cycle in {r['cycle'] for r in rows}:
        d=datetime.strptime(cycle,'%Y%m%d%H').date()
        dates.update(d+timedelta(days=k) for k in range(-1,5))
    return dates


def write_csv(path,rows):
    if not rows:path.write_text('',encoding='utf-8');return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True);banned=excluded_dates()
    (OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    (OUT/'excluded_utc_dates.txt').write_text('\n'.join(sorted(str(d) for d in banned)),encoding='utf-8')
    records=[];coverage={}
    for sid in ('CAB','GOB','IZA','SEL','TBL'):
        p=DATA/f'{sid}_aeronet_daily_2021_2025.txt';rows=rows_aeronet(p)
        valid=[]
        for r in rows:
            day=datetime.strptime(r['Date(dd:mm:yyyy)'],'%d:%m:%Y').date();v=aero550(r)
            if not 2021<=day.year<=2025 or v is None:continue
            valid.append((day,v))
            if v>=.30:records.append(dict(station_id=sid,date=str(day),daily_aod550=v,source='AERONET_L2',raw_file=str(p.relative_to(ROOT)),excluded=day in banned))
        coverage[sid]=dict(aeronet_valid_days=len(valid),aeronet_status='available' if valid else 'no_Level2_data_in_requested_period')
    for sid in ('DRA','GWN','TBL'):
        files=list((DATA/'noaa'/sid.lower()).glob('*.aod'));count=0;byday={};paths={}
        for p in files:
            points=noaa_points(p)
            for t,v in points:
                byday.setdefault(t.date(),{})[t]=v
                paths.setdefault(t.date(),set()).add(str(p.relative_to(ROOT)))
        for day,values in byday.items():
            if not 2021<=day.year<=2025:continue
            if len(values)<20:continue
            count+=1;v=float(np.mean(list(values.values())))
            if v>=.30:records.append(dict(station_id=sid,date=str(day),daily_aod550=v,source='NOAA_MFRSR_flag0',raw_file=';'.join(sorted(paths[day])),excluded=day in banned))
        coverage.setdefault(sid,{}).update(noaa_files=len(files),noaa_valid_days=count)
    # A source day can cross UTC midnight; preserve one strongest record per station/day.
    unique={}
    for r in records:
        key=(r['station_id'],r['date'])
        if key not in unique or r['daily_aod550']>unique[key]['daily_aod550']:unique[key]=r
    records=sorted(unique.values(),key=lambda r:(r['station_id'],r['date']))
    write_csv(OUT/'daily_candidates_all.csv',records)
    admitted=[r for r in records if not r['excluded']]
    write_csv(OUT/'daily_candidates_new_dates.csv',admitted)
    (OUT/'source_coverage.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
    print(json.dumps(dict(coverage=coverage,candidates=len(records),new_dates=len(admitted),
        new_by_station={s:sum(r['station_id']==s for r in admitted) for s in PROTOCOL['stations']}),indent=2))


if __name__=='__main__':main()
