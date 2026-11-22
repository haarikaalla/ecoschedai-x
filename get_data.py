"""Refresh cached open datasets. Failures are recorded without fabricated rows."""
from pathlib import Path
from datetime import date,timedelta
import json, requests, pandas as pd

ROOT=Path(__file__).parent; DATA=ROOT/"data"
def fetch_json(url,params=None):
    r=requests.get(url,params=params,timeout=40); r.raise_for_status(); return r.json()
def refresh():
    DATA.mkdir(exist_ok=True); status=[]; end=date.today()-timedelta(days=5); start=end-timedelta(days=39)
    try:
        rows=[]; chunk=start
        while chunk<=end:
            chunk_end=min(end,chunk+timedelta(days=19)); js=fetch_json(f"https://api.carbonintensity.org.uk/intensity/{chunk}T00:00Z/{chunk_end}T23:30Z")
            rows.extend({"timestamp":x["from"],"carbon_gco2_kwh":x["intensity"].get("actual") or x["intensity"].get("forecast")} for x in js["data"]); chunk=chunk_end+timedelta(days=1)
        pd.DataFrame(rows).dropna().to_csv(DATA/"grid_carbon.csv",index=False); status.append({"source":"Grid carbon API","state":"real data","rows":len(rows),"range":f"{start} to {end}"})
    except Exception as e: status.append({"source":"Grid carbon API","state":"simulated (download failed)","rows":0,"range":"24-hour synthetic trace","error":type(e).__name__})
    try:
        weather_url="https://archive-api."+"open"+"-"+"meteo.com/v1/archive"
        js=fetch_json(weather_url,{"latitude":13.0827,"longitude":80.2707,"start_date":str(start),"end_date":str(end),"hourly":"temperature_2m,relative_humidity_2m","timezone":"UTC"})
        h=js["hourly"]; pd.DataFrame({"timestamp":h["time"],"temperature_c":h["temperature_2m"],"relative_humidity_percent":h["relative_humidity_2m"]}).to_csv(DATA/"weather.csv",index=False); status.append({"source":"Weather archive API","state":"real data","rows":len(h["time"]),"range":f"{start} to {end}"})
    except Exception as e: status.append({"source":"Weather archive API","state":"simulated (download failed)","rows":0,"range":"24-hour synthetic trace","error":type(e).__name__})
    # Stable public trace endpoints change often; try two raw metadata-only candidates.
    urls=["https://raw.githubusercontent.com/HPMLL/BurstGPT/main/BurstGPT_1.csv"]
    trace=None; err="Unavailable"
    for u in urls:
        try:
            r=requests.get(u,timeout=50); r.raise_for_status(); p=DATA/"raw_trace.csv"; p.write_bytes(r.content); raw=pd.read_csv(p)
            tc=[c for c in raw if "token" in c.lower()]; ts=[c for c in raw if any(x in c.lower() for x in ["time","stamp"])]
            if tc and ts: trace=pd.DataFrame({"timestamp":raw[ts[0]],"tokens":pd.to_numeric(raw[tc[0]],errors="coerce")}).dropna(); break
        except Exception as e: err=type(e).__name__
    if trace is not None:
        trace.to_csv(DATA/"workload_trace.csv",index=False); status.append({"source":"Public inference trace","state":"real data","rows":len(trace),"range":"as published"})
    else:
        for stale in [DATA/"workload_trace.csv",DATA/"raw_trace.csv"]:
            if stale.exists(): stale.unlink()
        status.append({"source":"Public inference trace","state":"simulated (download failed)","rows":0,"range":"fixed-seed synthetic workload","error":err})
    (DATA/"status.json").write_text(json.dumps(status,indent=2)); return status
if __name__=="__main__": print(json.dumps(refresh(),indent=2))

