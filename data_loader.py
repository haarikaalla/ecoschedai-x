from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path(__file__).parent; DATA=ROOT/"data"
def wet_bulb_stull(t,rh):
    return t*np.arctan(.151977*np.sqrt(rh+8.313659))+np.arctan(t+rh)-np.arctan(rh-1.676331)+.00391838*rh**1.5*np.arctan(.023101*rh)-4.686035
def sources():
    p=DATA/"status.json"
    return json.loads(p.read_text()) if p.exists() else [{"source":x,"state":"simulated (not refreshed)","rows":0,"range":"24-hour synthetic trace"} for x in ["Grid carbon API","Weather archive API","Public inference trace"]]
def weather():
    p=DATA/"weather.csv"
    if not p.exists(): return None
    d=pd.read_csv(p); d["wet_bulb_c"]=wet_bulb_stull(d.temperature_c,d.relative_humidity_percent); return d

