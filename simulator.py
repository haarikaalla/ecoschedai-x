"""Deterministic metadata-only resource scheduling model."""
from dataclasses import dataclass, replace
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

POLICIES=["Run now","Run now + static settings","Energy-only","Carbon-only","Water-only","Balanced EcoSchedAI-X"]
_TRACE_OVERRIDE=None
_WORKLOAD_OVERRIDE=None

@dataclass(frozen=True)
class Settings:
    seed:int=42; window:int=8; idle_w:float=10.; error:float=0.; scenario:str="Conflict"
    offsite:bool=True; weights:tuple=(1.,1.,1.); controls:bool=True; routing:bool=True
    shadow:bool=False; max_latency:float=12.; power_limit:float=16.; fallback_threshold:float=.20
    power_caps:bool=True; cooling_choice:bool=True; shield_enabled:bool=True; data_version:str="default"

def set_user_data(trace=None,jobs=None):
    global _TRACE_OVERRIDE,_WORKLOAD_OVERRIDE
    _TRACE_OVERRIDE=trace.copy() if trace is not None else None
    _WORKLOAD_OVERRIDE=jobs.copy() if jobs is not None else None

def base_traces(cfg):
    if _TRACE_OVERRIDE is not None:
        return _TRACE_OVERRIDE.copy()
    h=np.arange(24); t=26+8*np.exp(-((h-15)/3.5)**2)
    wb=21+6*np.exp(-((h-12)/3)**2) if cfg.scenario=="Conflict" else 27-6*np.exp(-((h-13)/3)**2)
    c=460-330*np.exp(-((h-13)/2.5)**2)+50*np.exp(-((h-20)/2)**2)-40*(h<5)
    return pd.DataFrame({"hour":h,"carbon":c,"temp":t,"wet_bulb":wb,"pue":1.15+.012*np.maximum(t-15,0),"water_intensity":.1+.1*np.maximum(wb-12,0)})

def site_traces(cfg):
    b=base_traces(cfg); out=[]
    for i,(cm,to,ws,ne,nl) in enumerate([(1.,0,1.,0.,0.),(.72,-4,1.8,.08,.15),(1.18,3,2.5,.16,.30)]):
        x=b.copy(); x["site"]=f"Site {chr(65+i)}"; x["carbon"]*=cm; x["temp"]+=to
        x["pue"]=1.15+.012*np.maximum(x.temp-15,0); x["water_stress"]=ws; x["network_energy"]=ne; x["network_latency"]=nl; out.append(x)
    return pd.concat(out,ignore_index=True)

def workload(cfg):
    if _WORKLOAD_OVERRIDE is not None:
        x=_WORKLOAD_OVERRIDE.copy(); x["job"]=np.arange(len(x))
        if "tokens" not in x:x["tokens"]=1000
        if "restricted" not in x:x["restricted"]=False
        if "it_energy_kwh" not in x:x["it_energy_kwh"]=2.
        return x
    r=np.random.default_rng(cfg.seed); a=np.sort(r.integers(6,11,60))
    return pd.DataFrame({"job":np.arange(60),"arrival":a,"deadline":np.minimum(23,a+cfg.window),"restricted":np.arange(60)%5==0,"tokens":r.integers(300,2400,60),"it_energy_kwh":2.})

def gpu_model(util=.5,cap=1.,consolidate=False):
    idle,maximum,alpha=.35,2.8,1.4; util=.8 if consolidate else util
    requested=idle+(maximum-idle)*util**alpha; power=min(requested,maximum*cap)
    duration=.72/(util*(power/requested)**.7); energy=power*duration*(.78 if consolidate else 1.)
    return energy,power,duration

def _option(cfg,trace,job,h,site,cooling,enhanced,actual=False):
    lookup=trace.attrs["lookup"]; t=lookup[(site,int(h))]; si=ord(site[-1])-65
    consolidate=enhanced and cfg.power_caps; cap=.8 if consolidate else 1.; it,power,duration=gpu_model(.5,cap,consolidate)
    scale=float(job.get("it_energy_kwh",2.))/2.; it*=scale; duration*=scale
    pue=t.pue+(.13 if cooling=="Dry/chiller" else 0); wi=.08 if cooling=="Dry/chiller" else t.water_intensity
    if actual:
        r=np.random.default_rng(cfg.seed*1009+h*31+si*7); pue*=1+r.uniform(-cfg.error,cfg.error); wi*=1+r.uniform(-cfg.error,cfg.error); carbon=t.carbon*(1+r.uniform(-cfg.error,cfg.error))
    else: carbon=t.carbon
    wait=max(0,h-job.arrival); hold=cfg.idle_w/1000*wait*(.1 if consolidate else 1.)
    network=t.network_energy if site!="Site A" else 0.; energy=it*pue+hold+network
    onsite=it*wi; offsite=energy*4.52 if cfg.offsite else 0.; water=(onsite+offsite)*t.water_stress
    carb=(it*pue*carbon+network*carbon+hold*lookup[("Site A",int(job.arrival))].carbon)/1000
    latency=wait+duration+t.network_latency
    return dict(hour=int(h),site=site,cooling=cooling,consolidated=consolidate,power_cap=cap,power_kw=power,duration=duration,energy=energy,onsite=onsite,offsite=offsite,water=water,carbon=carb,latency=latency)

def _schedule(cfg,policy):
    jobs=workload(cfg); tr=site_traces(cfg); tr.attrs["lookup"]={(r.site,int(r.hour)):r for _,r in tr.iterrows()}
    enhanced=policy in ["Balanced EcoSchedAI-X","Run now + static settings"] and cfg.controls
    static=policy=="Run now + static settings"
    # Expanded site-hour capacity slots; assignment applies the cap inside optimisation.
    slots=[(s,h,k) for s in ["Site A","Site B","Site C"] for h in range(24) for k in range(6)]
    costs=np.full((len(jobs),len(slots)),1e6); opts={}
    for ji,j in jobs.iterrows():
        for qi,(site,h,_) in enumerate(slots):
            allowed=h>=j.arrival and h<=j.deadline and (site=="Site A" or (enhanced and not static and cfg.routing and not j.restricted))
            if not allowed: continue
            candidates=[]
            cooling_modes=["Evaporative","Dry/chiller"] if enhanced and cfg.cooling_choice and not static else ["Evaporative"]
            for cooling in cooling_modes:
                o=_option(cfg,tr,j,h,site,cooling,enhanced)
                shield=(not cfg.shield_enabled) or (o["latency"]<=cfg.max_latency and o["power_kw"]<=cfg.power_limit and not(j.restricted and site!="Site A"))
                if shield:candidates.append(o)
            if not candidates:continue
            vals=np.array([[o[x] for x in ["energy","water","carbon"]] for o in candidates])
            w={"Energy-only":(1,0,0),"Water-only":(0,1,0),"Carbon-only":(0,0,1)}.get(policy,cfg.weights)
            if policy in ["Run now","Run now + static settings"]:score=np.array([h*1e3+qi/1e6]*len(candidates))
            else:score=(vals/np.array([3.,20.,1.]))@np.array(w)
            bi=int(np.argmin(score)); opts[ji,qi]=candidates[bi]
            # Global carbon/energy/water objective with a tiny wait tie-break.
            raw={"Energy-only":"energy","Water-only":"water","Carbon-only":"carbon"}.get(policy)
            costs[ji,qi]=(opts[ji,qi][raw] if raw else score[bi])+1e-7*h
    rows,cols=linear_sum_assignment(costs); chosen={r:c for r,c in zip(rows,cols)}
    out=[]
    for ji,j in jobs.iterrows():
        qi=chosen.get(ji); o=opts.get((ji,qi))
        fallback=cfg.shield_enabled and cfg.error>cfg.fallback_threshold
        if fallback or o is None:
            viable=[(q,op) for (jj,q),op in opts.items() if jj==ji and op["site"]=="Site A"]
            if viable:qi,o=min(viable,key=lambda z:z[1]["hour"])
        if o is None:
            out.append({**j.to_dict(),"completed":False,"deadline_miss":True,"reason":"No shield-feasible slot"}); continue
        actual=_option(cfg,tr,j,o["hour"],o["site"],o["cooling"],enhanced,True)
        cf=_option(cfg,tr,j,j.arrival,"Site A","Evaporative",False,True)
        action=o if not cfg.shadow else cf
        reason="Uncertainty fallback to run now" if fallback else ("Shadow recommendation" if cfg.shadow else "Minimum shield-feasible cost")
        out.append({**j.to_dict(),**action,"completed":True,"deadline_miss":action["hour"]>j.deadline,"wait":action["hour"]-j.arrival,
                    "security_violation":bool(j.restricted and action["site"]!="Site A"),"reason":reason,"recommended_hour":o["hour"],"counterfactual_energy":cf["energy"],"counterfactual_water":cf["water"],"counterfactual_carbon":cf["carbon"]})
    return pd.DataFrame(out),tr

def simulate(cfg,policy): return _schedule(cfg,policy)

def summary(df):
    ok=df[df.completed]; sc=1000/max(len(ok),1)
    return {"Energy (kWh / 1,000 steps)":ok.energy.sum()*sc,"Water on-site (L / 1,000 steps)":ok.onsite.sum()*sc,
            "Water off-site (L / 1,000 steps)":ok.offsite.sum()*sc,"Water total, stress-weighted (L / 1,000 steps)":ok.water.sum()*sc,
            "Carbon (kgCO2e / 1,000 steps)":ok.carbon.sum()*sc,"Mean added wait (h)":ok.wait.mean(),"Mean completion time (h)":ok.latency.mean(),
            "Deadline misses (steps)":int(df.deadline_miss.sum()),"Restricted-routing violations (steps)":int(ok.security_violation.sum())}

def compare(cfg): return pd.DataFrame({p:summary(simulate(cfg,p)[0]) for p in POLICIES}).T

def ledger(cfg):
    rows=[]
    for seed in range(10):
        c=replace(cfg,seed=seed); b=summary(simulate(c,"Run now")[0]); x=summary(simulate(c,"Balanced EcoSchedAI-X")[0])
        for m in ["Energy (kWh / 1,000 steps)","Water total, stress-weighted (L / 1,000 steps)","Carbon (kgCO2e / 1,000 steps)"]:
            rows.append({"seed":seed,"metric":m,"saving_percent":100*(b[m]-x[m])/b[m]})
    d=pd.DataFrame(rows); g=d.groupby("metric").saving_percent
    return pd.DataFrame({"Mean simulated saving (%)":g.mean(),"95% CI low (%)":g.mean()-1.96*g.std()/np.sqrt(10),"95% CI high (%)":g.mean()+1.96*g.std()/np.sqrt(10)}).reset_index()

def audit_csv(df):
    cols=["job","arrival","deadline","restricted","tokens","recommended_hour","hour","site","cooling","power_cap","latency","energy","water","carbon","reason","counterfactual_energy","counterfactual_water","counterfactual_carbon"]
    return df[[c for c in cols if c in df]].to_csv(index=False).encode()

def sanity_test():
    cfg=Settings(error=0,idle_w=0,controls=False,routing=False); b=summary(simulate(cfg,"Run now")[0]); c=summary(simulate(cfg,"Carbon-only")[0])
    ok=c["Carbon (kgCO2e / 1,000 steps)"]<=b["Carbon (kgCO2e / 1,000 steps)"]+1e-9
    return {"passed":ok,"run_now":b["Carbon (kgCO2e / 1,000 steps)"],"carbon_only":c["Carbon (kgCO2e / 1,000 steps)"],"checks":"capacity applied during assignment; schedule and metrics share hour index"}
