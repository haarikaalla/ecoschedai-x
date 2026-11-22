"""Transparent simulated agent-side efficiency accounting."""
import numpy as np
import pandas as pd
from simulator import gpu_model

def batch_factor(efficiency):
    base=gpu_model(.5,1.,False)[0]; batched=gpu_model(.8,1.,True)[0]
    return 1-float(efficiency)*(1-batched/base)

def lever_accounting(p=.85,f=.30,cache_hit=.20,batch_eff=.60,retry_budget=2):
    cascade=f+1-p
    batching=batch_factor(batch_eff)
    retry_factor=(1+.05*retry_budget)/(1+.05*6)
    total=cascade*(1-cache_hit)*batching*retry_factor
    return {"energy_factor":total,"cascade_factor":cascade,"batch_factor":batching,"retry_factor":retry_factor,
            "escalation_rate":1-p,"failed_attempt_factor":(1-p)*f*(1-cache_hit),"quality":100.}

def configurations(run_now,infrastructure,p=.85,f=.30,cache_hit=.20,batch_eff=.60,retry_budget=2,steps=60):
    a=lever_accounting(p,f,cache_hit,batch_eff,retry_budget);rows=[]
    def add(name,base,agent):
        factor=a["energy_factor"] if agent else 1.;completion=base["Mean completion time (h)"]
        if agent:completion=completion*(1-.18*cache_hit)+(1-p)*.35+batch_eff*.15-.05*(6-retry_budget)
        deadlines=100*(1-base["Deadline misses (steps)"]/max(steps,1));deadlines=max(0,deadlines-(max(0,.75-p)*5 if agent else 0))
        rows.append({"Configuration":name,"Energy (kWh / 1,000 tasks)":base["Energy (kWh / 1,000 steps)"]*factor,
                     "Water (L / 1,000 tasks)":base["Water total, stress-weighted (L / 1,000 steps)"]*factor,
                     "Carbon (kgCO2e / 1,000 tasks)":base["Carbon (kgCO2e / 1,000 steps)"]*factor,
                     "Completion time (h)":max(.05,completion),"Quality maintained (%)":100.,"Deadlines met (%)":deadlines})
    add("Run now",run_now,False);add("Infrastructure only",infrastructure,False);add("Agent-side only",run_now,True);add("Both",infrastructure,True)
    return pd.DataFrame(rows),a

def batch_curve(efficiency=.60):
    base=gpu_model(.5,1.,False)[0];rows=[]
    for size in [1,2,4,8]:
        util=min(.9,.5+.06*(size-1));energy=gpu_model(util,1.,size>1)[0];effective=base-(base-energy)*efficiency
        rows.append({"Batch size":size,"GPU utilisation (%)":util*100,"Energy per equal-work step (kWh)":effective,"Change vs single (%)":100*(effective/base-1)})
    return pd.DataFrame(rows)
