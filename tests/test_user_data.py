import numpy as np
import pandas as pd
from simulator import Settings,simulate,set_user_data,summary

def test_user_trace_and_workload_recompute():
    h=np.arange(24);t=np.full(24,25.0);wb=np.full(24,18.0)
    trace=pd.DataFrame({"hour":h,"carbon":np.linspace(500,100,24),"temp":t,"wet_bulb":wb,"pue":1.27,"water_intensity":.7})
    jobs=pd.DataFrame({"arrival":[6,7],"deadline":[12,13],"restricted":[True,False],"it_energy_kwh":[1.5,3.0]})
    set_user_data(trace,jobs)
    d,_=simulate(Settings(data_version="test"),"Balanced EcoSchedAI-X")
    assert len(d)==2 and d.completed.all() and d.security_violation.sum()==0
    assert summary(d)["Energy (kWh / 1,000 steps)"]>0
    set_user_data(None,None)

def test_static_run_now_baseline():
    d,_=simulate(Settings(),"Run now + static settings")
    assert len(d)==60 and d.completed.all() and (d.site=="Site A").all()
