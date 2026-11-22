from dataclasses import replace
from pathlib import Path
import hashlib, io, numpy as np, pandas as pd, altair as alt, streamlit as st
from simulator import Settings,POLICIES,simulate,summary,audit_csv,sanity_test,base_traces,set_user_data,site_traces,_option,workload
from data_loader import sources,wet_bulb_stull
from agent_coordination import configurations,lever_accounting,batch_curve

ROOT=Path(__file__).parent
COLORS={"Run now":"#111111","Run now + static settings":"#6B7280","Energy-only":"#176B87","Carbon-only":"#4E735D","Water-only":"#279AB5","Balanced EcoSchedAI-X":"#7A4EAB"}
METRICS={"Energy":"Energy (kWh / 1,000 steps)","Water":"Water total, stress-weighted (L / 1,000 steps)","Carbon":"Carbon (kgCO2e / 1,000 steps)","Completion time":"Mean completion time (h)"}
FOOTER="Simulated results, not measurements. See Data sources."
st.set_page_config(page_title="Sustainable Scaling of AI Data Centers",layout="wide")
st.markdown("""<style>.block-container{padding-top:1.1rem;max-width:1500px}h1{font-size:2.15rem!important}#MainMenu,footer,[data-testid=stToolbar],.stDeployButton{visibility:hidden!important;height:0!important}.statusbar{background:#eef5f7;border-left:4px solid #176b87;padding:.65rem .8rem;margin:.4rem 0 1rem;font-size:.9rem}.metric-card{background:#f5f7f8;border:1px solid #dce4e8;border-radius:9px;padding:.8rem;min-height:120px}.metric-label{font-size:.86rem;color:#52636d}.metric-value{font-size:1.35rem;font-weight:600;white-space:normal}.metric-delta{font-size:.82rem;margin:.18rem 0}.metric-base{font-size:.76rem;color:#60727c}.footer-note{border-top:1px solid #dfe6e9;margin-top:1.2rem;padding-top:.55rem;color:#60727c;font-size:.78rem}</style>""",unsafe_allow_html=True)
for key,value in {"cascade_p":.85,"cascade_f":.30,"cache_hit":.20,"batch_eff":.60,"retry_budget":2}.items():
    if key not in st.session_state:st.session_state[key]=value

def footer():st.markdown(f'<div class="footer-note">{FOOTER}</div>',unsafe_allow_html=True)
def pct(v,b):return 100*(v/b-1) if b else 0
def read_upload(u):
    if u is None:return None,""
    raw=u.getvalue();return pd.read_csv(io.BytesIO(raw)),hashlib.sha1(raw).hexdigest()[:10]
def cached_trace():
    cp,wp=ROOT/"data/grid_carbon.csv",ROOT/"data/weather.csv"
    if not(cp.exists() and wp.exists()):return None,"default"
    c,w=pd.read_csv(cp),pd.read_csv(wp);c["timestamp"]=pd.to_datetime(c.timestamp,utc=True);c["hour"]=c.timestamp.dt.floor("h");c=c.groupby("hour",as_index=False).carbon_gco2_kwh.mean().tail(24);w=w.tail(24)
    if len(c)<24 or len(w)<24:return None,"default"
    t=w.temperature_c.to_numpy(float);rh=w.relative_humidity_percent.to_numpy(float);wb=wet_bulb_stull(t,rh)
    x=pd.DataFrame({"hour":range(24),"carbon":c.carbon_gco2_kwh.to_numpy(),"temp":t,"wet_bulb":wb});x["pue"]=1.15+.012*np.maximum(t-15,0);x["water_intensity"]=.1+.1*np.maximum(wb-12,0)
    return x,hashlib.sha1(pd.util.hash_pandas_object(x).values.tobytes()).hexdigest()[:10]
def prepare_inputs():
    errors=[];trace,ver=cached_trace();jobs=None;src={x["source"]:x for x in sources()};base=trace.copy() if trace is not None else base_traces(Settings())
    cf,cv=read_upload(st.session_state.get("carbon_upload"));wf,wv=read_upload(st.session_state.get("weather_upload"));jf,jv=read_upload(st.session_state.get("workload_upload"));paste=st.session_state.get("paste_carbon","").strip()
    if paste:
        try:
            vals=[float(v) for v in paste.replace("\n",",").split(",") if v.strip()]
            if len(vals)!=24:raise ValueError("Paste exactly 24 numeric carbon values.")
            cf=pd.DataFrame({"carbon_gco2_kwh":vals});cv=hashlib.sha1(paste.encode()).hexdigest()[:10]
        except Exception as e:errors.append(str(e))
    if cf is not None:
        if "carbon_gco2_kwh" not in cf or len(cf)<24:errors.append("Carbon CSV needs carbon_gco2_kwh and at least 24 rows.")
        else:base["carbon"]=pd.to_numeric(cf.carbon_gco2_kwh,errors="coerce").tail(24).to_numpy();ver+=cv
    if wf is not None:
        if not {"temperature_c","relative_humidity_percent"}.issubset(wf) or len(wf)<24:errors.append("Weather CSV needs temperature_c, relative_humidity_percent and at least 24 rows.")
        else:
            q=wf.tail(24);t=pd.to_numeric(q.temperature_c,errors="coerce").to_numpy();rh=pd.to_numeric(q.relative_humidity_percent,errors="coerce").to_numpy();base["temp"]=t;base["wet_bulb"]=wet_bulb_stull(t,rh);base["pue"]=1.15+.012*np.maximum(t-15,0);base["water_intensity"]=.1+.1*np.maximum(base.wet_bulb-12,0);ver+=wv
    if jf is not None:
        need={"timestamp","deferrable_flag","deadline_hours","restricted_flag"}
        if not need.issubset(jf) or not ({"it_energy_kwh","duration_hours"}&set(jf)):errors.append("Workload CSV needs timestamp, deferrable_flag, deadline_hours, restricted_flag, and it_energy_kwh or duration_hours.")
        else:
            flag=jf.deferrable_flag.astype(str).str.lower().isin(["true","1","yes"]);q=jf[flag].copy();ts=pd.to_datetime(q.timestamp,errors="coerce");q["arrival"]=ts.dt.hour;q["deadline"]=np.minimum(23,q.arrival+pd.to_numeric(q.deadline_hours,errors="coerce").fillna(0).astype(int));q["restricted"]=q.restricted_flag.astype(str).str.lower().isin(["true","1","yes"]);q["it_energy_kwh"]=pd.to_numeric(q["it_energy_kwh"] if "it_energy_kwh" in q else q.duration_hours*2,errors="coerce").fillna(2);jobs=q[["arrival","deadline","restricted","it_energy_kwh"]].reset_index(drop=True);ver+=jv
    cs="User data" if cf is not None and not any("Carbon" in e or "24 numeric" in e for e in errors) else ("real data, grid carbon API, "+src.get("Grid carbon API",{}).get("range","") if trace is not None else "simulated (download failed)")
    ws="User data" if wf is not None and not any("Weather" in e for e in errors) else ("real data, weather archive, "+src.get("Weather archive API",{}).get("range","") if trace is not None else "simulated (download failed)")
    js="User data" if jf is not None and not any("Workload" in e for e in errors) else src.get("Public inference trace",{}).get("state","simulated")
    return base,jobs,ver,errors,cs,ws,js

trace,jobs,version,input_errors,carbon_status,weather_status,workload_status=prepare_inputs();set_user_data(trace,jobs)
@st.cache_data(show_spinner=False,max_entries=192)
def run_policy(cfg,policy,version):return simulate(cfg,policy)[0]
def preset(name):
    st.session_state.energy_weight,st.session_state.water_weight,st.session_state.carbon_weight={"Energy-first":(3.,.5,.5),"Water-first":(.5,3.,.5),"Carbon-first":(.5,.5,3.),"Balanced":(1.,1.,1.)}[name]

st.title("Sustainable Scaling of AI Data Centers")
st.subheader("Reducing the Energy and Water Footprint of Agentic AI Workloads")
st.write("Agentic AI multiplies compute per request, so operators need to cut energy, water and carbon per completed task without breaking latency, deadlines or data-security rules.")
st.markdown(f'<div class="statusbar"><b>Carbon:</b> {carbon_status} | <b>Weather:</b> {weather_status} | <b>Workload:</b> {workload_status} | GPU, cooling, controller and agent coordination: simulated assumptions (cascade p={st.session_state.cascade_p:.2f}, f={st.session_state.cascade_f:.2f}; cache={st.session_state.cache_hit:.0%}; batch effectiveness={st.session_state.batch_eff:.0%})</div>',unsafe_allow_html=True)
with st.sidebar:
    st.header("Controls");scenario=st.selectbox("Scenario",["Conflict","Improve together"]);window=st.slider("Start flexibility (h)",1,12,8);idle=st.slider("Waiting power (W / step)",0,100,10);error=st.slider("Forecast error bound (%)",0,30,0);offsite=st.toggle("Include off-site water assumption",True);controls=st.toggle("Power and cooling controls",True);routing=st.toggle("Multi-site routing",True);shadow=st.toggle("Shadow mode: recommend only",False);st.caption("Off-site water assumption: 4.52 L / facility kWh")
    st.write("Presets");pc=st.columns(2)
    for i,name in enumerate(["Energy-first","Water-first","Carbon-first","Balanced"]):pc[i%2].button(name,use_container_width=True,on_click=preset,args=(name,))
    weights=(st.slider("Energy weight",0.,3.,st.session_state.get("energy_weight",1.),.1,key="energy_weight"),st.slider("Water weight",0.,3.,st.session_state.get("water_weight",1.),.1,key="water_weight"),st.slider("Carbon weight",0.,3.,st.session_state.get("carbon_weight",1.),.1,key="carbon_weight"))
cfg=Settings(window=window,idle_w=idle,error=error/100,scenario=scenario,offsite=offsite,weights=weights,controls=controls,routing=routing,shadow=shadow,data_version=version)
results={p:run_policy(cfg,p,version) for p in POLICIES};table=pd.DataFrame({p:summary(d) for p,d in results.items()}).T;b=table.loc["Run now"];sbase=table.loc["Run now + static settings"];x=table.loc["Balanced EcoSchedAI-X"]
agent_table,agent_account=configurations(b.to_dict(),x.to_dict(),st.session_state.cascade_p,st.session_state.cascade_f,st.session_state.cache_hit,st.session_state.batch_eff,st.session_state.retry_budget,len(results["Run now"]))
keys=list(METRICS.values());cards=st.columns(5)
specs=[("Facility energy",keys[0],"kWh"),("Total water",keys[1],"L"),("Carbon",keys[2],"kgCO2e"),("Job completion time",keys[3],"h"),("Deadlines met",None,"%")]
for col,(label,key,unit) in zip(cards,specs):
    if key is None:val=100*(1-x["Deadline misses (steps)"]/max(len(results["Balanced EcoSchedAI-X"]),1));base=100*(1-sbase["Deadline misses (steps)"]/max(len(results["Run now + static settings"]),1));run_base=100*(1-b["Deadline misses (steps)"]/max(len(results["Run now"]),1));delta=val-base
    else:val=x[key];base=sbase[key];run_base=b[key];delta=pct(val,base)
    word="cost" if label=="Job completion time" and delta>0 else ("improvement" if delta<0 else "change")
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val:,.2f} {unit}</div><div class="metric-delta">{delta:+.1f}% {word} vs static</div><div class="metric-base">Static: {base:,.2f} {unit}<br>Run now: {run_base:,.2f} {unit}</div></div>',unsafe_allow_html=True)
loss=[f"{m.lower()} rises {pct(x[k],sbase[k]):.1f}%" for m,k in METRICS.items() if pct(x[k],sbase[k])>0];wait=x["Mean added wait (h)"]-sbase["Mean added wait (h)"];dead=100*(1-x["Deadline misses (steps)"]/max(len(results["Balanced EcoSchedAI-X"]),1))
st.caption("Added wait = start hour − arrival hour. Completion time = added wait + execution duration + routing latency. Positive deltas are costs; negative deltas are improvements.")
st.info(f"Against best static settings, adaptive control changes energy {pct(x[keys[0]],sbase[keys[0]]):+.1f}%, water {pct(x[keys[1]],sbase[keys[1]]):+.1f}% and carbon {pct(x[keys[2]],sbase[keys[2]]):+.1f}%. Added wait changes {wait:+.2f} h; {dead:.1f}% of deadlines are met. "+("It loses on "+", ".join(loss)+"." if loss else "It does not lose on the four headline metrics in this setting."))
if max(abs(pct(x[k],sbase[k])) for k in keys[:3])<1:
    st.warning("Most of the saving here comes from power and cooling settings; adaptive scheduling adds little in this scenario.")

tabs=st.tabs(["Overview","Compare","Explore","Live replay","Your data","Robustness","Security and audit","Agent coordination","Method and limits","Pilot data"])
with tabs[0]:
    st.write("The controller jointly considers facility energy, stress-weighted direct and electricity-related water, carbon and completion time. Critical work is common to every policy and excluded.")
    melted=trace.melt("hour",value_vars=["carbon","temp","wet_bulb"],var_name="Trace",value_name="Value");st.altair_chart(alt.Chart(melted).mark_line().encode(x=alt.X("hour:Q",title="Hour",scale=alt.Scale(domain=[0,23])),y=alt.Y("Value:Q",title="Trace value (mixed units)",scale=alt.Scale(zero=False)),color="Trace:N").properties(height=300),width="stretch")
    st.subheader("Where adaptive control helps")
    cases=[("Strong carbon or wet-bulb swings",cfg),("Tight water stress at one site",replace(cfg,weights=(.5,3.,.5))),("Forecast changes",replace(cfg,error=.15)),("High capacity pressure",replace(cfg,window=3)),("Multi-site differences",replace(cfg,routing=True)),("Restricted tasks",replace(cfg,routing=True))];case_rows=[]
    for name,c in cases:
        controller=summary(run_policy(c,"Balanced EcoSchedAI-X",version));static=summary(run_policy(c,"Run now + static settings",version));deltas={m:100*(controller[k]/static[k]-1) for m,k in list(METRICS.items())[:3]};helps=any(v<-1 for v in deltas.values());case_rows.append({"Condition":name,"Energy vs static (%)":deltas["Energy"],"Water vs static (%)":deltas["Water"],"Carbon vs static (%)":deltas["Carbon"],"Adaptive value":"Helps" if helps else "Little or no help"})
    st.dataframe(pd.DataFrame(case_rows).round(2),width="stretch",hide_index=True);footer()
with tabs[1]:
    st.dataframe(table.round(3),width="stretch");st.download_button("Download comparison",table.to_csv().encode(),"comparison.csv","text/csv")
    long=[{"Policy":p,"Metric":m,"Run-now index":100*table.loc[p,k]/b[k]} for p in POLICIES for m,k in list(METRICS.items())[:3]];bar=pd.DataFrame(long)
    st.altair_chart(alt.Chart(bar).mark_bar().encode(x=alt.X("Policy:N",sort=POLICIES,axis=alt.Axis(labelAngle=-25)),y=alt.Y("Run-now index:Q",title="Run now = 100"),color=alt.Color("Policy:N",scale=alt.Scale(domain=POLICIES,range=[COLORS[p] for p in POLICIES])),column="Metric:N",tooltip=["Policy","Metric",alt.Tooltip("Run-now index:Q",format=".1f")]).properties(height=250),width="stretch")
    st.subheader("Infrastructure and agent coordination");st.dataframe(agent_table.round(2),width="stretch",hide_index=True);footer()
with tabs[2]:
    a,bx=st.columns(2);xn=a.selectbox("Horizontal axis",list(METRICS));yn=bx.selectbox("Vertical axis",[m for m in METRICS if m!=xn]);sn=next(m for m in METRICS if m not in [xn,yn]);pareto=pd.DataFrame([{"Policy":p,**{m:table.loc[p,k] for m,k in METRICS.items()}} for p in POLICIES]);common=dict(x=alt.X(f"{xn}:Q",title=f"← lower {xn}",scale=alt.Scale(zero=False,padding=20)),y=alt.Y(f"{yn}:Q",title=f"lower {yn} ↓",scale=alt.Scale(zero=False,padding=20)),color=alt.Color("Policy:N",scale=alt.Scale(domain=POLICIES,range=[COLORS[p] for p in POLICIES])),tooltip=["Policy"]+[alt.Tooltip(f"{m}:Q",format=".2f") for m in METRICS])
    pts=alt.Chart(pareto[pareto.Policy!="Run now"]).mark_circle(opacity=.85).encode(size=alt.Size(f"{sn}:Q",title=sn,scale=alt.Scale(range=[100,650])),**common);rn=alt.Chart(pareto[pareto.Policy=="Run now"]).mark_point(shape="diamond",filled=True,size=260,color="#111111").encode(x=common["x"],y=common["y"],tooltip=common["tooltip"]);labels=alt.Chart(pareto).mark_text(dx=9,dy=-9,align="left").encode(x=common["x"],y=common["y"],text="Policy:N",color=alt.Color("Policy:N",scale=alt.Scale(domain=POLICIES,range=[COLORS[p] for p in POLICIES]),legend=None));st.altair_chart((pts+rn+labels).properties(height=390),width="stretch")
    policy=st.selectbox("Timeline policy",POLICIES,index=len(POLICIES)-1);d=results[policy];ok=d[d.completed].copy();ok["lane_n"]=ok.groupby(["site","hour"]).cumcount()+1;ok["lane"]=ok.site+" · GPU "+ok.lane_n.astype(str);ok["end"]=np.minimum(24,ok.hour+ok.duration);ok["action"]=np.select([policy=="Run now",ok.site!="Site A",ok.cooling=="Dry/chiller",ok.power_cap<1,ok.wait>0],["Run-now timing","Routed","Dry cooling","Power-capped","Deferred"],default="Evaporative cooling");ad=["Run-now timing","Deferred","Power-capped","Routed","Evaporative cooling","Dry cooling"];ac=["#111111","#4E735D","#176B87","#7A4EAB","#279AB5","#B36B32"]
    gantt=alt.Chart(ok).mark_bar(cornerRadius=2).encode(x=alt.X("hour:Q",title="Hour",scale=alt.Scale(domain=[0,24]),axis=alt.Axis(values=list(range(24)))),x2="end:Q",y=alt.Y("lane:N",title="Site and GPU lane"),color=alt.Color("action:N",scale=alt.Scale(domain=ad,range=ac)),tooltip=["job","site","lane","action","arrival","hour","end","cooling","power_cap","energy","water","carbon","latency"]);marks=alt.Chart(ok).mark_tick(color="#667780",opacity=.3).encode(x="arrival:Q",y="lane:N");st.altair_chart((gantt+marks).properties(height=max(300,18*ok.lane.nunique())),width="stretch")
    pick=st.selectbox("Why this decision",ok.job.tolist());row=ok[ok.job==pick].iloc[0];st.subheader(f"Chosen action: hour {int(row.hour)}, {row.site}, {row.cooling}, {int(row.power_cap*100)}% power cap")
    tr=site_traces(cfg);tr.attrs["lookup"]={(r.site,int(r.hour)):r for _,r in tr.iterrows()};j=workload(cfg).loc[int(row.job)];alts=[]
    for name,h,s,cool in [("Run-now counterfactual",int(j.arrival),"Site A","Evaporative"),("Chosen",int(row.hour),row.site,row.cooling),("One hour earlier",max(int(j.arrival),int(row.hour)-1),"Site A","Evaporative"),("Deadline",int(j.deadline),"Site A","Dry/chiller")]:
        o=_option(cfg,tr,j,h,s,cool,policy=="Balanced EcoSchedAI-X");alts.append({"Alternative":name,"Hour":h,"Site":s,"Cooling":cool,"Energy (kWh)":o["energy"],"Water (L)":o["water"],"Carbon (kgCO2e)":o["carbon"],"Completion (h)":o["latency"],"Deadline":"Pass" if h<=j.deadline else "Fail","Latency":"Pass" if o["latency"]<=cfg.max_latency else "Fail","Power":"Pass" if o["power_kw"]<=cfg.power_limit else "Fail","Security":"Pass" if not(j.restricted and s!="Site A") else "Fail"})
    st.dataframe(pd.DataFrame(alts).drop_duplicates(["Hour","Site","Cooling"]).round(3),width="stretch",hide_index=True);footer()
with tabs[3]:
    @st.fragment(run_every=1.0)
    def replay_panel():
        if "replay_hour" not in st.session_state:st.session_state.replay_hour=0
        p,q=st.columns(2);p.button("Play",on_click=lambda:st.session_state.update(replay_playing=True),use_container_width=True);q.button("Pause",on_click=lambda:st.session_state.update(replay_playing=False),use_container_width=True)
        if st.session_state.get("replay_playing"):st.session_state.replay_hour=(st.session_state.replay_hour+1)%24
        hour=st.slider("Current hour",0,23,key="replay_hour");cur,base=results["Balanced EcoSchedAI-X"],results["Run now"];replay=[]
        for name,d in [("Run now",base),("Controller",cur)]:
            z=d[d.completed&(d.hour<=hour)];replay.append({"Policy":name,"Energy (kWh)":z.energy.sum(),"Water (L)":z.water.sum(),"Carbon (kgCO2e)":z.carbon.sum()})
        st.dataframe(pd.DataFrame(replay).round(2),width="stretch",hide_index=True);now=trace.iloc[hour];c1,c2=st.columns(2);c1.metric("Current carbon intensity",f"{now.carbon:.1f} gCO2e/kWh");c2.metric("Current wet-bulb temperature",f"{now.wet_bulb:.1f} °C")
        for metric in ["energy","water","carbon"]:
            lines=[]
            for name,d in [("Run now",base),("Controller",cur)]:
                z=d[d.completed].groupby("hour")[metric].sum().reindex(range(24),fill_value=0).cumsum();lines += [{"Hour":h,"Policy":name,"Value":v} for h,v in z.items()]
            unit={"energy":"kWh","water":"L","carbon":"kgCO2e"}[metric];chart=alt.Chart(pd.DataFrame(lines)).mark_line(point=True).encode(x=alt.X("Hour:Q",scale=alt.Scale(domain=[0,23])),y=alt.Y("Value:Q",title=f"Cumulative {metric} ({unit})"),color=alt.Color("Policy:N",scale=alt.Scale(domain=["Run now","Controller"],range=["#111111","#7A4EAB"])));st.altair_chart(chart.properties(height=165,padding={"left":85,"right":10,"top":10,"bottom":30}),width="stretch")
    replay_panel()
    footer()
with tabs[4]:
    st.write("Upload hourly series or paste 24 carbon-intensity values. Valid data replaces the current trace.");ct=pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=24,freq="h"),"carbon_gco2_kwh":np.linspace(420,180,24)});wt=pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=24,freq="h"),"temperature_c":np.linspace(24,32,24),"relative_humidity_percent":np.linspace(80,55,24)});jt=pd.DataFrame({"timestamp":["2026-01-01 06:00"],"it_energy_kwh":[2.],"deferrable_flag":[True],"deadline_hours":[8],"site":["Site A"],"restricted_flag":[False]});dc=st.columns(3);dc[0].download_button("Carbon template",ct.to_csv(index=False).encode(),"carbon_template.csv");dc[1].download_button("Weather template",wt.to_csv(index=False).encode(),"weather_template.csv");dc[2].download_button("Workload template",jt.to_csv(index=False).encode(),"workload_template.csv");st.file_uploader("Hourly carbon intensity CSV",type="csv",key="carbon_upload");st.file_uploader("Hourly temperature and humidity CSV",type="csv",key="weather_upload");st.file_uploader("Optional workload CSV",type="csv",key="workload_upload");st.text_area("Paste 24 comma-separated carbon-intensity values",key="paste_carbon")
    for e in input_errors:st.error(e)
    if not input_errors and any([st.session_state.get("carbon_upload"),st.session_state.get("weather_upload"),st.session_state.get("workload_upload"),st.session_state.get("paste_carbon","").strip()]):st.success("User data validated and applied.")
    footer()
with tabs[5]:
    chain=[("Run now","Run now",cfg),("+ time shifting only","Balanced EcoSchedAI-X",replace(cfg,controls=False,routing=False,shield_enabled=False)),("+ power caps","Balanced EcoSchedAI-X",replace(cfg,controls=True,power_caps=True,cooling_choice=False,routing=False,shield_enabled=False)),("+ cooling-mode choice","Balanced EcoSchedAI-X",replace(cfg,cooling_choice=True,routing=False,shield_enabled=False)),("+ multi-site routing","Balanced EcoSchedAI-X",replace(cfg,routing=True,shield_enabled=False)),("+ shield and fallback","Balanced EcoSchedAI-X",cfg)];rows=[];prev=None
    for label,pol,c in chain:
        z=summary(run_policy(c,pol,version));r={"Component":label,"Energy":z[keys[0]],"Water":z[keys[1]],"Carbon":z[keys[2]],"Completion time":z[keys[3]]}
        if prev:
            for m in ["Energy","Water","Carbon"]:r[f"Contribution: {m.lower()} (%)"]=100*(prev[m]-r[m])/prev[m]
        rows.append(r);prev=r
    both=agent_table[agent_table.Configuration=="Both"].iloc[0];r={"Component":"+ agent-side levers","Energy":both["Energy (kWh / 1,000 tasks)"],"Water":both["Water (L / 1,000 tasks)"],"Carbon":both["Carbon (kgCO2e / 1,000 tasks)"],"Completion time":both["Completion time (h)"]}
    for m in ["Energy","Water","Carbon"]:r[f"Contribution: {m.lower()} (%)"]=100*(prev[m]-r[m])/prev[m]
    rows.append(r);adf=pd.DataFrame(rows);st.dataframe(adf.round(2),width="stretch",hide_index=True);hurts=[]
    for _,r in adf.iloc[1:].iterrows():
        bad=[m for m in ["energy","water","carbon"] if r.get(f"Contribution: {m} (%)",0)<0]
        if bad:hurts.append(f"{r.Component} hurts "+", ".join(bad))
    st.write("Component check: "+("; ".join(hurts)+"." if hurts else "no sequential component hurts the three resource metrics in this setting."));wf=replace(cfg,weights=(.5,3.,.5));wd=run_policy(wf,"Balanced EcoSchedAI-X",version);st.write(f"Water-first outcome: {100*(wd.cooling=='Dry/chiller').mean():.1f}% dry-cooled and {100*(wd.site!='Site A').mean():.1f}% routed. Water changes {pct(summary(wd)[keys[1]],b[keys[1]]):+.1f}% versus run now.");footer()
with tabs[6]:
    d=results["Balanced EcoSchedAI-X"];c1,c2,c3=st.columns(3);c1.metric("Restricted tasks",f"{int(d.restricted.sum())}");c2.metric("Routing violations",f"{int(d.security_violation.sum())}");c3.metric("Deadline misses",f"{int(d.deadline_miss.sum())}");st.download_button("Download audit log",audit_csv(d),"audit_log.csv","text/csv");st.caption("Metadata only; no prompts, outputs or payloads.");footer()
with tabs[7]:
    st.write("Agents declare criticality, slack or deadline, restricted status and estimated energy. The coordinator combines that metadata with facility conditions; agent-side levers remain separate from infrastructure control.")
    c1,c2,c3,c4=st.columns(4);c1.slider("Cascade success probability p",.0,1.,key="cascade_p",step=.05);c2.slider("Small-model energy fraction f",.05,1.,key="cascade_f",step=.05);c3.slider("Cache hit rate",.0,.9,key="cache_hit",step=.05);c4.slider("Batch effectiveness",.0,1.,key="batch_eff",step=.05);st.slider("Retry budget per workflow",0,6,key="retry_budget")
    st.caption("Quality is equal by construction: unsuccessful small-model attempts escalate to the large model. Retry avoidance assumes 5% of workflows enter a loop without a budget.")
    st.dataframe(agent_table.round(2),width="stretch",hide_index=True)
    infra_save=b[keys[0]]-x[keys[0]];both_e=agent_table.loc[agent_table.Configuration=="Both","Energy (kWh / 1,000 tasks)"].iloc[0];agent_save=x[keys[0]]-both_e;total=max(1e-9,b[keys[0]]-both_e)
    hurt="Agent-side levers raise energy in this setting." if agent_account["energy_factor"]>1 else "Agent-side levers reduce energy in this setting."
    st.info(f"Of the modelled energy saving, infrastructure control contributes {100*infra_save/total:.1f}% and agent-side levers contribute {100*agent_save/total:.1f}%. Escalation rate is {agent_account['escalation_rate']:.1%}; failed small-model attempts add {b[keys[0]]*agent_account['failed_attempt_factor']:.1f} kWh per 1,000 tasks. {hurt} Low p and high f can reverse the saving.")
    chosen=results["Balanced EcoSchedAI-X"].iloc[0];dot=f'''digraph G {{rankdir=LR;node [shape=box style=filled fontname="Arial"];
    request [label="Agent request" fillcolor="#DCE7EC"];critical [label="Critical step\nSite A · hour {int(chosen.arrival)}" fillcolor="#B54A4A" fontcolor="white"];
    cached [label="Cached lookup\nno compute" fillcolor="#6A9F58"];cascade [label="Cascaded step\nsmall model" fillcolor="#4C8BB5"];
    escalate [label="Escalated on failure\nlarge model" fillcolor="#B88436"];defer [label="Deferrable step\n{chosen.site} · hour {int(chosen.hour)}" fillcolor="#7E65A5" fontcolor="white"];
    done [label="Completed task" fillcolor="#DCE7EC"];request->critical;request->cached;critical->cascade;cascade->done [label="success p"];cascade->escalate [label="failure 1-p"];escalate->defer;cached->done;defer->done;}}''';st.graphviz_chart(dot,use_container_width=True)
    curve=batch_curve(st.session_state.batch_eff);st.subheader("Batch-size effect");st.dataframe(curve.round(3),width="stretch",hide_index=True);st.altair_chart(alt.Chart(curve).mark_line(point=True).encode(x=alt.X("Batch size:O"),y=alt.Y("Energy per equal-work step (kWh):Q",scale=alt.Scale(zero=False)),tooltip=list(curve.columns)).properties(height=240),width="stretch");footer()
with tabs[8]:
    s=sanity_test();st.subheader("Mandatory Self-test");st.success(f"PASS — carbon-only {s['carbon_only']:.2f} ≤ run-now {s['run_now']:.2f} kgCO2e / 1,000 steps") if s["passed"] else st.error("FAIL");st.write("New here: joint accounting; power and cooling actions; protected routing; a constraint shield; uncertainty fallback; shadow recommendations; counterfactual audit; component attribution; and separately reported agent-side coordination.");st.write("Earlier supplied work used carbon opportunity windows and urgency constraints, a CNN trained on synthetic spatiotemporal carbon-intensity tensors (32, 32, 10, 1), and baseline, rule-based and DQN comparisons across regions. From paper: execution-time carbon intensity 720.5 / 685.2 / 650.3 gCO2/kWh, 4.88% / 9.72% lower. This was intensity at execution, not verified total emissions.");st.subheader("Data sources");st.dataframe(pd.DataFrame(sources()).drop(columns=["error"],errors="ignore"),width="stretch",hide_index=True);st.write("User uploads override the corresponding cached trace. Workload, hardware, cooling response, model cascading, batching, caching and retry-loop behaviour remain simulated.");st.write("Agent assumptions: small-model quality equals large-model quality by construction after escalation; default p=0.85 and f=0.30; cache hits avoid compute; batching interpolates the existing power-utilisation model; 5% of workflows are assumed loop-prone; retry budgets avoid only those modelled repeats.");st.write("Limits: approximate resource models, no full dependency graph, no measured agent success distribution, no lifecycle impact, and no production or certification claim. Novelty is a gap hypothesis, not a priority claim.");footer()
with tabs[9]:
    st.write("Shadow-mode validation needs time-aligned operator measurements before any operational conclusion.");pilot=pd.DataFrame({"Measurement":["Accelerator power and utilization","Facility inlet and wet-bulb temperature","Cooling electricity and direct water","Whole-facility power","Grid carbon intensity","Queue, start, completion and deadline","Network bytes, latency and energy","Forecast and action metadata"],"Unit":["kW, %","°C","kWh, L","kW","gCO2e/kWh","timestamp, h","GB, ms, kWh","timestamp"]});st.dataframe(pilot,width="stretch",hide_index=True);footer()
