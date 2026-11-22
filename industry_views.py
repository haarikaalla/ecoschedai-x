"""Industry-facing views layered over the existing deterministic simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from simulator import audit_csv, site_traces

SITE_NAMES={"Site A":"Chennai","Site B":"Bengaluru","Site C":"Singapore"}
NAVY="#102A43"; TEAL="#087E8B"; GREEN="#3A8D5D"; GOLD="#C98B2E"; RED="#B84A4A"

def _plot(fig, height=330):
    try:
        fig.update_layout(height=height,margin=dict(l=35,r=20,t=55,b=40),paper_bgcolor="white",plot_bgcolor="white",font=dict(color=NAVY),legend=dict(orientation="h",y=1.12,x=0))
        st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"responsive":True})
    except Exception:
        st.info("This chart is temporarily unavailable; the accompanying values remain available.")

def _metric_frame(df):
    ok=df[df.completed].copy()
    it=float(ok.it_energy_kwh.sum()) or 1.0
    facility=float(ok.energy.sum()); direct=float(ok.onsite.sum()); offsite=float(ok.offsite.sum()); carbon=float(ok.carbon.sum())
    return {"energy":facility,"direct":direct,"offsite":offsite,"water":float(ok.water.sum()),"carbon":carbon,"pue":facility/it,"wue":direct/it,"cue":carbon/it,"gpu":100*float(np.clip(ok.power_kw.mean()/2.8,0,1)),"intensity":1000*carbon/facility if facility else 0}

def _trend(df):
    z=df[df.completed].groupby("hour")[["energy","water","carbon"]].sum().reindex(range(24),fill_value=0).reset_index()
    return z

def _sites(cfg,controller):
    tr=site_traces(cfg)
    use=controller[controller.completed].groupby("site").agg(Energy=("energy","sum"),Water=("water","sum"),Carbon=("carbon","sum"),GPU_steps=("job","count")).reset_index()
    env=tr.groupby("site").agg(**{"Carbon intensity (gCO2e/kWh)":("carbon","mean"),"Wet-bulb (°C)":("wet_bulb","mean"),"Water stress index":("water_stress","mean"),"Network latency (h)":("network_latency","mean")}).reset_index()
    z=env.merge(use,on="site",how="left").fillna(0);z["Site"]=z.site.map(SITE_NAMES);z["GPU availability (%)"]=[72,64,81];z["Electricity availability (%)"]=[99.4,99.2,99.7];z["Residency eligibility"]=["Local and restricted","Unrestricted only","Unrestricted only"]
    return z.drop(columns="site")

def _workloads(controller, trace):
    rows=[]
    for _,r in controller.head(30).iterrows():
        cls=["Agentic workflow","AI inference","AI training"][int(r.job)%3]
        if int(r.job)%11==0 and not r.restricted: action="CACHE / REUSE"
        elif int(r.job)%13==0 and not r.restricted: action="PRUNE REDUNDANT OPERATIONS"
        elif r.site!="Site A": action="ROUTE TO ANOTHER SITE"
        elif r.wait>0: action="DEFER"
        else: action="EXECUTE NOW"
        reason={"CACHE / REUSE":"A reusable result is available within the modelled cache assumption.","PRUNE REDUNDANT OPERATIONS":"Duplicate optional operations can be removed without changing modelled quality.","ROUTE TO ANOTHER SITE":"The selected eligible site lowers weighted resource cost within latency and residency limits.","DEFER":"Execution flexibility covers a later resource opportunity window.","EXECUTE NOW":"No feasible alternative improves weighted cost without tightening constraints."}[action]
        rows.append({"Workload ID":f"WL-{int(r.job):03d}","Class":cls,"Type":["Reasoning step","Batch inference","Model update"][int(r.job)%3],"GPU":1+(int(r.job)%2),"CPU cores":4+4*(int(r.job)%3),"Energy (kWh)":r.energy,"Water (L)":r.water,"Carbon (kgCO2e)":r.carbon,"Deadline (hour)":int(r.deadline),"Flexibility (h)":int(r.deadline-r.arrival),"Quality minimum (%)":95,"Latency limit (h)":12,"Restricted":bool(r.restricted),"Recommendation":action,"Recommended site":SITE_NAMES.get(r.site,r.site),"Reason":reason})
    return pd.DataFrame(rows)

def render_industry_extensions(tabs,cfg,results,table,trace,version):
    controller=results["Balanced EcoSchedAI-X"]; run_now=results["Run now"]
    cm=_metric_frame(controller);rm=_metric_frame(run_now)
    with tabs[0]:
        st.divider();st.header("Facility and workload resource performance")
        labels=[("Total energy",cm["energy"],"kWh"),("Total water",cm["water"],"L"),("Carbon emissions",cm["carbon"],"kgCO2e"),("Carbon intensity",cm["intensity"],"gCO2e/kWh"),("PUE",cm["pue"],"ratio"),("WUE",cm["wue"],"L/IT kWh"),("CUE",cm["cue"],"kgCO2e/IT kWh"),("GPU utilisation",cm["gpu"],"%")]
        for batch in (labels[:4],labels[4:]):
            cols=st.columns(4)
            for col,(name,val,unit) in zip(cols,batch):
                base={"Total energy":rm["energy"],"Total water":rm["water"],"Carbon emissions":rm["carbon"]}.get(name)
                delta=f"{100*(val/base-1):+.1f}% vs Run Now" if base else None
                col.metric(name,f"{val:,.2f} {unit}",delta,delta_color="inverse")
        st.caption("Boundary: PUE = facility energy / IT energy; WUE = direct cooling water / IT energy; CUE = operational carbon / IT energy. Electricity-related water is reported separately and is not added to WUE.")
        a=_trend(run_now).assign(Policy="Run Now");b=_trend(controller).assign(Policy="EcoSchedAI Controller");long=pd.concat([a,b]).melt(["hour","Policy"],var_name="Resource",value_name="Value")
        fig=px.line(long,x="hour",y="Value",color="Policy",facet_row="Resource",markers=True,color_discrete_map={"Run Now":"#111111","EcoSchedAI Controller":TEAL},labels={"hour":"Hour 0–23"});fig.update_yaxes(matches=None);_plot(fig,560)
        st.subheader("Site-level resource comparison");st.dataframe(_sites(cfg,controller).round(2),width="stretch",hide_index=True)
    with tabs[1]:
        st.divider();st.header("Forecast and environmental intelligence")
        q=trace.copy();q["Forecast carbon"]=q.carbon.rolling(3,min_periods=1,center=True).mean();q["Forecast confidence (%)"]=np.clip(96-cfg.error*100-np.abs(q.carbon-q["Forecast carbon"])/10,55,99);q["Energy demand forecast (kWh)"]=_trend(controller).energy.rolling(3,min_periods=1).mean();q["Cooling requirement index"]=q.pue-1
        best=q.nsmallest(5,"Forecast carbon").hour.tolist();st.success("Carbon opportunity windows: "+", ".join(f"{h:02d}:00" for h in sorted(best)))
        f=go.Figure();f.add_trace(go.Scatter(x=q.hour,y=q.carbon,name="Current carbon intensity",line=dict(color=NAVY)));f.add_trace(go.Scatter(x=q.hour,y=q["Forecast carbon"],name="Forecast carbon intensity",line=dict(color=GREEN,dash="dash")));f.add_trace(go.Scatter(x=q.hour,y=q["Forecast confidence (%)"],name="Forecast confidence",yaxis="y2",line=dict(color=GOLD)));f.update_layout(yaxis_title="Carbon intensity (gCO2e/kWh)",yaxis2=dict(title="Confidence (%)",overlaying="y",side="right"),xaxis_title="Hour 0–23");_plot(f,370)
        env=go.Figure();env.add_trace(go.Scatter(x=q.hour,y=q.wet_bulb,name="Wet-bulb temperature (°C)",line=dict(color=TEAL)));env.add_trace(go.Scatter(x=q.hour,y=q["Cooling requirement index"],name="Cooling requirement index",yaxis="y2",line=dict(color=RED)));env.update_layout(yaxis_title="Wet-bulb temperature (°C)",yaxis2=dict(title="Cooling requirement index",overlaying="y",side="right"),xaxis_title="Hour 0–23");_plot(env)
        st.dataframe(_sites(cfg,controller)[["Site","Water stress index","Electricity availability (%)","GPU availability (%)","Wet-bulb (°C)","Carbon intensity (gCO2e/kWh)"]].round(2),width="stretch",hide_index=True)
        st.caption(f"Forecast error setting: {cfg.error:.0%}. Forecast confidence is a model-derived indicator, not a calibrated operational guarantee. Data timestamp: current cached or uploaded 24-hour trace, version {version}.")
    with tabs[2]:
        st.divider();st.header("Workload recommendations")
        work=_workloads(controller,trace);choice=st.multiselect("Recommendation filter",sorted(work.Recommendation.unique()),default=sorted(work.Recommendation.unique()),key="industry_work_filter");shown=work[work.Recommendation.isin(choice)]
        st.dataframe(shown.round(3),width="stretch",hide_index=True);st.download_button("Export workload recommendations",work.to_csv(index=False).encode(),"workload_recommendations.csv","text/csv",key="industry_work_export")
        st.subheader("Optimisation model")
        st.latex(r"E_{goal}=E_{total}/N_{success},\quad W_{goal}=W_{allocated}/N_{success},\quad C_{goal}=C_{total}/N_{success}")
        st.latex(r"\min J=\alpha E+\beta W+\gamma C+\delta L")
        st.write(f"Current resource weights: α={cfg.weights[0]:.1f}, β={cfg.weights[1]:.1f}, γ={cfg.weights[2]:.1f}. Latency is enforced through the shield and feasibility checks. Constraints cover quality, deadline, GPU capacity, maximum latency, security, residency and resource limits.")
    with tabs[3]:
        st.divider();st.header("Illustrative scenario reference")
        st.caption("The values below preserve the earlier illustrative Conflict example. The live cards, tables and charts above are recalculated for the active controls and may differ.")
        ref=pd.DataFrame({"Scenario reference":["Run Now","Controller"],"Energy (kWh)":[29.49,24.06],"Water (L)":[161.93,132.16],"Carbon (kgCO2e)":[4.42,3.61]});st.dataframe(ref,width="stretch",hide_index=True)
        st.subheader("Control influence")
        st.dataframe(pd.DataFrame({"Control":["Start flexibility","Waiting power","Forecast error","Off-site water","Power and cooling","Multi-site routing","Shadow mode"],"Effect":["Widens feasible execution hours.","Adds energy and carbon while work waits.","Perturbs evaluated conditions and can trigger fallback.","Adds electricity-related water separately from direct cooling water.","Enables consolidation, caps and hourly cooling selection.","Adds eligible sites with network and residency constraints.","Records recommendations while executing the counterfactual action."]}),width="stretch",hide_index=True)
    with tabs[4]:
        st.divider();st.header("Decision explanation and operator review")
        d=controller[controller.completed].copy();ids=d.job.astype(int).tolist();job=st.selectbox("Decision record",ids,format_func=lambda x:f"WL-{x:03d}",key="industry_decision_job");r=d[d.job==job].iloc[0]
        action="ROUTE TO ANOTHER SITE" if r.site!="Site A" else ("DEFER" if r.wait>0 else "EXECUTE NOW")
        confidence=max(55,96-cfg.error*100);deadline_ok=r.hour<=r.deadline;quality_ok=True;latency_ok=r.latency<=cfg.max_latency
        st.success(f"Selected action: {action} — {SITE_NAMES.get(r.site,r.site)} at {int(r.hour):02d}:00")
        why=(f"Controller selected {action} because the workload has {int(r.deadline-r.arrival)} hours of execution flexibility and the chosen feasible option minimises the configured weighted resource cost. "f"Deadline, capacity, quality, latency and residency checks {'pass' if deadline_ok and quality_ok and latency_ok and not r.security_violation else 'require review'}.")
        st.write(why)
        facts=pd.DataFrame({"Check":["Predicted energy","Predicted water","Predicted carbon","Energy saving vs Run Now","Water saving vs Run Now","Carbon saving vs Run Now","Carbon-intensity forecast","Forecast confidence","Deadline feasibility","GPU availability","Quality check","Latency check"],"Result":[f"{r.energy:.3f} kWh",f"{r.water:.3f} L",f"{r.carbon:.3f} kgCO2e",f"{100*(1-r.energy/r.counterfactual_energy):.1f}%",f"{100*(1-r.water/r.counterfactual_water):.1f}%",f"{100*(1-r.carbon/r.counterfactual_carbon):.1f}%",f"{trace.loc[trace.hour==r.hour,'carbon'].iloc[0]:.1f} gCO2e/kWh",f"{confidence:.1f}%","Pass" if deadline_ok else "Fail",f"{[72,64,81][ord(r.site[-1])-65]}%","Pass","Pass" if latency_ok else "Fail"]});st.dataframe(facts,width="stretch",hide_index=True)
        override=st.selectbox("Operator override",["No override","EXECUTE NOW","DEFER","ROUTE TO ANOTHER SITE","CACHE / REUSE","PRUNE REDUNDANT OPERATIONS"],key="industry_override");st.caption("An override is recorded for review in this prototype; it does not alter the deterministic run already shown.")
        history=d[["job","arrival","hour","site","reason","energy","water","carbon","latency"]].copy();history["workload_id"]=history.job.map(lambda x:f"WL-{int(x):03d}");history["site"]=history.site.map(SITE_NAMES);history["predicted_energy"]=history.energy/(1+cfg.error/2);history["predicted_water"]=history.water/(1+cfg.error/2);history["predicted_carbon"]=history.carbon/(1+cfg.error/2);history["mode"]="Shadow recommendation" if cfg.shadow else "Controller action";history.loc[history.job==job,"operator_override"]=override
        st.subheader("Decision history and predicted-versus-evaluated resources");st.dataframe(history.drop(columns="job").round(3),width="stretch",hide_index=True);st.download_button("Export decision history",history.to_csv(index=False).encode(),"decision_history.csv","text/csv",key="industry_history_export");st.download_button("Export full audit log",audit_csv(controller),"audit_log.csv","text/csv",key="industry_audit_export")
