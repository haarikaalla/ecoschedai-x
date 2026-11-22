"""Dependency-light test runner for the offline project environment."""
from dataclasses import replace
from simulator import Settings,POLICIES,simulate,sanity_test,ledger
from streamlit.testing.v1 import AppTest

assert sanity_test()["passed"]
for scenario in ["Conflict","Improve together"]:
    for policy in POLICIES:
        d,_=simulate(Settings(scenario=scenario),policy); ok=d[d.completed]
        assert len(d)==60 and (ok.hour>=ok.arrival).all() and (ok.hour<=ok.deadline).all()
        assert ok.groupby(["site","hour"]).size().max()<=6 and ok.security_violation.sum()==0
d,_=simulate(Settings(error=.3),"Balanced EcoSchedAI-X"); assert d.reason.str.contains("fallback").all()
d,_=simulate(Settings(shadow=True),"Balanced EcoSchedAI-X"); assert (d.hour==d.arrival).all()
assert len(ledger(Settings()))==3
app=AppTest.from_file("app.py",default_timeout=180).run(); assert not app.exception,app.exception
print("PASS: sanity, capacity, indices, security, fallback, shadow, ledger, and app execution")
