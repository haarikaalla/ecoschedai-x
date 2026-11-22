from dataclasses import replace
from simulator import Settings,POLICIES,simulate,sanity_test,ledger
def test_mandatory_sanity(): assert sanity_test()["passed"]
def test_capacity_deadline_indices_and_security():
    for scenario in ["Conflict","Improve together"]:
        for p in POLICIES:
            d,_=simulate(Settings(scenario=scenario),p); ok=d[d.completed]
            assert len(d)==60 and (ok.hour>=ok.arrival).all() and (ok.hour<=ok.deadline).all()
            assert ok.groupby(["site","hour"]).size().max()<=6
            assert ok.security_violation.sum()==0
def test_fallback_and_shadow():
    d,_=simulate(Settings(error=.3),"Balanced EcoSchedAI-X"); assert d.reason.str.contains("fallback").all()
    d,_=simulate(Settings(shadow=True),"Balanced EcoSchedAI-X"); assert (d.hour==d.arrival).all()
def test_ledger(): assert len(ledger(Settings()))==3
