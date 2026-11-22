from agent_coordination import lever_accounting,configurations,batch_curve
from simulator import Settings,simulate,summary

def test_agent_accounting_and_quality():
    run=summary(simulate(Settings(),"Run now")[0]);infra=summary(simulate(Settings(),"Balanced EcoSchedAI-X")[0])
    table,a=configurations(run,infra)
    assert len(table)==4 and (table["Quality maintained (%)"]==100).all()
    assert 0<=a["escalation_rate"]<=1 and a["failed_attempt_factor"]>=0

def test_low_success_can_hurt():
    assert lever_accounting(p=.2,f=.8,cache_hit=0,batch_eff=0,retry_budget=6)["energy_factor"]>1
    assert len(batch_curve())==4
