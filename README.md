# EcoSchedAI-X

EcoSchedAI-X is a new implementation, not the original research code. It is an offline-capable Streamlit decision-support simulator for energy, water, carbon, completion-time and constraint trade-offs. All operational outcomes and savings are simulated; cached open data supplies context only. The project makes no certification or production claim.

## Live app

[Open the deployed Streamlit app](https://ecoschedai-x.streamlit.app/)

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python get_data.py                 # optional refresh; app has simulated fallbacks
python train_models.py             # fixed CPU benchmark, no search
python -m pytest -q
streamlit run app.py
```

For a local demo, `./run.sh` starts the app from the project environment when available. The repository root contains `app.py`, `requirements.txt`, `runtime.txt`, and `.streamlit/config.toml` for direct deployment to Streamlit Community Cloud.

After dependencies are installed and data is cached, the app launches offline. `get_data.py` records failed downloads and never fabricates rows. The workload cache retains timestamps and token counts only.

## Scope

The base model uses 24 hourly slots, 60 fixed-seed deferrable steps, 2 kWh baseline IT energy per step, arrivals from 06:00–10:00, a selectable flexibility window and six starts per site-hour. Capacity is part of the assignment optimisation. Critical steps are common to all policies and excluded. Facility energy, direct and electricity-related water, carbon, waiting time, completion time, misses and protected-routing violations are reported.

Stage 1 adds the stated nonlinear accelerator power curve, caps, consolidation and cooling mode choice. Stage 2 adds three synthetic sites, network costs, water stress and protected routing. Stage 3 adds the constraint shield, uncertainty fallback, shadow mode, audit counterfactuals and ten-seed confidence intervals. Stage 4 adds a fixed CPU learning benchmark, calibrated forecast interval, masked planner and exact small-case reference. Stage 5 caches open carbon, weather and inference-request metadata with explicit fallbacks. Stage 6 provides the full analysis interface.

The prior supplied study concerned flexible training work, carbon opportunity windows and urgency constraints. It compared immediate, rule-based and learned scheduling using a CNN trained on synthetic spatiotemporal carbon-intensity tensors of shape `(32, 32, 10, 1)`. Its 4.88% and 9.72% results were reductions in carbon intensity at execution time, not verified total-emissions reductions. Its limits included approximate energy accounting, synthetic data, and no water or agentic workload model. Neutral record: DOI `10.1007/s44163-026-02022-4`.

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every material modelling choice.

