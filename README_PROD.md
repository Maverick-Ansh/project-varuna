# FloodTwin Patna — production package (`varuna`)

The five research notebooks, refactored into a config-driven, headless Python package with a
local-LLM chat front end. **Type/ask in natural language → it fetches weather, runs the flood
model, and tells you who's at risk, how urgently, and what to do.**

## Layout

```
varuna/
  config.py        # single source of truth (AOI, paths, thresholds) — config.yaml + env
  ee_auth.py       # Earth Engine init + image download (interactive or service account)
  io.py            # raster helpers, retrying HTTP, artifact-bundle validation
  build/           # "build the city model" (GPU, run rarely)  ── nb01/02/03/05
    sinks.py recharge.py validate.py twin.py
  serve/           # "run it daily" (CPU, headless)            ── nb04 + emulator + optimizer
    weather.py alerts.py emulator.py optimize.py
  agent/           # local-LLM tool-calling front end
    prompts.py tools.py llm.py chat.py
  cli.py           # python -m varuna {build|alert|chat}
tests/             # pytest (synthetic bundle; no Earth Engine needed)
notebooks/         # 00_setup_build.ipynb, 10_chat_agent.ipynb  (Colab)
config.example.yaml  requirements.txt
```

## Two stages

**1. Build the city model** (Colab GPU, once / occasionally) — `notebooks/00_setup_build.ipynb`
or `python -m varuna build`. Downloads DEM/land-cover/radar via Earth Engine, detects sinks,
ranks recharge sites, scores against Sentinel-1 (reports CSI), and trains the differentiable
twin. Writes the artifact bundle (`dem.tif`, `sinks.csv`, `emulator.pt`, …) to `CFG.work`.

**2. Serve / chat** (CPU for alerts; GPU for the LLM + optimizer) —
`notebooks/10_chat_agent.ipynb` or `python -m varuna chat`. Loads the bundle and answers
questions. The alert path (`varuna.serve.alerts.run_alerts`) is GPU-free so it never contends
with the local LLM.

## Quick start

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml      # set project_id, or export VARUNA_PROJECT_ID
export VARUNA_PROJECT_ID=my-ee-project

python -m varuna build                  # GEE auth + full build (GPU for twin)
python -m varuna alert --rain-mm 80     # dry-run the outlook with a forced 80 mm
python -m varuna chat                   # talk to it (local LLM)
```

In Python:

```python
from varuna.serve.alerts import run_alerts
from varuna.serve.emulator import whatif
from varuna.agent.llm import Agent

run_alerts(rain_mm=80)                       # structured outlook JSON
whatif(rain_mm=150, dig_sites={"3": 2.0})    # millisecond emulator what-if
Agent(four_bit=True).chat("Outlook for tomorrow? Which wards are urgent?")
```

## Prerequisites you must supply
1. **Earth Engine project** — register at earthengine.google.com, create a GCP project, set
   `VARUNA_PROJECT_ID`.
2. **Real groundwater** — replace the auto-written SAMPLE `gw_levels.csv` in `CFG.work` with
   pre-monsoon CGWB depth-to-water from India-WRIS, or recharge rankings are meaningless.
3. **Validation event date** — a Sentinel-1 pass right after heavy rain (the build lists them).

## Honest limitations (carried through to every alert + stated by the LLM)
- Alert thresholds (0.5/1.0 fill ratio) and the twin's Manning/infiltration params are
  **uncalibrated placeholders**. Calibrate against nb03's SAR observations before acting.
- No storm-sewer or pumping model — over-predicts flooding near working drains.
- Free DEMs carry 1–2 m vertical error in flat Patna — treat site rankings as shortlists.
- Outputs are **waterlogging likelihood**, not certified forecasts. Verify on the ground.

## Tests

```bash
pytest -m "not slow"     # fast: tool dispatch, alert engine, simulator gradients
pytest -m slow           # also trains a tiny emulator end-to-end on a synthetic bundle
```

## Local LLM notes
Default `Qwen/Qwen2.5-7B-Instruct` via transformers (native tool-calling template, needs
`transformers>=4.43`). On a 16 GB T4 sharing VRAM with the simulator, use 4-bit
(`Agent(four_bit=True)` / `chat --four-bit` / `VARUNA_LLM_4BIT=1`). A100/L4 on Colab Pro runs
fp16 comfortably. Swap models via `VARUNA_LLM_MODEL`.
```
