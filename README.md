# FloodTwin Patna — flood-to-drought water system

Map where Patna's monsoon water pools, push it into the depleted aquifer instead of onto
streets, verify everything against satellite radar, warn wards a day ahead, and let gradient
descent design the interventions. Everything runs on Google Colab's free tier (GPU only for
notebook 05) using free public data.

## Results on real Patna data (honest)

Measured on the differentiable twin (128×128 @ 60 m) against free public data — reported as-is, including a negative result.

- **SAR validation (the honest core):** scoring the *dynamic* twin against Sentinel-1 water masks
  (consistent `jrc<50` permanent-water mask on both sides) gives **mean CSI ≈ 0.033** across 8 storms
  (best 2025-08-02 = 0.112). Per-WorldCover-class Manning/infiltration calibration is a **near-null**
  (held-out 0.048→0.049): at 60 m, topographic-sink routing has limited co-location skill, and physics
  constants move water *level*, not *location*. See [`VALIDATION.md`](VALIDATION.md). The committed
  `validation_scores.json` 0.045 scored the *static* `depth.tif`, not the twin — superseded.
- **Canal / drainage optimizer:** the v2 router (basin sources, downhill-guaranteed Dijkstra,
  descending carved bed, river outfalls) cuts **20.2%** of built-land flood volume on a 100 mm storm
  (vs the v1 least-cost router's 16.1%). Strategy ladder: pits-only 7.8% → canals+pits 20.2%. See
  [`artifacts/patna/CANAL_RESULTS.md`](artifacts/patna/CANAL_RESULTS.md).
- **Adaptive distributed storage:** each storage site is sized to its own local depression (dynamic,
  non-linear by geography); the model reports *how many* sites a target cut needs — **727 → 30%,
  1359 → 50%, 2063 → 70%** (measured by re-simulating). See
  [`artifacts/patna/STORAGE_RESULTS.md`](artifacts/patna/STORAGE_RESULTS.md) and `varuna/serve/containers.py`.

## Notebooks (run in this order)

| Notebook | Idea | What it produces |
|---|---|---|
| `01_base_pipeline_sink_mapping.ipynb` | foundation | DEM, sinks ranked by volume, catchments, interactive map |
| `02_aquifer_recharge_matching.ipynb` | idea 2 | Recharge Suitability Index, ranked recharge-pit sites |
| `03_sar_ground_truth_loop.ipynb` | idea 3 | observed standing-water maps, POD/FAR/CSI validation, waterlogging frequency climatology |
| `04_realtime_ward_alerts.ipynb` | idea 4 | next-24h fill ratio per sink, ward-level alert map, optional Telegram push |
| `05_differentiable_flood_twin.ipynb` | idea 6 | PyTorch flood simulator, U-Net emulator, gradient-optimised dig plan |

`simulator_test.py` is an offline test of the notebook-05 physics (stability, mass balance,
correct pooling, gradient flow). It passed on a synthetic terrain; run it anywhere with PyTorch.

## One-time setup

1. **Google Earth Engine** (free, non-commercial): sign up at earthengine.google.com, create a
   Google Cloud project, register it for EE, and set `PROJECT_ID` at the top of notebooks
   01–03. First run will pop an auth flow.
2. **Colab**: upload the notebooks (or open from Drive). Mount Drive in notebook 01 if you want
   outputs to persist between sessions. Notebook 05 needs Runtime → Change runtime type → T4 GPU.
3. **Groundwater data** (notebook 02): download pre-monsoon depth-to-water for Patna district
   stations from India-WRIS (indiawris.gov.in → Groundwater Level) and replace the
   auto-generated `gw_levels.csv`. **The file the notebook writes contains SAMPLE values** —
   the pipeline runs with them but the rankings mean nothing until real data goes in.

## Things to edit (all marked `<-- EDIT` in the code)

- `PROJECT_ID` — your Earth Engine cloud project (notebooks 01, 02, 03)
- `AOI` — bounding box; preset to greater Patna
- `EVENT_DATE` (03) — pick a Sentinel-1 pass date right after heavy rain from the printed list
- `DESIGN_RAIN`, `BUDGET_M3` (05) — your design storm and excavation budget
- `TELEGRAM_TOKEN/CHAT` (04) — optional daily alert push

## How the pieces close the loop

01 predicts where water pools → 03 *observes* where it actually pooled (radar) and scores the
prediction → disagreement pixels are DEM-correction targets → 02 says which pooling sites sit
over a thirsty aquifer → 04 turns the same machinery into daily warnings → 05 wraps it all in a
differentiable simulator so the *design* of interventions becomes an optimisation problem, and
its calibration target is 03's observations.

## Honest limitations (say these out loud to anyone you show this)

- Free DEMs carry 1–2 m vertical error in flat terrain; treat site rankings as shortlists for
  ground verification, not final designs.
- No storm-sewer network or pumping in any model here — predictions over-estimate flooding near
  functioning drains (conservative for warnings, biased for design).
- The CN parameters, alert thresholds, Manning/infiltration values are textbook defaults until
  calibrated against notebook 03's observations and IMD rain records.
- Recharge sites need geotechnical and water-quality screening (parts of Bihar's shallow
  alluvium carry arsenic) before anything is dug.
- This is a planning/prioritisation tool aligned with CPHEEO drainage practice and Atal Bhujal
  Yojana framing — not a substitute for certified engineering design.

## Suggested next milestones

1. Run 01–03 end to end for monsoon 2024–2025; report the CSI score — that number is your credibility.
2. Replace sample groundwater data; hand the top-10 recharge list to someone who can site-visit.
3. Hindcast notebook 04 against dates where 03 saw water; tune alert thresholds.
4. Notebook 05's research extension: make Manning n and infiltration learnable and calibrate
   against SAR masks by gradient descent — that's a publishable, satellite-calibrated
   differentiable flood twin of an Indian city.
5. Re-run everything with CartoDEM 10 m from Bhuvan/Bhoonidhi once registered.
