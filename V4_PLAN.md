# V4 plan — one city, seen street by street, kept by an agent

V3 shipped metered recharge. V4's depth validation then proved the twin's core failure with two
independent instruments (SAR extent; crowdsourced depths): it knows *how much* water but not
*where*. Everything below is organised around fixing "where" — and around the vision: **join the
tiles into a city, see the water at street level, know what the city pours out, and put a
caretaker agent on duty.**

## Where this stands (V4 so far)

- **Sprint 1 (shipped)**: living map — flow arrows, named danger zones, satellite basemap.
- **Depth validation (shipped)**: the twin fails point depths (skill −0.46, wet-site corr −0.23);
  magnitude half is a 60 m resolution artifact (4.45× ⇒ ~810 m² ponds), ranking half is real.
- **Sprint 2 (shipped, this branch)**: Mumbai SAR validation finally run (baseline mean CSI
  0.039–0.047, FAR ≈ 0.95 — same regime as Patna). **Inferred drainage-sink field** fitted through
  the twin against Sentinel-1: held-out extent improves on all four tiles *while predicting less
  water* (mean CSI 0.0435 → 0.0479 at −10% wetness), beating a 1-parameter and the 16-parameter
  control; it does **not** move point-depth ranking at all. City outflow ≥ 5.3 Mm³ per 100 mm
  storm, saturating by 200 mm. Stitched one-Mumbai domain (817 × 586 @ 60 m) with a measured seam
  cost. Full results: [`SINKFIELD_RESULTS.md`](SINKFIELD_RESULTS.md); code in
  `varuna/build/sinkfield.py` + `varuna/build/city.py`.

## A. One Mumbai — the tile join

**Status: the city is complete and serving.** The four original tiles turned out to be a 2×3
grid with two cells empty — the eastern column's north row (Mulund, Nahur, Bhandup pumping
station, Thane creek) and south row (Trombay, Mahul, Mankhurd, Sewri, Vashi creek) — which the
mosaic had been silently filling with sea. Both are now built (`mumbai_northeast`,
`mumbai_harbour`), a test asserts every grid cell has exactly one owner, and
`varuna/build/city.py` mosaics all six into one 60 m domain that `varuna/serve/city_view.py`
serves from a precomputed storm ladder (`GET /api/city_domain`).

Tiles mosaic cleanly onto the union bounding box (72.745–73.061 E, 18.884–19.324 N) because they
share the DEM's 30 m pixel grid at integer offsets. Cross-seam flow then happens *inside one
Domain* instead of being truncated by six closed boundaries (gap G6).

1. ~~`build_city_domain(tile_works)` — mosaic dem/worldcover/sand/clay, union grid, one Domain~~
   **done** (`varuna/build/city.py`); next: wire it into `areas.py` as a first-class area so the
   API and dashboard can serve "Mumbai (one city)" instead of four tiles.
2. ~~Seam audit~~ **done**: depths near former boundaries disagree 40–85% more than in the
   interior. Next: kill the residual sub-pixel regrid phase difference (mean |Δz| 0.4–1.4 m)
   by cropping tiles from the city mosaic rather than re-deriving each tile's own crop.
3. City-wide flow field + danger zones for the map (Sprint 1 layers recomputed at city scale).
4. Tidal/storm-surge boundary at the western outfalls — for Mumbai the binding constraint
   (backlog #8); without it the city domain still closes at the sea.
5. Later: multi-resolution nesting — city at 60 m, hotspots at 5–10 m (workstream B).

## B. Street level — where the 4.45× goes to die

The depth validation left a number to beat: remove the 4.45× dilution ⇒ resolve ~28 m ponds.

1. Sub-grid street conveyance (backlog #4): streets as 1D channels inside 60 m cells using the
   existing `road_graph.json.gz`; carved drains stop being full-cell artifacts (G3).
2. CartoDEM 10 m rebuild of one ward (backlog #5): the clean test of whether finer relief alone
   fixes co-location — our own thesis, falsifiable.
3. ~~Map: zoom switches to satellite + street-projected flow arrows, per-street depth in mm~~
   **done** — `varuna/serve/streets.py` + `POST /api/streets` + the `StreetWater` layer. It
   keeps ponded streets (the flow layer drops them, and they are usually the worst place to be)
   and reports both the cell mean and a street-depth estimate using the measured 4.45×
   concentration. Next: the estimate is a single median ratio for the whole city — it should
   vary with how much of the cell is street rather than being constant, which needs the sub-grid
   conveyance work in item 1.

## C. Reverse-engineering the city — outflow and assimilation

The sink field *is* the reverse engineering: given observed extent (SAR) and depths (reports),
infer the drainage the city must have, and integrate `drained_volume_m3` per storm — "the city
poured out ≥ X m³ during this storm", per tile and city-wide, on the dashboard next to the
rain total. Next:

1. Surface outflow + per-tile drain maps on the map UI (a "what the city removes" layer).
2. Rolling assimilation: refit the field each monsoon month as new S1 passes + reports arrive;
   the field's *change* is drainage degradation/clogging seen from space (feeds backlog #23,
   the maintenance prioritiser).
3. Cross-check against BMC/BRIMSTOWAD pumping-station capacities where published — turns a
   lower bound into a calibration.

## D. The city caretaker — an agent on duty

All the pieces exist separately (chat agent + tools, nightly loop, alerts, news stage 2.5,
reward gate). The caretaker is their orchestration into one daemon with a memory:

1. **Watch**: Open-Meteo nowcast + IMD bulletins every few hours (backlog #19 lite).
2. **Simulate**: on rain signal, run the city domain for the nowcast total; refresh danger
   zones, flow arrows, ward alerts.
3. **Learn**: run the nightly citizen-report loop for real (backlog #20 — deployed, never
   exercised; the reward gate finally fires on live data), and the monthly sink-field refit (C2).
4. **Report**: a daily caretaker note — what it watched, simulated, changed, refused to change
   (the reward gate's refusals are the trust signal) — via the existing agent/chat surface.
5. Implementation: a single `varuna caretaker` loop (cron/HF Space scheduled job) + a state
   file; the LLM agent narrates and answers, the loop does the work.

## E. Learning systems — what "reinforcement" concretely means here

1. **Reward-gated nightly fine-tune** (exists): citizen reports nudge the emulator only if the
   replay-buffer anchor says the grid didn't degrade. Run it nightly for a month; report
   accepted/refused updates.
2. **SAR-supervised FloodGNN** (backlog #6): train the GNN on observed S1 water directly, not
   on the twin's own output — removes the "distills the twin" caveat; the biggest single
   scientific upgrade available.
3. **RL on interventions**: the differentiable optimiser already does portfolio design by
   gradient; the honest RL experiment is a policy (PPO) proposing drain/storage portfolios on
   the city domain with flood-cut-per-rupee as reward, compared against the gradient optimiser
   on identical budgets. GPU budget is available; free-tier T4 is no longer the constraint.
4. **Nowcast emulator**: train the U-Net on real hyetographs (hourly Open-Meteo shapes, not the
   1.5 h design storm) so "next 3 hours" is an emulator call (backlog #19) — removes the
   storm-shape excuse the depth validation had to grant the model.

## Sequencing

Sprint 2 (this branch) → **S3**: city join hardened + seam audit + tidal boundary + city map
layers → **S4**: street-level view + sub-grid conveyance + CartoDEM ward → **S5**: caretaker
daemon + nightly loop actually running → **S6**: GNN-on-SAR + RL-vs-gradient study + nowcast
emulator. The paper gains a V4 section after S3 (city + sink field are publishable together:
"satellite-inferred effective drainage of an Indian megacity").

## Validation discipline (unchanged, non-negotiable)

CSI is never quoted without a wetness figure. Depth reports and news labels never train
anything. Every learned field ships with its held-out numbers, including the ones that lose.
