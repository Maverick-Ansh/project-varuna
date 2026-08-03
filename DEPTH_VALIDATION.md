# Depth validation — the twin vs real observed water depths

**First point-depth validation in the project. The model does not pass it.**

Reproduce: `python scripts/run_depth_validation.py --days 365 --window 24 --min-rain 10 --sweep 3 6 12 24 48`
Artifacts: `artifacts/depth_observations.json` (the QC'd set), `artifacts/depth_validation.json` (metrics).

---

## Why this exists

Every validation Varuna had done measured **extent**: Sentinel-1 says wet or dry, the twin says
wet or dry, CSI compares the masks (`VALIDATION.md`, CSI ≈ 0.033–0.041). That never asked the
question a resident asks — *how deep is the water on my street, and were you right?*

This does. It is a completely independent test: different data source (people, not radar),
different quantity (depth, not extent), different failure modes.

## The data

**mumbaiflood.in** (IIT Bombay IDP Climate Studies + MCGM/MCMCR) exposes a public API. Its
`/cs/map/` endpoint carries crowdsourced water-depth reports: the reporter's height in feet and
inches, and the fraction of themselves the water reached. `depth = fraction × height`.

147 raw rows, 2026-02-11 → 2026-08-03. Quality control (every rule counted, `quality_control` in
the report):

| dropped | rule | why |
|---|---|---|
| 38 | `no_depth_reported` | the report exists, the depth field is null |
| 15 | `outside_region` | the app is reachable worldwide; the live feed contains points in Argentina and Australia |
| 2 | `rain_too_low_to_be_pluvial` | 1.42 m of standing water at Worli after **3.8 mm** of rain in 24 h — whatever that is, it is not rainfall-driven waterlogging |

**94 survive; 85 fall inside a built Mumbai tile; 83 scored.**
Personal data (`name`, free-text `feedback`) is read and discarded at parse time and is
test-pinned never to reach a file — see `test_personal_data_never_reaches_an_observation`.

Rainfall forcing is Open-Meteo (ERA5 archive / forecast `past_days`), summed over the window
before each report. The mumbaiflood AWS endpoint serves **today only** and cannot force a past storm.

## Result

83 observations, 24 h antecedent window, ≥10 mm rain:

| metric | value |
|---|---|
| MAE | **0.207 m** |
| RMSE | **0.310 m** |
| bias | **−0.162 m** (systematic under-prediction) |
| **skill vs best trivial baseline** | **−0.46** |
| correlation (all) | +0.055 |
| correlation (observed ≥ 0.15 m) | **−0.23** |
| CSI (wet/dry @ 0.15 m) | 0.244 (POD 0.29, FAR 0.39) |

Baseline RMSE: always-dry 0.379 · **observed mean 0.257** · observed median 0.324.

**Negative skill means the model is worse than predicting the same number everywhere.** Per tile
it ranges −0.40 (mumbai_west) to −0.64 (mumbai_south); no tile is positive.

## Is it an artifact of our choices? No.

### The antecedent window doesn't rescue it

| window | mean rain | RMSE | skill | correlation | CSI | predicted wet |
|---|---|---|---|---|---|---|
| 3 h | 17.8 mm | 0.363 | −0.93 | −0.096 | 0.000 | 1 / 22 |
| 6 h | 26.4 mm | 0.327 | −0.83 | +0.012 | 0.027 | 2 / 36 |
| 12 h | 47.3 mm | 0.309 | −0.65 | +0.047 | 0.100 | 7 / 37 |
| 24 h | 80.1 mm | 0.310 | −0.46 | +0.055 | 0.244 | 18 / 38 |
| 48 h | 142.7 mm | 0.376 | −0.77 | −0.057 | 0.393 | 38 / 40 |

Skill is negative at every window; correlation never leaves zero.

### …and this table contains a warning about CSI itself

Read the last two columns together. **CSI climbs 0.00 → 0.39 purely as the model gets wetter**
(1 → 38 wet predictions) while correlation stays flat at ~0 and skill stays negative. On a report
set biased toward places that flooded, CSI rewards a model for predicting more water, not for
knowing where it goes. A wetter model scores better while being no more right.

Since the project's headline validation number *is* a CSI, this is a caution that applies to
`VALIDATION.md` too: CSI must always be read next to a bias/wetness figure, or it flatters.

### Neighbourhood verification doesn't rescue it either

A 60 m model that puts a flood one cell over is not really wrong, so we re-score against the best
cell in a box around each point (standard practice in forecast verification).

| box | predicted mean | RMSE | correlation | correlation (wet) |
|---|---|---|---|---|
| 60 m | 0.117 m | 0.310 | +0.055 | −0.225 |
| 180 m | 0.145 m | 0.308 | −0.025 | −0.373 |
| 300 m | 0.176 m | 0.310 | −0.101 | **−0.454** |
| 420 m | 0.219 m | 0.316 | −0.148 | −0.420 |
| 540 m | 0.275 m | 0.333 | −0.038 | −0.148 |

Widening the box closes the *bias* (0.117 → 0.275 m) but never improves RMSE, and on the sites
that actually flooded the correlation gets **more negative**. Allowing the model to nominate the
worst cell within 300 m still leaves it unable to say which reported location was deeper.

## Two failure modes, separated

The two experiments above pull the failure apart cleanly.

**1. Magnitude — a resolution artifact, measured and curable.**
On wet sites the model under-predicts by a stable factor of **4.45×** (n=38, identical at the
0.15 m and 0.30 m thresholds). That ratio is not mysterious. A person reports the depth where
they are standing; the model reports the **mean of a 3,600 m² cell**. If the same water sits in
a pond of area *A* inside the cell, `observed/predicted ≈ cell_area / A`, so:

> **implied ponding footprint = 3,600 / 4.45 ≈ 810 m² — a square 28.5 m across.**

That is a flooded street junction. The model is not wrong about how much water there is; it is
spreading it over sixty metres of ground. **This one is fixable with grid resolution** — and it
is a quantitative target for the 5 m street twin, which should remove most of the 4.45×.

**2. Ranking — the real failure, and resolution alone will not fix it.**
Zero-to-negative correlation, robust across every window and every neighbourhood size. The model
cannot say which street floods worse. This is the same co-location failure the SAR work found —
now confirmed by an **independent data source measuring an independent quantity**. Two methods,
two datasets, same verdict.

## Honest limits of this result

- **n = 83, one city, one season.** The wet-site anti-correlation (−0.23, p≈0.08) is suggestive,
  not established. Do not quote it as a significant negative correlation.
- **Reporting bias.** People report where water is surprising or bad — plausibly the places the
  model least expects. That alone could produce anti-correlation without the model being wrong in
  the way it looks. It cannot be ruled out from these data.
- **Point vs cell average.** As above, these are genuinely different quantities. The dilution
  analysis quantifies the gap rather than pretending it away, but the comparison is not apples
  to apples and the 4.45× is a *consistency* argument, not proof.
- **Storm shape.** The emulator is forced with a rainfall total and internally assumes a 1.5 h
  design storm. A miss can be storm shape as easily as terrain.
- **Timing.** An observation is a moment; `hmax` is the storm maximum. Reports made hours after
  the peak are compared against a peak the model is entitled to predict.

## These observations never train anything

They are held for scoring only, so they stay a clean test set — mirroring the existing
`learn/news_labels.py` quarantine. The nightly learner is fed citizen reports through a separate
path with its own reward gate (`learn/reward.py`).

## What this changes

1. **The 5 m street twin gets a number to beat**: remove the 4.45× dilution, and the implied
   target resolution is ~28 m or finer.
2. **CSI alone is not a sufficient headline** anywhere in this project.
3. **Co-location is confirmed as the core problem** by a second, independent method — which
   strengthens the case for the things that change *where* water goes (spatially varying rainfall,
   an inferred drainage-sink field) over the things that change how deep it gets.
