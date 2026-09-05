# SAR Validation & Calibration — Honest Results (2026-06-27)

Run on Kaggle T4, branch `phase1-sar-calibration`. Twin scored against Sentinel-1 SAR water masks on
the 128x128 / 60 m domain grid, with a consistent JRC permanent-water mask (`jrc<50`) applied to BOTH
prediction and truth. Antecedent rain from the Open-Meteo archive, ECMWF IFS 9 km (2-day window).

## 1. Corrected honest baseline (replaces the stale static number)
The committed `validation_scores.json` (CSI 0.045) scored the **static** `depth.tif` depression-fill,
not the rainfall-driven twin. Scoring the **dynamic twin** instead gives, per storm (tau=0.05 m):

| date | antecedent rain | CSI |
|---|---|---|
| 2023-07-04 | 96 mm | 0.020 |
| 2023-08-09 | 108 mm | 0.015 |
| 2023-10-03 | 96 mm | 0.042 |
| 2024-07-05 | 106 mm | 0.031 |
| 2024-07-07 | 75 mm | 0.000 |
| 2025-07-17 | 69 mm | 0.013 |
| **2025-08-02** | 73 mm | **0.112** |
| 2025-10-04 | 66 mm | 0.035 |
| **mean** | | **0.033** |

Mean CSI by threshold tau: 0.05->0.033, 0.10->0.028, 0.15->0.022, 0.20->0.018, 0.30->0.018.

## 2. Physics calibration: near-null (honest negative result)
Fitting 16 per-WorldCover-class Manning/infiltration multipliers (Adam, soft-Dice vs SAR, 4 train
storms 2023-24, 2 held-out 2025, 20 iters) moved held-out CSI **0.0483 -> 0.0488** (train 0.0169 ->
0.0176); soft-Dice loss flat ~0.955; multipliers stayed in [0.94, 1.14]. See `calibration_report.json`.

**Why (structural, not a bug):** the twin floods ~2900 cells, SAR observes ~2600 — comparable *area*,
~12% *overlap*. The gap is **co-location**. Roughness/infiltration change water depth, not where
topography routes it, so the soft-Dice gradient w.r.t. these params is ~0.

## 3. Orientation check (ruling out a georef bug)
`flipud(SAR)` doubles CSI, which looked like a north-south georef bug. It is NOT: all rasters share
identical north-up geotransforms, and in `dem.tif` the Ganga (`jrc>50`) sits at 41.5 m vs 49.3 m mean
with `corr(-DEM, JRC) = 0.80` un-flipped vs 0.05 flipped. **The DEM is correctly oriented**; the flip
gain is a coincidental mirror with no physical basis and is deliberately NOT applied.

## 4. Conclusion
At 60 m resolution, observed monsoon waterlogging is poorly explained by topographic-sink routing, and
physics-constant calibration cannot close that gap. The contribution is therefore framed as a
reproducible differentiable urban flood twin + gradient canal/dig optimizer + live dashboard, with an
**honest** SAR validation that delimits where topographic routing falls short (motivating finer-DEM /
drainage-aware modeling as future work).

Artifacts: `observed_water_<date>.tif` (8 storms), `calibration_report.json`, `calibrated_params.json`,
`twin_scores_2025-08-02.json`.

---

# Mumbai SAR validation (2026-08-04) — the run G6 said we never did

Same protocol as Patna (dynamic twin, real 2-day antecedent rain per TILE centre — not Patna's,
`calibrate._rain_for_date` had priced Patna rain into every city until this branch; jrc<50 mask on
both sides), on all 4 Mumbai tiles x 11 monsoon storm dates (2025 + 2026, Sentinel-1 passes with
>=25 mm antecedent rain, incl. 213 mm before 2026-07-08). tau = 0.15 m.

| tile | mean CSI | typical FAR | POD at >=90 mm |
|---|---|---|---|
| mumbai_south | 0.039 | 0.93–0.98 | 0.53–0.71 |
| mumbai_west  | 0.047 | 0.92–0.96 | 0.41–0.70 |
| mumbai_east  | 0.043 | 0.93–0.97 | 0.59–0.71 |
| mumbai_north | 0.041 | 0.93–0.96 | 0.46–0.74 |

**Mumbai lands exactly in Patna's regime** (0.033–0.041): at high rain the twin finds most of the
SAR water (POD up to 0.74) while predicting 10–15x too many wet cells. Read with the CSI caution
from `DEPTH_VALIDATION.md`: these CSIs are bought with massive over-wetness (pred_wet vs sar_wet
is in `artifacts/mumbai_sar_baseline.json` per date). Over-flooding where working drainage exists
is precisely gap G2 — which the sink-field inversion on this branch attacks; its results live in
`artifacts/sinkfield_report.json` and the summary below.

## Sink-field inversion (same branch) — the first thing that improves held-out extent

Fitting a **per-cell drainage capacity** through the differentiable twin against these same SAR
masks (8 train storms, 3 held out) improves held-out CSI on **all four tiles while predicting
fewer wet cells** — mean 0.0435 → 0.0479 (+10%) at −10% wetness. The controls ladder on
`mumbai_east` shows the gain is spatial, not extra freedom: one uniform drain rate does nothing
(0.0442 → 0.0439) and the 16-parameter per-class physics calibration makes held-out extent *worse*
(0.0402), reproducing this document's §2 near-null under a stronger loss.

It does **not** transfer to point depths — replaying the 83 held-out depth reports with and without
the field moves nothing past the fourth decimal. Full numbers, method, and limits:
[`SINKFIELD_RESULTS.md`](SINKFIELD_RESULTS.md).
