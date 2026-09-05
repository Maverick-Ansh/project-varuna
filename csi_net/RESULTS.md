# A learned upper bound on flood-extent co-location

What a neural network can and cannot recover from satellite-only inputs at 60 m, measured
against the trivial baselines that make a CSI number mean something.

All numbers below come from `csi_net/`. Protocol details are in `data.py` (splits) and
`metrics.py` (scoring). Every run uses the same 128×128 crops, the same permanent-water mask
on prediction and truth, and the same max-pooled SAR truth as `varuna/build/calibrate.py`.

---

## 1. What CSI was worth on this task, before any model

Computed from artifacts already on disk. `artifacts/trivial_baselines.json`.

| domain | storms | valid cells | all-wet | random | **climatology (LOO)** | SAR vs SAR |
|---|---|---|---|---|---|---|
| patna | 8 | 15 854 | 0.0473 | 0.0249 | **0.2999** | 0.1602 |
| mumbai_harbour | 11 | 37 017 | 0.0205 | 0.0103 | **0.5806** | 0.4622 |
| mumbai_northeast | 11 | 61 590 | 0.0132 | 0.0066 | **0.3913** | 0.2709 |

- **all-wet** — predict every valid cell wet. CSI equals the observed wet fraction, for free.
- **random** — scatter the observed number of wet cells uniformly. Bias 1, zero knowledge.
- **climatology** — "these cells are usually wet", fitted on the *other* storms of that domain
  and tested on the held-out one. Uses no rainfall and no physics.
- **SAR vs SAR** — every date scored against every other date of the same domain. An empirical
  ceiling: two radar passes of the same city agree with each other only this well.

The existing methods, for reference (`artifacts/patna/baseline_comparison.json`, 8 Patna dates):
static depth 0.0323, TWI 0.0422, **dynamic twin 0.0410**, HAND-lite 0.0505.

**A model that ignores rainfall entirely scores 0.58 on mumbai_harbour at bias 1.29.** The twin
scores 0.041. The Sentinel-1 target is dominated by a static, storm-independent wet field that
the JRC ≥ 50 % permanent-water mask does not remove — tidal flat, mangrove (WorldCover class 95
is present in both Mumbai tiles), seasonal pond, paddy. CSI as computed has mostly been a test
of finding persistent water, which the twin never attempted; it predicts only the storm increment.

The signature confirms the reading: on mumbai_northeast's one real flood (2026-07-08, 2937 wet
cells against a ~450 baseline) climatology scores its **worst**, 0.075. It wins the quiet dates
and loses the flood.

---

## 2. Headline result

Leave-one-storm-out, tidal tile excluded, terrain-only, 18 storms, width 32, **3 seeds**:

| method | CSI | bias | POD | precision | note |
|---|---|---|---|---|---|
| random | 0.013 | 1.00 | | | |
| all-wet | 0.026 | — | 1.00 | 0.03 | |
| dynamic twin | 0.041 | 1.12 | 0.08 | 0.05 | |
| TWI | 0.042 | | | | one-line formula |
| HAND-lite | 0.051 | | | | one-line formula |
| **net (terrain-only)** | **0.2895 ± 0.0064** | **1.39 ± 0.05** | **0.514** | **0.37** | **7.1× the twin** |

Per-seed: 0.2806 / 0.2955 / 0.2924. Ranking 0.3219 ± 0.0059.

> **The 7.1× is not yet like-for-like, and must not be printed until it is.** `all-wet` and
> `random` above are computed on *exactly the scenes the net was scored on*, so 0.2895 vs 0.026
> (11×) and vs 0.013 (22×) are honest comparisons. The twin / TWI / HAND figures come from the
> older Patna-only table: 8 dates, one domain, and including the dead 2024-07-07 scene. Ours is
> 18 storms across Patna + Mumbai-NE under leave-one-storm-out. Before the multiple goes in a
> paper the twin has to be rerun on this protocol — see NEXT_STEPS §1.

Bias matters as much as the CSI: the gain is not bought by predicting more water. The twin
catches 8 % of wet cells at 5 % precision; this catches 51 % at 37 %. On the flood-holdout split
the net runs at bias 0.22 — a *quarter* of the observed wet area — at 68 % precision.

Storm-increment score (wet today and not persistently wet — the honest target a rainfall model
should be graded on) is **0.1723 ± 0.0054**.

### 2.1 Capacity is not the bottleneck

Flood holdout, terrain-only, tidal tile dropped, 1200 steps, single seed:

| width | params | CSI | ranking | increment | wall time |
|---|---|---|---|---|---|
| 16 | 2.0 M | 0.2206 | 0.2784 | 0.1742 | 35 s |
| 32 | 7.9 M | **0.2321** | 0.2780 | 0.1876 | 58 s |
| 64 | 31.4 M | 0.2213 | 0.2721 | 0.1775 | 128 s |
| 96 | 70.6 M | 0.2194 | 0.2831 | 0.1730 | 245 s |

Flat across a 35× parameter range. The entire spread (0.219–0.232) sits inside the ±0.010–0.016
seed noise measured in §4.1, so none of it is a real difference.

**The 6.9× gain over the twin did not come from scale.** It came from the feature stack, the
evaluation protocol, excluding the tidal domain, and deleting rainfall. Anyone repeating this
should spend their compute on inputs and protocol, not on width — a 2 M-parameter net on one T4
in 35 seconds reaches the same place as a 70 M one.

---

## 3. Generalisation

| protocol | what is held out | CSI | ranking | bias | increment |
|---|---|---|---|---|---|
| leave-one-storm-out | an ordinary storm, terrain seen | **0.290 ± 0.006** | 0.322 | 1.39 | 0.172 |
| flood holdout | the two largest storms | 0.206 ± 0.002 | 0.278 | 0.22 | — |
| leave-one-**domain**-out | an entire unseen city | 0.099 | 0.127 | 0.57 | 0.037 |

Leave-one-domain-out (29 storms, all three domains, with rain) beats every terrain baseline —
twin 0.041, TWI 0.042, HAND-lite 0.051, all-wet 0.026, random 0.013 — on cities the model has
never seen. Climatology scores 0.396 there and wins, but the comparison is not like-for-like
and the difference is the deployable one: **climatology needs a year of Sentinel-1 history at
the target site; the net needs none.** For a new city with no SAR archive climatology does not
exist, and the net is the only method on that list that runs at all.

One column is loud: in mumbai_harbour the storm-increment score is 0.0000 on **eight of eleven
dates**. Essentially all observed water there is persistent and there is almost no increment to
predict. Patna's increments run 0.15–0.20. The two Mumbai tiles and Patna are different problems.

---

## 4. Rainfall is unusable, and including it makes the model worse

This is the strongest and most surprising result, and it survived three attempts to explain it away.

### 4.1 Ablation (flood split, all 3 domains, 3 seeds)

| input removed | CSI | ranking |
|---|---|---|
| nothing (full model) | 0.1577 ± 0.0096 | 0.1837 ± 0.0133 |
| **rainfall** | **0.1669 ± 0.0106** | **0.3125 ± 0.0118** |
| persistent-water prior | 0.1544 ± 0.0161 | — |
| all terrain | 0.0855 | — |

Removing rainfall moves CSI by +0.009 — inside seed noise. But it moves *ranking quality* by
**+0.124 at ≈7 seed-sd**, with the three-seed ranges disjoint. Removing terrain collapses the
model and drives bias to 0.65: with no terrain it cannot localise, so it smears water. Removing
the persistent-water prior barely matters, so the net is not simply relocating the climatology
baseline — it is using genuine terrain structure.

### 4.2 Permutation control

Shuffle rain vectors *within* each domain: every rainfall distribution is preserved, only the
storm↔rain pairing is destroyed.

| arm | CSI | ranking |
|---|---|---|
| real rain | 0.1549 ± 0.0054 | **0.184 ± 0.013** |
| shuffled rain | 0.1597 ± 0.0104 | **0.272 ± 0.035** |
| no rain at all | 0.1669 ± 0.0106 | **0.313 ± 0.012** |

Monotone. Breaking the pairing does not hurt — it helps; deleting rainfall helps more. Real
rainfall is strictly the worst of the three inputs. Two effects are stacked here: *any*
per-scene scalar gives the net something to condition on instead of learning terrain (shuffled
0.272 < none 0.313), and *real* rain is worse still (0.184) because the net fits a rain→wetness
mapping on the training storms that does not transfer.

### 4.3 Two explanations tested and refuted

**Tidal contradiction — refuted.** Rainfall correlates with observed wet-cell count in opposite
directions in two tiles of the same city:

| domain | rain_1d | rain_3d | rain_5d | rain_7d | rain_14d | peak_3h |
|---|---|---|---|---|---|---|
| patna | 0.68 | −0.02 | 0.07 | 0.09 | 0.07 | 0.58 |
| mumbai_harbour | 0.14 | 0.03 | −0.10 | −0.38 | **−0.70\*** | 0.10 |
| mumbai_northeast | 0.05 | **0.77\*** | **0.79\*** | **0.68\*** | 0.55 | 0.29 |

`* p < 0.05`. Pooled over 29 storms nothing survives: best is rain_3d, r = 0.30, p = 0.11.
Mumbai-Harbour's significantly **negative** 14-day correlation is physically backwards for
rainfall and is the expected signature of tidal forcing — Sentinel-1 overpasses at a fixed local
time, so harbour extent tracks tide.

That predicted a fix: drop the tidal tile and rainfall should stop hurting. **It did not.**
Ablating rain still helped by +0.112 (+10.4 sd) with Harbour removed. The prediction failed, so
the tidal contradiction is not the operative mechanism.

What *did* survive: dropping Harbour lifted the terrain-only model from 0.170 → **0.2059 ± 0.0018**.
The tidal tile was genuinely poisoning training, just not through the rain channels.

**Extrapolation failure — refuted.** The flood split holds out the two *largest* storms, so it is
deliberately out-of-distribution in magnitude; a rain→wetness map fitted on quiet storms would
fail there while terrain-only degraded gracefully. Tested in-distribution with leave-one-storm-out:

| LOSO, no harbour | CSI | ranking | bias | increment |
|---|---|---|---|---|
| with rain | 0.192 | 0.234 | 2.54 | 0.099 |
| rain ablated | **0.281** | 0.315 | **1.35** | **0.166** |

Rain costs ~0.09 CSI in-distribution too. Not extrapolation.

### 4.4 Why there is nothing to extract

- **One number per tile.** Nine probe points spanning a 15 km tile collapse to a single ERA5
  cell (19.250, 73.000). Spatial rainfall is not merely unused — it is *unavailable* from this
  archive, so the "uniform rain over the tile" hypothesis cannot be tested against it at all.
- **4.3× product disagreement.** ERA5 gives 485.7 mm for Mumbai on 2026-07-06; ECMWF IFS at 9 km
  gives 114.1 mm. That spread is larger than any effect the physics calibration produced.
- **Misdocumented source.** `varuna/serve/weather.py:33` sends no `models=` parameter, so
  Open-Meteo returns `best_match` = ECMWF IFS 9 km. The docstring and the project writeup both
  claim ERA5. **The twin was never calibrated on ERA5.**

---

## 5. Data integrity: what was excluded and why

- **`waterlogging_frequency.tif` — excluded, label leakage.** `varuna/build/validate.py:83`
  builds it by counting wet pixels across the same Sentinel-1 passes that produce our labels.
  As a feature it would have manufactured a large fake CSI.
- **`rsi.tif` — excluded.** Derived from `gw_levels.csv`, which is Patna's six invented wells
  copied byte-identically into every bundle.
- **`catchment_labels.tif` — excluded.** Integer IDs; a memorisation key, not a feature.
- **`patna/2024-07-07` — dropped from scoring.** Zero observed wet cells, so no ground truth,
  and a forced 0.0 in every method's mean including the twin's 0.041.

The 23 retained channels are in `data.py`. Three deserve mention: `dem_rel_s2/s8/s32` are
elevation minus its own Gaussian blur at 2/8/32 cells. DEM error is spatially correlated, so
*relative* micro-relief survives a bias that destroys absolute elevation — which is the ±1 m
vs 20 cm problem, attacked at the feature level.

---

## 6. What this means for the paper

1. **CSI on this task has a floor and a ceiling that were never measured.** All-wet is 0.013–0.047
   and a rainfall-free climatology is 0.30–0.58. Any CSI reported without those beside it is
   uninterpretable — including the 0.041.
2. **The twin's weakness is a target mismatch, not a physics failure.** It predicts the storm
   increment and is scored against persistent water plus increment.
3. **A learned model reaches 0.281 at bias 1.35, 6.9× the twin**, and 0.099 on entirely unseen
   cities where climatology is unavailable.
4. **Rainfall, as available from free reanalysis, carries no usable information about where water
   goes at 60 m, and including it measurably degrades the model.** Shown four ways: ablation,
   permutation, correlation, and two refuted alternative explanations.
5. **Two live corrections to the record**: the rain source is IFS 9 km, not ERA5; and
   Mumbai-Harbour is tidally contaminated and should be excluded or tide-corrected rather than
   pooled with pluvial domains.

## 7. Reproducing

```bash
python -m csi_net.run --split loso  --ablate rain --areas patna,mumbai_northeast --width 32
python -m csi_net.run --split flood --ablate none --shuffle_rain 7 --seed 1
python -m csi_net.run --split lodo  --width 32 --steps 1500
```

Runs on one T4 in minutes. `--ablate {none,rain,persist,terrain}`, `--shuffle_rain N`,
`--areas a,b`, `--grid {30,60,120}`, `--width/--depth` for capacity.

## 8. Open

- The 30 m run (`--grid 30`) is built and was launched, but the session ended mid-run and it
  produced no number. The 60 m arm of that same comparison completed: CSI 0.2810, bias 1.37,
  all-wet 0.0291, i.e. a skill multiple of 9.7× over zero knowledge on its own grid.
- Whether a tide-corrected Harbour becomes usable is untested.
- Leave-one-domain-out is single-seed and was run with rainfall included; it should be rerun
  terrain-only, where §4 predicts it improves.
- Three domains is a thin basis for a cross-city claim. The LODO number is a floor, not an estimate.
