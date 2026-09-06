# A learned upper bound on flood-extent co-location

What a neural network can and cannot recover from satellite-only inputs at 60 m, measured
against the trivial baselines that make a CSI number mean something.

All numbers below come from `csi_net/`. Protocol details are in `data.py` (splits) and
`metrics.py` (scoring). Every run uses the same 128×128 crops, the same permanent-water mask
on prediction and truth, and the same max-pooled SAR truth as `varuna/build/calibrate.py`.
Every per-run JSON is committed under `csi_net/results/` — the Kaggle kernel that produced
them is not storage, and an earlier set of headline seeds was lost that way.

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

**A model that ignores rainfall entirely scores 0.58 on mumbai_harbour at bias 1.29.** The
Sentinel-1 target is dominated by a static, storm-independent wet field that the JRC ≥ 50 %
permanent-water mask does not remove — tidal flat, mangrove (WorldCover class 95 is present in
both Mumbai tiles), seasonal pond, paddy. CSI as computed has mostly been a test of finding
persistent water, which the twin never attempted; it predicts only the storm increment.

The signature confirms the reading: on mumbai_northeast's one real flood (2026-07-08, 2937 wet
cells against a ~450 baseline) climatology scores its **worst**, 0.075. It wins the quiet dates
and loses the flood.

---

## 2. The existing methods, re-measured on the scenes we actually report

The previous version of this document compared a net measured over 18 storms under
leave-one-storm-out against a twin measured on 8 Patna dates with one globally-swept threshold
and a dead scene included, and printed the ratio. Different test sets, so that ratio was not a
measurement. `twin_protocol.py` closes the gap: same 18 storms, same 60 m grid, same
permanent-water mask, same max-pooled truth, and a threshold chosen on the *other* storms of the
fold rather than on the scene being reported.

Grid alignment is proven, not assumed — the feature stack's truth and mask are checked cell for
cell against `calibrate.align_sar` / `valid_mask` on the twin's own crop before anything is
scored: **0 of 16 384 cells differ in Patna, 0 of 65 536 in Mumbai-NE**, and 0 of 851 968 truth
cells across all dates.

**The scoring path is verified against the project's own committed table** (`--reproduce`, old
protocol, 8 Patna dates, one global threshold):

| method | committed | re-derived here | delta |
|---|---|---|---|
| dynamic twin | 0.0412 | 0.0409 | −0.0003 |
| TWI | 0.0422 | 0.0429 | +0.0007 |
| HAND-lite | 0.0505 | 0.0504 | −0.0001 |
| static depression depth | 0.0323 | 0.0287 | −0.0036 |

Static depth is the one known difference: `baselines.py` max-pools `depth.tif` to 60 m and the
feature stack mean-pools it. The other three reproduce, so the new numbers below are the same
scoring code answering a different question, not a different scoring code.

That check earned its keep immediately. The first version of `twin_protocol.py` swept HAND-lite
over pooled quantiles 0.01–0.50, which top out below 1 m in Patna — while the threshold the
original table selected was **8.57 m**. HAND-lite was being scored at a threshold it would never
have chosen, and the regression check is what surfaced it.

### 2.1 The like-for-like table

18 storms, patna + mumbai_northeast, 60 m, dead scene dropped. `csi_net/results/twin_protocol.json`.

| method | LOSO threshold | per-domain threshold | old global rule | POD |
|---|---|---|---|---|
| **dynamic twin** | 0.0450 ± 0.0358 | **0.0536** | 0.0450 | 0.291 |
| TWI | 0.0387 | 0.0394 | 0.0403 | 0.262 |
| HAND-lite | 0.0324 | 0.0394 | 0.0352 | 0.897 |
| static depression depth | 0.0310 | 0.0315 | 0.0319 | 0.206 |
| all-wet | 0.0291 | — | — | 1.000 |
| random | 0.0140 | — | — | |
| **climatology** | **0.2881** | — | — | |

"per-domain threshold" lets each city keep its own, chosen on that city's other storms. It is
more generous than one pooled threshold — Patna's HAND runs to 15 m and Mumbai-NE's to 200 m —
and the net gets per-domain standardisation, so this is the column the baselines are quoted at.

**Two corrections to the project's own record, both against us:**

1. **The twin is better than its published 0.0410, not worse.** On this protocol it reaches
   **0.0536**. Dropping the dead 2024-07-07 scene (a forced 0.0 in every method's mean) and
   adding a second domain both help it. The old number understated the twin.
2. **The optimism in the old protocol was negligible.** Sweeping one threshold over all dates
   including the test scene bought between +0.000 and +0.005. The old table's weakness was its
   test set, not its threshold rule.

HAND-lite's 0.897 POD is worth reading beside its CSI: at its chosen threshold it calls almost
everything wet, so its 0.0394 is barely distinguishable from all-wet's 0.0291 by construction.

---

## 3. Headline result

Leave-one-storm-out, tidal tile excluded, terrain-only, 18 storms, width 32, **3 seeds**:

| method | CSI | bias | POD | precision |
|---|---|---|---|---|
| random | 0.0140 | 1.00 | | |
| all-wet | 0.0291 | — | 1.00 | 0.03 |
| static depression depth | 0.0315 | | 0.21 | |
| TWI | 0.0394 | | 0.26 | |
| HAND-lite | 0.0394 | | 0.90 | |
| dynamic twin | 0.0536 | | 0.29 | |
| **net (terrain-only)** | **0.2884 ± 0.0066** | **1.41 ± 0.05** | **0.517** | **0.37** |
| **climatology (no model, no rain)** | **0.2881** | 1.29 | | |

Per-seed: 0.2795 / 0.2954 / 0.2903. Ranking 0.3217 ± 0.0062.
`csi_net/results/loso_rain_w32_g60_s{0,1,2}_patna+mumbai_northeast.json`.

- **5.4× the physics twin**, on identical scenes, with both thresholds chosen off the test scene.
- **9.9× all-wet**, **21× random**.
- **1.001× climatology.** The net and "these cells are usually wet" are the same number to three
  decimal places: 0.2884 against 0.2881, a gap of **0.0003** against a seed standard deviation of
  0.0066.

That last line is the honest headline and it is not the one this document previously carried.
Where the city has been seen in training, a 7.9 M-parameter U-Net over 23 terrain channels does
not beat a rainfall-free, model-free climatology. What it buys is not accuracy on a known city.
It is §4: the climatology cannot be computed at all for a city with no Sentinel-1 archive, and
the net can.

Bias matters as much as the CSI: the gain over the twin is not bought by predicting more water.
The twin catches 29 % of wet cells; this catches 51 % at 37 % precision. On the flood-holdout
split the net runs at bias 0.26 — a *quarter* of the observed wet area — catching 22 % of wet
cells, so what it does predict there it predicts precisely.

Storm-increment score (wet today and not persistently wet) is **0.1711 ± 0.0059**.

### 3.1 Capacity is not the bottleneck

Flood holdout, terrain-only, tidal tile dropped, 1200 steps, single seed:

| width | params | CSI | ranking | increment | wall time |
|---|---|---|---|---|---|
| 16 | 2.0 M | 0.2206 | 0.2784 | 0.1742 | 35 s |
| 32 | 7.9 M | **0.2321** | 0.2780 | 0.1876 | 58 s |
| 64 | 31.4 M | 0.2213 | 0.2721 | 0.1775 | 128 s |
| 96 | 70.6 M | 0.2194 | 0.2831 | 0.1730 | 245 s |

Flat across a 35× parameter range; the entire spread sits inside seed noise. Anyone repeating
this should spend compute on inputs and protocol, not on width.

### 3.2 Resolution is not the bottleneck either

The project's standing explanation for poor co-location was scale: real ponds measure ~810 m²
(28.5 m across) and were being scored on 60 m cells. Tested directly, same protocol, same seed,
terrain-only, patna + mumbai_northeast:

| grid | CSI | all-wet | **CSI ÷ all-wet** | bias | increment |
|---|---|---|---|---|---|
| 60 m | 0.2814 | 0.0291 | **9.7×** | 1.35 | 0.1659 |
| 30 m | 0.2380 | 0.0228 | **10.4×** | 1.77 | 0.1516 |

Raw CSI is *not* comparable across grids — the base rates differ, which is exactly why the
skill multiple is the column to read. On that column the finer grid is 7 % better, well inside
what a single seed can produce. Quadrupling the number of cells does not unlock co-location, so
the 60 m cell size was not what was standing in the way.

---

## 4. Generalisation

| protocol | what is held out | CSI | ranking | bias | increment |
|---|---|---|---|---|---|
| leave-one-storm-out | an ordinary storm, terrain seen | **0.2884 ± 0.0066** | 0.322 | 1.41 | 0.171 |
| flood holdout | the two largest storms | 0.2211 ± 0.0064 | 0.273 | 0.26 | 0.177 |
| leave-one-**domain**-out, 3 domains | one tile, 2 domains to learn from | 0.0936 ± 0.0251 | 0.128 | 1.07 | 0.039 |
| leave-one-**domain**-out, 14 domains | one tile, 13 domains to learn from | **0.2698 ± 0.0146** | 0.303 | 1.20 | 0.110 |
| leave-one-**region**-out, 14 domains | an entire city or region | **0.1305 ± 0.0046** | 0.161 | 0.53 | 0.057 |

### 4.1 Eleven more domains, and what they did to the transfer claim

The three-domain LODO number was reported as *a floor, not an estimate*. That was the right call.
`scripts/fetch_sar_masks.py` fetched **110 Sentinel-1 masks across 11 new areas** (Bengaluru, the
four remaining Mumbai tiles, six Karnataka towns) after one interactive Earth Engine login, taking
the dataset from 3 domains to **14** and from 30 scenes to 140. All 110 cleared the wet-fraction
floor; median wet fractions 0.003–0.012, in line with the three existing domains.

`patna_east` and `patna_west` were deliberately **not** fetched: they carry `source_work="patna"`
and share its exact AOI, so they are sub-crops of one scene and would have added duplicate masks
under different names.

Retraining with 13 domains available instead of 2 nearly triples per-tile transfer, 0.0936 to
**0.2698**, and tightens the seed spread from ±0.0251 to ±0.0146.

### 4.2 The number that survives the control: leave-one-REGION-out

Per-tile LODO stops meaning "unseen city" once the dataset has six Mumbai tiles and seven
Karnataka domains. Holding out `mumbai_west` leaves five adjacent Mumbai tiles in training;
holding out Chitradurga leaves six other Karnataka towns with the same climate, geology and tank
morphology. The three best LODO folds are exactly Chikkaballapur, Chitradurga and Kolar.

`--split loro` groups the domains into `patna` / `mumbai` / `karnataka` and holds an entire region
out, so the target has no sibling of any kind in training. It **halves** the result:

| held out | CSI | × all-wet | seed sd |
|---|---|---|---|
| one tile, 3 domains, original stack | 0.0936 | 3.6 | 0.0251 |
| one tile, 3 domains, rebuilt stack (control) | 0.0960 | 3.7 | 0.0090 |
| one tile, 14 domains | 0.2698 | 12.5 | 0.0146 |
| **one region, 14 domains** | **0.1305** | **6.1** | **0.0046** |

So roughly half the apparent gain was siblings. What remains is real but smaller: **0.1305 against
0.0936, a 1.4× improvement at a fifth of the seed spread.** The rebuilt-stack control rules out
the alternative explanation — rebuilding the feature stack with `build_stack`'s own TWI
definition, on the same three domains, changes nothing.

### 4.3 Inland transfer works; coastal transfer does not

The per-region result is not uniform, and this is the part with operational consequences:

| region held out | test scenes | CSI | × all-wet |
|---|---|---|---|
| karnataka (7 domains) | 210 | 0.2131 | 10.3 |
| patna (1 domain) | 21 | 0.1660 | 3.1 |
| **mumbai (6 tiles)** | 186 | **0.0333** | **1.8** |

With no Mumbai tile in training, the model is barely above all-wet on Mumbai. Mumbai's Sentinel-1
water is tidal flat, creek and mangrove (WorldCover class 95); neither the Gangetic floodplain nor
the Karnataka plateau contains that surface, and the model does not invent it. The honest
statement is **inland-to-inland transfer works, inland-to-coastal does not** — deploy on an
unseen inland city, do not deploy on an unseen coastal one without a coastal domain in training.

Bias is also worth reading: 1.20 on unseen tiles but **0.53** on unseen regions. On genuinely new
ground the model predicts about half the water that is there, so it under-warns rather than
over-warns — the wrong direction for a flood product, and a calibration problem rather than a
ranking one (ranking holds up better, 0.161 against 0.303).

Rainfall on unseen regions changes nothing, as on unseen tiles: 0.1118 ± 0.0345 with rain against
0.1305 ± 0.0046 without. The claim of §5 now holds at 14 domains and at two grouping levels.

On an unseen region the net still beats every terrain baseline — twin 0.054, TWI 0.039,
HAND-lite 0.039, all-wet 0.022, random 0.011. Climatology scores 0.545 across these domains and
wins wherever it can be computed, but that uses the held-out city's *own* Sentinel-1 dates.
**For a new city with no SAR archive climatology does not exist, and the net is the only method on
this list that runs at all.** That, bounded by the coastal result, is the deployable claim.

One column is loud: in mumbai_harbour the storm-increment score is 0.0000 on **eight of eleven
dates**. Essentially all observed water there is persistent and there is almost no increment to
predict. Patna's increments run 0.15–0.20. The two Mumbai tiles and Patna are different problems.

---

## 5. Rainfall is unusable where the city is known — and simply irrelevant where it is not

This is the strongest and most surprising result, and it survived three attempts to explain it
away. It also now has a stated boundary, which it did not before.

### 5.1 Ablation (flood split, all 3 domains, 3 seeds)

| input removed | CSI | ranking |
|---|---|---|
| nothing (full model) | 0.1577 ± 0.0096 | 0.1837 ± 0.0133 |
| **rainfall** | **0.1669 ± 0.0106** | **0.3125 ± 0.0118** |
| persistent-water prior | 0.1544 ± 0.0161 | — |
| all terrain | 0.0855 | — |

Removing rainfall moves CSI by +0.009 — inside seed noise. But it moves *ranking quality* by
**+0.124 at ≈7 seed-sd**, with the three-seed ranges disjoint. Removing terrain collapses the
model and drives bias to 0.65. Removing the persistent-water prior barely matters, so the net is
not simply relocating the climatology baseline — it is using genuine terrain structure.

### 5.2 Permutation control

Shuffle rain vectors *within* each domain: every rainfall distribution is preserved, only the
storm↔rain pairing is destroyed.

| arm | CSI | ranking |
|---|---|---|
| real rain | 0.1549 ± 0.0054 | **0.184 ± 0.013** |
| shuffled rain | 0.1597 ± 0.0104 | **0.272 ± 0.035** |
| no rain at all | 0.1669 ± 0.0106 | **0.313 ± 0.012** |

Monotone. Breaking the pairing does not hurt — it helps; deleting rainfall helps more. Two
effects are stacked: *any* per-scene scalar gives the net something to condition on instead of
learning terrain, and *real* rain is worse still because the net fits a rain→wetness mapping on
the training storms that does not transfer.

### 5.3 Where the effect stops: unseen cities

Every rainfall result above is measured on a split where the domain appears in training. On
leave-one-domain-out, matched arms at 3 seeds each:

| LODO arm | CSI | bias | increment |
|---|---|---|---|
| with rain | 0.0939 ± 0.0161 | 0.70 ± 0.15 | 0.0398 |
| rain ablated | 0.0936 ± 0.0205 | 1.07 ± 0.06 | 0.0390 |

**Identical** — 0.0003 apart against seed sds of 0.016–0.021. Rainfall neither helps nor hurts
on a city the model has never seen. The claim is therefore not "rainfall is always harmful"; it
is *rainfall is harmful when the model can use it to shortcut a domain it has already seen, and
inert otherwise.* The one thing rainfall does change on unseen cities is bias — 0.70 with rain
against 1.07 without, i.e. it makes the model under-predict flood area by a third.

### 5.4 Two explanations tested and refuted

**Tidal contradiction — refuted twice now.** Rainfall correlates with observed wet-cell count in
opposite directions in two tiles of the same city:

| domain | rain_1d | rain_3d | rain_5d | rain_7d | rain_14d | peak_3h |
|---|---|---|---|---|---|---|
| patna | 0.68 | −0.02 | 0.07 | 0.09 | 0.07 | 0.58 |
| mumbai_harbour | 0.14 | 0.03 | −0.10 | −0.38 | **−0.70\*** | 0.10 |
| mumbai_northeast | 0.05 | **0.77\*** | **0.79\*** | **0.68\*** | 0.55 | 0.29 |

`* p < 0.05`. Pooled over 29 storms nothing survives: best is rain_3d, r = 0.30, p = 0.11.
Mumbai-Harbour's significantly **negative** 14-day correlation is physically backwards for
rainfall, and was read as the signature of tidal forcing.

That prediction was tested and failed: dropping the tidal tile should have stopped rainfall
hurting, and it did not — ablating rain still helped by +0.112 (+10.4 sd) with Harbour removed.
§6 tests the tidal reading itself, directly, and it fails there too.

What *did* survive: dropping Harbour lifted the terrain-only model from 0.170 → **0.2059 ± 0.0018**.
The tidal tile was genuinely poisoning training, just not through the rain channels.

**Extrapolation failure — refuted.** The flood split holds out the two *largest* storms, so a
rain→wetness map fitted on quiet storms would fail there while terrain-only degraded gracefully.
Tested in-distribution with leave-one-storm-out:

| LOSO, no harbour | CSI | ranking | bias | increment |
|---|---|---|---|---|
| with rain | 0.192 | 0.234 | 2.54 | 0.099 |
| rain ablated | **0.281** | 0.315 | **1.35** | **0.166** |

Rain costs ~0.09 CSI in-distribution too. Not extrapolation.

### 5.5 Why there is nothing to extract, and what the rain actually is

- **One number per tile.** Nine probe points spanning a 15 km tile collapse to a single archive
  cell. Spatial rainfall is not merely unused — it is *unavailable* from this archive.
- **The source is ECMWF IFS 9 km, and was never ERA5.** `varuna/serve/weather.py` sent no
  `models=` parameter, so Open-Meteo answered with `best_match`. Measured at the Mumbai-NE
  centre over 2026-06-06 → 2026-07-06:

  | `models=` | total |
  |---|---|
  | *(absent — `best_match`)* | 554.2 mm |
  | `ecmwf_ifs` | **554.2 mm** — identical |
  | `era5` | 1213.4 mm — **2.19×** |
  | `era5_land` | 0.0 mm — no data at this coastal cell |

  The docstring, `VALIDATION.md`, the paper and the project writeup all said ERA5. **The twin was
  never calibrated on ERA5.** The model is now pinned to `ecmwf_ifs`, which reproduces every
  existing number exactly (the function still returns 554.2 mm) and stops `best_match` drifting
  under the project later. Switching to ERA5 remains defensible but is not free: it costs a full
  recalibration, and the 2.2× spread is larger than any effect that calibration produced.

---

## 6. Mumbai-Harbour: the tidal explanation, tested

Harbour has been excluded from the headline on the strength of an *interpretation* — that its
backwards rain correlation is tidal. `tide_probe.py` tests that interpretation with the cheapest
instrument that can carry it. `data.tide_vec` derives tide phase from the calendar alone: the
sin/cos of the synodic phase and of its second harmonic, covering the spring–neap beat and the
~14.8-day aliasing of the lunar semidiurnal tide at Sentinel-1's fixed overpass hour.

Against each domain's observed wet fraction, with a permutation p-value because four predictors
on eleven scenes will fit noise well:

| domain | n | tide R² | perm p | rain R² | perm p |
|---|---|---|---|---|---|
| patna | 7 | 0.985 | 0.063 | 0.672 | 0.156 |
| **mumbai_harbour** | 11 | **0.405** | **0.475** | 0.489 | 0.072 |
| mumbai_northeast | 11 | 0.438 | 0.238 | 0.405 | 0.138 |

**The tide story does not survive.** In the one domain where it predicted signal, random
permutations of the labels fit as well roughly half the time. And Patna — 1 000 km inland, no
tide — produces the best tidal fit in the table, which is exactly what four predictors on seven
scenes buys and is the clearest possible warning against reading the R² instead of the p.

The instrument's limits belong in the claim: a calendar phase carries no amplitude, no
bathymetry and no surge, so this refutes *the cheap version* of the tide hypothesis, not the
existence of tidal forcing. A real gauge or an FES2014 sample could still find what this cannot.
What it does establish is that Harbour is not rescuable by anything derivable from the date, and
that the tidal reading in §5.4 was never measured before being used to justify dropping a domain.

### 6.1 Giving the network the tide phase makes it worse

The probe above is linear and works on a domain mean; a convolutional net could in principle use
the phase spatially. `--tide 1` appends the four channels and retrains. LOSO, all three domains,
terrain-only, 3 seeds (per-domain columns are the first 2 seeds):

| arm | CSI | bias | patna | mumbai_ne | **mumbai_harbour** |
|---|---|---|---|---|---|
| no tide (control) | **0.3391 ± 0.0062** | 1.43 | 0.2726 | 0.2587 | **0.4714** |
| + tide channels | 0.3264 ± 0.0033 | 1.19 | 0.3077 | 0.2193 | **0.4445** |

Tide phase costs 0.0127 CSI at ≈2–4 seed-sd, and it costs the most in **Harbour**, the domain it
was introduced to rescue. The per-domain column is the giveaway: the channels *help* landlocked
Patna (+0.035) and hurt both coastal tiles. That is the §5.2 fingerprint effect exactly — any
per-scene scalar gives the net something to condition on instead of learning terrain — and it is
positive evidence that these channels carry no tidal physics, only scene identity.

### 6.2 The decision, and a number that cuts the other way

Harbour stays excluded from the headline. The stated reason changes: not "it is tidally
contaminated" (never measured, and the cheap test failed), but *its observed water is almost
entirely persistent — a 0.0000 storm increment on 8 of 11 dates — so it is a different hazard
class from urban pluvial flooding.*

The effect of pooling it depends on the split, and it is worth stating both directions rather
than quoting the convenient one:

| split | with Harbour | without | effect of pooling |
|---|---|---|---|
| flood holdout (train quiet → test the 2 big floods) | 0.170 | 0.2059 ± 0.0018 | **−0.036** |
| leave-one-storm-out | 0.3391 ± 0.0062 | 0.2884 ± 0.0066 | **+0.051** |

On the flood split Harbour degrades the model, which is the number this document previously
quoted. On leave-one-storm-out it *inflates* the pooled mean by more than it degraded the other,
because its scenes are the easiest in the dataset — a climatology scores 0.58 there. Reporting a
headline over all three domains would therefore have been flattering, not conservative. Excluding
Harbour is the stricter choice on the split the headline uses, and that is the honest reason to
keep it excluded.

---

## 7. Two things that did not work, reported because they were tried

### 7.1 Training on the storm increment does not help

The net has been trained on the full SAR mask and *scored* on the increment, while the twin
predicts only the increment. `--target increment` trains on the same quantity it is graded on,
with the persistent field built from training scenes only so no test label leaks in. LOSO,
terrain-only, patna + mumbai_northeast, 3 seeds:

| training target | increment CSI | full-mask CSI |
|---|---|---|
| full SAR mask | 0.1711 ± 0.0059 | **0.2884 ± 0.0066** |
| storm increment | **0.1757 ± 0.0038** | 0.2444 ± 0.0064 |

+0.0046 on the increment, against seed sds of 0.004–0.006. A null. Training directly on the
quantity that matters does not make the storm increment more predictable, which is the strongest
evidence yet that the increment is close to unlearnable from these inputs rather than merely
mis-targeted. Full-mask CSI drops as expected once the model stops being asked to find
persistent water.

### 7.2 Threshold calibration helps only where the domain is unseen

Ranking quality has run consistently above achieved CSI in every experiment here, and that gap is
calibration loss. `calibrated_threshold` picks a per-scene threshold that matches the predicted
wet fraction to the *training* scenes' wet fraction — a number available without ever looking at
the test label.

| split | global threshold | calibrated | delta |
|---|---|---|---|
| LODO, with rain (3 seeds) | 0.0939 ± 0.0161 | 0.1093 ± 0.0111 | **+0.0154** |
| LOSO headline (3 seeds) | 0.2884 ± 0.0066 | 0.2616 ± 0.0019 | −0.0268 |
| LOSO, increment-trained (3 seeds) | 0.2444 ± 0.0064 | 0.2338 ± 0.0062 | −0.0106 |
| flood holdout (3 seeds) | 0.2211 ± 0.0064 | 0.1410 ± 0.0013 | −0.0801 |

The gain is real only where the domain is unseen, and the losses elsewhere are much larger than
it. That is coherent — a threshold fitted on *other cities* is miscalibrated for an unseen one,
while on a seen city the swept threshold is already right and forcing the wet fraction to a prior
can only move it away — but +0.0154 is about one seed sd, so it stays a suggestion. On the flood
split it is actively harmful (−0.0801), because the held-out storms are the two largest and their
wet fraction is nothing like the training prior.

---

## 8. The dataset had no builder, and now it does

Everything above was computed from an 8 MB `varuna_stack.npz` and a `rain_features.json` that
were produced in a scratch notebook cell and committed as binaries with no code behind them.
That made two things impossible: adding a domain — the highest-value experiment left — and
honouring the paper's claim that every headline number is regenerable from a committed artifact.

`build_stack.py` and `build_rain.py` close it. Both have a `--verify` mode that compares against
the committed files rather than asserting equality, because parts of the originals could not be
reproduced and pretending otherwise would be the wrong kind of tidy.

### 8.1 What reproduces, and what does not

| | result |
|---|---|
| SAR truth (the labels) | **identical**, all three areas |
| feature channels | **21 of 23** reproduce to float16 precision |
| `hand` | differs on mumbai_harbour only: 27 cells of 262 144, a percentile tie-break |
| `twi` | **differs everywhere** — see below |
| `rain_Nd` totals | **identical**, once the conventions were recovered |
| `peak_*h` | 117 of 120; the 3 misses are all mumbai_northeast 2026-08-01 |

`twi`'s committed values are not reproduced by `varuna/build/baselines.py`'s own formula, nor by
that formula on a filled DEM, a Horn/Sobel slope, a 60 m slope, or the full-AOI raster. Inverting
it gives a flow-accumulation multiplier of ~950 with scatter rather than a clean 900. The cell
that wrote it is gone, so the builder uses the project's own documented definition and says so.

Two conventions in the rain file had to be recovered by search, and both are the kind of thing
that silently poisons a comparison:

- **UTC, not IST.** The same date at the same point totals 30.0 mm in UTC and 48.5 mm in
  Asia/Kolkata (Patna, 2023-08-09) — 5.5 hours of a different day swap in.
- **`rain_Nd` is N calendar days *ending on* the overpass.** Open-Meteo's archive range is
  inclusive at both ends, so the natural `d - N` fetches N+1 days.
- `peak_*h` is the heaviest burst in the **7 days** ending on the overpass, not within the day.
  3-day and 5-day windows fit some dates; only 7 fits 117 of 120.

**A live inconsistency this surfaced:** `varuna/build/calibrate.py:_rain_for_date` uses IST *and*
the off-by-one, so the twin's "2-day antecedent rain" is three IST days, while csi_net's
`rain_2d` is two UTC days. Neither is wrong; they have never been the same number. It is left as
a stated decision rather than a silent fix, because changing it moves the twin's calibration —
and §2's regression check shows the gap is small for the twin anyway: re-deriving the committed
8-date table with the *cached UTC* rain lands at 0.0409 against the recorded 0.0412.

### 8.2 The difference does not move any result

One channel of 23, standardised before the net sees it, should not matter. Should-not is not a
measurement. The headline was re-run on a fully rebuilt stack, 3 seeds:

| stack | CSI | per-seed |
|---|---|---|
| committed | 0.2884 ± 0.0066 | 0.2795 / 0.2954 / 0.2903 |
| **rebuilt** | **0.2885 ± 0.0082** | 0.2772 / 0.2961 / 0.2923 |

**+0.0001, or 0.02 seed-sd.** Leave-one-domain-out agrees too (0.1223 → 0.1214, seed 0). The
builder is validated for adding new domains.

One rule comes with it: **rebuild every area or none.** A stack with old `twi` in two cities and
new `twi` in a third would have leave-one-domain-out comparing domains built two different ways,
which is exactly the failure mode this whole document exists to avoid.

### 8.3 Getting more cities

`scripts/fetch_sar_masks.py` is the remaining piece: Sentinel-1 monsoon passes over an area's
AOI, deduplicated to one per 10 days, ranked by 3-day antecedent rainfall, top N downloaded.
`--dry-run` imports no Earth Engine at all and ranks candidates from the rainfall archive, so the
selection is reviewable before any quota is spent.

**`--min-wet` did not do what this section claimed it did, and the proof is the scene it was
named after.** It measured the wet fraction of the *downloaded* raster. `csi_net.build_stack`
only ever reads `build_domain`'s crop window, and `patna/2024-07-07` is `0.0065` wet across the
full AOI while being exactly `0.0000` wet inside that window:

    full AOI  (521, 1040)   wet 0.006494   ->  old guard: KEEP
    crop      (256,  256)   wet 0.000000   ->  new guard: DROP

So the guard written specifically to reject `patna/2024-07-07` would have accepted
`patna/2024-07-07`. It now crops before measuring, and falls back to the full-AOI test only for
an area with no bundle yet, saying so when it does. `build_stack` additionally names any scene
with zero wet cells rather than letting it appear only as the low end of a printed range, which
is how this one hid for months.

---

## 9. Data integrity: what was excluded and why

- **`waterlogging_frequency.tif` — excluded, label leakage.** `varuna/build/validate.py:83`
  builds it by counting wet pixels across the same Sentinel-1 passes that produce our labels.
  As a feature it would have manufactured a large fake CSI.
- **`rsi.tif` — excluded.** Derived from `gw_levels.csv`, Patna's six invented wells copied
  byte-identically into every bundle.
- **`catchment_labels.tif` — excluded.** Integer IDs; a memorisation key, not a feature.
- **`patna/2024-07-07` — dropped from scoring.** Zero observed wet cells, so no ground truth,
  and a forced 0.0 in every method's mean including the twin's old 0.0410.

**The invented-wells audit is closed, and it is good news.** `gw_levels.csv` reaches only one
product: `gw_depth` → `rsi.tif` → the `rsi` / `recharge_score` *ranking* columns in
`recharge_sites.csv`. The V3 recharge headline — metered m³ into the aquifer — comes from
`varuna/serve/recharge_sites.py:193-195`, whose suitability layer is WorldCover perviousness
modulated by Cosby Ksat from SoilGrids sand/clay (`_suitability`, lines 83-99), and whose volumes
come from the twin's metered infiltration. Neither touches `gw_levels.csv`. The recharge
*volumes* are clean; only the site *ordering* inherits the placeholder wells, and that path
already carries a loud gate (`build/recharge.py:105-124`).

The 23 retained channels are in `data.py`. Three deserve mention: `dem_rel_s2/s8/s32` are
elevation minus its own Gaussian blur at 2/8/32 cells. DEM error is spatially correlated, so
*relative* micro-relief survives a bias that destroys absolute elevation — the ±1 m vs 20 cm
problem, attacked at the feature level.

---

## 10. What this means for the paper

The paper's §4 currently claims *topographic routing does not co-locate flat-city flooding*,
supported by four methods sitting in a 0.032–0.051 band. Both halves need to change.

1. **CSI on this task has a floor and a ceiling that were never measured.** All-wet is
   0.013–0.047 and a rainfall-free climatology is 0.30–0.58. Any CSI reported without those
   beside it is uninterpretable — including the 0.041, and including our own 0.2884.
2. **Terrain does carry the co-location signal; the twin was not extracting it.** A learned model
   on the same rasters reaches 5.4× the twin on identical scenes. "No topographic method can
   co-locate" is refuted; "this physics model does not" is what the evidence supports.
3. **But the learned model ties a model-free climatology** (0.2884 vs 0.2881) where the city is
   known. Its real contribution is transfer: on an unseen city it scores 0.094 where climatology
   cannot be computed at all. The paper should lead with transfer, not with the multiple.
4. **The twin's published 0.0410 understated it.** It is 0.0536 on a fair protocol. Correcting a
   number in the direction that weakens our own claim is worth stating plainly.
5. **Rainfall from free reanalysis carries no usable information about where water goes at 60 m**,
   and degrades the model where the domain is seen — shown four ways, with two alternative
   explanations refuted. On unseen cities it is simply inert. The source is IFS 9 km, not ERA5.
6. **Neither scale nor resolution is the lever.** Flat from 2.0 M to 70.6 M parameters, and flat
   from 60 m to 30 m on the skill multiple.

## 11. Reproducing

```bash
python -m csi_net.build_stack --verify        # rebuild the rasters, compare to the committed npz
python -m csi_net.build_rain  --verify        # same for the rainfall vectors
python -m csi_net.run --split loso  --ablate rain --areas patna,mumbai_northeast --width 32
python -m csi_net.run --split lodo  --ablate rain --width 32 --steps 1500
python -m csi_net.run --split loso  --ablate rain --target increment --areas patna,mumbai_northeast
python -m csi_net.twin_protocol --areas patna,mumbai_northeast    # CPU; sims cached after run 1
python -m csi_net.tide_probe                                      # CPU, seconds
```

Runs on one T4 in minutes. `--ablate {none,rain,persist,terrain,tide}`, `--target {mask,increment}`,
`--shuffle_rain N`, `--areas a,b`, `--grid {30,60,120}`, `--tide 1`, `--width/--depth`.

## 12. Open

- **Coastal transfer is the open problem now, and it is a data problem.** Held out entirely,
  Mumbai scores 0.0333 — 1.8× all-wet, essentially nothing. Every non-Mumbai domain is inland,
  so the model has never seen a tidal flat or a mangrove. The test is a second coastal city
  (Chennai, Kochi, Surat) held out against a training set containing Mumbai: if it recovers, the
  limit is coastal *coverage*; if it does not, tidal water is not predictable from terrain and
  the deployable claim is inland-only, permanently.
- **On unseen regions the model under-predicts, bias 0.53.** It finds roughly half the water that
  is there. Ranking degrades far less than calibration (0.161 against 0.303), so this is probably
  fixable with a per-domain threshold rule that does not need local SAR history — but the
  calibration experiment of §7.2 was run on the old 3-domain split and needs redoing at 14.
- **LOSO has not been re-measured at 14 domains.** The 0.2884 headline and the tie with the
  climatology are still 18 storms over 2 domains. Re-running is 139 folds per seed, which is
  hours rather than minutes, and would say whether the tie survives a tenfold larger dataset.
- Whether a *real* tide gauge (FES2014, or a tide table at the overpass hour) rescues Harbour is
  still untested. The calendar proxy is ruled out twice over — no correlation (p = 0.47) and a
  measurable loss when fed to the network (−0.0127 at ≈2–4 sd). This matters more now that
  Mumbai is six domains rather than two.
- The nightly loop **has now run** (`--areas patna --dry-run --skip-ee`, all stages reached). What
  it revealed is the real blocker: the reports dataset is empty, so the fine-tune and reward-gate
  half of the loop has nothing to learn from regardless of credentials.
