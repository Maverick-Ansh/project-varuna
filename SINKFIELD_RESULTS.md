# The inferred drainage-sink field — what a satellite can and cannot tell us about Mumbai's drains

**A per-cell drainage capacity, inferred from Sentinel-1 by gradient descent through the
differentiable twin. It improves flood *extent* on held-out storms while predicting *less* water —
and it does not improve point-depth ranking at all.**

Reproduce: `python scripts/run_sinkfield.py --tiles mumbai_east mumbai_west mumbai_north mumbai_south --min-rain 25 --test-dates 2025-06-24 2025-09-04 2026-07-20`
Artifacts: `artifacts/<tile>/sinkfield.pt` (fitted field), `artifacts/sinkfield_report.json` (all metrics),
`artifacts/mumbai_sar_baseline.json` (the baseline it is measured against).

---

## Why a field, and not more constants

Two independent validations said the same thing: the twin knows *how much* water, not *where*
(`VALIDATION.md`: SAR CSI 0.033–0.051 with FAR ≈ 0.95; `DEPTH_VALIDATION.md`: negative skill and
zero-to-negative correlation against 83 crowdsourced depths). The existing SAR calibration fits
~16 per-WorldCover-class Manning/infiltration multipliers — constants that change how *deep* water
gets, not where it goes, which is why that calibration was a documented near-null.

Meanwhile the largest known physics gap (G2) is that **no storm sewer is modelled at all**. The twin
over-floods precisely where working drains exist. So instead of more constants, we give the model
the missing object: a **per-cell drain capacity** (mm/h), positive by construction, restricted to
built land, regularised toward "no drains", and fitted through the simulator so that simulated wet
extent matches the radar. 65,536 parameters per tile instead of 16 — freedom to move water,
priors to keep it physical, held-out storms to keep it honest.

## Method in one paragraph

After each verbatim Bates (2010) step, water leaves a cell at rate `drain` (bounded by what is
present) — a sewer, not infiltration, so it never fills the soil column or counts as recharge. The
field is `softplus(θ) × max_rate × built`, fitted with Adam through the checkpointed simulator on
8 training storms, held out on 3. The loss is deliberately **all volume fractions of the storm's
rain**, so the terms are commensurate: soft-Dice against the SAR mask, plus hinges that push
predicted depth down on SAR-dry cells and up on SAR-wet ones, plus a **minimum-outflow prior** that
charges for every m³ drained.

Three design decisions worth stating, because each one was a failure first:

1. **Soft-Dice alone cannot train this field.** `sigmoid((h−τ)/β)` saturates in deeply flooded
   cells, so the gradient vanishes exactly where the model is most wrong. The magnitude hinges have
   constant slope regardless of depth. (First Colab run: 60 iterations, no learning.)
2. **The loss terms must share units.** With arbitrary weights the fit either drains everything
   (6× more outflow than truth in a synthetic recovery test) or nothing at all.
3. **The minimum-outflow prior makes the volume interpretable.** Fitting picks the *smallest*
   drainage that explains the observed extent, so **every outflow number below is a lower bound**,
   not an estimate.

Non-finite losses *and* non-finite gradients are skipped (a finite loss can still produce a NaN
adjoint through 1,440 near-CFL timesteps; without the gradient guard, `clip_grad_norm_` writes NaN
into θ and the fit is dead for good — this killed one full run).

## Result 1 — held-out storm extent improves, and gets there by predicting less water

All four tiles, mean CSI over the 3 held-out storm dates, with the wetness that bought it:

| tile | baseline CSI | sink-field CSI | change | predicted wet cells (held-out) |
|---|---|---|---|---|
| mumbai_east | 0.0442 | **0.0478** | +8% | 19,691 → **17,454** (−11%) |
| mumbai_west | 0.0354 | **0.0361** | +2% | 16,008 → **14,792** (−8%) |
| mumbai_north | 0.0460 | **0.0472** | +3% | 21,974 → **21,493** (−2%) |
| mumbai_south | 0.0485 | **0.0605** | +25% | 7,163 → **5,929** (−17%) |
| **mean** | **0.0435** | **0.0479** | **+10%** | **−10%** |

`DEPTH_VALIDATION.md` established that **CSI can be bought by predicting more water**. This result
is the opposite trade: every tile improves while getting *drier*. That is the only reason these
small numbers are worth reporting at all — and it is why the table above quotes wetness beside
every CSI, as that document requires.

## Result 2 — the controls ladder: it is the *field*, not the extra freedom

Same loss, same storms, same held-out dates on `mumbai_east`; only the parameter count changes.

| model | parameters | train CSI | held-out CSI | held-out wet cells |
|---|---|---|---|---|
| textbook physics | 0 | 0.0428 | 0.0442 | 19,691 |
| one uniform drain rate (1.06 mm/h) | 1 | 0.0426 | 0.0439 | 19,433 |
| per-WorldCover-class Manning/infiltration | 16 | 0.0354 | **0.0402** | 18,612 |
| **per-cell sink field** | **65,536** | **0.0488** | **0.0478** | **17,454** |

A single city-wide drain rate does nothing (0.0442 → 0.0439). The 16-parameter physics calibration
makes held-out extent **worse** (0.0402) — it fits the training storms by turning every multiplier
up ~1.8×, which is what "change depth, not location" looks like when you let it try harder. Only
the spatial field improves held-out extent. **The gain comes from spatial structure, not from
parameter count** — which is the claim the ladder exists to test.

## Result 3 — the decisive test: no transfer to point depths

The 83 quarantined crowdsourced depth reports, replayed with identical machinery on both arms
(same observations, same cells, same rainfall forcing; only the drain field differs):

| | skill vs best baseline | correlation | correlation (wet sites) | RMSE |
|---|---|---|---|---|
| textbook twin | −0.918 | +0.037 | −0.065 | 0.356 m |
| **with sink field** | −0.921 | +0.032 | −0.063 | 0.356 m |

**Nothing moves.** Differences are in the fourth decimal — far below meaningful for n = 83.

*Caveat on these absolute numbers*: this replay drives the **raw simulator** on both arms, whereas
the published depth validation used the **U-Net emulator** (skill −0.46, wet-site correlation
−0.23). The A/B is therefore valid — both arms are identical except for the field — but these
numbers are not comparable to `DEPTH_VALIDATION.md`'s. Incidentally, the emulator scores *better*
than the simulator it was trained on, which is worth a look someday: smoothing appears to help on
point comparisons.

**Interpretation.** Learning where water *leaves* the surface is not the same as learning where it
*goes*. The field is fitted on radar extent — a wet/dry mask — which constrains where drainage must
exist but says almost nothing about relative depth between two wet streets. So it improves the
quantity it was trained on and leaves the ranking failure untouched. Co-location remains the open
problem after three independent attacks (physics calibration, resolution analysis, drainage
inversion), and the next honest move is the one the depth validation already pointed at: **finer
grid** (the measured 4.45× dilution implies ~28 m), not more parameters at 60 m.

## Result 4 — what the city pours out

Integrating the drained volume over each storm answers a question the project could not previously
ask. Because the prior selects the smallest drainage consistent with the radar, these are **lower
bounds**.

Summed over the four tiles, per real storm: **2.6 – 4.5 million m³**, tracking storm size
(2025-09-04, 21 mm: 2.64 Mm³ · 2026-07-08, 213 mm: 4.48 Mm³).

On the complete six-tile city domain, with design storms:

| storm | inferred surface outflow | flooded land | of which built |
|---|---|---|---|
| 25 mm | ≥ 3.48 million m³ | — | — |
| 50 mm | ≥ 4.63 million m³ | 70.7 km² | 20.2 km² |
| 100 mm | ≥ 5.21 million m³ | 151.1 km² | 44.8 km² |
| 200 mm | ≥ 5.41 million m³ | 241.5 km² | 76.7 km² |

**The outflow saturates.** Eight times the rain (25 → 200 mm) buys 56% more drained water, and the
last doubling (100 → 200 mm) buys 4%. The field is *rate*-limited: once streets are wet for the
whole storm, extra rainfall cannot leave any faster — it ponds. That is a concrete, falsifiable
statement about Mumbai's drainage ceiling (~5.4 million m³ per storm, however hard it rains), and
the obvious next step is to check it against BMC/BRIMSTOWAD published pumping capacities.

Note how differently the two columns behave: drained volume flattens while flooded area keeps
climbing. That *is* the mechanism — beyond the drainage ceiling, additional rain has nowhere to go
but outward across the ground.

Inferred capacities are physically plausible for urban drainage: mean 0.3 – 3.0 mm/h over built
land, peaking at 4 – 50 mm/h.

## Result 5 — one Mumbai, complete

The tiles turned out to be a **2 × 3 grid with two cells empty**: the eastern column had no
northern row (Mulund, Nahur, the Bhandup pumping station, Thane creek) and no southern row
(Trombay, Mahul, Mankhurd, Sewri, Vashi creek). Nothing announced this — the mosaic simply filled
both with sea, which is why the four-tile city was only 70.9% real coverage. Both are now built
(`mumbai_northeast`, `mumbai_harbour`; emulator val RMSE 22.9 cm and 9.0 cm) and a test asserts
every grid cell has exactly one owner.

The six tiles mosaic onto one grid — **817 × 586 at 60 m, coverage 1.00**, of which **950 km² is
land** and 336 km² built — and a storm runs on the whole city in **under 3 seconds** on a T4, with
water crossing the former seams instead of hitting six closed boundaries.

Seam audit (100 mm storm, city domain vs each individual tile, land cells only):

| tile | mean Δh in the 600 m edge band | mean Δh interior | ratio | p99 Δh |
|---|---|---|---|---|
| mumbai_south | 4.67 cm | 3.23 cm | 1.4× | 35 cm |
| mumbai_west | 5.55 cm | 4.00 cm | 1.4× | 48 cm |
| mumbai_east | 5.73 cm | 3.66 cm | 1.6× | 59 cm |
| mumbai_north | 7.51 cm | 4.05 cm | 1.9× | 69 cm |
| mumbai_northeast | 10.85 cm | 4.56 cm | **2.4×** | 106 cm |
| mumbai_harbour | 7.15 cm | 2.91 cm | 2.5× | 54 cm |

Depths near former boundaries disagree with the tiled model **40–150% more than in the interior** —
the measured cost of closed tile boundaries. The two new creek tiles are the worst affected, which
is what you would expect: they are where the water was trying to leave.

Honest caveat: the interior disagreement is not zero either (2.9–4.6 cm), because mosaicking
re-samples the DEM at a slightly different sub-pixel phase (mean |Δz| 0.44–2.34 m between the two
croppings, worst on northeast). The seam signal is the *difference* between edge and interior, not
the absolute number.

## What this changes

1. **G2 (no storm sewer) is now modelled, not just admitted** — and the inferred drainage is
   testable against published municipal capacity.
2. **The city has an outflow number with a ceiling** — the saturation at ~5.5 Mm³ is the headline.
3. **Co-location survives a third attack.** Resolution, not parameterisation, is the remaining
   lever; the 5 m street twin has a target (~28 m) and now also a null result telling it what
   *won't* work.
4. **The tile join is real** and the seam cost is measured, opening the one-city workstream in
   `V4_PLAN.md`.

## Limits

- **n = 11 storms, one city, one season-and-a-bit.** Held-out CSI differences of +0.004 on three
  dates are small; the consistent sign across four tiles and the wetness direction are what carry
  the claim, not any single number.
- **The field is not a map of pipes.** It is the effective sink needed to reconcile a 60 m
  topographic model with radar. Real drains are sub-grid; a cell's value bundles pipes, pumping,
  and infiltration the twin underestimates.
- **Radar sees extent, not depth**, so nothing here constrains the depth ranking (Result 3 is the
  direct evidence).
- **The creek tiles are undertrained, for a numerical reason.** An explicit Bates step of 10 s
  violates CFL where a tidal creek makes the water deep, the storm returns NaN, and that iteration
  is thrown away: west lost 17 of 40, north and northeast 25 of 40. Their fields are correspondingly
  weak (northeast mean 0.33 mm/h against east's 2.98). `fit` now re-runs an unstable storm at a
  halved timestep before giving up, so a re-fit of those tiles should recover the lost training —
  the fields reported here predate that fix.
- SAR wet masks over a monsoon megacity include wet roofs, mudflats and creek margins the JRC mask
  does not remove.
