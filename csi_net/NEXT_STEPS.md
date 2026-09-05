# Next steps

Ranked by evidence gained per hour, not by how interesting they are. Numbers referenced here
are in `RESULTS.md`.

---

## Tier 1 — do these before anything goes in the paper

### 1. Rerun the twin on our protocol, so the multiple is honest
**~1 h, no GPU.** The headline says the net reaches 0.2895 and the twin reaches 0.041. Those are
measured on **different test sets**: the twin's number is 8 Patna dates from the old table
(including the dead 2024-07-07 scene), ours is 18 storms across Patna + Mumbai-NE under
leave-one-storm-out. The comparisons against `all-wet` (11×) and `random` (22×) *are*
like-for-like, because those are computed per-scene inside the same runs — the twin comparison
is not.

Run `score_twin` over the same 18 storms with the same mask and the same dead-scene exclusion,
and report whatever comes out. If the twin improves once 2024-07-07 is dropped, that is a
correction to the project's own record and should be stated plainly.

**Until this is done, quote 11× over all-wet, not 7.1× over the twin.**

### 2. Leave-one-domain-out, terrain-only, 3 seeds
**~25 min GPU.** The LODO 0.099 is the single most important number for deployment — it is the
only one measured on a city the model has never seen, and it is the regime where climatology is
unavailable because there is no local SAR history. But it was run **with rainfall included and
on a single seed**, and §4 shows rainfall costs ~0.09 CSI. That number is very likely a
substantial underestimate of its own method.

```bash
python -m csi_net.run --split lodo --ablate rain --width 32 --steps 1500 --seed 0   # 1,2
```

### 3. Finish the 30 m resolution test
**~15 min GPU.** Built and launched; the session ended mid-run so it produced nothing. This is
the direct test of the project's own headline measurement — 810 m² ponds, 28.5 m across, scored
on 60 m cells. The 60 m arm completed for comparison: CSI 0.2810 at all-wet 0.0291, a 9.7× skill
multiple.

```bash
python -m csi_net.run --split loso --ablate rain --areas patna,mumbai_northeast --grid 30 --seed 0
```

Compare **CSI ÷ all-wet on each grid**, never raw CSI across grids — the base rates differ, so
raw CSI is not comparable between 30 m and 60 m.

---

## Tier 2 — decisions the project has to make

### 4. Decide what Mumbai-Harbour is for
It is tidally contaminated: rainfall correlates **−0.70** with observed water at 14 days, its
storm increment is 0.0000 on 8 of 11 dates, and including it drags the terrain-only model from
0.206 down to 0.170. Three options, in order of preference:

- **Exclude it**, and say why — a tidal estuary is a different hazard class from urban pluvial
  flooding, which is a defensible scoping decision, not a convenience.
- **Add a tide covariate** (FES2014 or a tide table sampled at the Sentinel-1 overpass time) and
  test whether it becomes usable. This is the interesting version and it is a real experiment.
- Keep pooling it, and accept a model trained on two contradictory targets.

### 5. Fix the rainfall source, in code and in prose
`varuna/serve/weather.py:33` sends no `models=` parameter, so Open-Meteo returns `best_match` =
ECMWF IFS 9 km. The docstring at line 28, the Notion writeup, and the paper all say ERA5. They
disagree 4.3× (114.1 vs 485.7 mm on 2026-07-06). Pick one:

- pin `models=era5` to make the code match the claim, then re-verify the twin's calibration, or
- change the claim to IFS 9 km everywhere.

Either way the choice must be stated, because the spread is larger than any effect the physics
calibration produced.

### 6. Train on the storm increment directly
**~20 min GPU, untested.** The net is currently trained on the full SAR mask and *scored* on the
increment (0.172). Training against the increment target directly may do considerably better on
the quantity that actually matters — and it is the target the twin was implicitly built for, so
it is the fairest possible head-to-head.

---

## Tier 3 — strengthens the claim, costs more

### 7. More domains
Three domains is a thin basis for any cross-city claim, which is why LODO 0.099 should be read as
a floor rather than an estimate. The V3 build serves 14 areas; SAR masks exist for only 3.
Fetching `observed_water_*.tif` for more areas is a GEE job on the existing pipeline and would
materially strengthen the only result that speaks to generalisation.

### 8. Threshold calibration
Ranking quality (0.3219) sits consistently above achieved CSI (0.2895) across every run. The gap
is pure calibration loss — the model orders cells better than its single global threshold
exploits. A per-scene threshold rule (predicted wet fraction matched to a terrain-derived prior)
would recover part of it for free, with no retraining.

---

## Carried over from the pre-existing review, still open

These predate this work and are unchanged by it:

- **The nightly loop has still never run.** Either run it for 3–5 nights and get a real log, or
  cut §6 and keep it as one line of future work. Shipping the in-between state is the risk.
- **BMC / BRIMSTOWAD capacity lookup.** A literature lookup, not a compute job, and still the
  best value-per-hour item available: it could externally validate the 5.3 Mm³ drainage
  saturation ceiling, which would be the only externally-validated number in the project.
- **Repo is private** while the paper claims reproducibility on free hardware. Make it public
  before submission or drop the claim.
- **Audit which paper numbers touch `gw_levels.csv`** (Patna's six invented wells, copied
  byte-identically into every bundle). `csi_net` excludes `rsi.tif` for this reason, but the
  intervention numbers have not been audited.

---

## What the paper's spine should now be

The previous plan was *"at 60 m with satellite-only inputs, urban pluvial flood models cannot
co-locate."* That is no longer what the evidence says. Co-location **is** learnable from terrain —
0.2895 at bias 1.39, 51 % recall at 37 % precision, 11× a zero-knowledge baseline on identical
scenes. The revised claim is sharper and more useful:

> Flood extent at 60 m is predictable from terrain alone. Rainfall, as available from free
> reanalysis, contributes nothing and measurably degrades the model. And CSI on this task must be
> reported against trivial baselines, because a rainfall-free climatology scores 0.58 — higher
> than anything a physics model in this literature reports.

Three things carry that, and all three are measured rather than argued:

1. **What CSI is worth here** — all-wet 0.013–0.047, climatology 0.30–0.58, SAR-vs-SAR ceiling
   0.16–0.46. This is a methodological contribution that applies to other people's flood papers,
   not only this one.
2. **Rainfall is the wrong lever**, shown four independent ways with two alternative explanations
   tested and refuted. Negative results this well controlled are rare and citable.
3. **Scale is not the lever either** — flat from 2.0 M to 70.6 M parameters. Anyone repeating
   this should spend compute on inputs and protocol, not width.

The word "honest" still appears zero times. Every weakness above is stated as a number.
