# Next steps

Ranked by evidence gained per hour, not by how interesting they are. Numbers referenced here
are in `RESULTS.md`; every run's JSON is in `results/`.

---

## Closed on 2026-09-05

Kept, briefly, because what a test *ruled out* is as much a result as what it found.

| was | outcome |
|---|---|
| §1 rerun the twin on our protocol | **Done.** Twin 0.0536, not 0.0410 — it was *understated*. Multiple is 5.4×, not 7.1×. The scoring path reproduces the committed 8-date table to ±0.0007, and that check caught a real threshold-grid bug in our own script. |
| §2 LODO terrain-only, 3 seeds | **Done.** 0.0936 ± 0.0205. The prediction that ablating rain would *improve* LODO was wrong: with rain is 0.0939 ± 0.0161. Identical. |
| §3 finish the 30 m test | **Done.** 30 m 10.4× all-wet vs 60 m 9.7×. Resolution is not the lever. |
| §4 decide what Harbour is for | **Done, and the tidal reading was tested rather than asserted.** Calendar tide phase explains nothing in Harbour (R² 0.405, permutation p = 0.47) while landlocked Patna "fits" at 0.985. Excluded, with a measured reason: 0.0000 storm increment on 8 of 11 dates. |
| §5 fix the rainfall source | **Done.** `best_match` = `ecmwf_ifs` to the 0.1 mm (554.2 = 554.2); ERA5 would be 2.19× higher. Pinned `models=ecmwf_ifs`, which changes no existing number, and corrected the prose in 6 files. |
| §6 train on the storm increment | **Done. Null.** 0.1757 ± 0.0038 vs 0.1723 ± 0.0054 trained on the full mask. Training on the target you are graded on does not make the increment predictable. |
| §8 threshold calibration | **Partly.** Implemented and reported on every run. +0.0154 on LODO, −0.0106 on LOSO — one seed-sd, so kept as a suggestion, not a claim. |
| audit `gw_levels.csv` reach | **Done, clean.** The V3 recharge *volumes* come from WorldCover perviousness × Cosby Ksat and the twin's metered infiltration; `gw_levels.csv` reaches only the site *ordering*, which already carries a gate. |

The one result none of this was looking for: **the net ties the climatology baseline** (0.2895 vs
0.2881) on the split where the city is known. The headline is now transfer, not accuracy.

---

## Tier 1 — the paper cannot be submitted without these

### 1. Rewrite the paper's §4
**~3 h, no compute.** `paper/varuna-floodtwin.tex:174-215` claims *topographic routing does not
co-locate flat-city flooding*, evidenced by four methods in a 0.032–0.051 band. That claim is now
refuted by our own data: the same rasters, learned, reach 0.2895. The section needs:

- Table 3 replaced by the like-for-like table (`RESULTS.md` §2.1), which also *raises* the twin
  from 0.0410 to 0.0536 — a correction against our own headline, and it should be said as such.
- The trivial baselines added beside every CSI in the paper. All-wet 0.013–0.047 and climatology
  0.30–0.58 are the floor and the ceiling, and no CSI in this literature is interpretable
  without them. This is the part that generalises past this paper.
- The claim restated: co-location *is* learnable from terrain; this physics model was not
  extracting it; and the learned model's contribution is transfer to cities with no SAR archive,
  not accuracy on cities with one.
- §4's "why" (DEM uncertainty) survives as an explanation of why *the twin* fails, not of why the
  task is impossible.

### 2. Pick a venue
**~1 h.** The paper has been "done, not submitted" since 2026-07-12. Nothing below improves it
more than choosing a deadline does. The header still targets a CCAI/EGU-style workshop; the
learned-baseline result would also fit a remote-sensing venue, and the methodological point
(report CSI against trivial baselines) is the most citable thing here.

### 3. Decide the repo's visibility
**~10 min.** The paper claims reproducibility on free hardware while the repo is private. Make it
public before submission, or drop the claim. This is the user's call, not a code change.

---

## Tier 2 — strengthens the claim

### 4. More domains
**GEE job on the existing pipeline.** Three domains is a thin basis for a cross-city claim, and
the deployable claim now rests entirely on LODO (0.0936 ± 0.0205, n=3 cities). The V3 build
serves 14 areas; SAR masks exist for 3. Fetching `observed_water_*.tif` for even three more
would roughly double the evidence behind the only number that matters for deployment. **This is
the highest-value experiment left in the project.**

### 5. A real tide gauge for Harbour
**~2 h.** Only the calendar proxy has been ruled out. FES2014, or a tide table sampled at the
Sentinel-1 overpass time, is the version with amplitude in it. If Harbour becomes usable, it is a
fourth domain and a genuinely interesting result; if it does not, the exclusion is settled twice
over. Worth doing only after §4 — a new city is cheaper evidence than a rescued one.

### 6. Threshold calibration across more splits
**~20 min GPU.** One seed-sd on one split. Run it on LOSO/flood/LODO × 3 seeds with and without,
and either promote it to a claim or drop it.

---

## Tier 3 — carried over, unchanged by this work

- **The nightly loop has still never run.** Either run it for 3–5 nights and get a real log, or
  cut §6 of the paper and keep it as one line of future work. `scripts/nightly_update.py
  --dry-run` on Kaggle (where `HF_TOKEN` is a secret) is the zero-risk first step — it snapshots
  the Space and pushes nothing.
- **BMC / BRIMSTOWAD capacity lookup.** A literature lookup, not a compute job, and still the
  best value-per-hour item available: it could externally validate the 5.3 Mm³ drainage
  saturation ceiling, which would be the only externally-validated number in the project.
- **ERA5 vs IFS is a decision, not a bug, and it can be flipped in one line.**
  `varuna/serve/weather.py:ARCHIVE_MODEL`. Pinning IFS preserved every existing number; switching
  to ERA5 costs a full recalibration and moves rainfall by 2.19×. Stated either way, as required.

---

## What the paper's spine should now be

> Flood extent at 60 m is predictable from terrain alone — but so is a climatology, and the two
> score the same. What a learned model adds is that it runs on a city with no radar archive,
> where the climatology cannot be computed. Rainfall, as available from free reanalysis,
> contributes nothing and measurably degrades the model where the city is already known. And CSI
> on this task must be reported against trivial baselines, because a rainfall-free climatology
> scores 0.58 — higher than anything a physics model in this literature reports.

Four things carry that, and all four are measured rather than argued:

1. **What CSI is worth here** — all-wet 0.013–0.047, climatology 0.30–0.58, SAR-vs-SAR ceiling
   0.16–0.46. A methodological contribution that applies to other people's flood papers.
2. **Terrain is learnable and the twin was leaving it on the table** — 5.4× on identical scenes,
   with the twin's own number corrected *upward* in the process.
3. **Rainfall is the wrong lever**, shown four independent ways, with two alternative explanations
   tested and refuted, and now with a stated boundary: the effect vanishes on unseen cities.
4. **Neither scale nor resolution is the lever either** — flat from 2.0 M to 70.6 M parameters,
   flat from 60 m to 30 m.

The word "honest" still appears zero times. Every weakness above is stated as a number.
