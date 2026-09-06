# Next steps

Ranked by evidence gained per hour, not by how interesting they are. Numbers referenced here
are in `RESULTS.md`; every run's JSON is in `results/`.

---

## Closed on 2026-09-06

| was | outcome |
|---|---|
| §4 more domains — *the highest-value experiment left* | **Done.** One Earth Engine login, 110 Sentinel-1 masks across 11 areas, 3 domains → **14**. All 110 cleared the wet floor; none dropped. |
| the transfer claim rests on 3 cities | **Done, and the answer needed a control.** Per-tile LODO 0.0936 → **0.2698**. But per-tile stopped meaning "unseen city": leave-one-**region**-out halves it to **0.1305 ± 0.0046**, which is the honest number — still 1.4× the old floor at a fifth of the seed spread. |
| is the gain the rebuilt stack? | **No.** Same 3 domains on `varuna_stack14.npz`: 0.0960 ± 0.0090 against the committed 0.0936 ± 0.0251. The gain is domains. |
| §1 read §4 of the paper cold | **Done, and it found three defects.** Table 1's Patna row averaged in `2024-07-07` eleven lines below the paragraph saying that date is dropped; §4 used two different climatologies (Table 1 swept a per-domain threshold on the scenes it reported, Table 2 fixed 0.5) while the caption described the fixed one; and "no parameters" was false. All corrected. |
| §2 pick a venue | **Researched; one decision left.** The CCAI/NeurIPS 2026 deadline (abstract 22 Aug, full 29 Aug) **has passed**. Recommendation: *Environmental Data Science* (Cambridge) — rolling submission, Diamond OA, and it awards Open Practice Badges for exactly the committed-artifact reproducibility this project is built around. |
| `--min-wet` protects against empty masks | **It did not.** It measured the *downloaded* raster; `build_stack` reads only `build_domain`'s crop. `patna/2024-07-07` is 0.0065 wet full-AOI and **0.0000** cropped, so the guard would have accepted the very scene it was written to reject. Fixed, and `build_stack` now names any zero-wet scene. |
| `trivial_baselines.json` had no builder | **Fixed.** `build_trivial.py` reproduces all 12 committed values to 4 decimals, then corrects them. Extended to 14 domains: climatology 0.29–0.67, mean 0.55. |
| BMC / BRIMSTOWAD capacity lookup | **Done — the project's first external anchor.** The design *rate* (25/50 mm h⁻¹) is the wrong quantity. BMC's *measured* 16.45 Mm³ pumped 16–19 Aug 2025 through six stations (≈4.7 Mm³/day) sits just below our ≥5.41 Mm³ per-storm surface-outflow ceiling — the correct side, since ours counts 186 mostly gravity-fed outfalls. See `SINKFIELD_RESULTS.md`. |
| the nightly loop has never run | **It has now**, `--areas patna --dry-run --skip-ee`, every stage reached. It had been blocked by its own token check: `--dry-run` pushes nothing and both HF repos are public, but `main()` exited on a missing `HF_TOKEN` before reaching any of that. |

---

## Tier 1 — the paper cannot be submitted without these

### 1. ~~Decide the repo's visibility~~ — **done, the repo is public**
`github.com/Maverick-Ansh/project-varuna` went public on 2026-09-06 after a credential scan of
all 178 commits of history came back clean. The paper's reproducibility claim now holds.

### 2. Submit
The venue question is answered unless you disagree with it. What remains is one cold read of the
rewritten §4 (it now argues against two of the project's own earlier positions) and the
submission itself.

---

## Tier 2 — the open research question

### 3. ~~Coastal transfer~~ — **answered: the limit is coverage** (see RESULTS §4.4)
Chennai was built as the 15th domain and settles it. Mumbai, now with a coastal city in
training, goes **0.0333 → 0.0613 (+84 %, 1.8× → 3.3× all-wet)**; Chennai, scored with Mumbai in
training, reaches **0.1642 at 7.8×**, close to inland Karnataka's 10.6×. Tidal water *is*
learnable from terrain — the model has to have seen a coast. Every fold gained the same 10
storms and only the coastal region moved, so it is not data volume.

**What is left of it:** Mumbai at 3.3× still trails inland performance, so one coastal sibling
narrows the gap without closing it, and Chennai's own figure rests on 10 scenes. A third coastal
city (Kochi, Surat, Visakhapatnam) would say whether 3.3× keeps climbing with coastal coverage
or plateaus — the difference between "add coasts" and "coasts are simply harder". Same recipe:
one bundle build plus ~10 masks, about 25 minutes of GPU.

**Mind the pass window.** `SEASON` in `fetch_sar_masks.py` must be set for any area whose rain
does not arrive in Jun–Oct, and `list_passes` now takes the months through to the Earth Engine
query. Kochi is southwest-monsoon (the default is fine); Visakhapatnam and anything on the
Coromandel coast needs `(6, 12)` like Chennai.

### 4. The under-prediction on unseen regions
**~30 min GPU.** Bias falls to **0.53** on leave-one-region-out: the model finds about half the
water present, i.e. it under-warns, which is the wrong direction for a flood product. Ranking
degrades far less (0.161 vs 0.303), so this looks like calibration rather than skill. The
threshold-calibration experiment of §7.2 was run on the old 3-domain split and should be redone
on `loro` × 3 seeds — it is the one lever with a plausible mechanism behind it.

### 5. Re-measure LOSO at 14 domains
**Hours, not minutes** (139 folds per seed). The 0.2884 headline and the tie with the climatology
are still 18 storms over 2 domains, which is now the smallest evidence base in the paper. Worth
knowing whether the tie survives a tenfold larger dataset.

### 6. A real tide gauge for Harbour
**~2 h.** Only the calendar proxy is ruled out (R² 0.405, permutation p = 0.47, and −0.0127 CSI
when fed to the net). FES2014 or a tide table at the overpass hour is the version with amplitude
in it. This matters more now that Mumbai is six domains rather than two, and it is plausibly the
same question as §3.

---

## Tier 3 — carried over

- **The nightly loop's real blocker is not a secret, it is data.** The reports dataset is empty,
  so the fine-tune and reward-gate half has nothing to learn from even with `HF_TOKEN` attached.
  Running it for five nights would produce five nights of `0 model update(s)`. Either drive
  citizen reports to the deployed Space first, or scope §6 of the paper to the refresh half of
  the loop, which does work.
- **The twin and csi_net still use different rainfall windows.** `calibrate._rain_for_date` uses
  IST and Open-Meteo's inclusive range, so the twin's "2-day antecedent rain" is three IST days
  while csi_net's `rain_2d` is two UTC days. Measurably small for the twin (0.0412 → 0.0409); a
  decision about which convention to standardise on, not a fire.
- **ERA5 vs IFS is a decision, not a bug.** `varuna/serve/weather.py:ARCHIVE_MODEL`. Pinning IFS
  preserved every existing number; ERA5 costs a full recalibration and moves rainfall by 2.19×.

---

## What the paper's spine now is

> Flood extent at 60 m is predictable from terrain alone — but so is a climatology, and on a city
> with a radar archive the two score the same. What a learned model adds is that it runs where the
> climatology cannot be computed at all. That transfer is worth 0.131 CSI, 6.1× all-wet, on a
> region with no sibling in training — and it is **not uniform**: 10.6× inland, but 1.8× on a
> coastal city when no coast is in training, rising to 3.3× once one is. Tidal water is
> learnable; the model has to have seen a coast. Rainfall, as available from free reanalysis,
> contributes nothing at any grouping level. And CSI on this task must be reported against
> trivial baselines, because a rainfall-free climatology scores 0.29–0.67.

Five things carry it, all measured rather than argued:

1. **What CSI is worth here** — all-wet 0.006–0.054, climatology 0.29–0.67, SAR-vs-SAR ceiling
   0.21–0.59 over 14 domains. A methodological contribution that applies to other people's papers.
   In nine of fourteen domains the climatology beats the agreement between two radar passes of the
   same city, which caps what any model can be asked to do here.
2. **Terrain is learnable and the twin was leaving it on the table** — 5.4× on identical scenes,
   with the twin's own number corrected *upward* in the process.
3. **Transfer is real, smaller than it first looks, and bounded** — 0.0936 → 0.2698 per tile, but
   0.1305 per region, with the rebuilt-stack control ruling out the alternative explanation.
   The one place it nearly fails is a coast the model has not seen, and that is a coverage
   problem: one coastal city in training raises held-out coastal skill by 84 %.
4. **Rainfall is the wrong lever**, shown four independent ways, with two alternative explanations
   tested and refuted, and now replicated three times: at 14 domains per tile, per region, and
   again at 15 domains across four regions.
5. **Neither scale nor resolution is the lever either** — flat from 2.0 M to 70.6 M parameters,
   flat from 60 m to 30 m.

The word "honest" still appears zero times. Every weakness above is stated as a number.
