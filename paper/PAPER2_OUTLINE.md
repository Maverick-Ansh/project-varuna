# Paper 2 — outline

**Decision: this is a measurement paper, not a system paper.** The existing 13-page
`varuna-floodtwin.tex` is the system paper (twin, planning, GNN routing, participatory layer,
LLM brief). Paper 2 takes only the `csi_net` evidence and makes one argument well in 6–7 pages.
Nothing here needs a new experiment; every number is already committed in `csi_net/results/`.

---

## The one claim

> Critical Success Index on Sentinel-1 flood extent is uninterpretable without trivial baselines.
> Supply them and two things follow: a learned model's value is **transfer, not accuracy**, and
> that transfer is bounded by whether the model has seen the target's *terrain type* — not by
> capacity, resolution, or rainfall.

Three contributions, in the order a reader needs them:

1. **A baseline scale for this task.** A one-threshold climatology scores 0.29–0.67 across 15
   domains — above any physics-model CSI we can find in this literature. Applies to other
   people's papers, which is what makes it citable.
2. **"Held out" is ambiguous and the ambiguity is worth 2×.** Leave-one-tile-out reads 0.270;
   leave-one-*region*-out on the same data reads 0.147.
3. **Transfer is bounded by terrain-type coverage, and that is fixable.** A coastal city with no
   coast in training scores 1.8× all-wet; add one coastal city and it goes to 3.3× (+84 %).

---

## Section plan (target 7 pp incl. references)

### §1 Introduction — 0.75 p
Frame: flood-extent models are reported with CSI, and CSI is compared against nothing. State the
three contributions above. One sentence that the code, 15 bundles, 150 SAR masks and every run's
JSON are public.

### §2 Data and protocol — 0.75 p
- 15 domains, 150 Sentinel-1 scenes: Patna (7), 6 Mumbai tiles (62), 7 Karnataka (70),
  Chennai (10). 60 m scoring grid, JRC ≥ 50 % permanent water masked **on both sides**.
- Thresholds always chosen on the other storms of the fold, never the reported scene.
- Date selection: ranked by 3-day antecedent rain from the pinned `ecmwf_ifs` archive.
- **Two things stated as protocol, not buried:** `patna/2024-07-07` has zero wet cells in the
  crop and is dropped; Chennai's window is Jun–Dec because it floods on the northeast monsoon.
- Model: 7.9 M-parameter U-Net, 23 terrain channels. One paragraph. It is not the contribution.

### §3 What a CSI is worth before any model exists — 1 p · **Table 1**
Table 1 = the 15-domain trivial-baseline table (all-wet / random / climatology / SAR-vs-SAR).
Numbers: all-wet 0.006–0.054, climatology **0.29–0.67 mean 0.55**, SAR-vs-SAR 0.21–0.59.
Two observations that carry the section:
- The climatology beats published physics-model CSIs on this target.
- In 9 of 15 domains it beats the agreement between two radar passes of the *same city* —
  averaging many passes is more self-consistent than any two, which caps what any model can be
  asked to do here.

### §4 A learned model ties the climatology it was built to beat — 1 p · **Table 2**
Table 2 = all methods on identical scenes: random 0.014, all-wet 0.029, static depth 0.032,
TWI 0.039, HAND-lite 0.039, dynamic twin 0.054, **learned 0.288 ± 0.007**, climatology 0.288.
- Terrain *is* learnable: 5.4× the physics twin on identical scenes. What fails is routing water
  over a noisy DEM, not predicting where it goes.
- **And it ties the climatology to three decimals.** On a city with a radar archive the network
  is worth nothing over "these cells are usually wet". Say it plainly; it sets up §5.

### §5 Transfer, and what "held out" has to mean — 1.25 p · **Table 3**
The methodological core.
- Held-out *tile*: 0.270 ± 0.015, 12.5× all-wet.
- But with 6 Mumbai tiles and 7 Karnataka domains, holding out one leaves siblings in training.
- Held-out *region*: **0.147 ± 0.008**, 6.9×. Half the apparent gain was siblings.
- **Rebuilt-stack control** (0.096 ± 0.009 vs 0.094 ± 0.025 on the same 3 domains) rules out the
  alternative explanation. Keep this — it is what makes the comparison trustworthy.
- Bias falls 1.20 → 0.53 on unseen regions: the model under-warns. Ranking degrades far less
  (0.161 vs 0.303), so it is calibration, not skill.

### §6 The boundary: terrain-type coverage — 1 p · **Table 4**
The positive result, and the one a practitioner acts on.
- Per region: Karnataka 10.6×, Patna 2.8×, **Mumbai 1.8× — near nothing**.
- Chennai added as a 15th domain and its own region makes the test symmetric.
- **Mumbai 0.0333 → 0.0613 (+84 %); Chennai, with Mumbai in training, 0.164 at 7.8×.**
- **Internal control:** every fold gained the same 10 storms; only the coastal region moved
  (+84 % vs +3.6 % and −7.4 %). For scale, 3→14 domains was 4.6× the storms for +39 % overall.
- Rule: deploy on an unseen inland city; on an unseen coastal city only with a coast in training.

### §7 Three levers that are not levers — 0.5 p
Compress hard; one short paragraph each, no tables.
- **Rainfall**: nothing, replicated three ways (per tile, per region, 15 domains / 4 regions:
  0.1460 ± 0.0233 vs 0.1474 ± 0.0081). Plus the reason — 9 probe points over a 15 km tile
  collapse to one archive cell, so spatial rainfall is unavailable, not merely unused.
- **Capacity**: flat 2.0 M → 70.6 M parameters.
- **Resolution**: flat 60 m → 30 m (9.7× vs 10.4× on the skill multiple).

### §8 Limitations, and what this means for the literature — 0.5 p
- Mumbai at 3.3× still trails inland; one coastal sibling narrows the gap without closing it.
  Chennai's own figure rests on 10 scenes.
- LOSO is 18 storms over 2 domains — the smallest evidence base here. Say so.
- The recommendation: **report CSI against all-wet and a leave-one-out climatology, and state
  whether "held out" means a tile or a region.** That is the transferable output.

---

## What is deliberately NOT in this paper

Cut, and each belongs to the system paper: the differentiable twin's construction, canal routing
and storage siting, the GNN street-graph router, the participatory reporting layer and its reward
gate, the LLM brief, the deployment stack, the drainage-ceiling / BMC anchor, the DEM-uncertainty
ensemble (one sentence in §4 only, as the reason routing fails).

Keeping any of these turns it back into a 13-page system paper.

---

## Tables (4, all from committed JSON — no new runs)

| # | content | source |
|---|---|---|
| 1 | trivial baselines, 15 domains | `csi_net/data/trivial_baselines14.json` + chennai |
| 2 | all methods, identical scenes | existing Table 3 of the system paper |
| 3 | transfer: tile vs region + stack control | `results/lodo_*`, `loro_*`, `*_patna+*` |
| 4 | coastal coverage, 14 vs 15 domains | `results/loro_*_varuna_stack14/15.json` |

One figure at most. If any: the per-region bar chart of §6 (Karnataka / Patna / Mumbai / Chennai,
14 vs 15 domains). The DEM-uncertainty figure stays in the system paper.

---

## Practical notes for writing

- Regenerate Table 1 for 15 domains first: `python -m csi_net.build_trivial --stack
  csi_net/data/varuna_stack15.npz --out csi_net/data/trivial_baselines15.json` (~1 min, no GPU).
- Start from a fresh `.tex`; do not fork `varuna-floodtwin.tex` — the point is that this paper is
  short, and forking imports the structure that made the other one long.
- Venue: *Environmental Data Science* (Cambridge) — rolling, Diamond OA, Open Practice Badges,
  and 6–7 pp with a public artifact is exactly its shape.
- The word "honest" appears zero times. Every weakness is a number.
