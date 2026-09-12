# Paper — Varuna FloodTwin

**Target venues:** Climate Change AI workshop (NeurIPS/ICLR editions) or an EGU session on urban
hydrology / natural hazards. This is the **complete** version (three cities, five findings); to
meet a 4-page workshop limit, condense §2 and §6 and drop Table 3 — the abstract and findings
survive intact.

## Files
- `varuna-floodtwin.tex` — the paper (self-contained; compiles with any TeX distribution)
- `PAPER.md` — GitHub-readable mirror (the tex is canonical for submission)
- `references.bib` — bibliography (14 entries, all real)
- `figures/` — all figures, regenerable from the artifact bundles:
  - `spiderweb_patna.png`, `spiderweb_bengaluru.png`, `spiderweb_mumbai_south.png`,
    `containers_mumbai_west.png` — map overlays; regenerate with
    `python scripts/make_paper_figures.py --spiderweb <area> --containers <area>`
  - `flood_uncertainty.png`, `baseline_comparison.png`, `dose_response.png` —
    copied from `artifacts/patna/figures/` (built by `varuna/build/baselines.py` + serve suite)
  - `storage_dose.png` — `artifacts/patna/figures/storage_dose.png` (**buildable-siting** version,
    regenerated 2026-07-12)
  - `route_demo_patna.png` — FloodGNN flood-safe routing demo (production checkpoint)
- `GNN_SECTION.md` — provenance + experiment checklist for §5 (historical; all numbers now live
  in the tex)

## Figures in `varuna-short.tex` (the 5-page conference cut)

The short paper carries twelve figures: two colour flowcharts drawn in TikZ **inside the tex**
(no external file, nothing to regenerate), six charts built from the artifacts, and four map or
dashboard images that already lived in `figures/`.

| figure | file | how to regenerate |
|---|---|---|
| 1 teaser | `figures/live_patna_city.png` | dashboard screenshot |
| 2 the whole process | TikZ, inline in `varuna-short.tex` | edit the tex |
| 3 what a CSI is worth | `figures/f_csi_worth.pdf` | `python scripts/make_short_paper_figures.py` |
| 4 method ladder | `figures/f_method_ladder.pdf` | same script |
| 5 DEM ensemble | `figures/flood_uncertainty.png` | `varuna/build/baselines.py` |
| 6 transfer + calibration | `figures/f_transfer.pdf` | same script |
| 7 the three non-levers | `figures/f_nonlevers.pdf` | same script |
| 8 spiderweb algorithm | TikZ, inline in `varuna-short.tex` | edit the tex |
| 9 spiderweb on real streets | `figures/spiderweb_patna.png` | `python scripts/make_paper_figures.py --spiderweb patna` |
| 10 planning results | `figures/f_planning.pdf` | same script |
| 11 metered recharge | `figures/f_recharge.pdf` | same script |
| 12 flood-safe routing | `figures/route_demo_patna.png` | FloodGNN routing demo |

`scripts/make_short_paper_figures.py` holds no computation. Every value in it is transcribed from a
committed artifact named in a `SOURCE:` comment above the block that uses it, so a chart cannot
drift away from the text. The colour slots (blue, orange, aqua, yellow) are defined twice on
purpose, once in that script and once in the `varuna-short.tex` preamble, so a colour means the
same thing in a flowchart as it does in a chart.

## Build
```bash
pdflatex varuna-floodtwin && bibtex varuna-floodtwin && pdflatex varuna-floodtwin && pdflatex varuna-floodtwin
```
or upload the `paper/` folder to Overleaf as-is.

**On submission:** swap `\documentclass{article}` + the geometry/preamble block for the venue
template (CCAI uses a NeurIPS-based style); the `\rupee` macro is a plain-text `INR` fallback —
replace with the template's rupee glyph if available.

## Where every number comes from (single source of truth)

| claim | artifact |
|---|---|
| trivial baselines (all-wet 0.013–0.047, climatology 0.30–0.58, SAR-vs-SAR ceiling) | `artifacts/trivial_baselines.json` |
| like-for-like table on 18 storms (twin 0.054 / TWI 0.039 / HAND 0.039 / depth 0.032 / all-wet 0.029 / random 0.014 / climatology 0.288) | `csi_net/results/twin_protocol.json` |
| learned net 0.288 ± 0.007 (LOSO), 0.094 ± 0.021 (unseen city) | `csi_net/results/loso_rain_w32_g60_s*_patna+mumbai_northeast.json`, `lodo_rain_w32_g60_s*.json` |
| rainfall ablation, permutation control, tide arms, 30 m vs 60 m, increment target | `csi_net/results/*.json`, tabulated in `csi_net/RESULTS.md` |
| rain source is ECMWF IFS 9 km, not ERA5 (554.2 vs 1213.4 mm) | `varuna/serve/weather.py:ARCHIVE_MODEL` docstring (measurement inline) |
| *superseded:* baseline CSI table (0.032 / 0.042 / 0.051 / 0.041), twin 0.033 | `artifacts/patna/BASELINES.md`, `baseline_comparison.json`, `VALIDATION.md` — kept as the record the new table reproduces to ±0.0007 |
| calibration near-null 0.0483 → 0.0488 | `artifacts/patna/calibration_report.json` |
| DEM ensemble: Patna 10.79±0.35 km², 5% robust; Bengaluru 52% | `artifacts/{patna,bengaluru}/flood_uncertainty.json` |
| spiderweb cuts 80.9 / 89.4 / 80.3 / 72.6 % (Patna-family, Bengaluru) | `artifacts/<area>/canal_plan.json` (`reduction_pct`, `network`) |
| Mumbai spiderweb cuts 63.0 / 50.2 / 39.2 / 32.5 % | `artifacts/mumbai_{south,east,west,north}/canal_plan.json` |
| strategy ladder (pits 7.8%, sparse canals 20.2–20.8%) | `artifacts/patna/strategy_results.json`, `CANAL_RESULTS.md` |
| buildable sites / under-street share / 10-20-40% phases + ₹ | `artifacts/<area>/storage_sizing.json` (`sites`, `phases`; regenerated 2026-07-12 with building/water exclusion) |
| legacy storage 727 sites → 30% (pre-buildability, §4 ladder table) | git history of `artifacts/patna/storage_sizing.json` (pre-PR#13), `STORAGE_RESULTS.md` |
| INR 850/m³ (spiderweb) vs INR 1,309/m³ (sparse) | spiderweb: `network.total_length_m`×9000 + pit excavation×300 over volume removed; sparse: `artifacts/patna/costbenefit.json` |
| exposure: Patna 1,205/3,749 buildings; Mumbai-West 15,449/31,597 | `artifacts/{patna,mumbai_west}/exposure.json` |
| emulator val RMSE 5–21 cm (south 5.0, west 20.9, Bengaluru 10.4 cm) | `artifacts/<area>/twin_meta.pt` (`val_rmse_m`) |
| deterministic collapse (frozen 16.3/34.2 cm; 3 bit-identical failures) | build logs 2026-07-12; fix + tell documented in `varuna/build/twin.py` (`train_emulator` docstring, commit `d04561d`) |
| water-balance closure ~10⁻⁶ relative | `artifacts/<area>/water_balance.json` (`closure_err_pct`); assertion in `varuna/serve/waterbalance.py` |
| dose-response 25→200 mm monotone | `artifacts/patna/dose_response.json` |
| FloodGNN AUC 0.906 / routing 2,297→1,570→1,172 m / ρ 0.14–0.24 / 4 ms | `artifacts/gnn/gnn_report.json` |
| FloodGNN transfer (0.721 / 0.847 zero-shot) + ablation (0.866/0.890/0.906) | `artifacts/gnn/transfer_summary.json`, `ablation.json` |
| reward-gate thresholds (15 pts / 2 days / −5% / +10% / 70%) | `varuna/learn/reward.py` defaults; tested in `tests/test_learn.py` |
| citizen-report bands ±0.15–0.35 m, rate limits | `varuna/serve/reports.py` (`DEPTH_TOL_M`, `RATE_*`) |

Notes:
- `artifacts/patna/costbenefit.json` still prices the **old sparse plan** (kept as the paper's
  comparison point). Re-running `/api/costbenefit` reprices with the spiderweb network length.
- The §4 ladder table quotes the pre-buildability storage point (727 sites → 30%) for the Patna
  strategy comparison; Table 3 quotes the current buildable-siting phases. Both artifacts exist —
  the difference IS the buildability cost discussed in the text.
