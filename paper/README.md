# Workshop paper — Varuna FloodTwin

**Target venues:** Climate Change AI workshop (NeurIPS/ICLR editions) or an EGU session on urban
hydrology / natural hazards. Body is written to fit a 4-page limit excluding references.

## Files
- `varuna-floodtwin.tex` — the paper (self-contained; compiles with any TeX distribution)
- `references.bib` — bibliography
- `figures/` — all figures, regenerable from the artifact bundles:
  - `spiderweb_patna.png`, `spiderweb_bengaluru.png` — street-routed drain network over OSM roads
    (regenerate: overlay `canal_plan.json`'s `network.edges` on `exposure.json` road polylines)
  - `flood_uncertainty.png`, `baseline_comparison.png`, `storage_dose.png`, `dose_response.png` —
    copied from `artifacts/patna/figures/` (built by `varuna/build/baselines.py` + serve suite)
  - `route_demo_patna.png` — FloodGNN flood-safe routing demo (currently the CPU **smoke**
    checkpoint's render; regenerate with `scripts/run_gnn_colab.py --figures` after the
    production training and re-copy)
- `GNN_SECTION.md` — provenance + experiment checklist for §5 (the FloodGNN section); every
  `\todo{TODO-COLAB}` in the tex maps to a `run_gnn_colab.py` flag there

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
| baseline CSI table (0.032 / 0.042 / 0.051 / 0.041) | `artifacts/patna/BASELINES.md`, `baseline_comparison.json` |
| twin mean CSI 0.033 (2-day window), best storm 0.112 | `VALIDATION.md`, `twin_scores_2025-08-02.json` |
| calibration near-null 0.0483 → 0.0488 | `artifacts/patna/calibration_report.json` |
| DEM ensemble: Patna 10.79±0.35 km², 5% robust; Bengaluru 52% | `artifacts/{patna,bengaluru}/flood_uncertainty.json` |
| spiderweb cuts 80.9 / 89.4 / 80.3 / 72.6 % | `artifacts/<area>/canal_plan.json` (`reduction_pct`, `network`) |
| strategy ladder (pits 7.8%, sparse canals 20.2–20.8%) | `artifacts/patna/strategy_results.json`, `CANAL_RESULTS.md` |
| storage 727 sites → 30% (Patna), 431 → 30% (Bengaluru) | `artifacts/<area>/storage_sizing.json`, `STORAGE_RESULTS.md` |
| INR 850/m³ (spiderweb) vs INR 1,309/m³ (sparse) | spiderweb: `network.total_length_m`×9000 + pit excavation×300 over volume removed; sparse: `artifacts/patna/costbenefit.json` |
| exposure 1,205/3,749 buildings, 3,036/6,762 roads | `artifacts/patna/exposure.json` |
| dose-response 25→200 mm monotone | `artifacts/patna/dose_response.json` |
| FloodGNN smoke numbers quoted in §5 (AUC 0.856; routing 2,297→1,735→1,172 m) | CPU smoke run 2026-07-06, table in `paper/GNN_SECTION.md` |
| FloodGNN production AUC / routing / ρ / speed (`TODO-COLAB`) | `artifacts/gnn/gnn_report.json` after `scripts/run_gnn_colab.py --train --eval` |
| FloodGNN transfer + ablation (`TODO-COLAB`) | `artifacts/gnn/transfer_summary.json`, `ablation.json` after `--transfer --ablation` |

Note: `artifacts/patna/costbenefit.json` still prices the **old sparse plan** (kept as the paper's
comparison point). Re-running `/api/costbenefit` reprices with the spiderweb network length.
