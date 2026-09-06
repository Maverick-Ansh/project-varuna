# Paper 2 — outline and pre-extracted numbers

**Framing (corrected 2026-09-06): the GNN and the routing are the core.** An earlier draft of
this outline made the `csi_net` measurement work the thesis. That was wrong. The paper is about
**learned street-level flood risk on a real city graph, and routing on it** — the deliverable a
reader can see is Figure 1: a live map with a flow field, orange impassable nodes, a purple
detour, and a named road labelled *waist-deep*. Everything else is supporting evidence.

Write it long first, then cut. Ansh will say what stays.

---

## Figures available (all already in `paper/figures/`)

| file | use |
|---|---|
| `dashboard_patna_route.png` | **Figure 1** — live dashboard: flow arrows, flooded nodes, purple detour, "Birchand Patel Path (Gardner Road) waist-deep" |
| `route_demo_patna.png` | routing demo, shortest vs GNN vs oracle |
| `pred_vs_true.png` | GNN calibration |
| `auc_vs_rain.png` | AUC vs rainfall intensity — does skill hold at extremes |
| `spiderweb_patna.png` | drainage network on the OSM graph |
| `spiderweb_bengaluru.png`, `spiderweb_mumbai_south.png` | cross-city drainage |
| `containers_mumbai_west.png` | buildable storage siting |
| `dose_response.png`, `storage_dose.png` | monotone dose–response |
| `flood_uncertainty.png` | DEM ±1 m ensemble — why the physical router fails |
| `baseline_comparison.png` | CSI baselines |

Also in `artifacts/gnn/figures/`: `train_curve.png`.

---

## Numbers, already extracted — do not re-derive

### GNN per city (`artifacts/gnn/gnn_report.json`)

| area | nodes | edge AUC | wet RMSE (m) | GNN ms/query | emulator ms/query | speedup |
|---|---|---|---|---|---|---|
| patna | 37,017 (79,928 edges) | **0.9056** | 0.154 | 4.1 | 260.2 | 63× |
| patna_east | 13,590 | **0.9422** | — | 3.9 | 110.9 | 28× |
| patna_west | 31,961 | **0.8902** | — | 4.1 | 216.0 | 53× |
| bengaluru | 31,675 | **0.9155** | — | 5.0 | 238.4 | 48× |

Trained on patna + patna_east + patna_west + bengaluru; val rains 60/140/220 mm.
Patna flow_spearman 0.137 (weak — state honestly, the GNN predicts *wetness*, not flow direction).

### Ablation — message passing earns its keep (`artifacts/gnn/ablation.json`)

| area | 0 layers (no message passing) | full | Δ AUC |
|---|---|---|---|
| patna | 0.8661 (rmse 0.196) | 0.9056 | +0.040 |
| patna_east | 0.9082 (rmse 0.306) | 0.9422 | +0.034 |
| patna_west | 0.8793 (rmse 0.140) | 0.8902 | +0.011 |
| bengaluru | 0.8878 (rmse 0.406) | 0.9155 | +0.028 |

### Zero-shot cross-city transfer (`artifacts/gnn/transfer_summary.json`)

Rows = city held OUT of training; **bold = the zero-shot cell**.

| trained without | patna | patna_east | patna_west | bengaluru |
|---|---|---|---|---|
| bengaluru | 0.8938 | 0.9371 | 0.8904 | **0.7212** |
| patna | **0.8473** | 0.9386 | 0.8979 | 0.9216 |

Zero-shot on a genuinely unseen city: **0.7212** (Bengaluru) and **0.8473** (Patna). Features are
local and per-graph standardised, which is why it transfers at all.

### Routing, 150 routes per area (`gnn_report.json` → `routing`)

| area | wet m/route: shortest → **GNN** → oracle | detour %: GNN / oracle | avoidance recovered | % routes hitting water (shortest → GNN) |
|---|---|---|---|---|
| patna | 2297 → **1570** → 1172 | 21.2 / 37.1 | **64.6 %** at 57 % of the detour | 100.0 → 98.7 |
| patna_east | 585 → **406** → 369 | 4.9 / 16.5 | **82.9 %** at 30 % of the detour | 71.3 → 63.3 |
| patna_west | 2024 → **1727** → 1349 | 13.8 / 15.7 | **44.0 %** at 88 % of the detour | 98.7 → 99.3 |
| bengaluru | 1888 → **1172** → 1333 | 24.2 / 35.2 | **129 %** — beats the oracle | 99.3 → 98.0 |

Avoidance recovered = (shortest − GNN)/(shortest − oracle). **Bengaluru exceeds 100 %**: the
oracle is greedy on true depth and the GNN's smoothed risk field finds a better global path.
Worth a paragraph — it is the most interesting single number in the routing table.

### Drainage spiderweb (from the system paper, §"Planning")

Ladder: excavation 4.1 % → pits 7.8 % → sparse canals+pits 20.2–20.8 % → **street spiderweb
80.9 %** → distributed storage 30 %. Cost ₹850/m³ removed vs ₹1,309/m³ sparse (Bengaluru
₹1,121/m³). Per city: Patna 80.9, Patna-E 89.4, Patna-W 80.3, Bengaluru 72.6, Mumbai
S/E/W/N 63.0/50.2/39.2/32.5 %. Exposure at 100 mm: Patna 1,205/3,749 buildings and 3,036/6,762
road segments; Mumbai-West 15,449/31,597 buildings.

### Supporting validation (`csi_net`, compressed to ONE section)

15 domains, 150 SAR scenes. Climatology 0.29–0.67 (mean 0.55); all-wet 0.006–0.054; learned
U-Net 0.288 ± 0.007 **ties** climatology 0.288; twin 0.054. Transfer: per-tile 0.270, per-region
**0.147**. Coastal: Mumbai 0.033 → 0.061 (+84 %) once one coastal city is in training.

---

## Section plan (write long, then cut)

1. **Introduction** — lead with Figure 1. The claim: street-level flood risk is learnable on a
   city graph, transfers zero-shot, and is fast enough to route on.
2. **Related work** — flood mapping, GNNs on road networks, evacuation routing.
3. **System overview** — free satellite data → twin → emulator → GNN → router → dashboard.
4. **Building a city** — FABDEM/WorldCover/JRC/Sentinel-1 → bundle; OSM street graph.
5. **Why a physical router is not enough** — DEM ±1 m ensemble, only 5 % of flooded area robust
   in Patna vs 52 % in hillier Bengaluru. Motivates learning. (`flood_uncertainty.png`)
6. **Learned street-level risk: the GNN** — graph, features, training, AUC table, ablation,
   4.1 ms vs 260 ms. (`pred_vs_true.png`, `auc_vs_rain.png`)
7. **Zero-shot cross-city transfer** — the transfer matrix; why local + per-graph standardised
   features transfer.
8. **Flood-aware routing** — the routing table, the oracle-beating Bengaluru result,
   `route_demo_patna.png`, and Figure 1 revisited.
9. **From routing people to routing water: the drainage spiderweb** — the ladder table,
   `spiderweb_*.png`, cost per m³.
10. **Buildable distributed storage** — `containers_mumbai_west.png`, phase ladder.
11. **Validation: what a flood CSI is worth** — the whole `csi_net` story in one tight section.
12. **The deployed system** — dashboard, API, free-tier envelope.
13. **Limitations** — flow_spearman 0.137; zero-shot 0.72 is a real drop; oracle is greedy not
    optimal; 150 synthetic OD pairs, not observed trips.
14. **Conclusion.**

---

## Practical

- New file `paper/varuna-gnn-routing.tex`. Reuse the preamble from `varuna-floodtwin.tex`
  (article 10pt, geometry, booktabs, natbib, `\rupee` macro) — it compiles clean with `latexmk
  -pdf` from Git Bash. MiKTeX is on PATH.
- `references.bib` already exists and is shared.
- Write the whole thing first; Ansh trims after reading.
