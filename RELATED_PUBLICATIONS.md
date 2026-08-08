# Related Publications & Patents — Project Varuna / FloodTwin

**Compiled by:** Ansh Vivek (independent researcher) · anshvivek2003@gmail.com
**Date:** 8 August 2026
**System under study:** *Varuna FloodTwin* — an open, free-tier, satellite-built **differentiable** urban flood digital twin with street-graph drainage planning, a self-supervised street-graph GNN, buildability-constrained detention siting, managed-aquifer-recharge coupling, and a reward-gated citizen-report learning loop.

Live: [project-varuna-iota.vercel.app](https://project-varuna-iota.vercel.app) · API: [anshvivek-varuna-floodtwin.hf.space](https://anshvivek-varuna-floodtwin.hf.space) · Code: [github.com/Maverick-Ansh/project-varuna](https://github.com/Maverick-Ansh/project-varuna)

---

## What this document is

A curated reading list of **21 headline works — 15 peer-reviewed / preprint publications and 6 patents — that other people have published on the same problems Varuna works on**, each annotated with *what it does* and *how it relates to (and differs from) Varuna*. An appendix lists ~35 further works one line each.

This is a **literature and prior-art map, not a legal freedom-to-operate opinion**. It is the curated distillation of the full prior-art sweep in [`noveltySearch.md`](noveltySearch.md) (23 July 2026, ~99 items across five sub-domains), refreshed with a live search on **8 August 2026** that added two items (CN119090245A and HydroGAT, both marked *new*). Every URL here was returned by live web or patent search; **no citation or patent number is fabricated.** Paywalled or anti-bot-blocked pages are given as canonical DOI / publisher landing pages.

**Read this honestly.** Several works below are *closer* to Varuna than is comfortable — in particular **Inunda** (arXiv, July 2026) is a near-concurrent, near-identical differentiable Bates-2010-in-PyTorch core. It is related work to be cited and distinguished, **not** something Varuna predates. Likewise, Bates-2010, FiLM, HAND/TWI, VIIRS Black Marble and ASR/MAR are established methods Varuna *uses*; they are listed as methods and baselines, not as things Varuna claims.

**Varuna's own contributions**, referenced below as V1–V9, are: V1 free-tier satellite→twin→dashboard pipeline · V2 honest SAR validation + DEM-uncertainty diagnosis · V3 street-graph "spiderweb" drainage with re-simulated benefit · V4 FloodGNN self-supervised street-graph GNN · V5 buildability-constrained phased detention siting · V6 flood-to-drought aquifer-recharge coupling · V7 reward-gated citizen online learning · V8 seeded-collapse reproducibility finding · V9 grounded multilingual advisories + VIIRS outage overlay.

---

## Part A — 15 publications

### A. Differentiable & reduced-complexity flood physics (relates to V1)

**1. Inunda: a GPU-native differentiable flood solver** — Zhi Li, 2026, arXiv preprint
<https://arxiv.org/html/2607.09614>
*What it does:* Implements the Bates-2010 local-inertial 2-D shallow-water solver fully in PyTorch and calibrates Manning's *n*, bed elevation and infiltration by autodiff-through-the-time-loop plus gradient descent.
*Relation to Varuna:* **The closest single work in the entire literature — essentially Varuna's exact numerical core.** Differs in that it is a *pure solver* on US datasets: no U-Net emulator, no hosted twin or dashboard, no free-global-data developing-world framing, no drainage/GNN/recharge/citizen stack. **Near-concurrent (July 2026)** — related work, not predated.

**2. A simple inertial formulation of the shallow water equations** — Bates, Horritt & Fewtrell, 2010, *Journal of Hydrology*
<https://doi.org/10.1016/j.jhydrol.2010.03.027>
*What it does:* Introduces the local-inertial (LISFLOOD-ACC) scheme — the reduced-complexity formulation that made metre-scale 2-D urban inundation tractable.
*Relation to Varuna:* **This *is* Varuna's numerical core**, cited as method. The original is non-differentiable Fortran/C; Varuna's contribution is the PyTorch, gradient-optimisable re-implementation — **not the scheme itself.**

**3. LISFLOOD-FP 8.0** — Shaw, Sharifian, Bates, Neal et al., 2021, *Geoscientific Model Development*
<https://gmd.copernicus.org/articles/14/3577/2021/>
*What it does:* The operational descendant of Bates-2010 — DG/acceleration solvers, GPU support, subgrid channels; the field's gold-standard open hydraulic engine.
*Relation to Varuna:* Direct lineage and the correct numerical baseline to benchmark against. Differs: classic HPC code with no autodiff, no ML emulator, no web-served twin.

**4. Universal Shallow Water Equations with Differentiable Programming** — Liu et al., 2025, *Water Resources Research*
<https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025WR040265>
*What it does:* Embeds a neural flow-resistance closure inside the full 2-D SWE and trains it end-to-end by differentiable programming.
*Relation to Varuna:* Very close on differentiable roughness — the same "learn the closure through the solver" idea. Differs: full SWE rather than reduced local-inertial, and a methods paper on a closure — no emulator, twin, or open-data city pipeline. *(Paywalled; DOI verified.)*

**5. DassFlow2D-V3: variational data assimilation with an algorithmic-differentiation adjoint** — Pujol, Monnier, Larnier et al., 2022, *Geoscientific Model Development*
<https://gmd.copernicus.org/articles/15/6085/2022/>
*What it does:* A 2-D SWE finite-volume model whose auto-differentiated adjoint assimilates flood and satellite observations to infer parameters.
*Relation to Varuna:* **The closest differentiable-full-hydraulics lineage that clearly predates Varuna.** Differs: adjoint of Fortran rather than native PyTorch autodiff, full SWE, and a research assimilation tool — no ML emulator, no public twin.

### B. AI flood digital twins & emulators (relates to V1, V8)

**6. Toward AI-Driven Digital Twins for Metropolitan Floods (CLDNet)** — 2026, arXiv preprint
<https://arxiv.org/html/2605.13761v1>
*What it does:* An observation-coupled digital twin built on a fast low-dimensional latent-dynamics surrogate of the shallow-water equations, with a meshless decoder queried at gauge locations.
*Relation to Varuna:* The closest "AI digital twin + SWE surrogate" concept to Varuna's twin+emulator pair. Differs: a latent-dynamics surrogate rather than a differentiable Bates solver paired with a U-Net emulator; no open-satellite, developing-city, free-tier framing.

**7. FlowsDT: a geospatial digital twin for navigating urban flood dynamics** — 2026, *Computers, Environment & Urban Systems* (preprint: arXiv 2507.08850)
<https://www.sciencedirect.com/science/article/pii/S0198971526000165> · <https://arxiv.org/pdf/2507.08850>
*What it does:* A geospatial digital twin over 1D–2D coupled hydrodynamics for Galveston, Texas — topography, hydrography and infrastructure, validated against historical events and social sensing, navigable across return-period scenarios.
*Relation to Varuna:* The most direct "urban flood digital twin" peer, and it shares the social-sensing validation instinct behind V7. Differs: not differentiable, no PyTorch emulator, and free-tier reproduction from free global data is not its thesis.

**8. Global prediction of extreme floods in ungauged watersheds** — Nearing, Cohen et al. (Google Flood Hub), 2024, *Nature*
<https://www.nature.com/articles/s41586-024-07145-1>
*What it does:* An LSTM river-discharge model forecasting extreme streamflow globally, including ungauged basins, at up to 5-day lead time — the flagship operational AI-flood system.
*Relation to Varuna:* The reference point for "AI flood forecasting at global scale" and the strongest argument that data-scarce regions can be served. Differs fundamentally: forecasts *river-gauge hydrographs per basin*, not per-street urban inundation — no city graph, no drainage design, no routing, no simulator self-supervision.

### C. Graph neural networks for flood & drainage (relates to V4)

**9. SWE-GNN: hydraulics-based graph neural networks for flood modelling** — Bentivoglio, Isufi, Jonkman & Taormina, 2023, *HESS*
<https://hess.copernicus.org/articles/27/4227/2023/>
*What it does:* A GNN surrogate of the shallow-water equations on a finite-volume mesh graph — roughly 100× faster than the solver and generalising to unseen topographies.
*Relation to Varuna:* One of the two closest works to FloodGNN. Differs: the graph is *hydrodynamic mesh cells*, not the OSM road network; autoregressive depth rollout rather than single-pass per-street risk; boundary-driven rather than FiLM-conditioned on rainfall; no routing application.

**10. mSWE-GNN: multi-scale hydraulic graph neural networks** — Bentivoglio et al., 2025, *NHESS*
<https://nhess.copernicus.org/articles/25/335/2025/>
*What it does:* The multi-resolution successor to SWE-GNN, generalising across meshes, topographies and boundary conditions at 700–1000× speed-up.
*Relation to Varuna:* The closest analog to Varuna's cross-topography generalisation claim. Differs: a mesh-node autoregressive surrogate that still needs a fine-tuning simulation per new site, versus FloodGNN's zero-shot flat↔hilly transfer; no street graph, no routing.

**11. FloodGNN-GRU: a spatio-temporal graph neural network for flood prediction** — Kazadi, Doss-Gollin, Sebastian & Silva, 2024, *Environmental Data Science*
<https://www.cambridge.org/core/journals/environmental-data-science/article/floodgnngru-a-spatiotemporal-graph-neural-network-for-flood-prediction/93BA1DA8D6ECC93D985656C3BC1EA3DE>
*What it does:* A GNN+GRU predicting depth and velocity over a mesh graph, trained and tested on a single Hurricane Harvey simulation, ~1000× faster than the solver.
*Relation to Varuna:* The nearest namesake to FloodGNN and the most important work to distinguish by name. Differs: temporally autoregressive on *one* storm — the authors themselves flag poor rainfall generalisation — on a mesh rather than roads, with no FiLM continuous-storm conditioning, no cross-city transfer, and no routing.

**12. Pluvial flood emulation with hydraulics-informed message passing** — Kazadi, Doss-Gollin & Da Silva, 2024, *ICML*
<https://proceedings.mlr.press/v235/kazadi24a.html>
*What it does:* A hydraulics-informed message-passing GNN that predicts flood depth autoregressively from terrain and precipitation.
*Relation to Varuna:* Very close in spirit — a simulation-trained, rainfall-driven GNN emulator. Differs: full depth-field rollout on a terrain graph rather than per-street risk in a single pass; no explicit FiLM conditioning; no routing or transfer study.

**13. HydroGAT: distributed heterogeneous graph attention transformer for spatiotemporal flood prediction** *(new, added 8 Aug 2026)* — 2025, arXiv preprint
<https://arxiv.org/pdf/2509.02481>
*What it does:* A distributed heterogeneous graph-attention transformer for spatiotemporal flood prediction, scaling attention-based message passing across large hydrologic graphs.
*Relation to Varuna:* The current scaling frontier for graph-based flood models and the strongest argument that graph attention beats plain message passing at size. Differs: heterogeneous hydrologic graph and distributed training, not a self-supervised street-graph risk model conditioned on continuous rainfall, and no evacuation-routing or drainage-design consumer of its output.

### D. Validation, DEM uncertainty & citizen data (relates to V2, V7)

**14. Validation of a 30 m resolution flood hazard model of the conterminous United States** — Wing, Bates, Sampson et al., 2017, *Water Resources Research*
<https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017WR020917>
*What it does:* Benchmarks a CONUS-wide 2-D hydraulic model against observed flood extents, reporting Critical Success Index values up to ~0.87 — the canonical CSI validation study.
*Relation to Varuna:* Defines the metric and the standard Varuna's V2 validation is measured against. Differs sharply in outcome and intent: Varuna reports an *honest negative* (CSI ≈ 0.033–0.041 on flat deltaic terrain) and then diagnoses the mechanism, rather than presenting a headline CSI.

**15. Vertical accuracy of freely available global digital elevation models in flood-prone terrain** — Marsh et al., 2024, *International Journal of Digital Earth*
<https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2308734>
*What it does:* Quantifies vertical error of free global DEMs across floodplains, finding FABDEM the best performer at roughly 1.24 m RMSE.
*Relation to Varuna:* The empirical basis for Varuna's ±1 m spatially-correlated FABDEM perturbation ensemble. Differs: it *measures* DEM error; Varuna *propagates* that error through a dynamic twin to measure the resulting SAR co-location loss (~5 % flat vs ~52 % hilly) — the V2 contribution.

**16. Can assimilation of crowdsourced data in hydrological modelling improve flood prediction?** — Mazzoleni et al., 2017, *HESS*
<https://hess.copernicus.org/articles/21/839/2017/>
*What it does:* Assimilates (synthetic) crowdsourced observations of variable accuracy and irregular arrival into hydrological models, quantifying the forecast benefit.
*Relation to Varuna:* The closest citizen-data→model-update precedent to V7. Differs decisively: continuous state assimilation with synthetic data and **no deployment gate** — Varuna does reward-gated emulator fine-tuning on real citizen depth-band reports with an auditable ship/no-ship decision and rollback.

### E. Drainage design & flood-to-aquifer recharge (relates to V3, V5, V6)

**17. Generation of optimal (de)centralized layouts for urban drainage systems: a graph-theory combinatorial multi-objective optimisation** — Hesarkazzazi et al., 2022, *Sustainable Cities & Society*
<https://www.sciencedirect.com/science/article/pii/S2210670722001548>
*What it does:* Takes the **street network as the base graph**, then applies graph theory plus multi-objective optimisation to generate optimal centralised/decentralised sewer layouts.
*Relation to Varuna:* The closest prior art to V3's street-graph drainage generation. Differs in the evaluation loop: it optimises pipe topology and cost against a hydraulic model, whereas Varuna **re-simulates the 2-D twin to *measure* the street-flood-volume cut** (73–89 %) and routes from explicitly flood-safe non-river outfalls.

**18. Respond to urban floods and groundwater depletion: a managed aquifer recharge approach** — 2026, *Science of the Total Environment*
<https://www.sciencedirect.com/science/article/abs/pii/S0048969726001580>
*What it does:* Repurposes 23 abandoned wells for Aquifer Storage, Transfer and Recovery, injecting stormwater during peak rain; SWMM+MODFLOW over 2009–2023 shows up to ~5 % single-storm flood-volume reduction and 0.5–12 m water-table rise.
*Relation to Varuna:* **The closest MAR-coupling work — explicitly dual flood *and* groundwater**, and the right quantitative comparison for V6's metered recharge plans (6.3–16.5 % of runoff to aquifer in the Karnataka towns). Differs: well-injection hydrogeology on existing wells; Varuna adds street-graph drainage design, detention siting and a re-simulated flood cut, framed to Indian norms.

**19. Underground Taming of Floods for Irrigation (UTFI)** — Pavelic et al. / IWMI, 2015
<https://cgspace.cgiar.org/items/7d13ba30-dba2-42c0-af1b-f596b044cd64>
*What it does:* Introduces UTFI — diverting high monsoon flows into aquifer recharge structures in the Ganges basin, cutting flood peaks while banking dry-season irrigation water.
*Relation to Varuna:* **Conceptually the single closest prior art to Varuna's flood-to-drought thesis.** Differs: rural and river-basin scale via ponds and wells; Varuna is *urban*, operates on the city street graph, engineers the drainage network, and re-simulates the measured street-flood cut.

**20. NASA's Black Marble nighttime lights product suite** — Román et al., 2018, *Remote Sensing of Environment*
<https://www.sciencedirect.com/science/article/pii/S003442571830110X>
*What it does:* Defines the VNP46 daily VIIRS Day/Night-Band product at 500 m, atmospherically and lunar-BRDF corrected for day-to-day comparability.
*Relation to Varuna:* The exact data product behind V9's power-outage overlay, cited as **method**. Varuna operationalises it as a live flood-dashboard outage layer with an explicit "unknown ≠ dark" monsoon cloud-mask caveat — an integration, not a remote-sensing advance.

**21. FiLM: Visual Reasoning with a General Conditioning Layer** — Perez, Strub, de Vries, Dumoulin & Courville, 2018, *AAAI*
<https://arxiv.org/abs/1709.07871>
*What it does:* Introduces Feature-wise Linear Modulation — affine (γ, β) conditioning of a network's intermediate features on side information.
*Relation to Varuna:* The exact conditioning mechanism FloodGNN uses, cited as **method**. The claim is not FiLM but *applying FiLM-on-rainfall to a street-graph flood GNN so a single model spans a continuous storm range* instead of one model per return period.

---

## Part B — 6 patents

Summaries are read from published abstracts and claims as returned by Google Patents; **they are not a legal claim-chart**, and a proper chart should precede any filing.

**P1. WO2025106413A1 — "Machine learning deluge"** — Autodesk, filed 2024, published 2025
<https://patents.google.com/patent/WO2025106413A1/en>
*Claims:* A sequence of CNNs trained on ~10,000 stormwater simulations produces flood maps 16–25× faster than the underlying deluge solver, updating interactively as a designer places ponds and swales.
*Relation to Varuna:* **The closest patent to Varuna's fast ML twin**, and the closest to V5's interactive intervention siting. Differs: a CNN surrogate of a CAD solver for design-time what-ifs — not a *differentiable satellite/DEM-built* twin; no street-graph GNN, no re-simulation-measured drainage benefit, no citizen online learning, no aquifer coupling.

**P2. CN119090245A — Flood control dispatching management based on a digital twin basin** *(new, added 8 Aug 2026)* — Hunan Wuling Power Technology / Wuling Power, 2024
<https://patents.google.com/patent/CN119090245A/en>
*Claims:* Builds a 3-D digital model of a target watershed, trains LSTMs on historical weather and water-level data to forecast levels and flows, and issues scheduling recommendations to operators.
*Relation to Varuna:* The most recent "flood digital twin" patent found and the clearest evidence that twin+ML flood filings are accelerating. Differs: basin-scale reservoir/gate dispatch driven by time-series LSTMs — no 2-D differentiable hydraulics, no street-graph design, no urban per-street risk or recharge.

**P3. US20150134126A1 — Intelligent drainage system** — IBM (now GlobalFoundries), filed 2013, published 2015
<https://patents.google.com/patent/US20150134126A1/en>
*Claims:* Sensors feed a max-flow analysis that identifies an optimal diversion path through the drainage network and actuates valves to prevent overflow.
*Relation to Varuna:* **The closest smart-drainage patent** to V3. Differs: real-time valve control on a *physical, already-built* network via max-flow, rather than a re-simulation-measured *street-graph drainage design* generated inside a twin; no recharge or evacuation layer.

**P4. US20130116920A1 — Flood-aware travel routing** — IBM, filed 2011, published 2013
<https://patents.google.com/patent/US20130116920A1/en>
*Claims:* A geospatial database, route generator, flood simulator and risk model together produce multiple candidate routes scored by the flood risk a traveller would face.
*Relation to Varuna:* **The closest flood-safe-routing patent** to V4's evacuation endpoint — and notably it already couples a simulator to a router. Differs: an offline-trained risk model feeding a GPS router; no differentiable city twin generating the risk surface, no learned street-graph GNN, no online citizen updates, no drainage/recharge integration.

**P5. US20120264393A1 — Flood data collection and warning** — IBM (now TSMC), filed 2011, published 2012
<https://patents.google.com/patent/US20120264393A1/en>
*Claims:* Citizens report water level by SMS or voice call to "totem" nodes; the system predicts levels, returns feedback to reporters, and auto-subscribes them to alerts.
*Relation to Varuna:* **The closest crowdsourced-citizen-report patent** to V7. Differs where it matters most: citizen input feeds a threshold predictor — there is **no learned model-update loop**, no reward gate, no rollback, and no public audit log, which is precisely the layer Varuna adds.

**P6. WO2009138995A1 — Rainwater harvesting and artificial recharge trench** — V. K. Kedia (India), filed 2008, published 2009
<https://patents.google.com/patent/WO2009138995A1/en>
*Claims:* An underground PVC-lined boundary trench captures roughly 80 % of plot rainfall as artificial groundwater recharge while reducing runoff and erosion.
*Relation to Varuna:* The most relevant *Indian* stormwater-to-groundwater recharge patent, in the same policy space as V6 and Atal Bhujal Yojana. Differs: a passive civil structure at plot scale — no metered recharge control, no flood-to-aquifer optimisation model, no twin.

**Patent white-space observed.** Across the ~21 patents swept in the full search, none was found that (a) builds a **differentiable, simulation-ready city flood twin from satellite + DEM data**, (b) uses a **self-supervised street-graph GNN** for urban flood risk — only academic work exists there — or (c) couples **flood runoff to metered aquifer recharge as a twin-optimised objective**, since the recharge patents located are all passive physical infrastructure. Those three axes are where Varuna's patent-novelty concentrates. This is an observation from an abstract-level sweep, not a legal opinion.

---

## Appendix — further reading (one line each)

**Flood twins, solvers & commercial products**

- SFINCS — Deltares (Leijnse et al.), 2020– — open reduced-complexity compound-flood solver, non-differentiable — <https://github.com/Deltares/SFINCS>
- Subgrid local-inertial formulation for urban flood simulation — Nithila Devi & Kuiry, 2024, WRR — <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR035334>
- Adjoint-based assimilation for 2-D urban flood parameters — Zhang, Monnier, Garambois et al., 2024, J. Hydrology — <https://www.sciencedirect.com/science/article/abs/pii/S0022169424012812>
- smash v1.0: differentiable regionalizable hydrological modelling — Colleoni et al., 2025, GMD — <https://gmd.copernicus.org/articles/18/7003/2025/>
- Fully differentiable distributed rainfall-runoff modelling — 2025, HESS — <https://hess.copernicus.org/articles/29/6257/2025/>
- FloodCast / GeoPINS — Xu, Zhu, Bamber et al., 2024, Water Research — <https://arxiv.org/abs/2403.12226>
- Deep CNN for rapid fluvial flood inundation — Kabir, Patidar, Xia, Liang et al., 2020, J. Hydrology — <https://arxiv.org/pdf/2006.11555>
- Data-driven flood emulation with deep CNNs — Guo, Leitão, Simões & Moosavi, 2021, J. Flood Risk Management — <https://arxiv.org/abs/2004.08340>
- U-RNN: high-resolution spatiotemporal urban-flood nowcasting — Jiang, Chen et al., 2025, J. Hydrology — <https://www.sciencedirect.com/science/article/abs/pii/S002216942500455X>
- Digital Twins for Urban Flood Risk Management: a systematic review — 2025, Remote Sensing — <https://www.mdpi.com/2072-4292/17/17/3104>
- Probabilistic flood-risk mapping for Indian megacities (ALERT, GEE) — 2025 — <https://www.sciencedirect.com/science/article/pii/S2590123025039246>
- Copernicus/Sentinel-1 Global Flood Monitoring (GFM) — 2025, Remote Sensing of Environment — <https://www.sciencedirect.com/science/article/pii/S0034425725005127>
- Fathom Global Flood Map (commercial) — <https://www.fathom.global/product/global-flood-map/>
- JBA Global Flood Models (commercial) — <https://jbagr.com/digital-tools/global-flood-models/>
- Previsico Flood Intel Platform (commercial) — <https://previsico.com/en-us/intel-platform>

**AI flood prediction, GNNs & routing**

- Predictions in Ungauged Basins with Machine Learning — Kratzert, Klotz, Shalev et al., 2019, WRR — <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019WR026065>
- ASTGCN: spatial–temporal graph DL for urban flood nowcasting — Farahmand, Xu & Mostafavi, 2023, Scientific Reports — <https://www.nature.com/articles/s41598-023-32548-x>
- Interpretable physics-informed GNNs for flood forecasting — Taghizadeh et al., 2025, CACIE — <https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13484>
- GNN surrogate for real-time hydraulics of urban drainage networks — Zhang, Tian, Lu, Liao & Yuan, 2024, Water Research — <https://www.sciencedirect.com/science/article/abs/pii/S0043135424010418>
- Multi-fidelity GNNs for flood-hazard mapping — Bentivoglio group, 2025, Reliability Eng. & System Safety — <https://www.sciencedirect.com/science/article/abs/pii/S136481522500338X>
- GNNs for route recommendation under probabilistic hazards — Liu & Meidani, 2025, arXiv — <https://arxiv.org/abs/2501.09803>
- Flood numerical model + Dijkstra risk-avoidance routing — 2023, Water Resources Management — <https://link.springer.com/article/10.1007/s11269-023-03500-5>
- Emergency evacuation routing for dam-failure flooding — 2025, Applied Sciences — <https://www.mdpi.com/2076-3417/15/8/4518>
- Transferability of data-driven urban pluvial flood-depth models (Berlin) — Seleem et al., 2023, NHESS — <https://nhess.copernicus.org/articles/23/809/2023/>
- Cross-region streamflow forecasting at global scale (ED-DLSTM) — Li et al., 2024, The Innovation — <https://www.cell.com/the-innovation/fulltext/S2666-6758(24)00055-9>
- Deep learning for flood mapping: a review — Bentivoglio, Isufi, Jonkman & Taormina, 2022, HESS — <https://hess.copernicus.org/articles/26/4345/2022/>

**Validation, DEM uncertainty, citizen science & night-lights**

- First collective validation of global fluvial flood models — Bernhofen et al., 2018, Environmental Research Letters — <https://iopscience.iop.org/article/10.1088/1748-9326/aae014>
- Sen1Floods11 — Bonafilia, Tellman, Anderson & Issenberg, 2020, CVPR-W — <https://ieeexplore.ieee.org/document/9150760/>
- Robustness of Bayesian Sentinel-1 flood mapping: multi-event validation — 2025, Science of Remote Sensing — <https://www.sciencedirect.com/science/article/pii/S2666017225000161>
- Topographic index as prior for Sentinel-1 flood maps — 2023, Water (MDPI) — <https://www.mdpi.com/2073-4441/15/23/4034>
- DEM-resolution effects on coastal flood vulnerability — Fereshtehpour & Karamouz, 2018, WRR — <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2017WR022318>
- Modelling DEM errors in coastal flood inundation — Karamouz & Fereshtehpour, 2019, WRR — <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018WR024562>
- Evaluation of the NWM–HAND flood-mapping methodology — Johnson, Munasinghe, Eyelade & Cohen, 2019, NHESS — <https://nhess.copernicus.org/articles/19/2405/2019/>
- Citizen observations contributing to flood modelling — Assumpção, Popescu, Jonoski & Solomatine, 2018, HESS — <https://hess.copernicus.org/articles/22/1473/2018/>
- Improving hydrological models with crowdsourced data — Avellaneda et al., 2020, WRR — <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR026325>
- Crowdsourced social media for disaster management: PetaJakarta — Holderness & Turpin, 2018, CEUS — <https://www.sciencedirect.com/science/article/abs/pii/S0198971518301066>
- Satellite assessment of electricity restoration after Hurricane Maria — Román et al., 2019, PLOS ONE — <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0218883>
- Validation-gated multi-agent governance for online surrogate adaptation — Lim, Lee & Bang, 2026, arXiv — <https://arxiv.org/abs/2606.03321>

**Drainage optimisation, LID, sponge cities & recharge**

- Structural optimisation of urban drainage systems — 2024, Water (MDPI) — <https://www.mdpi.com/2073-4441/16/12/1696>
- Hydrograph-based storm-sewer design optimisation by GA — Afshar et al., 2005, Can. J. Civil Eng. — <https://cdnsciencepub.com/doi/10.1139/l05-121>
- Multi-objective optimisation of LID stormwater controls — 2018, J. Hydrology — <https://www.sciencedirect.com/science/article/abs/pii/S0022169418303214>
- SWMMLIDopt: LID optimisation tool for SWMM — 2024, J. Hydroinformatics — <https://iwaponline.com/jh/article/26/11/2781/105364/>
- Spatial optimisation of LID cost-effectiveness (SWMM-FTC) — 2022, J. Environmental Management — <https://www.sciencedirect.com/science/article/abs/pii/S0301479722001475>
- Review of Sponge City implementation in China — 2023, Water Science & Technology — <https://iwaponline.com/wst/article/88/10/2499/98252/>
- Enhanced Aquifer Recharge Using Stormwater: State of the Science — US EPA, 2021 — <https://www.gwpc.org/wp-content/uploads/2022/08/EAR-Using-Stormwater-State-of-Science-Review-USEPA2021.pdf>
- Lessons Learned from ASR Systems in the US — GWPC, 2022 — <https://www.gwpc.org/wp-content/uploads/2022/08/Lessons-Learned-from-Aquifer-Storage-and-Recovery-ASR-Systems-in-the-United-States.pdf>
- Floodwater recharge to offset groundwater depletion (Ramganga) — 2017, Sustainable Water Resources Management — <https://link.springer.com/article/10.1007/s40899-017-0168-6>
- Managing underground transfer of floods for irrigation (Ramganga) — 2019, J. Hydrology — <https://www.sciencedirect.com/science/article/abs/pii/S0022169419312533>
- Urban flood mitigation via deep aquifer recharge (Guadalajara) — Vanegas-Espinosa et al., 2022, IJERPH — <https://pmc.ncbi.nlm.nih.gov/articles/PMC8949559/>
- Atal Bhujal Yojana (Atal Jal) — CGWB / Ministry of Jal Shakti, 2020– — <https://www.cgwb.gov.in/en/participatory-ground-water-management-atal-bhujal-yojana>
- Legislation on Rainwater Harvesting (Chennai/Tamil Nadu) — CSE India, 2003 — <https://www.cseindia.org/legislation-on-rainwater-harvesting-1111>

**Further patents**

- CN109902801A — Bayesian NN flood ensemble forecasting — <https://patents.google.com/patent/CN109902801A/en>
- CN118692226A — AI-based flood analysis — <https://patents.google.com/patent/CN118692226A/en>
- CN110991776A — GRU water-level prediction — <https://patents.google.com/patent/CN110991776A/en>
- US12230116B2 — IoT smart-city flood early warning — <https://patents.google.com/patent/US12230116B2/en>
- US20160047663A1 — Evacuation navigation device — <https://patents.google.com/patent/US20160047663A1/en>
- US7349768B2 — Evacuation route planning tool — <https://patents.google.com/patent/US7349768B2/en>
- US9600997B1 — Localized flood alert system — <https://patents.google.com/patent/US9600997B1/en>
- US12094320B2 — Flood warning system — <https://patents.google.com/patent/US12094320B2/en>
- US20170193305A1 — Flash-flooding detection — <https://patents.google.com/patent/US20170193305A1/en>
- CN207924874U — Urban flood-control early-warning device — <https://patents.google.com/patent/CN207924874U/en>
- WO2018215746A1 — Manhole and sewer network — <https://patents.google.com/patent/WO2018215746A1/en>
- WO2020049310A1 — Smart sewer system — <https://patents.google.com/patent/WO2020049310A1/en>
- US9638334B2 — Flow-control gate for detention pond — <https://patents.google.com/patent/US9638334B2/en>
- US9011689B1 — Artificial recharge system — <https://patents.google.com/patent/US9011689B1/en>
- US7721799B2 — Flow-control packer & ASR system — <https://patents.google.com/patent/US7721799B2/en>
- WO2022118101A1 — Remote dam monitoring by satellite SAR — <https://patents.google.com/patent/WO2022118101A1/en>

---

## Where Varuna sits, in one paragraph

Every *ingredient* above has strong prior art, and two works — **Inunda** on the differentiable solver and the **2026 STOTEN MAR paper** on flood↔recharge coupling — are close enough that they must be cited and distinguished rather than treated as background. What no located publication or patent does is *assemble* them: a differentiable Bates-2010 PyTorch twin, a U-Net emulator, free global open satellite data, a self-supervised street-graph GNN with routing and zero-shot transfer, re-simulation-measured street-graph drainage, buildability-constrained detention siting, flood-to-aquifer recharge coupling, and a reward-gated citizen online-learning loop — served free-tier for data-scarce Indian cities. The four places novelty concentrates are the DEM-ensemble co-location diagnosis (V2), re-simulation-measured street-graph drainage with flood-safe non-river outfalls (V3), FloodGNN's five-ingredient combination (V4), and the reward-gated, audited citizen online-learning loop (V7), whose nearest analog is in nuclear surrogate modelling rather than geoscience.

*For the full sweep — ~99 items, per-theme comparison tables and the head-to-head novelty matrix — see [`noveltySearch.md`](noveltySearch.md) / [`noveltySearch.pdf`](noveltySearch.pdf).*
