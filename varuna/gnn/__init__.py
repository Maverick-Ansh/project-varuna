"""Graph neural network over the street network — learned flood-aware path planning.

The serve layer already routes on graphs twice: storm drains via a reverse Dijkstra with a
hand-set uphill penalty (`serve.roadnet`), and evacuation implicitly via the exposure overlay.
Both need the raster emulator + per-geometry sampling for every rainfall, and the routing cost
(`length + 2400·climb`) is a fixed guess. This package replaces the guess with a *learned*
model of how water behaves on the street graph:

  * `graph`   — bundle -> tensors: OSM street nodes/edges with local topographic + urban
                features (elevation, pit-ness, slope, flow accumulation, buildings, roads,
                SAR waterlogging frequency). Features are local and per-graph standardized,
                so one model transfers across cities.
  * `model`   — FloodGNN: pure-PyTorch edge-conditioned message passing (no torch-geometric),
                FiLM-conditioned on rainfall so ONE network covers the whole storm range.
                Heads: per-node flood depth, per-edge flood depth, per-node drainage flow.
  * `dataset` — supervision from the twin: emulator depth sweeps + the street drain planner's
                flow accumulation, cached per bundle as gnn_data.npz.
  * `train`   — multi-area training with wet-weighted losses (sparse-flood lesson from the
                emulator fix) and held-out rainfall validation.
  * `planner` — what it is all for: flood-safe evacuation routing between any two points at
                any rainfall (GNN backend, honest emulator fallback) and GNN-scored drain
                corridors.
  * `evaluate`— AUC / wet-RMSE / flow rank-correlation / routing quality + speed, and the
                cross-city transfer experiments (train Patna-family -> test Bengaluru).

Everything here is plain torch + numpy + scipy: it runs on a free Colab/Kaggle T4, this
laptop's CPU, and the deployed HF Space unchanged.
"""
