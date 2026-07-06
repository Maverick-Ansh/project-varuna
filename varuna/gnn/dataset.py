"""Supervision for FloodGNN, generated from the twin — no hand labels anywhere.

Two label families, cached per bundle as `gnn_data.npz`:

  DEPTH (the routing signal): the U-Net emulator's max-depth grid for a sweep of storms,
  sampled at every street node's cell and along every street edge (endpoints + midpoint —
  edges are ~30–100 m vs 60 m cells, so three samples cover the segment). The GNN learns to
  reproduce the raster emulator *on the graph*, which is what makes it portable: the raster
  needs this city's DEM window; the graph features travel.

  FLOW (the planning signal): run the street drain planner (`serve.canals.route_network`,
  the same reverse-Dijkstra spiderweb the dashboard ships) at several (rain, n_inlets)
  configurations and accumulate how many m³ each street node conveys. log1p-compressed —
  trunk nodes carry ~10⁶ m³, twigs ~10³. The GNN's flow head learns *where water wants to
  go* without running Dijkstra, and transfer tests ask whether that knowledge is city-free.

Emulator flood (not the physics sim) also places the planner's inlets here: it is available
in every bundle, millisecond-fast, and matches what the deployed planner sees. Honest note:
labels inherit the emulator's error vs the twin (val RMSE ~5–21 cm per bundle).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from ..config import CFG
from .graph import build_graph

log = logging.getLogger("varuna.gnn.dataset")

CACHE_FILE = "gnn_data.npz"
RAINS = tuple(float(r) for r in range(20, 241, 20))            # 12 storms, 20..240 mm
FLOW_CFGS = ((60.0, 25), (100.0, 40), (150.0, 60))             # (rain_mm, n_inlets)


def edge_sample_cells(sg):
    """[E, 3, 2] domain cells per directed edge: src cell, midpoint cell, dst cell."""
    ei = sg.edge_index.cpu().numpy()
    cu = sg.cell[ei[0]]
    cv = sg.cell[ei[1]]
    mid = cu.copy()
    cell_of = getattr(sg, "cell_of", None)
    if cell_of is not None:
        for k in range(ei.shape[1]):
            m = cell_of((sg.lat[ei[0, k]] + sg.lat[ei[1, k]]) / 2.0,
                        (sg.lon[ei[0, k]] + sg.lon[ei[1, k]]) / 2.0)
            if m is not None:
                mid[k] = m
    return np.stack([cu, mid, cv], axis=1)


def sample_depths(hmax, sg, esc=None):
    """Emulator max-depth grid -> (node_depth [N], edge_depth [E]) in metres. Pure."""
    hmax = np.asarray(hmax, dtype="float32")
    node = hmax[sg.cell[:, 0], sg.cell[:, 1]]
    esc = edge_sample_cells(sg) if esc is None else esc
    edge = hmax[esc[..., 0], esc[..., 1]].max(axis=1)
    return node, edge


def _flow_labels(work, sg, rain_mm, n_inlets):
    """Per-node drained m³ from the street drain planner at one configuration (zeros if the
    planner falls back to grid routing — e.g. no street outfall in reach)."""
    import torch
    from ..build.twin import candidate_sites
    from ..serve import canals, roadnet
    from ..serve.emulator import whatif_grid

    dom = sg.dom
    hmax, _dig, _s = whatif_grid(rain_mm, None, work=work, device="cpu")
    flood_np = (np.clip(hmax - 0.15, 0, None) * dom.built.cpu().numpy()).astype("float64")

    weight = None
    freq_grid = getattr(sg, "_freq_grid", None)
    if freq_grid is not None:
        weight = 1.0 + 2.0 * freq_grid                          # same priority plan_canals uses

    with torch.no_grad():
        sites, _m, _a, _e = candidate_sites(dom, work)
    routed = canals.route_network(
        dom.z0.cpu().numpy(), flood_np, roadnet.load_road_graph(work), sg.cell_of,
        lowland_cells=canals._lowland_cells(dom, work),
        boundary_cells=canals._lowest_boundary_cells(dom, k=5),
        pit_cells=list(sites), n_inlets=int(n_inlets), weight=weight, dx=dom.dx)

    flow = np.zeros(sg.n, dtype="float64")
    if routed is None:
        log.warning("flow labels @%.0f mm/%d inlets: planner fell back to grid — zeros", rain_mm, n_inlets)
        return flow
    for c in routed[0]:                                         # per-inlet street paths
        for orig in c["node_path"]:
            j = sg.compact_of[orig]
            if j >= 0:
                flow[j] += c["drained_m3"]
    return flow


def build_dataset(work=None, rains=RAINS, flow_cfgs=FLOW_CFGS, out=CACHE_FILE, device="cpu"):
    """Generate + cache all labels for one bundle. Returns a small summary dict."""
    from ..serve.emulator import whatif_grid

    work = work or CFG.work
    t0 = time.perf_counter()
    sg = build_graph(work, device="cpu")
    # stash the freq grid for label parity with plan_canals (feature col 5 = sar_freq)
    from .graph import feature_grids
    sg._freq_grid = feature_grids(work, sg.dom).get("freq")

    esc = edge_sample_cells(sg)
    node_depth = np.zeros((len(rains), sg.n), dtype="float16")
    edge_depth = np.zeros((len(rains), sg.e), dtype="float16")
    for i, r in enumerate(rains):
        hmax, _dig, _s = whatif_grid(float(r), None, work=work, device=device)
        nd, ed = sample_depths(hmax, sg, esc=esc)
        node_depth[i], edge_depth[i] = nd.astype("float16"), ed.astype("float16")

    node_flow = np.zeros((len(flow_cfgs), sg.n), dtype="float32")
    for i, (r, k) in enumerate(flow_cfgs):
        node_flow[i] = _flow_labels(work, sg, float(r), int(k))
        log.info("flow labels %d/%d: %.0f mm / %d inlets -> %d wet nodes",
                 i + 1, len(flow_cfgs), r, k, int((node_flow[i] > 0).sum()))

    path = os.path.join(work, out)
    np.savez_compressed(
        path, rains=np.asarray(rains, dtype="float32"),
        node_depth=node_depth, edge_depth=edge_depth,
        flow_rain=np.asarray([c[0] for c in flow_cfgs], dtype="float32"),
        flow_inlets=np.asarray([c[1] for c in flow_cfgs], dtype="int64"),
        node_flow=node_flow, n_nodes=sg.n, n_edges=sg.e)
    summary = dict(
        work=work, n_nodes=sg.n, n_edges=sg.e, rains=list(rains),
        flow_cfgs=[list(c) for c in flow_cfgs],
        wet_node_frac_100mm=round(float((node_depth[np.argmin(np.abs(np.asarray(rains) - 100.0))]
                                         > 0.15).mean()), 4),
        seconds=round(time.perf_counter() - t0, 1), file=path)
    log.info("gnn dataset -> %s (%d nodes, %d edges, %.1fs)", path, sg.n, sg.e, summary["seconds"])
    return summary


def load_dataset(work=None):
    """(StreetGraph, npz dict) for a bundle — raises with a clear message if not generated."""
    work = work or CFG.work
    path = os.path.join(work, CACHE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {CACHE_FILE} in {work} — run varuna.gnn.dataset.build_dataset "
                                f"(or scripts/run_gnn_colab.py --data)")
    sg = build_graph(work, device="cpu")
    data = dict(np.load(path))
    if int(data["n_nodes"]) != sg.n or int(data["n_edges"]) != sg.e:
        raise RuntimeError(f"{path} was built for a different graph "
                           f"({int(data['n_nodes'])}x{int(data['n_edges'])} vs {sg.n}x{sg.e}) — regenerate")
    return sg, data
