"""Street graph -> learning-ready tensors.

Reuses `serve.roadnet.prepare` (same pruning + node order as the drain planner, so labels
computed there index straight into these tensors) and adds per-node / per-edge features.

Feature design rules (they carry the transfer story):
  * LOCAL — nothing identifies the city; a node only sees its own cell and neighbours.
  * PER-GRAPH STANDARDIZED where units are city-scale (elevation), raw physical units where
    they are already comparable across cities (pit-ness in metres, grade in %).
  * MISSING-TOLERANT — bundles without SAR frequency (any non-Patna city) get zeros, not NaNs.

`graph_tensors` is pure (RoadNet + feature grids in, tensors out) and offline-testable;
`build_graph` is the bundle driver that assembles the grids from dem/acc/jrc/urban rasters.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

from ..config import CFG

log = logging.getLogger("varuna.gnn.graph")

NODE_FEATURES = ("z_std", "pitness_m", "grade_pct", "log_acc", "jrc", "sar_freq",
                 "bld_3x3", "road_3x3", "built", "degree", "boundary_dist")
EDGE_FEATURES = ("length_km", "dz_m", "grade_pct")


class StreetGraph:
    """One area's street graph as tensors + the aux arrays routing and labelling need.

    Node indexing is COMPACT (dead/pruned RoadNet nodes removed); `compact_of` maps the
    original roadnet index -> compact index (-1 if dropped) so planner outputs computed on
    the raw RoadNet (e.g. drain paths) can be re-indexed onto these tensors.
    """

    def __init__(self, x, edge_index, edge_attr, lat, lon, cell, z, compact_of, dom=None):
        self.x = x                        # [N, F] float32
        self.edge_index = edge_index      # [2, E] int64, directed (both ways per street)
        self.edge_attr = edge_attr        # [E, 3] float32
        self.lat, self.lon = lat, lon     # [N] float64 (numpy)
        self.cell = cell                  # [N, 2] int64 domain cells (numpy)
        self.z = z                        # [N] float64 elevation (numpy)
        self.compact_of = compact_of      # [N_orig] int64, -1 where dropped (numpy)
        self.dom = dom                    # twin Domain (bundle driver only; None in pure tests)
        self.n = int(x.shape[0])
        self.e = int(edge_index.shape[1])
        self._adj = None

    def adjacency(self):
        """[(v, length_m), ...] per node, from the directed edge list — for Dijkstra routing."""
        if self._adj is None:
            adj = [[] for _ in range(self.n)]
            ei = self.edge_index.cpu().numpy()
            ln = (self.edge_attr[:, 0].cpu().numpy() * 1000.0)
            for k in range(ei.shape[1]):
                adj[int(ei[0, k])].append((int(ei[1, k]), float(ln[k])))
            self._adj = adj
        return self._adj

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        return self


def _box3(mask, ndimage):
    """3x3 box mean of a {0,1} grid — building/road density around a cell."""
    return ndimage.uniform_filter(np.asarray(mask, dtype="float64"), size=3, mode="nearest")


def graph_tensors(net, grids, device="cpu"):
    """Pure core: RoadNet + domain-grid feature rasters -> StreetGraph.

    net:   serve.roadnet.RoadNet (prepare() output — alive mask + adjacency + per-node cell)
    grids: dict of N x N float arrays on the twin domain grid:
           z (m, required), acc (flow accumulation, optional), jrc (0-100, optional),
           freq (SAR wet frequency 0-1, optional), bld / road ({0,1}, optional),
           built ({0,1}, optional). Missing optional grids become zero features.
    """
    from scipy import ndimage

    z_grid = np.asarray(grids["z"], dtype="float64")
    H, W = z_grid.shape
    dx = float(grids.get("dx", CFG.dx))
    gy, gx = np.gradient(z_grid, dx)
    grade_grid = np.hypot(gy, gx) * 100.0                       # % grade of the terrain

    zeros = np.zeros_like(z_grid)
    acc = np.log1p(np.clip(np.asarray(grids.get("acc", zeros), dtype="float64"), 0, None)) / 10.0
    jrc = np.clip(np.asarray(grids.get("jrc", zeros), dtype="float64"), 0, 100) / 100.0
    freq = np.clip(np.asarray(grids.get("freq", zeros), dtype="float64"), 0, 1)
    bld = _box3(grids.get("bld", zeros), ndimage)
    road = _box3(grids.get("road", zeros), ndimage)
    built = np.asarray(grids.get("built", zeros), dtype="float64")

    keep = [i for i in range(net.n) if net.alive[i]]
    compact_of = np.full(net.n, -1, dtype="int64")
    compact_of[keep] = np.arange(len(keep))
    n = len(keep)
    lat = net.lat[keep].astype("float64")
    lon = net.lon[keep].astype("float64")
    z = net.z[keep].astype("float64")
    cell = np.array([net.cell[i] for i in keep], dtype="int64")

    zmean, zstd = float(z.mean()), float(z.std() + 1e-6)        # per-graph -> transferable
    deg = np.array([len(net.adj[i]) for i in keep], dtype="float64")
    nbr_mean = np.array([np.mean([net.z[v] for v, _ in net.adj[i]]) if net.adj[i] else net.z[i]
                         for i in keep])
    pitness = np.clip(nbr_mean - z, -5.0, 5.0)                  # +ve = lower than neighbours

    rr, cc = cell[:, 0], cell[:, 1]
    bdist = np.minimum(np.minimum(rr, H - 1 - rr), np.minimum(cc, W - 1 - cc)) / (0.5 * H)

    x = np.stack([
        (z - zmean) / zstd,
        pitness,
        np.clip(grade_grid[rr, cc], 0, 20),
        acc[rr, cc],
        jrc[rr, cc],
        freq[rr, cc],
        bld[rr, cc],
        road[rr, cc],
        built[rr, cc],
        deg / 4.0,
        np.clip(bdist, 0, 1),
    ], axis=1).astype("float32")

    src, dst, eattr = [], [], []
    for i in keep:
        u = compact_of[i]
        for v_orig, length in net.adj[i]:
            v = compact_of[v_orig]
            if v < 0:
                continue
            dz = float(net.z[v_orig] - net.z[i])                # signed: dst above src = +ve
            src.append(u)
            dst.append(v)
            eattr.append((length / 1000.0, np.clip(dz, -5, 5),
                          np.clip(100.0 * dz / max(length, 1.0), -20, 20)))
    edge_index = torch.tensor([src, dst], dtype=torch.int64, device=device)
    edge_attr = torch.tensor(eattr, dtype=torch.float32, device=device)

    sg = StreetGraph(torch.tensor(x, device=device), edge_index, edge_attr,
                     lat, lon, cell, z, compact_of)
    log.info("street graph tensors: %d nodes, %d directed edges, %d features",
             sg.n, sg.e, x.shape[1])
    return sg


# --------------------------------------------------------------------------- bundle driver

_CACHE = {}


def cell_of_fn(work, dom):
    """(lat, lon) -> domain cell (r, c) or None — the same mapping plan_canals uses."""
    import rasterio
    with rasterio.open(f"{work}/dem.tif") as src:
        inv = ~src.transform

    def cell_of(lat, lon):
        col30, row30 = inv * (lon, lat)
        dr, dc = int((row30 - dom.row0) // 2), int((col30 - dom.col0) // 2)
        return (dr, dc) if (0 <= dr < dom.N and 0 <= dc < dom.N) else None

    return cell_of


def feature_grids(work, dom):
    """Assemble the domain-grid feature rasters from whatever the bundle has."""
    import rasterio
    from ..build.calibrate import _crop_to_domain

    grids = {"z": dom.z0.cpu().numpy(), "built": dom.built.cpu().numpy(), "dx": dom.dx}

    def crop(name, agg):
        path = f"{work}/{name}"
        if not os.path.exists(path):
            return None
        with rasterio.open(path) as s:
            return _crop_to_domain(np.nan_to_num(s.read(1)), dom, agg=agg).cpu().numpy()

    for key, fname, agg in (("acc", "acc.tif", "max"), ("jrc", "jrc_occurrence.tif", "mean"),
                            ("freq", "waterlogging_frequency.tif", "mean")):
        g = crop(fname, agg)
        if g is not None:
            grids[key] = g
    ug_path = f"{work}/urban_grid.npz"
    if os.path.exists(ug_path):
        ug = np.load(ug_path)
        grids["bld"] = ug["buildings"].astype("float64")
        grids["road"] = ug["roads"].astype("float64")
    return grids


def build_graph(work=None, device="cpu", use_cache=True):
    """Bundle -> StreetGraph (cached per work dir; the graph is static per bundle)."""
    from ..build.twin import build_domain
    from ..serve import roadnet

    work = work or CFG.work
    if use_cache and work in _CACHE:
        return _CACHE[work].to(device)
    graph = roadnet.load_road_graph(work)
    if graph is None:
        raise FileNotFoundError(f"no {roadnet.CACHE_FILE} in {work} — run "
                                f"scripts/build_road_graphs.py for this area first")
    dom = build_domain(work, device="cpu")
    cell_of = cell_of_fn(work, dom)
    net = roadnet.prepare(graph, dom.z0.cpu().numpy(), cell_of)
    sg = graph_tensors(net, feature_grids(work, dom), device=device)
    sg.dom = dom
    sg.cell_of = cell_of
    if use_cache:
        _CACHE[work] = sg
    return sg
