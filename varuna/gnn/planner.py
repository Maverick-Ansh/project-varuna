"""Learned flood-aware path planning on the street graph.

`safe_route` answers the question the exposure panel only gestures at: *"how do I actually
get from A to B during an r-mm storm?"* It runs Dijkstra over the real OSM street graph with
edge costs inflated by predicted flood depth, and reports the dry-ignorant shortest path next
to it so the detour is honest.

Two interchangeable risk backends:
  gnn      — FloodGNN edge-depth head: one forward pass covers every street at any rainfall,
             works on any bundle with a road graph (even one the model never trained on).
  emulator — the U-Net raster sampled along each edge (exactly how training labels are made).
             Always available; it is the fallback until a checkpoint ships in the bundle, and
             the oracle the GNN is judged against in evaluate.py.

`drain_corridors` exposes the flow head: streets ranked by how much water the drain planner
would send along them — GNN-guessed trunk lines without running the planner.

Checkpoint discovery: <bundle>/gnn.pt, then $VARUNA_GNN, then <repo>/artifacts/gnn/gnn.pt.
"""
from __future__ import annotations

import heapq
import logging
import os

import numpy as np
import torch

from ..config import CFG
from .graph import build_graph

log = logging.getLogger("varuna.gnn.planner")

FLOOD_TAU = 0.15          # m; an edge deeper than this counts as flooded in reports
AVOID_DEPTH = 0.30        # m; edges deeper than this are near-forbidden (cost x25)
RISK_WEIGHT = 8.0
_MODELS = {}


def _checkpoint_path(work):
    for p in (os.path.join(work, "gnn.pt"), os.environ.get("VARUNA_GNN") or "",
              os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                  os.path.abspath(__file__)))), "artifacts", "gnn", "gnn.pt")):
        if p and os.path.exists(p):
            return p
    return None


def load_gnn(work=None, device="cpu"):
    """(model, checkpoint_path) or (None, None) when no checkpoint is discoverable."""
    from .model import load_checkpoint
    path = _checkpoint_path(work or CFG.work)
    if path is None:
        return None, None
    if path not in _MODELS:
        model, _ckpt = load_checkpoint(path, device=device)
        _MODELS[path] = model
    return _MODELS[path], path


def edge_depths(work=None, rain_mm=100.0, backend="auto", sg=None, device="cpu"):
    """Per-directed-edge flood depth [E] + the backend that produced it ('gnn'|'emulator')."""
    work = work or CFG.work
    sg = sg or build_graph(work, device=device)
    if backend in ("auto", "gnn"):
        model, path = load_gnn(work, device=device)
        if model is not None:
            with torch.no_grad():
                out = model(sg.x, sg.edge_index, sg.edge_attr, float(rain_mm))
            return out["edge_depth"].cpu().numpy().astype("float64"), "gnn"
        if backend == "gnn":
            raise FileNotFoundError("no FloodGNN checkpoint found (bundle gnn.pt / $VARUNA_GNN "
                                    "/ artifacts/gnn/gnn.pt) — train one or use backend='auto'")
    from ..serve.emulator import whatif_grid
    from .dataset import sample_depths
    hmax, _dig, _s = whatif_grid(float(rain_mm), None, work=work, device=device)
    _node, edge = sample_depths(hmax, sg)
    return edge.astype("float64"), "emulator"


def _nearest_node(sg, lat, lon, max_m=800.0):
    """Nearest street node to a clicked point, or None beyond max_m."""
    coslat = np.cos(np.radians(lat))
    d2 = ((sg.lat - lat) ** 2 + ((sg.lon - lon) * coslat) ** 2)
    i = int(np.argmin(d2))
    if np.sqrt(d2[i]) * 111320.0 > max_m:
        return None
    return i


def _dijkstra(n, adj, s, t):
    """Least-cost path s->t over adj[u] = [(v, cost, edge_id)] -> (node path, edge ids)."""
    dist = np.full(n, np.inf)
    came = {}
    dist[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > dist[u]:
            continue
        for v, c, k in adj[u]:
            nd = d + c
            if nd < dist[v]:
                dist[v] = nd
                came[v] = (u, k)
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[t]):
        return None, None
    path, edges, u = [t], [], t
    while u != s:
        u, k = came[u]
        path.append(u)
        edges.append(k)
    return path[::-1], edges[::-1]


def _route_stats(sg, lengths, depth, node_path, edge_ids):
    d = depth[edge_ids] if len(edge_ids) else np.zeros(0)
    return dict(
        length_m=round(float(lengths[edge_ids].sum())),
        flooded_edges=int((d > FLOOD_TAU).sum()),
        wet_length_m=round(float(lengths[edge_ids][d > FLOOD_TAU].sum())) if len(edge_ids) else 0,
        max_depth_m=round(float(d.max()), 2) if len(d) else 0.0,
        path_latlon=[[round(float(sg.lat[i]), 6), round(float(sg.lon[i]), 6)] for i in node_path],
    )


def safe_route(start, end, rain_mm=100.0, work=None, backend="auto",
               avoid_depth_m=AVOID_DEPTH, risk_weight=RISK_WEIGHT, device="cpu"):
    """Flood-aware route start->end ([lat, lon] each) vs the flood-ignorant shortest path.

    Cost per street edge = length x (1 + risk_weight·min(depth, 1.5) + 25 if depth > avoid).
    Water above `avoid_depth_m` is only crossed when there is genuinely no drier way — the
    route never refuses; it reports max depth so the caller can warn instead.
    """
    work = work or CFG.work
    sg = build_graph(work, device=device)
    depth, used = edge_depths(work, rain_mm, backend=backend, sg=sg, device=device)

    s = _nearest_node(sg, float(start[0]), float(start[1]))
    t = _nearest_node(sg, float(end[0]), float(end[1]))
    if s is None or t is None:
        raise ValueError("start/end is >800 m from any mapped street in this area")
    if s == t:
        raise ValueError("start and end snap to the same street node — pick points further apart")

    ei = sg.edge_index.cpu().numpy()
    lengths = sg.edge_attr[:, 0].cpu().numpy() * 1000.0
    mult = 1.0 + risk_weight * np.minimum(depth, 1.5) + 25.0 * (depth > avoid_depth_m)
    adj_safe = [[] for _ in range(sg.n)]
    adj_short = [[] for _ in range(sg.n)]
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        adj_safe[u].append((v, float(lengths[k] * mult[k]), k))
        adj_short[u].append((v, float(lengths[k]), k))

    p_safe, e_safe = _dijkstra(sg.n, adj_safe, s, t)
    p_short, e_short = _dijkstra(sg.n, adj_short, s, t)
    if p_safe is None:
        raise ValueError("no street route connects these points inside the domain")

    safe = _route_stats(sg, lengths, depth, p_safe, e_safe)
    short = _route_stats(sg, lengths, depth, p_short, e_short)
    return dict(
        rain_mm=float(rain_mm), backend=used,
        start_snapped=[round(float(sg.lat[s]), 6), round(float(sg.lon[s]), 6)],
        end_snapped=[round(float(sg.lat[t]), 6), round(float(sg.lon[t]), 6)],
        route=safe, shortest=short,
        detour_pct=round(100.0 * (safe["length_m"] / max(short["length_m"], 1) - 1.0), 1),
        avoided_wet_m=max(0, short["wet_length_m"] - safe["wet_length_m"]),
        note="Learned street-flood risk (%s backend); depth > %.2f m near-forbidden. "
             "Planning aid, not a life-safety certification." % (used, avoid_depth_m),
    )


def drain_corridors(rain_mm=100.0, work=None, top_k=30, device="cpu"):
    """GNN flow head -> the streets predicted to carry the most drainage (trunk corridors).

    Requires a checkpoint (this head has no raster fallback). Edge score = the smaller of its
    endpoint flows — both ends of a trunk segment carry water.
    """
    work = work or CFG.work
    sg = build_graph(work, device=device)
    model, path = load_gnn(work, device=device)
    if model is None:
        raise FileNotFoundError("drain_corridors needs a trained FloodGNN checkpoint")
    with torch.no_grad():
        out = model(sg.x, sg.edge_index, sg.edge_attr, float(rain_mm))
    flow = out["node_flow"].cpu().numpy()                      # log1p(m³)
    ei = sg.edge_index.cpu().numpy()
    undirected = ei[0] < ei[1]
    score = np.minimum(flow[ei[0]], flow[ei[1]]) * undirected
    order = np.argsort(-score)[:top_k]
    smax = float(score[order[0]]) if len(order) else 1.0
    return dict(
        rain_mm=float(rain_mm), checkpoint=path,
        corridors=[dict(
            latlon=[[round(float(sg.lat[ei[0, k]]), 6), round(float(sg.lon[ei[0, k]]), 6)],
                    [round(float(sg.lat[ei[1, k]]), 6), round(float(sg.lon[ei[1, k]]), 6)]],
            score=round(float(score[k] / max(smax, 1e-9)), 3),
            flow_m3=round(float(np.expm1(min(flow[ei[0, k]], flow[ei[1, k]])))))
            for k in order],
        note="log-flow score from the GNN flow head; compare with canal_plan.json trunks.",
    )


def status(work=None):
    """What the route endpoint would use right now — for the dashboard badge."""
    work = work or CFG.work
    model, path = load_gnn(work)
    has_graph = True
    try:
        from ..serve import roadnet
        has_graph = roadnet.load_road_graph(work) is not None
    except Exception:  # noqa: BLE001
        has_graph = False
    return dict(road_graph=has_graph, checkpoint=path,
                backend="gnn" if model is not None else "emulator")
