"""OSM road-network storm drains — fetch real streets once, route drainage along them forever.

The v2 canal engine routes least-cost corridors over the 60 m DEM grid, merely *preferring* road
cells — the result is a blocky line that ignores actual street geometry. This module upgrades
routing to the real OSM road network so the planned storm drains follow streets you can see:

  * FETCH ONCE — a plain Overpass API query (no osmnx/geopandas; `requests` only, three public
    mirrors) pulls every drivable street in the domain. The graph is cached in the bundle as
    `road_graph.json.gz`, so tests, the deployed Space and re-plans never touch the network.
  * REAL GRAPH — nodes are OSM way vertices (full street geometry), edges consecutive vertex
    pairs with haversine lengths; nodes carry the DEM elevation of their grid cell.
  * ONE REVERSE DIJKSTRA — a single multi-source pass from all outfalls (river / lowest boundary /
    storage pits) labels every street node with its cheapest gravity route out; the flow-direction
    cost `length + 2400·max(0, climb)` is the grid router's `dx·(1+40·Δz)` expressed per metre.
  * SPIDERWEB — many inlets (deepest cells of each flood basin) each follow their `succ` pointer
    chain; the union of chains is a forest whose shared trunks accumulate the drained volume of
    every branch — trunk vs twig is reported per segment for flow-weighted rendering.

Only geometry lives here. Carving, re-simulation and the honest before/after measurement stay in
`serve.canals` (the drawn web is finer than the 60 m cells it is carved into — disclosed there).
"""
from __future__ import annotations

import gzip
import heapq
import json
import logging
import math
import os
from collections import defaultdict, deque

import numpy as np

from ..config import CFG

log = logging.getLogger("varuna.serve.roadnet")

CACHE_FILE = "road_graph.json.gz"
ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                "residential", "living_street", "service",
                "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link")
OVERPASS_MIRRORS = ("https://overpass-api.de/api/interpreter",
                    "https://overpass.kumi.systems/api/interpreter",
                    "https://maps.mail.ru/osm/tools/overpass/api/interpreter")


# --------------------------------------------------------------------------- fetch + persist


def fetch_overpass(bbox, classes=ROAD_CLASSES, timeout=90, mirrors=OVERPASS_MIRRORS):
    """Raw Overpass JSON for every way of the given highway classes in bbox=(south,west,north,east).

    `out geom` returns each way's node ids AND vertex lat/lons in one request. Footways/paths are
    excluded by default: storm drains are trenched along streets, and paths only add clutter.
    """
    import requests
    s, w, n, e = bbox
    query = ('[out:json][timeout:%d];way["highway"~"^(%s)$"](%f,%f,%f,%f);out geom;'
             % (timeout, "|".join(classes), s, w, n, e))
    last = None
    for url in mirrors:
        try:
            r = requests.post(url, data={"data": query}, timeout=timeout + 30,
                              headers={"User-Agent": "varuna-floodtwin/1.0 "
                                       "(github.com/Maverick-Ansh/project-varuna)"})
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("overpass mirror failed (%s): %s", url, exc)
            last = exc
    raise RuntimeError(f"all Overpass mirrors failed (last: {last}); "
                       f"use the bundle's cached {CACHE_FILE} instead")


def graph_from_overpass(osm, bbox=None, classes=ROAD_CLASSES):
    """Overpass JSON -> the persisted artifact: deduped nodes + ways as node-index chains."""
    idx, nodes, ways = {}, [], []
    for el in osm.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el or len(el.get("nodes", [])) < 2:
            continue
        chain = []
        for nid, pt in zip(el["nodes"], el["geometry"]):
            if nid not in idx:
                idx[nid] = len(nodes)
                nodes.append([round(pt["lat"], 6), round(pt["lon"], 6)])
            chain.append(idx[nid])
        ways.append(chain)
    return {"bbox": list(bbox) if bbox else None, "classes": list(classes),
            "nodes": nodes, "ways": ways}


def save_road_graph(work, graph):
    path = os.path.join(work, CACHE_FILE)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(graph, f, separators=(",", ":"))
    log.info("road graph cached -> %s (%d nodes, %d ways)", path,
             len(graph["nodes"]), len(graph["ways"]))
    return path


def load_road_graph(work=None):
    """The bundle's cached road graph, or None if it was never fetched."""
    path = os.path.join(work or CFG.work, CACHE_FILE)
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_road_graph(work=None, margin=0.002):
    """Fetch + cache the road graph for a bundle's domain (the only step needing internet)."""
    import torch
    import rasterio
    work = work or CFG.work
    meta = torch.load(f"{work}/twin_meta.pt", map_location="cpu", weights_only=False)
    row0, col0, N = meta["row0"], meta["col0"], meta["n_grid"]
    with rasterio.open(f"{work}/dem.tif") as s:
        T = s.transform
    lon0, lat_top = T * (col0, row0)
    lon1, lat_bot = T * (col0 + N * 2, row0 + N * 2)
    bbox = (min(lat_top, lat_bot) - margin, min(lon0, lon1) - margin,
            max(lat_top, lat_bot) + margin, max(lon0, lon1) + margin)
    graph = graph_from_overpass(fetch_overpass(bbox), bbox=bbox)
    if not graph["ways"]:
        raise RuntimeError(f"Overpass returned no roads for bbox {bbox}")
    save_road_graph(work, graph)
    return graph


# --------------------------------------------------------------------------- geometry helpers


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _bresenham(a, b):
    """Grid cells on the integer line a->b inclusive (the inlet->street snap bridge)."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 >= r0 else -1), (1 if c1 >= c0 else -1)
    err, r, c = dr - dc, r0, c0
    cells = []
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            return cells
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


# --------------------------------------------------------------------------- routable street net


class RoadNet:
    """One domain's street graph, ready to route: geometry + DEM elevation + adjacency.

    Built at plan time from the cached artifact (`prepare`); not persisted itself.
    """

    def __init__(self, lat, lon, z, cell, adj, alive, cell_index, cell_of):
        self.lat, self.lon, self.z = lat, lon, z
        self.cell = cell                  # per-node domain cell (r, c) or None
        self.adj = adj                    # per-node [(neighbor, length_m), ...]
        self.alive = alive                # in-domain AND in the largest connected component
        self.cell_index = cell_index      # (r, c) -> [node, ...] for snapping
        self.cell_of = cell_of            # (lat, lon) -> (r, c) | None
        self.n = len(lat)


def prepare(graph, z_np, cell_of):
    """Build a routable RoadNet from the persisted graph.

    cell_of(lat, lon) -> domain cell (r, c) or None keeps this testable without rasterio;
    the bundle driver passes the dem.tif inverse-transform version. Nodes outside the domain
    are pruned (ways crossing the boundary break there) and only the largest connected
    component is kept — fragments can't reach any outfall anyway.
    """
    pts = graph["nodes"]
    n = len(pts)
    lat = np.array([p[0] for p in pts], dtype="float64")
    lon = np.array([p[1] for p in pts], dtype="float64")
    cell = [cell_of(p[0], p[1]) for p in pts]
    adj = [[] for _ in range(n)]
    seen = set()
    for chain in graph["ways"]:
        for a, b in zip(chain, chain[1:]):
            if a == b or cell[a] is None or cell[b] is None:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            length = haversine_m(lat[a], lon[a], lat[b], lon[b])
            adj[a].append((b, length))
            adj[b].append((a, length))

    comp = np.full(n, -1, dtype="int64")                     # largest connected component
    best_root, best_size = -1, 0
    for s in range(n):
        if comp[s] >= 0 or not adj[s]:
            continue
        comp[s] = s
        size, q = 1, deque([s])
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if comp[v] < 0:
                    comp[v] = s
                    size += 1
                    q.append(v)
        if size > best_size:
            best_root, best_size = s, size
    if best_root < 0:
        raise RuntimeError("road graph has no in-domain edges")
    alive = comp == best_root
    dropped = int((~alive & (comp >= 0)).sum())
    if dropped:
        log.info("road net: kept component of %d nodes, dropped %d fragment nodes",
                 best_size, dropped)
    cell_index = defaultdict(list)
    z = np.zeros(n, dtype="float64")
    for i in range(n):
        if alive[i]:
            cell_index[cell[i]].append(i)
            z[i] = float(z_np[cell[i]])
        else:
            adj[i] = []
    return RoadNet(lat, lon, z, cell, adj, alive, dict(cell_index), cell_of)


def snap(net, cell, max_cells=3):
    """Nearest street node within max_cells (~180 m) of a grid cell, else None."""
    r0, c0 = int(cell[0]), int(cell[1])
    best, best_d2 = None, None
    for dr in range(-max_cells, max_cells + 1):
        for dc in range(-max_cells, max_cells + 1):
            for i in net.cell_index.get((r0 + dr, c0 + dc), ()):
                d2 = dr * dr + dc * dc
                if best is None or d2 < best_d2:
                    best, best_d2 = i, d2
    return best


def reverse_dijkstra(net, outfall_nodes, uphill_penalty_m=2400.0):
    """One multi-source pass from all outfalls: per-node cost + next hop toward the cheapest one.

    Flow-direction cost of draining y through x (x one hop nearer the outfall):
    length + uphill_penalty_m * max(0, z[x] - z[y]) — i.e. the grid router's
    dx*(1+40*climb) per metre, so ~2.4 km of flat street trades against 1 m of climb.
    """
    dist = np.full(net.n, np.inf)
    succ = np.full(net.n, -1, dtype="int64")
    pq = []
    for o in set(int(i) for i in outfall_nodes):
        if net.alive[o]:
            dist[o] = 0.0
            heapq.heappush(pq, (0.0, o))
    while pq:
        d, x = heapq.heappop(pq)
        if d > dist[x]:
            continue
        for y, length in net.adj[x]:
            nd = d + length + uphill_penalty_m * max(0.0, net.z[x] - net.z[y])
            if nd < dist[y]:
                dist[y] = nd
                succ[y] = x
                heapq.heappush(pq, (nd, y))
    return dist, succ


def inlet_path(succ, node):
    """Follow the next-hop chain from an inlet node to its outfall -> [node, ...]."""
    path, u = [int(node)], int(node)
    while succ[u] >= 0:
        u = int(succ[u])
        path.append(u)
        if len(path) > len(succ):                            # cycle guard (cannot happen)
            raise RuntimeError("succ chain does not terminate")
    return path


def nodes_to_cells(net, node_path, inlet_cell=None, step_deg=0.00027, z_np=None):
    """Street polyline -> deduped domain-cell path, 4-connected so the carved trench conveys.

    Densifies each edge at ~30 m (half a cell), prepends a straight bridge from the inlet's
    flood cell to its snapped street node, and inserts the lower of the two corner cells at
    diagonal steps (face-flow simulators don't pass water through cell corners).
    """
    cells = []

    def push(rc):
        if rc is None:
            return
        if cells and cells[-1] == rc:
            return
        if cells:                                            # 4-connect diagonal steps
            r0, c0 = cells[-1]
            r1, c1 = rc
            if abs(r1 - r0) == 1 and abs(c1 - c0) == 1:
                via = (r0, c1), (r1, c0)
                if z_np is not None:
                    via = sorted(via, key=lambda rc_: float(z_np[rc_]))
                cells.append(via[0])
        cells.append(rc)

    if inlet_cell is not None:
        first = net.cell[node_path[0]]
        for rc in _bresenham((int(inlet_cell[0]), int(inlet_cell[1])), first):
            push(rc)
    else:
        push(net.cell[node_path[0]])
    for a, b in zip(node_path, node_path[1:]):
        la1, lo1, la2, lo2 = net.lat[a], net.lon[a], net.lat[b], net.lon[b]
        k = max(2, int(max(abs(la2 - la1), abs(lo2 - lo1)) / step_deg) + 1)
        for f in np.linspace(0.0, 1.0, k)[1:]:
            push(net.cell_of(la1 + f * (la2 - la1), lo1 + f * (lo2 - lo1)))
    return cells


def build_network(net, inlets, succ, dist):
    """Union of inlet drain paths -> spiderweb segments with accumulated drained volume.

    inlets: [{node, cell, drained_m3, basin}, ...] (already routed: dist[node] < inf).
    Each inlet's volume is added along its whole chain; volumes therefore change exactly at
    junction nodes (in-degree >= 2) and at inlet nodes sitting on another inlet's path, so
    splitting chains at those heads yields segments of constant flow — trunk edges carry the
    sum of every branch upstream, which is what the flow-weighted rendering needs.
    """
    edge_drain = defaultdict(float)
    used_in = defaultdict(int)
    paths = {}
    for inlet in inlets:
        p = inlet_path(succ, inlet["node"])
        paths[inlet["node"]] = p
        for u, v in zip(p, p[1:]):
            if edge_drain[(u, v)] == 0.0:
                used_in[v] += 1
            edge_drain[(u, v)] += inlet["drained_m3"]

    heads = set(i["node"] for i in inlets) | set(v for v, k in used_in.items() if k >= 2)
    elen = {}
    for (u, v) in edge_drain:
        for w, length in net.adj[u]:
            if w == v:
                elen[(u, v)] = length
                break

    segments, done = [], set()
    for h in sorted(heads):
        u = h
        while succ[u] >= 0 and (u, int(succ[u])) in edge_drain and (u, int(succ[u])) not in done:
            seg_nodes, length = [u], 0.0
            drained = edge_drain[(u, int(succ[u]))]
            while succ[u] >= 0:
                v = int(succ[u])
                done.add((u, v))
                length += elen[(u, v)]
                seg_nodes.append(v)
                u = v
                if v in heads or succ[v] < 0:
                    break
            segments.append({
                "nodes": seg_nodes, "drained_m3": drained, "length_m": length,
                "path_latlon": [[round(float(net.lat[i]), 6), round(float(net.lon[i]), 6)]
                                for i in seg_nodes],
            })
    return {"segments": segments, "paths": paths,
            "total_length_m": sum(s["length_m"] for s in segments)}
