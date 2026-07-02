"""Exposure / evacuation view — who and what floods, and which roads stay dry.

Overlays OSM building footprints and the road network on the twin's flood-depth grid for a storm,
so the dashboard can show *at-risk buildings* and *roads that stay passable* (evacuation routes).

Needs osmnx (an OSM/Overpass query). It is lazy-imported and the API returns 503 where osmnx is
absent, so the rest of the dashboard is unaffected. Flood depth comes from the per-area emulator
(millisecond), so this is cheap once OSM data is fetched.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from ..config import CFG

log = logging.getLogger("varuna.serve.exposure")

TAU = 0.15               # m; depth above which a cell counts as flooded
MAX_FEATURES = 1500      # cap returned geometries so the payload stays light
CACHE_FILE = "exposure.json"   # per-bundle cache; lets osmnx-less deploys serve exposure


def _flood_grid(rain_mm, work, device="cpu"):
    """Per-cell max flood depth over the domain (emulator), plus the domain for geo-referencing."""
    from .emulator import load_emulator, whatif_grid
    hmax, _dig, _summary = whatif_grid(rain_mm, None, work=work, device=device)
    dom = load_emulator(work, device)["dom"]
    return hmax, dom


def _depth_sampler(work, hmax, dom):
    """Return depth_at(lat, lon) -> flood depth (m) at that point, or None if outside the domain."""
    import rasterio
    with rasterio.open(f"{work}/dem.tif") as s:
        T = s.transform
    inv = ~T
    N, row0, col0 = dom.N, dom.row0, dom.col0

    def depth_at(lat, lon):
        col30, row30 = inv * (lon, lat)              # 30 m grid indices
        dr, dc = int((row30 - row0) // 2), int((col30 - col0) // 2)
        if 0 <= dr < N and 0 <= dc < N:
            return float(hmax[dr, dc])
        return None

    return depth_at


def _domain_bbox(work, dom):
    """(north, south, east, west) lat/lon of the domain, for the OSM query."""
    import rasterio
    with rasterio.open(f"{work}/dem.tif") as s:
        T = s.transform
    lon0, lat_top = T * (dom.col0, dom.row0)
    lon1, lat_bot = T * (dom.col0 + dom.N * 2, dom.row0 + dom.N * 2)
    return max(lat_top, lat_bot), min(lat_top, lat_bot), max(lon0, lon1), min(lon0, lon1)


def _osm_features(north, south, east, west, tags):
    """osmnx features query, tolerant of the 1.x/2.x API rename."""
    import osmnx as ox
    try:                                             # osmnx >= 2.0
        return ox.features_from_bbox((west, south, east, north), tags)
    except Exception:
        pass
    try:                                             # osmnx 1.x positional
        return ox.features_from_bbox(north, south, east, west, tags)
    except Exception:                                # older name
        return ox.geometries_from_bbox(north, south, east, west, tags)


def assess_exposure(rain_mm=None, work=None, center=None, device="cpu"):
    """At-risk buildings + flooded/dry roads for a storm. center is accepted for API parity."""
    work = work or CFG.work
    rain_mm = CFG.design_rain_mm if rain_mm is None else float(rain_mm)
    hmax, dom = _flood_grid(rain_mm, work, device)
    depth_at = _depth_sampler(work, hmax, dom)
    north, south, east, west = _domain_bbox(work, dom)

    buildings, roads_flooded, roads_dry = [], [], []
    n_buildings = n_at_risk = 0

    try:
        b = _osm_features(north, south, east, west, {"building": True})
        for geom in b.geometry:
            c = geom.centroid
            d = depth_at(c.y, c.x)
            if d is None:
                continue
            n_buildings += 1
            if d > TAU:
                n_at_risk += 1
                if len(buildings) < MAX_FEATURES:
                    buildings.append({"lat": round(c.y, 6), "lon": round(c.x, 6),
                                      "depth_m": round(d, 2)})
    except Exception as e:  # noqa: BLE001
        log.warning("building exposure failed: %s", e)

    n_roads = n_flooded_roads = 0
    try:
        r = _osm_features(north, south, east, west, {"highway": True})
        for geom in r.geometry:
            coords = _line_coords(geom)
            if not coords:
                continue
            n_roads += 1
            depths = [depth_at(lat, lon) for lat, lon in coords]
            depths = [d for d in depths if d is not None]
            if not depths:
                continue
            latlon = [[round(lat, 6), round(lon, 6)] for lat, lon in coords]
            if max(depths) > TAU:
                n_flooded_roads += 1
                if len(roads_flooded) < MAX_FEATURES:
                    roads_flooded.append(latlon)
            elif len(roads_dry) < MAX_FEATURES:
                roads_dry.append(latlon)
    except Exception as e:  # noqa: BLE001
        log.warning("road exposure failed: %s", e)

    buildings.sort(key=lambda x: -x["depth_m"])
    return dict(
        rain_mm=rain_mm, tau_m=TAU,
        buildings=dict(total_in_domain=n_buildings, at_risk=n_at_risk,
                       at_risk_pct=round(100 * n_at_risk / max(n_buildings, 1), 1),
                       points=buildings[:MAX_FEATURES]),
        roads=dict(total=n_roads, flooded=n_flooded_roads,
                   flooded_lines=roads_flooded, dry_lines=roads_dry),
        note="At-risk = flood depth > %.2f m at the building/road (emulated). Dry roads are candidate "
             "evacuation routes. OSM footprints; not a life-safety certification." % TAU,
    )


def _line_coords(geom):
    """[(lat, lon), ...] for a LineString-like geometry; [] otherwise."""
    try:
        if geom.geom_type == "LineString":
            return [(y, x) for x, y in geom.coords]
        if geom.geom_type == "MultiLineString":
            return [(y, x) for part in geom.geoms for x, y in part.coords]
    except Exception:  # noqa: BLE001
        pass
    return []


def save_urban_grid(work=None):
    """Rasterize OSM buildings + roads onto the twin's domain grid -> urban_grid.npz.

    The canal router uses this to avoid houses (near-forbidden cells) and prefer road
    corridors (where storm drains are actually dug). Buildings are marked by centroid,
    roads by densified polyline samples — adequate at 60 m cells. Needs osmnx; run on
    Colab/Kaggle alongside save_exposure.
    """
    import torch
    import rasterio
    work = work or CFG.work
    meta = torch.load(f"{work}/twin_meta.pt", map_location="cpu", weights_only=False)
    row0, col0, N = meta["row0"], meta["col0"], meta["n_grid"]
    with rasterio.open(f"{work}/dem.tif") as s:
        T = s.transform
    inv = ~T

    def cell(lat, lon):
        col30, row30 = inv * (lon, lat)
        dr, dc = int((row30 - row0) // 2), int((col30 - col0) // 2)
        return (dr, dc) if (0 <= dr < N and 0 <= dc < N) else None

    lon0, lat_top = T * (col0, row0)
    lon1, lat_bot = T * (col0 + N * 2, row0 + N * 2)
    north, south = max(lat_top, lat_bot), min(lat_top, lat_bot)
    east, west = max(lon0, lon1), min(lon0, lon1)

    buildings = np.zeros((N, N), dtype=bool)
    roads = np.zeros((N, N), dtype=bool)
    b = _osm_features(north, south, east, west, {"building": True})
    for geom in b.geometry:
        try:
            c = geom.centroid
            at = cell(c.y, c.x)
        except Exception:  # noqa: BLE001
            continue
        if at:
            buildings[at] = True
    r = _osm_features(north, south, east, west, {"highway": True})
    step = 0.00027                                     # ~30 m in degrees: half a domain cell
    for geom in r.geometry:
        pts = _line_coords(geom)
        for (la1, lo1), (la2, lo2) in zip(pts, pts[1:]):
            k = max(2, int(max(abs(la2 - la1), abs(lo2 - lo1)) / step) + 1)
            for f in np.linspace(0.0, 1.0, k):
                at = cell(la1 + f * (la2 - la1), lo1 + f * (lo2 - lo1))
                if at:
                    roads[at] = True
    np.savez(f"{work}/urban_grid.npz", buildings=buildings, roads=roads)
    log.info("urban grid cached -> %s/urban_grid.npz (building cells %d, road cells %d)",
             work, int(buildings.sum()), int(roads.sum()))
    return dict(building_cells=int(buildings.sum()), road_cells=int(roads.sum()))


def save_exposure(rain_mm=None, work=None, device="cpu"):
    """Compute exposure and cache it in the bundle as exposure.json.

    Run this where osmnx exists (Colab); the deployed API then serves the cached copy on
    hosts without osmnx / Overpass access instead of a 503.
    """
    from ..io import save_json
    work = work or CFG.work
    out = assess_exposure(rain_mm=rain_mm, work=work, device=device)
    path = save_json(os.path.join(work, CACHE_FILE), out)
    log.info("exposure cached -> %s (buildings at risk: %s)",
             path, out["buildings"]["at_risk"])
    return out


def load_cached(work=None):
    """The bundle's cached exposure.json, or None if it was never computed."""
    from ..io import load_json
    work = work or CFG.work
    return load_json(os.path.join(work, CACHE_FILE))
