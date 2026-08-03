"""Named danger zones — turn a depth grid into pins a human can act on.

The dashboard could already draw a blue blob over a city. A blob is not a decision. What a
resident or a ward officer needs is "Rajendra Nagar, waist-deep, and it holds 40,000 m3" — a
place with a name, a severity anyone can picture, and a size you can rank against other places.

Two pieces:

  * `load_places` / `build_places` — every named place inside the domain, fetched once from
    Overpass (OSM `place=` nodes plus admin_level 9/10 boundary centroids) and cached in the
    bundle as `places.json.gz`. Same contract as `roadnet.py`: fetched at build time, never at
    serve time, so the deployed Space and the tests never touch the network.
  * `danger_zones` — connected wet regions of a depth grid, ranked by the volume of water
    standing on BUILT land, each labelled with its nearest named place.

Severity reuses the citizen-report vocabulary (`reports.DEPTH_M`: ankle / knee / waist / chest)
on purpose. The model and the people reporting into it should describe the world with the same
words, or the two halves of the loop cannot be compared.

`alerts.py::_aggregate_wards` fetches OSM ward polygons and then throws the names away
(`ward_0`, `ward_1`, ...). This module is where the names come back.
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import os

import numpy as np

from ..config import CFG
from .reports import DEPTH_M

log = logging.getLogger("varuna.serve.zones")

CACHE_FILE = "places.json.gz"

# OSM place types worth naming a flood after, coarse -> fine. Villages/hamlets are included
# because the Karnataka Tier B towns are small and their neighbourhoods are not mapped.
PLACE_TYPES = ("city", "town", "borough", "suburb", "quarter", "neighbourhood",
               "village", "hamlet", "city_block")

# Roads carry most of the usable naming in Indian cities. A probe of the Patna domain found
# SEVEN `place=` nodes and ZERO admin_level 8/9/10 ward boundaries — Patna's wards are simply not
# in OSM — while its named arterials (Bailey Road, Ashok Rajpath, Boring Road) are all mapped and
# are what residents actually navigate by. So named major roads are sampled as naming anchors.
NAMED_ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary")
ROAD_SAMPLE_M = 250.0            # spacing of naming anchors along a named road

# Nearest-anchor wins, but not all anchors are equally useful as a flood label. These penalties
# (in km) are added to the true distance before comparing: a suburb name 600 m away should beat
# a road name 400 m away, and "Patna" — true of the whole domain — should almost never win.
PRIORITY_KM = {
    "admin9": 0.0, "admin10": 0.0, "admin8": 0.15,
    "suburb": 0.0, "neighbourhood": 0.05, "quarter": 0.05, "city_block": 0.2,
    "village": 0.10, "hamlet": 0.15, "borough": 0.30, "town": 0.30, "city": 0.80,
    "road": 0.25, "station": 0.30, "landuse": 0.50,
}
DEFAULT_PRIORITY_KM = 0.40
LANDMARK_KINDS = ("road", "station")

# Some OSM ways are named for what they ARE, not where they are — Patna has several ways whose
# entire name is "Flyover". As a flood label that is worse than useless, so these are dropped as
# naming anchors. Compound names that merely contain the word ("Bailey Road Flyover") are kept.
GENERIC_NAMES = {
    "flyover", "bridge", "overbridge", "foot overbridge", "underpass", "subway", "culvert",
    "road", "street", "lane", "path", "marg", "gali", "highway", "expressway",
    "service road", "link road", "main road", "approach road", "access road", "side road",
    "bypass", "slip road", "connector", "ramp", "roundabout", "flyover bridge",
    "railway station", "bus stand", "bus stop", "parking", "unnamed road",
}

# A single connected wet region larger than this is not one place any more — at 200 mm the whole
# Patna floodplain becomes one blob. Over-large regions are split at their deep cores (see
# `_partition_large`) so the pins stay at the scale a person can act on.
MAX_ZONE_KM2 = 2.5
SPLIT_STEP_M = 0.25          # raise the water level by this much to find separate cores
MAX_SPLIT_DEPTH = 4

# Peak depth -> the band a person would use. Ordered deepest-first for the lookup below.
BANDS = [("chest", DEPTH_M["chest"], "EXTREME"),
         ("waist", DEPTH_M["waist"], "HIGH"),
         ("knee", DEPTH_M["knee"], "MODERATE"),
         ("ankle", DEPTH_M["ankle"], "LOW")]

# Beyond this the nearest named place is not really "where the flood is" — say "near X" instead
# of claiming the name. Neighbourhood spacing in Indian cities is well under a kilometre.
NAME_CONFIDENT_KM = 1.2


def band_for(depth_m):
    """(band, severity) for a peak depth, using the citizen-report vocabulary."""
    for name, thresh, sev in BANDS:
        if depth_m >= thresh:
            return name, sev
    return "damp", "MINIMAL"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- fetch + persist


def fetch_places(bbox, timeout=90, mirrors=None):
    """Naming anchors in bbox=(south, west, north, east) from Overpass.

    Three `out` statements in one request, because the three families need different geometry:
    point features come back bare, named roads need `out geom` so they can be sampled along their
    length, and boundaries/landuse need `out center` so we never have to parse a polygon.
    """
    import requests
    from .roadnet import OVERPASS_MIRRORS
    mirrors = mirrors or OVERPASS_MIRRORS
    s, w, n, e = bbox
    box = f"({s:f},{w:f},{n:f},{e:f})"
    admin = '["boundary"="administrative"]["admin_level"~"^(8|9|10)$"]["name"]'
    query = (
        f'[out:json][timeout:{timeout}];'
        f'(node["place"~"^({"|".join(PLACE_TYPES)})$"]["name"]{box};'
        f' node["railway"="station"]["name"]{box};);out;'
        f'(way["highway"~"^({"|".join(NAMED_ROAD_CLASSES)})$"]["name"]{box};);out geom;'
        f'(way{admin}{box};relation{admin}{box};'
        f' way["landuse"~"^(residential|industrial|commercial)$"]["name"]{box};);out center;'
    )
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
    raise RuntimeError(f"all Overpass mirrors failed (last: {last})")


def _kind_of(tags):
    if tags.get("boundary") == "administrative":
        return f"admin{tags.get('admin_level', '')}"
    if tags.get("place"):
        return tags["place"]
    if tags.get("railway") == "station":
        return "station"
    if tags.get("highway"):
        return "road"
    if tags.get("landuse"):
        return "landuse"
    return "place"


def _sample_way(geometry, every_m=ROAD_SAMPLE_M):
    """Anchor points along a way's geometry, roughly `every_m` apart (always both ends)."""
    pts = [(g["lat"], g["lon"]) for g in geometry if "lat" in g and "lon" in g]
    if not pts:
        return []
    if len(pts) == 1:
        return pts
    out = [pts[0]]
    acc = 0.0
    for (la1, lo1), (la2, lo2) in zip(pts, pts[1:]):
        acc += haversine_km(la1, lo1, la2, lo2) * 1000.0
        if acc >= every_m:
            out.append((la2, lo2))
            acc = 0.0
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def places_from_overpass(osm, bbox=None):
    """Overpass JSON -> the persisted artifact: [{name, lat, lon, kind}, ...].

    Named roads become a RUN of anchors along their length, so "Bailey Road" can name a flood
    anywhere along Bailey Road rather than only near its midpoint. Anchors are deduped on
    (name, kind, ~100 m cell) to keep the cache small.
    """
    out, seen = [], set()

    def push(name, lat, lon, kind):
        if lat is None or lon is None:
            return
        key = (name.lower(), kind, round(float(lat), 3), round(float(lon), 3))
        if key in seen:
            return
        seen.add(key)
        out.append({"name": name, "lat": round(float(lat), 6), "lon": round(float(lon), 6),
                    "kind": kind})

    for el in osm.get("elements", []):
        tags = el.get("tags") or {}
        name = (tags.get("name") or "").strip()
        if not name or name.lower() in GENERIC_NAMES:
            continue
        kind = _kind_of(tags)
        if "geometry" in el:                                  # named road -> sample along it
            for lat, lon in _sample_way(el["geometry"]):
                push(name, lat, lon, kind)
        elif el.get("type") == "node":
            push(name, el.get("lat"), el.get("lon"), kind)
        else:
            c = el.get("center") or {}
            push(name, c.get("lat"), c.get("lon"), kind)
    return {"bbox": list(bbox) if bbox else None, "places": out}


def save_places(work, payload):
    path = os.path.join(work, CACHE_FILE)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    log.info("places cached -> %s (%d named)", path, len(payload["places"]))
    return path


def load_places(work=None):
    """The bundle's cached named places, or [] if they were never fetched."""
    path = os.path.join(work or CFG.work, CACHE_FILE)
    if not os.path.exists(path):
        return []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f).get("places", [])
    except Exception as e:  # noqa: BLE001
        log.warning("could not read %s: %s", path, e)
        return []


def build_places(work=None, margin=0.005):
    """Fetch + cache named places for a bundle's domain (the only step needing internet)."""
    import rasterio
    import torch
    work = work or CFG.work
    meta = torch.load(f"{work}/twin_meta.pt", map_location="cpu", weights_only=False)
    row0, col0, N = meta["row0"], meta["col0"], meta["n_grid"]
    with rasterio.open(f"{work}/dem.tif") as s:
        T = s.transform
    lon0, lat_top = T * (col0, row0)
    lon1, lat_bot = T * (col0 + N * 2, row0 + N * 2)
    bbox = (min(lat_top, lat_bot) - margin, min(lon0, lon1) - margin,
            max(lat_top, lat_bot) + margin, max(lon0, lon1) + margin)
    payload = places_from_overpass(fetch_places(bbox), bbox=bbox)
    save_places(work, payload)
    return payload


# --------------------------------------------------------------------------- the zones


def name_for(lat, lon, places):
    """Best naming anchor for a point -> dict(label, name, kind, km, named, landmark).

    "Best" is nearest AFTER a per-kind penalty (`PRIORITY_KM`), so a neighbourhood name a little
    further off wins over an arterial road, and the city's own name — true everywhere in the
    domain and therefore useless as a label — effectively never does.

    `landmark` is the nearest road/station regardless of what won, so the UI can say
    "Rajendra Nagar · near Bailey Road" and be recognisable to someone standing in the water.
    """
    empty = {"label": None, "name": None, "kind": None, "km": None, "named": False,
             "landmark": None}
    if not places:
        return empty
    best, best_score, best_km = None, None, None
    land, land_km = None, None
    for p in places:
        km = haversine_km(lat, lon, p["lat"], p["lon"])
        score = km + PRIORITY_KM.get(p["kind"], DEFAULT_PRIORITY_KM)
        if best_score is None or score < best_score:
            best, best_score, best_km = p, score, km
        if p["kind"] in LANDMARK_KINDS and (land_km is None or km < land_km):
            land, land_km = p, km
    confident = best_km <= NAME_CONFIDENT_KM
    return {"label": best["name"] if confident else f"near {best['name']}",
            "name": best["name"], "kind": best["kind"], "km": round(best_km, 2),
            "named": bool(confident),
            "landmark": (land["name"] if land is not None and land_km <= NAME_CONFIDENT_KM
                         else None)}


def _cores(mask, hmax, level, min_cells, max_cells, depth=0):
    """Boolean masks of the distinct deep cores inside one over-large wet region.

    Raising the water level splits a merged sheet back into the separate pools that made it: at
    60 mm Patna has a dozen ponds, at 200 mm they touch and label them as one 20 km2 blob. Each
    step up the level asks "are these still one pool at THIS depth?" and recurses while any piece
    is still too big to be a single place.
    """
    from scipy import ndimage
    if mask.sum() <= max_cells or depth >= MAX_SPLIT_DEPTH:
        return [mask]
    level = level + SPLIT_STEP_M
    if level >= float(hmax[mask].max()):
        return [mask]
    lab, n = ndimage.label(mask & (hmax > level))
    subs = [lab == i for i in range(1, n + 1)]
    subs = [s for s in subs if s.sum() >= min_cells]
    if len(subs) <= 1:
        return _cores(mask, hmax, level, min_cells, max_cells, depth + 1)
    out = []
    for s in subs:
        out.extend(_cores(s, hmax, level, min_cells, max_cells, depth + 1))
    return out


def _partition_large(mask, hmax, min_cells, max_cells, min_depth):
    """Split an over-large wet region into sub-regions, one per deep core.

    The cores are found by raising the water level; every cell of the ORIGINAL region is then
    assigned to its nearest core, so the shallow fringe is redistributed rather than discarded
    and the sub-regions still sum to the parent's area and volume exactly.
    """
    from scipy import ndimage
    cores = _cores(mask, hmax, min_depth, min_cells, max_cells)
    if len(cores) <= 1:
        return [mask]
    seeds = np.zeros(mask.shape, dtype="int32")
    for i, cm in enumerate(cores, start=1):
        seeds[cm] = i
    # nearest-core assignment: the index of the closest non-zero seed cell for every cell
    _, (ri, ci) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    owner = seeds[ri, ci]
    return [(owner == i) & mask for i in range(1, len(cores) + 1)]


def danger_zones(hmax, built, latlon, places=None, dx=None, min_depth=None, max_zones=12,
                 min_cells=2, max_zone_km2=MAX_ZONE_KM2):
    """Connected wet regions ranked by the volume of water standing on BUILT land.

    `latlon(row, col) -> [lat, lon]` keeps this testable without rasterio. Zones with no built
    cells at all are dropped: a flooded field is not a danger zone, and on the Patna tile the
    biggest wet blob by far is the Ganga itself.

    Ranking is by built flooded volume — the same quantity `/api/whatif` already reports as
    `flooded_volume_m3`, so the zone list always sums to something the summary agrees with.
    """
    from scipy import ndimage
    dx = float(dx or CFG.dx)
    min_depth = float(CFG.min_depth_m if min_depth is None else min_depth)
    hmax = np.asarray(hmax, dtype="float64")
    built = np.asarray(built, dtype="float64")
    places = places or []

    wet = hmax > min_depth
    lab, n = ndimage.label(wet)
    if n == 0:
        return []

    cell_area = dx * dx
    max_cells = max(int(min_cells), int(max_zone_km2 * 1e6 / cell_area))
    excess = np.clip(hmax - min_depth, 0.0, None)

    regions = []
    for i in range(1, n + 1):
        m = lab == i
        if int(m.sum()) < min_cells:
            continue
        regions.extend(_partition_large(m, hmax, min_cells, max_cells, min_depth))

    zones = []
    for m in regions:
        n_cells = int(m.sum())
        if n_cells < min_cells:
            continue
        built_m = m & (built > 0)
        built_cells = int(built_m.sum())
        if built_cells == 0:
            continue                                    # open water / farmland, not a danger zone
        built_vol = float(excess[built_m].sum()) * cell_area
        peak = float(hmax[m].max())
        # centroid of the DEEPEST built part — the pin should sit where the water is, not at the
        # geometric middle of a long thin sliver that may be dry in the middle
        w = excess * built_m
        if w.sum() <= 0:
            rr, cc = ndimage.center_of_mass(built_m)
        else:
            rr, cc = ndimage.center_of_mass(w)
        lat, lon = latlon(int(round(rr)), int(round(cc)))
        band, severity = band_for(peak)
        nm = name_for(lat, lon, places)
        zones.append({
            "latlon": [lat, lon],
            "place": nm["label"] or "unnamed area",
            "place_name": nm["name"],
            "place_kind": nm["kind"],
            "place_km": nm["km"],
            "landmark": nm["landmark"],
            "named": nm["named"],
            "band": band,
            "severity": severity,
            "peak_depth_m": round(peak, 2),
            "mean_depth_m": round(float(hmax[built_m].mean()), 2),
            "area_km2": round(n_cells * cell_area / 1e6, 3),
            "built_area_km2": round(built_cells * cell_area / 1e6, 3),
            "volume_m3": round(built_vol),
            "n_cells": n_cells,
        })
    zones.sort(key=lambda z: z["volume_m3"], reverse=True)
    for rank, z in enumerate(zones[:max_zones], start=1):
        z["rank"] = rank
    return zones[:max_zones]


def zones_for_grid(hmax, work=None, dom=None, transform=None, max_zones=12, device="cpu"):
    """Bundle-aware wrapper: build the latlon mapper from the bundle and name the zones."""
    from .emulator import load_emulator
    work = work or CFG.work
    if dom is None:
        dom = load_emulator(work, device)["dom"]
    if transform is None:
        import rasterio
        with rasterio.open(f"{work}/dem.tif") as src:
            transform = src.transform
    T = transform

    def latlon(r, c):
        r = int(np.clip(r, 0, dom.N - 1))
        c = int(np.clip(c, 0, dom.N - 1))
        lon, lat = T * (dom.col0 + c * 2 + 1, dom.row0 + r * 2 + 1)
        return [round(lat, 5), round(lon, 5)]

    return danger_zones(hmax, dom.built.cpu().numpy(), latlon,
                        places=load_places(work), dx=dom.dx, max_zones=max_zones)
