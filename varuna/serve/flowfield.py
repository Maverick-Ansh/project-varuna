"""Where the water GOES — a velocity field to draw arrows with.

The twin has always known this. `Domain.step` computes `qx`/`qy`, the unit-width discharge on
every cell face, and then `simulate` throws them away and returns depth alone. So the dashboard
could say how deep a street gets but never which way the water runs down it.

This module reconstructs the flow direction from a depth grid the cheap way, so it works on the
*emulator's* instant output and keeps up with the rainfall slider:

    direction  =  -grad(z + h)            steepest descent of the WATER SURFACE
    speed      =  (1/n) * h^(2/3) * |grad(z + h)|^(1/2)      (Manning, steady uniform flow)

Taking the gradient of the water surface rather than the bed is what makes ponds behave: inside
a filled depression the surface is flat and the water crawls, and it spills over the lowest lip
instead of pointing at the bottom of the hole forever.

CAVEAT, stated once and repeated in the payload: the emulator returns `hmax`, the per-cell maximum
over the storm, and cells do not peak simultaneously — so `z + hmax` is not an instantaneous water
surface. For *direction* it is a good proxy (bed relief dominates the gradient everywhere except
inside ponds, which is exactly where the h-term is meant to correct it). For physically exact,
time-resolved velocity use `Domain.rollout(track_flux=True)`; this is the millisecond approximation.

Pure numpy — no rasterio, no torch — so it unit-tests offline against synthetic terrain.
"""
from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger("varuna.serve.flowfield")

_NET_CACHE = {}             # work dir -> prepared RoadNet (rebuilding it costs ~2 s on Patna)

MIN_DEPTH_M = 0.05          # below this a cell is "damp", not flowing — no arrow
MIN_SLOPE = 1e-5            # gradient floor; stops flat cells dividing by ~0
MAX_SPEED_MS = 4.0          # backstop: DEM pits/spikes can manufacture absurd local slopes
SMOOTH_PASSES = 1           # 3x3 box passes over the water surface before differencing


def _box3(a):
    """One 3x3 box-filter pass with edge replication (no scipy dependency)."""
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def surface_velocity(h, z, mann, dx, min_depth=MIN_DEPTH_M, max_speed=MAX_SPEED_MS,
                     smooth=SMOOTH_PASSES):
    """Depth-averaged velocity (m/s) at cell centres from a depth grid and the bed.

    Returns (u, v): u eastward (+column), v northward (-row), both zero where the cell is dry.
    Rasters are north-up, so row increases southward — the sign flip on the row gradient is what
    turns "downhill in grid space" into "northward in map space".

    The water surface is box-smoothed before differencing. A 30 m DEM carries metre-scale noise,
    and a single noisy pixel next to a real one manufactures a slope steep enough to saturate any
    plausible velocity — smoothing kills the one-cell spikes while leaving the drainage corridors
    (which are many cells long) untouched. `smooth=0` gives the raw gradient.
    """
    h = np.asarray(h, dtype="float64")
    z = np.asarray(z, dtype="float64")
    eta = z + h                                              # water-surface elevation
    for _ in range(int(smooth)):
        eta = _box3(eta)

    # central differences, one-sided at the edges (np.gradient does exactly this)
    d_drow, d_dcol = np.gradient(eta, dx)
    sx = d_dcol                                              # d(eta)/d(east)
    sy = -d_drow                                             # d(eta)/d(north)

    smag = np.hypot(sx, sy)
    smag_safe = np.maximum(smag, MIN_SLOPE)

    n = np.asarray(mann, dtype="float64")
    n = np.where(n > 1e-6, n, 0.035)                         # defensive: never divide by zero
    speed = (1.0 / n) * np.power(np.maximum(h, 0.0), 2.0 / 3.0) * np.sqrt(smag_safe)
    speed = np.clip(speed, 0.0, max_speed)

    wet = h >= min_depth
    speed = np.where(wet, speed, 0.0)

    # unit vector points DOWN the surface gradient
    u = -speed * sx / smag_safe
    v = -speed * sy / smag_safe
    return u, v


def bearing_deg(u, v):
    """Compass bearing (0 = north, 90 = east) of an east/north vector."""
    return float((math.degrees(math.atan2(u, v)) + 360.0) % 360.0)


def arrow_points(u, v, h, latlon, step=4, min_depth=MIN_DEPTH_M, max_arrows=1200,
                 min_speed=0.02):
    """Subsample the field into renderable arrows, biggest flow first.

    `latlon(row, col) -> [lat, lon]` keeps this testable without rasterio (the bundle driver
    passes the dem.tif transform version). `step` thins the grid so the map is readable rather
    than a hairball; the cap then keeps the arrows that carry the most water.

    Ranking is by DISCHARGE (speed x depth), not raw speed: a 5 cm film sliding fast down a
    kerb is not the story, a slow half-metre of water crossing a junction is. Ranking on speed
    alone promotes exactly the thin-and-fast cells that DEM noise creates.
    """
    R, C = h.shape
    out = []
    for r in range(0, R, step):
        for c in range(0, C, step):
            d = float(h[r, c])
            if d < min_depth:
                continue
            uu, vv = float(u[r, c]), float(v[r, c])
            sp = math.hypot(uu, vv)
            if sp < min_speed:
                continue
            out.append({"latlon": latlon(r, c), "bearing": round(bearing_deg(uu, vv), 1),
                        "speed_ms": round(sp, 3), "depth_m": round(d, 3),
                        "flux_m2s": round(sp * d, 4)})
    out.sort(key=lambda a: a["flux_m2s"], reverse=True)
    return out[:max_arrows]


def _coarsen(a, k):
    """Block-mean an array down by an integer factor, trimming any ragged remainder."""
    if k <= 1:
        return a
    ny, nx = a.shape
    ny, nx = (ny // k) * k, (nx // k) * k
    return a[:ny, :nx].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))


def velocity_layer(u, v, bounds, ref_time=None, max_cells=128):
    """The field as a leaflet-velocity / GRIB-style payload for animated particle rendering.

    bounds = [[lat_min, lon_min], [lat_max, lon_max]]. leaflet-velocity reads data row-major
    starting at (la1, lo1) — the NORTH-WEST corner — walking south, which is exactly raster order,
    so the grids pass straight through. parameterNumber 2 = U (eastward), 3 = V (northward).

    Grids larger than `max_cells` a side are block-averaged down first: a 256^2 tile would
    otherwise ship ~130k floats on every slider drag, and the particle animation interpolates
    the field anyway so the extra resolution never reaches the screen.
    """
    (lat_min, lon_min), (lat_max, lon_max) = bounds
    k = max(1, int(np.ceil(max(u.shape) / float(max_cells))))
    u, v = _coarsen(np.asarray(u), k), _coarsen(np.asarray(v), k)
    ny, nx = u.shape
    # dx/dy are the grid spacings in degrees; nx-1 intervals span the box
    dx_deg = (lon_max - lon_min) / max(nx - 1, 1)
    dy_deg = (lat_max - lat_min) / max(ny - 1, 1)

    def header(param):
        return {"parameterUnit": "m.s-1", "parameterNumber": param, "parameterNumberName":
                "eastward_wind" if param == 2 else "northward_wind", "parameterCategory": 2,
                "nx": int(nx), "ny": int(ny), "lo1": float(lon_min), "la1": float(lat_max),
                "lo2": float(lon_max), "la2": float(lat_min), "dx": float(dx_deg),
                "dy": float(dy_deg), "refTime": ref_time or "2026-01-01T00:00:00Z"}

    clean = lambda a: [round(float(x), 4) for x in np.nan_to_num(a).ravel()]   # noqa: E731
    return [{"header": header(2), "data": clean(u)},
            {"header": header(3), "data": clean(v)}]


def road_segments(net, shape):
    """Static per-edge street geometry as flat numpy arrays — everything that does NOT depend
    on the storm. Built once per bundle; `road_flow` then reduces to vectorised arithmetic.

    Returns row/col of each edge's midpoint cell, the edge's east/north unit vector, its length
    and its midpoint lat/lon. Edges are deduplicated and clipped to the domain.
    """
    R, C = shape
    rows, cols, easts, norths, lens, mlat, mlon = [], [], [], [], [], [], []
    seen = set()
    for a in range(net.n):
        if not net.alive[a]:
            continue
        for b, length in net.adj[a]:
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            ca, cb = net.cell[a], net.cell[b]
            if ca is None or cb is None:
                continue
            r, c = (ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2
            if not (0 <= r < R and 0 <= c < C):
                continue
            la, lo = float(net.lat[a]), float(net.lon[a])
            dlat, dlon = float(net.lat[b]) - la, float(net.lon[b]) - lo
            east, north = dlon * math.cos(math.radians(la)), dlat
            norm = math.hypot(east, north)
            if norm < 1e-12:
                continue
            rows.append(r)
            cols.append(c)
            easts.append(east / norm)
            norths.append(north / norm)
            lens.append(float(length))
            mlat.append((la + float(net.lat[b])) / 2.0)
            mlon.append((lo + float(net.lon[b])) / 2.0)
    return {"row": np.asarray(rows, dtype="int64"), "col": np.asarray(cols, dtype="int64"),
            "east": np.asarray(easts), "north": np.asarray(norths),
            "length_m": np.asarray(lens), "lat": np.asarray(mlat), "lon": np.asarray(mlon)}


def road_flow(net, u, v, h, min_depth=MIN_DEPTH_M, min_speed=0.02, max_arrows=800,
              segments=None, exclude=None, flag=None):
    """Which way does the water run ALONG the streets — one arrow per 60 m cell.

    Each street edge samples the field at its midpoint cell and projects the velocity onto the
    edge's own bearing. The cross-street component is dropped on purpose: water crossing a road
    is not "flow down the road", and drawing it as such would be a lie the eye believes.

    Results are AGGREGATED TO THE DOMAIN CELL, keeping the strongest-carrying street in each.
    OSM way vertices sit metres apart, so a raw per-edge answer is thousands of 5 m arrows — a
    hairball, and fake precision: the velocity field underneath has 60 m cells, so there is no
    information to justify more than one arrow per cell. The reported bearing is already
    sign-corrected, so a renderer can point the arrow at `bearing` and be done.

    Pass `segments` (from `road_segments`) to skip rebuilding the static geometry, `exclude` (a
    boolean grid) to drop cells that are not street at all — permanent water above all — and
    `flag` to mark cells whose depth should not be read as a street depth with `near_water`.

    The distinction matters and is not cosmetic. In Bengaluru the tanks ARE the low points, so a
    road on a tank bund sits one 60 m cell from four metres of water and inherits a depth that is
    the lake's, not the street's. Deleting those arrows would hide real flooding; reporting them
    unqualified would claim a road is chest-deep when it is the tank beside it. So they are
    served, and marked.
    """
    seg = road_segments(net, h.shape) if segments is None else segments
    if len(seg["row"]) == 0:
        return []
    r, c = seg["row"], seg["col"]
    depth = np.asarray(h)[r, c]
    along = np.asarray(u)[r, c] * seg["east"] + np.asarray(v)[r, c] * seg["north"]
    speed = np.abs(along)
    keep = (depth >= min_depth) & (speed >= min_speed)
    if exclude is not None:
        keep &= ~np.asarray(exclude, dtype=bool)[r, c]
    if not keep.any():
        return []

    flux = speed * depth
    # one winner per cell: sort by flux ascending so the last write per cell is the strongest
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(flux[idx], kind="stable")]
    best = {}
    for i in idx:
        best[(int(r[i]), int(c[i]))] = int(i)

    flagged = None if flag is None else np.asarray(flag, dtype=bool)[r, c]
    sign = np.where(along >= 0, 1.0, -1.0)                   # flip the arrow to follow the water
    out = []
    for i in best.values():
        entry = {
            "latlon": [round(float(seg["lat"][i]), 6), round(float(seg["lon"][i]), 6)],
            "bearing": round(bearing_deg(sign[i] * seg["east"][i], sign[i] * seg["north"][i]), 1),
            "speed_ms": round(float(speed[i]), 3),
            "depth_m": round(float(depth[i]), 3),
            "flux_m2s": round(float(flux[i]), 4),
            "length_m": round(float(seg["length_m"][i]), 1),
        }
        if flagged is not None and bool(flagged[i]):
            entry["near_water"] = True
        out.append(entry)
    out.sort(key=lambda e: e["flux_m2s"], reverse=True)
    return out[:max_arrows]


# --------------------------------------------------------------------------- bundle driver


def flow_for_area(rain_mm, work=None, step=4, max_arrows=1200, with_roads=True,
                  with_layer=False, device="cpu"):
    """End-to-end: emulate the storm, build the field, return everything the map needs.

    `with_layer` (the animated-particle grid) is OFF by default: it is the single heaviest part
    of the payload and the dashboard only needs it when that layer is switched on.

    Needs the bundle (emulator + dem.tif); the pure functions above are the testable core.
    """
    import rasterio
    from ..config import CFG
    from .emulator import load_emulator, whatif_grid

    work = work or CFG.work
    hmax, _dig, summary = whatif_grid(rain_mm, None, work=work, device=device)
    b = load_emulator(work, device)
    dom = b["dom"]
    z = dom.z0.cpu().numpy()
    mann = dom.mann.cpu().numpy()

    u, v = surface_velocity(hmax, z, mann, dom.dx)

    with rasterio.open(f"{work}/dem.tif") as src:
        T = src.transform

    def latlon(r, c):
        # +1 lands on the centre of the 60 m domain cell (two 30 m DEM pixels)
        lon, lat = T * (dom.col0 + c * 2 + 1, dom.row0 + r * 2 + 1)
        return [round(lat, 5), round(lon, 5)]

    N = dom.N
    lat_nw, lon_nw = latlon(0, 0)[0], latlon(0, 0)[1]
    lat_se, lon_se = latlon(N - 1, N - 1)[0], latlon(N - 1, N - 1)[1]
    bounds = [[min(lat_nw, lat_se), min(lon_nw, lon_se)],
              [max(lat_nw, lat_se), max(lon_nw, lon_se)]]

    # The emulator's own held-out error, shipped with every depth so nothing downstream can
    # render "0.69 m" as if it were surveyed. Bundles report both a whole-grid and a flooded-cell
    # RMSE; the flooded-cell one is the honest bar for a number about standing water.
    meta = b["meta"]
    rmse = meta.get("flooded_rmse_m") or meta.get("val_rmse_m")

    res = {
        "rain_mm": float(rain_mm),
        "summary": summary,
        "bounds": bounds,
        "arrows": arrow_points(u, v, hmax, latlon, step=step, max_arrows=max_arrows),
        "max_speed_ms": round(float(np.hypot(u, v).max()), 3),
        "depth_rmse_m": None if rmse is None else round(float(rmse), 3),
        "note": ("Direction is steepest descent of the water surface (z+h); speed is Manning "
                 "steady-flow. Depth is the storm maximum per cell, so this is a peak-conditions "
                 "sketch of the flow, not a time-resolved velocity field. Depths carry the "
                 "emulator's held-out RMSE and are least trustworthy in the deep tail, which no "
                 "validation covers."),
    }
    if with_layer:
        res["velocity_layer"] = velocity_layer(u, v, bounds)
    if with_roads:
        try:
            net, seg = _road_net(work, dom, z, T)
            if net is not None:
                water = _permanent_water(dom)
                arrows = road_flow(net, u, v, hmax, segments=seg, exclude=water,
                                   flag=_dilate(water))
                res["road_flow"] = _name_streets(arrows, work)
                res["n_near_water"] = sum(1 for a in res["road_flow"] if a.get("near_water"))
        except Exception as e:  # noqa: BLE001 — arrows still render without street projection
            log.warning("road flow unavailable: %s", e)
    return res


def _permanent_water(dom):
    """Cells that are lake/river/sea in WorldCover — never a "flooded street".

    `build_domain` stashes the per-cell WorldCover class on the domain; the canonical
    permanently-wet classes live in build.landcover.NO_RECHARGE (80 water, 90 wetland, 95
    mangrove). Returns None when the domain predates the `wc` attribute.
    """
    wc = getattr(dom, "wc", None)
    if wc is None:
        return None
    from ..build.landcover import NO_RECHARGE
    return np.isin(np.asarray(wc), list(NO_RECHARGE))


def _dilate(mask, iterations=1):
    """Grow a mask by `iterations` cells — the neighbours that inherit its water."""
    if mask is None:
        return None
    from scipy import ndimage
    return ndimage.binary_dilation(mask, iterations=iterations)


def _name_streets(arrows, work):
    """Attach the nearest named road to each street arrow, so a hover can say WHERE.

    `roadnet`'s cached graph deliberately keeps only node chains — way names are dropped — so the
    names come from the naming anchors that the danger-zone pins already use. Arrows with no road
    anchor within range keep `street: None` rather than borrowing a name from streets away.
    """
    if not arrows:
        return arrows
    try:
        from .zones import load_places, nearest_names
    except Exception as e:  # noqa: BLE001
        log.debug("street naming unavailable: %s", e)
        return arrows
    places = load_places(work)
    if not places:
        return arrows
    names = nearest_names([a["latlon"][0] for a in arrows],
                          [a["latlon"][1] for a in arrows], places)
    for a, nm in zip(arrows, names):
        a["street"] = nm
    return arrows


def _road_net(work, dom, z, T):
    """The bundle's prepared street graph + static edge geometry, cached. (None, None) when the
    bundle has no road graph.

    `roadnet.prepare` walks every OSM vertex and runs a connected-components pass — ~2 s on
    Patna's 37k nodes — and `road_segments` walks the adjacency again. Both depend only on the
    graph and the terrain, which are fixed per bundle, so both are cached: only the storm changes
    between requests, and the storm only touches vectorised arithmetic.
    """
    if work in _NET_CACHE:
        return _NET_CACHE[work]
    from . import roadnet
    graph = roadnet.load_road_graph(work)
    if not graph:
        _NET_CACHE[work] = (None, None)
        return _NET_CACHE[work]
    inv = ~T
    N = dom.N

    def cell_of(lat, lon):
        col30, row30 = inv * (lon, lat)
        dr, dc = int((row30 - dom.row0) // 2), int((col30 - dom.col0) // 2)
        return (dr, dc) if (0 <= dr < N and 0 <= dc < N) else None

    net = roadnet.prepare(graph, z, cell_of)
    _NET_CACHE[work] = (net, road_segments(net, (N, N)))
    return _NET_CACHE[work]
