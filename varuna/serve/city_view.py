"""Serve the joined city: one Mumbai, from a precomputed storm ladder.

The tiles each have an emulator that answers the rainfall slider in milliseconds. The joined
city domain has no emulator — training one on an 817 x 586 grid is its own project — and running
the simulator live would take minutes on the CPU the public Space runs on. It does not need to:
rainfall is a single scalar, so `build.city.build_city_bundle` precomputes a ladder of storms and
this module interpolates between the rungs.

Interpolation is honest here in a way it would not be for, say, an intervention: flood depth is
monotone and smooth in rainfall total (the dose-response curve was measured and is monotone),
and the answer is always bracketed by two REAL simulations rather than extrapolated from one.
Outside the ladder it clamps and says so rather than extrapolating a storm nobody simulated.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

log = logging.getLogger("varuna.serve.city_view")

_CACHE = {}


def load_city(work):
    """(meta, npz) for a city bundle, cached. Raises FileNotFoundError if it was never built."""
    if work in _CACHE:
        return _CACHE[work]
    meta_path = os.path.join(work, "city_meta.json")
    grid_path = os.path.join(work, "city_hmax.npz")
    if not (os.path.exists(meta_path) and os.path.exists(grid_path)):
        raise FileNotFoundError(
            f"no city bundle in {work} — build it with varuna.build.city.build_city_bundle")
    with open(meta_path) as f:
        meta = json.load(f)
    npz = np.load(grid_path)
    _CACHE[work] = (meta, npz)
    return _CACHE[work]


def bounds(meta):
    """[[south, west], [north, east]] of the city grid, for a map overlay."""
    (lat0, lon0), (h, w), pix = meta["origin"], meta["grid"], meta["pix_deg"]
    # the grid is 60 m cells built from 30 m pixels, so one cell spans 2 raster pixels
    south = lat0 - h * 2 * pix
    east = lon0 + w * 2 * pix
    return [[south, lon0], [lat0, east]]


def depth_grid(rain_mm, work):
    """Interpolated city depth grid (metres) at `rain_mm`, plus what it was derived from."""
    meta, npz = load_city(work)
    rungs = sorted(float(r) for r in meta["rains"])
    r = float(rain_mm)
    clamped = None
    if r <= rungs[0]:
        clamped = "below" if r < rungs[0] else None
        grid = npz[str(rungs[0])].astype("float32")
        used, frac = (rungs[0], rungs[0]), 0.0
    elif r >= rungs[-1]:
        clamped = "above" if r > rungs[-1] else None
        grid = npz[str(rungs[-1])].astype("float32")
        used, frac = (rungs[-1], rungs[-1]), 0.0
    else:
        hi = next(x for x in rungs if x >= r)
        lo = max(x for x in rungs if x <= r)
        if hi == lo:
            grid = npz[str(lo)].astype("float32")
            used, frac = (lo, hi), 0.0
        else:
            frac = (r - lo) / (hi - lo)
            grid = ((1 - frac) * npz[str(lo)].astype("float32")
                    + frac * npz[str(hi)].astype("float32"))
            used = (lo, hi)
    return grid, {"rungs_used": list(used), "blend": round(float(frac), 3),
                  "clamped": clamped, "ladder": rungs}


def city_flood(rain_mm, work, min_depth=0.15):
    """Headline city-wide numbers at a rainfall total — the joined-domain answer."""
    meta, npz = load_city(work)
    grid, prov = depth_grid(rain_mm, work)
    covered = npz["covered"]
    built = npz["built"]
    dx = float(meta.get("dx", 60.0))
    cell_km2 = dx * dx / 1e6

    wet = (grid > min_depth) & covered
    wet_built = wet & built
    volume = float(np.where(covered, grid, 0.0).sum()) * dx * dx

    out = {
        "rain_mm": float(rain_mm),
        "flood_km2": round(float(wet.sum()) * cell_km2, 3),
        "flood_built_km2": round(float(wet_built.sum()) * cell_km2, 3),
        "volume_m3": round(volume),
        "max_depth_m": round(float(grid[covered].max()), 2),
        "land_km2": round(float(covered.sum()) * cell_km2, 1),
        "built_km2": round(float(built.sum()) * cell_km2, 1),
        "min_depth_m": min_depth,
        "grid": meta["grid"],
        "bounds": bounds(meta),
        "with_sink_field": meta.get("with_sink_field", False),
        "provenance": prov,
        "note": meta.get("note"),
    }
    if meta.get("outflow_m3"):
        lo, hi = prov["rungs_used"]
        o = meta["outflow_m3"]
        if str(lo) in o and str(hi) in o:
            f = prov["blend"]
            out["outflow_m3_lower_bound"] = round((1 - f) * o[str(lo)] + f * o[str(hi)])
    if prov["clamped"]:
        out["warning"] = (
            f"{rain_mm} mm is {prov['clamped']} the simulated ladder "
            f"({prov['ladder'][0]:.0f}-{prov['ladder'][-1]:.0f} mm); showing the nearest rung "
            f"rather than extrapolating a storm that was never run.")
    return out
