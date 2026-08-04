"""Street-level water: how deep is it on THIS road, and which way is it running.

The map's flood overlay answers "is this neighbourhood flooded". Standing at a junction, nobody
asks that — they ask how deep the water is on their street. That is a different question and it
needs a different answer, because the twin's cell is 60 m x 60 m and a street is not.

Two things this does that the existing arrow layer (`flowfield.road_flow`) does not:

1. **Every street in view, not just the flowing ones.** `road_flow` keeps one arrow per 60 m cell
   and drops anything slower than 2 cm/s, because at city zoom more than that is a hairball. A
   ponded street with nowhere to drain is exactly the street a resident cares about and exactly
   the one that filter removes. Here segments are returned per-segment, bounded by the viewport
   instead of by a global cap.

2. **A street-depth estimate, not just the cell mean.** `DEPTH_VALIDATION.md` measured this gap
   rather than assuming it: against 83 crowdsourced reports the twin read **4.45x shallower** than
   people did on wet sites, and that ratio implies the water occupies ~810 m^2 of the 3,600 m^2
   cell — a flooded junction, not an evenly wet square. So the cell mean systematically
   understates the puddle. Both numbers are served: `depth_cell_mm` (what the model computes) and
   `depth_street_mm` (that, scaled by the measured concentration), never one silently standing in
   for the other.

The honest limits of the correction, which the API echoes to every caller: it was measured on ONE
city, ONE season, n=38 wet observations; it is a median ratio, not a per-street quantity; and
below the wet threshold there is no measurement at all, so the factor is ramped from 1x rather
than extrapolated. It moves a systematic bias, it does not make the twin able to rank streets —
the same document showed the ranking failure survives everything, including the drainage
inversion.
"""
from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger("varuna.serve.streets")

# Measured in DEPTH_VALIDATION.md: median(observed / predicted) on wet sites, n=38, identical at
# the 0.15 m and 0.30 m thresholds. NOT a tuned parameter — changing it invalidates that doc.
DILUTION_FACTOR = 4.45
WET_THRESHOLD_M = 0.15          # CFG.min_depth_m — where the ratio was measured
MIN_DEPTH_M = 0.02              # below this a street is "damp", not worth a row
MAX_SEGMENTS = 1500


def concentration_factor(depth_m, factor=DILUTION_FACTOR, wet=WET_THRESHOLD_M):
    """Cell-mean -> street multiplier: 1x when dry, ramping to `factor` at the wet threshold.

    The ratio was only ever measured AT wet sites, so above the threshold it is applied as
    measured and below it is interpolated from 1x. The ramp is an interpolation choice made to
    avoid a discontinuity at 0.15 m (a 15 cm cell mean would otherwise jump to 67 cm); it is not
    a measurement, and it is deliberately the conservative direction.
    """
    d = np.asarray(depth_m, dtype="float64")
    ramp = np.clip(d / max(wet, 1e-9), 0.0, 1.0)
    return 1.0 + (factor - 1.0) * ramp


def street_depth_mm(cell_depth_m, factor=DILUTION_FACTOR, wet=WET_THRESHOLD_M):
    """Estimated depth where a person would stand, in mm."""
    d = np.asarray(cell_depth_m, dtype="float64")
    return d * concentration_factor(d, factor, wet) * 1000.0


def _in_bbox(lat, lon, bbox):
    if bbox is None:
        return np.ones(len(lat), dtype=bool)
    (s, w), (n, e) = bbox
    return (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)


def streets_in_view(segments, h, u=None, v=None, bbox=None, min_depth=MIN_DEPTH_M,
                    max_segments=MAX_SEGMENTS, exclude=None, flag=None,
                    factor=DILUTION_FACTOR):
    """Per-street-segment depth and flow direction inside `bbox`.

    `segments` is `flowfield.road_segments(net, h.shape)` — the static per-edge geometry. Returns
    (rows, meta): rows are ready for the map, meta records what was filtered or truncated so a
    caller never mistakes a capped list for the whole picture.
    """
    n_total = len(segments["row"])
    if n_total == 0:
        return [], {"n_segments": 0, "n_returned": 0, "truncated": False}

    r, c = segments["row"], segments["col"]
    lat, lon = segments["lat"], segments["lon"]
    depth = np.asarray(h)[r, c]

    keep = _in_bbox(lat, lon, bbox) & (depth >= min_depth)
    n_in_view = int(_in_bbox(lat, lon, bbox).sum())
    if exclude is not None:
        keep &= ~np.asarray(exclude, dtype=bool)[r, c]
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return [], {"n_segments": n_total, "n_in_view": n_in_view, "n_returned": 0,
                    "truncated": False}

    # deepest first, so a truncated list keeps the streets that matter
    idx = idx[np.argsort(-depth[idx], kind="stable")]
    truncated = len(idx) > max_segments
    shown = idx[:max_segments]

    if u is not None and v is not None:
        along = (np.asarray(u)[r, c] * segments["east"]
                 + np.asarray(v)[r, c] * segments["north"])
    else:
        along = np.zeros(n_total)

    flagged = None if flag is None else np.asarray(flag, dtype=bool)[r, c]
    cell_mm = depth * 1000.0
    st_mm = street_depth_mm(depth, factor)

    rows = []
    for i in shown:
        sign = 1.0 if along[i] >= 0 else -1.0
        speed = abs(float(along[i]))
        entry = {
            "latlon": [round(float(lat[i]), 6), round(float(lon[i]), 6)],
            "depth_cell_mm": int(round(float(cell_mm[i]))),
            "depth_street_mm": int(round(float(st_mm[i]))),
            "length_m": round(float(segments["length_m"][i]), 1),
            "cell": [int(r[i]), int(c[i])],
        }
        if speed >= 0.02:
            from .flowfield import bearing_deg
            entry["bearing"] = round(bearing_deg(sign * segments["east"][i],
                                                 sign * segments["north"][i]), 1)
            entry["speed_ms"] = round(speed, 3)
        else:
            entry["ponded"] = True          # water sits here with nowhere to go
        if flagged is not None and bool(flagged[i]):
            entry["near_water"] = True
        rows.append(entry)

    meta = {
        "n_segments": n_total,
        "n_in_view": n_in_view,
        "n_returned": len(rows),
        "truncated": bool(truncated),
        "min_depth_mm": int(round(min_depth * 1000)),
        "dilution_factor": factor,
        "depth_note": (
            "depth_cell_mm is the twin's 60 m cell mean. depth_street_mm scales it by the "
            "concentration factor measured in DEPTH_VALIDATION.md (median 4.45x on n=38 wet "
            "sites, Mumbai, one season), ramped from 1x below 0.15 m where it was never "
            "measured. It corrects a systematic bias; it does not make the model able to rank "
            "which street floods worse — that failure is documented and unresolved."),
    }
    if truncated:
        meta["dropped"] = int(len(idx) - len(rows))
        log.info("streets_in_view: showing %d of %d in view (deepest first)",
                 len(rows), len(idx))
    return rows, meta


def summarise(rows):
    """Headline numbers for a viewport: worst street, how much is impassable."""
    if not rows:
        return {"n": 0}
    st = np.array([r["depth_street_mm"] for r in rows], dtype="float64")
    length = np.array([r["length_m"] for r in rows], dtype="float64")
    # thresholds people can act on, not model units
    bands = {"ankle_100mm": 100, "knee_400mm": 400, "impassable_600mm": 600}
    out = {
        "n": len(rows),
        "max_street_mm": int(st.max()),
        "median_street_mm": int(np.median(st)),
        "flooded_length_m": int(length.sum()),
        "ponded_share": round(float(sum(1 for r in rows if r.get("ponded")) / len(rows)), 3),
    }
    for name, mm in bands.items():
        out[f"length_over_{name}"] = int(length[st >= mm].sum())
    return out


def streets_for_area(rain_mm, work=None, bbox=None, max_segments=MAX_SEGMENTS,
                     min_depth=MIN_DEPTH_M, device="cpu"):
    """Bundle driver: emulate the storm, then answer the street question inside `bbox`.

    Reuses `flowfield`'s cached street graph and velocity field, so a viewport request after the
    first one for a bundle is vectorised arithmetic only.
    """
    import rasterio
    from ..config import CFG
    from . import flowfield
    from .emulator import load_emulator, whatif_grid

    work = work or CFG.work
    hmax, _dig, summary = whatif_grid(rain_mm, None, work=work, device=device)
    b = load_emulator(work, device)
    dom = b["dom"]
    z = dom.z0.cpu().numpy()
    u, v = flowfield.surface_velocity(hmax, z, dom.mann.cpu().numpy(), dom.dx)

    with rasterio.open(f"{work}/dem.tif") as src:
        T = src.transform
    net, segments = flowfield._road_net(work, dom, z, T)
    if net is None or segments is None or len(segments["row"]) == 0:
        return {"rain_mm": rain_mm, "streets": [],
                "meta": {"n_segments": 0, "n_returned": 0, "truncated": False,
                         "reason": "this bundle has no cached road graph"}}

    water = flowfield._permanent_water(dom)
    rows, meta = streets_in_view(segments, hmax, u, v, bbox=bbox, min_depth=min_depth,
                                 max_segments=max_segments, exclude=water,
                                 flag=flowfield._dilate(water, 1))
    rows = flowfield._name_streets(rows, work)
    return {"rain_mm": rain_mm, "streets": rows, "meta": meta,
            "summary": summarise(rows), "flood_km2": summary.get("flood_km2")}
