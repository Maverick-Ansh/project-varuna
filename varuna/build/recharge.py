"""Notebook 02 -> functions: Recharge Suitability Index (RSI) + ranked recharge sites.

RSI = weighted mix of depth-to-water, soil permeability (Cosby Ksat), pervious fraction and
flow-accumulation availability. Sites are scored as volume x RSI.

Outputs (into CFG.work): sand.tif, clay.tif, rsi.tif, rsi_overlay.png, recharge_map.html,
recharge_sites.csv.

NOTE: requires a real pre-monsoon groundwater CSV (gw_levels.csv) from India-WRIS. A SAMPLE
file is written if absent so the pipeline runs, but rankings are MEANINGLESS until replaced.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from ..config import CFG
from ..ee_auth import download_ee_image, init_ee, region
from ..io import read1, read_aligned, write_raster

log = logging.getLogger("varuna.build.recharge")

SAMPLE_GW = [
    # SAMPLE / PLACEHOLDER VALUES — NOT REAL MEASUREMENTS. Replace with India-WRIS export.
    ("SAMPLE_Danapur", 25.633, 85.046, 8.5),
    ("SAMPLE_Phulwari", 25.575, 85.080, 7.2),
    ("SAMPLE_PatnaSadar", 25.610, 85.140, 6.0),
    ("SAMPLE_Sampatchak", 25.555, 85.180, 9.1),
    ("SAMPLE_PatnaCityE", 25.595, 85.230, 5.4),
    ("SAMPLE_Fatuha", 25.560, 85.290, 10.3),
]

# Every recharge payload ships this. It began as notebook-02 markdown, got dropped in the port,
# and the omission is exactly how a prioritisation quietly becomes a certification.
QUALITY_CAVEAT = (
    "Recharge siting requires geotechnical and water-quality screening before construction — "
    "this ranking prioritises, it does not certify. Region-specific risks: parts of Bihar's "
    "shallow alluvium carry arsenic; Karnataka's hard-rock (gneiss) aquifers carry fluoride; "
    "urban runoff carries sewage and heavy metals."
)

# A groundwater station farther than this from the area centre cannot support an IDW field
# inside it. Patna's wells sat ~1,600 km from Bengaluru's bundle and still ranked its sites.
GW_MAX_STATION_KM = 50.0


def station_km(center, lats, lons):
    """Great-circle (haversine) distance in km from center=(lat, lon) to each station."""
    lat0, lon0 = np.radians(center[0]), np.radians(center[1])
    lat, lon = np.radians(np.asarray(lats, dtype="float64")), np.radians(np.asarray(lons, dtype="float64"))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def bundle_center(work=None):
    """(lat, lon) of the bundle's raster mid-point from depth.tif — the bundle itself is the
    ground truth of where it sits, so provenance checks can't be fooled by a stale registry."""
    import rasterio
    work = work or CFG.work
    with rasterio.open(f"{work}/depth.tif") as s:
        T, h, w = s.transform, s.height, s.width
    lon, lat = T * (w / 2, h / 2)
    return float(lat), float(lon)


def load_groundwater(work=None, center=None, max_km=GW_MAX_STATION_KM):
    """Read gw_levels.csv; write the SAMPLE file (with a loud warning) if missing.

    When `center` is given, stations get a `_km_to_center` column and the AOI sanity gate runs:
    if no station falls within `max_km`, every row is forced `_is_sample=True` — an IDW field
    built entirely from far-away wells is structured noise, not data.
    """
    import pandas as pd
    work = work or CFG.work
    path = f"{work}/gw_levels.csv"
    if not os.path.exists(path):
        pd.DataFrame(SAMPLE_GW, columns=["station", "lat", "lon", "depth_to_water_m"]).to_csv(path, index=False)
        log.warning("WROTE SAMPLE gw_levels.csv with FAKE depths — recharge rankings are "
                    "meaningless until you replace it with real CGWB/India-WRIS data.")
    gw = pd.read_csv(path)
    gw["_is_sample"] = gw["station"].astype(str).str.startswith("SAMPLE")
    if center is not None and len(gw):
        gw["_km_to_center"] = station_km(center, gw["lat"].values, gw["lon"].values)
        nearest = float(gw["_km_to_center"].min())
        if nearest > max_km:
            gw["_is_sample"] = True
            log.warning("NO groundwater station within %.0f km of area centre %s (nearest %.0f km) "
                        "— treating gw_levels.csv as SAMPLE; recharge rankings are meaningless "
                        "for this area.", max_km, tuple(round(c, 3) for c in center), nearest)
    return gw


def groundwater_status(work=None, center=None, max_km=GW_MAX_STATION_KM):
    """Provenance summary of the groundwater input behind RSI / recharge ranks.

    `sample` is True if ANY station is a SAMPLE placeholder or none sits within `max_km` of the
    centre — a partially fake input still taints the ranking. Centre defaults to the bundle's
    own raster mid-point; `aoi_check` says whether the distance gate actually ran (absence of a
    check is not a pass — same three-state doctrine as nightlights).
    """
    work = work or CFG.work
    if center is None:
        try:
            center = bundle_center(work)
        except Exception as e:  # noqa: BLE001 — no rasterio / no depth.tif: skip gate, say so
            log.warning("groundwater_status: no bundle centre (%s); AOI gate skipped", e)
    gw = load_groundwater(work, center=center, max_km=max_km)
    n_placeholder = int(gw["_is_sample"].sum()) if len(gw) else 0
    nearest = float(gw["_km_to_center"].min()) if "_km_to_center" in gw else None
    sample = bool(gw["_is_sample"].any()) or not len(gw)
    if not len(gw):
        reason = "gw_levels.csv has no stations"
    elif gw["station"].astype(str).str.startswith("SAMPLE").any():
        reason = "gw_levels.csv contains SAMPLE placeholder stations (not real measurements)"
    elif nearest is not None and nearest > max_km:
        reason = f"no groundwater station within {max_km:.0f} km of area centre (nearest {nearest:.0f} km)"
    else:
        reason = None
    return {"sample": sample, "reason": reason, "n_stations": int(len(gw)),
            "n_placeholder": n_placeholder, "nearest_station_km": nearest,
            "aoi_check": "ok" if center is not None else "skipped — bundle centre unavailable",
            "caveat": QUALITY_CAVEAT}


def idw(LON, LAT, pts, vals, power=2.0):
    """Inverse-distance-weighted interpolation of station values onto a grid."""
    num = np.zeros(LON.shape)
    den = np.zeros(LON.shape)
    for (plat, plon), v in zip(pts, vals):
        d2 = (LON - plon) ** 2 + (LAT - plat) ** 2 + 1e-12
        w = 1.0 / d2 ** (power / 2)
        num += w * v
        den += w
    return num / den


def download_soil(work=None, aoi=None, scale=None):
    """SoilGrids sand/clay mean (0-30 cm) -> sand.tif, clay.tif."""
    import ee
    work = work or CFG.work
    reg = region(aoi)
    scale = scale or CFG.scale
    sand = ee.Image("projects/soilgrids-isric/sand_mean").select(
        ["sand_0-5cm_mean", "sand_5-15cm_mean", "sand_15-30cm_mean"]).reduce(ee.Reducer.mean())
    clay = ee.Image("projects/soilgrids-isric/clay_mean").select(
        ["clay_0-5cm_mean", "clay_5-15cm_mean", "clay_15-30cm_mean"]).reduce(ee.Reducer.mean())
    download_ee_image(sand, f"{work}/sand.tif", reg, scale)
    download_ee_image(clay, f"{work}/clay.tif", reg, scale)


def cosby_ksat(sand_pct, clay_pct):
    """Cosby (1984) pedotransfer: sand/clay % -> saturated conductivity (mm/hr).

    The formula itself, separated from raster I/O so the state screen can apply it to
    per-block mean sand/clay tables without downloading a single pixel.
    """
    return (10 ** (-0.6 + 0.0126 * np.asarray(sand_pct) - 0.0064 * np.asarray(clay_pct))) * 25.4


def compute_ksat(work=None, R=None, C=None):
    """Cosby Ksat over the bundle's SoilGrids rasters (values are g/kg, hence /10 -> %)."""
    work = work or CFG.work
    sand_pct = read_aligned(f"{work}/sand.tif", R, C) / 10.0
    clay_pct = read_aligned(f"{work}/clay.tif", R, C) / 10.0
    return cosby_ksat(sand_pct, clay_pct)


def compute_rsi(work=None, weights=None):
    """Combine depth-to-water, Ksat, pervious fraction, availability -> rsi.tif. Returns rsi array."""
    from scipy import ndimage
    work = work or CFG.work
    w_gw, w_ks, w_perv, w_avail = weights or CFG.rsi_weights

    depth, transform, R, C = read1(f"{work}/depth.tif")
    cols = np.arange(C) + 0.5
    rows = np.arange(R) + 0.5
    lons = transform.c + cols * transform.a
    lats = transform.f + rows * transform.e
    LON, LAT = np.meshgrid(lons, lats)

    # centre from the grid being ranked, so building e.g. Bengaluru on Patna wells warns loudly
    gw = load_groundwater(work, center=(float(np.mean(lats)), float(np.mean(lons))))
    gw_depth = idw(LON, LAT, gw[["lat", "lon"]].values, gw["depth_to_water_m"].values)
    ksat = compute_ksat(work, R, C)

    # canonical pervious set — the old ~isin([50, 80]) counted saturated wetland (90),
    # mangrove (95) and nodata as pervious, which flattered RSI on exactly the wrong ground
    from .landcover import NO_RECHARGE, PERVIOUS
    wc = read_aligned(f"{work}/worldcover.tif", R, C)
    pervious = np.isin(wc, sorted(PERVIOUS)).astype("float64")
    pervious = ndimage.uniform_filter(pervious, size=7)
    avail = np.log1p(read_aligned(f"{work}/acc.tif", R, C))

    def norm(a, lo=2, hi=98):
        a = a.astype("float64")
        p1, p2 = np.nanpercentile(a, lo), np.nanpercentile(a, hi)
        return np.clip((a - p1) / max(p2 - p1, 1e-9), 0, 1)

    rsi = (w_gw * norm(gw_depth) + w_ks * norm(ksat) + w_perv * pervious + w_avail * norm(avail))
    rsi = np.where(np.isin(wc, sorted(NO_RECHARGE)), 0, rsi)
    write_raster(f"{work}/rsi.tif", rsi.astype("float32"), transform, dtype="float32")
    return rsi, gw_depth, ksat, (R, C)


def rank_recharge_sites(rsi, gw_depth, ksat, shape, work=None):
    """Score each nb01 sink by volume x local RSI -> recharge_sites.csv."""
    import pandas as pd
    work = work or CFG.work
    R, C = shape
    sinks = pd.read_csv(f"{work}/sinks.csv")

    def rsi_at(row, col, win=3):
        r0, r1 = max(0, row - win), min(R, row + win + 1)
        c0, c1 = max(0, col - win), min(C, col + win + 1)
        return float(np.nanmean(rsi[r0:r1, c0:c1]))

    sinks["rsi"] = [rsi_at(int(rr), int(cc)) for rr, cc in zip(sinks.row, sinks.col)]
    sinks["gw_depth_m"] = [float(gw_depth[int(rr), int(cc)]) for rr, cc in zip(sinks.row, sinks.col)]
    sinks["ksat_mm_hr"] = [float(ksat[int(rr), int(cc)]) for rr, cc in zip(sinks.row, sinks.col)]
    sinks["recharge_score"] = sinks.volume_m3 * sinks.rsi
    ranked = sinks.sort_values("recharge_score", ascending=False).reset_index(drop=True)
    ranked.to_csv(f"{work}/recharge_sites.csv", index=False)
    return ranked


def run(work=None, aoi=None, project_id=None, skip_download=False):
    """Full nb02 pipeline. Returns ranked recharge sites DataFrame."""
    work = work or CFG.work
    from ..io import require_bundle
    require_bundle(work, ["sinks.csv", "depth.tif", "acc.tif", "worldcover.tif"])
    init_ee(project_id)
    if not skip_download:
        download_soil(work, aoi)
    rsi, gw_depth, ksat, shape = compute_rsi(work)
    ranked = rank_recharge_sites(rsi, gw_depth, ksat, shape, work)
    log.info("nb02 done: %d recharge sites -> %s/recharge_sites.csv", len(ranked), work)
    return ranked
