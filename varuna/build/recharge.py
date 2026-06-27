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


def load_groundwater(work=None):
    """Read gw_levels.csv; write the SAMPLE file (with a loud warning) if missing."""
    import pandas as pd
    work = work or CFG.work
    path = f"{work}/gw_levels.csv"
    if not os.path.exists(path):
        pd.DataFrame(SAMPLE_GW, columns=["station", "lat", "lon", "depth_to_water_m"]).to_csv(path, index=False)
        log.warning("WROTE SAMPLE gw_levels.csv with FAKE depths — recharge rankings are "
                    "meaningless until you replace it with real CGWB/India-WRIS data.")
    gw = pd.read_csv(path)
    gw["_is_sample"] = gw["station"].astype(str).str.startswith("SAMPLE")
    return gw


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


def compute_ksat(work=None, R=None, C=None):
    """Cosby (1984) pedotransfer: sand/clay % -> saturated conductivity (mm/hr)."""
    work = work or CFG.work
    sand_pct = read_aligned(f"{work}/sand.tif", R, C) / 10.0
    clay_pct = read_aligned(f"{work}/clay.tif", R, C) / 10.0
    ksat = (10 ** (-0.6 + 0.0126 * sand_pct - 0.0064 * clay_pct)) * 25.4
    return ksat


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

    gw = load_groundwater(work)
    gw_depth = idw(LON, LAT, gw[["lat", "lon"]].values, gw["depth_to_water_m"].values)
    ksat = compute_ksat(work, R, C)

    wc = read_aligned(f"{work}/worldcover.tif", R, C)
    pervious = (~np.isin(wc, [50, 80])).astype("float64")
    pervious = ndimage.uniform_filter(pervious, size=7)
    avail = np.log1p(read_aligned(f"{work}/acc.tif", R, C))

    def norm(a, lo=2, hi=98):
        a = a.astype("float64")
        p1, p2 = np.nanpercentile(a, lo), np.nanpercentile(a, hi)
        return np.clip((a - p1) / max(p2 - p1, 1e-9), 0, 1)

    rsi = (w_gw * norm(gw_depth) + w_ks * norm(ksat) + w_perv * pervious + w_avail * norm(avail))
    rsi = np.where(wc == 80, 0, rsi)
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
