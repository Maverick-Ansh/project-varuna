"""Notebook 03 -> functions: Sentinel-1 SAR ground-truth loop.

Detect observed standing water, score nb01's prediction (POD/FAR/CSI), and build a multi-year
waterlogging-frequency climatology. The CSI score is the project's credibility metric.

Outputs (into CFG.work): observed_water_<date>.tif, waterlogging_frequency.tif,
validation_scores.json.
"""
from __future__ import annotations

import datetime as _dt
import logging

import numpy as np

from ..config import CFG
from ..ee_auth import download_ee_image, init_ee, region
from ..io import read1, save_json

log = logging.getLogger("varuna.build.validate")


def _s1(reg, start, end):
    """Sentinel-1 IW VV collection, speckle-filtered, in dB."""
    import ee
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(reg).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .select("VV"))
    return col.map(lambda im: im.focal_median(50, "circle", "meters")
                   .copyProperties(im, ["system:time_start"]))


def list_passes(year=2025, reg=None):
    """Return available Sentinel-1 monsoon acquisition dates (ISO strings) for picking EVENT_DATE."""
    reg = reg if reg is not None else region()
    monsoon = _s1(reg, f"{year}-06-01", f"{year}-10-15")
    ts = monsoon.aggregate_array("system:time_start").getInfo()
    return [_dt.datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d") for t in ts]


def observed_water(event_date, work=None, dry_year=2025, reg=None):
    """Map standing water for one event date; writes observed_water_<date>.tif."""
    import ee
    work = work or CFG.work
    reg = reg if reg is not None else region()
    dry_ref = _s1(reg, f"{dry_year}-02-01", f"{dry_year}-03-31").median()
    d0 = _dt.date.fromisoformat(event_date)
    event_img = _s1(reg, str(d0 - _dt.timedelta(days=1)), str(d0 + _dt.timedelta(days=2))).mosaic()
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    water = (event_img.lt(-16).Or(event_img.subtract(dry_ref).lt(-4))
             .updateMask(jrc.lt(50)).unmask(0)
             .focal_mode(60, "circle", "meters"))
    path = f"{work}/observed_water_{event_date}.tif"
    download_ee_image(water, path, reg)
    return path, dry_ref


def score(event_date, work=None):
    """POD / FAR / CSI of nb01's depth>0.15 prediction vs observed water."""
    work = work or CFG.work
    obs = read1(f"{work}/observed_water_{event_date}.tif")[0]
    pred_depth = read1(f"{work}/depth.tif")[0]
    r = min(obs.shape[0], pred_depth.shape[0])
    c = min(obs.shape[1], pred_depth.shape[1])
    obs = obs[:r, :c] > 0.5
    pred = pred_depth[:r, :c] > CFG.min_depth_m
    hits = int((obs & pred).sum())
    misses = int((obs & ~pred).sum())
    fa = int((~obs & pred).sum())
    scores = dict(event_date=event_date,
                  pod=hits / max(hits + misses, 1),
                  far=fa / max(hits + fa, 1),
                  csi=hits / max(hits + misses + fa, 1),
                  hits=hits, misses=misses, false_alarms=fa)
    save_json(f"{work}/validation_scores.json", scores)
    log.info("POD=%.2f FAR=%.2f CSI=%.2f", scores["pod"], scores["far"], scores["csi"])
    return scores


def frequency_climatology(work=None, years=None, dry_ref=None, reg=None):
    """Count how often each pixel is wet across monsoon passes -> waterlogging_frequency.tif."""
    import ee
    work = work or CFG.work
    years = years or [2023, 2024, 2025]
    reg = reg if reg is not None else region()
    if dry_ref is None:
        dry_ref = _s1(reg, "2025-02-01", "2025-03-31").median()
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    freq = ee.Image(0).rename("freq")
    npass = 0
    for y in years:
        col = _s1(reg, f"{y}-06-01", f"{y}-10-15")
        nimg = col.size().getInfo()
        if nimg == 0:
            continue
        wet = col.map(lambda im: im.lt(-16).Or(im.subtract(dry_ref).lt(-4)).unmask(0))
        freq = freq.add(wet.sum())
        npass += nimg
    freq = freq.divide(max(npass, 1)).updateMask(jrc.lt(50))
    download_ee_image(freq, f"{work}/waterlogging_frequency.tif", reg)
    log.info("frequency map from %d radar passes", npass)
    return npass


def run(event_date=None, work=None, project_id=None, years=None, do_climatology=True):
    """Full nb03 pipeline. event_date=None -> only lists available passes (manual pick required)."""
    work = work or CFG.work
    from ..io import require_bundle
    require_bundle(work, ["depth.tif"])
    init_ee(project_id)
    reg = region()
    if event_date is None:
        passes = list_passes(reg=reg)
        log.info("Pick EVENT_DATE from these Sentinel-1 passes: %s", passes)
        return {"passes": passes}
    _, dry_ref = observed_water(event_date, work, reg=reg)
    scores = score(event_date, work)
    if do_climatology:
        frequency_climatology(work, years, dry_ref, reg)
    return scores
