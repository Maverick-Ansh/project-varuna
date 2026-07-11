"""NASA Black Marble (VIIRS VNP46A2) night lights -> flood power-outage layer.

Outage logic: a built-up 500 m cell whose latest good-quality radiance drops more than
`drop_frac` below its 90-day median baseline is flagged dark. The gap-filled BRDF-corrected
band + Mandatory_Quality_Flag masking keeps moonlight and clouds from faking outages; a
baseline floor keeps never-lit cells (mangroves, sea) out of the denominator.

Runs where earthengine-api is available (Kaggle/Colab nightly job — NEVER on the Space;
the Space only serves the committed nightlights_outage.json + nightlights.png).
The GEE archive lags a few days — the JSON carries the actual data date and the dashboard
must show it ("power status as of <date>"), never imply real-time.

compute_outage() is pure numpy so the logic is offline-testable without EE.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os

import numpy as np

from ..config import CFG
from ..io import save_json

log = logging.getLogger("varuna.build.nightlights")

NTL_BAND = "Gap_Filled_DNB_BRDF_Corrected_NTL"
QF_BAND = "Mandatory_Quality_Flag"          # 0/1 = good, 2 = poor, 255 = no retrieval
SCALE_M = 500


def compute_outage(latest, baseline, quality=None, built=None,
                   min_baseline=1.0, drop_frac=0.5):
    """Pure-numpy outage detection. Returns (outage_mask, drop_pct, stats).

    latest/baseline: radiance grids (nW/cm^2/sr); quality: latest-composite quality grid
    (cells >= 2 or 255 are untrusted); built: {0..1} built-up fraction (cells < 0.05 skipped).
    """
    latest = np.asarray(latest, dtype="float64")
    baseline = np.asarray(baseline, dtype="float64")
    valid = np.isfinite(latest) & np.isfinite(baseline) & (baseline >= min_baseline)
    if quality is not None:
        q = np.asarray(quality, dtype="float64")
        valid &= np.isfinite(q) & (q < 2)
    if built is not None:
        valid &= np.asarray(built, dtype="float64") >= 0.05
    drop = np.zeros_like(baseline)
    np.divide(baseline - latest, baseline, out=drop, where=baseline > 0)
    outage = valid & (drop > drop_frac)
    stats = dict(n_valid=int(valid.sum()), n_outage=int(outage.sum()),
                 pct_built_dark=round(100.0 * outage.sum() / max(valid.sum(), 1), 2),
                 median_drop_pct=round(float(np.median(drop[outage]) * 100), 1)
                 if outage.any() else 0.0)
    return outage, drop, stats


def _read(path):
    import rasterio
    with rasterio.open(path) as s:
        return s.read(1).astype("float64"), s.transform


def _built_fraction(work, shape, transform):
    """Fraction of each 500 m night-lights cell that is WorldCover built-up (class 50)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(os.path.join(work, "worldcover.tif")) as s:
        built30 = (s.read(1) == 50).astype("float32")
        src_transform, src_crs = s.transform, s.crs
    out = np.zeros(shape, dtype="float32")
    reproject(built30, out, src_transform=src_transform, src_crs=src_crs,
              dst_transform=transform, dst_crs=src_crs or "EPSG:4326",
              resampling=Resampling.average)
    return out


def fetch_vnp46a2(work, aoi=None, days_back=90, good_days=3, max_lag_days=14):
    """Download latest-good + baseline VNP46A2 composites for the AOI (needs init_ee done).

    Writes ntl_latest.tif / ntl_baseline.tif / ntl_quality.tif into the bundle; returns the
    latest composite's most recent image date (ISO) or raises if the archive has nothing
    fresh enough.
    """
    import ee
    from ..ee_auth import download_ee_image, region

    aoi = aoi or CFG.aoi
    reg = region(aoi)
    today = _dt.date.today()
    col = (ee.ImageCollection("NASA/VIIRS/002/VNP46A2")
           .filterBounds(reg)
           .filterDate(str(today - _dt.timedelta(days=days_back + max_lag_days)),
                       str(today + _dt.timedelta(days=1))))

    def good_ntl(img):
        q = img.select(QF_BAND)
        return (img.select(NTL_BAND)
                .updateMask(q.lt(2))
                .copyProperties(img, ["system:time_start"]))

    goodc = col.map(good_ntl)
    recent = goodc.sort("system:time_start", False).limit(good_days)
    n_recent = recent.size().getInfo()
    if n_recent == 0:
        raise RuntimeError(f"no VNP46A2 images in the last {days_back + max_lag_days} d for {aoi}")
    latest_ms = ee.Date(recent.first().get("system:time_start")).format("YYYY-MM-dd").getInfo()
    lag = (today - _dt.date.fromisoformat(latest_ms)).days
    if lag > max_lag_days:
        raise RuntimeError(f"VNP46A2 archive is {lag} d stale (> {max_lag_days})")

    latest = recent.median()
    baseline = goodc.filterDate(str(today - _dt.timedelta(days=days_back + max_lag_days)),
                                str(_dt.date.fromisoformat(latest_ms)
                                    - _dt.timedelta(days=good_days))).median()
    quality = (col.sort("system:time_start", False).limit(good_days)
               .select(QF_BAND).reduce(ee.Reducer.min()))

    download_ee_image(latest, os.path.join(work, "ntl_latest.tif"), reg=reg, scale=SCALE_M)
    download_ee_image(baseline, os.path.join(work, "ntl_baseline.tif"), reg=reg, scale=SCALE_M)
    download_ee_image(quality, os.path.join(work, "ntl_quality.tif"), reg=reg, scale=SCALE_M)
    log.info("VNP46A2 for %s: latest good night %s (lag %d d)", work, latest_ms, lag)
    return latest_ms


def _overlay_png(path, outage, drop, upscale=8):
    """Crisp red overlay PNG (transparent where powered) for the Leaflet layer."""
    from PIL import Image
    o = np.kron(outage.astype("uint8"), np.ones((upscale, upscale), dtype="uint8"))
    d = np.kron(np.clip(drop, 0, 1), np.ones((upscale, upscale)))
    rgba = np.zeros((*o.shape, 4), dtype="uint8")
    rgba[..., 0] = 239                                  # red-ish (#ef4444 family)
    rgba[..., 1] = 68
    rgba[..., 2] = 68
    rgba[..., 3] = np.where(o > 0, (120 + 100 * d).astype("uint8"), 0)
    Image.fromarray(rgba, "RGBA").save(path)


def update_area(work, aoi=None, drop_frac=0.5, min_baseline=1.0, fetch=True):
    """Full per-area refresh: fetch composites (optional), detect outages, write artifacts.

    Writes {work}/nightlights_outage.json + {work}/nightlights.png. Returns the JSON dict.
    """
    date = None
    if fetch:
        date = fetch_vnp46a2(work, aoi=aoi)
    latest, T = _read(os.path.join(work, "ntl_latest.tif"))
    baseline, _ = _read(os.path.join(work, "ntl_baseline.tif"))
    quality = None
    qp = os.path.join(work, "ntl_quality.tif")
    if os.path.exists(qp):
        quality, _ = _read(qp)
    built = _built_fraction(work, latest.shape, T)
    outage, drop, stats = compute_outage(latest, baseline, quality, built,
                                         min_baseline=min_baseline, drop_frac=drop_frac)

    rows, cols = np.nonzero(outage)
    cells = [dict(lat=round(T.f + T.e * (r + 0.5), 5), lon=round(T.c + T.a * (c + 0.5), 5),
                  drop_pct=round(float(drop[r, c]) * 100, 1))
             for r, c in zip(rows.tolist(), cols.tolist())]
    h, w = latest.shape
    bounds = [[T.f + T.e * h, T.c], [T.f, T.c + T.a * w]]
    bounds = [[min(bounds[0][0], bounds[1][0]), min(bounds[0][1], bounds[1][1])],
              [max(bounds[0][0], bounds[1][0]), max(bounds[0][1], bounds[1][1])]]
    js = dict(date=date, updated_at=_dt.datetime.now(_dt.timezone.utc)
              .isoformat(timespec="seconds"),
              product="NASA Black Marble VNP46A2 (gap-filled BRDF-corrected, 500 m)",
              drop_frac=drop_frac, min_baseline=min_baseline, baseline_days=90,
              n_outage_cells=stats["n_outage"], pct_built_dark=stats["pct_built_dark"],
              median_drop_pct=stats["median_drop_pct"], n_valid_cells=stats["n_valid"],
              bounds=bounds, cells=cells, image="nightlights.png",
              note=("Dark = built-up 500 m cell whose latest good-quality night radiance is "
                    f">{int(drop_frac * 100)}% below its 90-day median. Data date above — "
                    "the satellite archive lags a few days; this is NOT real-time."))
    _overlay_png(os.path.join(work, "nightlights.png"), outage, drop)
    save_json(os.path.join(work, "nightlights_outage.json"), js)
    log.info("night lights %s: %d dark cells (%.1f%% of lit built land)", work,
             stats["n_outage"], stats["pct_built_dark"])
    return js
