"""I/O helpers: GeoTIFF read/write, retrying HTTP, artifact-bundle validation.

Centralises the raster helpers and `requests` calls that the notebooks scattered (and that
crashed hard on a single failed download). Everything here is dependency-light and CPU-only.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

log = logging.getLogger("varuna.io")

# Artifacts produced by the build stage. Required ones gate the serve/agent layers.
REQUIRED_ARTIFACTS = ["dem.tif", "depth.tif", "worldcover.tif", "catchment_labels.tif", "sinks.csv"]
OPTIONAL_ARTIFACTS = [
    "acc.tif", "jrc_occurrence.tif", "sand.tif", "clay.tif", "rsi.tif", "recharge_sites.csv",
    "waterlogging_frequency.tif", "validation_scores.json", "emulator.pt", "twin_meta.pt",
]


def read1(path):
    """Read band 1 of a GeoTIFF -> (array, transform, height, width)."""
    import rasterio
    with rasterio.open(path) as s:
        return s.read(1), s.transform, s.height, s.width


def read_aligned(path, R, C):
    """Read band 1 and crop/pad to exactly (R, C) — downloads can differ by a pixel."""
    a = read1(path)[0].astype("float64")
    if a.shape[0] >= R and a.shape[1] >= C:
        return a[:R, :C]
    return np.pad(a, ((0, max(0, R - a.shape[0])), (0, max(0, C - a.shape[1]))), mode="edge")[:R, :C]


def write_raster(path, array, transform, dtype=None, crs="EPSG:4326"):
    """Write a single-band GeoTIFF."""
    import rasterio
    a = array if dtype is None else array.astype(dtype)
    meta = dict(driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                crs=crs, transform=transform, dtype=str(a.dtype))
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(a, 1)
    return path


def http_get(url, timeout=60, retries=3, backoff=2.0):
    """GET with retry + backoff. Raises RuntimeError after the final attempt."""
    import requests
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - we retry then re-raise wrapped
            last = e
            log.warning("GET failed (%d/%d): %s", i + 1, retries, e)
            if i < retries - 1:
                time.sleep(backoff * (i + 1))
    raise RuntimeError(f"GET {url[:80]}... failed after {retries} tries: {last}")


def http_get_json(url, **kw):
    return http_get(url, **kw).json()


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def bundle_status(work):
    """Map artifact filename -> bool(present)."""
    return {f: os.path.exists(os.path.join(work, f))
            for f in REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS}


def require_bundle(work, needed=None):
    """Raise a clear, actionable error if required artifacts are missing."""
    needed = needed or REQUIRED_ARTIFACTS
    missing = [f for f in needed if not os.path.exists(os.path.join(work, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {work}: {missing}. Run the build stage first "
            f"(notebooks/00_setup_build.ipynb or `python -m varuna build`).")
