"""Earth Engine initialisation + image download.

Interactive OAuth on first run (as in the notebooks); pass a service-account JSON to run
headlessly (cron / API). This is the one true GEE entry point — build modules call `init_ee()`
once and `download_ee_image()` for every layer.
"""
from __future__ import annotations

import logging

from .config import CFG
from .io import http_get

log = logging.getLogger("varuna.ee")
_initialized = False


def init_ee(project_id: str | None = None, service_account_json: str | None = None):
    """Initialise Earth Engine. Idempotent.

    service_account_json: path to a GCP service-account key for headless use. When omitted,
    falls back to interactive `ee.Authenticate()` exactly like the original notebooks.
    """
    global _initialized
    if _initialized:
        return
    import ee

    project = project_id or CFG.project_id
    if project in (None, "", "your-cloud-project-id"):
        raise ValueError(
            "Earth Engine project not set. Register at earthengine.google.com, create a Google "
            "Cloud project, then set VARUNA_PROJECT_ID (or project_id in config.yaml).")
    try:
        if service_account_json:
            import json
            with open(service_account_json) as f:
                email = json.load(f)["client_email"]
            creds = ee.ServiceAccountCredentials(email, service_account_json)
            ee.Initialize(creds, project=project)
        else:
            ee.Initialize(project=project)
    except Exception as e:  # noqa: BLE001
        log.info("ee.Initialize failed (%s); launching interactive auth", e)
        ee.Authenticate()
        ee.Initialize(project=project)
    _initialized = True
    log.info("Earth Engine ready (project=%s)", project)


def region(aoi=None):
    """ee.Geometry.Rectangle for the AOI."""
    import ee
    return ee.Geometry.Rectangle(list(aoi or CFG.aoi))


def download_ee_image(img, path, reg=None, scale=None, bands=None):
    """Download a (small) EE image as a GeoTIFF via getDownloadURL, with retries."""
    if bands is not None:
        img = img.select(bands)
    url = img.getDownloadURL({
        "region": reg if reg is not None else region(),
        "scale": scale or CFG.scale,
        "format": "GEO_TIFF",
        "crs": "EPSG:4326",
    })
    r = http_get(url, timeout=600)
    with open(path, "wb") as f:
        f.write(r.content)
    log.info("saved %s (%.1f MB)", path, len(r.content) / 1e6)
    return path
