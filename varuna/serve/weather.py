"""Rainfall input (notebook 04, cell 1).

Open-Meteo (free, no key) 24-h precipitation. To swap in IMD nowcasts, replace
`forecast_rain_mm` with your own function returning total mm over `hours` — the rest of the
pipeline is agnostic to the source.
"""
from __future__ import annotations

import logging

from ..config import CFG
from ..io import http_get_json

log = logging.getLogger("varuna.serve.weather")


def forecast_rain_mm(lat, lon, hours=24):
    """Total forecast precipitation (mm) over the next `hours` at one point."""
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&hourly=precipitation&forecast_days=2&timezone=Asia/Kolkata")
    js = http_get_json(url, timeout=60)
    p = js["hourly"]["precipitation"][:hours]
    return float(sum(v or 0 for v in p))


def aoi_max_rain(aoi=None, hours=24):
    """Sample centre + 4 corners of the AOI and take the max (conservative for alerts)."""
    aoi = aoi or CFG.aoi
    pts = [((aoi[1] + aoi[3]) / 2, (aoi[0] + aoi[2]) / 2),
           (aoi[1], aoi[0]), (aoi[1], aoi[2]), (aoi[3], aoi[0]), (aoi[3], aoi[2])]
    p = max(forecast_rain_mm(la, lo, hours) for la, lo in pts)
    log.info("AOI max 24-h rainfall forecast: %.1f mm", p)
    return p
