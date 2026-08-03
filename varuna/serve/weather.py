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


def historical_rain_mm(lat, lon, start_date, end_date):
    """Total observed precipitation (mm) over [start_date, end_date] inclusive, from Open-Meteo's
    archive (ERA5 reanalysis; free, no key). Dates are ISO 'YYYY-MM-DD'.

    Used by varuna.build.calibrate to force the twin with the real rain leading into a Sentinel-1
    overpass, so the simulated water extent is comparable to the SAR water mask of that day.
    """
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
           "&hourly=precipitation&timezone=Asia/Kolkata")
    js = http_get_json(url, timeout=60)
    p = js.get("hourly", {}).get("precipitation") or []
    total = float(sum(v or 0 for v in p))
    log.info("archive rain %s..%s @ (%.3f,%.3f): %.1f mm", start_date, end_date, lat, lon, total)
    return total


def hourly_rain_series(lat, lon, start_date, end_date):
    """Hourly precipitation (mm) over [start_date, end_date] inclusive -> (times, precip).

    `historical_rain_mm` sums the whole range, which is all the SAR calibration needs. Scoring a
    point observation needs the shape kept: "how much rain had fallen in the six hours BEFORE
    this person waded through it" is a different question from the daily total.

    The ERA5 archive lags roughly five days, so recent dates come from the forecast endpoint's
    `past_days` window instead. Both return the same hourly schema; the caller cannot tell.
    """
    import datetime as _dt
    lag_days = (_dt.date.today() - _dt.date.fromisoformat(str(end_date))).days
    if lag_days >= 6:
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
               "&hourly=precipitation&timezone=Asia/Kolkata")
    else:
        past = min(92, max(1, (_dt.date.today() - _dt.date.fromisoformat(str(start_date))).days + 1))
        url = ("https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}&hourly=precipitation"
               f"&past_days={past}&forecast_days=1&timezone=Asia/Kolkata")
    js = http_get_json(url, timeout=60)
    h = js.get("hourly") or {}
    times = h.get("time") or []
    precip = [float(v or 0) for v in (h.get("precipitation") or [])]
    return times, precip


def aoi_max_rain(aoi=None, hours=24):
    """Sample centre + 4 corners of the AOI and take the max (conservative for alerts)."""
    aoi = aoi or CFG.aoi
    pts = [((aoi[1] + aoi[3]) / 2, (aoi[0] + aoi[2]) / 2),
           (aoi[1], aoi[0]), (aoi[1], aoi[2]), (aoi[3], aoi[0]), (aoi[3], aoi[2])]
    p = max(forecast_rain_mm(la, lo, hours) for la, lo in pts)
    log.info("AOI max 24-h rainfall forecast: %.1f mm", p)
    return p


def forecast_hyetograph(lat, lon, hours=48):
    """Hourly forecast precipitation at one point, arrays kept (not summed away).

    Returns {"times": ISO hours, "precip_mm": mm/hr, "past24_mm": observed total of the
    last 24 h} — past_days=1 makes recent actual rain part of the same free call.
    """
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&hourly=precipitation"
           "&forecast_days=3&past_days=1&timezone=Asia/Kolkata")
    js = http_get_json(url, timeout=60)
    times = js["hourly"]["time"]
    p = [float(v or 0) for v in js["hourly"]["precipitation"]]
    # past_days=1 -> the first 24 entries are the previous day (observed/analysed)
    past24 = sum(p[:24])
    return {"times": times[24:24 + hours], "precip_mm": p[24:24 + hours],
            "past24_mm": round(past24, 1)}


def area_weather(aoi=None, center=None, hours=24):
    """One dashboard-ready live-weather dict for an area (the /api/weather payload).

    rain_24h_mm is the conservative AOI max (alerts use the same number); the hyetograph is
    sampled at the twin-crop centre where the dashboard's storm actually falls.
    """
    import datetime as _dt
    aoi = aoi or CFG.aoi
    center = center or CFG.center
    hyeto = forecast_hyetograph(center[0], center[1], hours=48)
    return dict(rain_24h_mm=round(aoi_max_rain(aoi, hours), 1),
                rain_center_24h_mm=round(sum(hyeto["precip_mm"][:24]), 1),
                rain_center_48h_mm=round(sum(hyeto["precip_mm"]), 1),
                past24_mm=hyeto["past24_mm"],
                hyetograph={"times": hyeto["times"], "precip_mm": hyeto["precip_mm"]},
                fetched_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                source="open-meteo")
