"""Citizen reports -> supervision labels for the emulator fine-tune.

A report is only a usable label together with the storm that caused it, so reports are
grouped by IST calendar day and each day is tagged with the actual rain that fell (Open-Meteo:
ERA5 archive for older days, the forecast API's past_days window for the last week — the
archive lags ~5 days). Dry-day reports are dropped (wet report + <5 mm rain is far more
likely mischief or a burst pipe than a storm signal), and multiple reports in the same cell
on the same day collapse to their median depth so one street corner can't dominate the loss.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging

from ..io import http_get_json
from ..serve.reports import DEPTH_M

log = logging.getLogger("varuna.learn.labels")

IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
MIN_RAIN_MM = 5.0


def rain_for_day(lat, lon, day):
    """Actual rain (mm) that fell on IST calendar day `day` at (lat, lon).

    Recent days (ERA5 archive lag) come from the forecast API's past_days backfill;
    older days from the archive endpoint.
    """
    day = _dt.date.fromisoformat(str(day))
    age = (_dt.date.today() - day).days
    if age <= 6:
        url = ("https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}&hourly=precipitation"
               "&past_days=7&forecast_days=1&timezone=Asia/Kolkata")
        js = http_get_json(url, timeout=60)
        total = sum(float(v or 0)
                    for t, v in zip(js["hourly"]["time"], js["hourly"]["precipitation"])
                    if t.startswith(day.isoformat()))
        return float(total)
    from ..serve.weather import historical_rain_mm
    return float(historical_rain_mm(lat, lon, day.isoformat(), day.isoformat()))


def collect_labels(reports, center, days_back=30, min_rain_mm=MIN_RAIN_MM, rain_fn=None):
    """Reports (dicts with ts/cell/depth_m/depth_band) -> [{date, rain_mm, points}].

    points = [(row, col, depth_m, band)]; per (cell, day) the median depth wins.
    rain_fn(day) is injectable for offline tests; defaults to rain_for_day at `center`.
    """
    rain_fn = rain_fn or (lambda d: rain_for_day(center[0], center[1], d))
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days_back)
    by_day = {}
    for rep in reports:
        if not rep.get("cell"):
            continue
        try:
            ts = _dt.datetime.fromisoformat(rep["ts"].replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            continue
        if ts < cutoff:
            continue
        day = ts.astimezone(IST).date().isoformat()
        key = (day, tuple(rep["cell"]))
        by_day.setdefault(day, {}).setdefault(key, []).append(rep)

    labelset = []
    for day in sorted(by_day):
        try:
            rain = rain_fn(day)
        except Exception as e:  # noqa: BLE001
            log.warning("rain lookup failed for %s: %s — day dropped", day, e)
            continue
        if rain < min_rain_mm:
            log.info("day %s: %.1f mm < %.0f mm — %d reports dropped (dry-day filter)",
                     day, rain, min_rain_mm, sum(len(v) for v in by_day[day].values()))
            continue
        points = []
        for (d, cell), reps in by_day[day].items():
            depths = sorted(r["depth_m"] for r in reps)
            med = depths[len(depths) // 2]
            band = next((b for b, m in DEPTH_M.items() if abs(m - med) < 1e-6), "knee")
            points.append((int(cell[0]), int(cell[1]), float(med), band))
        if points:
            labelset.append(dict(date=day, rain_mm=float(rain), points=points))
    log.info("labels: %d storm-days, %d points",
             len(labelset), sum(len(d["points"]) for d in labelset))
    return labelset


def split_holdout(labelset, seed=0, holdout_frac=0.34):
    """Deterministic DAY-wise split (never by point — points within a day are correlated)."""
    if len(labelset) < 2:
        return labelset, []
    scored = sorted(labelset,
                    key=lambda d: hashlib.sha256(f"{seed}:{d['date']}".encode()).hexdigest())
    n_hold = max(1, int(round(len(labelset) * holdout_frac)))
    n_hold = min(n_hold, len(labelset) - 1)
    hold = scored[:n_hold]
    train = scored[n_hold:]
    return (sorted(train, key=lambda d: d["date"]),
            sorted(hold, key=lambda d: d["date"]))
