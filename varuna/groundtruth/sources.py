"""Adapters that turn real-world feeds into `Observation` rows — with the quality control.

MUMBAI FLOOD (mumbaiflood.in, IIT Bombay IDP Climate Studies + MCGM/MCMCR)
    `/cs/map/?days=N`   crowdsourced water depth: a person's height in feet+inches, a
                        `water_level_factor` (the fraction of themselves the water reached) and
                        a rendered `water_level` string. depth = factor x height.
    `/aws/stations/`    35 municipal automatic weather stations, with coordinates and live rain.
    `/aws/hourly-aws-data/<id>/`  that station's rainfall by hour — TODAY ONLY, no history.

The crowdsourced feed is real data with real data problems, and the QC below is the difference
between a validation set and a pile of numbers. Measured on the 147-row pull of 2026-08-03:

  * points land as far away as Argentina and Australia — the app is reachable from anywhere and
    people tap the map. Anything outside the target region goes.
  * 41 rows carry `water_level: null` — the report exists, the depth does not.
  * some rows report the water reaching the reporter's own full height (factor ~1.0). Chest-deep
    is real; "the water was exactly as tall as me" is a slider pushed to its stop.
  * the same person submits twice within a minute (two rows, identical depth and place).

Each rule is counted, never silent: `ObservationSet.rejected` carries the ledger so the size of
the surviving set can always be checked against what it was filtered from.

PRIVACY: `name` and `feedback` are read and discarded here. They never enter an Observation.
"""
from __future__ import annotations

import logging
from collections import Counter

from .observations import Observation, ObservationSet

log = logging.getLogger("varuna.groundtruth.sources")

MUMBAIFLOOD_API = "https://api.mumbaiflood.in"
USER_AGENT = ("varuna-floodtwin/1.0 research validation "
              "(github.com/Maverick-Ansh/project-varuna)")

# Greater Mumbai plus a margin. The feed is globally reachable, so a region gate is mandatory.
MUMBAI_BBOX = (18.85, 72.70, 19.35, 73.10)          # south, west, north, east

MAX_BODY_FRACTION = 0.95      # water at ~100% of the reporter's height = a maxed-out slider
MIN_DEPTH_M = 0.01
MAX_DEPTH_M = 3.0             # deeper than any credible standing water on a street


def _get_json(url, timeout=60):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()


def fetch_crowdsourced(days=365, api=MUMBAIFLOOD_API):
    """Raw crowdsourced depth rows. `days` is the only filter the API honours (others are ignored
    — passing a start/end date silently returns just today, which is a good way to think you have
    a date-filtered set when you have one row)."""
    return _get_json(f"{api}/cs/map/?days={int(days)}")


def fetch_stations(api=MUMBAIFLOOD_API):
    """The 35 municipal AWS rain gauges: id, name, lat/lon, and their current rainfall."""
    return _get_json(f"{api}/aws/stations/")


def fetch_station_hourly(station_id, api=MUMBAIFLOOD_API):
    """One station's hourly rainfall. TODAY ONLY — the endpoint ignores every date parameter, so
    this cannot force a past storm. Historical forcing comes from `serve.weather` instead."""
    return _get_json(f"{api}/aws/hourly-aws-data/{int(station_id)}/")


def _depth_from_row(row):
    """metres of standing water, from the reporter's height x the fraction it reached."""
    factor = row.get("water_level_factor")
    if factor is None:
        return None, None
    height_in = (row.get("feet") or 0) * 12 + (row.get("inch") or 0)
    if height_in <= 0:
        return None, float(factor)
    return float(factor) * height_in * 0.0254, float(factor)


def crowdsourced_observations(raw=None, days=365, bbox=MUMBAI_BBOX, api=MUMBAIFLOOD_API):
    """Raw crowdsourced rows -> a QC'd ObservationSet. `raw` lets tests skip the network.

    Every rejection is counted by rule. Nothing personal is copied out of the source row.
    """
    rows = fetch_crowdsourced(days=days, api=api) if raw is None else raw
    rejected = Counter()
    seen = set()
    out = []
    for row in rows:
        lat, lon = row.get("latitude"), row.get("longitude")
        if lat is None or lon is None:
            rejected["no_location"] += 1
            continue
        lat, lon = float(lat), float(lon)
        if bbox and not (bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]):
            rejected["outside_region"] += 1
            continue
        if not row.get("water_level"):
            rejected["no_depth_reported"] += 1
            continue
        depth, factor = _depth_from_row(row)
        if depth is None:
            rejected["no_reporter_height"] += 1
            continue
        if factor is not None and factor >= MAX_BODY_FRACTION:
            rejected["depth_at_reporter_height"] += 1
            continue
        if depth < MIN_DEPTH_M:
            rejected["zero_depth"] += 1
            continue
        if depth > MAX_DEPTH_M:
            rejected["implausible_depth"] += 1
            continue
        ts = row.get("timestamp")
        if not ts:
            rejected["no_timestamp"] += 1
            continue
        key = (round(lat, 5), round(lon, 5), ts[:16], round(depth, 3))
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        place = (row.get("location") or "").strip(" ,") or None
        out.append(Observation(
            lat=lat, lon=lon, ts=ts, depth_m=round(depth, 3),
            source="mumbaiflood_cs", source_id=str(row.get("id")), place=place,
            # `name` and `feedback` are read above and deliberately not carried
            meta={"reported": row.get("water_level"), "body_fraction": factor},
        ))
    log.info("crowdsourced: kept %d of %d (%s)", len(out), len(rows), dict(rejected))
    return ObservationSet(out, dict(rejected),
                          meta={"source": "mumbaiflood_cs", "n_raw": len(rows),
                                "days": days, "bbox": list(bbox) if bbox else None,
                                "api": api})
