"""Notebook 04 -> the daily alert engine: rain -> runoff -> fill ratio -> GREEN/AMBER/RED -> wards.

`run_alerts()` is the hot path the LLM calls. CPU-only, headless (map render is opt-in so it
never floods MCP tool results with HTML). Returns structured JSON.

CALIBRATION WARNING: the amber/red fill-ratio thresholds are placeholders (see README and
varuna.build.validate). Outputs are 'waterlogging likelihood', not certified forecasts.
"""
from __future__ import annotations

import datetime as _dt
import logging

import numpy as np

from ..config import CFG
from ..io import read_aligned, read1, require_bundle

log = logging.getLogger("varuna.serve.alerts")

# WorldCover class -> SCS Curve Number for hydrologic soil groups (A, B, C, D)
CN_TABLE = {
    10: (36, 60, 73, 79), 20: (45, 66, 77, 83), 30: (49, 69, 79, 84),
    40: (67, 78, 85, 89), 50: (89, 92, 94, 95), 60: (77, 86, 91, 94),
    80: (98, 98, 98, 98), 90: (98, 98, 98, 98),
}


def _soil_group(work, shape):
    """Hydrologic soil group (1=A..4=D) from clay %, else assume C (Gangetic silty clay)."""
    import os
    R, C = shape
    if os.path.exists(f"{work}/clay.tif"):
        clay = read_aligned(f"{work}/clay.tif", R, C) / 10.0
        sg = np.full(clay.shape, 2)
        sg[clay > 25] = 3
        sg[clay > 40] = 4
        sg[clay < 10] = 1
        return sg
    return np.full(shape, 3)


def _runoff_depth(P, work):
    """SCS-CN runoff depth (mm) per cell for total rainfall P (mm). Returns (Q, labels, shape)."""
    wc, _, R, C = read1(f"{work}/worldcover.tif")
    labels = read1(f"{work}/catchment_labels.tif")[0]
    r = min(wc.shape[0], labels.shape[0])
    c = min(wc.shape[1], labels.shape[1])
    wc, labels = wc[:r, :c], labels[:r, :c]
    sg = _soil_group(work, wc.shape)

    cn = np.full(wc.shape, 80.0)
    for cls, vals in CN_TABLE.items():
        for g in range(1, 5):
            cn[(wc == cls) & (sg == g)] = vals[g - 1]
    S = 25400.0 / cn - 254.0
    Ia = 0.2 * S
    Q = np.where(P > Ia, (P - Ia) ** 2 / (P + 0.8 * S), 0.0)
    return Q, labels, (r, c)


def _aggregate_wards(alerts, aoi):
    """Aggregate sink alerts to wards (OSM admin boundaries, else a ~1 km grid). Returns (wards_gdf, ward_level)."""
    import geopandas as gpd
    from shapely.geometry import Point, box

    bbox_poly = box(aoi[0], aoi[1], aoi[2], aoi[3])
    wards = None
    try:
        import osmnx as ox
        g = ox.features_from_polygon(bbox_poly, tags={"boundary": "administrative",
                                                       "admin_level": ["9", "10"]})
        g = g[g.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index()
        if len(g) >= 5:
            wards = g[["geometry"]].copy()
            wards["ward"] = [f"ward_{i}" for i in range(len(wards))]
            log.info("OSM wards: %d", len(wards))
    except Exception as e:  # noqa: BLE001
        log.warning("OSM ward fetch failed (%s); using grid", e)

    if wards is None:
        cellsz = 0.01
        cells = [box(x, y, x + cellsz, y + cellsz)
                 for x in np.arange(aoi[0], aoi[2], cellsz)
                 for y in np.arange(aoi[1], aoi[3], cellsz)]
        wards = gpd.GeoDataFrame({"ward": [f"zone_{i}" for i in range(len(cells))]},
                                 geometry=cells, crs="EPSG:4326")
        log.info("ward grid fallback: %d zones", len(wards))

    import pandas as pd
    al = pd.DataFrame(alerts)
    pts = gpd.GeoDataFrame(al, geometry=[Point(lo, la) for la, lo in zip(al.lat, al.lon)],
                           crs="EPSG:4326")
    joined = gpd.sjoin(pts, wards.set_crs("EPSG:4326", allow_override=True), predicate="within")
    score = {"GREEN": 0, "AMBER": 1, "RED": 2}
    ward_level = joined.groupby("ward")["level"].agg(lambda s: max(s, key=lambda v: score[v]))
    wards["alert"] = wards["ward"].map(ward_level).fillna("GREEN")
    return wards, ward_level


def _save_map(work, alerts, wards, aoi):
    import folium
    colors = {"GREEN": "#2e7d32", "AMBER": "#ef6c00", "RED": "#c62828"}
    m = folium.Map(location=[(aoi[1] + aoi[3]) / 2, (aoi[0] + aoi[2]) / 2],
                   zoom_start=12, tiles="cartodbpositron")
    for _, w in wards.iterrows():
        if w.alert == "GREEN":
            continue
        folium.GeoJson(w.geometry.__geo_interface__,
                       style_function=lambda f, col=colors[w.alert]:
                       dict(fillColor=col, color=col, weight=1, fillOpacity=0.35)).add_to(m)
    for a in alerts:
        folium.CircleMarker([a["lat"], a["lon"]], radius=6, color=colors[a["level"]], fill=True,
                            popup=f"sink {a['sink_id']}: {a['level']}<br>fill {a['fill_ratio']}").add_to(m)
    path = f"{work}/alert_map.html"
    m.save(path)
    return path


def run_alerts(rain_mm=None, work=None, aoi=None, make_map=False, aggregate_wards=True):
    """Compute today's flood outlook.

    rain_mm: total 24-h rainfall; if None, fetch the AOI max from Open-Meteo.
    Returns a JSON-serialisable dict: forecast_rain_mm, sinks[], wards{}, summary{}, plus a
    'caveats' line so any consumer (LLM, dashboard) surfaces the limitations.
    """
    import pandas as pd
    work = work or CFG.work
    aoi = aoi or CFG.aoi
    require_bundle(work, ["sinks.csv", "catchment_labels.tif", "worldcover.tif"])

    if rain_mm is None:
        from .weather import aoi_max_rain
        rain_mm = aoi_max_rain(aoi)
    P = float(rain_mm)

    Q, labels, (r, c) = _runoff_depth(P, work)
    sinks = pd.read_csv(f"{work}/sinks.csv")
    alerts = []
    for _, s in sinks.iterrows():
        cat = labels == int(s.sink_id)
        if cat.sum() == 0:
            continue
        inflow_m3 = float(Q[cat].mean() / 1000.0 * cat.sum() * CFG.cell_area_m2)
        ratio = inflow_m3 / max(s.volume_m3, 1.0)
        level = ("RED" if ratio >= CFG.red_ratio
                 else ("AMBER" if ratio >= CFG.amber_ratio else "GREEN"))
        alerts.append(dict(sink_id=int(s.sink_id), lat=float(s.lat), lon=float(s.lon),
                           inflow_m3=round(inflow_m3), capacity_m3=float(s.volume_m3),
                           fill_ratio=round(ratio, 2), level=level))
    alerts.sort(key=lambda a: a["fill_ratio"], reverse=True)
    pd.DataFrame(alerts).to_csv(f"{work}/alerts_today.csv", index=False)

    wards = {}
    if aggregate_wards and alerts:
        try:
            wgdf, ward_level = _aggregate_wards(alerts, aoi)
            wards = {k: v for k, v in ward_level.items() if v != "GREEN"}
            if make_map:
                _save_map(work, alerts, wgdf, aoi)
        except Exception as e:  # noqa: BLE001
            log.warning("ward aggregation failed: %s", e)

    summary = dict(
        red=sum(a["level"] == "RED" for a in alerts),
        amber=sum(a["level"] == "AMBER" for a in alerts),
        green=sum(a["level"] == "GREEN" for a in alerts),
    )
    return dict(
        forecast_rain_mm=round(P, 1),
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        sinks=alerts,
        wards=wards,
        summary=summary,
        caveats=("Provisional 'waterlogging likelihood', not a certified forecast. Alert "
                 "thresholds are uncalibrated placeholders; the model ignores storm sewers and "
                 "pumping (over-predicts near working drains)."),
    )


def telegram_push(result, token, chat_id, top=5):
    """Optional: push a one-line summary of an alert result to Telegram."""
    from ..io import http_get
    import requests
    reds = [a for a in result["sinks"] if a["level"] == "RED"]
    ambers = [a for a in result["sinks"] if a["level"] == "AMBER"]
    msg = (f"Patna flood outlook, next 24h: {result['forecast_rain_mm']:.0f} mm forecast.\n"
           f"RED sinks: {len(reds)}  AMBER: {len(ambers)}.\n"
           + "\n".join(f"- sink {a['sink_id']} fill {a['fill_ratio']}" for a in reds[:top]))
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  data={"chat_id": chat_id, "text": msg}, timeout=30)
    return msg
