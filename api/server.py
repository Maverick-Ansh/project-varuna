"""FastAPI backend for the Varuna dashboard.

Serves per-area artifact bundles (sinks, recharge, alerts, validation, canal plan, rendered maps)
and runs the twin live for rainfall what-ifs and interventions. Areas come from `varuna.areas`;
every endpoint takes an optional `area` id and resolves that area's bundle directory. With no
`area`, it falls back to $VARUNA_WORK (if set) or the default area's bundle.

Heavy endpoints (emulator / canals / storage / optimize / chat) lazy-import torch+rasterio and
return 503 with a clear message if those aren't installed, so the light endpoints always work.

Run:  uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import base64
import io
import math
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from varuna.config import CFG
from varuna.areas import (list_areas, get_area, area_work, default_area_id, is_built, CITIES)

app = FastAPI(title="Varuna FloodTwin API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_ENV_WORK = os.environ.get("VARUNA_WORK")

# In-process TTL cache for live upstream calls (Open-Meteo, alerts, advisory). Races between
# worker threads just cause a duplicate fetch — acceptable for these read-only payloads.
_TTL_CACHE: dict = {}


def _cached(key, ttl_s, fn):
    import time
    now = time.monotonic()
    hit = _TTL_CACHE.get(key)
    if hit is not None and now - hit[0] < ttl_s:
        return hit[1]
    val = fn()
    if len(_TTL_CACHE) >= 512:            # user-influenced keys (e.g. rain_mm) must not grow unbounded
        for k in sorted(_TTL_CACHE, key=lambda k: _TTL_CACHE[k][0])[:256]:
            _TTL_CACHE.pop(k, None)
    _TTL_CACHE[key] = (now, val)
    return val


# Citizen report store: memory + JSONL + HF-dataset persistence (see varuna/serve/reports.py).
from varuna.serve.reports import ReportStore, ip_hash as _ip_hash  # noqa: E402
# News store: PARALLEL to reports (news/ prefix, source="news", quarantined from the learner).
from varuna.serve.news import NewsStore  # noqa: E402

REPORTS = ReportStore()
NEWS = NewsStore()


@app.on_event("startup")
def _startup():
    REPORTS.start()
    NEWS.start()


@app.on_event("shutdown")
def _shutdown():
    REPORTS.flush()
    NEWS.flush()


def _work(area: str | None = None) -> str:
    """Resolve a bundle directory: explicit area id -> $VARUNA_WORK -> default area."""
    if area:
        try:
            return area_work(area)
        except KeyError as e:  # noqa: BLE001
            raise HTTPException(404, str(e))
    return _ENV_WORK or area_work(default_area_id())


def _center(area: str | None = None) -> list:
    if area:
        try:
            return list(get_area(area).center)
        except KeyError:  # noqa: BLE001
            pass
    return list(CFG.center)


def _pj(work, name):
    return os.path.join(work, name)


def _load_json(work, name, default=None):
    import json
    path = _pj(work, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _domain_bounds(work, center):
    """[[south, west], [north, east]] of the twin domain. Exact via the bundle's twin_meta.pt +
    dem.tif transform if present, else a center+size approximation (fine for a map overlay)."""
    lat, lon = center
    half_m = CFG.n_grid * CFG.dx / 2.0
    try:
        import rasterio
        import torch
        meta = torch.load(_pj(work, "twin_meta.pt"), map_location="cpu", weights_only=False)
        with rasterio.open(_pj(work, "dem.tif")) as s:
            T = s.transform
        r0, c0, N = meta["row0"], meta["col0"], meta["n_grid"]
        lon0, lat_top = T * (c0, r0)
        lon1, lat_bot = T * (c0 + N * 2, r0 + N * 2)
        return [[min(lat_bot, lat_top), min(lon0, lon1)], [max(lat_bot, lat_top), max(lon0, lon1)]]
    except Exception:
        dlat = half_m / 111320.0
        dlon = half_m / (111320.0 * max(math.cos(math.radians(lat)), 1e-3))
        return [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]


# ----------------------------------------------------------------------------- light endpoints


@app.get("/api/health")
def health():
    return {"ok": True, "default_area": default_area_id(),
            "areas": [a.id for a in list_areas()]}


@app.get("/api/areas")
def areas():
    """Registered areas + whether each has a serveable bundle."""
    return [{"id": a.id, "name": a.name, "built": is_built(a.id),
             "aoi": list(a.aoi), "center": list(a.center), "note": a.note, "city": a.city}
            for a in list_areas()]


@app.get("/api/meta")
def meta(area: str | None = None):
    work = _work(area)
    center = _center(area)
    a = None
    try:
        a = get_area(area) if area else None
    except KeyError:
        a = None
    return {"area": area or default_area_id(),
            "aoi": list(a.aoi) if a else list(CFG.aoi), "center": center,
            "domain_bounds": _domain_bounds(work, center),
            "design_rain_mm": CFG.design_rain_mm, "min_depth_m": CFG.min_depth_m}


@app.get("/api/sinks")
def sinks(area: str | None = None):
    gj = _load_json(_work(area), "sinks.geojson")
    if gj is None:
        raise HTTPException(404, "sinks.geojson not in bundle")
    return gj


@app.get("/api/recharge")
def recharge(area: str | None = None):
    import pandas as pd
    from varuna.build.recharge import groundwater_status
    work = _work(area)
    path = _pj(work, "recharge_sites.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "recharge_sites.csv not in bundle")
    df = pd.read_csv(path)
    feats = [{"type": "Feature",
              "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
              "properties": {k: (float(r[k]) if k != "sink_id" else int(r[k]))
                             for k in df.columns if k not in ("lat", "lon")}}
             for _, r in df.iterrows()]
    # provenance of the groundwater input; sample=True means the ranking must not be trusted
    status = groundwater_status(work)
    return {"type": "FeatureCollection", "features": feats,
            "sample": status["sample"], "gw_status": status, "caveat": status["caveat"]}


@app.get("/api/alerts")
def alerts(area: str | None = None):
    import pandas as pd
    path = _pj(_work(area), "alerts_today.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "alerts_today.csv not in bundle")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/api/validation")
def validation(area: str | None = None):
    work = _work(area)
    out = {"static_depth_vs_sar": _load_json(work, "validation_scores.json"),
           "calibration_report": _load_json(work, "calibration_report.json"),
           "calibrated_params": _load_json(work, "calibrated_params.json"),
           "baseline_comparison": _load_json(work, "baseline_comparison.json"),
           "flood_uncertainty": _load_json(work, "flood_uncertainty.json")}
    twin = {f[len("twin_scores_"):-5]: _load_json(work, f)
            for f in (os.listdir(work) if os.path.isdir(work) else [])
            if f.startswith("twin_scores_") and f.endswith(".json")}
    out["dynamic_twin_vs_sar"] = twin
    return out


@app.get("/api/canal_plan")
def canal_plan(area: str | None = None):
    gj = _load_json(_work(area), "canal_plan.json")
    if gj is None:
        raise HTTPException(404, "canal_plan.json not in bundle — run plan_canals")
    return gj


@app.get("/api/image/{name}")
def image(name: str, area: str | None = None):
    if not name.endswith(".png") or "/" in name or "\\" in name:
        raise HTTPException(400, "bad image name")
    path = _pj(_work(area), name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not in bundle")
    return FileResponse(path, media_type="image/png")


# ----------------------------------------------------------------------------- live data endpoints


def _area_aoi(area: str | None) -> list:
    try:
        return list(get_area(area).aoi) if area else list(CFG.aoi)
    except KeyError:
        return list(CFG.aoi)


@app.get("/api/weather")
def weather(area: str | None = None):
    """Live Open-Meteo forecast for the area: 24-h AOI max + hourly hyetograph. Cached 15 min."""
    from varuna.serve.weather import area_weather
    aid = area or default_area_id()

    def fetch():
        try:
            return area_weather(aoi=_area_aoi(area), center=tuple(_center(area)))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"weather fetch failed: {e}")
    return _cached(("weather", aid), 900, fetch)


@app.get("/api/alerts_live")
def alerts_live(area: str | None = None):
    """Today's outlook at the LIVE forecast (read-only: never rewrites alerts_today.csv).

    Cached 15 min; the committed /api/alerts stays as the nightly-refreshed fallback."""
    from varuna.serve.alerts import run_alerts
    aid = area or default_area_id()
    work = _work(area)

    def fetch():
        try:
            return run_alerts(rain_mm=None, work=work, aoi=_area_aoi(area),
                              aggregate_wards=False, save_csv=False)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"live alerts unavailable: {e}")
    return _cached(("alerts_live", aid), 900, fetch)


class CitizenReport(BaseModel):
    area: str
    lat: float
    lon: float
    depth_band: str                       # ankle | knee | waist | chest
    note: str | None = Field(None, max_length=280)
    client_ts: str | None = None
    website: str | None = None            # honeypot: bots fill it; humans never see it


@app.post("/api/reports")
def post_report(req: CitizenReport, request: Request):
    """Citizen flood report: pin + water level + note. Rate-limited, validated, persisted."""
    if req.website:                        # honeypot tripped: pretend success, store nothing
        return {"ok": True, "id": "0" * 12}
    ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "?"))
    try:
        work = _work(req.area)             # 404 on unknown area
        rep = REPORTS.add(req.area, work, req.lat, req.lon, req.depth_band,
                          note=req.note or "", client_ts=req.client_ts, ip_hash=_ip_hash(ip))
        return {"ok": True, **rep}
    except ValueError as e:
        msg = str(e)
        raise HTTPException(429 if msg.startswith("rate_limited") else 422, msg)


@app.get("/api/reports")
def get_reports(area: str | None = None, hours: float = 24):
    """Recent citizen reports for the map layer (public fields only)."""
    aid = area or default_area_id()
    reps = REPORTS.recent(area=aid, hours=min(max(hours, 1), 24 * 7))
    return {"area": aid, "count": len(reps), "reports": reps}


@app.get("/api/advisory")
def advisory(area: str | None = None):
    """Grounded citizen advisory (EN+HI, +MR for Mumbai). Hosted LLM or template. 30-min cache."""
    from varuna.serve.advisory import make_advisory
    aid = area or default_area_id()
    work = _work(area)

    def build():
        try:
            wx = weather(area)
        except HTTPException:
            wx = None
        try:
            al = alerts_live(area)
        except HTTPException:
            al = None                       # advisory renders honestly from partial facts
        a = None
        try:
            a = get_area(aid)
        except KeyError:
            pass
        langs = ("en", "hi", "mr") if (a and a.city == "mumbai") else ("en", "hi")
        return make_advisory(work, a.name if a else aid, weather=wx, alerts=al,
                             reports=REPORTS.summary(area=aid, hours=24), langs=langs)
    return _cached(("advisory", aid), 1800, build)


@app.get("/api/waterbalance")
def waterbalance(area: str | None = None, rain_mm: float = 100.0, efficiency: float = 1.0):
    """Storm mass balance (rain = soil/vegetation + ponded) from the committed ladder. Read-only."""
    from varuna.serve.waterbalance import water_balance
    try:
        return water_balance(work=_work(area), rain_mm=rain_mm, efficiency=efficiency)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"water balance failed: {e}")


@app.get("/api/storage_plan")
def storage_plan(area: str | None = None):
    """The committed storage-sizing curve (for the client-side container designer)."""
    plan = _load_json(_work(area), "storage_sizing.json")
    if plan is None:
        raise HTTPException(404, "storage_sizing.json not in bundle — run plan_storage")
    return plan


@app.get("/api/recharge_plan")
def recharge_plan(area: str | None = None):
    """The committed metered recharge-siting plan (m³ measured by re-simulation, Phase 3).

    Read-only, like storage_plan: the Space never computes, it serves what Colab committed."""
    plan = _load_json(_work(area), "recharge_plan.json")
    if plan is None:
        raise HTTPException(404, "recharge_plan.json not in bundle — run plan_recharge on Colab")
    return plan


@app.get("/api/state_screen")
def state_screen(state: str = "karnataka"):
    """Tier A recharge-opportunity screen: ranked admin units + sensitivity band.

    A committed table (scripts/run_state_screen.py), never computed here. The payload's own
    note says what it is: a screen, not a siting — it cannot say 'build here'."""
    import re as _re
    if not _re.fullmatch(r"[a-z][a-z0-9_]{1,40}", state):
        raise HTTPException(400, "bad state id")
    from varuna.areas import artifacts_root
    js = _load_json(os.path.join(artifacts_root(), state), "state_screen.json")
    if js is None:
        raise HTTPException(404, f"no state screen for '{state}' — run run_state_screen.py")
    return js


@app.get("/api/news")
def news(area: str | None = None, hours: float = 24):
    """Quarantined public-news feed: every item carries source='news' and the payload says
    trained_on=false — news is displayed, never used for training (see varuna/serve/news.py)."""
    aid = area or default_area_id()
    return NEWS.feed(aid, hours=min(max(hours, 1), 24 * 7))


@app.get("/api/nightlights")
def nightlights(area: str | None = None):
    """VIIRS Black Marble power-outage summary (written by the nightly job)."""
    js = _load_json(_work(area), "nightlights_outage.json")
    if js is None:
        raise HTTPException(404, "no night-lights data yet — the nightly job populates this")
    return js


@app.get("/api/learning_log")
def learning_log(area: str | None = None):
    """Nightly self-improvement history (reward-gated emulator updates)."""
    js = _load_json(_work(area), "learning_log.json")
    return {"area": area or default_area_id(), "entries": js or []}


@app.get("/api/city")
def city(city: str = "mumbai", rain_mm: float | None = None):
    """City-wide aggregate over a tile group. Tiles tessellate exactly, so sums don't double-count."""
    grp = CITIES.get(city)
    if not grp:
        raise HTTPException(404, f"unknown city '{city}'; known: {sorted(CITIES)}")
    if rain_mm is not None:                # quantize + clamp: it is a cache key
        rain_mm = round(min(max(float(rain_mm), 0.0), 500.0))

    def build():
        tiles, totals = [], dict(flooded_area_m2=0.0, flooded_volume_m3=0.0, red=0, amber=0,
                                 reports_24h=0, outage_cells=0)
        emulator_ok = True
        for tid in grp["tiles"]:
            built = is_built(tid)
            a = get_area(tid)
            work = a.work_dir()
            t = dict(id=tid, name=a.name, built=built, center=list(a.center))
            if built:
                t["bounds"] = _domain_bounds(work, list(a.center))
                r = rain_mm
                if r is None:
                    try:
                        r = weather(tid)["rain_24h_mm"]
                    except HTTPException:
                        r = CFG.design_rain_mm
                t["rain_mm"] = r
                try:
                    from varuna.serve.emulator import whatif
                    s = whatif(r, work=work)
                    t["summary"] = s
                    totals["flooded_area_m2"] += s.get("flooded_area_m2", 0)
                    totals["flooded_volume_m3"] += s.get("flooded_volume_m3", 0)
                except Exception as e:  # noqa: BLE001
                    emulator_ok = False
                    t["summary_error"] = str(e)
                try:
                    al = alerts_live(tid)
                    t["alerts"] = al.get("summary", {})
                    totals["red"] += t["alerts"].get("red", 0)
                    totals["amber"] += t["alerts"].get("amber", 0)
                except HTTPException:
                    pass
                rs = REPORTS.summary(area=tid, hours=24)
                t["reports_24h"] = rs["count"]
                totals["reports_24h"] += rs["count"]
                nl = _load_json(work, "nightlights_outage.json")
                if nl:
                    t["outage_cells"] = nl.get("n_outage_cells", 0)
                    t["nightlights_date"] = nl.get("date")
                    totals["outage_cells"] += t.get("outage_cells", 0)
            tiles.append(t)
        return dict(city=city, name=grp["name"], note=grp.get("note"),
                    tiles=tiles, totals=totals, emulator=emulator_ok)
    return _cached(("city", city, rain_mm), 60, build)


# ----------------------------------------------------------------------------- live / heavy endpoints


def _flood_png(hmax, tau):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from PIL import Image
    try:
        from matplotlib import colormaps
        cmap = colormaps["Blues"]
    except Exception:  # older matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap("Blues")
    norm = np.clip(hmax / 1.0, 0, 1)
    rgba = (cmap(norm) * 255).astype("uint8")
    rgba[..., 3] = np.where(hmax > tau, 200, 0).astype("uint8")     # transparent where dry
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class WhatIf(BaseModel):
    rain_mm: float = 100.0
    dig_sites: dict | None = None
    area: str | None = None


@app.post("/api/whatif")
def whatif(req: WhatIf):
    """Live emulator what-if: returns flood stats + a depth overlay PNG (data URL) + bounds."""
    work = _work(req.area)
    try:
        from varuna.serve.emulator import whatif_grid
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"emulator unavailable (need torch+rasterio+bundle): {e}")
    try:
        hmax, _dig, summary = whatif_grid(req.rain_mm, req.dig_sites, work=work)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"emulator failed: {e}")
    return {"summary": summary, "bounds": _domain_bounds(work, _center(req.area)),
            "overlay_png": _flood_png(hmax, CFG.min_depth_m)}


class CanalReq(BaseModel):
    rain_mm: float = 100.0
    n_canals: int = 3
    use_river: bool = True
    optimize_depths: bool = False
    n_inlets: int = 40          # spiderweb density when the bundle has a road graph
    area: str | None = None


@app.post("/api/canals")
def canals(req: CanalReq):
    try:
        from varuna.serve.canals import plan_canals
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"canals unavailable: {e}")
    try:
        return plan_canals(rain_mm=req.rain_mm, n_canals=req.n_canals, use_river=req.use_river,
                           optimize_depths=req.optimize_depths, n_inlets=req.n_inlets,
                           work=_work(req.area))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"plan_canals failed: {e}")


class StorageReq(BaseModel):
    rain_mm: float = 100.0
    targets: list | None = None
    unit_m3: float | None = None          # container unit size for the equiv_units readout
    area: str | None = None


@app.post("/api/storage")
def storage(req: StorageReq):
    """Adaptive distributed-storage sizing: flood-cut vs #sites curve + sites for target cuts."""
    try:
        from varuna.serve.containers import plan_storage
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"storage unavailable: {e}")
    try:
        targets = tuple(req.targets) if req.targets else (30, 50, 70)
        kw = {"unit_m3": float(req.unit_m3)} if req.unit_m3 else {}
        return plan_storage(rain_mm=req.rain_mm, targets=targets, work=_work(req.area), **kw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"plan_storage failed: {e}")


class OptReq(BaseModel):
    design_rain: float = 100.0
    budget_m3: float = 150000.0
    area: str | None = None


@app.post("/api/optimize")
def optimize(req: OptReq):
    try:
        from varuna.serve.optimize import optimize_design
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"optimize unavailable: {e}")
    try:
        return optimize_design(design_rain=req.design_rain, budget_m3=req.budget_m3,
                               work=_work(req.area))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"optimize failed: {e}")


class CostBenefitReq(BaseModel):
    rain_mm: float = 100.0
    area: str | None = None


@app.post("/api/costbenefit")
def costbenefit(req: CostBenefitReq):
    """Rank canal / storage / dig interventions by flood-m³ reduced per rupee."""
    try:
        from varuna.serve.costbenefit import rank_interventions
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"costbenefit unavailable: {e}")
    try:
        return rank_interventions(rain_mm=req.rain_mm, work=_work(req.area))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"costbenefit failed: {e}")


class ExposureReq(BaseModel):
    rain_mm: float = 100.0
    area: str | None = None


def _osmnx_present() -> bool:
    import importlib.util
    return importlib.util.find_spec("osmnx") is not None


@app.post("/api/exposure")
def exposure(req: ExposureReq):
    """At-risk buildings + roads that stay dry (evacuation routes) for the storm.

    Computes live where osmnx is installed; otherwise serves the bundle's cached
    exposure.json (written by `save_exposure` on Colab), so deploys without osmnx
    still get the exposure view (at its cached rainfall, flagged `cached`)."""
    work = _work(req.area)
    from varuna.serve.exposure import load_cached
    if not _osmnx_present():
        cached = load_cached(work)
        if cached is not None:
            cached["cached"] = True
            return cached
        raise HTTPException(503, "exposure unavailable: osmnx not installed and no "
                                 "cached exposure.json in the bundle")
    try:
        from varuna.serve.exposure import assess_exposure
        return assess_exposure(rain_mm=req.rain_mm, work=work, center=_center(req.area))
    except Exception as e:  # noqa: BLE001
        cached = load_cached(work)
        if cached is not None:
            cached["cached"] = True
            cached["note"] = f"live exposure failed ({e}); serving cached copy. " \
                             + str(cached.get("note", ""))
            return cached
        raise HTTPException(500, f"exposure failed: {e}")


class RouteReq(BaseModel):
    start: list                 # [lat, lon]
    end: list                   # [lat, lon]
    rain_mm: float = 100.0
    avoid_depth_m: float = 0.3
    area: str | None = None


@app.post("/api/route")
def route(req: RouteReq):
    """Flood-safe street route A->B at the given storm, next to the flood-ignorant shortest
    path. Risk backend is the FloodGNN checkpoint when the bundle ships one, else the raster
    emulator sampled per street (response says which)."""
    work = _work(req.area)
    try:
        from varuna.gnn.planner import safe_route
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"routing unavailable (need torch+rasterio+bundle): {e}")
    try:
        return safe_route(req.start, req.end, rain_mm=req.rain_mm, work=work,
                          avoid_depth_m=req.avoid_depth_m)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"safe_route failed: {e}")


class ReportReq(BaseModel):
    area: str | None = None
    rain_mm: float = 100.0


@app.post("/api/report")
def report(req: ReportReq):
    """LLM-written plain-English intervention report from the area's tool outputs."""
    try:
        from api.report import make_report
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"report unavailable: {e}")
    try:
        return make_report(work=_work(req.area), area=req.area, rain_mm=req.rain_mm)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"report failed: {e}")


class ChatReq(BaseModel):
    message: str
    history: list | None = None
    area: str | None = None


@app.post("/api/chat")
def chat(req: ChatReq):
    """Chat about the twin. Prefers a free hosted LLM API (set LLM_API_KEY); falls back to the local
    Qwen agent if installed; else 503."""
    work = _work(req.area)
    # 1) free hosted OpenAI-compatible API (Groq by default) — no GPU needed
    try:
        from api.chat_hosted import available, chat_once as hosted_chat
    except Exception:  # noqa: BLE001
        available = lambda: False  # noqa: E731
        hosted_chat = None
    if available():
        try:
            return {"reply": hosted_chat(req.message, history=req.history or [], work=work),
                    "backend": "hosted"}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"hosted LLM failed: {e}")
    # 2) local agent (only where transformers + model are present)
    try:
        from varuna.agent.tools import dispatch  # noqa: F401
        from varuna.agent.llm import chat_once
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"chat unavailable: set LLM_API_KEY for the free hosted API, "
                                 f"or install the local agent ({e}).")
    try:
        return {"reply": chat_once(req.message, history=req.history or [], work=work),
                "backend": "local"}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"chat failed: {e}")
