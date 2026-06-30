"""FastAPI backend for the Varuna dashboard.

Serves the artifact bundle (sinks, recharge, alerts, validation, canal plan, rendered maps) and runs
the U-Net emulator live for rainfall what-ifs. Point it at the bundle with VARUNA_WORK (defaults to
artifacts/patna). Heavy endpoints (emulator / canals / optimize / chat) lazy-import torch+rasterio and
return 503 with a clear message if those aren't installed, so the light endpoints always work.

Run:  VARUNA_WORK=artifacts/patna uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import base64
import io
import math
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from varuna.config import CFG

app = FastAPI(title="Varuna FloodTwin API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WORK = os.environ.get("VARUNA_WORK", CFG.work)


def _p(name):
    return os.path.join(WORK, name)


def _load_json(name, default=None):
    import json
    path = _p(name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _domain_bounds():
    """[[south, west], [north, east]] of the 128x128 twin domain. Exact via rasterio if present,
    else a center+size approximation (good enough for a map overlay)."""
    lat, lon = CFG.center
    half_m = CFG.n_grid * CFG.dx / 2.0
    try:
        import rasterio
        import torch
        meta = torch.load(_p("twin_meta.pt"), map_location="cpu", weights_only=False)
        with rasterio.open(_p("dem.tif")) as s:
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
    return {"ok": True, "work": WORK,
            "artifacts": sorted(f for f in os.listdir(WORK)) if os.path.isdir(WORK) else []}


@app.get("/api/meta")
def meta():
    return {"aoi": list(CFG.aoi), "center": list(CFG.center),
            "domain_bounds": _domain_bounds(),
            "design_rain_mm": CFG.design_rain_mm, "min_depth_m": CFG.min_depth_m}


@app.get("/api/sinks")
def sinks():
    gj = _load_json("sinks.geojson")
    if gj is None:
        raise HTTPException(404, "sinks.geojson not in bundle")
    return gj


@app.get("/api/recharge")
def recharge():
    import pandas as pd
    path = _p("recharge_sites.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "recharge_sites.csv not in bundle")
    df = pd.read_csv(path)
    feats = [{"type": "Feature",
              "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
              "properties": {k: (float(r[k]) if k != "sink_id" else int(r[k]))
                             for k in df.columns if k not in ("lat", "lon")}}
             for _, r in df.iterrows()]
    return {"type": "FeatureCollection", "features": feats}


@app.get("/api/alerts")
def alerts():
    import pandas as pd
    path = _p("alerts_today.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "alerts_today.csv not in bundle")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/api/validation")
def validation():
    out = {"static_depth_vs_sar": _load_json("validation_scores.json"),
           "calibration_report": _load_json("calibration_report.json"),
           "calibrated_params": _load_json("calibrated_params.json")}
    twin = {f[len("twin_scores_"):-5]: _load_json(f)
            for f in (os.listdir(WORK) if os.path.isdir(WORK) else [])
            if f.startswith("twin_scores_") and f.endswith(".json")}
    out["dynamic_twin_vs_sar"] = twin
    return out


@app.get("/api/canal_plan")
def canal_plan():
    gj = _load_json("canal_plan.json")
    if gj is None:
        raise HTTPException(404, "canal_plan.json not in bundle — run plan_canals")
    return gj


@app.get("/api/image/{name}")
def image(name: str):
    if not name.endswith(".png") or "/" in name or "\\" in name:
        raise HTTPException(400, "bad image name")
    path = _p(name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not in bundle")
    return FileResponse(path, media_type="image/png")


# ----------------------------------------------------------------------------- live / heavy endpoints


class WhatIf(BaseModel):
    rain_mm: float = 100.0
    dig_sites: dict | None = None


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


@app.post("/api/whatif")
def whatif(req: WhatIf):
    """Live emulator what-if: returns flood stats + a depth overlay PNG (data URL) + bounds."""
    try:
        from varuna.serve.emulator import whatif_grid
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"emulator unavailable (need torch+rasterio+bundle): {e}")
    try:
        hmax, _dig, summary = whatif_grid(req.rain_mm, req.dig_sites, work=WORK)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"emulator failed: {e}")
    return {"summary": summary, "bounds": _domain_bounds(),
            "overlay_png": _flood_png(hmax, CFG.min_depth_m)}


class CanalReq(BaseModel):
    rain_mm: float = 100.0
    n_canals: int = 3
    use_river: bool = True
    optimize_depths: bool = False


@app.post("/api/canals")
def canals(req: CanalReq):
    try:
        from varuna.serve.canals import plan_canals
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"canals unavailable: {e}")
    try:
        return plan_canals(rain_mm=req.rain_mm, n_canals=req.n_canals, use_river=req.use_river,
                           optimize_depths=req.optimize_depths, work=WORK)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"plan_canals failed: {e}")


class OptReq(BaseModel):
    design_rain: float = 100.0
    budget_m3: float = 150000.0


@app.post("/api/optimize")
def optimize(req: OptReq):
    try:
        from varuna.serve.optimize import optimize_design
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"optimize unavailable: {e}")
    try:
        return optimize_design(design_rain=req.design_rain, budget_m3=req.budget_m3, work=WORK)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"optimize failed: {e}")


class ChatReq(BaseModel):
    message: str
    history: list | None = None


@app.post("/api/chat")
def chat(req: ChatReq):
    """Chat about the twin. Prefers a free hosted LLM API (set LLM_API_KEY); falls back to the local
    Qwen agent if installed; else 503."""
    # 1) free hosted OpenAI-compatible API (Groq by default) — no GPU needed
    try:
        from api.chat_hosted import available, chat_once as hosted_chat
    except Exception:  # noqa: BLE001
        available = lambda: False  # noqa: E731
        hosted_chat = None
    if available():
        try:
            return {"reply": hosted_chat(req.message, history=req.history or [], work=WORK),
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
        return {"reply": chat_once(req.message, history=req.history or [], work=WORK),
                "backend": "local"}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"chat failed: {e}")
