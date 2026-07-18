"""Tool schemas (OpenAI/Qwen function format) + dispatch into the serve layer.

Heavy deps (torch, geopandas) are imported lazily inside each handler so this module — and the
tool-dispatch tests — stay light.
"""
from __future__ import annotations

import logging
import traceback

from ..config import CFG

log = logging.getLogger("varuna.agent.tools")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch the next-24h rainfall forecast (mm) for Patna (max over the AOI).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_outlook",
            "description": ("Today's flood outlook: per-sink and per-ward GREEN/AMBER/RED alert "
                            "levels from rainfall. If the user names a rainfall amount, pass it as "
                            "rain_mm; omit rain_mm ONLY to use the live forecast."),
            "parameters": {
                "type": "object",
                "properties": {
                    "rain_mm": {"type": "number",
                                "description": "24-h rainfall total in mm. Pass the amount the user "
                                               "states (e.g. 30 for 'what if 30 mm'); omit for live forecast."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "whatif",
            "description": ("Fast emulator what-if: flooded area/volume and peak depth for a given "
                            "storm, optionally with excavation at candidate sites."),
            "parameters": {
                "type": "object",
                "properties": {
                    "rain_mm": {"type": "number", "description": "24-h rainfall total (mm)."},
                    "dig_sites": {
                        "type": "object",
                        "description": "Optional map of site index -> dig depth in metres, e.g. {\"3\": 2.0}.",
                    },
                },
                "required": ["rain_mm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_design",
            "description": ("Physics-based optimisation of where to excavate to minimise flooding "
                            "for a design storm and excavation budget. Slow (GPU); use only when "
                            "asked to plan interventions."),
            "parameters": {
                "type": "object",
                "properties": {
                    "design_rain_mm": {"type": "number", "description": "Design storm total (mm)."},
                    "budget_m3": {"type": "number", "description": "Total excavation budget (m3)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_canals",
            "description": ("Plan drainage canals: route channels from the worst-flooded areas to "
                            "storage pits to cut net flood volume. Returns routes + before/after "
                            "flooded volume + % reduction. Slow (GPU); use when asked where/how to "
                            "dig canals or channels."),
            "parameters": {
                "type": "object",
                "properties": {
                    "rain_mm": {"type": "number", "description": "Design storm total (mm)."},
                    "n_canals": {"type": "integer", "description": "How many canals to route (default 3)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recharge_sites",
            "description": "Top aquifer-recharge candidate sites (flood water -> groundwater), ranked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "How many sites to return (default 10)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recharge_plan",
            "description": ("Metered recharge-siting plan for this area: m3 actually infiltrated "
                            "into the aquifer per number of recharge structures, MEASURED by "
                            "re-simulation (not a score). Includes soil capacity and groundwater "
                            "provenance."),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_sites": {"type": "integer",
                                  "description": "How many best sites to include (default 10)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "state_screen",
            "description": ("State-wide recharge-opportunity screen (Tier A): admin units ranked "
                            "by groundwater need x runoff x soil x pervious land, with a "
                            "sensitivity band. A screen, not a siting — it cannot say 'build "
                            "here'."),
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": "string",
                              "description": "State id (default 'karnataka')."},
                    "top_n": {"type": "integer",
                              "description": "How many top units to return (default 10)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validation_scores",
            "description": "Model accuracy vs Sentinel-1 radar observations (POD/FAR/CSI).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# --- handlers -------------------------------------------------------------

def _get_weather(**_):
    from ..serve.weather import aoi_max_rain
    return {"rain_mm_24h": round(aoi_max_rain(), 1)}


def _get_outlook(rain_mm=None, top=10, **_):
    from ..serve.alerts import run_alerts
    r = run_alerts(rain_mm=rain_mm, make_map=False)
    # trim the sink list so we don't flood the LLM context
    r = dict(r)
    r["sinks"] = r["sinks"][:top]
    r["sinks_total"] = r["summary"]["red"] + r["summary"]["amber"] + r["summary"]["green"]
    r["fill_ratio_meaning"] = ("inflow/capacity ratio (NOT a percentage): 1.0 = full, "
                               "0.5 = half-full, 18.9 = 18.9x over capacity")
    return r


def _whatif(rain_mm, dig_sites=None, **_):
    from ..serve.emulator import whatif
    return whatif(rain_mm=rain_mm, dig_sites=dig_sites)


def _optimize_design(design_rain_mm=None, budget_m3=None, **_):
    from ..serve.optimize import optimize_design
    return optimize_design(design_rain=design_rain_mm, budget_m3=budget_m3)


def _plan_canals(rain_mm=None, n_canals=3, **_):
    from ..serve.canals import plan_canals
    return plan_canals(rain_mm=rain_mm, n_canals=int(n_canals))


def _recharge_sites(top_n=10, **_):
    import pandas as pd
    from ..build.recharge import groundwater_status
    path = CFG.path("recharge_sites.csv")
    df = pd.read_csv(path).head(int(top_n))
    cols = [c for c in ["sink_id", "lat", "lon", "volume_m3", "gw_depth_m", "ksat_mm_hr",
                        "rsi", "recharge_score"] if c in df.columns]
    # provenance comes from gw_levels.csv (the input), not recharge_sites.csv (the product,
    # which carries no station column — checking it here is how the old warning went dead)
    status = groundwater_status(CFG.work)
    return {"sites": df[cols].to_dict(orient="records"),
            "sample": status["sample"],
            "warning": (f"Groundwater input is SAMPLE data ({status['reason']}) — rankings are "
                        "not meaningful until real CGWB/India-WRIS data is loaded."
                        if status["sample"] else None),
            "caveat": status["caveat"]}


def _recharge_plan(top_sites=10, **_):
    from ..io import load_json
    js = load_json(CFG.path("recharge_plan.json"))
    if js is None:
        return {"error": "No recharge_plan.json in this bundle — run plan_recharge (Colab GPU)."}
    best = max((c for c in js["curve"] if not c.get("unstable")),
               key=lambda c: c.get("recharge_m3", 0), default=None)
    return {"rain_mm": js["rain_mm"], "metered": js.get("metered", True),
            "would_be_runoff_m3": js.get("would_be_runoff_m3"),
            "soil_capacity_m3": js.get("soil_capacity_m3"),
            "curve": js["curve"],
            "best_dose": best,
            "top_sites": js.get("sites", [])[:int(top_sites)],
            "phases": js.get("phases"),
            "gw_status": js.get("gw_status"), "note": js.get("note")}


def _state_screen(state="karnataka", top_n=10, **_):
    import os
    import re
    from ..areas import artifacts_root
    from ..io import load_json
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", str(state)):
        return {"error": f"bad state id '{state}'"}
    js = load_json(os.path.join(artifacts_root(), str(state), "state_screen.json"))
    if js is None:
        return {"error": f"No state screen for '{state}' — run scripts/run_state_screen.py."}
    blocks = sorted(js.get("blocks", []), key=lambda b: b.get("rank", 1e9))
    keep = ["unit", "district", "rank", "roi", "category", "stage_pct", "gws_trend_mm_yr",
            "ksat_mm_hr", "pervious_frac", "rank_p5", "rank_p95", "top10_freq"]
    return {"state": js["state"], "admin_level": js["admin_level"],
            "weights": js.get("weights"),
            "top": [{k: b.get(k) for k in keep} for b in blocks[:int(top_n)]],
            "sensitivity": js.get("sensitivity"),
            "note": js.get("note"), "caveat": js.get("caveat")}


def _validation_scores(**_):
    from ..io import load_json
    s = load_json(CFG.path("validation_scores.json"))
    if s is None:
        return {"error": "No validation scores yet — run the SAR validation build step (nb03)."}
    return s


_HANDLERS = {
    "get_weather": _get_weather,
    "get_outlook": _get_outlook,
    "whatif": _whatif,
    "optimize_design": _optimize_design,
    "plan_canals": _plan_canals,
    "recharge_sites": _recharge_sites,
    "recharge_plan": _recharge_plan,
    "state_screen": _state_screen,
    "validation_scores": _validation_scores,
}


def dispatch(name, arguments=None):
    """Run a tool by name with a dict of arguments. Always returns a JSON-serialisable dict."""
    arguments = arguments or {}
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return handler(**arguments)
    except Exception as e:  # noqa: BLE001
        log.warning("tool %s failed: %s", name, e)
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=2)}
