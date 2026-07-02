"""One-click plain-English intervention report for an area.

Gathers the area's computed outputs (alerts, canal plan, storage sizing, dig plan, cost-benefit,
SAR validation) and turns them into a structured markdown brief. If a hosted LLM is configured
(LLM_API_KEY, same as api/chat_hosted) it writes the prose; otherwise it falls back to a
deterministic template so the feature always works. It never invents numbers — the LLM is
instructed to use only the facts passed in, and the template just formats them.
"""
from __future__ import annotations

import json
import os


def _load(work, name):
    p = os.path.join(work, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _count_alerts(work):
    import csv
    p = os.path.join(work, "alerts_today.csv")
    if not os.path.exists(p):
        return None
    levels = {"RED": 0, "AMBER": 0, "GREEN": 0}
    try:
        with open(p) as f:
            for row in csv.DictReader(f):
                lv = str(row.get("level", "")).upper()
                if lv in levels:
                    levels[lv] += 1
    except Exception:
        return None
    return levels


def gather_facts(work, area=None, rain_mm=100.0):
    canal = _load(work, "canal_plan.json")
    storage = _load(work, "storage_sizing.json")
    dig = _load(work, "dig_plan.json")
    cb = _load(work, "costbenefit.json")
    val = _load(work, "calibration_report.json")
    alerts = _count_alerts(work)
    return dict(area=area or "the area", rain_mm=rain_mm, alerts=alerts,
                canal=canal, storage=storage, dig=dig, costbenefit=cb, validation=val)


def _template_report(f):
    """Deterministic markdown from the gathered facts (used when no LLM key is set)."""
    L = [f"# Flood intervention brief — {f['area']}", ""]
    a = f.get("alerts")
    if a:
        L += [f"**Current outlook:** {a['RED']} RED, {a['AMBER']} AMBER, {a['GREEN']} GREEN sinks.", ""]
    c = f.get("canal")
    if c:
        v = c["flooded_volume_m3"]
        L += [f"**Canals + storage pits:** {c['n_canals']} downhill canals to {c['outfalls']} cut "
              f"built-land flooding **{c['reduction_pct']}%** "
              f"({v['before']:,} → {v['after']:,} m³) at {c['rain_mm']:.0f} mm.", ""]
    s = f.get("storage")
    if s and s.get("targets"):
        parts = ", ".join(f"{k} cut ≈ {v['sites']} sites" for k, v in s["targets"].items())
        L += [f"**Distributed storage:** sized to local depressions — {parts}.", ""]
    d = f.get("dig")
    if d:
        fv = d["flooded_volume_m3"]
        L += [f"**Optimised excavation:** {d['total_excavation_m3']:,} m³ across {len(d['dig_plan'])} "
              f"sites cuts flooding **{d['reduction_pct']}%** vs do-nothing "
              f"({fv['do_nothing']:,} → {fv['optimal']:,} m³).", ""]
    cb = f.get("costbenefit")
    if cb and cb.get("interventions"):
        L += ["**Cost-benefit (₹ per m³ of flood removed, best first):**"]
        for it in cb["interventions"]:
            cpm = it.get("cost_per_m3_reduced_inr")
            L.append(f"- {it['name']}: ~₹{cpm}/m³ · {it['cost_crore_inr']} cr · "
                     f"{it['reduction_pct']}% cut ({it['detail']})")
        L.append("")
    v = f.get("validation")
    if v:
        L += [f"**Model skill (honest):** held-out CSI vs Sentinel-1 SAR "
              f"{v.get('baseline', {}).get('mean_csi_test')} → "
              f"{v.get('calibrated', {}).get('mean_csi_test')} (calibrated). Modest — treat outputs as "
              f"planning estimates of waterlogging *likelihood*, not certified forecasts.", ""]
    L += ["_Costs are indicative unit rates; flood volumes are measured on built-up land by "
          "re-simulation._"]
    return "\n".join(L)


SYSTEM = (
    "You are a flood-resilience planning assistant for Varuna FloodTwin. Write a concise, structured "
    "markdown brief for a city engineer, using ONLY the JSON facts provided — never invent numbers. "
    "Cover: current outlook, recommended interventions (canals/storage/excavation) with their measured "
    "flood-cut %, the cost-benefit ranking (highest ROI first), and an HONEST note on model skill "
    "(SAR-validated CSI is modest, so these are planning estimates, not certified forecasts). "
    "Prefer bullet points and bold headers; keep it under ~300 words."
)


def make_report(work, area=None, rain_mm=100.0):
    facts = gather_facts(work, area, rain_mm)
    try:
        from api.chat_hosted import available, DEFAULT_BASE, DEFAULT_MODEL
    except Exception:
        available = lambda: False  # noqa: E731
        DEFAULT_BASE = DEFAULT_MODEL = None

    if not available():
        return {"markdown": _template_report(facts), "backend": "template", "facts": facts}

    import requests
    key = os.environ["LLM_API_KEY"]
    base = os.environ.get("LLM_API_BASE", DEFAULT_BASE).rstrip("/")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Facts (JSON):\n" + json.dumps(facts, default=float)}]
    try:
        r = requests.post(f"{base}/chat/completions",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": model, "messages": msgs, "temperature": 0.3, "max_tokens": 800},
                          timeout=45)
        r.raise_for_status()
        md = r.json()["choices"][0]["message"]["content"].strip()
        return {"markdown": md, "backend": "hosted", "facts": facts}
    except Exception as e:  # noqa: BLE001 — never fail the endpoint; fall back to the template
        return {"markdown": _template_report(facts), "backend": f"template (LLM failed: {e})",
                "facts": facts}
