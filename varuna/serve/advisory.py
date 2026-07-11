"""Citizen-facing flood advisory: the dashboard's "communicate back" channel.

Grounded strictly in numbers the system actually produced — live forecast, today's sink
alerts, citizen reports of the last 24 h, and the committed plan headlines. A hosted LLM
(Groq free tier via LLM_API_KEY, same client pattern as api/chat_hosted) writes it in
English + Hindi (+ Marathi for Mumbai areas); without a key a deterministic template
renders the same facts, so the panel always works.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os

from ..io import load_json

log = logging.getLogger("varuna.serve.advisory")

DEFAULT_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def gather_facts(work, area_name, weather=None, alerts=None, reports=None):
    """Assemble every number the advisory may cite. All inputs optional -> honest partials."""
    facts = dict(area=area_name,
                 generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    if weather:
        facts["rain_next24h_mm"] = weather.get("rain_24h_mm")
        facts["rain_past24h_mm"] = weather.get("past24_mm")
    if alerts:
        s = alerts.get("summary", {})
        facts["sinks_red"] = s.get("red", 0)
        facts["sinks_amber"] = s.get("amber", 0)
        worst = [a for a in alerts.get("sinks", []) if a.get("level") in ("RED", "AMBER")][:3]
        facts["worst_spots"] = [dict(lat=a["lat"], lon=a["lon"], level=a["level"],
                                     fill_ratio=a["fill_ratio"]) for a in worst]
    if reports:
        facts["reports_24h"] = reports.get("count", 0)
        facts["reports_max_depth_m"] = reports.get("max_depth_m", 0.0)
    canal = load_json(os.path.join(work, "canal_plan.json"))
    if canal and canal.get("reduction_pct") is not None:
        facts["drain_plan_cut_pct"] = round(float(canal["reduction_pct"]), 1)
    return facts


def _severity(facts):
    rain = float(facts.get("rain_next24h_mm") or 0)
    red = int(facts.get("sinks_red") or 0)
    reps = int(facts.get("reports_24h") or 0)
    if red > 0 or rain >= 80 or (reps >= 5 and float(facts.get("reports_max_depth_m") or 0) >= 0.8):
        return "high"
    if rain >= 40 or int(facts.get("sinks_amber") or 0) > 0 or reps >= 3:
        return "moderate"
    return "low"


_TEMPLATES = {
    "low": {
        "en": ("No significant waterlogging expected in {area} — about {rain} mm of rain is "
               "forecast for the next 24 hours. {reports_line} Roads should stay passable; "
               "follow local advisories."),
        "hi": ("{area} में अगले 24 घंटों में लगभग {rain} मिमी बारिश का अनुमान है — बड़े जलभराव की "
               "आशंका नहीं है। {reports_line_hi} सड़कें सामान्य रहने की उम्मीद है; स्थानीय सूचनाओं "
               "का पालन करें।"),
        "mr": ("{area} मध्ये पुढील २४ तासांत सुमारे {rain} मिमी पावसाचा अंदाज आहे — मोठ्या "
               "पाणी साचण्याची शक्यता नाही. रस्ते सुरळीत राहतील; स्थानिक सूचनांचे पालन करा."),
    },
    "moderate": {
        "en": ("Watch for waterlogging in {area}: ~{rain} mm of rain is forecast in the next "
               "24 hours and {amber} low-lying spots may fill ({red} critical). {reports_line} "
               "Avoid underpasses and known low points during heavy spells; plan extra travel time."),
        "hi": ("{area} में जलभराव की संभावना: अगले 24 घंटों में ~{rain} मिमी बारिश का अनुमान है और "
               "{amber} निचले इलाके भर सकते हैं ({red} गंभीर)। {reports_line_hi} तेज़ बारिश के "
               "दौरान अंडरपास और निचले इलाकों से बचें; यात्रा में अतिरिक्त समय रखें।"),
        "mr": ("{area} मध्ये पाणी साचण्याची शक्यता: पुढील २४ तासांत ~{rain} मिमी पाऊस अपेक्षित आहे "
               "आणि {amber} सखल भाग भरू शकतात ({red} गंभीर). जोरदार पावसात अंडरपास आणि सखल "
               "भाग टाळा; प्रवासासाठी जास्त वेळ ठेवा."),
    },
    "high": {
        "en": ("Heavy waterlogging risk in {area}: ~{rain} mm of rain is forecast in the next "
               "24 hours; {red} spots are at critical fill and {amber} more are elevated. "
               "{reports_line} Avoid travel through low-lying areas, never walk or drive through "
               "moving water, and move vehicles/valuables off basement and street level where "
               "flooding is reported."),
        "hi": ("{area} में भारी जलभराव का खतरा: अगले 24 घंटों में ~{rain} मिमी बारिश का अनुमान है; "
               "{red} स्थान गंभीर स्थिति में हैं और {amber} और भी खतरे में हैं। {reports_line_hi} "
               "निचले इलाकों की यात्रा से बचें, बहते पानी में पैदल या वाहन से न जाएँ, और जहाँ पानी "
               "भरने की सूचना है वहाँ वाहन/सामान सुरक्षित स्थान पर रखें।"),
        "mr": ("{area} मध्ये तीव्र पाणी साचण्याचा धोका: पुढील २४ तासांत ~{rain} मिमी पावसाचा अंदाज; "
               "{red} ठिकाणे गंभीर स्थितीत आहेत आणि आणखी {amber} धोक्यात आहेत. सखल भागातून प्रवास "
               "टाळा, वाहत्या पाण्यातून चालू/गाडी चालवू नका, आणि पाणी साचल्याची नोंद असलेल्या भागात "
               "वाहने/मौल्यवान वस्तू सुरक्षित ठिकाणी हलवा."),
    },
}


def render_template(facts, langs=("en", "hi")):
    sev = _severity(facts)
    n = int(facts.get("reports_24h") or 0)
    depth = float(facts.get("reports_max_depth_m") or 0)
    reports_line = (f"Citizens filed {n} flooding reports in the last 24 h "
                    f"(deepest ~{depth:.1f} m)." if n else "")
    reports_line_hi = (f"पिछले 24 घंटों में नागरिकों ने {n} जलभराव रिपोर्ट दर्ज कीं "
                       f"(अधिकतम ~{depth:.1f} मी)।" if n else "")
    vals = dict(area=facts.get("area", "this area"),
                rain=round(float(facts.get("rain_next24h_mm") or 0)),
                red=int(facts.get("sinks_red") or 0),
                amber=int(facts.get("sinks_amber") or 0),
                reports_line=reports_line, reports_line_hi=reports_line_hi)
    out = {}
    for lang in langs:
        tpl = _TEMPLATES[sev].get(lang) or _TEMPLATES[sev]["en"]
        out[lang] = " ".join(tpl.format(**vals).split())
    return out, sev


_LLM_SYSTEM = (
    "You write short public flood advisories for Indian cities. Use ONLY the facts provided — "
    "never invent numbers or place names. Tone: calm, practical, specific. For each requested "
    "language write ONE paragraph of at most 4 short sentences. Output STRICT JSON: an object "
    'with one key per language code (e.g. {{"en": "...", "hi": "..."}}). Languages: {langs}. '
    "The 'hi' text must be Hindi in Devanagari; 'mr' must be Marathi in Devanagari."
)


def make_advisory(work, area_name, weather=None, alerts=None, reports=None, langs=("en", "hi"),
                  timeout=30.0):
    """Advisory dict {advisory:{lang:text}, severity, backend, facts, generated_at}."""
    facts = gather_facts(work, area_name, weather=weather, alerts=alerts, reports=reports)
    tmpl, sev = render_template(facts, langs)
    key = os.environ.get("LLM_API_KEY")
    if key:
        try:
            import json as _json
            import requests
            base = os.environ.get("LLM_API_BASE", DEFAULT_BASE).rstrip("/")
            model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
            r = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [
                          {"role": "system", "content": _LLM_SYSTEM.format(langs=", ".join(langs))},
                          {"role": "user",
                           "content": "Facts (JSON):\n" + _json.dumps(facts, ensure_ascii=False)
                                      + f"\nSeverity: {sev}"}],
                      "temperature": 0.2, "max_tokens": 600,
                      "response_format": {"type": "json_object"}},
                timeout=timeout)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            parsed = _json.loads(txt)
            adv = {lang: str(parsed[lang]).strip() for lang in langs if parsed.get(lang)}
            if adv:
                for lang in langs:                      # LLM skipped a language -> template fills in
                    adv.setdefault(lang, tmpl[lang])
                return dict(advisory=adv, severity=sev, backend="hosted", facts=facts,
                            generated_at=facts["generated_at"])
        except Exception as e:  # noqa: BLE001
            log.warning("hosted advisory failed (%s) — template fallback", e)
    return dict(advisory=tmpl, severity=sev, backend="template", facts=facts,
                generated_at=facts["generated_at"])
