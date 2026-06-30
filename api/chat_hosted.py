"""Hosted-LLM chat for the dashboard — a free, provider-agnostic fallback for the local Qwen agent.

Calls any OpenAI-compatible chat endpoint (Groq's free tier by default; also OpenRouter, Together,
a local vLLM, etc.) so the deployed Space needs no GPU. Grounds the model in the committed artifact
bundle (validation CSI, canal cut, storage sizing, sink/alert counts) so answers stay on-topic and
honest. Enabled only when LLM_API_KEY is set; otherwise the caller returns 503.

Env:
  LLM_API_KEY   (required)  — provider API key (e.g. a free Groq key)
  LLM_API_BASE  (optional)  — OpenAI-compatible base URL; default Groq
  LLM_MODEL     (optional)  — model id; default a free Groq Llama-3.3
"""
from __future__ import annotations

import json
import os

DEFAULT_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def available() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


def _bundle_facts(work: str) -> str:
    """A compact, honest summary of what's in the bundle, for the system prompt."""
    def load(name):
        p = os.path.join(work, name)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    lines = []
    val = load("calibration_report.json")
    twin = [f for f in (os.listdir(work) if os.path.isdir(work) else []) if f.startswith("twin_scores_")]
    if twin:
        lines.append(f"Dynamic twin vs Sentinel-1 SAR: mean CSI ~0.033 across storms (honest baseline; "
                     f"per-class physics calibration is a near-null result).")
    canal = load("canal_plan.json")
    if canal:
        lines.append(f"Canal optimizer (100 mm storm): {canal.get('reduction_pct')}% built-land flood cut, "
                     f"{canal.get('n_canals')} canals to {canal.get('outfalls')}.")
    stor = load("storage_sizing.json")
    if stor and stor.get("targets"):
        t = stor["targets"]
        lines.append("Adaptive distributed storage (sites sized to local minima, non-linear): "
                     + ", ".join(f"{k} cut needs ~{v['sites']} sites" for k, v in t.items()) + ".")
    sinks = None
    sp = os.path.join(work, "sinks.csv")
    if os.path.exists(sp):
        try:
            import csv
            with open(sp) as f:
                sinks = sum(1 for _ in csv.reader(f)) - 1
        except Exception:
            sinks = None
    if sinks:
        lines.append(f"{sinks} candidate flood-sink locations mapped from the DEM.")
    return "\n".join(f"- {x}" for x in lines) or "- (bundle facts unavailable)"


SYSTEM = (
    "You are the assistant for Varuna FloodTwin, a satellite-calibrated differentiable flood model of "
    "Patna, India. Answer concisely and HONESTLY about flood risk, the twin's skill, canal/storage "
    "interventions, and what the numbers mean. Do not overstate accuracy — the SAR-validated skill is "
    "modest (CSI ~0.033) and you should say so when relevant. Use the project facts below.\n\n"
    "Project facts:\n{facts}"
)


def chat_once(message: str, history=None, work: str = "artifacts/patna",
              timeout: float = 30.0) -> str:
    import requests

    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY not set")
    base = os.environ.get("LLM_API_BASE", DEFAULT_BASE).rstrip("/")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    msgs = [{"role": "system", "content": SYSTEM.format(facts=_bundle_facts(work))}]
    for turn in (history or []):
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": message})

    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": msgs, "temperature": 0.3, "max_tokens": 600},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
