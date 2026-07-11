"""Advisory rendering + live-weather parsing, all network monkeypatched."""
import pytest


CANNED = {
    "hourly": {
        "time": [f"2026-07-{10 + h // 24:02d}T{h % 24:02d}:00" for h in range(96)],
        "precipitation": [1.0] * 24 + [2.0] * 72,     # past day 24 mm, then 2 mm/hr
    }
}
# forecast_rain_mm calls the API WITHOUT past_days -> its array starts at "now"
CANNED_NOPAST = {"hourly": {"time": CANNED["hourly"]["time"][24:],
                            "precipitation": [2.0] * 72}}


def _fake_http(url, **k):
    return CANNED if "past_days" in url else CANNED_NOPAST


def test_forecast_hyetograph(monkeypatch):
    import varuna.serve.weather as W
    monkeypatch.setattr(W, "http_get_json", _fake_http)
    h = W.forecast_hyetograph(19.0, 72.8, hours=48)
    assert h["past24_mm"] == 24.0
    assert len(h["precip_mm"]) == 48 and h["precip_mm"][0] == 2.0
    assert sum(h["precip_mm"][:24]) == 48.0


def test_area_weather(monkeypatch):
    import varuna.serve.weather as W
    monkeypatch.setattr(W, "http_get_json", _fake_http)
    out = W.area_weather(aoi=(72.7, 18.9, 73.0, 19.3), center=(19.0, 72.8))
    assert out["rain_24h_mm"] == 48.0                 # max over sampled points (all canned equal)
    assert out["rain_center_48h_mm"] == 96.0
    assert out["hyetograph"]["precip_mm"][0] == 2.0


def test_advisory_template_severities():
    from varuna.serve.advisory import render_template, _severity
    low, sev = render_template(dict(area="Patna", rain_next24h_mm=10))
    assert sev == "low" and "Patna" in low["en"] and low["hi"]
    facts = dict(area="Mumbai", rain_next24h_mm=90, sinks_red=2, sinks_amber=3,
                 reports_24h=4, reports_max_depth_m=0.8)
    assert _severity(facts) == "high"
    out, _ = render_template(facts, langs=("en", "hi", "mr"))
    assert "90" in out["en"] and out["mr"]


def test_make_advisory_without_key(tmp_path, monkeypatch):
    from varuna.serve.advisory import make_advisory
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    out = make_advisory(str(tmp_path), "Testville",
                        weather=dict(rain_24h_mm=50, past24_mm=5),
                        reports=dict(count=2, max_depth_m=0.4, latest=None))
    assert out["backend"] == "template"
    assert out["severity"] == "moderate"
    assert "Testville" in out["advisory"]["en"]


def test_alerts_save_csv_off(synth_bundle, monkeypatch):
    import os
    from varuna.serve.alerts import run_alerts
    csv = os.path.join(synth_bundle, "alerts_today.csv")
    out = run_alerts(rain_mm=60, work=synth_bundle, aggregate_wards=False, save_csv=False)
    assert out["sinks"] and not os.path.exists(csv)
    run_alerts(rain_mm=60, work=synth_bundle, aggregate_wards=False)
    assert os.path.exists(csv)                         # default still writes
