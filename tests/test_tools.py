"""Pure tool-layer tests: tool-call parsing + dispatch routing (no torch/GEE needed)."""
from varuna.agent import tools
from varuna.agent.llm import parse_tool_calls


def test_parse_single_tool_call():
    text = 'sure\n<tool_call>\n{"name": "get_outlook", "arguments": {"rain_mm": 80}}\n</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [{"name": "get_outlook", "arguments": {"rain_mm": 80}}]


def test_parse_multiple_and_malformed():
    text = ('<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
            '<tool_call>{bad json}</tool_call>'
            '<tool_call>{"name": "validation_scores", "arguments": {}}</tool_call>')
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["get_weather", "validation_scores"]


def test_parse_no_calls():
    assert parse_tool_calls("RED in 3 wards; act now.") == []


def test_dispatch_unknown_tool():
    out = tools.dispatch("does_not_exist", {})
    assert "error" in out


def test_dispatch_routes_and_catches(monkeypatch):
    monkeypatch.setitem(tools._HANDLERS, "get_weather", lambda **_: {"rain_mm_24h": 42.0})
    assert tools.dispatch("get_weather", {}) == {"rain_mm_24h": 42.0}

    def boom(**_):
        raise ValueError("nope")
    monkeypatch.setitem(tools._HANDLERS, "get_weather", boom)
    out = tools.dispatch("get_weather", {})
    assert out["error"].startswith("ValueError")


def test_tool_schema_names_match_handlers():
    schema_names = {t["function"]["name"] for t in tools.TOOLS}
    assert schema_names == set(tools._HANDLERS)


def test_state_screen_tool(tmp_path, monkeypatch):
    """Reads the committed screen, ranks by `rank`, and carries note + caveat through —
    the screen's own honesty text must reach the LLM, not just the dashboard."""
    import json
    ktk = tmp_path / "karnataka"
    ktk.mkdir()
    blocks = [dict(unit=f"u{i}", district=f"d{i}", rank=3 - i, roi=0.1 * i, category="Safe",
                   stage_pct=50.0, gws_trend_mm_yr=-1.0, ksat_mm_hr=10.0, pervious_frac=0.5,
                   rank_p5=1, rank_p95=5, top10_freq=0.9) for i in range(3)]
    (ktk / "state_screen.json").write_text(json.dumps(dict(
        state="karnataka", admin_level="district", weights={}, sensitivity={},
        note="A screen, not a siting.", caveat="quality screening required", blocks=blocks)))
    monkeypatch.setenv("VARUNA_ARTIFACTS", str(tmp_path))
    out = tools.dispatch("state_screen", {"top_n": 2})
    assert out["admin_level"] == "district" and len(out["top"]) == 2
    assert out["top"][0]["unit"] == "u2"                    # lowest rank number first
    assert "screen" in out["note"] and out["caveat"]
    assert "error" in tools.dispatch("state_screen", {"state": "no_such_state"})
    assert "error" in tools.dispatch("state_screen", {"state": "BAD ID!"})


def test_recharge_plan_tool(tmp_path, monkeypatch):
    import json
    from varuna.config import CFG
    curve = [dict(sites=50, recharge_m3=1000, reduction_pct=1.0, unstable=False),
             dict(sites=800, recharge_m3=9000, reduction_pct=3.0, unstable=True)]
    (tmp_path / "recharge_plan.json").write_text(json.dumps(dict(
        rain_mm=100.0, metered=True, would_be_runoff_m3=5e5, soil_capacity_m3=1e6,
        curve=curve, sites=[dict(lat=1, lon=2)] * 20, phases=[], gw_status={"sample": True},
        note="measured by re-simulation")))
    monkeypatch.setattr(CFG, "work", str(tmp_path))
    out = tools.dispatch("recharge_plan", {"top_sites": 5})
    assert out["best_dose"]["sites"] == 50                  # unstable dose never wins
    assert len(out["top_sites"]) == 5 and out["gw_status"]["sample"] is True
