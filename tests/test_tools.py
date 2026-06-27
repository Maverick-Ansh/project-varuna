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
