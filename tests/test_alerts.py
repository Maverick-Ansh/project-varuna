"""Alert-engine smoke test on the synthetic bundle (no Earth Engine, no GPU)."""
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pandas")


def test_run_alerts_structure_and_levels(synth_bundle):
    from varuna.serve.alerts import run_alerts

    # heavy rain, ward aggregation off (no network/osmnx in CI)
    r = run_alerts(rain_mm=150, work=synth_bundle, aggregate_wards=False)

    assert r["forecast_rain_mm"] == 150.0
    assert set(r["summary"]) == {"red", "amber", "green"}
    assert len(r["sinks"]) >= 1
    s0 = r["sinks"][0]
    assert {"sink_id", "lat", "lon", "inflow_m3", "capacity_m3", "fill_ratio", "level"} <= set(s0)
    assert s0["level"] in {"RED", "AMBER", "GREEN"}
    # sinks are sorted by fill ratio descending
    ratios = [s["fill_ratio"] for s in r["sinks"]]
    assert ratios == sorted(ratios, reverse=True)
    assert "caveats" in r


def test_no_rain_is_all_green(synth_bundle):
    from varuna.serve.alerts import run_alerts
    r = run_alerts(rain_mm=0.0, work=synth_bundle, aggregate_wards=False)
    assert r["summary"]["red"] == 0 and r["summary"]["amber"] == 0


def test_heavier_rain_raises_fill_ratio(synth_bundle):
    from varuna.serve.alerts import run_alerts
    low = run_alerts(rain_mm=20, work=synth_bundle, aggregate_wards=False)["sinks"]
    high = run_alerts(rain_mm=200, work=synth_bundle, aggregate_wards=False)["sinks"]
    by_id_low = {s["sink_id"]: s["fill_ratio"] for s in low}
    by_id_high = {s["sink_id"]: s["fill_ratio"] for s in high}
    for sid in by_id_low:
        assert by_id_high[sid] >= by_id_low[sid]
