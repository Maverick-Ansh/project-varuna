"""End-to-end twin smoke test: train a tiny emulator on the synthetic bundle, then what-if.

Marked slow (runs the CPU simulator a handful of times). Run with: pytest -m slow
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rasterio")

pytestmark = pytest.mark.slow


def test_train_then_whatif(synth_bundle):
    from varuna.build.twin import train_twin
    from varuna.serve.emulator import whatif

    meta = train_twin(work=synth_bundle, n_samples=4, epochs=3, device="cpu")
    assert "sites" in meta and len(meta["sites"]) > 0

    dry = whatif(rain_mm=20, work=synth_bundle, device="cpu")
    wet = whatif(rain_mm=150, work=synth_bundle, device="cpu")
    for r in (dry, wet):
        assert {"flooded_area_m2", "flooded_volume_m3", "peak_depth_m"} <= set(r)
        assert r["flooded_volume_m3"] >= 0
    # more rain -> at least as much peak depth (emulator is monotone-ish, allow equality)
    assert wet["peak_depth_m"] >= dry["peak_depth_m"] - 0.05
