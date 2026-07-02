"""Offline tests for the paper-baselines module (synthetic bundle; no network/EE)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("scipy")

from varuna.build import baselines  # noqa: E402


def _write(path, arr, dtype="float32"):
    from rasterio.transform import Affine
    T = Affine(0.00027, 0, 85.05, 0, -0.00027, 25.66)
    meta = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                crs="EPSG:4326", transform=T, dtype=dtype)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr.astype(dtype), 1)


def test_sweep_picks_best_threshold():
    valid = torch.ones(8, 8)
    truth = torch.zeros(8, 8)
    truth[:4] = 1
    grid = torch.zeros(8, 8)
    grid[:4] = 5.0                          # perfect separation at any t in (0, 5)
    best = baselines.sweep(["d"], {"d": truth}, valid, [1.0, 6.0], "above", grid=grid)
    assert best["threshold"] == 1.0
    assert best["mean_csi"] == 1.0


def test_terrain_layers_shapes(synth_bundle):
    from varuna.build.twin import build_domain
    dom = build_domain(synth_bundle, device="cpu")
    layers = baselines.terrain_layers(synth_bundle, dom)
    for k in ("DEPTH", "TWI", "HAND"):
        assert tuple(layers[k].shape) == (dom.N, dom.N)
    assert torch.isfinite(layers["TWI"]).all()


def test_compare_baselines_offline(synth_bundle, monkeypatch):
    # fake SAR truth: water where the synthetic bowl is (30 m rows 56:72, cols 72:88)
    obs = np.zeros((256, 256), dtype="uint8")
    obs[56:72, 72:88] = 1
    _write(f"{synth_bundle}/observed_water_2024-01-01.tif", obs, "uint8")
    monkeypatch.setattr(baselines, "_rain_for_date", lambda d, w: 80.0)
    rep = baselines.compare_baselines(work=synth_bundle, windows=(2,), taus=(0.05, 0.15),
                                      storm_hr=0.5, total_hr=1.0, make_figure=False)
    assert set(rep) >= {"static_depth", "twi", "hand_lite", "dynamic_twin"}
    for k in ("static_depth", "twi", "hand_lite", "dynamic_twin"):
        assert 0.0 <= rep[k]["mean_csi"] <= 1.0
    assert rep["dynamic_twin"]["window_days"] == 2


def test_dem_uncertainty_offline(synth_bundle):
    s = baselines.dem_uncertainty(work=synth_bundle, rain_mm=80, n_members=2,
                                  storm_hr=0.5, total_hr=1.0, make_figure=False)
    assert s["n_members"] == 2
    assert s["flooded_km2_mean"] >= 0
    assert s["km2_prob_ge_90pct"] <= s["km2_prob_ge_50pct"] + 1e-9
