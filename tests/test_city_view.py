"""Offline tests for serving the joined city from its precomputed storm ladder."""
import json

import numpy as np
import pytest

from varuna.serve import city_view as CV


@pytest.fixture
def city_bundle(tmp_path):
    """A 3-rung ladder on a 10 x 8 grid: half the grid is land, a quarter of that is built."""
    h, w = 10, 8
    covered = np.zeros((h, w), dtype=bool)
    covered[:, :4] = True                       # left half is land, right half is sea
    built = np.zeros((h, w), dtype=bool)
    built[:5, :2] = True
    grids = {}
    for r, depth in ((50.0, 0.10), (100.0, 0.30), (200.0, 0.50)):
        g = np.zeros((h, w), dtype="float16")
        g[covered] = depth
        g[0, 0] = depth * 2                     # a deepest cell to check max
        grids[str(r)] = g
    land = covered.copy()
    land[:, 3] = False                          # one land column is actually creek
    np.savez_compressed(tmp_path / "city_hmax.npz", covered=covered, land=land, built=built,
                        **grids)
    meta = {"grid": [h, w], "pix_deg": 0.00027, "origin": [19.3, 72.75], "dx": 60.0,
            "coverage": 0.5, "built_frac": 0.125, "rains": [50.0, 100.0, 200.0],
            "tiles": ["a", "b"], "offsets60": {}, "with_sink_field": True,
            "outflow_m3": {"50.0": 1000, "100.0": 2000, "200.0": 3000},
            "note": "test ladder"}
    (tmp_path / "city_meta.json").write_text(json.dumps(meta))
    CV._CACHE.clear()
    yield str(tmp_path)
    CV._CACHE.clear()


def test_missing_bundle_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_city_bundle"):
        CV.load_city(str(tmp_path))


def test_exact_rung_is_not_blended(city_bundle):
    grid, prov = CV.depth_grid(100.0, city_bundle)
    assert prov["blend"] == 0.0 and prov["clamped"] is None
    assert grid[1, 1] == pytest.approx(0.30, abs=1e-3)


def test_between_rungs_interpolates_and_reports_both(city_bundle):
    grid, prov = CV.depth_grid(150.0, city_bundle)
    assert prov["rungs_used"] == [100.0, 200.0]
    assert prov["blend"] == pytest.approx(0.5)
    assert grid[1, 1] == pytest.approx(0.40, abs=1e-3)      # halfway between 0.30 and 0.50


def test_outside_the_ladder_clamps_and_warns(city_bundle):
    out_hi = CV.city_flood(500.0, city_bundle)
    out_lo = CV.city_flood(5.0, city_bundle)
    assert "warning" in out_hi and "above" in out_hi["warning"]
    assert "warning" in out_lo and "below" in out_lo["warning"]
    # clamped, never extrapolated
    assert out_hi["max_depth_m"] == pytest.approx(CV.city_flood(200.0, city_bundle)["max_depth_m"])


def test_sea_and_creek_are_excluded_from_every_total(city_bundle):
    out = CV.city_flood(100.0, city_bundle)
    # 10 x 8 grid: 4 covered columns, but one of them is creek -> 30 land cells of 3600 m^2
    assert out["land_km2"] == pytest.approx(round(30 * 0.0036, 1))
    assert out["grid_km2"] == pytest.approx(round(40 * 0.0036, 1))   # covered, incl. the creek
    assert out["built_km2"] == pytest.approx(round(10 * 0.0036, 1))
    assert out["flood_km2"] <= 30 * 0.0036 + 1e-9        # never counts sea or creek
    assert out["flood_built_km2"] <= out["flood_km2"]
    assert "land_is_approximate" not in out


def test_a_bundle_without_a_land_mask_says_its_totals_are_approximate(tmp_path):
    """Old bundles only have `covered`; they must flag that, not silently overstate."""
    h, w = 6, 6
    covered = np.ones((h, w), dtype=bool)
    np.savez_compressed(tmp_path / "city_hmax.npz", covered=covered,
                        built=np.zeros((h, w), dtype=bool),
                        **{"100.0": np.full((h, w), 0.3, dtype="float16")})
    (tmp_path / "city_meta.json").write_text(json.dumps(
        {"grid": [h, w], "pix_deg": 0.00027, "origin": [19.3, 72.75], "dx": 60.0,
         "rains": [100.0], "tiles": []}))
    CV._CACHE.clear()
    out = CV.city_flood(100.0, str(tmp_path))
    assert out["land_is_approximate"] is True
    assert "sea" in out["land_note"]
    CV._CACHE.clear()


def test_volume_and_area_rise_with_rain(city_bundle):
    a = CV.city_flood(50.0, city_bundle)
    b = CV.city_flood(200.0, city_bundle)
    assert b["volume_m3"] > a["volume_m3"]
    assert b["max_depth_m"] > a["max_depth_m"]


def test_outflow_is_interpolated_and_labelled_a_lower_bound(city_bundle):
    out = CV.city_flood(150.0, city_bundle)
    assert out["outflow_m3_lower_bound"] == pytest.approx(2500, abs=1)
    assert out["with_sink_field"] is True


def test_bounds_span_the_grid(city_bundle):
    meta, _ = CV.load_city(city_bundle)
    (s, w), (n, e) = CV.bounds(meta)
    assert s < n and w < e
    assert n == pytest.approx(19.3)                          # origin is the NW corner
    assert e == pytest.approx(72.75 + 8 * 2 * 0.00027)
