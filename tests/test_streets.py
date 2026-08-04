"""Offline tests for the street-level water layer (no bundle / rasterio needed)."""
import numpy as np
import pytest

from varuna.serve import streets as S


def _segments(n=6, R=20, C=20):
    """n segments laid along one row, alternating east/north bearings."""
    rows = np.full(n, 5, dtype="int64")
    cols = np.arange(n, dtype="int64") + 2
    return {
        "row": rows, "col": cols,
        "east": np.where(np.arange(n) % 2 == 0, 1.0, 0.0),
        "north": np.where(np.arange(n) % 2 == 0, 0.0, 1.0),
        "length_m": np.full(n, 50.0),
        "lat": 19.0 + 0.001 * np.arange(n),
        "lon": 72.8 + 0.001 * np.arange(n),
    }


def _grid(R=20, C=20, val=0.0):
    return np.full((R, C), val)


def test_concentration_factor_rises_over_the_measured_range():
    assert S.concentration_factor(0.0) == pytest.approx(1.0)
    assert S.concentration_factor(S.WET_THRESHOLD_M) == pytest.approx(S.DILUTION_FACTOR)
    assert S.concentration_factor(S.MEASURED_MAX_M) == pytest.approx(S.DILUTION_FACTOR)
    mid = S.concentration_factor(S.WET_THRESHOLD_M / 2)
    assert 1.0 < mid < S.DILUTION_FACTOR


def test_concentration_decays_to_one_when_the_cell_is_uniformly_deep():
    """Regression: without this a 4.45 m creekside cell mean became 19.8 m 'on the street'."""
    assert S.concentration_factor(S.UNIFORM_DEPTH_M) == pytest.approx(1.0)
    assert S.concentration_factor(4.45) == pytest.approx(1.0)
    between = S.concentration_factor((S.MEASURED_MAX_M + S.UNIFORM_DEPTH_M) / 2)
    assert 1.0 < between < S.DILUTION_FACTOR
    assert S.street_depth_mm(4.45) == pytest.approx(4450, abs=1)      # reports the cell mean


def test_street_depth_exceeds_cell_mean_but_never_below_it():
    d = np.array([0.0, 0.05, 0.15, 0.5, 2.0, 5.0])
    mm = S.street_depth_mm(d)
    assert np.all(mm >= d * 1000 - 1e-9)
    assert mm[2] == pytest.approx(0.15 * S.DILUTION_FACTOR * 1000)


def test_street_depth_peaks_in_the_measured_band():
    """The correction must not make deep water deeper than the deepest corrected puddle."""
    d = np.linspace(0.0, 6.0, 200)
    mm = S.street_depth_mm(d)
    boost = mm / 1000.0 - d
    assert boost.argmax() < len(d) // 2                  # the lift lives at shallow depths
    assert boost[-1] == pytest.approx(0.0, abs=1e-6)     # none left in deep water


def test_ponded_streets_are_kept_and_marked():
    """The whole point: a street with water and no flow is exactly what road_flow drops."""
    seg = _segments()
    h = _grid()
    h[5, 2:8] = 0.3
    rows, meta = S.streets_in_view(seg, h, u=_grid(), v=_grid())     # zero velocity everywhere
    assert len(rows) == 6
    assert all(r.get("ponded") for r in rows)
    assert all("bearing" not in r for r in rows)
    assert meta["n_returned"] == 6


def test_flowing_streets_get_a_bearing_and_no_ponded_flag():
    seg = _segments()
    h = _grid()
    h[5, 2:8] = 0.3
    u = _grid(val=1.0)
    rows, _ = S.streets_in_view(seg, h, u=u, v=_grid())
    east_rows = [r for r in rows if r["cell"][1] % 2 == 0]           # east-aligned segments
    assert east_rows and all("bearing" in r for r in east_rows)
    assert all(not r.get("ponded") for r in east_rows)


def test_bbox_filters_and_reports_what_was_in_view():
    seg = _segments()
    h = _grid()
    h[5, 2:8] = 0.3
    bbox = [[18.999, 72.799], [19.0015, 72.8015]]                    # first ~2 midpoints
    rows, meta = S.streets_in_view(seg, h, bbox=bbox)
    assert 0 < len(rows) < 6
    assert meta["n_in_view"] == len(rows)
    assert meta["n_segments"] == 6


def test_dry_streets_are_dropped():
    seg = _segments()
    h = _grid(val=0.001)
    rows, meta = S.streets_in_view(seg, h)
    assert rows == [] and meta["n_returned"] == 0


def test_truncation_keeps_the_deepest_and_says_so():
    seg = _segments(n=6)
    h = _grid()
    h[5, 2:8] = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
    rows, meta = S.streets_in_view(seg, h, max_segments=2)
    assert meta["truncated"] is True and meta["dropped"] == 4
    assert [r["depth_cell_mm"] for r in rows] == [900, 800]          # deepest first


def test_excluded_and_flagged_cells():
    seg = _segments()
    h = _grid()
    h[5, 2:8] = 0.4
    excl = np.zeros((20, 20), dtype=bool)
    excl[5, 2] = True                                               # permanent water
    flag = np.zeros((20, 20), dtype=bool)
    flag[5, 3] = True                                               # neighbour of water
    rows, _ = S.streets_in_view(seg, h, exclude=excl, flag=flag)
    assert len(rows) == 5                                           # the water cell is gone
    assert sum(1 for r in rows if r.get("near_water")) == 1


def test_summarise_bands_and_lengths():
    rows = [
        {"depth_street_mm": 50, "length_m": 100.0},
        {"depth_street_mm": 450, "length_m": 200.0, "ponded": True},
        {"depth_street_mm": 900, "length_m": 300.0},
    ]
    s = S.summarise(rows)
    assert s["n"] == 3 and s["max_street_mm"] == 900
    assert s["flooded_length_m"] == 600
    assert s["length_over_knee_400mm"] == 500                       # 200 + 300
    assert s["length_over_impassable_600mm"] == 300
    assert s["ponded_share"] == pytest.approx(1 / 3, abs=1e-3)


def test_empty_segments_is_not_a_crash():
    empty = {k: np.array([]) for k in
             ("row", "col", "east", "north", "length_m", "lat", "lon")}
    empty["row"] = empty["row"].astype("int64")
    empty["col"] = empty["col"].astype("int64")
    rows, meta = S.streets_in_view(empty, _grid())
    assert rows == [] and meta["n_segments"] == 0
