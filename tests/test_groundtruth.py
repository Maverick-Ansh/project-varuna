"""Observed-depth ground truth: the QC rules, the privacy guarantee, and the metrics.

Offline only — every network adapter takes an injectable `raw`, and the scoring maths is pure.
"""
import datetime as _dt

import numpy as np
import pytest

from varuna.groundtruth.observations import Observation, ObservationSet
from varuna.groundtruth.score import (_binary, _errors, antecedent_rain_mm, metrics,
                                     neighbourhood_scores, subgrid_dilution)
from varuna.groundtruth.sources import MUMBAI_BBOX, crowdsourced_observations


def row(**kw):
    """A well-formed crowdsourced row; override any field to break it."""
    base = dict(id=1, name="A Person", latitude=19.10, longitude=72.85, feet=5, inch=6,
                water_level="1 ft 0 in", water_level_factor=0.18, location="Kurla",
                feedback="knee deep", timestamp="2026-07-20T18:30:00+05:30")
    base.update(kw)
    return base


# --- quality control ------------------------------------------------------------------------


def test_depth_is_reporter_height_times_the_fraction():
    got = crowdsourced_observations(raw=[row(feet=6, inch=0, water_level_factor=0.5)])
    assert len(got) == 1
    assert got.observations[0].depth_m == pytest.approx(6 * 12 * 0.0254 * 0.5, abs=1e-3)


def test_points_outside_the_region_are_dropped():
    """The app is reachable worldwide and the live feed really does contain Argentina."""
    got = crowdsourced_observations(raw=[row(), row(id=2, latitude=-34.58, longitude=-58.43)])
    assert len(got) == 1
    assert got.rejected["outside_region"] == 1


def test_rows_without_a_depth_are_dropped():
    got = crowdsourced_observations(raw=[row(water_level=None, water_level_factor=None)])
    assert len(got) == 0
    assert got.rejected["no_depth_reported"] == 1


def test_water_at_the_reporters_own_height_is_dropped():
    """Chest-deep is real; 'the water was exactly as tall as me' is a slider at its stop."""
    got = crowdsourced_observations(raw=[row(water_level="5 ft 6 in", water_level_factor=1.0)])
    assert len(got) == 0
    assert got.rejected["depth_at_reporter_height"] == 1


def test_zero_and_implausible_depths_are_dropped():
    got = crowdsourced_observations(raw=[
        row(id=1, water_level="0 ft 0 in", water_level_factor=0.0),
        row(id=2, feet=6, inch=0, water_level_factor=0.94, water_level="5 ft 8 in"),
    ])
    assert got.rejected["zero_depth"] == 1
    assert len(got) == 1                                  # 0.94 x 6ft = 1.72 m, steep but kept


def test_duplicate_submissions_are_collapsed():
    got = crowdsourced_observations(raw=[row(id=1), row(id=2)])
    assert len(got) == 1
    assert got.rejected["duplicate"] == 1


def test_rows_missing_a_timestamp_are_dropped():
    got = crowdsourced_observations(raw=[row(timestamp=None)])
    assert got.rejected["no_timestamp"] == 1


def test_every_rejection_is_counted_so_the_ledger_reconciles():
    raw = [row(id=1), row(id=2, latitude=-34.0, longitude=-58.0), row(id=3, water_level=None),
           row(id=4, water_level_factor=1.0), row(id=5, timestamp=None)]
    got = crowdsourced_observations(raw=raw)
    assert len(got) + sum(got.rejected.values()) == len(raw)
    assert got.meta["n_raw"] == len(raw)


# --- privacy --------------------------------------------------------------------------------


def test_personal_data_never_reaches_an_observation():
    """The source ships a reporter NAME and free text on every row. Neither may survive parsing —
    these files get committed."""
    got = crowdsourced_observations(raw=[row(name="Kartik Bhatt", feedback="I am at home")])
    o = got.observations[0]
    blob = repr(o).lower() + str(o.meta).lower()
    assert "kartik" not in blob and "bhatt" not in blob
    assert "i am at home" not in blob
    assert not hasattr(o, "name") and not hasattr(o, "feedback")
    assert o.place == "Kurla"                             # a locality is fine; a person is not


def test_saved_file_carries_no_personal_data(tmp_path):
    got = crowdsourced_observations(raw=[row(name="Kartik Bhatt", feedback="secret note")])
    p = got.save(str(tmp_path / "obs.json"))
    text = open(p, encoding="utf-8").read().lower()
    assert "kartik" not in text and "secret note" not in text


# --- the record -----------------------------------------------------------------------------


def test_observation_roundtrips_through_json(tmp_path):
    s = crowdsourced_observations(raw=[row()])
    p = s.save(str(tmp_path / "o.json"))
    back = ObservationSet.load(p)
    assert len(back) == 1
    assert back.observations[0].depth_m == s.observations[0].depth_m
    assert back.rejected == s.rejected


def test_naive_timestamps_are_read_as_utc_not_guessed():
    o = Observation(lat=19.1, lon=72.8, ts="2026-07-20T18:30:00", depth_m=0.3, source="t")
    assert o.dt().tzinfo is not None
    assert o.dt().utcoffset() == _dt.timedelta(0)


def test_date_property():
    assert crowdsourced_observations(raw=[row()]).observations[0].date == "2026-07-20"


# --- metrics --------------------------------------------------------------------------------


def test_errors_are_the_textbook_definitions():
    e = _errors([1.0, 2.0], [0.0, 0.0])
    assert e["mae_m"] == pytest.approx(1.5)
    assert e["rmse_m"] == pytest.approx(np.sqrt(2.5), abs=1e-4)   # metrics round to 4 dp
    assert e["bias_m"] == pytest.approx(1.5)


def test_binary_scores_match_the_sar_convention():
    pred = [0.5, 0.5, 0.0, 0.0]
    obs = [0.5, 0.0, 0.5, 0.0]
    b = _binary(pred, obs, thresh=0.15)
    assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (1, 1, 1, 1)
    assert b["csi"] == pytest.approx(1 / 3, abs=1e-4)             # metrics round to 4 dp
    assert b["pod"] == pytest.approx(0.5)
    assert b["far"] == pytest.approx(0.5)


def _rows(pred, obs):
    return [{"observed_m": o, "predicted_m": p, "area": "a"} for p, o in zip(pred, obs)]


def test_a_perfect_model_scores_skill_one():
    obs = [0.1, 0.4, 0.9, 0.2]
    m = metrics(_rows(obs, obs))
    assert m["model"]["rmse_m"] == 0.0
    assert m["skill_vs_best_baseline"] == pytest.approx(1.0)


def test_predicting_zero_everywhere_cannot_look_good():
    """The failure mode this whole baseline block exists to catch: on a set whose median depth is
    a few centimetres, always-dry has a small absolute error and must still score <= 0 skill."""
    obs = [0.05, 0.08, 0.10, 0.06, 0.9]
    m = metrics(_rows([0.0] * len(obs), obs))
    assert m["model"]["mae_m"] < 0.25                      # looks respectable in isolation...
    assert m["skill_vs_best_baseline"] <= 0                # ...and is worthless against a baseline


def test_a_model_worse_than_the_mean_gets_negative_skill():
    obs = [0.1, 0.2, 0.3, 0.4]
    m = metrics(_rows([5.0, 0.0, 5.0, 0.0], obs))
    assert m["skill_vs_best_baseline"] < 0


def test_baselines_are_all_reported_for_comparison():
    m = metrics(_rows([0.2, 0.2], [0.1, 0.3]))
    assert set(m["baselines"]) == {"always_dry", "observed_mean", "observed_median"}
    assert m["n"] == 2


def test_correlation_needs_variation_and_enough_points():
    from varuna.groundtruth.score import _correlation
    assert _correlation([1.0, 2.0], [1.0, 2.0]) is None            # too few
    assert _correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # no spread in the prediction
    assert _correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_metrics_on_an_empty_set_do_not_explode():
    assert metrics([]) == {"n": 0}


# --- rainfall windowing ---------------------------------------------------------------------


def test_antecedent_rain_sums_only_the_window_before_the_report():
    tz = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    when = _dt.datetime(2026, 7, 20, 12, 0, tzinfo=tz)
    times = [f"2026-07-20T{h:02d}:00" for h in range(24)]
    precip = [1.0] * 24                                    # 1 mm every hour, all day
    cache = {(round(19.1, 2), round(72.85, 2), "2026-07-20"): (times, precip)}
    got = antecedent_rain_mm(19.1, 72.85, when, window_h=6, cache=cache)
    assert got == pytest.approx(7.0)                       # hours 06:00..12:00 inclusive
    assert antecedent_rain_mm(19.1, 72.85, when, window_h=3, cache=cache) == pytest.approx(4.0)


def test_antecedent_rain_returns_none_when_the_series_is_empty():
    tz = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    when = _dt.datetime(2026, 7, 20, 12, 0, tzinfo=tz)
    cache = {(19.1, 72.85, "2026-07-20"): ([], [])}
    assert antecedent_rain_mm(19.1, 72.85, when, cache=cache) is None


def test_bbox_covers_the_mumbai_tiles():
    s, w, n, e = MUMBAI_BBOX
    for lat, lon in [(18.9655, 72.83), (19.1040, 72.83), (19.1040, 72.976), (19.2425, 72.83)]:
        assert s <= lat <= n and w <= lon <= e


# --- diagnostics ----------------------------------------------------------------------------


def _drows(pairs, area="a", rain=80.0):
    return [{"observed_m": o, "predicted_m": p, "area": area, "rain_mm": rain,
             "cell": [5, 5]} for o, p in pairs]


def test_subgrid_dilution_recovers_a_known_ponding_area():
    """If the model spreads the same water over the whole 3600 m2 cell while the pond is really
    900 m2, the observed/predicted ratio must be 4 and the implied footprint 900 m2."""
    rows = _drows([(0.4, 0.1), (0.8, 0.2), (1.2, 0.3), (0.6, 0.15), (2.0, 0.5)])
    d = subgrid_dilution(rows, dx=60.0)
    assert d["median_observed_over_predicted"] == pytest.approx(4.0, abs=1e-6)
    assert d["implied_ponding_area_m2"] == 900
    assert d["implied_ponding_span_m"] == pytest.approx(30.0, abs=0.1)


def test_subgrid_dilution_ignores_dry_and_zero_prediction_rows():
    rows = _drows([(0.4, 0.1)] * 4 + [(0.01, 0.0)] * 10)     # only 4 usable pairs
    assert subgrid_dilution(rows) is None


def test_neighbourhood_scores_improve_when_the_model_is_merely_misplaced():
    """A flood predicted one cell over is a location error: widening the box must fix it."""
    grid = np.zeros((11, 11))
    grid[5, 7] = 0.8                                          # true depth, two cells east
    rows = _drows([(0.8, 0.0)])
    out = neighbourhood_scores(rows, lambda a, r: grid, radii=(0, 1, 2))
    assert out["0"]["mae_m"] == pytest.approx(0.8)            # nothing in the cell itself
    assert out["1"]["mae_m"] == pytest.approx(0.8)            # still outside a 3x3 box
    assert out["2"]["mae_m"] == pytest.approx(0.0)            # inside a 5x5 box
    assert out["2"]["box_m"] == 300


def test_neighbourhood_scores_do_not_rescue_a_model_that_has_no_water_anywhere():
    """The real Mumbai outcome: widening the box cannot invent skill that isn't there."""
    grid = np.zeros((11, 11))
    rows = _drows([(0.8, 0.0), (0.5, 0.0)])
    out = neighbourhood_scores(rows, lambda a, r: grid, radii=(0, 2, 4))
    assert out["0"]["mae_m"] == pytest.approx(out["4"]["mae_m"])


def test_neighbourhood_box_sizes_are_reported():
    grid = np.zeros((11, 11))
    out = neighbourhood_scores(_drows([(0.3, 0.0)]), lambda a, r: grid, radii=(0, 1, 3))
    assert [out[k]["box_m"] for k in ("0", "1", "3")] == [60, 180, 420]
