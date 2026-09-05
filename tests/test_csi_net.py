"""Guards on csi_net's scoring contract.

These are not accuracy tests. They pin the three properties that, if they silently broke,
would turn a reported number into a wrong number rather than a noisy one:

  * the increment target never sees the held-out scene's own mask,
  * the calibrated threshold uses a prior wet fraction and not the test label,
  * ablating rainfall does not also ablate the tide channels.
"""
import numpy as np
import pytest

from csi_net.data import N_RAIN, N_TIDE, RAIN_KEYS, pool, tide_vec
from csi_net.metrics import csi_report, persistent_field, storm_increment
from csi_net.run import ablation_mask, calibrated_threshold, fold_targets


class _FakeD:
    """Two domains, hand-built masks, so the leakage question has a checkable answer."""

    def __init__(self):
        a = np.zeros((3, 4, 4), bool)
        a[:, 0, :] = True                      # row 0 wet in every scene -> persistent
        a[0, 3, 3] = True                      # a mark unique to scene 0
        a[1, 2, 2] = True
        self.Y = {"x": a}
        self.valid = {"x": np.ones((4, 4), bool)}


def _samples(idxs):
    return [dict(area="x", idx=i, date=f"d{i}", flood=False) for i in idxs]


def test_fold_targets_cannot_see_the_held_out_scene():
    """Scene 0 is held out. Its unique wet cell must not shape any training target."""
    D = _FakeD()
    train = _samples([1, 2])                       # 0 is the test scene
    t = fold_targets(D, train, "increment")
    for s in train:
        y, v = t[(s["area"], s["idx"])]
        # (3,3) is wet ONLY in the held-out scene. If it were pooled into the persistent field
        # it would be masked out of the training target, which is how the leak would show.
        assert v[3, 3], "held-out scene's mask reached a training scene's valid mask"
    # and the persistent row really is removed from both targets
    for s in train:
        y, v = t[(s["area"], s["idx"])]
        assert not y[0, :].any(), "persistent row survived into the increment target"
        assert not v[0, :].any()


def test_fold_targets_is_identity_for_the_mask_target():
    assert fold_targets(_FakeD(), _samples([0, 1]), "mask") is None


def test_fold_targets_single_scene_domain_has_no_persistent_field():
    """One training scene in a domain: nothing to average, so the target is the raw mask."""
    D = _FakeD()
    t = fold_targets(D, _samples([1]), "increment")
    y, v = t[("x", 1)]
    assert np.array_equal(y, D.Y["x"][1]), "a lone scene should keep its whole mask"
    assert v.all()


def test_calibrated_threshold_hits_the_requested_wet_fraction():
    rng = np.random.default_rng(0)
    p = rng.random((64, 64)).astype("float32")
    valid = np.ones((64, 64), bool)
    for frac in (0.02, 0.15, 0.5):
        thr = calibrated_threshold(p, valid, frac)
        got = float((p >= thr).sum()) / p.size
        assert abs(got - frac) < 0.01, f"asked {frac}, predicted {got}"


def test_calibrated_threshold_respects_the_valid_mask():
    """Cells outside the mask must not influence the threshold."""
    p = np.zeros((10, 10), "float32")
    p[:5] = 0.9                                    # the invalid half is all-high
    valid = np.zeros((10, 10), bool)
    valid[5:] = True                               # only the all-zero half is scoreable
    assert calibrated_threshold(p, valid, 0.5) == pytest.approx(0.0, abs=1e-6)


def test_calibrated_threshold_survives_an_empty_mask():
    assert calibrated_threshold(np.zeros((4, 4), "float32"), np.zeros((4, 4), bool), 0.1) == 0.5


def test_ablating_rain_leaves_the_tide_channels_alone():
    names = ["dem_z", "twi"]
    n_static, n_rain = len(names), N_RAIN + N_TIDE
    m = ablation_mask(names, "rain", n_static, n_rain)
    assert m[:n_static].all(), "terrain must survive a rain ablation"
    assert not m[n_static:n_static + N_RAIN].any(), "rain channels must be zeroed"
    assert m[n_static + N_RAIN:].all(), "tide channels are not rainfall"
    t = ablation_mask(names, "tide", n_static, n_rain)
    assert t[:n_static + N_RAIN].all() and not t[n_static + N_RAIN:].any()


def test_ablation_mask_terrain_and_persist():
    names = ["dem_z", "jrc_occurrence", "log_dist_perm_water"]
    m = ablation_mask(names, "persist", len(names))
    assert m[0] == 1.0 and m[1] == 0.0 and m[2] == 0.0
    assert ablation_mask(names, "terrain", len(names))[:len(names)].sum() == 0.0
    assert ablation_mask(names, "none", len(names)).all()


def test_tide_vec_is_periodic_and_bounded():
    """A synodic month later the phase should return to (nearly) where it started."""
    a, b = tide_vec("2025-06-24"), tide_vec("2025-07-24")     # 30 d vs the 29.53 d month
    assert len(a) == N_TIDE
    assert np.abs(a).max() <= 1.0
    assert np.hypot(a[0], a[1]) == pytest.approx(1.0, abs=1e-5)
    assert np.hypot(a[2], a[3]) == pytest.approx(1.0, abs=1e-5)
    assert np.abs(a[:2] - b[:2]).max() < 0.12, "one month apart should be nearly in phase"


def test_tide_vec_separates_dates_within_a_month():
    """It has to carry information, not just be bounded."""
    v = np.stack([tide_vec(f"2025-06-{d:02d}") for d in (1, 8, 15, 22)])
    assert np.abs(v[:, None, :] - v[None, :, :]).sum(-1).max() > 1.0


def test_pool_matches_the_calibrate_convention():
    """Truth is max-pooled and features mean-pooled; getting this backwards moves every number."""
    a = np.arange(16, dtype="float32").reshape(1, 4, 4)
    assert pool(a, 2, "max")[0, 0, 0] == 5.0
    assert pool(a, 2, "mean")[0, 0, 0] == pytest.approx(2.5)
    assert pool(a, 1, "max").shape == a.shape
    assert pool(a, 2, "mean").shape == (1, 2, 2)


def test_storm_increment_is_the_non_persistent_part():
    masks = [np.array([[1, 0], [0, 0]], bool), np.array([[1, 1], [0, 0]], bool),
             np.array([[1, 0], [1, 0]], bool)]
    pers = persistent_field(masks, exclude=2)          # (0,0) wet in both others
    assert pers[0, 0] and not pers[1, 0]
    inc = storm_increment(masks[2], pers)
    assert inc[1, 0] and not inc[0, 0], "persistent water must not count as storm increment"


def test_csi_report_ignores_cells_outside_the_valid_mask():
    pred = np.ones((2, 2), bool)
    obs = np.array([[1, 0], [0, 0]], bool)
    valid = np.array([[1, 1], [0, 0]], bool)
    r = csi_report(pred, obs, valid)
    assert (r["hits"], r["fa"], r["misses"]) == (1, 1, 0)
    assert r["csi"] == pytest.approx(0.5)


def test_rain_keys_exist_in_the_committed_features():
    """The cached rain file must still carry every key the feature vector reads."""
    import json
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "csi_net", "data", "rain_features.json")
    if not os.path.exists(p):
        pytest.skip("rain_features.json not in this checkout")
    rain = json.load(open(p))
    area = next(iter(rain))
    day = next(iter(rain[area]["rain"]))
    for k in RAIN_KEYS + ["doy"]:
        assert k in rain[area]["rain"][day], f"{k} missing from cached rain features"
