"""Night-lights outage detection: pure-numpy logic, no Earth Engine."""
import numpy as np

from varuna.build.nightlights import compute_outage


def _grids():
    baseline = np.array([[10.0, 10.0, 0.5, 10.0],
                         [10.0, 10.0, 10.0, 10.0]])
    latest = np.array([[3.0, 8.0, 0.1, 3.0],       # 70% drop, 20% drop, dark-baseline, 70% drop
                       [-1.0, 3.0, 3.0, 10.0]])    # sentinel-missing, 70%, 70%, no drop
    quality = np.zeros_like(baseline)
    built = np.ones_like(baseline)
    return baseline, latest, quality, built


def test_basic_outage():
    baseline, latest, quality, built = _grids()
    outage, drop, stats = compute_outage(latest, baseline, quality, built)
    assert outage[0, 0] and not outage[0, 1]        # 70% flagged, 20% not
    assert not outage[0, 2]                          # baseline below floor -> never flagged
    assert not outage[1, 0]                          # -1 sentinel = missing, NOT dark
    assert not outage[1, 3]
    assert stats["n_outage"] == int(outage.sum())


def test_quality_and_built_masks():
    baseline, latest, quality, built = _grids()
    quality[0, 0] = 2                                # poor quality -> excluded
    built[0, 3] = 0.0                                # not built -> excluded
    outage, _, _ = compute_outage(latest, baseline, quality, built)
    assert not outage[0, 0] and not outage[0, 3]
    assert outage[1, 1] and outage[1, 2]


def test_monsoon_all_masked_is_unknown_not_blackout():
    """A fully cloud-masked latest composite must yield ZERO outages (the 2026-07-11 bug)."""
    baseline = np.full((4, 4), 20.0)
    latest = np.full((4, 4), -1.0)                   # everything missing
    outage, _, stats = compute_outage(latest, baseline, None, np.ones((4, 4)))
    assert stats["n_outage"] == 0 and stats["n_valid"] == 0
