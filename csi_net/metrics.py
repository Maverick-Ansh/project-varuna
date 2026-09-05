"""CSI, and the baselines without which a CSI number means nothing.

Standing rule from the project's own methodology note: never print a CSI without the
bias score beside it, and never print a table without the trivial baselines in it.
"""
import numpy as np


def confusion(pred, obs, valid):
    p = (pred > 0.5) & valid
    o = (obs > 0.5) & valid
    return int((p & o).sum()), int((~p & o).sum()), int((p & ~o).sum())


def csi_report(pred, obs, valid):
    h, m, f = confusion(pred, obs, valid)
    return dict(csi=h / max(h + m + f, 1), pod=h / max(h + m, 1), far=f / max(h + f, 1),
                bias=(h + f) / max(h + m, 1), hits=h, misses=m, fa=f,
                pred_wet=h + f, obs_wet=h + m)


def sweep_threshold(prob, obs, valid, lo=0.02, hi=0.98, n=49):
    """Best single global threshold. Returns (threshold, csi). Caller decides which
    split it was chosen on - choosing on the test split is optimistic and must be labelled."""
    best = (-1.0, 0.5)
    for t in np.linspace(lo, hi, n):
        c = csi_report(prob >= t, obs, valid)["csi"]
        if c > best[0]:
            best = (c, float(t))
    return best[1], best[0]


# --------------------------------------------------------------------- trivial baselines

def csi_allwet(obs, valid):
    """Predict every valid cell wet. CSI == the observed wet fraction, for free, with zero knowledge."""
    o = (obs > 0.5) & valid
    return int(o.sum()) / max(int(valid.sum()), 1)


def csi_random(obs, valid, k=None, seed=0):
    """Scatter k wet cells uniformly at random. k defaults to the observed wet count (bias == 1)."""
    o = (obs > 0.5) & valid
    n = int(valid.sum())
    k = int(o.sum()) if k is None else int(k)
    idx = np.flatnonzero(valid.ravel())
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
    p = np.zeros(valid.size, bool); p[pick] = True
    return csi_report(p.reshape(valid.shape), obs, valid)["csi"]


def persistent_field(masks, exclude, thresh=0.5):
    """Cells wet in >= `thresh` of the OTHER dates. This is the storm-independent water that
    dominates the SAR target: seasonal ponds, tidal flats, mangrove, paddy - everything the
    JRC >= 50 % permanent-water mask does not remove."""
    others = [m for i, m in enumerate(masks) if i != exclude]
    if not others:
        return np.zeros_like(masks[exclude], bool)
    return np.mean(others, axis=0) >= thresh


def storm_increment(obs, persistent):
    """The honest target: wet today AND not persistently wet. What a rainfall model should predict."""
    return (obs > 0.5) & (~persistent)
