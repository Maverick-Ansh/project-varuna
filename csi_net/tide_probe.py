"""Does a calendar-derived tide phase explain Mumbai-Harbour's water, where rainfall cannot?

RESULTS.md §4.3 found Harbour's observed wet-cell count correlates -0.70 with 14-day rainfall:
significant, and backwards for rain. The reading offered was tidal forcing - Sentinel-1 crosses
at a fixed local time, so harbour extent should track where the tide is in its cycle. That was
an interpretation, never a measurement, and it was used to justify dropping a whole domain.

This measures it. `data.tide_vec` turns a date into the sin/cos of the synodic phase and its
second harmonic - the spring-neap beat and the ~14.8-day aliasing of the lunar semidiurnal tide
at a fixed observation hour. Cheap enough to run before deciding whether a GPU experiment on it
is worth anything.

Two tests per domain, both against observed wet fraction:

  * single-channel Pearson r, with the n that produced it, so a lucky r on 11 dates is visible
  * multiple R^2 of all four tide channels, with a permutation p-value - four predictors on
    eleven scenes will fit noise well, and the permutation says how well noise alone does

Rainfall is included in the same table as the reference the tide has to beat.

    python -m csi_net.tide_probe
"""
import json
import os

import numpy as np

from .data import RAIN_KEYS, VarunaData, tide_vec

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = ["patna", "mumbai_harbour", "mumbai_northeast"]
TIDE_NAMES = ["moon_sin", "moon_cos", "moon_sin2", "moon_cos2"]


def r2_multiple(X, y):
    """Ordinary least squares R^2 with an intercept."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / max(ss_tot, 1e-12)


def perm_p(X, y, n=5000, seed=0):
    """P(noise fits at least this well) - the honest reading of R^2 at n=11, k=4."""
    rng = np.random.default_rng(seed)
    obs = r2_multiple(X, y)
    hits = sum(r2_multiple(X, rng.permutation(y)) >= obs for _ in range(n))
    return obs, (hits + 1) / (n + 1)


def main():
    D = VarunaData(os.path.join(HERE, "data", "varuna_stack.npz"),
                   os.path.join(HERE, "data", "rain_features.json"), grid_m=60, tide=True)
    rain = json.load(open(os.path.join(HERE, "data", "rain_features.json")))

    print(f"{'domain':<18}{'n':>4}   " + "".join(f"{k:>10}" for k in TIDE_NAMES)
          + f"{'rain_14d':>10}")
    print("-" * 84)
    summary = {}
    for a in ALL:
        rows = [s for s in D.samples if s["area"] == a]
        if not rows:
            continue
        y = np.array([float((D.Y[a][s["idx"]] & D.valid[a]).sum()) / int(D.valid[a].sum())
                      for s in rows])
        T = np.stack([tide_vec(s["date"]) for s in rows])
        r14 = np.array([rain[a]["rain"][s["date"]]["rain_14d"] for s in rows])
        rs = [float(np.corrcoef(T[:, i], y)[0, 1]) for i in range(4)]
        rr = float(np.corrcoef(r14, y)[0, 1])
        print(f"{a:<18}{len(rows):>4}   " + "".join(f"{v:>10.2f}" for v in rs) + f"{rr:>10.2f}")
        r2, p = perm_p(T, y)
        r2r, pr = perm_p(np.column_stack([r14, np.log1p(r14)]), y)
        summary[a] = dict(n=len(rows), tide_r=[round(v, 3) for v in rs], rain14_r=round(rr, 3),
                          tide_r2=round(r2, 3), tide_p=round(p, 4),
                          rain_r2=round(r2r, 3), rain_p=round(pr, 4))

    print(f"\n{'domain':<18}{'tide R^2':>10}{'perm p':>9}   {'rain R^2':>10}{'perm p':>9}")
    print("-" * 60)
    for a, v in summary.items():
        print(f"{a:<18}{v['tide_r2']:>10.3f}{v['tide_p']:>9.4f}   "
              f"{v['rain_r2']:>10.3f}{v['rain_p']:>9.4f}")
    print("\n4 tide predictors on 11 scenes fit ~0.4 R^2 from noise alone; the permutation p is "
          "the number\nto read, not the R^2. A tide story survives only if p is small in Harbour "
          "and not elsewhere.")

    out = os.path.join(HERE, "results", "tide_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print("wrote", out)
    return summary


if __name__ == "__main__":
    main()
