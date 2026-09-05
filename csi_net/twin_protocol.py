"""Re-score the physics twin and the terrain indices on the *net's* protocol.

The headline in RESULTS.md compared a net measured under leave-one-storm-out over 18 storms in
two domains against a twin measured on 8 Patna dates with one globally-swept threshold and a
dead scene included. Those are different test sets, so the multiple between them was not a
measurement. This script removes that gap:

  * same storms      - patna + mumbai_northeast, 2024-07-07 dropped, 18 total
  * same grid        - 60 m, from the same varuna_stack.npz array the net was scored on
  * same mask        - jrc occurrence < 50 % on prediction AND truth
  * same truth       - max-pooled Sentinel-1 water mask
  * same threshold discipline - chosen on the OTHER storms of the fold, never on the test scene

Alignment is asserted, not assumed: the stack's truth/mask are checked cell-for-cell against
`calibrate.align_sar` / `calibrate.valid_mask` on the twin's own domain crop before anything is
scored. If a bundle ever moves its crop window, this fails loudly instead of quietly comparing
two different pieces of ground.

Both protocols are reported. `loso` is the like-for-like one. `global` reproduces the old
table's rule - one threshold maximising mean CSI over every date, including the test scene -
so the old numbers remain traceable and the size of that optimism is visible.

    python -m csi_net.twin_protocol --areas patna,mumbai_northeast

No internet: rainfall comes from the cached rain_features.json the net itself used, whose
rain_{1,2,3,5}d windows are exactly the WINDOWS the twin sweeps.
"""
import argparse
import json
import os
import time

import numpy as np

from .data import VarunaData
from .metrics import csi_allwet, csi_random, persistent_field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WINDOWS = (1, 2, 3, 5)                                    # antecedent-rain windows, days
TWIN_TAUS = (0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)
STORM_HR, TOTAL_HR = 2.0, 4.0                             # baselines.py defaults, unchanged


def csi(pred, obs, valid):
    """Hits / (hits + misses + false alarms) over the scoreable cells only."""
    p = pred & valid
    o = obs & valid
    hits = int((p & o).sum())
    return dict(csi=hits / max(int((p | o).sum()), 1),
                pod=hits / max(int(o.sum()), 1),
                bias=int(p.sum()) / max(int(o.sum()), 1),
                precision=hits / max(int(p.sum()), 1))


def twin_grids(area, dates, work, device=None):
    """hmax for every (date, window): the twin forced with that date's antecedent rain."""
    import torch

    from varuna.build.twin import build_domain

    rain = json.load(open(os.path.join(HERE, "data", "rain_features.json")))[area]["rain"]
    dom = build_domain(work, device=device)
    out = {}
    for d in dates:
        for w in WINDOWS:
            mm = float(rain[d][f"rain_{w}d"])
            with torch.no_grad():
                h = dom.simulate(dom.z0, rain_mm=mm, storm_hr=STORM_HR, total_hr=TOTAL_HR)
            out[(d, w)] = h.detach().cpu().numpy()
    return dom, out


def check_alignment(D, area, work, dom):
    """Prove the stack's truth/mask are the same cells the twin baseline scores on."""
    from varuna.build.calibrate import align_sar, valid_mask
    v_twin = valid_mask(work, dom).cpu().numpy() > 0.5
    v_stack = D.valid[area]
    if v_twin.shape != v_stack.shape:
        raise AssertionError(f"{area}: grid mismatch {v_twin.shape} vs stack {v_stack.shape}")
    dv = int((v_twin != v_stack).sum())
    dy = 0
    for i, d in enumerate(D.dates[area]):
        y_twin = align_sar(work, d, dom).cpu().numpy() > 0.5
        dy += int((y_twin != D.Y[area][i]).sum())
    n = v_stack.size
    print(f"  alignment {area}: mask differs in {dv}/{n} cells, truth in {dy}/{n * len(D.dates[area])}")
    if dv or dy:
        raise AssertionError(f"{area}: stack and twin crop disagree - not like-for-like")


def score_candidates(cands, samples, Y, valid):
    """CSI of every candidate (threshold, ...) on every storm -> {cand: {storm_key: csi}}."""
    return {c: {(s["area"], s["date"]): csi(pred_fn(s), Y[s["area"]][s["idx"]], valid[s["area"]])["csi"]
                for s in samples}
            for c, pred_fn in cands.items()}


def pick_and_score(table, samples, protocol):
    """Select one candidate per fold, return per-storm CSI under the chosen selection rule.

    loso   - the candidate maximising mean CSI over the OTHER 17 storms, matching the net's
             "threshold swept on train scenes only" rule.
    global - one candidate maximising mean CSI over ALL storms, the old table's rule. The gap
             between the two is the optimism the old number carried.
    """
    keys = [(s["area"], s["date"]) for s in samples]
    out = {}
    if protocol == "global":
        best = max(table, key=lambda c: np.mean([table[c][k] for k in keys]))
        return {k: table[best][k] for k in keys}, {k: best for k in keys}
    chosen = {}
    for k in keys:
        others = [o for o in keys if o != k]
        best = max(table, key=lambda c: np.mean([table[c][o] for o in others]))
        out[k] = table[best][k]
        chosen[k] = best
    return out, chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", default="patna,mumbai_northeast")
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "twin_protocol.json"))
    args = ap.parse_args(argv)

    areas = [a for a in args.areas.split(",") if a]
    D = VarunaData(os.path.join(HERE, "data", "varuna_stack.npz"),
                   os.path.join(HERE, "data", "rain_features.json"),
                   grid_m=args.grid, keep_areas=areas)
    print(f"grid {args.grid} m | {len(D.samples)} storms | areas {areas}")

    t0 = time.time()
    doms, grids = {}, {}
    for a in areas:
        work = os.path.join(args.artifacts, a)
        dom, g = twin_grids(a, D.dates[a], work)
        check_alignment(D, a, work, dom)
        doms[a], grids[a] = dom, g
    print(f"  {sum(len(g) for g in grids.values())} twin simulations in {time.time() - t0:.0f}s")

    # ---- candidate predictors -------------------------------------------------------------
    # The twin sweeps (tau, window) exactly as baselines.py does. The terrain indices are read
    # from the same feature stack the net saw, so no method gets a different version of the DEM.
    F = D.feature_names
    layer = {a: {n: D.X[a][F.index(n)] for n in ("depth_depression", "twi", "hand")} for a in areas}

    def q(name, lo, hi, n):
        """Thresholds as quantiles pooled over both domains: one global threshold, as before."""
        v = np.concatenate([layer[a][name][D.valid[a]].ravel() for a in areas])
        return np.quantile(v, np.linspace(lo, hi, n))

    methods = {}
    methods["dynamic_twin"] = {
        (round(float(t), 4), w): (lambda s, t=t, w=w: grids[s["area"]][(s["date"], w)] > t)
        for t in TWIN_TAUS for w in WINDOWS}
    methods["static_depth"] = {
        (round(float(t), 4),): (lambda s, t=t: layer[s["area"]]["depth_depression"] > t)
        for t in np.linspace(0.05, 1.5, 30)}
    methods["twi"] = {
        (round(float(t), 4),): (lambda s, t=t: layer[s["area"]]["twi"] > t)
        for t in q("twi", 0.50, 0.99, 25)}
    methods["hand_lite"] = {
        (round(float(t), 4),): (lambda s, t=t: layer[s["area"]]["hand"] < t)
        for t in q("hand", 0.01, 0.50, 40)}

    # ---- zero-knowledge references, computed on these same scenes -------------------------
    ref = {"all_wet": {}, "random": {}, "climatology": {}}
    for s in D.samples:
        k = (s["area"], s["date"])
        y, v = D.Y[s["area"]][s["idx"]], D.valid[s["area"]]
        ref["all_wet"][k] = csi_allwet(y, v)
        ref["random"][k] = csi_random(y, v)
        masks = [D.Y[s["area"]][i] for i in range(len(D.dates[s["area"]]))]
        ref["climatology"][k] = csi(persistent_field(masks, s["idx"]), y, v)["csi"]

    # ---- score ----------------------------------------------------------------------------
    report = {"meta": dict(areas=areas, grid_m=args.grid, storms=len(D.samples),
                           dates={a: D.dates[a] for a in areas},
                           dropped=["patna/2024-07-07 (zero observed wet cells)"],
                           windows=list(WINDOWS), taus=list(TWIN_TAUS),
                           rain="cached rain_features.json (same file the net used)"),
              "reference": {k: round(float(np.mean(list(v.values()))), 4) for k, v in ref.items()},
              "methods": {}}

    print(f"\n{'method':<16}{'LOSO CSI':>10}{'sd':>8}{'global CSI':>12}{'optimism':>10}")
    print("-" * 56)
    for name, cands in methods.items():
        table = score_candidates(cands, D.samples, D.Y, D.valid)
        loso, chosen = pick_and_score(table, D.samples, "loso")
        glob, gpick = pick_and_score(table, D.samples, "global")
        lv = np.array(list(loso.values()))
        gv = np.array(list(glob.values()))
        # bias/POD at the LOSO choice, so the twin is described the same way the net is
        det = []
        for s in D.samples:
            k = (s["area"], s["date"])
            det.append(csi(cands[chosen[k]](s), D.Y[s["area"]][s["idx"]], D.valid[s["area"]]))
        report["methods"][name] = dict(
            loso_csi=round(float(lv.mean()), 4), loso_sd=round(float(lv.std()), 4),
            global_csi=round(float(gv.mean()), 4), global_param=list(gpick[list(gpick)[0]]),
            pod=round(float(np.mean([d["pod"] for d in det])), 4),
            bias=round(float(np.mean([d["bias"] for d in det])), 4),
            precision=round(float(np.mean([d["precision"] for d in det])), 4),
            per_storm={f"{a}/{d}": round(loso[(a, d)], 4) for a, d in loso})
        print(f"{name:<16}{lv.mean():>10.4f}{lv.std():>8.4f}{gv.mean():>12.4f}"
              f"{gv.mean() - lv.mean():>+10.4f}")

    print("-" * 56)
    for k, v in report["reference"].items():
        print(f"{k:<16}{v:>10.4f}")

    print(f"\nnet (terrain-only, this protocol, 3 seeds): 0.2895 +/- 0.0064")
    tw = report["methods"]["dynamic_twin"]["loso_csi"]
    print(f"net / twin on identical scenes: {0.2895 / max(tw, 1e-9):.1f}x   "
          f"(RESULTS.md previously quoted 7.1x from the 8-date Patna table)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return report


if __name__ == "__main__":
    main()
