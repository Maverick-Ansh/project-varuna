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


def domain(work, device=None):
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    from varuna.build.twin import build_domain
    return build_domain(work, device=device)


def twin_grids(area, dates, dom, cache=None):
    """hmax for every (date, window): the twin forced with that date's antecedent rain.

    28 CPU-minutes of Bates-2010 routing, so it is cached. The cache is keyed by the physics
    that produced it - area, date, window, storm/total hours - and is a pure function of them,
    so a stale entry can only appear if the simulator itself changes. Delete the file after
    touching twin.step. It is deliberately not committed: recomputable, and 14 MB.
    """
    import torch
    rain = json.load(open(os.path.join(HERE, "data", "rain_features.json")))[area]["rain"]
    have = {}
    if cache and os.path.exists(cache):
        z = np.load(cache)
        have = {k: z[k] for k in z.files}
    out, fresh = {}, 0
    n = len(dates) * len(WINDOWS)
    for j, d in enumerate(dates):
        t = time.time()
        for w in WINDOWS:
            key = f"{area}|{d}|{w}|{STORM_HR}|{TOTAL_HR}"
            if key in have:
                out[(d, w)] = have[key]
                continue
            mm = float(rain[d][f"rain_{w}d"])
            with torch.no_grad():
                h = dom.simulate(dom.z0, rain_mm=mm, storm_hr=STORM_HR, total_hr=TOTAL_HR)
            out[(d, w)] = have[key] = h.detach().cpu().numpy().astype("float32")
            fresh += 1
        if fresh:
            print(f"    {area} {d}  sims in {time.time() - t:.0f}s ({(j + 1) * len(WINDOWS)}/{n})",
                  flush=True)
    if cache and fresh:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **have)
        print(f"    cached {len(have)} grids -> {cache}", flush=True)
    elif not fresh:
        print(f"    {area}: all {n} grids from cache", flush=True)
    return out


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

    loso        - the candidate maximising mean CSI over the OTHER 17 storms, matching the
                  net's "threshold swept on train scenes only" rule.
    loso_domain - same, restricted to the held-out storm's own domain. More generous to the
                  terrain indices, whose useful threshold is domain-specific.
    global - one candidate maximising mean CSI over ALL storms, the old table's rule. The gap
             between the two is the optimism the old number carried.
    """
    keys = [(s["area"], s["date"]) for s in samples]
    out, chosen = {}, {}
    if protocol == "global":
        best = max(table, key=lambda c: np.mean([table[c][k] for k in keys]))
        return {k: table[best][k] for k in keys}, {k: best for k in keys}
    for k in keys:
        # loso: every other storm.  loso_domain: every other storm OF THAT DOMAIN. The second
        # exists because Patna's HAND runs to 15 m and Mumbai-NE's to 200 m, so one pooled
        # threshold is a handicap the net does not carry - it gets per-domain standardisation.
        others = [o for o in keys if o != k and (protocol == "loso" or o[0] == k[0])]
        if not others:
            others = [o for o in keys if o != k]
        best = max(table, key=lambda c: np.mean([table[c][o] for o in others]))
        out[k] = table[best][k]
        chosen[k] = best
    return out, chosen


def reproduce_old_table(args, doms):
    """Re-derive the committed 8-date Patna table, to prove this file scores the way that one did.

    Same domain, same 8 dates INCLUDING the dead 2024-07-07, one global threshold swept over all
    of them - the old protocol exactly. Two differences are known and expected, and are the point
    of printing the deltas rather than asserting equality:

      * static depth: the old table max-pooled depth.tif to 60 m, the feature stack mean-pools it.
      * the twin: the old table fetched rain live; we replay the cached rain_features.json.

    TWI and HAND-lite share pooling and inputs with the original, so those two are the real test.
    """
    ref_path = os.path.join(args.artifacts, "patna", "baseline_comparison.json")
    if not os.path.exists(ref_path):
        print("\nno baseline_comparison.json - skipping reproduction check")
        return None
    ref = json.load(open(ref_path))
    D8 = VarunaData(os.path.join(HERE, "data", "varuna_stack.npz"),
                    os.path.join(HERE, "data", "rain_features.json"),
                    grid_m=args.grid, keep_areas=["patna"], drop_dead=False)
    g = twin_grids("patna", D8.dates["patna"], doms["patna"], args.cache)
    F = D8.feature_names
    L = {n: D8.X["patna"][F.index(n)] for n in ("depth_depression", "twi", "hand")}
    V = D8.valid["patna"]
    cands = {
        "dynamic_twin": {(round(float(t), 4), w): (lambda s, t=t, w=w: g[(s["date"], w)] > t)
                         for t in TWIN_TAUS for w in WINDOWS},
        "static_depth": {(round(float(t), 4),): (lambda s, t=t: L["depth_depression"] > t)
                         for t in np.linspace(0.05, 1.5, 30)},
        "twi": {(round(float(t), 4),): (lambda s, t=t: L["twi"] > t)
                for t in np.quantile(L["twi"][V], np.linspace(0.50, 0.99, 25))},
        "hand_lite": {(round(float(t), 4),): (lambda s, t=t: L["hand"] < t)
                      for t in np.linspace(0.25, 15, 40)}}
    print("\nREGRESSION CHECK - old protocol, 8 Patna dates, one global threshold")
    print(f"{'method':<16}{'committed':>11}{'here':>9}{'delta':>9}")
    print("-" * 45)
    out = {}
    for name, c in cands.items():
        t = score_candidates(c, D8.samples, D8.Y, D8.valid)
        v, _ = pick_and_score(t, D8.samples, "global")
        mine = float(np.mean(list(v.values())))
        theirs = ref.get(name, {}).get("mean_csi")
        out[name] = dict(committed=theirs, here=round(mine, 4))
        shown = "n/a" if theirs is None else f"{theirs:.4f}"
        delta = 0.0 if theirs is None else mine - theirs
        print(f"{name:<16}{shown:>11}{mine:>9.4f}{delta:>+9.4f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", default="patna,mumbai_northeast")
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "twin_protocol.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "results", "twin_grids.npz"))
    ap.add_argument("--reproduce", type=int, default=1,
                    help="also re-derive the committed 8-date Patna table as a regression check")
    args = ap.parse_args(argv)

    areas = [a for a in args.areas.split(",") if a]
    D = VarunaData(os.path.join(HERE, "data", "varuna_stack.npz"),
                   os.path.join(HERE, "data", "rain_features.json"),
                   grid_m=args.grid, keep_areas=areas)
    print(f"grid {args.grid} m | {len(D.samples)} storms | areas {areas}", flush=True)

    doms = {a: domain(os.path.join(args.artifacts, a)) for a in areas}
    for a in areas:                          # cheap, and it is the assumption everything rests on
        check_alignment(D, a, os.path.join(args.artifacts, a), doms[a])

    t0 = time.time()
    grids = {a: twin_grids(a, D.dates[a], doms[a], args.cache) for a in areas}
    print(f"  {sum(len(g) for g in grids.values())} twin simulations in {time.time() - t0:.0f}s",
          flush=True)

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
    # baselines.py's own grid, verbatim. An earlier version of this file swept HAND over pooled
    # quantiles 0.01-0.50, which tops out near 0.8 m in Patna - below the 8.57 m the original
    # table selected - and so scored HAND-lite at a threshold it would never have chosen.
    methods["hand_lite"] = {
        (round(float(t), 4),): (lambda s, t=t: layer[s["area"]]["hand"] < t)
        for t in np.linspace(0.25, 15, 40)}

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

    print(f"\n{'method':<16}{'LOSO':>9}{'sd':>8}{'LOSO/dom':>10}{'global':>9}{'bias':>7}{'POD':>7}")
    print("-" * 66)
    for name, cands in methods.items():
        table = score_candidates(cands, D.samples, D.Y, D.valid)
        loso, chosen = pick_and_score(table, D.samples, "loso")
        dom_, _ = pick_and_score(table, D.samples, "loso_domain")
        glob, gpick = pick_and_score(table, D.samples, "global")
        lv = np.array(list(loso.values()))
        dv = np.array(list(dom_.values()))
        gv = np.array(list(glob.values()))
        # bias/POD at the LOSO choice, so the twin is described the same way the net is
        det = []
        for s in D.samples:
            k = (s["area"], s["date"])
            det.append(csi(cands[chosen[k]](s), D.Y[s["area"]][s["idx"]], D.valid[s["area"]]))
        report["methods"][name] = dict(
            loso_csi=round(float(lv.mean()), 4), loso_sd=round(float(lv.std()), 4),
            loso_domain_csi=round(float(dv.mean()), 4),
            best_csi=round(float(max(lv.mean(), dv.mean())), 4),
            global_csi=round(float(gv.mean()), 4), global_param=list(gpick[list(gpick)[0]]),
            pod=round(float(np.mean([d["pod"] for d in det])), 4),
            bias=round(float(np.mean([d["bias"] for d in det])), 4),
            precision=round(float(np.mean([d["precision"] for d in det])), 4),
            per_storm={f"{a}/{d}": round(loso[(a, d)], 4) for a, d in loso})
        print(f"{name:<16}{lv.mean():>9.4f}{lv.std():>8.4f}{dv.mean():>10.4f}{gv.mean():>9.4f}"
              f"{np.mean([d['bias'] for d in det]):>7.2f}"
              f"{np.mean([d['pod'] for d in det]):>7.3f}")

    print("-" * 56)
    for k, v in report["reference"].items():
        print(f"{k:<16}{v:>10.4f}")

    print("\nLOSO/dom lets each domain keep its own threshold - more generous to the terrain\n"
          "indices than the pooled rule, and the number quoted for them below.")
    print(f"\nnet (terrain-only, this protocol, 3 seeds): 0.2895 +/- 0.0064")
    for name, m in report["methods"].items():
        print(f"  net / {name:<14} {0.2895 / max(m['best_csi'], 1e-9):>5.1f}x   "
              f"(best of its protocols: {m['best_csi']:.4f})")
    print(f"  net / all-wet        {0.2895 / max(report['reference']['all_wet'], 1e-9):>5.1f}x")
    print(f"  net / climatology    {0.2895 / max(report['reference']['climatology'], 1e-9):>5.1f}x")

    if args.reproduce:
        report["reproduction"] = reproduce_old_table(args, doms)


    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return report


if __name__ == "__main__":
    main()
