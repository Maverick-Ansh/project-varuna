"""Rebuild `data/trivial_baselines.json`: what a CSI is worth on this task with no model.

Table~\ref{tab:trivial} of the paper is the project's most portable claim -- that a
parameter-free climatology scores 0.30-0.58 on Sentinel-1 flood extent, higher than physics
models in this literature report -- and it was the last number still coming from a scratch cell
with no code behind it. This is that code.

    python -m csi_net.build_trivial --verify              # reproduce the committed 8-date table
    python -m csi_net.build_trivial --stack data/varuna_stack14.npz --out data/trivial_baselines14.json

`--keep-dead` reproduces the committed file exactly. The default DROPS scenes with no observed
wet cells, because a scene with no ground truth contributes a forced 0.0 to every method's mean:
the committed Patna row averages 2024-07-07 into all four of its columns while the paper's own
protocol paragraph says that date is dropped.
"""
import argparse
import json
import os

import numpy as np

from .data import DEAD_SCENES, pool

HERE = os.path.dirname(os.path.abspath(__file__))
CLIM_GRID = np.arange(0.05, 1.0, 0.05)   # the committed file's thresholds are all multiples of 0.05

# The committed climatology sweeps this threshold PER DOMAIN and keeps the best (Patna 0.15,
# Harbour 0.35, Mumbai-NE 0.45), i.e. it is tuned on the same scenes it is then reported on.
# That makes it a legitimate *ceiling* for "how far does 'these cells are usually wet' go", but
# it is not parameter-free, and the paper's caption should not call it that. Both are reported:
# `mean_csi` (swept, reproduces the committed table) and `mean_csi_fixed` at a single threshold
# shared by every domain, which is the honestly out-of-sample version.
FIXED_THRESHOLD = 0.15


def csi(pred, obs, valid):
    p, o = pred & valid, obs & valid
    hits = int((p & o).sum())
    return hits / max(hits + int((~p & o).sum()) + int((p & ~o).sum()), 1)


def area_baselines(Y, valid, dates):
    """all-wet, random, leave-one-out climatology and SAR-vs-SAR agreement for one domain."""
    n_valid = int(valid.sum())
    allwet, rand, per_date, freqs = [], [], {}, []
    for i, d in enumerate(dates):
        obs = Y[i] & valid
        w = int(obs.sum())
        frac = w / max(n_valid, 1)
        # all-wet predicts every valid cell: hits = w, misses = 0, false alarms = n_valid - w
        a = w / max(n_valid, 1)
        # a random predictor firing at the base rate: W / (2N - W)
        r = w / max(2 * n_valid - w, 1)
        # climatology: cells wet in enough of the OTHER storms of this domain
        others = [Y[j] for j in range(len(dates)) if j != i]
        freq = (np.mean([o.astype("float32") for o in others], axis=0) if others
                else np.zeros_like(obs, "float32"))
        freqs.append(freq)
        allwet.append(a); rand.append(r)
        per_date[d] = dict(obs_wet=w, wet_frac=round(frac, 6),
                           csi_allwet=round(a, 4), csi_random=round(r, 4))
    def clim_at(t):
        return [csi(f >= t, Y[i], valid) for i, f in enumerate(freqs)]

    best_t = max(CLIM_GRID, key=lambda t: float(np.mean(clim_at(t))))
    clim = clim_at(best_t)
    clim_fixed = clim_at(FIXED_THRESHOLD)
    for d, c in zip(dates, clim):
        per_date[d]["csi_clim"] = round(c, 4)

    pairs = [csi(Y[i], Y[j], valid) for i in range(len(dates)) for j in range(i + 1, len(dates))]
    return dict(
        n_valid_cells=n_valid, n_dates=len(dates), dates=list(dates),
        mean_csi_allwet=round(float(np.mean(allwet)), 4),
        mean_csi_random=round(float(np.mean(rand)), 4),
        climatology=dict(threshold=round(float(best_t), 2),
                         mean_csi=round(float(np.mean(clim)), 4),
                         fixed_threshold=FIXED_THRESHOLD,
                         mean_csi_fixed=round(float(np.mean(clim_fixed)), 4),
                         per_date={d: per_date[d]["csi_clim"] for d in dates}),
        sar_self_consistency=dict(mean=round(float(np.mean(pairs)), 4) if pairs else None,
                                  min=round(float(np.min(pairs)), 4) if pairs else None,
                                  max=round(float(np.max(pairs)), 4) if pairs else None,
                                  n_pairs=len(pairs)),
        per_date=per_date)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default=os.path.join(HERE, "data", "varuna_stack.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "trivial_baselines.json"))
    ap.add_argument("--grid", type=int, default=60, choices=[30, 60, 120])
    ap.add_argument("--keep-dead", action="store_true",
                    help="keep scenes with zero observed wet cells (reproduces the committed file)")
    ap.add_argument("--verify", default="", help="compare against this json instead of writing")
    args = ap.parse_args(argv)

    z = np.load(args.stack, allow_pickle=True)
    k = {30: 1, 60: 2, 120: 4}[args.grid]
    out = {}
    for a in [str(s) for s in z["areas"]]:
        Y = pool(z[f"{a}__Y"].astype("float32"), k, "max") > 0.5
        valid = pool(z[f"{a}__jrc30"].astype("float32"), k, "mean") < 50
        dates = [str(s) for s in z[f"{a}__dates"]]
        keep = [i for i, d in enumerate(dates)
                if args.keep_dead or ((a, d) not in DEAD_SCENES and int((Y[i] & valid).sum()) > 0)]
        dropped = [dates[i] for i in range(len(dates)) if i not in keep]
        if dropped:
            print(f"  {a}: dropped {len(dropped)} scene(s) with no ground truth: {', '.join(dropped)}")
        out[a] = area_baselines(Y[keep], valid, [dates[i] for i in keep])
        b = out[a]
        print(f"  {a:<18} {b['n_dates']:2d} storms  allwet {b['mean_csi_allwet']:.4f}  "
              f"random {b['mean_csi_random']:.4f}  "
              f"clim {b['climatology']['mean_csi']:.4f}@{b['climatology']['threshold']:.2f} "
              f"(fixed {b['climatology']['mean_csi_fixed']:.4f})  "
              f"SARvSAR {b['sar_self_consistency']['mean']}")

    if args.verify:
        ref = json.load(open(args.verify))
        print(f"\nverify against {os.path.basename(args.verify)}")
        for a in out:
            if a not in ref:
                print(f"  {a}: absent from reference"); continue
            for key, new, old in [
                    ("allwet", out[a]["mean_csi_allwet"], ref[a]["mean_csi_allwet"]),
                    ("random", out[a]["mean_csi_random"], ref[a]["mean_csi_random"]),
                    ("clim", out[a]["climatology"]["mean_csi"], ref[a]["climatology"]["mean_csi"]),
                    ("sarvsar", out[a]["sar_self_consistency"]["mean"],
                     ref[a]["sar_self_consistency"]["mean"])]:
                flag = "ok " if abs(new - old) < 5e-4 else "DIFF"
                print(f"  {flag} {a:<18} {key:<8} {new:.4f} vs {old:.4f}")
        return out

    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)
    return out


if __name__ == "__main__":
    main()
