"""Aggregate the threshold-calibration result that was already sitting in `results/`.

Every run has always written `csi_cal` and `thr_cal` per scene alongside `csi`, but they were only
ever read once, on a 3-domain LODO split, where the gain was +0.0154 at about one seed standard
deviation and was written up as a suggestion. At 14 and 15 domains, on the leave-one-REGION-out
split that actually corresponds to deployment, it is a replicated effect and it changes sign
between the two holdout schemes. That is the mechanism, not noise: hold out a tile and its siblings
keep the training wet-fraction prior valid, so the swept global threshold is already close to right;
hold out a region and the prior is genuinely wrong for ground the model has never seen.

This is the builder for Tables `tab:calibration` and `tab:calregion` of the paper, written so those
numbers are not another result that exists only in a scratch cell.

    python -m csi_net.calibration_report
    python -m csi_net.calibration_report --json out.json
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# `--ablate rain` ZEROES the rain channels, so `*_rain_*` is the terrain-only run and `*_none_*` is
# the run with rainfall present. The filenames read backwards and this has caught us out before.
TERRAIN_ONLY, WITH_RAIN = "rain", "none"

SPLITS = [
    ("region, 15 areas", "loro", TERRAIN_ONLY, "varuna_stack15"),
    ("region, 14 areas", "loro", TERRAIN_ONLY, "varuna_stack14"),
    ("tile, 14 areas", "lodo", TERRAIN_ONLY, "varuna_stack14"),
    ("region, 15 areas (with rain)", "loro", WITH_RAIN, "varuna_stack15"),
    ("region, 14 areas (with rain)", "loro", WITH_RAIN, "varuna_stack14"),
]


def region_of(area):
    """Same grouping as run.REGIONS. Chennai is its own region on purpose: pooling it with Mumbai
    into one 'coastal' region would hold both out together and destroy the coverage test."""
    if area.startswith("mumbai"):
        return "mumbai"
    if area.startswith("patna"):
        return "patna"
    if area.startswith("chennai"):
        return "chennai"
    return "karnataka"


def load(split, ablate, stack):
    """Seed files for one configuration. The single-digit seed glob is deliberate: it excludes the
    `_patna+mumbai_*` area-subset runs, which average a different dataset and would contaminate."""
    pat = f"{split}_{ablate}_w32_g60_s?_{stack}.json"
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, pat)))]


def summarise(runs):
    g = [r["mean"]["csi"] for r in runs]
    c = [r["mean"]["csi_cal"] for r in runs]
    o = [r["mean"]["csi_opt"] for r in runs]
    delta = st.mean(c) - st.mean(g)
    gap = st.mean(o) - st.mean(g)
    sd = st.stdev(g) if len(g) > 1 else float("nan")
    return dict(n_seeds=len(runs), csi=st.mean(g), csi_sd=sd, csi_cal=st.mean(c),
                csi_cal_sd=st.stdev(c) if len(c) > 1 else float("nan"),
                csi_opt=st.mean(o), delta=delta, delta_in_sd=delta / sd if sd else float("nan"),
                gap_closed_pct=100.0 * delta / gap if gap else float("nan"))


def by_region(runs):
    """Per-region means, averaged over seeds. `bias` is predicted wet cells over observed wet cells
    at the global threshold, so below 1.0 means the model is under-warning."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in runs:
        rows = defaultdict(list)
        for row in r["rows"]:
            rows[region_of(row["area"])].append(row)
        for reg, rs in rows.items():
            for key in ("csi", "csi_cal", "bias", "csi_allwet"):
                acc[reg][key].append(st.mean(x[key] for x in rs))
    out = {}
    for reg, a in acc.items():
        csi, cal = st.mean(a["csi"]), st.mean(a["csi_cal"])
        out[reg] = dict(csi=csi, csi_cal=cal, bias=st.mean(a["bias"]),
                        change_pct=100.0 * (cal - csi) / csi if csi else float("nan"),
                        x_allwet=csi / st.mean(a["csi_allwet"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="also write the report to this path")
    args = ap.parse_args(argv)

    report = {"splits": {}, "regions": {}}

    print("Threshold calibration: match the predicted wet fraction to the TRAINING scenes' wet")
    print("fraction. No test label is touched, so this is a free correction and not a peek.\n")
    print(f"{'held out':30s} {'global':>8s} {'calib':>8s} {'delta':>8s} {'in sd':>7s} {'gap closed':>11s}")
    for label, split, ablate, stack in SPLITS:
        runs = load(split, ablate, stack)
        if not runs:
            print(f"{label:30s} {'MISSING':>8s}")
            continue
        s = summarise(runs)
        report["splits"][label] = s
        print(f"{label:30s} {s['csi']:8.4f} {s['csi_cal']:8.4f} {s['delta']:+8.4f} "
              f"{s['delta_in_sd']:7.1f} {s['gap_closed_pct']:10.0f}%")

    runs = load("loro", TERRAIN_ONLY, "varuna_stack15")
    if runs:
        report["regions"] = by_region(runs)
        print(f"\nBy held-out region, terrain only, 15 areas, {len(runs)} seeds:")
        print(f"{'region':14s} {'global':>8s} {'calib':>8s} {'change':>8s} {'bias':>6s} {'x all-wet':>10s}")
        for reg in sorted(report["regions"], key=lambda r: report["regions"][r]["csi"]):
            v = report["regions"][reg]
            print(f"{reg:14s} {v['csi']:8.4f} {v['csi_cal']:8.4f} {v['change_pct']:+7.1f}% "
                  f"{v['bias']:6.2f} {v['x_allwet']:9.1f}x")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.json}")
    return report


if __name__ == "__main__":
    main()
