"""Turnkey sink-field inversion — the Colab driver behind V4 Sprint 2.

For each tile: ensure SAR wet masks exist for the storm dates (GEE), fit the per-cell
drainage-sink field through the differentiable twin, evaluate on train + held-out storm
dates (CSI always next to wetness), account the per-storm drained volume (a LOWER bound
on what the city pours out), and replay the quarantined depth-validation rows with the
fitted field for the apples-to-apples textbook-vs-sinkfield pair.

Example (Mumbai, the Sprint 2 configuration):
    python scripts/run_sinkfield.py --tiles mumbai_east mumbai_west mumbai_north mumbai_south \
        --min-rain 25 --test-dates 2025-06-24 2025-09-04 2026-07-20 --project-id <gee-project>

Needs GEE only for mask downloads; with masks already in the bundles, runs offline on GPU.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--min-rain", type=float, default=25.0, help="2-day antecedent mm to keep a pass")
    ap.add_argument("--test-dates", nargs="+", default=[])
    ap.add_argument("--project-id", default=None, help="GEE project (only to download missing masks)")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out", default="artifacts/sinkfield_report.json")
    args = ap.parse_args()

    import numpy as np
    import torch

    from varuna.areas import get_area
    from varuna.build import sinkfield as SF
    from varuna.build.calibrate import align_sar, valid_mask
    from varuna.build.twin import build_domain
    from varuna.build.validate import _s1, observed_water
    from varuna.serve.weather import historical_rain_mm

    hp = dict(SF.DEFAULT_HP)
    if args.iters:
        hp["iters"] = args.iters
    if args.batch:
        hp["batch"] = args.batch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- storm dates: Sentinel-1 passes over the union of tile AOIs, with antecedent rain
    areas = [get_area(t) for t in args.tiles]
    lons = [c for a in areas for c in (a.aoi[0], a.aoi[2])]
    lats = [c for a in areas for c in (a.aoi[1], a.aoi[3])]
    ctr = (float(np.mean([a.center[0] for a in areas])),
           float(np.mean([a.center[1] for a in areas])))

    import ee
    from varuna.ee_auth import init_ee
    need_ee = False
    dates = None
    cache = f"artifacts/_s1_dates_{'_'.join(sorted(args.tiles))}.json"
    if os.path.exists(cache):
        dates = json.load(open(cache))
    else:
        init_ee(args.project_id)
        reg = ee.Geometry.Rectangle([min(lons), min(lats), max(lons), max(lats)])
        ts = []
        for y in args.years:
            end = f"{y}-10-15"
            ts += _s1(reg, f"{y}-06-01", end).aggregate_array("system:time_start").getInfo()
        dates = sorted({dt.datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d") for t in ts})
        json.dump(dates, open(cache, "w"))

    rain_city = {}
    for d in dates:
        d1 = dt.date.fromisoformat(d)
        rain_city[d] = historical_rain_mm(ctr[0], ctr[1], str(d1 - dt.timedelta(days=2)), d)
    wet_dates = [d for d in dates if rain_city[d] >= args.min_rain]
    test_dates = [d for d in args.test_dates if d in wet_dates] or wet_dates[-1:]
    train_dates = [d for d in wet_dates if d not in test_dates]
    print(f"{len(wet_dates)} storm dates | train {len(train_dates)} | test {test_dates}")

    res = {"hp": hp, "train_dates": train_dates, "test_dates": test_dates,
           "sink_eval": {}, "outflow": {}, "field_stats": {}}
    sink_doms, base_doms = {}, {}
    for tid in args.tiles:
        a = get_area(tid)
        work = a.work_dir() if hasattr(a, "work_dir") else f"artifacts/{tid}"
        for d in wet_dates:
            if not os.path.exists(f"{work}/observed_water_{d}.tif"):
                if not need_ee:
                    init_ee(args.project_id)
                    need_ee = True
                observed_water(d, work=work, reg=ee.Geometry.Rectangle(list(a.aoi)))
        dom0 = build_domain(work=work, device=device)
        base_doms[tid] = dom0
        valid = valid_mask(work, dom0)
        sars = {d: align_sar(work, d, dom0) for d in wet_dates}
        rain = {}
        for d in wet_dates:
            d1 = dt.date.fromisoformat(d)
            rain[d] = historical_rain_mm(a.center[0], a.center[1],
                                         str(d1 - dt.timedelta(days=2)), d)

        dom, sf, hist = SF.fit(dom0, sars, valid, rain, train_dates, hp=hp)
        sink_doms[tid] = dom
        res["sink_eval"][tid] = SF.eval_dates(dom, sars, valid, rain, wet_dates, test_dates,
                                              tau=hp["tau"], storm_hr=hp["storm_hr"],
                                              total_hr=hp["total_hr"])
        res["outflow"][tid] = {d: round(dom.drained_volume_m3(
            rain[d], storm_hr=hp["storm_hr"], total_hr=hp["total_hr"])[1]) for d in wet_dates}
        dmm = sf.drain_mm_h()
        res["field_stats"][tid] = dict(mean_mm_h=round(float(dmm.mean()), 3),
                                       max_mm_h=round(float(dmm.max()), 2))
        SF.save_sinkfield(f"{work}/sinkfield.pt", sf, hp)
        json.dump(hist, open(f"{work}/sinkfield_fit.json", "w"))
        print(tid, res["field_stats"][tid])

    # ---- depth-pair replay (only for tiles that appear in the stored validation rows)
    if os.path.exists("artifacts/depth_validation.json"):
        from varuna.groundtruth.score import metrics as gt_metrics
        rows = [r for r in json.load(open("artifacts/depth_validation.json"))["rows"]
                if r["area"] in sink_doms]
        if rows:
            pair = {}
            for name, doms in (("textbook", base_doms), ("sinkfield", sink_doms)):
                cache_g, out_rows = {}, []
                for r in rows:
                    key = (r["area"], round(r["rain_mm"], 1))
                    if key not in cache_g:
                        with torch.no_grad():
                            cache_g[key] = doms[r["area"]].simulate(
                                doms[r["area"]].z0, rain_mm=key[1], storm_hr=1.5,
                                total_hr=3.0).cpu().numpy()
                    rr, cc = r["cell"]
                    out_rows.append({**r, "predicted_m": round(float(cache_g[key][rr, cc]), 3)})
                m = gt_metrics(out_rows)
                obs = np.array([x["observed_m"] for x in out_rows])
                prd = np.array([x["predicted_m"] for x in out_rows])
                w = obs >= 0.15
                m["correlation_wet_only"] = (round(float(np.corrcoef(prd[w], obs[w])[0, 1]), 4)
                                             if w.sum() >= 3 and prd[w].std() > 1e-12 else None)
                pair[name] = m
            res["depth_pair"] = pair
    json.dump(res, open(args.out, "w"), default=str)
    print("report ->", args.out)


if __name__ == "__main__":
    main()
