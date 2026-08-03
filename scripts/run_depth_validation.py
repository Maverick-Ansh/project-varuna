"""Score the twin against REAL observed water depths.

    python scripts/run_depth_validation.py --days 365 --window 24

Pulls crowdsourced depth reports (mumbaiflood.in, IIT Bombay + MCGM), quality-controls them,
looks up the rain that actually fell at each point in the hours before it was reported
(Open-Meteo), runs the emulator for that storm and compares the predicted depth in the
observation's own cell against what was reported.

Writes `artifacts/depth_validation.json` (metrics + QC ledger) and `depth_observations.json`
(the QC'd set, no personal data). Re-running with --cache-only skips the network and rescores
the saved observations, which is what you want when changing the scoring, not the data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from varuna.areas import artifacts_root  # noqa: E402
from varuna.groundtruth.observations import ObservationSet  # noqa: E402
from varuna.groundtruth.score import (assign_areas, neighbourhood_scores, predict_depths,  # noqa: E402
                                      report)
from varuna.groundtruth.sources import crowdsourced_observations  # noqa: E402


def _fmt(m, label):
    if not m or m.get("n", 0) == 0:
        return f"{label:<16} (no observations)"
    mo = m["model"]
    return (f"{label:<16} n={m['n']:<4} MAE {mo['mae_m']:.3f} m  RMSE {mo['rmse_m']:.3f} m  "
            f"bias {mo['bias_m']:+.3f} m  skill {m['skill_vs_best_baseline']}  "
            f"CSI {m['binary']['csi']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=365, help="how far back to pull observations")
    ap.add_argument("--window", type=int, default=24, help="antecedent rainfall window (hours)")
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--cache-only", action="store_true",
                    help="rescore the saved observations without fetching")
    ap.add_argument("--sweep", nargs="*", type=int, default=None,
                    help="also score across these antecedent windows (hours), e.g. --sweep 3 6 12 24 48")
    ap.add_argument("--min-rain", type=float, default=0.0,
                    help="drop observations with less antecedent rain than this (mm): water that "
                         "no rain could have caused is not pluvial flooding")
    ap.add_argument("--out", default=None, help="output dir (default: artifacts/)")
    args = ap.parse_args()

    out_dir = args.out or artifacts_root()
    obs_path = os.path.join(out_dir, "depth_observations.json")

    if args.cache_only:
        obs = ObservationSet.load(obs_path)
        print(f"loaded {len(obs)} observations from {obs_path}")
    else:
        obs = crowdsourced_observations(days=args.days)
        print(f"crowdsourced: kept {len(obs)} of {obs.meta['n_raw']} raw")
        for rule, n in sorted(obs.rejected.items(), key=lambda kv: -kv[1]):
            print(f"    rejected {n:4d}  {rule}")
        obs.save(obs_path)
        print(f"saved -> {obs_path}")

    tagged, unmatched = assign_areas(obs.observations, args.areas)
    inside = [o for o in tagged if o.area]
    print(f"\ninside a built domain: {len(inside)} / {len(obs)} "
          f"({len(unmatched)} fall outside every area)")
    if not inside:
        sys.exit("no observations inside any built area — nothing to score")
    from collections import Counter
    for aid, n in Counter(o.area for o in inside).most_common():
        print(f"    {aid:<16} {n}")

    print(f"\nscoring with a {args.window} h antecedent rainfall window "
          f"(Open-Meteo; one call per ~1 km cell per day)…")
    rows, skipped = predict_depths(
        inside, window_h=args.window, area_ids=args.areas, min_rain_mm=args.min_rain,
        progress=lambda i, n: print(f"    {i}/{n}", end="\r", flush=True))
    print(f"    scored {len(rows)}, skipped {len(skipped)}          ")

    rep = report(rows, skipped, obs)
    rep["window_h"] = args.window
    rep["min_rain_mm"] = args.min_rain
    rep["rows"] = rows

    # A single window choice is a free parameter, and a negative result that only holds at one
    # setting is not a result. Re-scoring across windows shows whether the verdict is a property
    # of the model or of the window we happened to pick. The rainfall cache is shared, so the
    # sweep costs almost nothing beyond the first pass.
    if args.sweep:
        sweep = {}
        cache = {}
        for w in args.sweep:
            r_w, _sk = predict_depths(inside, window_h=w, area_ids=args.areas,
                                      rain_cache=cache, min_rain_mm=args.min_rain)
            m = report(r_w, None, None)["overall"]
            sweep[str(w)] = {"n": m.get("n"),
                             "rmse_m": m.get("model", {}).get("rmse_m"),
                             "mae_m": m.get("model", {}).get("mae_m"),
                             "bias_m": m.get("model", {}).get("bias_m"),
                             "skill": m.get("skill_vs_best_baseline"),
                             "correlation": m.get("correlation"),
                             "csi": m.get("binary", {}).get("csi"),
                             "pred_wet": m.get("predicted", {}).get("n_wet"),
                             "obs_wet": m.get("observed", {}).get("n_wet"),
                             "pred_mean_m": m.get("predicted", {}).get("mean_m"),
                             "mean_rain_mm": round(sum(x["rain_mm"] for x in r_w) / max(len(r_w), 1), 1)}
            print(f"    window {w:3d} h  rain~{sweep[str(w)]['mean_rain_mm']:6.1f} mm  "
                  f"RMSE {sweep[str(w)]['rmse_m']}  skill {sweep[str(w)]['skill']}  "
                  f"r {sweep[str(w)]['correlation']}  CSI {sweep[str(w)]['csi']}  "
                  f"pred_wet {sweep[str(w)]['pred_wet']}/{sweep[str(w)]['obs_wet']}  "
                  f"pred_mean {sweep[str(w)]['pred_mean_m']} m")
        rep["window_sweep"] = sweep

    # Neighbourhood verification: let the model be right NEARBY. If the error falls as the box
    # grows, it knows where the water is and is misplacing it; if not, it does not know.
    if rows:
        from varuna.areas import area_work
        from varuna.serve.emulator import whatif_grid
        _g = {}

        def grid_fn(area, rain_mm):
            k = (area, round(rain_mm, 1))
            if k not in _g:
                _g[k] = whatif_grid(round(rain_mm, 1), None, work=area_work(area))[0]
            return _g[k]

        rep["neighbourhood"] = neighbourhood_scores(rows, grid_fn)
        print("\nneighbourhood verification (best cell within the box):")
        for rad, m in rep["neighbourhood"].items():
            print(f"    {m['box_m']:4d} m box  pred_mean {m['predicted_mean_m']:.3f}  "
                  f"RMSE {m['rmse_m']:.3f}  r {m['correlation']}  r(wet) {m['correlation_wet_only']}")

    path = os.path.join(out_dir, "depth_validation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)

    ov = rep["overall"]
    print("\n" + "=" * 78)
    print(_fmt(ov, "ALL"))
    for aid, m in sorted(rep["by_area"].items()):
        print(_fmt(m, "  " + aid))
    print("=" * 78)
    if ov.get("n"):
        print(f"observed : mean {ov['observed']['mean_m']:.3f} m  median "
              f"{ov['observed']['median_m']:.3f} m  max {ov['observed']['max_m']:.3f} m  "
              f"wet(>=0.15m) {ov['observed']['n_wet']}/{ov['n']}")
        print(f"predicted: mean {ov['predicted']['mean_m']:.3f} m  median "
              f"{ov['predicted']['median_m']:.3f} m  max {ov['predicted']['max_m']:.3f} m  "
              f"wet(>=0.15m) {ov['predicted']['n_wet']}/{ov['n']}")
        print("baselines (RMSE m): " + ", ".join(
            f"{k} {v['rmse_m']:.3f}" for k, v in ov["baselines"].items()))
        print(f"correlation observed vs predicted: {ov['correlation']}")
        print(f"\nskill > 0 means the model beats the best trivial baseline; "
              f"<= 0 means it does not.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
