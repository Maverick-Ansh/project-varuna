"""Tier A: run the Karnataka state screen (tables only — no rasters ever cross the wire).

On Colab (EE authed):
    python scripts/run_state_screen.py --project <gcp-project>

The EE table is cached to artifacts/karnataka/state_screen_raw.json on first fetch; every
later run rescores from the cache, so weight/hyperparameter tweaking is offline and instant:
    python scripts/run_state_screen.py --w-need 0.55 --jitter 0.4

Taluk upgrade (after scripts/spike_boundaries.py shows join coverage worth the claim):
    python scripts/run_state_screen.py --project <p> --admin-asset users/<you>/karnataka_taluks \
        --admin-level taluk --name-prop KGISTalukName
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from varuna.build import state_screen  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="karnataka", choices=["karnataka"])
    ap.add_argument("--project", default=None, help="GCP project for Earth Engine")
    ap.add_argument("--cgwb", default="data/cgwb_karnataka_2024_district.csv")
    ap.add_argument("--out-dir", default="artifacts/karnataka")
    ap.add_argument("--admin-asset", default=None,
                    help="EE asset id of taluk polygons (default: GAUL districts)")
    ap.add_argument("--admin-level", default="district", choices=["district", "taluk"])
    ap.add_argument("--name-prop", default=None, help="name property on --admin-asset")
    ap.add_argument("--refetch", action="store_true",
                    help="delete the cached EE table and fetch fresh")
    for k, v in state_screen.ROI_WEIGHTS.items():
        ap.add_argument(f"--w-{k}", type=float, default=v, help=f"ROI weight (default {v})")
    ap.add_argument("--ksat-cap", type=float, default=state_screen.KSAT_CAP_MM_HR)
    ap.add_argument("--n-perturb", type=int, default=state_screen.SENSITIVITY_N)
    ap.add_argument("--jitter", type=float, default=state_screen.SENSITIVITY_JITTER)
    args = ap.parse_args()

    raw_path = os.path.join(args.out_dir, "state_screen_raw.json")
    if args.refetch and os.path.exists(raw_path):
        os.remove(raw_path)

    weights = {k: getattr(args, f"w_{k}") for k in state_screen.ROI_WEIGHTS}
    payload = state_screen.run(
        cgwb_csv=args.cgwb, out_dir=args.out_dir, project_id=args.project,
        admin_asset=args.admin_asset, admin_level=args.admin_level, name_prop=args.name_prop,
        weights=weights, ksat_cap=args.ksat_cap, n_perturb=args.n_perturb, jitter=args.jitter)

    print(f"\nadmin_level={payload['admin_level']}  "
          f"join coverage={payload['join_report']['coverage']:.0%}")
    if payload["join_report"]["unmatched_cgwb"]:
        print("UNMATCHED CGWB rows:", payload["join_report"]["unmatched_cgwb"])
    print(f"\ntop 10 of {len(payload['blocks'])} units "
          f"(rank | unit | category | stage% | trend mm/yr | ROI | rank p5-p95):")
    for b in payload["blocks"][:10]:
        print(f"  {b['rank']:>3} | {b['unit']:<22} | {b['category']:<14} "
              f"| {b['stage_pct']:>6.1f} | {b['gws_trend_mm_yr']:>7.1f} "
              f"| {b['roi']:.3f} | {b['rank_p5']}-{b['rank_p95']}")
    print(f"\nstable top-10 ({payload['sensitivity']['n_perturbations']} perturbations, "
          f"±{payload['sensitivity']['jitter']:.0%} weights): "
          f"{payload['sensitivity']['stable_top10']}")
    for n in payload["screen_notes"]:
        print("note:", n)
    print("\nReality check (plan §Verification 7): Kolar, Chikkaballapur and Bengaluru Urban "
          "are long-standing over-exploited units — if they are not near the top, the screen "
          "is wrong.")


if __name__ == "__main__":
    main()
