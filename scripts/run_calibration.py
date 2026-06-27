"""Colab driver for Phase-1 SAR calibration of the differentiable twin.

Auto-selects Sentinel-1 monsoon passes that actually had antecedent rain (so there is a flood
signal to fit), downloads their SAR water masks, then gradient-fits per-WorldCover-class Manning /
infiltration multipliers and reports held-out CSI before (textbook) vs after (calibrated).

Run on Colab (GPU runtime) AFTER authenticating Earth Engine in the notebook:

    !git clone -b phase1-sar-calibration https://github.com/Maverick-Ansh/project-varuna.git
    %cd project-varuna
    !pip -q install rasterio pysheds earthengine-api scikit-image
    import ee; ee.Authenticate(auth_mode='notebook'); ee.Initialize(project='YOUR_GEE_PROJECT')
    import os; os.environ['VARUNA_WORK']='artifacts/patna'; os.environ['VARUNA_PROJECT_ID']='YOUR_GEE_PROJECT'
    !python scripts/run_calibration.py --years 2024 2025 --max-dates 6 --n-test 2 --min-rain 25

The committed artifacts/patna bundle already has dem/worldcover/jrc — only the per-date
observed_water_<date>.tif masks are downloaded here, so no full rebuild is needed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("run_calibration")


def pick_dates(years, max_dates, min_rain, window_days):
    """List Sentinel-1 passes for the given years, keep those with >= min_rain antecedent rain."""
    from varuna.build.validate import list_passes
    from varuna.build.calibrate import _rain_for_date
    cands = []
    for y in years:
        try:
            cands += list_passes(year=y)
        except Exception as e:  # noqa: BLE001
            log.warning("list_passes(%s) failed: %s", y, e)
    cands = sorted(set(cands))
    log.info("found %d candidate passes across %s", len(cands), years)
    scored = []
    for d in cands:
        try:
            mm = _rain_for_date(d, window_days)
        except Exception as e:  # noqa: BLE001
            log.warning("rain lookup failed for %s: %s", d, e)
            continue
        if mm >= min_rain:
            scored.append((d, mm))
            log.info("  keep %s  (%.0f mm antecedent)", d, mm)
    scored.sort(key=lambda x: x[0])                      # chronological -> later dates held out
    return [d for d, _ in scored[:max_dates]]


def main():
    ap = argparse.ArgumentParser(description="SAR-calibrate the FloodTwin (Colab/GPU)")
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--max-dates", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=2, help="dates held out for validation")
    ap.add_argument("--min-rain", type=float, default=25.0, help="min antecedent rain (mm) to use a date")
    ap.add_argument("--window-days", type=int, default=2)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--dates", nargs="+", default=None, help="skip auto-pick; use these dates")
    ap.add_argument("--project-id", default=os.environ.get("VARUNA_PROJECT_ID"))
    args = ap.parse_args()

    from varuna.config import CFG
    from varuna.ee_auth import init_ee
    from varuna.build.calibrate import run
    log.info("work dir: %s", CFG.work)
    init_ee(args.project_id)

    dates = args.dates or pick_dates(args.years, args.max_dates, args.min_rain, args.window_days)
    if len(dates) < 2:
        raise SystemExit(f"need >=2 usable dates, got {dates}. Lower --min-rain or add --years.")
    log.info("calibrating on %d dates: %s", len(dates), dates)

    report = run(dates=dates, n_test=args.n_test, project_id=args.project_id,
                 iters=args.iters, lr=args.lr, window_days=args.window_days)

    print("\n" + "=" * 64)
    print("SAR CALIBRATION RESULT")
    print("=" * 64)
    print(f"train dates: {report['dates_train']}")
    print(f"test  dates: {report['dates_test']}  (held out)")
    print(f"  held-out CSI   textbook -> calibrated : "
          f"{report['baseline']['mean_csi_test']}  ->  {report['calibrated']['mean_csi_test']}")
    print(f"  train    CSI   textbook -> calibrated : "
          f"{report['baseline']['mean_csi_train']}  ->  {report['calibrated']['mean_csi_train']}")
    print("\nlearned per-WorldCover-class multipliers (n_mult, infil_mult):")
    print(json.dumps(report["multipliers"], indent=2))
    print(f"\nsaved: {CFG.work}/calibration_report.json  +  calibrated_params.json")


if __name__ == "__main__":
    main()
