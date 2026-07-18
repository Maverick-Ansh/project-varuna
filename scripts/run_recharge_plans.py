"""Run plan_recharge on every built area bundle -> {work}/recharge_plan.json + dose figure.

CPU-friendly (a few rollouts per area); GPU just makes it faster. Usage:
    python scripts/run_recharge_plans.py [--areas patna bengaluru ...] [--rain 100]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("run_recharge_plans")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="*", default=None, help="area ids (default: all built)")
    ap.add_argument("--rain", type=float, default=100.0)
    args = ap.parse_args()

    from varuna.areas import area_work, is_built, list_areas
    from varuna.serve.recharge_sites import plan_recharge, plot_recharge_dose

    ids = args.areas or [a.id for a in list_areas() if is_built(a.id)]
    for aid in ids:
        work = area_work(aid)
        t0 = time.time()
        log.info("=== %s (%s) ===", aid, work)
        rep = plan_recharge(rain_mm=args.rain, work=work)
        fig_dir = Path(work) / "figures"
        fig_dir.mkdir(exist_ok=True)
        plot_recharge_dose(rep, out=str(fig_dir / "recharge_dose.png"))
        best = max((c for c in rep["curve"] if not c["unstable"]),
                   key=lambda c: c["recharge_m3"], default=None)
        log.info("%s: %d candidates, best %s m3 (%s%% of runoff), metered=%s, sample_gw=%s [%.0fs]",
                 aid, rep["max_sites"],
                 best and best["recharge_m3"], best and best["reduction_pct"],
                 rep["metered"], rep["gw_status"].get("sample"), time.time() - t0)


if __name__ == "__main__":
    main()
