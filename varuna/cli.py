"""Command-line entry point: python -m varuna {build|alert|chat}."""
from __future__ import annotations

import argparse
import json
import logging

from .config import CFG


def _cmd_build(args):
    if args.area:
        from .build.areas_build import build_area
        steps = [s for s in args.steps if s in ("sinks", "recharge", "twin")]
        build_area(args.area, project_id=args.project_id, n_samples=args.n_samples,
                   epochs=args.epochs, steps=steps or None)
        return
    from .build import sinks, recharge, validate, twin
    if "sinks" in args.steps:
        sinks.run(project_id=args.project_id)
    if "recharge" in args.steps:
        recharge.run(project_id=args.project_id)
    if "validate" in args.steps:
        validate.run(event_date=args.event_date, project_id=args.project_id)
    if "twin" in args.steps:
        twin.train_twin(n_samples=args.n_samples)


def _cmd_alert(args):
    from .serve.alerts import run_alerts
    r = run_alerts(rain_mm=args.rain_mm, make_map=args.map)
    print(json.dumps(r, indent=2, default=float))


def _cmd_chat(args):
    from .agent.chat import repl
    repl(four_bit=args.four_bit)


def _cmd_calibrate(args):
    from .build.calibrate import run
    rep = run(dates=args.dates, n_test=args.n_test, project_id=args.project_id,
              iters=args.iters, lr=args.lr, window_days=args.window_days)
    print(json.dumps({"baseline_test_csi": rep["baseline"]["mean_csi_test"],
                      "calibrated_test_csi": rep["calibrated"]["mean_csi_test"],
                      "multipliers": rep["multipliers"]}, indent=2, default=float))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="varuna", description="FloodTwin Patna")
    p.add_argument("--work", help="override work/artifact directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the city model (GEE; GPU for the twin)")
    b.add_argument("--steps", nargs="+",
                   default=["sinks", "recharge", "validate", "twin"],
                   choices=["sinks", "recharge", "validate", "twin"])
    b.add_argument("--project-id", default=None, help="Earth Engine project id")
    b.add_argument("--event-date", default=None, help="Sentinel-1 pass date for validation, YYYY-MM-DD")
    b.add_argument("--n-samples", type=int, default=CFG.n_samples)
    b.add_argument("--area", default=None,
                   help="build a registered area by id (see varuna.areas); sub-crops need no Earth Engine")
    b.add_argument("--epochs", type=int, default=40, help="twin emulator training epochs")
    b.set_defaults(func=_cmd_build)

    a = sub.add_parser("alert", help="run the daily ward alert (CPU)")
    a.add_argument("--rain-mm", type=float, default=None, help="override rainfall (mm); default: live forecast")
    a.add_argument("--map", action="store_true", help="also render alert_map.html")
    a.set_defaults(func=_cmd_alert)

    c = sub.add_parser("chat", help="talk to the flood model (local LLM, GPU)")
    c.add_argument("--four-bit", action="store_true", help="load the LLM in 4-bit (16 GB GPUs)")
    c.set_defaults(func=_cmd_chat)

    cal = sub.add_parser("calibrate", help="SAR-calibrate the twin's Manning/infiltration (GEE; GPU)")
    cal.add_argument("--dates", nargs="+", required=True,
                     help="Sentinel-1 overpass dates with antecedent rain, YYYY-MM-DD (last n held out)")
    cal.add_argument("--n-test", type=int, default=1, help="dates to hold out for validation")
    cal.add_argument("--iters", type=int, default=40)
    cal.add_argument("--lr", type=float, default=0.05)
    cal.add_argument("--window-days", type=int, default=2, help="antecedent-rain accumulation window")
    cal.add_argument("--project-id", default=None, help="Earth Engine project id")
    cal.set_defaults(func=_cmd_calibrate)

    args = p.parse_args(argv)
    if args.work:
        CFG.work = args.work
    args.func(args)


if __name__ == "__main__":
    main()
