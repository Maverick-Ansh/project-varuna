"""Command-line entry point: python -m varuna {build|alert|chat}."""
from __future__ import annotations

import argparse
import json
import logging

from .config import CFG


def _cmd_build(args):
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
    b.set_defaults(func=_cmd_build)

    a = sub.add_parser("alert", help="run the daily ward alert (CPU)")
    a.add_argument("--rain-mm", type=float, default=None, help="override rainfall (mm); default: live forecast")
    a.add_argument("--map", action="store_true", help="also render alert_map.html")
    a.set_defaults(func=_cmd_alert)

    c = sub.add_parser("chat", help="talk to the flood model (local LLM, GPU)")
    c.add_argument("--four-bit", action="store_true", help="load the LLM in 4-bit (16 GB GPUs)")
    c.set_defaults(func=_cmd_chat)

    args = p.parse_args(argv)
    if args.work:
        CFG.work = args.work
    args.func(args)


if __name__ == "__main__":
    main()
