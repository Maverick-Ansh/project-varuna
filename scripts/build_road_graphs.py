"""Fetch OSM road graphs + regenerate street-spiderweb canal plans for the built areas.

Runs anywhere with internet + torch + rasterio (no osmnx — the fetch is a plain Overpass
query). Typical full refresh:

    python scripts/build_road_graphs.py --fetch --plan

The graphs land in each bundle as road_graph.json.gz (committed, so the deployed Space and
offline tests never fetch); the plans rewrite canal_plan.json + the map_canal_plan viz arrays.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from varuna.areas import list_areas, is_built  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="fetch + cache road_graph.json.gz per area")
    ap.add_argument("--plan", action="store_true", help="run plan_canals + render map per area")
    ap.add_argument("--areas", nargs="*", default=None, help="area ids (default: all built)")
    ap.add_argument("--rain", type=float, default=100.0)
    ap.add_argument("--n-inlets", type=int, default=40)
    args = ap.parse_args()

    areas = [a for a in list_areas() if is_built(a.id)
             and (args.areas is None or a.id in args.areas)]
    if not areas:
        sys.exit("no built areas matched")
    print("areas:", ", ".join(a.id for a in areas))

    if args.fetch:
        from varuna.serve.roadnet import build_road_graph
        for a in areas:
            t0 = time.time()
            g = build_road_graph(work=a.work_dir())
            print(f"[{a.id}] road graph: {len(g['nodes'])} nodes, {len(g['ways'])} ways "
                  f"({time.time() - t0:.1f}s)")
            time.sleep(2)                                    # be polite to Overpass

    if args.plan:
        from varuna.serve.canals import plan_canals
        from varuna.viz import map_canal_plan
        rows = []
        for a in areas:
            work = a.work_dir()
            old = None
            try:
                with open(os.path.join(work, "canal_plan.json"), encoding="utf-8") as f:
                    old = json.load(f)["reduction_pct"]
            except Exception:  # noqa: BLE001
                pass
            t0 = time.time()
            res = plan_canals(rain_mm=args.rain, n_inlets=args.n_inlets, work=work)
            net = res.get("network") or {}
            try:
                map_canal_plan(work=work)
            except Exception as e:  # noqa: BLE001
                print(f"[{a.id}] viz failed: {e}")
            rows.append((a.id, old, res["reduction_pct"], net.get("n_inlets"),
                         net.get("total_length_m"), net.get("excavation_m3"),
                         round(time.time() - t0, 1)))
            print(f"[{a.id}] cut {old} -> {res['reduction_pct']}% "
                  f"({net.get('n_inlets')} inlets, {net.get('total_length_m')} m) "
                  f"in {rows[-1][-1]}s")
        print(f"\n{'area':<12}{'old %':>7}{'new %':>7}{'inlets':>8}{'length m':>10}{'excav m3':>10}{'sec':>6}")
        for r in rows:
            print(f"{r[0]:<12}{str(r[1]):>7}{r[2]:>7}{str(r[3]):>8}{str(r[4]):>10}{str(r[5]):>10}{r[6]:>6}")


if __name__ == "__main__":
    main()
