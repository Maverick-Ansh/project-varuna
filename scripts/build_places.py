"""Fetch OSM naming anchors (places / named roads / stations / wards) for the built areas.

The danger-zone pins name themselves from these. Runs anywhere with internet + torch +
rasterio — the fetch is a plain Overpass query, no osmnx. Typical full refresh:

    python scripts/build_places.py                    # all built areas, skipping ones already done
    python scripts/build_places.py --areas patna --force

Anchors land in each bundle as places.json.gz (committed, so the deployed Space and the offline
tests never fetch). Overpass is a shared free service — the delay between areas is deliberate,
and the public mirrors 504 often enough that --retries earns its keep.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from varuna.areas import list_areas, is_built  # noqa: E402
from varuna.serve.zones import CACHE_FILE, build_places, load_places  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--areas", nargs="*", default=None, help="area ids (default: all built)")
    ap.add_argument("--force", action="store_true", help="refetch areas that already have a cache")
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between areas")
    ap.add_argument("--retries", type=int, default=3, help="attempts per area")
    args = ap.parse_args()

    areas = [a for a in list_areas() if is_built(a.id)
             and (args.areas is None or a.id in args.areas)]
    if not areas:
        sys.exit("no built areas matched")

    rows, failed = [], []
    for a in areas:
        work = a.work_dir()
        if os.path.exists(os.path.join(work, CACHE_FILE)) and not args.force:
            n = len(load_places(work))
            print(f"[{a.id}] cached ({n} anchors) — use --force to refetch")
            rows.append((a.id, n, "cached"))
            continue
        for attempt in range(1, args.retries + 1):
            try:
                t0 = time.time()
                payload = build_places(work=work)
                kinds = Counter(p["kind"] for p in payload["places"])
                roads = len({p["name"] for p in payload["places"] if p["kind"] == "road"})
                print(f"[{a.id}] {len(payload['places'])} anchors, {roads} named roads "
                      f"({time.time() - t0:.1f}s) {dict(kinds)}")
                rows.append((a.id, len(payload["places"]), f"{roads} roads"))
                break
            except Exception as e:  # noqa: BLE001
                print(f"[{a.id}] attempt {attempt}/{args.retries} failed: {e}")
                if attempt == args.retries:
                    failed.append(a.id)
                else:
                    time.sleep(args.sleep * attempt * 2)       # back off; mirrors 504 under load
        time.sleep(args.sleep)

    print(f"\n{'area':<18}{'anchors':>9}  note")
    for aid, n, note in rows:
        print(f"{aid:<18}{n:>9}  {note}")
    if failed:
        print("\nFAILED (rerun to retry):", " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
