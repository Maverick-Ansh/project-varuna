"""Fetch Sentinel-1 water masks for an area, so it can become a domain in csi_net.

The learned model's only real claim is transfer -- it scores 0.094 on a city it has never seen,
where a climatology cannot be computed at all -- and that claim rests on three domains, because
only three of the sixteen built bundles have `observed_water_<date>.tif`. More cities is the
highest-value measurement left in this project, and this is the script that gets them.

    python scripts/fetch_sar_masks.py --areas bengaluru --n 10 --dry-run
    python scripts/fetch_sar_masks.py --areas bengaluru,mumbai_south --n 10

Needs Earth Engine, which needs a one-time interactive login:

    earthengine authenticate
    export VARUNA_PROJECT_ID=<your-google-cloud-project>

`--dry-run` needs neither: it lists the Sentinel-1 passes it *would* fetch, ranked, using only
the rainfall archive, so the date selection can be reviewed before spending GEE quota.

Date selection matters more than it looks. The existing three domains were hand-picked, and one
of them (patna/2024-07-07) has zero observed wet cells -- no ground truth at all -- which sat in
every published mean as a forced 0.0 until it was found. Picking by antecedent rainfall biases
towards dates where something actually happened, and `--min-wet` refuses to write a mask that is
empty enough to repeat that mistake.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("fetch_sar")

MONSOON = (6, 10)          # inclusive month range: the SOUTHWEST monsoon, most of India
MIN_GAP_DAYS = 10          # two passes three days apart are nearly the same scene

# India has two monsoons and they do not overlap. The southwest monsoon (Jun-Oct) soaks Patna,
# Mumbai and interior Karnataka; the Coromandel coast gets most of its rain from the NORTHEAST
# monsoon (Oct-Dec) instead -- the 2015 Chennai floods were in November and December. Ranking
# candidates by antecedent rain inside a Jun-Oct window would therefore sample Chennai's dry
# half of the year and make it look like a domain where nothing happens, which is exactly the
# wrong answer to get from a coastal-transfer experiment. Areas whose season differs from the
# default are listed here; the rainfall ranking still chooses which dates to keep, this only
# decides which ones it is allowed to see.
SEASON = {
    "chennai": (6, 12),    # both monsoons; the ranking picks the wet ones, and they are Oct-Dec
}


def season_for(area_id):
    return SEASON.get(area_id, MONSOON)


def antecedent_rain(lat, lon, date, window=3):
    """Total mm over the `window` days ending on `date`, from the pinned archive model."""
    from varuna.serve.weather import historical_rain_mm
    d1 = _dt.date.fromisoformat(date)
    d0 = d1 - _dt.timedelta(days=window)
    try:
        return float(historical_rain_mm(lat, lon, d0.isoformat(), d1.isoformat()))
    except Exception as e:  # noqa: BLE001 - one bad date must not kill the run
        log.warning("rain lookup failed for %s: %s", date, e)
        return -1.0


def candidate_dates(area, years, reg=None):
    """Sentinel-1 monsoon passes over the area's AOI, deduplicated to one per MIN_GAP_DAYS."""
    from varuna.build.validate import list_passes
    lo, hi = season_for(area)
    seen = []
    for y in years:
        try:
            passes = list_passes(year=y, reg=reg)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: no passes for %d (%s)", area, y, e)
            continue
        for d in sorted(set(passes)):
            m = int(d[5:7])
            if not (lo <= m <= hi):
                continue
            dt = _dt.date.fromisoformat(d)
            if seen and (dt - _dt.date.fromisoformat(seen[-1])).days < MIN_GAP_DAYS:
                continue
            seen.append(d)
    return seen


def rank(area_id, dates, center):
    """Rank candidate dates by 3-day antecedent rain, wettest first."""
    lat, lon = center
    scored = [(d, antecedent_rain(lat, lon, d)) for d in dates]
    scored.sort(key=lambda t: -t[1])
    return scored


def _domain_crop(work):
    """Return a function cropping a full-AOI raster to the window csi_net actually trains on.

    This guard exists because of patna/2024-07-07, and measuring the *downloaded* raster would
    not have caught it: that scene is 0.0065 wet across the full AOI and exactly 0.0000 wet
    inside `build_domain`'s window, which is the only part `csi_net.build_stack` ever reads. A
    wet-fraction test on the download therefore passes the very mask it was written to reject.
    Fall back to the identity crop if the bundle has no domain yet, so a fetch into a fresh
    area still works -- it just reverts to the weaker full-AOI test, and says so.
    """
    try:
        from varuna.build.twin import build_domain
        dom = build_domain(work)
        n, r0, c0 = dom.N * 2, dom.row0, dom.col0
    except Exception as e:  # noqa: BLE001 - a missing bundle is not a reason to refuse to fetch
        log.warning("%s: no twin domain (%s); wet fraction measured on the full AOI, "
                    "which is the weaker test", work, type(e).__name__)
        return lambda a: a
    return lambda a: a[r0:r0 + n, c0:c0 + n]


def fetch(area_id, dates, artifacts, min_wet, dry_run):
    """Write observed_water_<date>.tif for each date; skip masks that are effectively empty."""
    from varuna.areas import get_area

    a = get_area(area_id)
    work = os.path.join(artifacts, area_id)
    written, skipped = [], []
    if dry_run:
        # A dry run must not import or initialise Earth Engine at all - the whole point is that
        # it is reviewable before anyone has authenticated.
        return [d for d in dates
                if not os.path.exists(os.path.join(work, f"observed_water_{d}.tif"))], []

    from varuna.build.validate import observed_water
    from varuna.ee_auth import region
    from varuna.io import read1

    os.makedirs(work, exist_ok=True)
    reg = region(a.aoi)
    crop = _domain_crop(work)
    for d in dates:
        path = os.path.join(work, f"observed_water_{d}.tif")
        if os.path.exists(path):
            log.info("%s %s: already present", area_id, d)
            continue
        observed_water(d, work=work, reg=reg)
        wet = float((crop(read1(path)[0]) > 0.5).mean())
        if wet < min_wet:
            # A mask with (almost) no water is not ground truth; it is a forced zero in every
            # method's mean. patna/2024-07-07 is exactly this, and it cost the project a
            # wrong twin baseline until someone noticed.
            os.remove(path)
            skipped.append((d, wet))
            log.warning("%s %s: wet fraction %.5f < %.5f — dropped, not ground truth",
                        area_id, d, wet, min_wet)
        else:
            written.append(d)
            log.info("%s %s: wet fraction %.5f", area_id, d, wet)
    return written, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", required=True, help="comma-separated area ids")
    ap.add_argument("--years", default="2023,2024,2025,2026")
    ap.add_argument("--n", type=int, default=10, help="masks to keep per area")
    ap.add_argument("--min-wet", type=float, default=1e-4,
                    help="reject a mask whose wet fraction is below this (no ground truth)")
    ap.add_argument("--artifacts", default=os.path.join(REPO_ROOT, "artifacts"))
    ap.add_argument("--dry-run", action="store_true",
                    help="rank and print candidate dates without touching Earth Engine downloads")
    args = ap.parse_args(argv)

    from varuna.areas import get_area
    years = [int(y) for y in args.years.split(",") if y]
    report = {}

    if not args.dry_run:
        from varuna.ee_auth import init_ee
        init_ee(os.environ.get("VARUNA_PROJECT_ID"))

    for area_id in [a for a in args.areas.split(",") if a]:
        a = get_area(area_id)
        print(f"\n=== {area_id}  ({a.name})  centre {a.center}")
        reg = None
        if not args.dry_run:
            from varuna.ee_auth import region
            reg = region(a.aoi)
            cands = candidate_dates(area_id, years, reg)
        else:
            # Without EE we cannot list real passes; Sentinel-1 revisits every 6 or 12 days, so
            # a 12-day grid over the monsoon is the right shape to review, clearly labelled.
            cands = []
            for y in years:
                d = _dt.date(y, MONSOON[0], 1)
                while d.month <= MONSOON[1]:
                    cands.append(d.isoformat())
                    d += _dt.timedelta(days=12)
            print("  [dry-run] no Earth Engine: showing a 12-day synthetic grid, "
                  "NOT real acquisition dates")
        scored = rank(area_id, cands, a.center)[:args.n]
        for d, mm in scored:
            print(f"    {d}   {mm:7.1f} mm antecedent (3 d)")
        picked = [d for d, _ in scored]
        written, skipped = fetch(area_id, picked, args.artifacts, args.min_wet, args.dry_run)
        report[area_id] = dict(candidates=len(cands), picked=picked,
                               written=written, skipped=skipped)
        print(f"  {'would write' if args.dry_run else 'wrote'} {len(written)} mask(s)"
              + (f", dropped {len(skipped)} empty" if skipped else ""))

    out = os.path.join(args.artifacts, "sar_fetch_report.json")
    if not args.dry_run:
        json.dump(report, open(out, "w"), indent=2)
        print("\nwrote", out)
    print("\nNext: python -m csi_net.build_stack --areas <all areas, old and new> "
          "--out csi_net/data/varuna_stack.npz")
    print("Rebuild EVERY area in one go - see build_stack's docstring for why mixing is unsafe.")
    return report


if __name__ == "__main__":
    main()
