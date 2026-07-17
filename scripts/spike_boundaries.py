"""Phase 1 §1.1 boundary spike: can we honestly claim taluk-level precision for Karnataka?

CGWB categorization is per taluk (~176 in the 2011 frame, more today), but FAO/GAUL/2015/level2
— the only drop-in EE admin layer — is DISTRICT only. This script measures, for one or more
candidate polygon assets, the thing that actually decides the question: how many CGWB rows
join to a polygon by name. Run on Colab with EE authed:

    python scripts/spike_boundaries.py --project <gcp-project>                    # GAUL baseline
    python scripts/spike_boundaries.py --project <p> --asset users/<you>/karnataka_taluks

Candidate taluk sources to upload as an EE asset (in preference order, see plan §1.1):
  - datameet/maps india taluks (2011 census frame; Survey-of-India derived GeoJSON)
  - Karnataka-GIS (KGIS) published taluk boundaries
Whichever is chosen, `run_state_screen.py --admin-asset ... --admin-level taluk` records the
level in the output JSON — a reader must never assume taluk precision from a district product.

Verdict rule of thumb printed at the end: >=90% join coverage AND >=150 units -> taluk is
honest; anything less -> stay at district level and say so.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from varuna.build import state_screen  # noqa: E402
from varuna.ee_auth import init_ee  # noqa: E402


def probe(fc, name_prop, cgwb):
    names = fc.aggregate_array(name_prop).getInfo()
    joined, report = state_screen.join_units(cgwb, names)
    print(f"  units in asset: {len(names)}")
    print(f"  sample names: {sorted(names)[:8]} ...")
    print(f"  CGWB rows joined: {len(joined)}/{len(cgwb)}  (coverage {report['coverage']:.0%})")
    for kind in ("alias", "fuzzy", "merged"):
        if report[kind]:
            print(f"  {kind}: {json.dumps(report[kind], ensure_ascii=False)}")
    if report["unmatched_cgwb"]:
        print(f"  UNMATCHED CGWB: {report['unmatched_cgwb']}")
    if report["unmatched_polygons"]:
        print(f"  polygons with no CGWB row ({len(report['unmatched_polygons'])}): "
              f"{report['unmatched_polygons'][:12]}")
    return len(names), report["coverage"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=None)
    ap.add_argument("--cgwb", default="data/cgwb_karnataka_2024_district.csv",
                    help="CGWB table to join (district CSV today; taluk CSV once parsed)")
    ap.add_argument("--asset", default=None, help="candidate taluk polygon EE asset id")
    ap.add_argument("--name-prop", default=None,
                    help="name property on --asset (script tries common ones if omitted)")
    args = ap.parse_args()

    import ee
    init_ee(args.project)
    cgwb = state_screen.load_cgwb(args.cgwb)
    print(f"CGWB rows: {len(cgwb)} (from {args.cgwb})\n")

    print("== baseline: FAO/GAUL/2015/level2 (districts) ==")
    fc, name_prop, _ = state_screen.karnataka_units()
    n_gaul, cov_gaul = probe(fc, name_prop, cgwb)

    if args.asset:
        print(f"\n== candidate: {args.asset} ==")
        fc = ee.FeatureCollection(args.asset)
        name_prop = args.name_prop
        if name_prop is None:
            props = fc.first().propertyNames().getInfo()
            guesses = [p for p in props if "name" in p.lower() or "taluk" in p.lower()]
            print(f"  properties: {props}\n  trying name props: {guesses}")
            name_prop = guesses[0] if guesses else props[0]
        n, cov = probe(fc, name_prop, cgwb)
        verdict = "TALUK VIABLE" if (cov >= 0.90 and n >= 150) else "STAY AT DISTRICT"
        print(f"\nVERDICT: {verdict}  ({n} units, {cov:.0%} join coverage; "
              "note: coverage vs a district CGWB table only proves names parse — rerun with "
              "the taluk CGWB table before claiming taluk precision)")
    else:
        print(f"\nVERDICT: no taluk asset supplied — district level "
              f"({n_gaul} units, {cov_gaul:.0%} coverage) is the honest floor. "
              "Upload a taluk asset and rerun with --asset to upgrade.")


if __name__ == "__main__":
    main()
