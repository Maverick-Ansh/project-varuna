"""Physics-change audit: what does the V3 aquifer do to the headline numbers?

Adding soil storage capacity + the Ksat cap makes the twin LESS able to absorb water, so
predicted flooding must not decrease. The plan's honesty note requires reporting that change,
not discovering it later in a reviewer's question. This script runs the same storms through
the legacy physics (rate-only infiltration, no Ksat cap) and the V3 physics (metered) on real
bundles and writes the deltas to artifacts/aquifer_impact.json.

    python scripts/compare_aquifer.py --areas patna bengaluru --rains 60 100 150
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from varuna.areas import area_work  # noqa: E402
from varuna.build.twin import F_TABLE, Domain, _soil_infil_factor, _soil_pct, build_domain  # noqa: E402


def legacy_domain(dom, work):
    """Reconstruct the pre-V3 physics for the same crop: F_TABLE x clay factor, no Ksat cap,
    no capacity. Uses the V3 domain's own crop window so the comparison is cell-for-cell."""
    N = dom.N
    clay_pct = _soil_pct(work, "clay", dom.row0, dom.col0, N * 2, N)
    infil_mm = np.vectorize(lambda v: F_TABLE.get(int(v), 5.0))(dom.wc) * _soil_infil_factor(clay_pct)
    return Domain(dom.z0.cpu().numpy(), dom.mann.cpu().numpy(),
                  infil_mm / 1000.0 / 3600.0, dom.built.cpu().numpy(),
                  dx=dom.dx, device=str(dom.device), capacity=None)


def audit(area, rains, depth_thresh=0.10):
    work = area_work(area)
    dom_v3 = build_domain(work)
    dom_old = legacy_domain(dom_v3, work)
    cell = dom_v3.dx * dom_v3.dx
    cap_m3 = float(dom_v3.capacity.sum()) * cell if dom_v3.capacity is not None else None
    rows = []
    for r in rains:
        o = dom_old.rollout(dom_old.z0, r, track_infil=True)
        n = dom_v3.rollout(dom_v3.z0, r, track_infil=True)
        row = dict(
            rain_mm=r,
            infiltrated_m3_old=round(float(o["infil_grid"].sum()) * cell),
            infiltrated_m3_new=round(float(n["infil_grid"].sum()) * cell),
            ponded_final_m3_old=round(float(o["volume"][-1])),
            ponded_final_m3_new=round(float(n["volume"][-1])),
            flooded_km2_old=round(float((o["hmax"] > depth_thresh).sum()) * cell / 1e6, 2),
            flooded_km2_new=round(float((n["hmax"] > depth_thresh).sum()) * cell / 1e6, 2),
        )
        row["ponded_change_pct"] = round(
            100.0 * (row["ponded_final_m3_new"] - row["ponded_final_m3_old"])
            / max(row["ponded_final_m3_old"], 1), 1)
        if cap_m3:
            row["soil_filled_pct"] = round(100.0 * row["infiltrated_m3_new"] / cap_m3, 1)
        rows.append(row)
        print(f"{area} {r:>3} mm: ponded {row['ponded_final_m3_old']:>9,} -> "
              f"{row['ponded_final_m3_new']:>9,} m3 ({row['ponded_change_pct']:+.1f}%) | "
              f"flooded {row['flooded_km2_old']:.2f} -> {row['flooded_km2_new']:.2f} km2 | "
              f"soil filled {row.get('soil_filled_pct', 'n/a')}%")
    return dict(area=area, n_grid=dom_v3.N, dx=dom_v3.dx, depth_thresh_m=depth_thresh,
                soil_capacity_m3=cap_m3, entries=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--areas", nargs="+", default=["patna", "bengaluru"])
    ap.add_argument("--rains", nargs="+", type=float, default=[60.0, 100.0, 150.0])
    ap.add_argument("--out", default="artifacts/aquifer_impact.json")
    args = ap.parse_args()
    report = {
        "note": ("Legacy (rate-only infiltration) vs V3 (soil-capacity-metered, Ksat-capped) "
                 "on identical crops and storms. Predicted flooding INCREASES by design — the "
                 "old model let every pervious cell absorb at a constant rate forever. "
                 "Committed so the physics change is auditable, per the plan's honesty note."),
        "areas": [audit(a, args.rains) for a in args.areas],
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("->", args.out)


if __name__ == "__main__":
    main()
