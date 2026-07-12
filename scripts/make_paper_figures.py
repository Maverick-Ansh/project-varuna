"""Regenerate the paper's map-overlay figures from committed artifacts (no GEE, no network).

    python scripts/make_paper_figures.py --spiderweb mumbai_south --containers mumbai_west

Reads exposure.json (road polylines), canal_plan.json (drain network) and storage_sizing.json
(ranked container sites) from each area's bundle and writes paper/figures/*.png. Colors match
the dashboard palette so paper and product tell the same story.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "paper", "figures")

DRAIN = "#7b2fbe"
INLET = "#d97706"
OUTFALL = {"lowland": "#0d9488", "pit": "#ec4899", "boundary": "#64748b", "river": "#3a6ea5"}
CONTAINER = "#0ea5e9"
ROAD = "#c9ced6"


def _load(work, name):
    with open(os.path.join(work, name), encoding="utf-8") as f:
        return json.load(f)


def _roads(ax, work):
    exp = _load(work, "exposure.json")
    for key in ("dry_lines", "flooded_lines"):
        for ln in exp["roads"].get(key, []):
            ax.plot([p[1] for p in ln], [p[0] for p in ln], color=ROAD, lw=0.35, zorder=1)


def _frame(ax, title):
    ax.set_title(title, fontsize=9, wrap=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _clip(ax, xs, ys, pad=0.12):
    """Zoom to the content (padded bbox) so harbour bridges don't stretch the frame."""
    if not xs:
        return
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = max(x1 - x0, 1e-4), max(y1 - y0, 1e-4)
    ax.set_xlim(x0 - pad * dx, x1 + pad * dx)
    ax.set_ylim(y0 - pad * dy, y1 + pad * dy)


def spiderweb(area, work):
    cp = _load(work, "canal_plan.json")
    net = cp["network"]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _roads(ax, work)
    for e in net["edges"]:
        ln = e["path_latlon"]
        ax.plot([p[1] for p in ln], [p[0] for p in ln], color=DRAIN,
                lw=0.6 + 2.5 * float(e.get("weight", 0.3)), alpha=0.9, zorder=3)
    for i in net["inlets"]:
        ax.plot(i["latlon"][1], i["latlon"][0], "o", ms=3.5, color=INLET, mec="#92400e",
                mew=0.5, zorder=4)
    for o in net["outfall_points"]:
        ax.plot(o["latlon"][1], o["latlon"][0], "s", ms=6,
                color=OUTFALL.get(o.get("kind", "pit"), "#ec4899"), mec="white", mew=0.8, zorder=5)
    xs = [p[1] for e in net["edges"] for p in e["path_latlon"]]
    ys = [p[0] for e in net["edges"] for p in e["path_latlon"]]
    _clip(ax, xs, ys, pad=0.35)
    _frame(ax, f"{area}: street-routed storm-drain network\n"
               f"measured cut {cp['reduction_pct']:.1f}% @ {cp['rain_mm']:.0f} mm "
               f"({net['n_inlets']} inlets, {net['total_length_m']/1000:.1f} km)")
    out = os.path.join(OUT, f"spiderweb_{area}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def containers(area, work):
    sz = _load(work, "storage_sizing.json")
    sites = [s for s in sz.get("sites", []) if s.get("latlon")]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _roads(ax, work)
    if sites:
        xs = [s["latlon"][1] for s in sites]
        ys = [s["latlon"][0] for s in sites]
        ss = [max(2.0, min(60.0, s["site_m3"] / 120.0)) for s in sites]
        ax.scatter(xs, ys, s=ss, c=CONTAINER, edgecolors="#0369a1", linewidths=0.3,
                   alpha=0.8, zorder=3)
    if sites:
        _clip(ax, xs, ys, pad=0.15)
    ph = next((p for p in sz.get("phases", []) if p.get("reachable")), None)
    p1 = (f"\nphase 1: {ph['add_sites']} sites → {ph['target_cut_pct']}% cut, "
          f"INR {ph['add_cost_crore_inr']:.0f} cr" if ph else "")
    _frame(ax, f"{area}: {len(sites)} buildable container sites "
               f"(marker ∝ volume; no buildings, no water){p1}")
    out = os.path.join(OUT, f"containers_{area}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spiderweb", nargs="*", default=[])
    ap.add_argument("--containers", nargs="*", default=[])
    args = ap.parse_args()
    from varuna.areas import get_area
    os.makedirs(OUT, exist_ok=True)
    for a in args.spiderweb:
        spiderweb(a, get_area(a).work_dir())
    for a in args.containers:
        containers(a, get_area(a).work_dir())


if __name__ == "__main__":
    main()
