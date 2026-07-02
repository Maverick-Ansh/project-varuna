"""Adaptive distributed storage — size flood-detention to the local geography.

Where `serve/canals.py` *conveys* water downhill and `build/twin.candidate_sites` digs a handful of
big pits, this module answers a different planner question: *"how many distributed storage units do we
need, and where, to cut the flood by X%?"*

The key idea (and why it is non-linear): a storage site is **sized to its own local depression**, not a
fixed unit. Water pools at the terrain's local minima; the depth it pools to *is* that minimum's natural
capacity. We rank flooded built cells by their gradient-pooled depth (deepest sink first — the true
local minima), give each site a detention volume equal to the water it holds, and **re-simulate** the
storm so the reported cut is measured, not assumed. Deep sinks absorb a lot per site; shallow ones add
little — so the flood-cut-vs-#sites curve bends, and the marginal value of each extra site falls off.

`plan_storage` returns the dose curve + the #sites needed for target reductions (with an equivalent
count of standard modular units for procurement intuition). Pure-ish: needs a built Domain (GPU optional).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from ..config import CFG
from ..io import save_json

log = logging.getLogger("varuna.serve.containers")

_DEFAULT_COUNTS = (50, 100, 200, 500, 1000, 2000, 4000)
TAU = 0.15


def _ranked_minima(dom, rain_mm):
    """Flooded built cells ranked deepest-first by pooled depth = the dynamic local minima."""
    with torch.no_grad():
        hmax0 = dom.simulate(dom.z0, rain_mm=rain_mm)
    f = torch.relu(hmax0 - TAU) * dom.built
    fnp = f.cpu().numpy()
    ys, xs = np.where(fnp > 0.0)
    order = np.argsort(-fnp[ys, xs])
    return ys[order], xs[order], fnp[ys[order], xs[order]], f


def _cut_for_sites(dom, ys, xs, depths, f, K, rain_mm):
    """Place K geography-sized storage sites (deepen each by its pooled depth), re-simulate, measure cut."""
    cell = dom.dx * dom.dx
    rr, cc = ys[:K], xs[:K]
    D = torch.zeros_like(dom.z0)
    D[rr, cc] = torch.as_tensor(depths[:K], dtype=torch.float32, device=dom.device)
    smask = torch.zeros_like(dom.z0)
    smask[rr, cc] = 1.0
    streets = dom.built * (1.0 - smask)
    with torch.no_grad():
        h1 = dom.simulate(dom.z0 - D, rain_mm=rain_mm)
    f1 = torch.relu(h1 - TAU) * dom.built
    base = float((f * streets).sum()) * cell
    new = float((f1 * streets).sum()) * cell
    return dict(sites=int(K),
                reduction_pct=round(100 * (1 - new / max(base, 1)), 1),
                storage_m3=round(float(D.sum()) * cell),
                median_site_m3=round(float(np.median(depths[:K])) * cell))


def plan_storage(rain_mm=None, work=None, device=None, site_counts=None,
                 targets=(30, 50, 70), unit_m3=50.0, dom=None):
    """Sweep distributed-storage count -> flood cut, and size the #sites for target reductions."""
    from ..build.twin import build_domain

    work = work or CFG.work
    rain_mm = CFG.design_rain_mm if rain_mm is None else float(rain_mm)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dom = dom or build_domain(work, device=device)
    cell = dom.dx * dom.dx

    ys, xs, depths, f = _ranked_minima(dom, rain_mm)
    n_flood = len(ys)
    counts = sorted(set([k for k in (site_counts or _DEFAULT_COUNTS) if k < n_flood] + [n_flood]))
    V0 = float(f.sum()) * cell

    curve = [_cut_for_sites(dom, ys, xs, depths, f, K, rain_mm) for K in counts]
    # Carving thousands of pits can destabilise the explicit solver (depth blows up and the
    # "cut" goes absurdly negative). Flag those points and keep them out of target sizing.
    for c in curve:
        c["unstable"] = bool(not np.isfinite(c["reduction_pct"]) or c["reduction_pct"] < -50)
        log.info("storage %5d sites -> cut %5.1f%% | %d m3%s", c["sites"], c["reduction_pct"],
                 c["storage_m3"], "  [UNSTABLE — excluded from targets]" if c["unstable"] else "")
    stable = [c for c in curve if not c["unstable"]]

    S = np.array([c["sites"] for c in stable], dtype="float64")
    R = np.maximum.accumulate(np.array([c["reduction_pct"] for c in stable], dtype="float64"))
    SV = np.array([c["storage_m3"] for c in stable], dtype="float64")
    tgt = {}
    for t in targets:
        if len(S) == 0 or t > R.max():
            tgt[f"{int(t)}%"] = dict(sites=None, storage_m3=None, equiv_units=None,
                                     note="target beyond the stable part of the curve")
            continue
        n = int(np.interp(t, R, S))
        sv = float(np.interp(n, S, SV))
        tgt[f"{int(t)}%"] = dict(sites=n, storage_m3=round(sv), equiv_units=round(sv / unit_m3))

    report = dict(rain_mm=rain_mm, total_flood_m3=round(V0), max_sites=int(n_flood),
                  unit_m3=unit_m3, curve=curve, targets=tgt,
                  note="Each storage site sized to its local pooled depth (dynamic, non-linear by "
                       "geography); sites ranked by gradient-pooled depth = local minima, deepest first.")
    save_json(f"{work}/storage_sizing.json", report)
    return report


def plot_storage_dose(report, out="storage_dose.png"):
    """Flood-cut vs #storage-sites curve with target-reduction markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = [c["sites"] for c in report["curve"]]
    R = [c["reduction_pct"] for c in report["curve"]]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(S, R, "o-", color="#2a7f62", lw=2)
    for (lbl, v), col in zip(report["targets"].items(), ["#888", "#c0504d", "#3a6ea5"]):
        n = v["sites"]
        ax.axhline(float(lbl[:-1]), ls="--", color=col, alpha=.5)
        ax.axvline(n, ls="--", color=col, alpha=.5)
        ax.annotate(f"{lbl} cut\n{n} sites", (n, float(lbl[:-1])),
                    textcoords="offset points", xytext=(8, -28), color=col, fontsize=9)
    ax.set_xlabel("number of distributed storage sites (geography-sized)")
    ax.set_ylabel("flood-volume cut on built land (%)")
    ax.set_title(f"Patna adaptive storage — flood cut vs # sites ({report['rain_mm']:.0f} mm storm)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
