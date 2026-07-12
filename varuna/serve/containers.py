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


def _placeable(dom, work):
    """0/1 mask of cells where a container CAN physically go.

    Excludes building footprints (urban_grid.npz, rasterized OSM — you don't dig a detention
    tank under an occupied house) and permanent water (WorldCover 80/90 — the harbour is not a
    site). Streets and other open built land stay in: under-road detention is standard practice
    (BMC's Hindmata tanks). Missing layers degrade gracefully to all-permissive, so synthetic
    test domains keep the legacy behavior."""
    m = torch.ones_like(dom.z0)
    wc = getattr(dom, "wc", None)
    if wc is not None:
        w = torch.as_tensor(np.asarray(wc), device=dom.z0.device)
        m = m * ((w != 80) & (w != 90)).float()
    try:
        g = np.load(f"{work}/urban_grid.npz")
        b = torch.as_tensor(g["buildings"], dtype=torch.float32, device=dom.z0.device)
        if b.shape == dom.z0.shape:
            m = m * (1.0 - (b > 0).float())
    except Exception:  # noqa: BLE001 - no urban grid cached (synthetic bundle)
        pass
    return m


def _ranked_minima(dom, rain_mm, placeable=None):
    """Flooded built cells ranked deepest-first by pooled depth = the dynamic local minima.

    `placeable` only filters the CANDIDATE cells; the returned flood field `f` stays unmasked
    so the measured baseline (and therefore the reported cut) is unchanged by siting rules."""
    with torch.no_grad():
        hmax0 = dom.simulate(dom.z0, rain_mm=rain_mm)
    f = torch.relu(hmax0 - TAU) * dom.built
    fs = f * placeable if placeable is not None else f
    fnp = fs.cpu().numpy()
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
                 targets=(30, 50, 70), unit_m3=50.0, dom=None,
                 phase_targets=(10, 20, 40), site_list_max=1500, save=True):
    """Sweep distributed-storage count -> flood cut, size the #sites for target reductions,
    emit the explicit buildable site list (lat/lon + per-site volume) and a phased,
    indicatively-costed implementation plan."""
    from ..build.twin import build_domain

    work = work or CFG.work
    rain_mm = CFG.design_rain_mm if rain_mm is None else float(rain_mm)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dom = dom or build_domain(work, device=device)
    cell = dom.dx * dom.dx

    ys, xs, depths, f = _ranked_minima(dom, rain_mm, placeable=_placeable(dom, work))
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

    # ---- explicit site list: WHERE the containers actually go (buildable cells only)
    roads = None
    try:
        roads = np.load(f"{work}/urban_grid.npz")["roads"]
    except Exception:  # noqa: BLE001
        pass
    to_ll = None
    try:
        import rasterio
        with rasterio.open(f"{work}/dem.tif") as src:
            T = src.transform
        def to_ll(r, c):
            lon, lat = T * (dom.col0 + c * 2, dom.row0 + r * 2)
            return [round(lat, 5), round(lon, 5)]
    except Exception:  # noqa: BLE001 - synthetic bundle without rasters
        pass
    sites_out = []
    for i in range(int(min(site_list_max, n_flood))):
        r, c = int(ys[i]), int(xs[i])
        s = dict(rank=i + 1, row=r, col=c, site_m3=round(float(depths[i]) * cell))
        if to_ll is not None:
            s["latlon"] = to_ll(r, c)
        if roads is not None:
            s["on_road"] = bool(roads[r, c] > 0)
        sites_out.append(s)

    # ---- phased implementation plan with indicative costs (RCC detention rate)
    from .costbenefit import DEFAULT_COSTS
    rate = float(DEFAULT_COSTS["storage_inr_per_m3"])
    phases, prev_n, prev_v = [], 0, 0.0
    for t in phase_targets:
        if len(S) == 0 or t > R.max():
            phases.append(dict(phase=len(phases) + 1, target_cut_pct=int(t), reachable=False,
                               note="beyond the stable part of the dose curve"))
            continue
        n = int(np.interp(t, R, S))
        sv = float(np.interp(n, S, SV))
        dv = max(0.0, sv - prev_v)
        phases.append(dict(phase=len(phases) + 1, target_cut_pct=int(t), reachable=True,
                           cumulative_sites=n, cumulative_storage_m3=round(sv),
                           add_sites=max(0, n - prev_n), add_storage_m3=round(dv),
                           add_units=round(dv / unit_m3),
                           add_cost_inr=round(dv * rate),
                           add_cost_crore_inr=round(dv * rate / 1e7, 2)))
        prev_n, prev_v = n, sv

    report = dict(rain_mm=rain_mm, total_flood_m3=round(V0), max_sites=int(n_flood),
                  unit_m3=unit_m3, curve=curve, targets=tgt,
                  sites=sites_out, phases=phases,
                  storage_inr_per_m3=rate,
                  note="Each storage site sized to its local pooled depth (dynamic, non-linear by "
                       "geography); sites ranked by gradient-pooled depth = local minima, deepest "
                       "first. Sites are placed only where buildable: building footprints and "
                       "permanent water are excluded; under-street detention is allowed. Phase "
                       "costs use the indicative RCC detention rate — relative ROI, not a bid.")
    if save:                        # costbenefit probes with its own targets — must not clobber
        save_json(f"{work}/storage_sizing.json", report)
    return report


def plot_storage_dose(report, out="storage_dose.png"):
    """Flood-cut vs #storage-sites curve with target-reduction markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    good = [c for c in report["curve"] if not c.get("unstable")]
    S = [c["sites"] for c in good]
    R = [c["reduction_pct"] for c in good]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(S, R, "o-", color="#2a7f62", lw=2)
    for (lbl, v), col in zip(report["targets"].items(), ["#888", "#c0504d", "#3a6ea5"]):
        n = v["sites"]
        if n is None:                 # target beyond the stable part of the curve
            continue
        ax.axhline(float(lbl[:-1]), ls="--", color=col, alpha=.5)
        ax.axvline(n, ls="--", color=col, alpha=.5)
        ax.annotate(f"{lbl} cut\n{n} sites", (n, float(lbl[:-1])),
                    textcoords="offset points", xytext=(8, -28), color=col, fontsize=9)
    ax.set_xlabel("number of distributed storage sites (geography-sized)")
    ax.set_ylabel("flood-volume cut on built land (%)")
    ax.set_title(f"Adaptive storage — flood cut vs # sites ({report['rain_mm']:.0f} mm storm)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
