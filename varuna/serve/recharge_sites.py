"""Metered recharge siting — put monsoon runoff into the aquifer and MEASURE the m³.

`plan_storage`'s sibling with the opposite land rule and a different metric. Storage detains
flood water and may sit under streets; a recharge structure must put water INTO the ground, so
it needs pervious land, away from foundations. And where `rank_recharge_sites` (nb02) multiplies
`volume × RSI` and calls it a score, this module places structures, re-simulates the storm and
reads the m³ that actually entered the soil from `rollout(track_infil=True)`'s `infil_grid` —
which, since the V3 aquifer (Phase 2), is metered by per-cell soil capacity. A full soil column
stops accepting water no matter how good the structure looks on paper. Measure, don't assert.

Ranking: candidates are scored `pooled_depth × local_suitability`, where suitability is
normalized Cosby Ksat (soil throughput) × smoothed pervious fraction (open ground, not a lone
gap between buildings). Groundwater depth deliberately does NOT enter the rank: the Phase-2
capacity already encodes "room in the aquifer" inside the *measurement*, and the Phase-0 gate
means most bundles have no real groundwater data to rank with anyway — a fake-well term here
would be exactly the bug V3 exists to remove.

Every payload ships `gw_status` (sample/provenance) and the water-quality caveat: recharge is
not a neutral act — Karnataka's hard-rock aquifers carry fluoride, urban runoff carries sewage.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from ..config import CFG
from ..io import save_json
from .containers import _DEFAULT_COUNTS, TAU as FLOOD_TAU

log = logging.getLogger("varuna.serve.recharge_sites")

# Water shallower than this is numerical film, not a harvestable pool.
TAU_R = 0.05


def _placeable(dom, work, buffer_cells=1):
    """0/1 mask of cells where a recharge structure CAN go: pervious ground only.

    Opposite land rule to storage: streets are out (wc==50 is IMPERVIOUS), water/wetland are
    out (already saturated), building footprints are out AND get a `buffer_cells` safety ring
    (recharging against a foundation is a geotechnical risk, not a win). Road cells from
    urban_grid.npz are excluded too — "free places away from concrete". Missing layers degrade
    gracefully so torch-only synthetic domains still work."""
    from scipy import ndimage

    from ..build.landcover import PERVIOUS
    wc = getattr(dom, "wc", None)
    if wc is not None:
        perv = np.isin(np.asarray(wc), sorted(PERVIOUS))
    else:
        log.warning("domain has no WorldCover grid — falling back to built==0 as 'pervious'")
        perv = (dom.built.cpu().numpy() == 0)
    m = perv.astype("float32")
    try:
        g = np.load(f"{work}/urban_grid.npz")
        b = np.asarray(g["buildings"]) > 0
        if b.shape == m.shape:
            if buffer_cells > 0:
                b = ndimage.binary_dilation(b, structure=np.ones((3, 3), bool),
                                            iterations=int(buffer_cells))
            m = m * (~b).astype("float32")
        r = np.asarray(g["roads"]) > 0
        if r.shape == m.shape:
            m = m * (~r).astype("float32")
    except Exception:  # noqa: BLE001 - no urban grid cached (synthetic bundle)
        pass
    return torch.as_tensor(m, dtype=torch.float32, device=dom.z0.device)


def _suitability(dom, work):
    """Per-cell local suitability weight for ranking + the Ksat grid (mm/hr) if soil data exists.

    suitability = norm(Ksat, capped at p95) × pervious fraction (7-cell smoothing — a candidate
    surrounded by pervious ground beats a lone pervious gap). Without sand/clay rasters the Ksat
    term drops out (weight 1), which keeps torch-only test domains honest rather than inventing
    soil numbers."""
    from scipy import ndimage

    from ..build.landcover import PERVIOUS
    N = dom.N
    wc = getattr(dom, "wc", None)
    perv = (np.isin(np.asarray(wc), sorted(PERVIOUS)).astype("float64")
            if wc is not None else (dom.built.cpu().numpy() == 0).astype("float64"))
    suit = ndimage.uniform_filter(perv, size=7)
    ksat = None
    try:
        from ..build.recharge import cosby_ksat
        from ..build.twin import _soil_pct
        sand = _soil_pct(work, "sand", dom.row0, dom.col0, 2 * N, N)
        clay = _soil_pct(work, "clay", dom.row0, dom.col0, 2 * N, N)
        if sand is not None and clay is not None:
            ksat = cosby_ksat(sand, clay)
            cap = max(float(np.nanpercentile(ksat, 95)), 1e-9)   # sand that drains to nowhere
            suit = suit * np.clip(ksat / cap, 0.0, 1.0)          # useful is not an infinite win
    except Exception as e:  # noqa: BLE001 - no rasterio / no soil rasters
        log.debug("no Ksat for suitability (%s)", e)
    return torch.as_tensor(suit, dtype=torch.float32, device=dom.z0.device), ksat


def _ranked_candidates(dom, rain_mm, placeable, suit):
    """Pooled cells ranked by pooled_depth × suitability, plus the UNMASKED baseline rollout.

    Same critical pattern as containers._ranked_minima: `placeable`/`suit` filter and order the
    CANDIDATES only; the returned baseline (ponded volume, infiltration, hmax) stays unmasked so
    the measured numbers are not flattered by siting rules."""
    with torch.no_grad():
        base = dom.rollout(dom.z0, rain_mm=rain_mm, track_infil=True)
    f = torch.relu(base["hmax"] - TAU_R)
    score = (f * placeable * suit).cpu().numpy()
    fnp = f.cpu().numpy()
    ys, xs = np.where(score > 0.0)
    order = np.argsort(-score[ys, xs])
    ys, xs = ys[order], xs[order]
    return ys, xs, fnp[ys, xs], base


def _land_runoff_m3(dom, base):
    """The storm's would-be runoff ON LAND: final ponded volume excluding permanent water —
    rain that fell on the Ganga or the harbour is not runoff anyone can recharge with."""
    from ..build.landcover import NO_RECHARGE
    h_end = np.asarray(base["frames"][-1])
    wc = getattr(dom, "wc", None)
    land = (~np.isin(np.asarray(wc), sorted(NO_RECHARGE))).astype("float64") \
        if wc is not None else np.ones_like(h_end, dtype="float64")
    return float((h_end * land).sum()) * dom.dx * dom.dx


def _recharge_for_sites(dom, ys, xs, depths, base, K, rain_mm, ksat_dom, runoff_m3=None):
    """Place K cell-sized recharge basins (carve by pooled depth + open the soil to Ksat),
    re-simulate, and measure the ADDED m³ that entered the ground vs the baseline rollout."""
    cell = dom.dx * dom.dx
    rr, cc = ys[:K], xs[:K]
    D = torch.zeros_like(dom.z0)
    D[rr, cc] = torch.as_tensor(depths[:K], dtype=torch.float32, device=dom.device)
    infil_new = dom.infil
    if ksat_dom is not None:
        # the structure removes the land-cover surface cap; the soil still limits throughput
        infil_new = dom.infil.clone()
        k_ms = torch.as_tensor(np.asarray(ksat_dom, dtype="float32") / 1000.0 / 3600.0,
                               device=dom.device)
        infil_new[rr, cc] = torch.maximum(infil_new[rr, cc], k_ms[rr, cc])
    keep = dom.infil
    try:
        dom.infil = infil_new
        with torch.no_grad():
            res = dom.rollout(dom.z0 - D, rain_mm=rain_mm, track_infil=True)
    finally:
        dom.infil = keep
    added = (float(res["infil_grid"].sum()) - float(base["infil_grid"].sum())) * cell
    would_be_runoff = max(runoff_m3 if runoff_m3 is not None else _land_runoff_m3(dom, base), 1.0)
    # flood co-benefit on built land, measured with the storage/flood threshold
    f0 = torch.relu(base["hmax"] - FLOOD_TAU) * dom.built
    f1 = torch.relu(res["hmax"] - FLOOD_TAU) * dom.built
    b0 = max(float(f0.sum()) * cell, 1.0)
    return dict(sites=int(K),
                recharge_m3=round(added),
                reduction_pct=round(100.0 * added / would_be_runoff, 1),
                flood_cut_pct=round(100.0 * (1 - float(f1.sum()) * cell / b0), 1),
                storage_m3=round(float(D.sum()) * cell),
                median_site_m3=round(float(np.median(depths[:K])) * cell))


def _flag_unstable(curve):
    """Mark dose-curve points where the explicit solver blew up (mirrors containers' guard):
    non-finite numbers, or a 'recharge' / flood response so negative it can only be numerics."""
    for c in curve:
        vals = [c.get("reduction_pct", 0.0), c.get("flood_cut_pct", 0.0), c.get("recharge_m3", 0.0)]
        c["unstable"] = bool(any(not np.isfinite(float(v)) for v in vals)
                             or c.get("reduction_pct", 0.0) < -50
                             or c.get("flood_cut_pct", 0.0) < -50)
    return curve


def plan_recharge(rain_mm=None, work=None, device=None, site_counts=None,
                  targets=(30, 50, 70), unit_m3=50.0, dom=None,
                  phase_targets=(10, 20, 40), site_list_max=1500,
                  building_buffer_cells=1, save=True):
    """Sweep recharge-structure count -> measured m³ into the aquifer; emit the buildable site
    list (pervious ground only, lat/lon + per-site volume) and a phased, indicatively-costed
    plan. Output mirrors storage_sizing.json (curve/targets/sites/phases) so the dashboard
    panel machinery is reusable — but reduction_pct here means 'share of the storm's would-be
    runoff sent underground', and recharge_m3 is the headline: measured, metered, not scored."""
    from ..build.twin import build_domain

    work = work or CFG.work
    rain_mm = CFG.design_rain_mm if rain_mm is None else float(rain_mm)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dom = dom or build_domain(work, device=device)
    cell = dom.dx * dom.dx

    suit, ksat_dom = _suitability(dom, work)
    placeable = _placeable(dom, work, buffer_cells=building_buffer_cells)
    ys, xs, depths, base = _ranked_candidates(dom, rain_mm, placeable, suit)
    n_cand = len(ys)
    if n_cand == 0:
        log.warning("no placeable pooled cells at %.0f mm — nothing to plan", rain_mm)
    counts = sorted(set([k for k in (site_counts or _DEFAULT_COUNTS) if k < n_cand] + ([n_cand] if n_cand else [])))
    would_be_runoff = _land_runoff_m3(dom, base)
    base_infil_m3 = float(base["infil_grid"].sum()) * cell

    curve = _flag_unstable([_recharge_for_sites(dom, ys, xs, depths, base, K, rain_mm, ksat_dom,
                                                runoff_m3=would_be_runoff)
                            for K in counts])
    for c in curve:
        log.info("recharge %5d sites -> %8.0f m3 to aquifer (%5.1f%% of runoff)%s",
                 c["sites"], c["recharge_m3"], c["reduction_pct"],
                 "  [UNSTABLE — excluded from targets]" if c["unstable"] else "")
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

    # ---- explicit site list: WHERE the structures actually go (pervious cells only)
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
    for i in range(int(min(site_list_max, n_cand))):
        r, c = int(ys[i]), int(xs[i])
        s = dict(rank=i + 1, row=r, col=c, site_m3=round(float(depths[i]) * cell))
        if to_ll is not None:
            s["latlon"] = to_ll(r, c)
        if ksat_dom is not None:
            s["ksat_mm_hr"] = round(float(ksat_dom[r, c]), 2)
        sites_out.append(s)

    # ---- phased implementation plan with indicative costs (percolation-structure rate)
    from .costbenefit import DEFAULT_COSTS
    rate = float(DEFAULT_COSTS["recharge_inr_per_m3"])
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

    # groundwater provenance: this plan does not certify, and it says who fed it
    try:
        from ..build.recharge import QUALITY_CAVEAT, groundwater_status
        gw_status = groundwater_status(work)
        caveat = QUALITY_CAVEAT
    except Exception as e:  # noqa: BLE001 - no pandas / unreadable bundle
        log.warning("groundwater status unavailable (%s)", e)
        gw_status, caveat = {"sample": True, "reason": f"status unavailable: {e}"}, ""

    soil_capacity_m3 = (round(float(dom.capacity.sum()) * cell)
                        if dom.capacity is not None else None)
    report = dict(rain_mm=rain_mm,
                  would_be_runoff_m3=round(would_be_runoff),
                  base_infil_m3=round(base_infil_m3),
                  soil_capacity_m3=soil_capacity_m3,
                  metered=dom.capacity is not None,
                  max_sites=int(n_cand), unit_m3=unit_m3,
                  curve=curve, targets=tgt, sites=sites_out, phases=phases,
                  recharge_inr_per_m3=rate, gw_status=gw_status,
                  note="Each site is a cell-sized recharge basin carved to its local pooled "
                       "depth with the soil opened to Ksat; the reported m³ is measured by "
                       "re-simulation and metered by per-cell soil capacity (metered=false "
                       "means the bundle predates the aquifer and numbers are an upper bound). "
                       "reduction_pct = share of the storm's would-be runoff ON LAND (permanent "
                       "water excluded) sent underground. "
                       "Sites sit on pervious ground only, buffered away from buildings. "
                       "Costs use an indicative percolation-structure rate — relative ROI, "
                       "not a bid. " + caveat)
    if save:
        save_json(f"{work}/recharge_plan.json", report)
    return report


def plot_recharge_dose(report, out="recharge_dose.png"):
    """Measured aquifer-recharge vs #structures curve with target markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    good = [c for c in report["curve"] if not c.get("unstable")]
    S = [c["sites"] for c in good]
    R = [c["reduction_pct"] for c in good]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(S, R, "o-", color="#0d9488", lw=2)
    for (lbl, v), col in zip(report["targets"].items(), ["#888", "#c0504d", "#3a6ea5"]):
        n = v["sites"]
        if n is None:                 # target beyond the stable part of the curve
            continue
        ax.axhline(float(lbl[:-1]), ls="--", color=col, alpha=.5)
        ax.axvline(n, ls="--", color=col, alpha=.5)
        ax.annotate(f"{lbl}\n{n} sites", (n, float(lbl[:-1])),
                    textcoords="offset points", xytext=(8, -28), color=col, fontsize=9)
    ax.set_xlabel("number of recharge structures (geography-sized, pervious ground only)")
    ax.set_ylabel("share of would-be runoff sent to the aquifer (%)")
    ax.set_title(f"Metered recharge — measured m³ vs # structures ({report['rain_mm']:.0f} mm storm)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
