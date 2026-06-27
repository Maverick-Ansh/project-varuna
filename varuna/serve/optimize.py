"""Gradient-based intervention design (notebook 05, part 3).

Adam descends through the differentiable physics simulator to allocate excavation depth across
candidate sites so that flooded volume on built-up land is minimised under a budget. GPU strongly
recommended (autograd through hundreds of timesteps). Compares do-nothing / equal-split / optimal.
"""
from __future__ import annotations

import logging

import torch

from ..config import CFG
from ..io import save_json

log = logging.getLogger("varuna.serve.optimize")


def optimize_design(design_rain=None, budget_m3=None, iters=50, lam=5.0,
                    work=None, device=None, save=True):
    """Optimise dig depths for a design storm + budget.

    Returns dig_plan (per-site lat/lon/depth/excavation), flooded volumes for the three
    strategies, and the percentage reduction vs doing nothing.
    """
    from ..build.twin import build_domain, candidate_sites, dig_map

    work = work or CFG.work
    design_rain = CFG.design_rain_mm if design_rain is None else float(design_rain)
    budget_m3 = CFG.budget_m3 if budget_m3 is None else float(budget_m3)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        log.warning("optimize_design on CPU is slow (autograd through the simulator); prefer GPU.")

    dom = build_domain(work, device=device)
    sites, masks, site_area, eval_mask = candidate_sites(dom, work)
    K = len(sites)

    theta = torch.full((K,), -2.0, device=device, requires_grad=True)
    optt = torch.optim.Adam([theta], lr=0.15)
    hist = []
    for it in range(iters):
        D, depths = dig_map(theta, masks)
        hmax = dom.simulate(dom.z0 - D, rain_mm=design_rain, grad=True)
        flood = (torch.relu(hmax - 0.15) * eval_mask).sum() * dom.dx * dom.dx
        cost = (depths * site_area).sum()
        loss = flood + lam * torch.relu(cost - budget_m3)
        optt.zero_grad()
        loss.backward()
        optt.step()
        hist.append((float(flood.detach()), float(cost.detach())))
        if it % 5 == 0:
            log.info("iter %3d flooded %12.0f m3  excavation %10.0f m3", it, float(flood), float(cost))

    def fv(h):
        return float((torch.relu(h - 0.15) * eval_mask).sum() * dom.dx * dom.dx)

    with torch.no_grad():
        D_opt, depths_opt = dig_map(theta, masks)
        base = fv(dom.simulate(dom.z0, rain_mm=design_rain))
        opt_v = fv(dom.simulate(dom.z0 - D_opt, rain_mm=design_rain))
        eq_depth = min(budget_m3 / float(site_area.sum()), CFG.max_dig_m)
        D_eq = torch.full((K,), eq_depth, device=device).view(K, 1, 1).mul(masks).sum(0)
        eq_v = fv(dom.simulate(dom.z0 - D_eq, rain_mm=design_rain))

    # map domain site row/col back to lat/lon via the nb01 transform
    import rasterio
    with rasterio.open(f"{work}/dem.tif") as src:
        T = src.transform
    plan = []
    for i, (rr, cc) in enumerate(sites):
        row30, col30 = dom.row0 + rr * 2, dom.col0 + cc * 2
        lon, lat = T * (col30, row30)
        d = float(depths_opt[i])
        plan.append(dict(site=i, lat=round(lat, 5), lon=round(lon, 5),
                         dig_depth_m=round(d, 2),
                         excavation_m3=round(d * float(site_area[i]))))

    result = dict(
        design_rain_mm=design_rain, budget_m3=budget_m3,
        dig_plan=plan,
        total_excavation_m3=round(sum(p["excavation_m3"] for p in plan)),
        flooded_volume_m3=dict(do_nothing=round(base), equal_split=round(eq_v), optimal=round(opt_v)),
        reduction_pct=round(100 * (1 - opt_v / max(base, 1)), 1),
    )
    if save:
        save_json(f"{work}/dig_plan.json", result)
    log.info("optimal dig cuts flooding %.0f%% vs do-nothing", result["reduction_pct"])
    return result
