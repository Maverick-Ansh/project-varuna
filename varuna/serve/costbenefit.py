"""Cost-benefit ranking of flood interventions — flood volume removed per rupee.

Runs the three intervention families (canals+pits, distributed storage, gradient excavation) on the
selected bundle and ranks them by cost per cubic-metre of flood removed, so a planner sees the
highest-ROI move first. Unit costs are INDICATIVE Indian civil-works rates (editable) — treat the
ranking as relative, not a construction estimate. Every reduction is MEASURED by re-simulation
inside the underlying optimizers, not assumed.

Heavy (GPU strongly preferred): the excavation optimizer runs autograd through the simulator.
"""
from __future__ import annotations

import logging

from ..config import CFG
from ..io import save_json

log = logging.getLogger("varuna.serve.costbenefit")

# Indicative Indian civil-works unit rates (INR). Override via the `costs` argument / API.
DEFAULT_COSTS = {
    "excavation_inr_per_m3": 300.0,     # earthwork excavation + disposal
    "canal_inr_per_m": 9000.0,          # lined trapezoidal storm drain, per metre
    "storage_inr_per_m3": 6000.0,       # RCC detention capacity, per m3
}


def _item(name, reduced_m3, cost_inr, reduction_pct, detail=""):
    reduced_m3, cost_inr = float(reduced_m3), float(cost_inr)
    cpm = cost_inr / reduced_m3 if reduced_m3 > 0 else None
    return dict(name=name, detail=detail,
                flood_m3_reduced=round(reduced_m3), reduction_pct=round(float(reduction_pct), 1),
                cost_inr=round(cost_inr), cost_crore_inr=round(cost_inr / 1e7, 2),
                cost_per_m3_reduced_inr=round(cpm, 1) if cpm else None)


def rank_interventions(rain_mm=None, work=None, costs=None, device=None, iters=40):
    """Compare canals / storage / excavation and rank by cost per m³ of flood removed."""
    from .canals import plan_canals
    from .containers import plan_storage
    from .optimize import optimize_design

    work = work or CFG.work
    rain_mm = CFG.design_rain_mm if rain_mm is None else float(rain_mm)
    c = {**DEFAULT_COSTS, **(costs or {})}
    items = []

    # 1) canals + storage pits
    try:
        cp = plan_canals(rain_mm=rain_mm, work=work, device=device)
        v = cp["flooded_volume_m3"]
        reduced = max(v["before"] - v["after"], 0)
        canal_len = sum(x["length_m"] for x in cp["canals"])
        pit_excav = sum(x.get("excavation_m3", 0) for x in cp["storage_sites"])
        cost = canal_len * c["canal_inr_per_m"] + pit_excav * c["excavation_inr_per_m3"]
        items.append(_item("Canals + storage pits", reduced, cost, cp["reduction_pct"],
                           detail=f"{cp['n_canals']} canals ({round(canal_len)} m) to {cp['outfalls']}"))
    except Exception as e:  # noqa: BLE001
        log.warning("canals cost-benefit failed: %s", e)

    # 2) distributed storage sized to ~50% cut
    try:
        st = plan_storage(rain_mm=rain_mm, work=work, device=device,
                          site_counts=(100, 300, 600, 1000, 2000), targets=(50,))
        tg = st.get("targets", {}).get("50%")
        if tg:
            reduced = 0.5 * st["total_flood_m3"]
            cost = tg["storage_m3"] * c["storage_inr_per_m3"]
            items.append(_item("Distributed storage (~50% cut)", reduced, cost, 50.0,
                               detail=f"{tg['sites']} sites, {round(tg['storage_m3'])} m³ capacity"))
    except Exception as e:  # noqa: BLE001
        log.warning("storage cost-benefit failed: %s", e)

    # 3) gradient-optimised excavation
    try:
        dg = optimize_design(design_rain=rain_mm, work=work, device=device, iters=iters, save=False)
        fv = dg["flooded_volume_m3"]
        reduced = max(fv["do_nothing"] - fv["optimal"], 0)
        cost = dg["total_excavation_m3"] * c["excavation_inr_per_m3"]
        items.append(_item("Optimised excavation", reduced, cost, dg["reduction_pct"],
                           detail=f"{round(dg['total_excavation_m3'])} m³ across {len(dg['dig_plan'])} sites"))
    except Exception as e:  # noqa: BLE001
        log.warning("excavation cost-benefit failed: %s", e)

    items.sort(key=lambda x: x["cost_per_m3_reduced_inr"] if x["cost_per_m3_reduced_inr"] else 1e18)
    for i, it in enumerate(items):
        it["rank"] = i + 1
    report = dict(rain_mm=rain_mm, costs_assumed_inr=c, interventions=items,
                  note="Indicative unit rates; ranking is relative flood-reduction ROI, not a bid "
                       "estimate. Flood volume is measured on built-up land by re-simulation.")
    save_json(f"{work}/costbenefit.json", report)
    log.info("cost-benefit: %s", " | ".join(f"{it['name']}={it['cost_per_m3_reduced_inr']}₹/m³"
                                             for it in items))
    return report
