"""Storm water balance: where does the rain go?

The twin's boundaries are closed, so every storm closes exactly:

    rain_in == ponded (final water on the ground) + infiltrated (absorbed by soil/vegetation)

`build_ladder` runs one instrumented rollout per rain on a ladder of storms and commits the
result as {work}/water_balance.json — the dashboard's mass-balance panel is then served by
interpolation (milliseconds, read-only), never by running physics on the free-tier Space.
The per-WorldCover-class split is what powers "volume locally reduced by soil & forestation";
the all-built counterfactual ("what if this ground were concrete?") makes it causal.

CPU cost: ~10 s/rain at 128^2, ~40 s at 256^2 — a build-time / nightly-job step.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from ..config import CFG
from ..io import load_json, save_json

log = logging.getLogger("varuna.serve.waterbalance")

# WorldCover classes grouped for the dashboard story. "vegetation" is the forestation lever
# (tree 10 / shrub 20 / grass 30 / crop 40); built 50 is ~impervious; water/wetland absorb 0
# (the canonical NO_RECHARGE set — see build.landcover).
from ..build.landcover import NO_RECHARGE as _NO_RECHARGE  # noqa: E402

CLASS_GROUPS = {
    "vegetation": (10, 20, 30, 40),
    "built": (50,),
    "bare": (60,),
    "water_wetland": tuple(sorted(_NO_RECHARGE)),
}

LADDER = tuple(range(10, 251, 20))          # 10, 30, ..., 250 mm

# The built-up infiltration rate (mm/hr) used for the "all concrete" counterfactual — matches
# F_TABLE[50] in build.twin so the counterfactual is "everything behaves like built-up land".
_BUILT_INFIL_MM_HR = 1.0


def _budget_for_rain(dom, rain_mm, storm_hr=1.5, total_hr=3.0):
    """One tracked rollout -> exact budget terms + per-class-group infiltration split.

    Closure `rain = infiltrated + ponded_final + runoff_out` is an ACCOUNTING IDENTITY of a
    closed-boundary run (runoff_out = 0 by construction), kept as a solver smoke test — it is
    not a validation against observations. ponded_final is the would-be runoff a real drainage
    path would export. With the V3 aquifer on, infiltrated is metered by soil capacity.
    """
    import torch

    cell = dom.dx * dom.dx
    out = dom.rollout(dom.z0, rain_mm, storm_hr=storm_hr, total_hr=total_hr, track_infil=True)
    rain_m3 = rain_mm / 1000.0 * dom.N * dom.N * cell
    infil_m3 = float(out["infil_grid"].sum()) * cell
    ponded_final_m3 = float(out["volume"][-1])
    ponded_peak_m3 = float(out["volume"].max())
    runoff_out_m3 = 0.0                                 # closed boundaries — nothing leaves
    closure = abs(rain_m3 - infil_m3 - ponded_final_m3 - runoff_out_m3) / max(rain_m3, 1.0)

    by_class = {}
    wc = torch.as_tensor(np.asarray(dom.wc), device=out["infil_grid"].device)
    for group, classes in CLASS_GROUPS.items():
        m = torch.zeros_like(wc, dtype=torch.bool)
        for c in classes:
            m |= wc == c
        by_class[group] = round(float(out["infil_grid"][m].sum()) * cell)
    known = sum(by_class.values())
    by_class["other"] = max(0, round(infil_m3) - known)

    e = dict(rain_mm=float(rain_mm), rain_m3=round(rain_m3),
             infiltrated_m3=round(infil_m3), infil_by_class=by_class,
             ponded_final_m3=round(ponded_final_m3), ponded_peak_m3=round(ponded_peak_m3),
             runoff_out_m3=round(runoff_out_m3),
             closure_err_pct=round(closure * 100, 3))
    if dom.capacity is not None:
        cap_m3 = float(dom.capacity.sum()) * cell
        e["soil_capacity_m3"] = round(cap_m3)
        e["soil_filled_pct"] = round(100.0 * infil_m3 / max(cap_m3, 1.0), 1)
    return e


def _counterfactual_ponded(dom, rain_mm, storm_hr=1.5, total_hr=3.0):
    """Final ponded m^3 if every cell infiltrated like built-up land (soil/vegetation removed)."""
    import torch
    from ..build.twin import Domain

    infil_cf = torch.full_like(dom.infil, _BUILT_INFIL_MM_HR / 1000.0 / 3600.0)
    # capacity deliberately None: the counterfactual is all-concrete, and concrete has no
    # soil column to fill — its ~1 mm/hr seepage is not storage-limited
    cf = Domain(dom.z0, dom.mann, infil_cf, dom.built, dx=dom.dx, device=str(dom.device))
    out = cf.rollout(cf.z0, rain_mm, storm_hr=storm_hr, total_hr=total_hr)
    return float(out["volume"][-1])


def build_ladder(work=None, rains=LADDER, device=None, counterfactual=True,
                 storm_hr=1.5, total_hr=3.0):
    """Run the ladder of storms, verify closure, write {work}/water_balance.json."""
    from ..build.twin import build_domain

    work = work or CFG.work
    dom = build_domain(work, device=device)
    entries = []
    for r in rains:
        e = _budget_for_rain(dom, r, storm_hr=storm_hr, total_hr=total_hr)
        if e["closure_err_pct"] > 0.5:
            raise AssertionError(
                f"water budget does not close at {r} mm: err {e['closure_err_pct']}% "
                f"(rain {e['rain_m3']} != infil {e['infiltrated_m3']} + ponded "
                f"{e['ponded_final_m3']} + runoff_out {e['runoff_out_m3']}) — "
                "solver smoke test failed (this is an accounting identity, so a miss means "
                "numerical blowup, not a bad calibration)")
        if counterfactual:
            cf = _counterfactual_ponded(dom, r, storm_hr=storm_hr, total_hr=total_hr)
            e["ponded_if_all_built_m3"] = round(cf)
            e["reduced_by_nature_m3"] = max(0, round(cf - e["ponded_final_m3"]))
        entries.append(e)
        log.info("ladder %3d mm: rain %.2e m3 | infil %.2e | ponded %.2e | nature saves %s m3",
                 r, e["rain_m3"], e["infiltrated_m3"], e["ponded_final_m3"],
                 e.get("reduced_by_nature_m3", "n/a"))
    report = dict(n_grid=dom.N, dx=dom.dx, storm_hr=storm_hr, total_hr=total_hr,
                  counterfactual_infil_mm_hr=_BUILT_INFIL_MM_HR if counterfactual else None,
                  entries=entries,
                  note=("Closed-boundary storm budget: rain = infiltrated + ponded + runoff_out "
                        "with runoff_out = 0 by construction. This closure is an accounting "
                        "identity (a solver smoke test), NOT a validation against observations; "
                        "ponded_final is the would-be runoff a real drainage path would export. "
                        "Where soil data exists, infiltration is metered by per-cell soil "
                        "storage capacity (V3 aquifer). 'reduced_by_nature' = extra final "
                        "ponding if all ground infiltrated like built-up land."))
    save_json(f"{work}/water_balance.json", report)
    return report


def _interp(entries, key, rain_mm):
    xs = np.asarray([e["rain_mm"] for e in entries], dtype="float64")
    ys = np.asarray([float(e.get(key) or 0.0) for e in entries], dtype="float64")
    return float(np.interp(rain_mm, xs, ys))


def water_balance(work=None, rain_mm=None, efficiency=1.0):
    """Serve-time budget at any rain by interpolating the committed ladder. Read-only.

    Adds the storable-volume view from storage_sizing.json (nominal capacity x `efficiency`,
    the dashboard's usable-fraction knob for silted / unmaintained containers).
    """
    work = work or CFG.work
    rain_mm = CFG.design_rain_mm if rain_mm is None else float(rain_mm)
    ladder = load_json(f"{work}/water_balance.json")
    if not ladder or not ladder.get("entries"):
        raise FileNotFoundError("water_balance.json not in bundle — run "
                                "varuna.serve.waterbalance.build_ladder (build/nightly step)")
    e = ladder["entries"]
    lo, hi = e[0]["rain_mm"], e[-1]["rain_mm"]
    r = min(max(rain_mm, lo), hi)
    out = dict(rain_mm=rain_mm, clamped_to=r if r != rain_mm else None,
               rain_m3=round(_interp(e, "rain_m3", r)),
               infiltrated_m3=round(_interp(e, "infiltrated_m3", r)),
               ponded_final_m3=round(_interp(e, "ponded_final_m3", r)),
               ponded_peak_m3=round(_interp(e, "ponded_peak_m3", r)),
               reduced_by_nature_m3=round(_interp(e, "reduced_by_nature_m3", r)),
               infil_by_class={g: round(_interp(
                   [dict(rain_mm=x["rain_mm"], v=x["infil_by_class"].get(g, 0)) for x in e],
                   "v", r)) for g in list(CLASS_GROUPS) + ["other"]},
               n_grid=ladder.get("n_grid"), dx=ladder.get("dx"),
               storm_hr=ladder.get("storm_hr"), note=ladder.get("note"))

    sizing = load_json(os.path.join(work, "storage_sizing.json"))
    if sizing and sizing.get("targets"):
        eff = min(max(float(efficiency), 0.05), 1.0)
        storable = {}
        for pct, t in sizing["targets"].items():
            if t.get("storage_m3"):
                storable[pct] = dict(nominal_m3=t["storage_m3"],
                                     usable_m3=round(t["storage_m3"] * eff),
                                     sites=t.get("sites"))
        out["storable"] = storable
        out["efficiency"] = eff
        out["storage_rain_mm"] = sizing.get("rain_mm")
    return out
