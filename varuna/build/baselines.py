"""Paper baselines: score classic terrain indices against the same SAR truth as the twin.

Answers the reviewer question "is the differentiable twin better than trivial topographic
indices?" honestly: every method is scored on the identical 128x128 domain grid with the
identical jrc<50 permanent-water mask, using ONE global threshold per method chosen to
maximise its mean CSI over all SAR dates (best case for every method).

Methods: static depression depth (nb01), TWI, HAND-lite (height above the nearest
permanent-water cell — a Euclidean drainage proxy), and the dynamic twin swept over the
antecedent-rain window. Also `dem_uncertainty`: an ensemble of twin runs under spatially
correlated DEM perturbations, quantifying how DEM error bounds cell-level co-location skill
(total flooded area is robust; WHERE it floods is not).

Needs the bundle's observed_water_<date>.tif SAR masks (committed for Patna) and internet
for the ERA5 antecedent rain — run on Colab/Kaggle; no Earth Engine.
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import torch

from ..config import CFG
from ..io import save_json
from .calibrate import _crop_to_domain, _rain_for_date, align_sar, csi_hard, valid_mask

log = logging.getLogger("varuna.build.baselines")

WINDOWS = (1, 2, 3, 5)                                     # antecedent-rain windows (days)
TWIN_TAUS = (0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)


def sar_dates(work):
    """Dates with a committed observed_water_<date>.tif mask."""
    return sorted(os.path.basename(f)[len("observed_water_"):-4]
                  for f in glob.glob(os.path.join(work, "observed_water_*.tif")))


def terrain_layers(work, dom):
    """Static baseline layers on the domain grid: depression DEPTH, TWI, HAND-lite."""
    import rasterio
    from scipy.ndimage import distance_transform_edt

    def _read(name):
        with rasterio.open(os.path.join(work, name)) as s:
            return s.read(1).astype("float64")

    dem30, acc30, dep30, jrc30 = (_read(n) for n in
                                  ("dem.tif", "acc.tif", "depth.tif", "jrc_occurrence.tif"))
    gy, gx = np.gradient(dem30, 30.0)
    twi30 = np.log((acc30 * 900.0 + 1.0) / (np.hypot(gy, gx) + 1e-3))
    drain = jrc30 >= 50
    if drain.any():
        _, idx = distance_transform_edt(~drain, return_indices=True)
        hand30 = dem30 - dem30[idx[0], idx[1]]
    else:                                             # no permanent water in the AOI
        hand30 = np.zeros_like(dem30)
    return dict(DEPTH=_crop_to_domain(dep30, dom, agg="max"),
                TWI=_crop_to_domain(twi30, dom, agg="mean"),
                HAND=_crop_to_domain(hand30, dom, agg="mean"))


def sweep(dates, sar, valid, thresholds, wet_if="above", grid=None, grids=None):
    """Best single global threshold by mean CSI across dates.

    grid: one static layer scored against every date; grids: {date: layer} (dynamic twin).
    """
    best = None
    for t in thresholds:
        per = {}
        for d in dates:
            g = grid if grids is None else grids.get(d)
            if g is None:
                continue
            pred = (g > t) if wet_if == "above" else (g < t)
            per[d] = csi_hard(pred.float(), sar[d], tau=0.5, valid=valid)["csi"]
        if not per:
            continue
        mean = float(np.mean(list(per.values())))
        if best is None or mean > best["mean_csi"]:
            best = dict(threshold=round(float(t), 4), mean_csi=round(mean, 4),
                        per_date={d: round(v, 4) for d, v in per.items()})
    return best


def compare_baselines(work=None, dates=None, windows=WINDOWS, taus=TWIN_TAUS,
                      storm_hr=2.0, total_hr=4.0, device=None, make_figure=True):
    """Full comparison table -> baseline_comparison.json + BASELINES.md (+ figure)."""
    from .twin import build_domain
    work = work or CFG.work
    dom = build_domain(work, device=device)
    dates = dates or sar_dates(work)
    if len(dates) < 1:
        raise FileNotFoundError(f"no observed_water_<date>.tif masks in {work}")
    valid = valid_mask(work, dom)
    sar = {d: align_sar(work, d, dom) for d in dates}
    layers = terrain_layers(work, dom)

    results = {"static_depth": sweep(dates, sar, valid, np.linspace(0.05, 1.5, 30),
                                     "above", grid=layers["DEPTH"]),
               "twi": sweep(dates, sar, valid,
                            np.quantile(layers["TWI"].cpu().numpy(), np.linspace(0.50, 0.99, 25)),
                            "above", grid=layers["TWI"]),
               "hand_lite": sweep(dates, sar, valid, np.linspace(0.25, 15, 40),
                                  "below", grid=layers["HAND"])}

    best_twin, twin_all = None, {}
    for w in windows:
        grids = {}
        for d in dates:
            try:
                r = _rain_for_date(d, w)
            except Exception as e:  # noqa: BLE001 - rain archive hiccup: skip the date
                log.warning("rain lookup failed for %s (w=%dd): %s", d, w, e)
                continue
            with torch.no_grad():
                grids[d] = dom.simulate(dom.z0, rain_mm=r, storm_hr=storm_hr, total_hr=total_hr)
        r = sweep(dates, sar, valid, np.array(taus), "above", grids=grids)
        if r is None:
            continue
        r["window_days"] = w
        twin_all[f"window_{w}d"] = r
        if best_twin is None or r["mean_csi"] > best_twin["mean_csi"]:
            best_twin = r
    results["dynamic_twin"] = best_twin
    results["dynamic_twin_all_windows"] = twin_all

    report = dict(meta=dict(
        dates=dates, n_valid_cells=int(valid.sum()),
        protocol="single global threshold per method maximising mean CSI across all SAR dates; "
                 "identical domain grid and jrc<50 permanent-water mask on prediction AND truth",
        rain_source="Open-Meteo ERA5 archive, per-date antecedent totals"), **results)
    save_json(os.path.join(work, "baseline_comparison.json"), report)

    rows = [("static depression depth", results["static_depth"]),
            ("TWI (topographic wetness)", results["twi"]),
            ("HAND-lite (nearest drainage)", results["hand_lite"])]
    if best_twin:
        rows.append((f"dynamic twin (w={best_twin['window_days']}d)", best_twin))
    md = [f"# Baseline skill comparison vs Sentinel-1 SAR ({len(dates)} storms)\n",
          "Single global threshold per method (best mean CSI); same domain grid + "
          "permanent-water mask for all.\n",
          "| method | best threshold | mean CSI |", "|---|---|---|"]
    md += [f"| {n} | {r['threshold']} | **{r['mean_csi']:.3f}** |" for n, r in rows if r]
    md.append("\nPer-date CSI in baseline_comparison.json (twin antecedent-window sweep included).")
    with open(os.path.join(work, "BASELINES.md"), "w") as f:
        f.write("\n".join(md))
    if make_figure:
        _bar_figure(rows, work)
    for n, r in rows:
        log.info("%-32s thr=%-8s meanCSI=%.3f", n, r["threshold"], r["mean_csi"])
    return report


def _bar_figure(rows, work):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [n for n, r in rows if r]
    vals = [r["mean_csi"] for _, r in rows if r]
    ax.barh(names, vals, color=["#999"] * (len(vals) - 1) + ["#2a6fb0"])
    ax.set_xlabel("mean CSI vs SAR")
    ax.set_title("Waterlogging skill: baselines vs differentiable twin")
    for i, v in enumerate(vals):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center")
    os.makedirs(os.path.join(work, "figures"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(work, "figures", "baseline_comparison.png"), dpi=150)
    plt.close(fig)


def dem_uncertainty(work=None, rain_mm=None, n_members=10, sigma_m=1.0, corr_cells=5,
                    tau=0.15, storm_hr=2.0, total_hr=4.0, device=None, seed=0,
                    make_figure=True):
    """Twin ensemble under correlated DEM noise -> flood_uncertainty.json (+ figure).

    Key outputs: spread of TOTAL flooded area (robust) vs the area flooding in >=90% of
    members (small) — quantifies how DEM error caps cell-level co-location skill.
    """
    from scipy.ndimage import gaussian_filter
    from .twin import build_domain
    work = work or CFG.work
    rain_mm = CFG.design_rain_mm if rain_mm is None else float(rain_mm)
    dom = build_domain(work, device=device)
    base = dom.z0.cpu().numpy()
    rng = np.random.default_rng(seed)
    stack = []
    for _ in range(n_members):
        noise = gaussian_filter(rng.normal(0, 1, base.shape), sigma=corr_cells)
        noise *= sigma_m / max(noise.std(), 1e-9)
        z = torch.as_tensor(base + noise, device=dom.z0.device, dtype=dom.z0.dtype)
        with torch.no_grad():
            stack.append(dom.simulate(z, rain_mm=rain_mm, storm_hr=storm_hr, total_hr=total_hr))
    stack = torch.stack(stack)
    prob = (stack > tau).float().mean(0)
    cell_km2 = dom.dx * dom.dx / 1e6
    areas = [float((h > tau).sum()) * cell_km2 for h in stack]
    summary = dict(rain_mm=rain_mm, n_members=n_members, dem_sigma_m=sigma_m,
                   corr_len_cells=corr_cells, tau_m=tau,
                   flooded_km2_members=[round(a, 2) for a in areas],
                   flooded_km2_mean=round(float(np.mean(areas)), 2),
                   flooded_km2_std=round(float(np.std(areas)), 2),
                   km2_prob_ge_50pct=round(float((prob >= 0.5).sum()) * cell_km2, 2),
                   km2_prob_ge_90pct=round(float((prob >= 0.9).sum()) * cell_km2, 2),
                   note="ensemble of twin sims under spatially-correlated DEM perturbations; "
                        "robust flood zones = cells flooding in >=90% of members")
    save_json(os.path.join(work, "flood_uncertainty.json"), summary)
    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        im0 = axes[0].imshow(prob.cpu(), cmap="Blues", vmin=0, vmax=1)
        axes[0].set_title(f"P(flood > {tau} m) @ {rain_mm:.0f} mm, DEM ±{sigma_m} m")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(stack.std(0).cpu(), cmap="magma")
        axes[1].set_title("ensemble spread of max depth (m)")
        fig.colorbar(im1, ax=axes[1])
        for a in axes:
            a.set_xticks([]), a.set_yticks([])
        os.makedirs(os.path.join(work, "figures"), exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(work, "figures", "flood_uncertainty.png"), dpi=150)
        plt.close(fig)
    return summary
