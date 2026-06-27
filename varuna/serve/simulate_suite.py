"""Flood simulation suite — the "I need simulations of it" deliverable.

Built on the differentiable twin's no-grad `rollout` (build/twin.Domain.rollout). Produces the
evidence a planner / paper / dashboard wants:
  * dose_response   — flooded volume vs rainfall intensity (the design curve)
  * strategy_sweep  — do-nothing vs storage-pits, flooded volume + % cut
  * timelapse       — water rising & draining over time (frames for a GIF/MP4)
  * mass_curve      — total water in the domain over time (storm response hydrograph)

Everything here takes a Domain so it is testable offline on a synthetic domain; `domain_from_work`
rebuilds the real Patna domain from the artifact bundle (needs rasterio).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from ..config import CFG

log = logging.getLogger("varuna.serve.simulate_suite")


def domain_from_work(work=None, device="cpu"):
    from ..build.twin import build_domain
    return build_domain(work or CFG.work, device=device)


def dose_response(dom, rains=(25, 50, 75, 100, 150, 200), tau=0.15, built_only=True):
    """Flooded area/volume + peak depth at each rainfall intensity — the storm design curve."""
    rows = []
    cell = dom.dx * dom.dx
    for r in rains:
        with torch.no_grad():
            h = dom.simulate(dom.z0, rain_mm=float(r))
        flood = torch.relu(h - tau)
        if built_only:
            flood = flood * dom.built
        rows.append(dict(rain_mm=float(r),
                         flooded_area_m2=round(float((flood > 0).sum()) * cell),
                         flooded_volume_m3=round(float(flood.sum()) * cell),
                         peak_depth_m=round(float(h.max()), 3)))
        log.info("dose %3.0f mm -> %d m3 flooded on built land", r, rows[-1]["flooded_volume_m3"])
    return rows


def strategy_sweep(dom, masks, rain_mm=100.0, pit_depth=None, tau=0.15):
    """Flooded volume on built land for do-nothing vs digging all candidate storage pits."""
    pit_depth = CFG.max_dig_m if pit_depth is None else float(pit_depth)
    cell = dom.dx * dom.dx
    pit_any = masks.sum(0).clamp(max=1.0)
    streets = dom.built * (1.0 - pit_any)
    D = (pit_depth * masks).sum(0)
    with torch.no_grad():
        v0 = float((torch.relu(dom.simulate(dom.z0, rain_mm=rain_mm) - tau) * streets).sum()) * cell
        v1 = float((torch.relu(dom.simulate(dom.z0 - D, rain_mm=rain_mm) - tau) * streets).sum()) * cell
    return dict(rain_mm=float(rain_mm),
                do_nothing_m3=round(v0), with_pits_m3=round(v1),
                reduction_pct=round(100 * (1 - v1 / max(v0, 1)), 1))


def timelapse(dom, rain_mm=100.0, every=30, storm_hr=1.5, total_hr=3.0, probes=None):
    """Record the storm: depth frames + mass curve (+ optional probe-cell hydrographs)."""
    return dom.rollout(dom.z0, rain_mm=float(rain_mm), every=every,
                       storm_hr=storm_hr, total_hr=total_hr, probes=probes)


# --------------------------------------------------------------------------- renderers (matplotlib)


def plot_dose_response(rows, out="dose_response.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = [x["rain_mm"] for x in rows]
    v = [x["flooded_volume_m3"] for x in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r, v, "o-", color="steelblue")
    ax.set_xlabel("rainfall (mm)"); ax.set_ylabel("flooded volume on built land (m³)")
    ax.set_title("Patna flood dose–response"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def plot_mass_curve(roll, out="mass_curve.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(roll["times"], roll["volume"], "-", color="navy")
    ax.set_xlabel("time (hours)"); ax.set_ylabel("water stored in domain (m³)")
    ax.set_title("Storm response — rise and drain"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def render_timelapse(roll, out="flood_timelapse.gif", z=None, tau=0.15, fps=6):
    """Animate the depth frames into a GIF (Pillow writer). z optional hillshade-ish backdrop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    frames, times = roll["frames"], roll["times"]
    fig, ax = plt.subplots(figsize=(6, 6))
    if z is not None:
        ax.imshow(np.asarray(z), cmap="gray", origin="upper")
    im = ax.imshow(np.ma.masked_less(frames[0], tau), cmap="Blues", vmin=0, vmax=1.0,
                   alpha=0.85, origin="upper")
    ttl = ax.set_title(f"t = {times[0]:.2f} h"); ax.set_xticks([]); ax.set_yticks([])

    def upd(i):
        im.set_data(np.ma.masked_less(frames[i], tau))
        ttl.set_text(f"t = {times[i]:.2f} h")
        return im, ttl

    anim = FuncAnimation(fig, upd, frames=len(frames), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps)); plt.close(fig)
    log.info("wrote %s (%d frames)", out, len(frames))
    return out
