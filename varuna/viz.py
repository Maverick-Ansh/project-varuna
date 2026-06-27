"""Static map renders (PNG) of the build outputs — viewable on GitHub, no map tiles needed.

DEM hillshade basemap + overlays. Kept to matplotlib-to-file (never inline) so large image/HTML
blobs don't overflow MCP tool results (memory note). Each function returns the saved path.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import CFG

log = logging.getLogger("varuna.viz")


def _hillshade(z, azimuth=315, altitude=45):
    """Simple analytical hillshade in [0,1] for a basemap."""
    z = np.asarray(z, dtype="float64")
    dy, dx = np.gradient(z)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return (hs + 1.0) / 2.0


def _dem_extent(transform, H, W):
    """[lon_min, lon_max, lat_min, lat_max] for imshow extent."""
    lon0, lat_top = transform.c, transform.f
    lon1 = transform.c + W * transform.a
    lat_bot = transform.f + H * transform.e
    return [lon0, lon1, lat_bot, lat_top]


def map_storage_sites(work=None, out=None, top_label=8):
    """Hillshade + recharge/storage sites (size=recharge score, colour=RSI)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import rasterio

    work = work or CFG.work
    out = out or f"{work}/map_storage_sites.png"
    with rasterio.open(f"{work}/dem.tif") as s:
        z = s.read(1)
        ext = _dem_extent(s.transform, s.height, s.width)
    df = pd.read_csv(f"{work}/recharge_sites.csv")
    smax = max(float(df.recharge_score.max()), 1.0)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(_hillshade(z), cmap="gray", extent=ext, origin="upper")
    sc = ax.scatter(df.lon, df.lat, s=30 + (df.recharge_score / smax) * 500,
                    c=df.rsi, cmap="YlGn", vmin=0, vmax=1, edgecolor="k", linewidth=0.6, zorder=3)
    for _, r in df.head(top_label).iterrows():
        ax.annotate(f" {int(r.sink_id)}", (r.lon, r.lat), fontsize=8, color="darkred", zorder=4)
    plt.colorbar(sc, ax=ax, label="Recharge Suitability Index (0–1)")
    ax.set_title("Patna — candidate storage / recharge sites  (marker size = recharge score)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close(fig)
    log.info("wrote %s", out)
    return out


def _domain_extent(work, dom):
    import rasterio
    with rasterio.open(f"{work}/dem.tif") as s:
        T = s.transform
    lon0, lat_top = T * (dom.col0, dom.row0)
    lon1, lat_bot = T * (dom.col0 + dom.N * 2, dom.row0 + dom.N * 2)
    return [lon0, lon1, lat_bot, lat_top]


def map_flood(work=None, rain_mm=100, out=None, device="cpu"):
    """Simulate a storm on the real crop and overlay max water depth on the hillshade."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from .build.twin import build_domain

    work = work or CFG.work
    out = out or f"{work}/map_flood_{int(rain_mm)}mm.png"
    dom = build_domain(work, device=device)
    with torch.no_grad():
        h = dom.simulate(dom.z0, rain_mm=float(rain_mm)).cpu().numpy()
    z = dom.z0.cpu().numpy()
    ext = _domain_extent(work, dom)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(_hillshade(z), cmap="gray", extent=ext, origin="upper")
    im = ax.imshow(np.ma.masked_less(h, 0.15), cmap="Blues", extent=ext, origin="upper",
                   vmin=0, vmax=1.0, alpha=0.85)
    plt.colorbar(im, ax=ax, label="max water depth (m)")
    ax.set_title(f"Patna flood simulation — {float(rain_mm):.0f} mm storm  (central 7.7 km window)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close(fig)
    log.info("wrote %s", out)
    return out


def map_canal_plan(work=None, out=None):
    """Side-by-side flood before/after, with canal routes + storage pits on the 'after' panel.

    Requires plan_canals() to have run (reads _flood_before/after.npy, _canal_geom.npz, canal_plan.json).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .io import load_json

    work = work or CFG.work
    out = out or f"{work}/map_canal_plan.png"
    fb = np.load(f"{work}/_flood_before.npy")
    fa = np.load(f"{work}/_flood_after.npy")
    g = np.load(f"{work}/_canal_geom.npz", allow_pickle=True)
    res = load_json(f"{work}/canal_plan.json", {})

    z = g["z"]; N = fb.shape[0]
    zc = z if z.shape == (N, N) else z[:N, :N]   # saved z is already the 128^2 domain
    hs = _hillshade(zc)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    for a, F, ttl in [(ax[0], fb, "do nothing"), (ax[1], fa, "with canals + storage pits")]:
        a.imshow(hs, cmap="gray", origin="upper")
        a.imshow(np.ma.masked_less(F, 0.15), cmap="Blues", vmin=0, vmax=1.0, alpha=0.85, origin="upper")
        a.set_title(f"flood — {ttl}"); a.set_xticks([]); a.set_yticks([])
    for p in g["paths"]:
        p = np.asarray(p)
        if p.size:
            ax[1].plot(p[:, 1], p[:, 0], "-", color="red", lw=2.2, zorder=3)
    sites = np.asarray(g["sites"])
    if sites.size:
        ax[1].scatter(sites[:, 1], sites[:, 0], c="lime", edgecolor="k", s=70, marker="s", zorder=4,
                      label="storage pits")
        ax[1].legend(loc="upper right", fontsize=8)
    red = res.get("reduction_pct", "?")
    fig.suptitle(f"Patna canal + storage plan — {res.get('rain_mm', '?')} mm storm  →  "
                 f"net flood cut {red}%  (red = canals)", fontsize=13)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close(fig)
    log.info("wrote %s", out)
    return out


def render_all(work=None, rain_mm=100):
    """Convenience: storage-sites + flood + (if canal_plan ran) canal-plan maps."""
    import os
    work = work or CFG.work
    outs = [map_storage_sites(work), map_flood(work, rain_mm)]
    if os.path.exists(f"{work}/_canal_geom.npz"):
        outs.append(map_canal_plan(work))
    return outs
