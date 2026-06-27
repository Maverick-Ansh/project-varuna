"""Canal / drainage-channel planning — extends the dig optimiser from storage pits to routing.

Pipeline: simulate the design storm -> find the worst-flooded built-up cells (sources) -> route a
least-cost channel from each to its nearest candidate storage pit over the DEM (water follows low
ground) -> carve the channels (lower the bed, smooth the roughness) and dig the pits -> re-simulate
-> report the net flood-volume reduction and the canal geometry (lat/lon paths) for mapping.

Routing is heuristic (least-cost path over elevation) verified by the physics simulator — the path
geometry is not differentiable, but the flood-reduction it produces is measured, not assumed.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from ..config import CFG
from ..io import save_json

log = logging.getLogger("varuna.serve.canals")


def plan_canals(rain_mm=None, n_canals=3, channel_depth=2.0, channel_mann=0.02,
                pit_depth=None, work=None, device=None):
    """Plan up to n_canals draining the worst flooding into storage pits. Returns a plan dict
    and writes canal_plan.json + viz arrays into work."""
    from ..build.twin import build_domain, candidate_sites

    work = work or CFG.work
    rain_mm = CFG.design_rain_mm if rain_mm is None else float(rain_mm)
    pit_depth = CFG.max_dig_m if pit_depth is None else float(pit_depth)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dom = build_domain(work, device=device)
    sites, masks, site_area, eval_mask = candidate_sites(dom, work)
    N, DX = dom.N, dom.dx

    # baseline flooding on built-up land
    with torch.no_grad():
        hmax0 = dom.simulate(dom.z0, rain_mm=rain_mm)
    flood0 = torch.relu(hmax0 - 0.15) * dom.built

    # sources = distinct worst-flooded built cells
    fl = flood0.cpu().numpy().copy()
    sources = []
    for _ in range(n_canals):
        rr, cc = np.unravel_index(np.argmax(fl), fl.shape)
        if fl[rr, cc] <= 0:
            break
        sources.append((int(rr), int(cc)))
        fl[max(0, rr - 10):rr + 10, max(0, cc - 10):cc + 10] = 0

    z_np = dom.z0.cpu().numpy()
    z_carved = dom.z0.clone()
    mann_carved = dom.mann.clone()
    canal_mask = torch.zeros_like(dom.z0)
    canals = []
    if sources:
        from skimage.graph import MCP_Geometric
        # cost to traverse a cell = its elevation (water prefers the lowest corridor)
        cost = (z_np - z_np.min() + 1.0).astype("float64")
        for (sr, sc) in sources:
            tgt = min(sites, key=lambda p: (p[0] - sr) ** 2 + (p[1] - sc) ** 2)
            mcp = MCP_Geometric(cost)
            mcp.find_costs([list(tgt)])
            try:
                path = mcp.traceback([sr, sc])      # list of (row, col) from source to target
            except Exception as e:  # noqa: BLE001
                log.warning("canal routing failed for %s: %s", (sr, sc), e)
                continue
            for (rr, cc) in path:
                z_carved[rr, cc] = z_carved[rr, cc] - channel_depth
                mann_carved[rr, cc] = channel_mann
                canal_mask[rr, cc] = 1.0
            canals.append({"source": (sr, sc), "target": list(tgt), "path": path,
                           "length_m": len(path) * DX})

    # dig the storage pits too, then re-simulate the carved terrain
    pit_any = masks.sum(0).clamp(max=1.0)
    D_pit = (pit_depth * masks).sum(0)
    dom.mann = mann_carved
    with torch.no_grad():
        hmax1 = dom.simulate(z_carved - D_pit, rain_mm=rain_mm)
    flood1 = torch.relu(hmax1 - 0.15) * dom.built

    # net flood = water on streets EXCLUDING the engineered drainage (canals + pits hold water
    # by design — that's relocation, not flooding). Same accounting on before/after for fairness.
    streets = dom.built * (1.0 - (canal_mask + pit_any).clamp(max=1.0))
    base_vol = float((flood0 * streets).sum() * DX * DX)
    new_vol = float((flood1 * streets).sum() * DX * DX)

    # map domain cells -> lat/lon
    import rasterio
    with rasterio.open(f"{work}/dem.tif") as src:
        T = src.transform

    def latlon(rr, cc):
        lon, lat = T * (dom.col0 + cc * 2, dom.row0 + rr * 2)
        return [round(lat, 5), round(lon, 5)]

    canal_out = [{"length_m": round(c["length_m"]),
                  "from_latlon": latlon(*c["source"]),
                  "to_latlon": latlon(*c["target"]),
                  "path_latlon": [latlon(*p) for p in c["path"][::3]]} for c in canals]
    storage = [{"site": i, "latlon": latlon(*rc),
                "excavation_m3": round(pit_depth * float(site_area[i]))} for i, rc in enumerate(sites)]

    result = {
        "rain_mm": rain_mm, "n_canals": len(canal_out),
        "flooded_volume_m3": {"before": round(base_vol), "after": round(new_vol)},
        "reduction_pct": round(100 * (1 - new_vol / max(base_vol, 1)), 1),
        "canals": canal_out, "storage_sites": storage,
    }
    save_json(f"{work}/canal_plan.json", result)
    # arrays for the map renderer
    np.save(f"{work}/_flood_before.npy", flood0.cpu().numpy())
    np.save(f"{work}/_flood_after.npy", flood1.cpu().numpy())
    np.savez(f"{work}/_canal_geom.npz",
             z=z_np, row0=dom.row0, col0=dom.col0,
             sites=np.array(sites) if sites else np.zeros((0, 2)),
             paths=np.array([np.array(c["path"]) for c in canals], dtype=object))
    log.info("canals+pits cut flooding %.1f%% (%.0f -> %.0f m3)",
             result["reduction_pct"], base_vol, new_vol)
    return result
