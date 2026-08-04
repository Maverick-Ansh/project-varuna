"""Join tiles into one city domain — so water crosses the seams.

Tiles were a memory-budget decision, and they cost physics: each tile simulates with closed
boundaries, so water that should flow from Powai into Bhandup instead piles up against an
invisible wall. This module mosaics a city's tile bundles onto one grid and builds a single
`Domain` over it, which is both more correct and — because one 817x586 storm costs less than
four 256x256 storms plus the bookkeeping — no slower.

The mosaic is exact, not resampled: tiles share the DEM's 30 m pixel grid (same CRS, same pixel
size, integer-offset transforms), so each tile's raster drops into place at an integer offset.
Cells no tile covers are sea or outside the city and are marked in `covered`.

`seam_audit` is the honest measurement of what tiling was costing: it re-runs a storm on the
city domain and on each individual tile, and compares depths in the edge band against the
interior. Edge disagreement above interior disagreement is the closed-boundary artifact.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from ..config import CFG
from .twin import F_TABLE, N_TABLE, Domain, _soil_infil_factor

log = logging.getLogger("varuna.build.city")

SEA_CLASS = 80          # ESA WorldCover "permanent water" — the fill for uncovered cells


def _read(work, name):
    import rasterio
    with rasterio.open(f"{work}/{name}.tif") as s:
        return s.read(1), s.transform


def city_grid(tile_works):
    """Union 30 m grid over the tiles: (height, width, pixel_deg, lat_top, lon_left, offsets)."""
    shapes, transforms = {}, {}
    for w in tile_works:
        arr, T = _read(w, "dem")
        shapes[w], transforms[w] = arr.shape, T
    pix = abs(transforms[tile_works[0]].a)
    lon0 = min(T.c for T in transforms.values())
    lat0 = max(T.f for T in transforms.values())
    right = max(T.c + abs(T.a) * shapes[w][1] for w, T in transforms.items())
    bottom = min(T.f - abs(T.e) * shapes[w][0] for w, T in transforms.items())
    H = int(round((lat0 - bottom) / pix))
    W = int(round((right - lon0) / pix))
    offsets = {w: (int(round((lat0 - T.f) / pix)), int(round((T.c - lon0) / pix)))
               for w, T in transforms.items()}
    return H, W, pix, lat0, lon0, offsets


def mosaic(tile_works, name, offsets, H, W, fill=np.nan):
    out = np.full((H, W), fill, dtype="float64")
    for w in tile_works:
        try:
            arr, _ = _read(w, name)
        except Exception as e:  # noqa: BLE001 — an optional layer (sand/clay) may be absent
            log.info("mosaic: %s has no %s (%s)", w, name, e)
            continue
        r0, c0 = offsets[w]
        out[r0:r0 + arr.shape[0], c0:c0 + arr.shape[1]] = arr
    return out


def build_city_domain(tile_works, device=None, dx=None):
    """One Domain over the union of the tiles' rasters, built exactly as `twin.build_domain`.

    Returns (domain, info) where info carries the grid geometry and the per-tile 60 m offsets
    needed to place a tile's array (or fitted sink field) into the city grid.
    """
    H, W, pix, lat0, lon0, off30 = city_grid(tile_works)
    dem30 = mosaic(tile_works, "dem", off30, H, W)
    wc30 = mosaic(tile_works, "worldcover", off30, H, W, fill=SEA_CLASS)
    sand30 = mosaic(tile_works, "sand", off30, H, W)
    clay30 = mosaic(tile_works, "clay", off30, H, W)
    covered30 = ~np.isnan(dem30)
    dem30 = np.where(covered30, dem30, 0.0)          # uncovered = sea level
    wc30 = np.where(np.isnan(wc30), SEA_CLASS, wc30)

    # 30 -> 60 m exactly as build_domain: mean-pool the DEM, stride the categorical layers
    H2, W2 = H // 2, W // 2
    dem = dem30[:H2 * 2, :W2 * 2].reshape(H2, 2, W2, 2).mean(axis=(1, 3))
    wc = wc30[:H2 * 2:2, :W2 * 2:2]
    covered = covered30[:H2 * 2:2, :W2 * 2:2]
    sand = np.nan_to_num(sand30[:H2 * 2:2, :W2 * 2:2] / 10.0, nan=35.0)
    clay = np.nan_to_num(clay30[:H2 * 2:2, :W2 * 2:2] / 10.0, nan=25.0)

    mann = np.vectorize(lambda v: N_TABLE.get(int(v), 0.035))(wc)
    infil_mm_hr = np.vectorize(lambda v: F_TABLE.get(int(v), 5.0))(wc) * _soil_infil_factor(clay)
    try:
        from .recharge import cosby_ksat
        infil_mm_hr = np.minimum(infil_mm_hr, cosby_ksat(sand, clay))
    except Exception as e:  # noqa: BLE001
        log.info("city: no Ksat reconciliation (%s)", e)
    built = (wc == 50).astype("float32")
    porosity = np.clip(0.505 - 0.00142 * sand - 0.00037 * clay, 0.05, 0.6)
    capacity = (porosity * CFG.root_zone_m * CFG.soil_avail_frac).astype("float32")

    dom = Domain(dem.astype("float32"), mann.astype("float32"),
                 (infil_mm_hr / 1000.0 / 3600.0).astype("float32"), built,
                 dx=float(dx or CFG.dx), device=device, capacity=capacity)
    dom.wc = wc.astype(np.int32)
    info = {"grid": (H2, W2), "pix_deg": pix, "origin": (lat0, lon0),
            "coverage": float(covered.mean()), "built_frac": float(built.mean()),
            "offsets60": {w: (r // 2, c // 2) for w, (r, c) in off30.items()},
            "offsets30": off30, "covered": covered}
    log.info("city domain %dx%d | coverage %.1f%% | built %.1f%%",
             H2, W2, 100 * info["coverage"], 100 * info["built_frac"])
    return dom, info


def place(city_arr, tile_arr, work, info, tile_row0=0, tile_col0=0):
    """Write a tile-domain array into the city grid at the tile's own crop offset.

    `tile_row0/col0` are the tile Domain's `row0`/`col0` (its crop inside its own raster) —
    forgetting them silently misaligns everything by tens of cells.
    """
    r0, c0 = info["offsets60"][work]
    r0 += tile_row0 // 2
    c0 += tile_col0 // 2
    n, m = np.shape(tile_arr)
    city_arr[r0:r0 + n, c0:c0 + m] = tile_arr
    return city_arr


def crop(city_arr, work, info, n, tile_row0=0, tile_col0=0):
    """Inverse of `place`: the n x n window of the city grid matching a tile's domain."""
    r0, c0 = info["offsets60"][work]
    r0 += tile_row0 // 2
    c0 += tile_col0 // 2
    return city_arr[r0:r0 + n, c0:c0 + n]


def seam_audit(city_dom, info, tile_doms, rain_mm=100.0, edge_cells=10,
               storm_hr=2.0, total_hr=4.0):
    """What did closed tile boundaries cost? Depth disagreement at the edges vs the interior.

    `tile_doms` maps a tile's work dir to its Domain. For each tile we run the same storm on
    the tile (closed boundaries) and read the same window out of the city run (open), then
    compare inside an `edge_cells`-wide band against the interior, over land only.
    """
    with torch.no_grad():
        hc = city_dom.simulate(city_dom.z0, rain_mm=rain_mm, storm_hr=storm_hr,
                               total_hr=total_hr)
    out = {}
    for work, dom in tile_doms.items():
        with torch.no_grad():
            ht = dom.simulate(dom.z0, rain_mm=rain_mm, storm_hr=storm_hr, total_hr=total_hr)
        r0, c0 = getattr(dom, "row0", 0), getattr(dom, "col0", 0)
        win = crop(hc, work, info, dom.N, r0, c0)
        zwin = crop(city_dom.z0, work, info, dom.N, r0, c0)
        wc_city = torch.as_tensor(np.asarray(city_dom.wc), device=hc.device)
        land = crop(wc_city, work, info, dom.N, r0, c0) != SEA_CLASS
        dh = (win - ht).abs()
        edge = torch.zeros_like(dh, dtype=torch.bool)
        edge[:edge_cells, :] = edge[-edge_cells:, :] = True
        edge[:, :edge_cells] = edge[:, -edge_cells:] = True
        out[work] = {
            "mean_abs_dz_m": round(float((zwin - dom.z0).abs()[land].mean()), 3),
            "edge_band_mean_dh_cm": round(float(dh[edge & land].mean()) * 100, 2),
            "interior_mean_dh_cm": round(float(dh[(~edge) & land].mean()) * 100, 2),
            "p99_dh_cm": round(float(torch.quantile(dh[land], 0.99)) * 100, 1),
        }
    return out


def city_drain_field(sink_paths, info, tile_offsets=None):
    """Mosaic the tiles' fitted sink fields (mm/h -> m/s) onto the city grid."""
    drain = np.zeros(info["grid"], dtype="float32")
    tile_offsets = tile_offsets or {}
    for work, path in sink_paths.items():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        r0, c0 = tile_offsets.get(work, (0, 0))
        place(drain, blob["drain_mm_h"].numpy() / 3.6e6, work, info, r0, c0)
    return drain


RAIN_LADDER = (25.0, 50.0, 75.0, 100.0, 150.0, 200.0)


def build_city_bundle(tile_works, out_dir, rains=RAIN_LADDER, sink_paths=None,
                      tile_offsets=None, device=None, storm_hr=2.0, total_hr=4.0):
    """Precompute the joined city's storms once, so serving needs no physics.

    A city storm is ~3 s on a T4 and minutes on the CPU the public Space runs on, so the live
    rainfall slider cannot drive the simulator at city scale. It does not need to: rainfall is
    one scalar, so a ladder of storms plus interpolation reproduces the slider exactly the way
    `serve/waterbalance.py` already does for the water budget.

    Writes `city_hmax.npz` (depth grids, float16 — a 60 m depth carries nowhere near 4 decimal
    digits of meaning) and `city_meta.json`. With `sink_paths`, the ladder is simulated WITH the
    inferred drainage and the drained volume per rung is recorded.
    """
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    dom, info = build_city_domain(tile_works, device=device)
    drain = None
    if sink_paths:
        from .sinkfield import DrainDomain
        drain = city_drain_field(sink_paths, info, tile_offsets)
        sim = DrainDomain(dom)
        sim.drain = torch.as_tensor(drain, device=sim.device)
    else:
        sim = dom

    grids, outflow, dt_used = {}, {}, {}
    for r in rains:
        # The Bates scheme is explicit: on tiles full of tidal creek the 10 s step can violate
        # CFL at high rainfall and the whole grid goes non-finite. A NaN ladder rung would be
        # served as a flood map, so halve the step and re-run rather than storing it.
        dt = 10.0
        for _attempt in range(4):
            if drain is not None:
                hmax, vol = sim.drained_volume_m3(float(r), storm_hr=storm_hr,
                                                  total_hr=total_hr, dt=dt)
            else:
                with torch.no_grad():
                    hmax = sim.simulate(sim.z0, rain_mm=float(r), storm_hr=storm_hr,
                                        total_hr=total_hr, dt=dt)
                vol = None
            if bool(torch.isfinite(hmax).all()):
                break
            dt /= 2.0
            log.warning("city ladder %.0f mm went non-finite — retrying at dt=%.2f s", r, dt)
        else:
            raise RuntimeError(
                f"city storm at {r} mm is non-finite even at dt={dt} s — the domain is "
                f"numerically unstable, do not serve this ladder")
        if vol is not None:
            outflow[str(r)] = round(vol)
        dt_used[str(r)] = dt
        grids[str(r)] = hmax.cpu().numpy().astype("float16")
        log.info("city ladder %.0f mm (dt %.1f s): wet@0.15 %.3f", r, dt,
                 float((hmax > 0.15).float().mean()))

    # `covered` means "some tile supplies data here" — with a complete grid that is everything,
    # INCLUDING the Arabian Sea and Thane creek. Reporting totals over it would count the sea as
    # flooded land (908 km^2 of a 1,723 km^2 grid at 200 mm, against ~600 km^2 of real Mumbai).
    # `land` is the mask every headline number must use.
    from .landcover import NO_RECHARGE
    land = info["covered"] & ~np.isin(np.asarray(dom.wc), list(NO_RECHARGE))
    log.info("city: %.0f km2 of grid, %.0f km2 land, %.0f km2 built",
             info["covered"].sum() * dom.dx ** 2 / 1e6, land.sum() * dom.dx ** 2 / 1e6,
             float(dom.built.sum()) * dom.dx ** 2 / 1e6)
    np.savez_compressed(os.path.join(out_dir, "city_hmax.npz"),
                        covered=info["covered"], land=land,
                        built=dom.built.cpu().numpy().astype(bool),
                        wc=np.asarray(dom.wc).astype("int16"), **grids)
    meta = {
        "grid": list(info["grid"]),
        "pix_deg": info["pix_deg"],
        "origin": list(info["origin"]),
        "dx": dom.dx,
        "coverage": round(info["coverage"], 4),
        "built_frac": round(info["built_frac"], 4),
        "rains": list(rains),
        "tiles": list(tile_works),
        "offsets60": {k: list(v) for k, v in info["offsets60"].items()},
        "with_sink_field": drain is not None,
        "outflow_m3": outflow,
        "dt_s": dt_used,
        "note": ("Depths are a ladder of design storms on the JOINED city domain (water crosses "
                 "the former tile seams). Between rungs the server interpolates; it does not "
                 "re-simulate. Outflow, when present, is a lower bound - see SINKFIELD_RESULTS.md."),
    }
    with open(os.path.join(out_dir, "city_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log.info("city bundle -> %s (%d rungs)", out_dir, len(grids))
    return meta


def city_outflow(city_dom, sink_paths, info, tile_offsets=None, rains=(50.0, 100.0, 200.0),
                 storm_hr=2.0, total_hr=4.0):
    """Drained volume per design storm on the city domain, using the tiles' fitted sink fields.

    `sink_paths` maps a tile work dir to its `sinkfield.pt`; `tile_offsets` maps it to that
    tile Domain's (row0, col0). Because the fit selects the smallest drainage that explains the
    observed extent, these are LOWER BOUNDS on what the city's drainage actually removes.
    """
    from .sinkfield import DrainDomain
    dd = DrainDomain(city_dom)
    dd.drain = torch.as_tensor(city_drain_field(sink_paths, info, tile_offsets),
                               device=dd.device)
    return {float(r): round(dd.drained_volume_m3(float(r), storm_hr=storm_hr,
                                                 total_hr=total_hr)[1])
            for r in rains}
