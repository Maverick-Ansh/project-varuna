"""Millisecond what-ifs via the trained U-Net emulator (notebook 05, part 2).

`whatif()` answers "how bad is an X mm storm?" and "what if we dig at these sites?" without
running the physics simulator. CPU is fine — a forward pass is milliseconds.
"""
from __future__ import annotations

import logging

import torch

from ..config import CFG

log = logging.getLogger("varuna.serve.emulator")

_CACHE = {}


def load_emulator(work=None, device="cpu"):
    """Load emulator.pt + twin_meta.pt + reconstruct the domain. Cached per work dir."""
    work = work or CFG.work
    key = (work, device)
    if key in _CACHE:
        return _CACHE[key]
    from ..build.twin import UNet, build_domain
    from ..io import require_bundle
    require_bundle(work, ["emulator.pt", "twin_meta.pt", "dem.tif", "worldcover.tif"])
    meta = torch.load(f"{work}/twin_meta.pt", map_location=device, weights_only=False)
    emu = UNet().to(device)
    emu.load_state_dict(torch.load(f"{work}/emulator.pt", map_location=device))
    emu.eval()
    dom = build_domain(work, center=meta.get("center"), device=device)
    N = dom.N
    sites = meta["sites"]
    masks = torch.zeros(len(sites), N, N, device=device)
    yy, xx = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing="ij")
    for i, (rr, cc) in enumerate(sites):
        masks[i] = ((yy - rr) ** 2 + (xx - cc) ** 2 <= CFG.site_radius ** 2).float()
    bundle = dict(emu=emu, dom=dom, masks=masks, meta=meta)
    _CACHE[key] = bundle
    return bundle


def _dig_to_map(dig_sites, masks, device):
    """dig_sites: list[float] length K, or dict{site_index: depth_m}. -> depth map tensor."""
    K = masks.shape[0]
    depths = torch.zeros(K, device=device)
    if dig_sites:
        if isinstance(dig_sites, dict):
            for i, d in dig_sites.items():
                depths[int(i)] = float(d)
        else:
            for i, d in enumerate(dig_sites[:K]):
                depths[i] = float(d)
    depths = depths.clamp(0, CFG.max_dig_m)
    return (depths.view(K, 1, 1) * masks).sum(0), depths


def whatif_grid(rain_mm, dig_sites=None, work=None, device="cpu"):
    """Like whatif but also returns the full max-depth grid (numpy) + dig map, for the dashboard's
    live flood overlay. Returns (hmax_np, dig_map_np, summary)."""
    b = load_emulator(work, device)
    emu, dom, masks, meta = b["emu"], b["dom"], b["masks"], b["meta"]
    zmean, zstd = meta["zmean"], meta["zstd"]
    D, depths = _dig_to_map(dig_sites, masks, device)
    x = torch.stack([(dom.z0 - D - zmean) / zstd,
                     torch.full_like(dom.z0, float(rain_mm) / 100.0), D / 3.0]).unsqueeze(0)
    with torch.no_grad():
        hmax = emu(x)[0, 0]
    cell = dom.dx * dom.dx
    flooded = torch.relu(hmax - 0.15) * dom.built
    summary = dict(
        rain_mm=float(rain_mm),
        dig_depths_m=[round(float(d), 2) for d in depths],
        flooded_area_m2=round(float((flooded > 0).sum()) * cell),
        flooded_volume_m3=round(float(flooded.sum()) * cell),
        peak_depth_m=round(float(hmax.max()), 2),
        note="Emulated (U-Net surrogate); see optimize_design for physics-grade dig planning.",
    )
    return hmax.cpu().numpy(), D.cpu().numpy(), summary


def whatif(rain_mm, dig_sites=None, work=None, device="cpu"):
    """Predict the max-depth grid for a storm (+ optional dig plan) via the emulator.

    Returns a compact summary: flooded area/volume on built-up land, peak depth, and the
    per-site dig depths applied. depth>0.15 m counts as flooded.
    """
    _, _, summary = whatif_grid(rain_mm, dig_sites, work, device)
    return summary
