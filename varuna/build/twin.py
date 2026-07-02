"""Notebook 05 + simulator_test -> the differentiable flood twin.

A 2-D inertial shallow-water simulator (Bates et al. 2010) written as pure torch ops, so it is
differentiable end to end; a U-Net emulator trained on simulated storms for millisecond what-ifs;
and the candidate-site / dig-map machinery used by the gradient optimiser (varuna.serve.optimize).

The simulator `step`/`simulate` are lifted verbatim from the unit-tested simulator_test.py.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from ..config import CFG

log = logging.getLogger("varuna.build.twin")

G, HMIN = 9.81, 1e-3

# Manning n and infiltration (mm/hr) by ESA WorldCover class
N_TABLE = {10: 0.060, 20: 0.050, 30: 0.040, 40: 0.040, 50: 0.015, 60: 0.030, 80: 0.030, 90: 0.045}
F_TABLE = {10: 10.0, 20: 8.0, 30: 8.0, 40: 8.0, 50: 1.0, 60: 5.0, 80: 0.0, 90: 0.0}


def _device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class Domain:
    """A flood-model domain: bed elevation z0, per-cell Manning n + infiltration, built mask.

    Holds the simulator. `step`/`simulate` are the verbatim Bates 2010 scheme; the only change
    from the notebook is binding mann/infil/dx to the instance instead of module globals.
    """

    def __init__(self, z0, mann, infil, built, dx=None, device=None):
        self.device = _device(device)
        self.dx = float(dx or CFG.dx)
        self.z0 = torch.as_tensor(z0, dtype=torch.float32, device=self.device)
        self.mann = torch.as_tensor(mann, dtype=torch.float32, device=self.device)
        self.infil = torch.as_tensor(infil, dtype=torch.float32, device=self.device)
        self.built = torch.as_tensor(built, dtype=torch.float32, device=self.device)
        self.N = self.z0.shape[0]
        self.row0 = 0
        self.col0 = 0

    def step(self, h, qx, qy, z, rain_ms, dt):
        mann, infil, DX = self.mann, self.infil, self.dx
        eta = z + h
        hf = torch.clamp(torch.maximum(eta[:, :-1], eta[:, 1:]) - torch.maximum(z[:, :-1], z[:, 1:]), min=0.0)
        Sx = (eta[:, 1:] - eta[:, :-1]) / DX
        nx = 0.5 * (mann[:, :-1] + mann[:, 1:])
        hf73 = torch.clamp(hf, min=HMIN) ** (7.0 / 3.0)
        qx = (qx - G * hf * dt * Sx) / (1.0 + G * dt * nx ** 2 * torch.abs(qx) / hf73)
        qcap = 0.25 * hf * DX / dt
        qx = torch.where(hf > HMIN, torch.clamp(qx, -qcap, qcap), torch.zeros_like(qx))
        hfy = torch.clamp(torch.maximum(eta[:-1, :], eta[1:, :]) - torch.maximum(z[:-1, :], z[1:, :]), min=0.0)
        Sy = (eta[1:, :] - eta[:-1, :]) / DX
        ny = 0.5 * (mann[:-1, :] + mann[1:, :])
        hfy73 = torch.clamp(hfy, min=HMIN) ** (7.0 / 3.0)
        qy = (qy - G * hfy * dt * Sy) / (1.0 + G * dt * ny ** 2 * torch.abs(qy) / hfy73)
        qcapy = 0.25 * hfy * DX / dt
        qy = torch.where(hfy > HMIN, torch.clamp(qy, -qcapy, qcapy), torch.zeros_like(qy))
        Qx = torch.nn.functional.pad(qx, (1, 1, 0, 0))     # closed boundaries
        Qy = torch.nn.functional.pad(qy, (0, 0, 1, 1))
        dh = dt / DX * (Qx[:, :-1] - Qx[:, 1:] + Qy[:-1, :] - Qy[1:, :])
        h = torch.clamp(h + dh + rain_ms * dt - infil * dt, min=0.0)
        return h, qx, qy

    def rollout(self, z, rain_mm, storm_hr=1.5, total_hr=3.0, dt=10.0, every=30, probes=None):
        """No-grad storm rollout that RECORDS dynamics (for animation / hydrographs / mass curves).

        Returns dict: frames (T,N,N depth snapshots every `every` steps), times (hours), volume
        (total water m^3 per frame), hmax, and probe depth time-series at `probes` [(row,col),...].
        """
        with torch.no_grad():
            h = torch.zeros_like(z)
            qx = torch.zeros(z.shape[0], z.shape[1] - 1, device=z.device)
            qy = torch.zeros(z.shape[0] - 1, z.shape[1], device=z.device)
            hmax = torch.zeros_like(z)
            nsteps = int(total_hr * 3600 / dt)
            rain_steps = int(storm_hr * 3600 / dt)
            rain_rate = rain_mm / 1000.0 / (storm_hr * 3600.0)
            cell = self.dx * self.dx
            frames, times, volume, probe_ts = [], [], [], []
            for k in range(nsteps):
                r = rain_rate if k < rain_steps else 0.0
                h, qx, qy = self.step(h, qx, qy, z, r, dt)
                hmax = torch.maximum(hmax, h)
                if k % every == 0 or k == nsteps - 1:
                    frames.append(h.cpu().numpy().copy())
                    times.append(k * dt / 3600.0)
                    volume.append(float(h.sum()) * cell)
                    if probes:
                        probe_ts.append([float(h[r0, c0]) for (r0, c0) in probes])
        return dict(frames=np.stack(frames), times=np.asarray(times),
                    volume=np.asarray(volume), hmax=hmax,
                    probes=np.asarray(probe_ts) if probes else None)

    def simulate(self, z, rain_mm, storm_hr=1.5, total_hr=3.0, dt=10.0, chunk=60,
                 grad=False, return_final=False):
        """Run a storm; return max water depth grid (`hmax`). rain_mm falls uniformly over storm_hr.

        return_final=True -> return (hmax, h_end). hmax is the per-cell max over time (what the
        emulator/optimizer use); h_end is the final depth (the mass-conserving state).
        """
        h = torch.zeros_like(z)
        qx = torch.zeros(z.shape[0], z.shape[1] - 1, device=z.device)
        qy = torch.zeros(z.shape[0] - 1, z.shape[1], device=z.device)
        hmax = torch.zeros_like(z)
        nsteps = int(total_hr * 3600 / dt)
        rain_steps = int(storm_hr * 3600 / dt)
        rain_rate = rain_mm / 1000.0 / (storm_hr * 3600.0)

        def run_chunk(h, qx, qy, hmax, k0, k1):
            for k in range(int(k0), int(k1)):
                r = rain_rate if k < rain_steps else 0.0
                h, qx, qy = self.step(h, qx, qy, z, r, dt)
                hmax = torch.maximum(hmax, h)
            return h, qx, qy, hmax

        k = 0
        while k < nsteps:
            k1 = min(k + chunk, nsteps)
            if grad:
                h, qx, qy, hmax = torch.utils.checkpoint.checkpoint(
                    run_chunk, h, qx, qy, hmax, torch.tensor(k), torch.tensor(k1), use_reentrant=False)
            else:
                h, qx, qy, hmax = run_chunk(h, qx, qy, hmax, k, k1)
            k = k1
        return (hmax, h) if return_final else hmax


def _bundle_meta(work):
    """Load twin_meta.pt (crop centre, grid, dx) from a bundle; {} if absent/unreadable."""
    import os
    path = f"{work}/twin_meta.pt"
    if not os.path.exists(path):
        return {}
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        log.warning("could not read %s: %s", path, e)
        return {}


def build_domain(work=None, center=None, device=None):
    """Crop an N x N / dx-m domain over `center` from nb01's dem.tif + worldcover.tif.

    Area-correct from the bundle: when `center` is not given explicitly, the crop centre, grid
    size and cell size are read from the bundle's twin_meta.pt (so a non-Patna bundle crops at
    its own location), falling back to CFG for a not-yet-trained bundle.
    """
    import rasterio
    work = work or CFG.work
    meta = {} if center is not None else _bundle_meta(work)
    center = center or tuple(meta.get("center", CFG.center))
    N = int(meta.get("n_grid", CFG.n_grid))
    dx = float(meta.get("dx", CFG.dx))
    N30 = N * 2
    with rasterio.open(f"{work}/dem.tif") as src:
        dem30 = src.read(1).astype("float64")
        T = src.transform
    with rasterio.open(f"{work}/worldcover.tif") as src:
        wc30 = src.read(1)
    col0 = int((center[1] - T.c) / T.a) - N30 // 2
    row0 = int((center[0] - T.f) / T.e) - N30 // 2
    row0 = max(0, min(row0, dem30.shape[0] - N30))
    col0 = max(0, min(col0, dem30.shape[1] - N30))
    dem_c = dem30[row0:row0 + N30, col0:col0 + N30]
    wc_c = wc30[row0:row0 + N30, col0:col0 + N30]
    z_np = dem_c.reshape(N, 2, N, 2).mean(axis=(1, 3))
    wc_np = wc_c[::2, ::2]
    mann_np = np.vectorize(lambda v: N_TABLE.get(int(v), 0.035))(wc_np)
    infil_mm_hr = np.vectorize(lambda v: F_TABLE.get(int(v), 5.0))(wc_np)   # by land cover

    # Soil modulation: scale land-cover infiltration by the soil's absorbability. If nb02's
    # clay.tif exists, clayey ground soaks slower, sandy ground faster (factor ~[0.3, 1.5]).
    # Concrete (built-up) stays ~impervious regardless because its base rate is already ~1 mm/hr.
    soil_factor = _soil_infil_factor(work, row0, col0, N30, N)
    infil_np = (infil_mm_hr * soil_factor) / 1000.0 / 3600.0                # m/s

    built_np = (wc_np == 50).astype("float32")
    dom = Domain(z_np, mann_np, infil_np, built_np, dx=dx, device=device)
    dom.row0, dom.col0 = row0, col0
    dom.wc = wc_np.astype(np.int32)        # WorldCover class per cell -> calibrate.py learnable physics
    return dom


def _soil_infil_factor(work, row0, col0, N30, N):
    """Per-cell infiltration multiplier from soil clay %. 1.0 (neutral) if clay.tif absent."""
    import os
    import rasterio
    path = f"{work}/clay.tif"
    if not os.path.exists(path):
        return 1.0
    with rasterio.open(path) as src:
        clay = src.read(1).astype("float64")
    clay = clay[row0:row0 + N30, col0:col0 + N30][::2, ::2] / 10.0          # SoilGrids g/kg -> %
    if clay.shape != (N, N):                                                # defensive (edge crops)
        clay = np.pad(clay, ((0, max(0, N - clay.shape[0])), (0, max(0, N - clay.shape[1]))),
                      mode="edge")[:N, :N]
    # clay 10% -> 1.0x, 30% -> 0.6x, >=50% -> 0.3x (floor); sandier than 10% speeds up to 1.5x
    return np.clip(1.2 - 0.02 * clay, 0.3, 1.5)


def candidate_sites(dom, work=None, k=None, radius=None):
    """K intervention sites (domain row,col) from nb01 sinks, padded with deepest cells.

    Returns (sites, masks, site_area, eval_mask).
    """
    import pandas as pd
    work = work or CFG.work
    K = k or CFG.n_sites
    RADIUS = radius or CFG.site_radius
    N = dom.N
    sites = []
    try:
        sk = pd.read_csv(f"{work}/sinks.csv")
        for _, s in sk.iterrows():
            rr, cc = (int(s.row) - dom.row0) // 2, (int(s.col) - dom.col0) // 2
            if 5 <= rr < N - 5 and 5 <= cc < N - 5:
                sites.append((rr, cc))
            if len(sites) == K:
                break
    except Exception as e:  # noqa: BLE001
        log.warning("sinks.csv unavailable for sites: %s", e)
    with torch.no_grad():
        hm = dom.simulate(dom.z0, rain_mm=80.0).cpu().numpy().copy()
    while len(sites) < K:
        rr, cc = np.unravel_index(np.argmax(hm), hm.shape)
        hm[max(0, rr - 8):rr + 8, max(0, cc - 8):cc + 8] = 0
        if 5 <= rr < N - 5 and 5 <= cc < N - 5:
            sites.append((int(rr), int(cc)))
    masks = torch.zeros(K, N, N, device=dom.device)
    yy, xx = torch.meshgrid(torch.arange(N, device=dom.device),
                            torch.arange(N, device=dom.device), indexing="ij")
    for i, (rr, cc) in enumerate(sites):
        masks[i] = ((yy - rr) ** 2 + (xx - cc) ** 2 <= RADIUS ** 2).float()
    site_area = masks.sum(dim=(1, 2)) * dom.dx * dom.dx
    site_any = masks.sum(0).clamp(max=1.0)
    eval_mask = dom.built * (1.0 - site_any)   # judge flooding on built land EXCLUDING the pits
    return sites, masks, site_area, eval_mask


def dig_map(theta, masks, max_dig=None):
    """theta (unconstrained) -> (depth_map, per-site depths). depths in [0, max_dig]."""
    max_dig = CFG.max_dig_m if max_dig is None else max_dig
    depths = max_dig * torch.sigmoid(theta)
    return (depths.view(-1, 1, 1) * masks).sum(0), depths


class UNet(nn.Module):
    """Tiny 3-level U-Net: 3 input channels (norm elevation, rain/100, dig/3) -> 1 (max depth)."""

    def __init__(s, ch=3):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.ReLU())

        s.e1, s.e2, s.e3 = blk(ch, 16), blk(16, 32), blk(32, 64)
        s.pool = nn.MaxPool2d(2)
        s.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        s.d2 = blk(64, 32)
        s.up1 = nn.ConvTranspose2d(32, 16, 2, 2)
        s.d1 = blk(32, 16)
        s.out = nn.Conv2d(16, 1, 1)

    def forward(s, x):
        e1 = s.e1(x)
        e2 = s.e2(s.pool(e1))
        e3 = s.e3(s.pool(e2))
        d2 = s.d2(torch.cat([s.up2(e3), e2], 1))
        d1 = s.d1(torch.cat([s.up1(d2), e1], 1))
        return torch.nn.functional.softplus(s.out(d1))


def gen_dataset(dom, masks, n_samples=None, seed=None):
    """Random storms x random dig allocations -> (X, Y) tensors. Seeded for reproducibility."""
    n_samples = n_samples or CFG.n_samples
    seed = CFG.seed if seed is None else seed
    g = np.random.default_rng(seed)
    torch.manual_seed(seed)
    K = masks.shape[0]
    X, Y = [], []
    zmean, zstd = dom.z0.mean(), dom.z0.std()
    with torch.no_grad():
        for i in range(n_samples):
            rain = float(g.uniform(20, 150))
            theta = torch.randn(K, device=dom.device) * 2.0
            D, _ = dig_map(theta, masks)
            if g.random() < 0.2:
                D = torch.zeros_like(D)
            hmax = dom.simulate(dom.z0 - D, rain_mm=rain)
            x = torch.stack([(dom.z0 - D - zmean) / zstd,
                             torch.full_like(dom.z0, rain / 100.0), D / 3.0])
            X.append(x.cpu())
            Y.append(hmax.unsqueeze(0).cpu())
            if (i + 1) % 20 == 0:
                log.info("simulated %d/%d", i + 1, n_samples)
    return torch.stack(X), torch.stack(Y)


def train_emulator(X, Y, epochs=40, device=None, save_path=None, flood_weight=20.0, tau=0.05):
    """Train the U-Net; returns (model, final_val_rmse_m).

    Flooding is sparse (most cells stay dry), so a plain full-grid MSE is minimised by predicting
    ~0 everywhere — the emulator collapses to "no flood" while the val RMSE still looks small. We
    therefore up-weight wet cells (weight 1 + flood_weight where target > tau) and clip gradients
    so the net actually fits the flood signal. We also report a flooded-cell RMSE, which exposes the
    collapse the whole-grid RMSE hides.
    """
    device = _device(device)
    emu = UNet().to(device)
    opt = torch.optim.Adam(emu.parameters(), lr=1e-3)
    ntr = int(0.9 * len(X))
    Xtr, Ytr = X[:ntr].to(device), Y[:ntr].to(device)
    Xv, Yv = X[ntr:].to(device), Y[ntr:].to(device)
    vl = torch.tensor(float("nan"))
    for ep in range(epochs):
        perm = torch.randperm(ntr)
        for b in range(0, ntr, 8):
            idx = perm[b:b + 8]
            pred, tgt = emu(Xtr[idx]), Ytr[idx]
            w = 1.0 + flood_weight * (tgt > tau).float()          # emphasise the sparse wet cells
            loss = (w * (pred - tgt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(emu.parameters(), 5.0)
            opt.step()
        with torch.no_grad():
            pv = emu(Xv)
            vl = ((pv - Yv) ** 2).mean().sqrt()
            wet = Yv > 0.15
            vwet = ((pv - Yv) ** 2)[wet].mean().sqrt() if bool(wet.any()) else torch.zeros(())
        if ep % 5 == 0:
            log.info("epoch %d: val RMSE %.1f cm (flooded-cell RMSE %.1f cm)",
                     ep, float(vl) * 100, float(vwet) * 100)
    if save_path:
        torch.save(emu.state_dict(), save_path)
        log.info("saved emulator -> %s", save_path)
    return emu, float(vl)


def train_twin(work=None, center=None, n_samples=None, epochs=40, device=None):
    """End-to-end nb05 training: build domain -> sites -> dataset -> train -> persist.

    Saves emulator.pt (weights) and twin_meta.pt (sites, z stats, crop offsets) so the serve
    layer can reconstruct inputs without re-running the simulator.
    """
    work = work or CFG.work
    from ..io import require_bundle
    require_bundle(work, ["dem.tif", "worldcover.tif"])
    dom = build_domain(work, center, device)
    sites, masks, site_area, eval_mask = candidate_sites(dom, work)
    log.info("domain %s | %d sites: %s", tuple(dom.z0.shape), len(sites), sites)
    X, Y = gen_dataset(dom, masks, n_samples)
    torch.save({"X": X, "Y": Y}, f"{work}/twin_dataset.pt")
    emu, rmse = train_emulator(X, Y, epochs, dom.device, save_path=f"{work}/emulator.pt")
    meta = dict(sites=sites, zmean=float(dom.z0.mean()), zstd=float(dom.z0.std()),
                row0=dom.row0, col0=dom.col0, dx=dom.dx, n_grid=dom.N, val_rmse_m=rmse,
                center=list(center or CFG.center))
    torch.save(meta, f"{work}/twin_meta.pt")
    log.info("nb05 done: emulator val RMSE %.1f cm -> %s/emulator.pt", rmse * 100, work)
    return meta
