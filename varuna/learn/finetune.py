"""Fine-tune the U-Net emulator on citizen report points, anchored by the replay buffer.

Anchor + band tolerance are what keep this honest:
- the replay buffer (committed sim pairs spanning the rain range) appears in every batch with
  the original wet-weighted MSE, so a handful of wet points cannot drag the whole grid;
- report depths are coarse ("knee-deep"), so the observation loss is a Huber on the error
  OUTSIDE the band's half-width (ankle ±0.15 m ... chest ±0.35 m) — a prediction inside the
  band is already correct and contributes zero gradient.

Never touches emulator.pt: writes {work}/emulator_candidate.pt and lets the reward gate
decide (varuna.learn.reward).
"""
from __future__ import annotations

import logging
import os

import torch

from ..serve.reports import DEPTH_TOL_M

log = logging.getLogger("varuna.learn.finetune")

FLOOD_WEIGHT = 20.0          # same recipe as build.twin.train_emulator
TAU = 0.05


def band_huber(pred, depth_m, band, delta=0.5):
    """Huber on the error outside the report band's tolerance half-width."""
    tol = DEPTH_TOL_M.get(band, 0.2)
    err = torch.clamp(torch.abs(pred - depth_m) - tol, min=0.0)
    quad = torch.clamp(err, max=delta)
    return 0.5 * quad ** 2 + delta * (err - quad)


def load_replay(work, device="cpu"):
    buf = torch.load(os.path.join(work, "replay_buffer.pt"), map_location=device,
                     weights_only=False)
    return buf["X"].float(), buf["Y"].float()


def load_emulator_net(work, device="cpu", filename="emulator.pt"):
    from ..build.twin import UNet
    net = UNet().to(device)
    net.load_state_dict(torch.load(os.path.join(work, filename), map_location=device,
                                   weights_only=False))
    return net


def obs_batch(terrain_ch, days):
    """Report days -> stacked emulator inputs [D,3,N,N] + point index lists.

    terrain_ch: the normalized-elevation channel (replay X[0,0] — identical across replay
    samples because dig=0 there), so no rasterio is needed at learn time.
    """
    xs, points = [], []
    n = terrain_ch.shape[-1]
    zeros = torch.zeros_like(terrain_ch)
    for d in days:
        rain = torch.full_like(terrain_ch, float(d["rain_mm"]) / 100.0)
        xs.append(torch.stack([terrain_ch, rain, zeros]))
        points.append([(r, c, m, b) for (r, c, m, b) in d["points"]
                       if 0 <= r < n and 0 <= c < n])
    return torch.stack(xs), points


def replay_loss(net, X, Y):
    pred = net(X)
    w = 1.0 + FLOOD_WEIGHT * (Y > TAU).float()
    return (w * (pred - Y) ** 2).mean()


def obs_loss(net, X_obs, points):
    pred = net(X_obs)
    losses = []
    for i, pts in enumerate(points):
        for (r, c, depth_m, band) in pts:
            losses.append(band_huber(pred[i, 0, r, c], depth_m, band))
    if not losses:
        return torch.zeros((), device=X_obs.device)
    return torch.stack(losses).mean()


def finetune_emulator(work, train_days, epochs=15, lr=1e-4, obs_weight=3.0, device="cpu",
                      seed=0):
    """Train a candidate on report points + replay anchor. Returns candidate path."""
    torch.manual_seed(seed)
    Xr, Yr = load_replay(work, device)
    net = load_emulator_net(work, device)
    X_obs, points = obs_batch(Xr[0, 0], train_days)
    X_obs = X_obs.to(device)
    n_pts = sum(len(p) for p in points)
    if n_pts == 0:
        raise ValueError("no in-domain report points to train on")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        lr_ = replay_loss(net, Xr, Yr)
        lo = obs_loss(net, X_obs, points)
        loss = lr_ + obs_weight * lo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            log.info("finetune ep %d: replay %.5f obs %.5f (%d pts, %d days)",
                     ep, float(lr_.detach()), float(lo.detach()), n_pts, len(train_days))
    path = os.path.join(work, "emulator_candidate.pt")
    torch.save(net.state_dict(), path)
    return path
