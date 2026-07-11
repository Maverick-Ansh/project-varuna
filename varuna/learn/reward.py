"""The reward gate: a candidate emulator ships only if it is measurably better and not worse.

Citizen reports carry no dry labels — a model that predicts deep water everywhere scores a
perfect wet-point error. The gate therefore requires ALL of:

1. enough evidence:      >= min_points held-out points from >= min_days distinct storm-days
2. better on people:     held-out band-tolerant point error improves by >= improve_frac
3. not worse on physics: replay-buffer wet-weighted RMSE degrades by <= replay_slack
4. robustly better:      candidate wins >= win_rate of bootstrap resamples of the held-out
                         point errors (protects against one lucky cluster)

Everything is seeded/deterministic; every decision is appended to learning_log.json, which
the dashboard surfaces — accepted or not, the loop is auditable.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os

import numpy as np
import torch

from ..io import load_json, save_json
from .finetune import (band_huber, load_emulator_net, load_replay, obs_batch, FLOOD_WEIGHT,
                       TAU)

log = logging.getLogger("varuna.learn.reward")


def _point_errors(net, terrain_ch, days, device="cpu"):
    """Per-point band-tolerant errors (numpy) over a label set."""
    X, points = obs_batch(terrain_ch, days)
    errs = []
    with torch.no_grad():
        pred = net(X.to(device))
        for i, pts in enumerate(points):
            for (r, c, depth_m, band) in pts:
                errs.append(float(band_huber(pred[i, 0, r, c], torch.tensor(depth_m), band)))
    return np.asarray(errs, dtype="float64")


def point_error(net, terrain_ch, days, device="cpu"):
    e = _point_errors(net, terrain_ch, days, device)
    return float(e.mean()) if len(e) else float("nan")


def replay_rmse(net, X, Y):
    with torch.no_grad():
        pred = net(X)
        w = 1.0 + FLOOD_WEIGHT * (Y > TAU).float()
        return float(((w * (pred - Y) ** 2).mean()).sqrt())


def gate(work, candidate_path, holdout_days, min_points=15, min_days=2, improve_frac=0.05,
         replay_slack=0.10, boot_n=200, win_rate=0.70, seed=0, device="cpu"):
    """Compare candidate vs serving emulator. Returns the decision record (does NOT deploy)."""
    entry = dict(date=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 accepted=False)
    n_points = sum(len(d["points"]) for d in holdout_days)
    entry["n_holdout_points"] = n_points
    entry["n_holdout_days"] = len(holdout_days)
    if n_points < min_points or len(holdout_days) < min_days:
        entry["reason"] = (f"insufficient data: {n_points} pts/{len(holdout_days)} days "
                           f"(need {min_points}/{min_days})")
        return entry

    Xr, Yr = load_replay(work, device)
    terrain = Xr[0, 0]
    old = load_emulator_net(work, device)
    new = load_emulator_net(work, device,
                            filename=os.path.basename(candidate_path))

    e_old = _point_errors(old, terrain, holdout_days, device)
    e_new = _point_errors(new, terrain, holdout_days, device)
    entry["point_err_before"] = round(float(e_old.mean()), 5)
    entry["point_err_after"] = round(float(e_new.mean()), 5)
    r_old, r_new = replay_rmse(old, Xr, Yr), replay_rmse(new, Xr, Yr)
    entry["replay_rmse_before"] = round(r_old, 5)
    entry["replay_rmse_after"] = round(r_new, 5)

    if e_old.mean() <= 0:
        entry["reason"] = "serving model already at zero held-out error"
        return entry
    if e_new.mean() > e_old.mean() * (1 - improve_frac):
        entry["reason"] = f"held-out point error did not improve >= {improve_frac:.0%}"
        return entry
    if r_new > r_old * (1 + replay_slack):
        entry["reason"] = (f"replay RMSE degraded {r_old:.4f}->{r_new:.4f} "
                           f"(> {replay_slack:.0%} slack) — all-wet guard")
        return entry

    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(boot_n):
        idx = rng.integers(0, len(e_old), len(e_old))
        wins += int(e_new[idx].mean() < e_old[idx].mean())
    entry["bootstrap_win_rate"] = round(wins / boot_n, 3)
    if entry["bootstrap_win_rate"] < win_rate:
        entry["reason"] = f"bootstrap win rate {entry['bootstrap_win_rate']} < {win_rate}"
        return entry

    entry["accepted"] = True
    entry["reason"] = "improved on held-out reports without degrading the simulated grid"
    return entry


def append_log(work, entry, cap=365):
    path = os.path.join(work, "learning_log.json")
    logl = load_json(path, default=[]) or []
    logl.append(entry)
    save_json(path, logl[-cap:])
    return path
