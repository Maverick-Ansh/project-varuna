"""Train FloodGNN across areas, validating on held-out storms.

The flood target is sparse (a few % of streets are wet at 100 mm) — exactly the failure mode
that made the first Patna emulator predict zero everywhere. Same cure here: wet cells get a
×20 loss weight, gradients are clipped, and the reported metric is wet-RMSE / AUC, never the
full-graph MSE that hides a constant-dry model.

Validation splits by RAINFALL, not by node: every 4th storm of the sweep is held out, so
"val" asks the model to interpolate to storm sizes it never saw — the question the rainfall
slider actually poses. Cross-CITY generalization is a separate experiment (train with one
area held out entirely; see evaluate.transfer and the Colab runbook).

The saved checkpoint is the BEST-validation epoch (edge AUC, ties broken by wet-RMSE),
not the last one — cosine LR makes late epochs similar, but this caps any late overfit.

CPU-friendly (a full Patna epoch is seconds); on a T4 pass device='cuda' and amp=True
(fp16 — T4 has no bf16).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch

from .dataset import load_dataset
from .model import FloodGNN, save_checkpoint

log = logging.getLogger("varuna.gnn.train")

WET_TAU = 0.05        # m; a node/edge this deep counts as wet for loss weighting + metrics
WET_WEIGHT = 20.0
FLOW_LOSS_SCALE = 0.3


def _weighted_huber(pred, target, wet_weight=WET_WEIGHT, delta=0.25):
    w = 1.0 + (wet_weight - 1.0) * (target > WET_TAU).float()
    return (torch.nn.functional.huber_loss(pred, target, delta=delta, reduction="none") * w).mean()


def _epoch_batches(areas, rains_idx, flow_idx, rng):
    """Shuffled (area_key, kind, index) work items — one full-graph forward each."""
    items = [(k, "depth", i) for k in areas for i in rains_idx]
    items += [(k, "flow", i) for k in areas for i in flow_idx]
    rng.shuffle(items)
    return items


@torch.no_grad()
def validate(model, areas, val_idx, device):
    """Wet-RMSE (m) + flooded-edge AUC over the held-out storms, averaged across areas."""
    from .evaluate import auc as _auc
    model.eval()
    rmses, aucs = [], []
    for a in areas.values():
        sg, data = a["sg"], a["data"]
        for i in val_idx:
            out = model(sg.x, sg.edge_index, sg.edge_attr, float(data["rains"][i]))
            y = torch.as_tensor(data["edge_depth"][i], dtype=torch.float32, device=device)
            p = out["edge_depth"]
            wet = y > WET_TAU
            if wet.any():
                rmses.append(float(torch.sqrt(((p[wet] - y[wet]) ** 2).mean())))
            aucs.append(_auc(p.cpu().numpy(), (y > 0.15).cpu().numpy()))
    model.train()
    return (float(np.mean(rmses)) if rmses else float("nan"),
            float(np.nanmean(aucs)) if aucs else float("nan"))


def train_gnn(works, out="artifacts/gnn/gnn.pt", hidden=96, layers=4, dropout=0.0,
              epochs=60, lr=2e-3, weight_decay=1e-5, device=None, amp=False, seed=0,
              val_every=5, log_fn=None):
    """Train on the listed bundle dirs; returns (checkpoint path, history).

    works: list of bundle dirs that already have gnn_data.npz (dataset.build_dataset).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    say = log_fn or (lambda s: log.info("%s", s))

    areas = {}
    for w in works:
        sg, data = load_dataset(w)
        sg.to(device)
        areas[w] = {"sg": sg, "data": data}
        say(f"loaded {w}: {sg.n} nodes / {sg.e} edges")

    any_data = next(iter(areas.values()))["data"]
    n_rains = len(any_data["rains"])
    val_idx = list(range(2, n_rains, 4))                        # held-out storm sizes
    train_idx = [i for i in range(n_rains) if i not in val_idx]
    flow_idx = list(range(len(any_data["flow_rain"])))
    say(f"train storms {[float(any_data['rains'][i]) for i in train_idx]} | "
        f"val storms {[float(any_data['rains'][i]) for i in val_idx]}")

    model = FloodGNN(hidden=hidden, layers=layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device == "cuda")

    history = []
    best = dict(key=(float("-inf"), float("-inf")), epoch=0, auc=float("nan"),
                rmse=float("nan"), state=None)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        losses = []
        for key, kind, i in _epoch_batches(areas, train_idx, flow_idx, rng):
            sg, data = areas[key]["sg"], areas[key]["data"]
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=amp and device == "cuda"):
                if kind == "depth":
                    pred = model(sg.x, sg.edge_index, sg.edge_attr, float(data["rains"][i]))
                    yn = torch.as_tensor(data["node_depth"][i], dtype=torch.float32, device=device)
                    ye = torch.as_tensor(data["edge_depth"][i], dtype=torch.float32, device=device)
                    loss = _weighted_huber(pred["node_depth"], yn) \
                        + _weighted_huber(pred["edge_depth"], ye)
                else:
                    pred = model(sg.x, sg.edge_index, sg.edge_attr, float(data["flow_rain"][i]))
                    yf = torch.log1p(torch.as_tensor(data["node_flow"][i], dtype=torch.float32,
                                                     device=device))
                    loss = FLOW_LOSS_SCALE * torch.nn.functional.huber_loss(
                        pred["node_flow"], yf, delta=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        sched.step()
        if ep % val_every == 0 or ep == epochs:
            wet_rmse, edge_auc = validate(model, areas, val_idx, device)
            history.append(dict(epoch=ep, loss=round(float(np.mean(losses)), 4),
                                val_wet_rmse_m=round(wet_rmse, 4),
                                val_edge_auc=round(edge_auc, 4)))
            say(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  "
                f"val wet-RMSE {wet_rmse:.3f} m  edge-AUC {edge_auc:.3f}")
            key = (-1.0 if np.isnan(edge_auc) else edge_auc,
                   -np.nan_to_num(wet_rmse, nan=1e9))
            if key > best["key"]:
                best.update(key=key, epoch=ep, auc=edge_auc, rmse=wet_rmse,
                            state={k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()})

    if best["state"] is not None and best["epoch"] != epochs:
        model.load_state_dict(best["state"])
        say(f"kept best epoch {best['epoch']} (AUC {best['auc']:.3f}, "
            f"wet-RMSE {best['rmse']:.3f} m) over final epoch {epochs}")
    path = save_checkpoint(model, out, extra=dict(
        works=list(works), history=history, epochs=epochs, lr=lr, seed=seed,
        best_epoch=best["epoch"] or epochs,
        val_rains=[float(any_data["rains"][i]) for i in val_idx],
        train_seconds=round(time.perf_counter() - t0, 1)))
    say(f"checkpoint -> {path} ({sum(p.numel() for p in model.parameters()):,} params, "
        f"{time.perf_counter() - t0:.0f}s)")
    return path, history
