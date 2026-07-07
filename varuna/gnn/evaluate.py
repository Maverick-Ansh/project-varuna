"""Evaluation + the cross-city transfer experiment — the numbers the paper section needs.

Per area (against held-out storms, labels = the emulator oracle):
  edge_auc        — ranking quality: does the GNN order streets by flood risk correctly?
  wet_rmse_m      — depth accuracy where it matters (truly wet edges only; a constant-dry
                    model scores NaN here, not a flattering 0).
  flow_spearman   — rank correlation of the flow head vs the drain planner's accumulation
                    (does the model know where the trunk drains belong?).
  routing         — the end task: random origin->destination pairs routed three ways
                    (shortest / GNN-risk-aware / oracle-risk-aware). Reported: how much wet
                    street each policy crosses and the detour it pays. GNN ≈ oracle with a
                    fraction of the machinery is the win condition.
  speed           — ms per full-graph risk query, GNN vs emulator+sampling.

`transfer` wraps the question the paper actually asks: does street-flood knowledge learned
on flat Patna work on hilly Bengaluru (and back)? Train with a city held out entirely, then
evaluate on it zero-shot — the learned analogue of the DEM-uncertainty co-location finding.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import torch

from .dataset import load_dataset
from .planner import FLOOD_TAU, _dijkstra

log = logging.getLogger("varuna.gnn.evaluate")


def auc(scores, labels):
    """Rank-based ROC AUC, pure numpy (no sklearn). NaN if labels are one-class."""
    scores = np.asarray(scores, dtype="float64")
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # midranks (ties averaged) so equal scores don't fake discrimination
    _uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = (cum - (counts - 1) / 2.0)
    ranks = avg_rank[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _routing_eval(sg, depth_true, depth_gnn, n_routes=150, seed=0, min_m=1500.0):
    """Route random OD pairs by three policies; report wet exposure + detour of each."""
    rng = np.random.default_rng(seed)
    ei = sg.edge_index.cpu().numpy()
    lengths = sg.edge_attr[:, 0].cpu().numpy() * 1000.0

    def adjacency(depth):
        from .planner import AVOID_DEPTH, RISK_WEIGHT
        mult = (np.ones_like(lengths) if depth is None else
                1.0 + RISK_WEIGHT * np.minimum(depth, 1.5) + 25.0 * (depth > AVOID_DEPTH))
        adj = [[] for _ in range(sg.n)]
        for k in range(ei.shape[1]):
            adj[int(ei[0, k])].append((int(ei[1, k]), float(lengths[k] * mult[k]), k))
        return adj

    adjs = {"shortest": adjacency(None), "gnn": adjacency(depth_gnn),
            "oracle": adjacency(depth_true)}
    coslat = np.cos(np.radians(float(np.mean(sg.lat))))
    stats = {k: dict(wet_m=0.0, len_m=0.0, flooded_routes=0) for k in adjs}
    done = 0
    for _ in range(n_routes * 4):
        if done >= n_routes:
            break
        s, t = int(rng.integers(sg.n)), int(rng.integers(sg.n))
        sep = np.hypot(sg.lat[s] - sg.lat[t], (sg.lon[s] - sg.lon[t]) * coslat) * 111320.0
        if s == t or sep < min_m:
            continue
        paths = {k: _dijkstra(sg.n, adj, s, t) for k, adj in adjs.items()}
        if any(p[0] is None for p in paths.values()):
            continue
        done += 1
        for k, (_p, eids) in paths.items():
            d = depth_true[eids]                       # judged on the ORACLE depths, always
            stats[k]["wet_m"] += float(lengths[eids][d > FLOOD_TAU].sum())
            stats[k]["len_m"] += float(lengths[eids].sum())
            stats[k]["flooded_routes"] += int((d > FLOOD_TAU).any())
    if done == 0:
        return dict(n_routes=0)
    out = dict(n_routes=done)
    for k, s_ in stats.items():
        out[k] = dict(wet_m_per_route=round(s_["wet_m"] / done),
                      pct_routes_hitting_water=round(100.0 * s_["flooded_routes"] / done, 1),
                      mean_length_m=round(s_["len_m"] / done))
    base = out["shortest"]["mean_length_m"]
    for k in ("gnn", "oracle"):
        out[k]["detour_pct"] = round(100.0 * (out[k]["mean_length_m"] / max(base, 1) - 1.0), 1)
    return out


def evaluate_model(checkpoint, works, n_routes=150, seed=0, device="cpu", out=None):
    """Full report for one checkpoint over the given bundles -> dict (optionally saved)."""
    from scipy.stats import spearmanr
    from .model import load_checkpoint

    model, ckpt = load_checkpoint(checkpoint, device=device)
    val_rains = ckpt.get("val_rains") or [60.0, 140.0, 220.0]
    report = dict(checkpoint=checkpoint, val_rains=[float(r) for r in val_rains],
                  trained_on=ckpt.get("works", []), areas={})

    for w in works:
        sg, data = load_dataset(w)
        sg.to(device)
        rains = [float(r) for r in data["rains"]]
        idx = [rains.index(r) for r in val_rains if r in rains] or list(range(2, len(rains), 4))

        aucs, rmses, t_gnn = [], [], []
        depth_pairs = {}
        for i in idx:
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = model(sg.x, sg.edge_index, sg.edge_attr, rains[i])
            t_gnn.append(time.perf_counter() - t0)
            p = pred["edge_depth"].cpu().numpy().astype("float64")
            y = data["edge_depth"][i].astype("float64")
            depth_pairs[rains[i]] = (y, p)
            aucs.append(auc(p, y > FLOOD_TAU))
            wet = y > FLOOD_TAU
            if wet.any():
                rmses.append(float(np.sqrt(((p[wet] - y[wet]) ** 2).mean())))

        with torch.no_grad():
            fpred = model(sg.x, sg.edge_index, sg.edge_attr, float(data["flow_rain"][1]
                          if len(data["flow_rain"]) > 1 else data["flow_rain"][0]))
        yflow = np.log1p(data["node_flow"][1 if len(data["node_flow"]) > 1 else 0])
        rho = float(spearmanr(fpred["node_flow"].cpu().numpy(), yflow).statistic) \
            if yflow.max() > 0 else float("nan")

        # emulator timing = raster forward + per-edge sampling (the pipeline GNN replaces)
        from ..serve.emulator import whatif_grid
        from .dataset import sample_depths
        t0 = time.perf_counter()
        hmax, _d, _s = whatif_grid(100.0, None, work=w, device="cpu")
        sample_depths(hmax, sg)
        t_emu = time.perf_counter() - t0

        r_mid = val_rains[len(val_rains) // 2]
        y_true, y_gnn = depth_pairs.get(r_mid, next(iter(depth_pairs.values())))
        routing = _routing_eval(sg, y_true, y_gnn, n_routes=n_routes, seed=seed)

        report["areas"][os.path.basename(os.path.normpath(w))] = dict(
            n_nodes=sg.n, n_edges=sg.e,
            edge_auc=round(float(np.nanmean(aucs)), 4),
            wet_rmse_m=round(float(np.mean(rmses)), 3) if rmses else None,
            flow_spearman=round(rho, 3) if np.isfinite(rho) else None,
            ms_per_query_gnn=round(1000.0 * float(np.mean(t_gnn)), 1),
            ms_per_query_emulator=round(1000.0 * t_emu, 1),
            routing=routing)
        log.info("eval %-12s AUC %.3f wet-RMSE %s flow-rho %s",
                 w, report["areas"][os.path.basename(os.path.normpath(w))]["edge_auc"],
                 report["areas"][os.path.basename(os.path.normpath(w))]["wet_rmse_m"],
                 report["areas"][os.path.basename(os.path.normpath(w))]["flow_spearman"])

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
        log.info("report -> %s", out)
    return report


def transfer(all_works, holdout_work, out_dir="artifacts/gnn", seed=0, device=None, **train_kw):
    """Train with `holdout_work` excluded, then evaluate on it zero-shot (+ on the train set).

    Returns (checkpoint path, report). The Colab driver calls this twice — once holding out
    Bengaluru (flat->hilly) and once holding out Patna (hilly->flat).
    """
    from .train import train_gnn
    name = os.path.basename(os.path.normpath(holdout_work))
    train_works = [w for w in all_works if w != holdout_work]
    ckpt, _hist = train_gnn(train_works, out=os.path.join(out_dir, f"gnn_no_{name}.pt"),
                            seed=seed, device=device, **train_kw)
    report = evaluate_model(ckpt, all_works, seed=seed, device=device or "cpu",
                            out=os.path.join(out_dir, f"transfer_no_{name}.json"))
    report["holdout"] = name
    return ckpt, report
