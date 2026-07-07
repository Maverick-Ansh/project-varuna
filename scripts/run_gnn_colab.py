"""Colab/Kaggle turnkey for the FloodGNN path-planning stack (see RUNBOOK_GNN.md).

Jobs (combine flags freely; each is restartable and idempotent):

  --data       build gnn_data.npz (emulator depth sweeps + drain-planner flow labels) for
               every built area with a cached road graph (or --areas ...). CPU-fine, no EE.
  --train      train the production checkpoint on all areas with data -> artifacts/gnn/gnn.pt
  --ablation   train the no-graph (layers=0) and shallow (layers=2) baselines and evaluate
               them next to the production model -> artifacts/gnn/ablation.json
  --transfer   the paper experiment: hold one city out entirely, evaluate on it zero-shot.
               Runs bengaluru-holdout (flat->hilly) and patna-holdout (hilly->flat).
  --eval       evaluate the production checkpoint on every area -> artifacts/gnn/gnn_report.json
  --figures    paper/dashboard figures -> artifacts/gnn/figures/ (train curve, AUC vs rain,
               predicted-vs-true, a routing demo map)
  --push       commit gnn_data.npz per bundle + everything under artifacts/gnn and push.
               GITHUB_TOKEN from env / Colab secret / Kaggle secret — never printed.

GPU: pass --device cuda on a T4; fp16 autocast is ON by default there (T4 has no bf16 —
use --no-amp to opt out). Everything also runs on CPU, just slower (Patna epoch ~15 s).

Colab, end to end:

    !git clone -b feature/gnn-path-planning https://github.com/Maverick-Ansh/project-varuna.git
    %cd project-varuna
    !pip -q install rasterio
    !python scripts/run_gnn_colab.py --data --train --ablation --transfer --eval --figures --device cuda
    !python scripts/run_gnn_colab.py --push --branch feature/gnn-path-planning
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("run_gnn")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GNN_DIR = os.path.join(REPO_ROOT, "artifacts", "gnn")
CKPT = os.path.join(GNN_DIR, "gnn.pt")


def gnn_areas(only=None, need_data=False):
    """Bundle dirs that can play: built + road graph (+ gnn_data.npz when need_data)."""
    from varuna.areas import list_areas, is_built
    out = []
    for a in list_areas():
        if only and a.id not in only:
            continue
        w = a.work_dir()
        if not is_built(a.id):
            log.warning("area '%s' has no built bundle — skipped", a.id)
            continue
        if not os.path.exists(os.path.join(w, "road_graph.json.gz")):
            log.warning("area '%s' has no road_graph.json.gz — skipped "
                        "(scripts/build_road_graphs.py)", a.id)
            continue
        if need_data and not os.path.exists(os.path.join(w, "gnn_data.npz")):
            log.warning("area '%s' has no gnn_data.npz — run --data first", a.id)
            continue
        out.append((a.id, w))
    if not out:
        raise SystemExit("no eligible areas")
    return out


# ------------------------------------------------------------------ jobs

def do_data(areas, device):
    from varuna.gnn.dataset import build_dataset
    for aid, w in areas:
        log.info("=== dataset for %s ===", aid)
        s = build_dataset(work=w, device=device)
        log.info("%s: %d nodes / %d edges, wet@100mm %.1f%%, %.0fs",
                 aid, s["n_nodes"], s["n_edges"], 100 * s["wet_node_frac_100mm"], s["seconds"])


def do_train(areas, args):
    from varuna.gnn.train import train_gnn
    works = [w for _aid, w in areas]
    path, hist = train_gnn(works, out=CKPT, hidden=args.hidden, layers=args.layers,
                           epochs=args.epochs, lr=args.lr, device=args.device,
                           amp=args.amp, seed=args.seed)
    log.info("production checkpoint: %s (final %s)", path, hist[-1] if hist else "?")


def do_ablation(areas, args):
    """Does message passing earn its keep? layers 0 (plain MLP) and 2 vs the production K."""
    from varuna.gnn.train import train_gnn
    from varuna.gnn.evaluate import evaluate_model
    works = [w for _aid, w in areas]
    out = {}
    for L in sorted({0, 2, args.layers}):
        ck = CKPT if L == args.layers and os.path.exists(CKPT) else None
        if ck is None:
            ck, _ = train_gnn(works, out=os.path.join(GNN_DIR, f"gnn_L{L}.pt"),
                              hidden=args.hidden, layers=L, epochs=args.epochs, lr=args.lr,
                              device=args.device, amp=args.amp, seed=args.seed)
        rep = evaluate_model(ck, works, seed=args.seed, device=args.device or "cpu")
        out[f"layers_{L}"] = {a: dict(edge_auc=v["edge_auc"], wet_rmse_m=v["wet_rmse_m"])
                              for a, v in rep["areas"].items()}
        log.info("ablation layers=%d: %s", L, out[f"layers_{L}"])
    _save(os.path.join(GNN_DIR, "ablation.json"), out)


def do_transfer(areas, args):
    from varuna.gnn.evaluate import transfer
    works = {aid: w for aid, w in areas}
    all_works = list(works.values())
    results = {}
    for hold in ("bengaluru", "patna"):
        if hold not in works:
            log.warning("transfer: '%s' not available — skipped", hold)
            continue
        log.info("=== transfer: hold out %s ===", hold)
        _ck, rep = transfer(all_works, works[hold], out_dir=GNN_DIR, seed=args.seed,
                            device=args.device, hidden=args.hidden, layers=args.layers,
                            epochs=args.epochs, lr=args.lr, amp=args.amp)
        results[hold] = {a: v["edge_auc"] for a, v in rep["areas"].items()}
        log.info("zero-shot on %s: AUC %s", hold, results[hold].get(hold))
    _save(os.path.join(GNN_DIR, "transfer_summary.json"), results)


def do_eval(areas, args):
    from varuna.gnn.evaluate import evaluate_model
    if not os.path.exists(CKPT):
        raise SystemExit(f"no {CKPT} — run --train first")
    evaluate_model(CKPT, [w for _aid, w in areas], seed=args.seed,
                   device=args.device or "cpu", out=os.path.join(GNN_DIR, "gnn_report.json"))


def do_figures(areas, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from varuna.gnn.dataset import load_dataset
    from varuna.gnn.evaluate import auc as _auc
    from varuna.gnn.model import load_checkpoint

    figdir = os.path.join(GNN_DIR, "figures")
    os.makedirs(figdir, exist_ok=True)
    model, ckpt = load_checkpoint(CKPT, device="cpu")

    hist = ckpt.get("history") or []
    if hist:
        fig, ax1 = plt.subplots(figsize=(6, 3.4))
        ep = [h["epoch"] for h in hist]
        ax1.plot(ep, [h["loss"] for h in hist], "-o", ms=3, label="train loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(ep, [h["val_edge_auc"] for h in hist], "-s", ms=3, color="tab:green",
                 label="val edge AUC")
        ax2.set_ylabel("held-out storm edge AUC")
        fig.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{figdir}/train_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    val = set(float(r) for r in (ckpt.get("val_rains") or []))
    for aid, w in areas:
        sg, data = load_dataset(w)
        rains = [float(r) for r in data["rains"]]
        aucs = []
        for i, r in enumerate(rains):
            with torch.no_grad():
                p = model(sg.x, sg.edge_index, sg.edge_attr, r)["edge_depth"].numpy()
            aucs.append(_auc(p, data["edge_depth"][i].astype("float64") > 0.15))
        ax.plot(rains, aucs, "-o", ms=3, label=aid)
    for r in sorted(val):
        ax.axvline(r, color="0.85", lw=1, zorder=0)
    ax.set_xlabel("storm (mm / 24 h) — grey lines were held out of training")
    ax.set_ylabel("flooded-street ranking AUC"); ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{figdir}/auc_vs_rain.png", dpi=150); plt.close(fig)

    aid, w = areas[0]
    sg, data = load_dataset(w)
    i = int(np.argmin(np.abs(np.asarray([float(r) for r in data["rains"]]) - 100.0)))
    with torch.no_grad():
        p = model(sg.x, sg.edge_index, sg.edge_attr, float(data["rains"][i]))["edge_depth"].numpy()
    y = data["edge_depth"][i].astype("float64")
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.hexbin(y, p, gridsize=40, bins="log", cmap="Blues")
    lim = max(1.0, float(y.max()) + 0.1)
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel(f"emulator depth (m), {aid} @ {float(data['rains'][i]):.0f} mm")
    ax.set_ylabel("GNN depth (m)")
    fig.tight_layout(); fig.savefig(f"{figdir}/pred_vs_true.png", dpi=150); plt.close(fig)

    _route_figure(aid, w, figdir)
    log.info("figures -> %s", figdir)


def _route_figure(aid, work, figdir, rain=120.0):
    """Streets grey, predicted-wet streets blue-scaled, safe vs shortest route overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from varuna.gnn.graph import build_graph
    from varuna.gnn.planner import edge_depths, safe_route

    sg = build_graph(work)
    depth, backend = edge_depths(work, rain)
    ei = sg.edge_index.cpu().numpy()
    und = ei[0] < ei[1]
    lat0, lat1 = np.percentile(sg.lat, [35, 65])
    lon0, lon1 = np.percentile(sg.lon, [35, 65])
    r = safe_route([float(lat1), float(lon0)], [float(lat0), float(lon1)],
                   rain_mm=rain, work=work)

    fig, ax = plt.subplots(figsize=(6.4, 6))
    dry = und & (depth <= 0.15)
    for m, color, lw, alpha in ((dry, "0.75", 0.4, 0.6),):
        for k in np.where(m)[0]:
            ax.plot([sg.lon[ei[0, k]], sg.lon[ei[1, k]]], [sg.lat[ei[0, k]], sg.lat[ei[1, k]]],
                    color=color, lw=lw, alpha=alpha, zorder=1)
    wet = np.where(und & (depth > 0.15))[0]
    dmax = max(float(depth[wet].max()), 0.5) if len(wet) else 1.0
    cmap = plt.get_cmap("Blues")
    for k in wet:
        ax.plot([sg.lon[ei[0, k]], sg.lon[ei[1, k]]], [sg.lat[ei[0, k]], sg.lat[ei[1, k]]],
                color=cmap(0.35 + 0.65 * min(depth[k] / dmax, 1.0)), lw=1.4, zorder=2)
    sp = np.asarray(r["shortest"]["path_latlon"])
    rp = np.asarray(r["route"]["path_latlon"])
    ax.plot(sp[:, 1], sp[:, 0], "--", color="#64748b", lw=1.6, zorder=3, label="shortest")
    ax.plot(rp[:, 1], rp[:, 0], color="#1d4ed8", lw=2.4, zorder=4,
            label=f"flood-safe (+{r['detour_pct']}%)")
    ax.plot(*rp[0][::-1], "o", color="#1d4ed8", mfc="white", ms=8, zorder=5)
    ax.plot(*rp[-1][::-1], "o", color="#1d4ed8", ms=8, zorder=5)
    ax.set_title(f"{aid} @ {rain:.0f} mm — streets flooded per {backend} risk")
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(sg.lat)))))
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{figdir}/route_demo_{aid}.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ push (phase-3 pattern)

def _github_token():
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok.strip()
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN").strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN").strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _git(*args, token=None, check=True):
    r = subprocess.run(["git", "-C", REPO_ROOT, *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if token:
        out = out.replace(token, "***")
    if r.returncode != 0 and check:
        raise RuntimeError(f"git {args[0]} failed: {out}")
    return out


def do_push(message, branch=None):
    tok = _github_token()
    if not tok:
        raise SystemExit("no GITHUB_TOKEN (env var / Colab secret / Kaggle secret). "
                         "On Kaggle, attach the secret to THIS notebook (Add-ons -> Secrets).")
    if not _git("config", "user.email", check=False):
        _git("config", "user.email", "colab@varuna.local")
        _git("config", "user.name", "Varuna Colab")
    from varuna.areas import list_areas, is_built
    for a in list_areas():
        p = os.path.join(a.work_dir(), "gnn_data.npz")
        if is_built(a.id) and os.path.exists(p):
            _git("add", "-f", os.path.relpath(p, REPO_ROOT))
    if os.path.isdir(GNN_DIR):
        _git("add", "-f", os.path.relpath(GNN_DIR, REPO_ROOT))
    if not _git("status", "--porcelain", check=False):
        log.info("nothing to push")
        return
    _git("commit", "-m", message)
    remote = _git("remote", "get-url", "origin")
    slug = remote.split("github.com")[-1].lstrip(":/").removesuffix(".git")
    url = f"https://x-access-token:{tok}@github.com/{slug}.git"
    branch = branch or _git("rev-parse", "--abbrev-ref", "HEAD")
    out = _git("push", url, f"HEAD:{branch}", token=tok)
    log.info("pushed -> %s: %s", branch, out or "ok")


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    log.info("-> %s", path)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Varuna FloodGNN (Colab/Kaggle)")
    for flag, help_ in (("--data", "build gnn_data.npz per area"),
                        ("--train", "train the production checkpoint"),
                        ("--ablation", "layers-0/2 baselines + eval"),
                        ("--transfer", "hold-one-city-out experiments"),
                        ("--eval", "evaluate production checkpoint"),
                        ("--figures", "paper figures"),
                        ("--push", "commit + push gnn artifacts")):
        ap.add_argument(flag, action="store_true", help=help_)
    ap.add_argument("--areas", nargs="+", default=None, help="restrict to these area ids")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="disable fp16 autocast on cuda")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--message", default="FloodGNN: datasets + checkpoints + reports from Colab")
    ap.add_argument("--branch", default=None)
    args = ap.parse_args()

    import torch
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = args.amp and args.device == "cuda"
    log.info("device=%s amp=%s", args.device, args.amp)

    if not any((args.data, args.train, args.ablation, args.transfer, args.eval,
                args.figures, args.push)):
        raise SystemExit("nothing to do — pass --data/--train/--ablation/--transfer/"
                         "--eval/--figures/--push")
    if args.data:
        do_data(gnn_areas(args.areas), args.device)
    if args.train:
        do_train(gnn_areas(args.areas, need_data=True), args)
    if args.ablation:
        do_ablation(gnn_areas(args.areas, need_data=True), args)
    if args.transfer:
        do_transfer(gnn_areas(args.areas, need_data=True), args)
    if args.eval:
        do_eval(gnn_areas(args.areas, need_data=True), args)
    if args.figures:
        do_figures(gnn_areas(args.areas, need_data=True), args)
    if args.push:
        do_push(args.message, branch=args.branch)


if __name__ == "__main__":
    sys.path.insert(0, REPO_ROOT)
    main()
