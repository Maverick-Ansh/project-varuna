"""Retrofit replay_buffer.pt (and optionally water_balance.json) onto already-built bundles.

New builds get both from train_twin / run_phase3_colab; the four pre-v2 bundles (patna x3,
bengaluru) need this one-time backfill. CPU-friendly at 128^2 (~10 s/sim).

    python scripts/make_replay.py --areas all            # replay buffers
    python scripts/make_replay.py --areas all --ladder   # + water-balance ladders
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("make_replay")


def replay_for(work, k=16, seed=0):
    """Re-simulate k seeded storms (no dig — the anchor is the base terrain) -> replay_buffer.pt."""
    import numpy as np
    import torch
    from varuna.build.twin import build_domain, save_replay

    dom = build_domain(work)
    g = np.random.default_rng(seed)
    torch.manual_seed(seed)
    rains = np.linspace(20, 150, k) + g.uniform(-3, 3, k)
    zmean, zstd = dom.z0.mean(), dom.z0.std()
    X, Y = [], []
    with torch.no_grad():
        for i, rain in enumerate(rains):
            hmax = dom.simulate(dom.z0, rain_mm=float(rain))
            D = torch.zeros_like(dom.z0)
            x = torch.stack([(dom.z0 - D - zmean) / zstd,
                             torch.full_like(dom.z0, float(rain) / 100.0), D / 3.0])
            X.append(x.cpu())
            Y.append(hmax.unsqueeze(0).cpu())
            log.info("%s: replay sim %d/%d (%.0f mm)", work, i + 1, k, rain)
    save_replay(work, torch.stack(X), torch.stack(Y), k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="+", default=["all"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ladder", action="store_true", help="also build water_balance.json")
    ap.add_argument("--force", action="store_true", help="rebuild even if files exist")
    args = ap.parse_args()

    from varuna.areas import list_areas, get_area, is_built
    ids = ([a.id for a in list_areas() if is_built(a.id)]
           if args.areas == ["all"] else args.areas)
    for aid in ids:
        work = get_area(aid).work_dir()
        if args.force or not os.path.exists(os.path.join(work, "replay_buffer.pt")):
            replay_for(work, k=args.k)
        else:
            log.info("%s: replay_buffer.pt exists — skip", aid)
        if args.ladder:
            if args.force or not os.path.exists(os.path.join(work, "water_balance.json")):
                from varuna.serve.waterbalance import build_ladder
                build_ladder(work=work)
            else:
                log.info("%s: water_balance.json exists — skip", aid)
    log.info("done: %s", ids)


if __name__ == "__main__":
    main()
