"""Train and evaluate the flood-extent net against every baseline that makes CSI interpretable.

  python -m csi_net.run --split lodo --width 64 --steps 3000
  python -m csi_net.run --split flood --ablate rain      # does the net use rainfall at all?
"""
import argparse, json, os, time
import numpy as np
import torch

from .data import VarunaData, N_RAIN
from .model import UNet, masked_loss, count_params
from .metrics import (csi_report, sweep_threshold, csi_allwet, csi_random,
                      persistent_field, storm_increment)

HERE = os.path.dirname(os.path.abspath(__file__))


def dihedral(t, g):
    """Element g of the 8-fold symmetry group, applied to the trailing 2 axes.

    Safe here: gravity is vertical, and no channel encodes a compass direction (slope is a
    magnitude, curvature is a Laplacian). This is NOT the banned SAR flip - the features and
    the truth are transformed together, so their alignment is preserved.
    """
    if g & 4:
        t = torch.flip(t, [-1])
    return torch.rot90(t, g & 3, (-2, -1))


def ablation_mask(names, kind, n_static):
    """Zero-out mask over input channels. Ablating an input is how we find out what the net used."""
    m = np.ones(n_static + N_RAIN, dtype="float32")
    if kind == "rain":
        m[n_static:] = 0.0
    elif kind == "persist":                      # the persistent-water prior
        for i, n in enumerate(names):
            if n in ("jrc_occurrence", "log_dist_perm_water"):
                m[i] = 0.0
    elif kind == "terrain":
        m[:n_static] = 0.0
    elif kind != "none":
        raise ValueError(kind)
    return m


def fold_targets(D, train, target):
    """Per-scene (truth, valid) under the requested target.

    mask       - the full Sentinel-1 water mask. Dominated by storm-independent water:
                 tidal flat, mangrove, seasonal pond, paddy. What the net was trained on so far.
    increment  - wet today AND not persistently wet. This is the quantity the physics twin
                 actually predicts, so training on it is the fairest possible head-to-head,
                 and it is the only target for which "did the storm do this" is the question.

    The persistent field for a scene is built from the OTHER TRAINING scenes of its domain, never
    from the held-out scene, so switching target cannot leak the test label into training.
    """
    if target == "mask":
        return None
    by_area = {}
    for s in train:
        by_area.setdefault(s["area"], []).append(s["idx"])
    out = {}
    for s in train:
        others = [D.Y[s["area"]][i] for i in by_area[s["area"]] if i != s["idx"]]
        pers = (np.mean(others, 0) >= 0.5 if others
                else np.zeros_like(D.Y[s["area"]][s["idx"]], bool))
        out[(s["area"], s["idx"])] = (D.Y[s["area"]][s["idx"]] & ~pers,
                                      D.valid[s["area"]] & ~pers)
    return out


def calibrated_threshold(prob, valid, wet_frac):
    """Per-scene threshold that makes the predicted wet fraction match a prior wet fraction.

    Ranking quality has run consistently above achieved CSI in every experiment here, which is
    calibration loss: the model orders cells better than one global threshold can exploit.
    `wet_frac` is the mean observed wet fraction of the TRAINING scenes - a number available
    without ever looking at the test label - so this is a free correction, not a peek.
    """
    v = prob[valid]
    if v.size == 0:
        return 0.5
    return float(np.quantile(v, 1.0 - min(max(wet_frac, 1e-6), 1.0)))


def train_fold(D, train, stats, args, dev, chan_mask, targets=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net = UNet(D.n_in, args.width, args.depth, args.dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.15)
    # T4 is compute-capability 7.5: fp16 tensor cores, no real bf16. fp16 + GradScaler.
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    cm = torch.as_tensor(chan_mask, device=dev)[None, :, None, None]

    cache = []
    for s in train:
        x, y, v = D.tensor(s, stats)
        if targets is not None:
            y, v = targets[(s["area"], s["idx"])]
        cache.append((x, y, v))
    rng = np.random.default_rng(args.seed)
    net.train()
    for step in range(args.steps):
        xs, ys, vs = [], [], []
        for _ in range(args.batch):
            x, y, v = cache[rng.integers(len(cache))]
            H, W = y.shape
            c = min(args.crop, H, W)
            i = rng.integers(0, H - c + 1)
            j = rng.integers(0, W - c + 1)
            xs.append(x[:, i:i + c, j:j + c])
            ys.append(y[i:i + c, j:j + c])
            vs.append(v[i:i + c, j:j + c])
        x = torch.as_tensor(np.stack(xs), device=dev)
        y = torch.as_tensor(np.stack(ys), device=dev).float()
        v = torch.as_tensor(np.stack(vs), device=dev).float()
        g = int(rng.integers(8))
        x, y, v = dihedral(x, g), dihedral(y, g), dihedral(v, g)
        x = x * cm
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            loss = masked_loss(net(x), y, v, args.dice_w, args.bce_w, args.pos_weight)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if args.verbose and (step % max(1, args.steps // 6) == 0 or step == args.steps - 1):
            print(f"    step {step:>5}/{args.steps}  loss {loss.item():.4f}"
                  f"  lr {sched.get_last_lr()[0]:.2e}", flush=True)
    return net


@torch.no_grad()
def predict(net, D, s, stats, dev, chan_mask):
    x, y, v = D.tensor(s, stats)
    cm = torch.as_tensor(chan_mask, device=dev)[None, :, None, None]
    xt = torch.as_tensor(x[None], device=dev) * cm
    net.eval()
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
        p = torch.sigmoid(net(xt).float())[0].cpu().numpy()
    return p, y, v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lodo", choices=["loso", "lodo", "flood"])
    ap.add_argument("--grid", type=int, default=60, choices=[30, 60, 120])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--bce_w", type=float, default=0.5)
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--ablate", default="none", choices=["none", "rain", "persist", "terrain"])
    ap.add_argument("--target", default="mask", choices=["mask", "increment"],
                    help="mask: full SAR water. increment: wet today and not persistently wet "
                         "- the quantity the physics twin predicts.")
    ap.add_argument("--norm", default="per_domain", choices=["per_domain", "global"])
    ap.add_argument("--areas", default="", help="comma-separated areas to KEEP")
    ap.add_argument("--shuffle_rain", type=int, default=0,
                    help="permutation control: shuffle rain vectors within each area")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = VarunaData(os.path.join(HERE, "data", "varuna_stack.npz"),
                   os.path.join(HERE, "data", "rain_features.json"), grid_m=args.grid,
                   shuffle_rain=args.shuffle_rain,
                   keep_areas=[a for a in args.areas.split(',') if a] or None)
    print(f"grid {args.grid} m | {len(D.samples)} storms | {D.n_static} static + {N_RAIN} rain "
          f"= {D.n_in} channels | split={args.split} ablate={args.ablate}")
    cmask = ablation_mask(D.feature_names, args.ablate, D.n_static)

    rows, t0 = [], time.time()
    for fold in range(D.n_folds(args.split)):
        train, test = D.split(args.split, fold)
        stats = D.stats(train, per_domain=(args.norm == "per_domain"))
        targets = fold_targets(D, train, args.target)
        net = train_fold(D, train, stats, args, dev, cmask, targets)
        if fold == 0:
            print(f"  net: {count_params(net) / 1e6:.2f}M params  target={args.target}")

        # threshold picked on TRAIN scenes only - never on the scenes we then report
        tp = []
        for s in train[:min(len(train), 12)]:
            p, y, v = predict(net, D, s, stats, dev, cmask)
            if targets is not None:
                y, v = targets[(s["area"], s["idx"])]
            tp.append((p, y, v))
        thr, _ = sweep_threshold(np.concatenate([p.ravel() for p, _, _ in tp]),
                                 np.concatenate([y.ravel() for _, y, _ in tp]),
                                 np.concatenate([v.ravel() for _, _, v in tp]))

        # Prior wet fraction from the TRAINING scenes, per domain where the domain was seen
        # (loso/flood) and pooled when it was not (lodo). Feeds the calibrated threshold.
        wf = {}
        for s in train:
            y, v = (targets[(s["area"], s["idx"])] if targets is not None
                    else (D.Y[s["area"]][s["idx"]], D.valid[s["area"]]))
            wf.setdefault(s["area"], []).append(float((y & v).sum()) / max(int(v.sum()), 1))
        wf_pooled = float(np.mean([x for xs in wf.values() for x in xs]))

        for s in test:
            p, y, v = predict(net, D, s, stats, dev, cmask)
            r = csi_report(p >= thr, y, v)
            _, csi_opt = sweep_threshold(p, y, v)                     # optimistic, labelled as such
            masks = [D.Y[s["area"]][i] for i in range(len(D.dates[s["area"]]))]
            pers = persistent_field(masks, s["idx"])
            inc = storm_increment(y, pers)
            vi = v & ~pers
            frac = float(np.mean(wf[s["area"]])) if s["area"] in wf else wf_pooled
            thr_c = calibrated_threshold(p, v, frac)
            thr_ci = calibrated_threshold(p, vi, frac)
            rows.append(dict(
                fold=fold, area=s["area"], date=s["date"], flood=s["flood"], thr=round(thr, 3),
                csi=r["csi"], csi_opt=csi_opt, pod=r["pod"], far=r["far"], bias=r["bias"],
                pred_wet=r["pred_wet"], obs_wet=r["obs_wet"],
                csi_allwet=csi_allwet(y, v), csi_random=csi_random(y, v),
                csi_clim=csi_report(pers, y, v)["csi"],
                csi_increment=csi_report(p >= thr, inc, vi)["csi"],
                csi_cal=csi_report(p >= thr_c, y, v)["csi"],
                csi_inc_cal=csi_report(p >= thr_ci, inc, vi)["csi"],
                thr_cal=round(float(thr_c), 4),
                inc_wet=int((inc & v).sum())))

    hdr = (f"{'area':<17}{'date':<12}{'CSI':>7}{'opt':>7}{'bias':>7}{'POD':>6}"
           f"{'allwet':>8}{'rand':>7}{'clim':>7}{'incr':>7}{'cal':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        star = "*" if r["flood"] else " "
        print(f"{r['area']:<17}{r['date'] + star:<12}{r['csi']:>7.4f}{r['csi_opt']:>7.4f}"
              f"{r['bias']:>7.2f}{r['pod']:>6.3f}{r['csi_allwet']:>8.4f}{r['csi_random']:>7.4f}"
              f"{r['csi_clim']:>7.4f}{r['csi_increment']:>7.4f}{r['csi_cal']:>7.4f}")

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    print("-" * len(hdr))
    print(f"{'MEAN':<29}{mean('csi'):>7.4f}{mean('csi_opt'):>7.4f}{mean('bias'):>7.2f}"
          f"{mean('pod'):>6.3f}{mean('csi_allwet'):>8.4f}{mean('csi_random'):>7.4f}"
          f"{mean('csi_clim'):>7.4f}{mean('csi_increment'):>7.4f}{mean('csi_cal'):>7.4f}")
    print("\n(* = real flood event.  'opt' = threshold swept on the test scene itself: "
          "optimistic, shown only for comparability with the existing baseline table.)")
    print("reference: dynamic twin 0.0410 | TWI 0.0422 | HAND-lite 0.0505  (Patna, 8 dates)")
    print(f"elapsed {time.time() - t0:.0f}s")

    os.makedirs(args.out, exist_ok=True)
    name = (f"{args.split}_{args.ablate}_w{args.width}_g{args.grid}_s{args.seed}"
        f"{'_inc' if args.target == 'increment' else ''}"
        f"{'_shuf' + str(args.shuffle_rain) if args.shuffle_rain else ''}"
        f"{'_' + args.areas.replace(',', '+') if args.areas else ''}{args.tag}.json")
    json.dump(dict(args=vars(args), rows=rows,
                   mean={k: mean(k) for k in ("csi", "csi_opt", "bias", "pod", "csi_allwet",
                                              "csi_random", "csi_clim", "csi_increment", "csi_cal",
                                              "csi_inc_cal")}),
              open(os.path.join(args.out, name), "w"), indent=2)
    print("wrote", os.path.join(args.out, name))
    return rows


if __name__ == "__main__":
    main()
