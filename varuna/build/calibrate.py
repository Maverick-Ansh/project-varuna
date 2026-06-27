"""SAR calibration of the differentiable flood twin — the novel, paper-worthy loop.

Manning's n and infiltration ship as textbook lookup tables (twin.py N_TABLE/F_TABLE). Here we make
them *learnable* — one multiplier per ESA WorldCover class — and fit them by gradient descent so the
twin's simulated water extent agrees with Sentinel-1 SAR water masks on real storm dates. Calibrate
on a set of dates, report held-out CSI before (textbook) vs after (calibrated): a satellite-calibrated
differentiable flood twin of an Indian city.

Two halves:
  * pure-torch core (LearnablePhysics, soft_wet, soft_dice_loss, csi_hard) — unit-tested offline;
  * raster/GEE glue (align_sar, valid_mask) + the calibrate()/score_twin drivers — run on Colab where
    rasterio + the artifact bundle (observed_water_<date>.tif, jrc_occurrence.tif) are present.

The twin is scored on the 128x128 / 60 m domain grid (build_domain's crop), NOT the full-AOI static
depth.tif that validate.score uses, and the same jrc<50 permanent-water mask is applied to BOTH the
prediction and the SAR truth (so the Ganga channel is excluded on both sides, not just the truth).
"""
from __future__ import annotations

import datetime as _dt
import logging

import numpy as np
import torch

from ..config import CFG
from ..io import save_json

log = logging.getLogger("varuna.build.calibrate")


# --------------------------------------------------------------------------- pure-torch core


class LearnablePhysics:
    """Per-WorldCover-class learnable multipliers on Manning n and infiltration.

    n_eff = n_base * exp(log_n[class]),  f_eff = f_base * exp(log_f[class]).
    log_* init at 0 => exp = 1 => exactly the textbook physics (so calibration step 0 == baseline).
    exp() keeps values positive; gradients flow from the small per-class vectors to per-cell tensors
    via an index map, then through dom.simulate (pure torch) to the soft-Dice SAR loss.
    """

    def __init__(self, dom, device=None):
        if not hasattr(dom, "wc"):
            raise ValueError("dom.wc missing — rebuild the domain with the current build_domain "
                             "(it stashes the WorldCover class map for calibration).")
        dev = device or dom.device
        self.dom = dom
        wc = np.asarray(dom.wc)
        self.classes = [int(c) for c in np.unique(wc)]
        c2i = {c: i for i, c in enumerate(self.classes)}
        idx = np.vectorize(c2i.get)(wc).astype("int64")
        self.idx = torch.as_tensor(idx, device=dev)
        self.mann_base = dom.mann.detach().clone().to(dev)
        self.infil_base = dom.infil.detach().clone().to(dev)
        self.log_n = torch.zeros(len(self.classes), device=dev, requires_grad=True)
        self.log_f = torch.zeros(len(self.classes), device=dev, requires_grad=True)

    def params(self):
        return [self.log_n, self.log_f]

    def apply(self):
        """Write the current effective n/infil onto the domain (keeps the autograd graph)."""
        self.dom.mann = self.mann_base * torch.exp(self.log_n[self.idx])
        self.dom.infil = self.infil_base * torch.exp(self.log_f[self.idx])

    def reset(self):
        """Restore textbook physics (multipliers = 1) on the domain."""
        self.dom.mann = self.mann_base.clone()
        self.dom.infil = self.infil_base.clone()

    def multipliers(self):
        """{worldcover_class: {'n_mult':.., 'infil_mult':..}} — interpretable calibration result."""
        return {int(c): {"n_mult": round(float(torch.exp(self.log_n[i])), 4),
                         "infil_mult": round(float(torch.exp(self.log_f[i])), 4)}
                for i, c in enumerate(self.classes)}

    def prior_penalty(self):
        return (self.log_n ** 2).sum() + (self.log_f ** 2).sum()


def soft_wet(hmax, tau=0.15, beta=0.05):
    """Differentiable wet probability: ~1 where depth >> tau, ~0 where depth << tau."""
    return torch.sigmoid((hmax - tau) / beta)


def soft_dice_loss(pred_prob, target, valid=None, eps=1.0):
    """1 - soft Dice between a [0,1] probability map and a {0,1} target. valid masks scored cells."""
    if valid is not None:
        pred_prob = pred_prob * valid
        target = target * valid
    inter = (pred_prob * target).sum()
    denom = pred_prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def csi_hard(hmax, sar, tau=0.15, valid=None):
    """Non-differentiable CSI/POD/FAR for reporting. hmax: depth grid; sar: {0,1} truth."""
    pred = hmax > tau
    obs = sar > 0.5
    if valid is not None:
        m = valid > 0.5
        pred = pred & m
        obs = obs & m
    hits = int((pred & obs).sum())
    misses = int((~pred & obs).sum())
    fa = int((pred & ~obs).sum())
    return dict(csi=hits / max(hits + misses + fa, 1),
                pod=hits / max(hits + misses, 1),
                far=fa / max(hits + fa, 1),
                hits=hits, misses=misses, false_alarms=fa)


# --------------------------------------------------------------------------- raster / GEE glue (Colab)


def _crop_to_domain(arr, dom, agg="max"):
    """Crop a full-AOI 30 m array to build_domain's 256x256 window, then 2x-pool to 128x128."""
    N30 = dom.N * 2
    crop = np.asarray(arr, dtype="float32")[dom.row0:dom.row0 + N30, dom.col0:dom.col0 + N30]
    if crop.shape != (N30, N30):                                   # defensive: edge crops
        crop = np.pad(crop, ((0, max(0, N30 - crop.shape[0])), (0, max(0, N30 - crop.shape[1]))),
                      mode="edge")[:N30, :N30]
    t = torch.as_tensor(crop, device=dom.device).view(dom.N, 2, dom.N, 2)
    return t.amax(dim=(1, 3)) if agg == "max" else t.mean(dim=(1, 3))


def align_sar(work, date, dom):
    """observed_water_<date>.tif -> {0,1} water mask on the 128x128 domain grid (max-pooled)."""
    import rasterio
    with rasterio.open(f"{work}/observed_water_{date}.tif") as s:
        obs = s.read(1)
    return (_crop_to_domain(obs, dom, agg="max") > 0.5).float()


def valid_mask(work, dom):
    """Scoreable cells = NOT permanent water (jrc occurrence < 50%), on the domain grid.

    Applied to BOTH prediction and SAR truth so river channels (which the SAR mask already removes)
    are excluded from the prediction too — the single biggest false-alarm sink in the raw scoring.
    """
    import os
    import rasterio
    path = f"{work}/jrc_occurrence.tif"
    if not os.path.exists(path):
        log.warning("jrc_occurrence.tif absent — scoring without the permanent-water mask.")
        return torch.ones((dom.N, dom.N), device=dom.device)
    with rasterio.open(path) as s:
        jrc = s.read(1)
    occ = _crop_to_domain(jrc, dom, agg="mean")        # mean occurrence % over each 60 m cell
    return (occ < 50).float()


def _rain_for_date(date, window_days=2, center=None):
    """Total ERA5 archive rain (mm) over the `window_days` ending on the SAR overpass date."""
    from ..serve.weather import historical_rain_mm
    center = center or CFG.center
    d1 = _dt.date.fromisoformat(date)
    d0 = d1 - _dt.timedelta(days=window_days)
    return historical_rain_mm(center[0], center[1], d0.isoformat(), d1.isoformat())


# --------------------------------------------------------------------------- drivers


def score_twin(date, work=None, tau=None, rain_mm=None, window_days=2,
               storm_hr=2.0, total_hr=4.0, device=None):
    """Honest baseline: score the *dynamic twin* (real rain) vs SAR on the domain grid.

    This replaces validate.score's static depth.tif comparison. Returns CSI/POD/FAR and writes
    twin_scores_<date>.json. Run after the bundle + observed_water_<date>.tif exist.
    """
    from .twin import build_domain
    work = work or CFG.work
    tau = CFG.min_depth_m if tau is None else tau
    dom = build_domain(work, device=device)
    rain = _rain_for_date(date, window_days) if rain_mm is None else float(rain_mm)
    sar = align_sar(work, date, dom)
    valid = valid_mask(work, dom)
    with torch.no_grad():
        hmax = dom.simulate(dom.z0, rain_mm=rain, storm_hr=storm_hr, total_hr=total_hr)
    s = csi_hard(hmax, sar, tau=tau, valid=valid)
    s.update(event_date=date, rain_mm=round(rain, 1), tau_m=tau, model="dynamic_twin")
    save_json(f"{work}/twin_scores_{date}.json", s)
    log.info("twin vs SAR %s: CSI=%.3f POD=%.3f FAR=%.3f (rain %.0f mm)",
             date, s["csi"], s["pod"], s["far"], rain)
    return s


def calibrate(dates_train, dates_test=None, work=None, iters=40, lr=0.05, lam=1e-2,
              tau=None, beta=0.05, window_days=2, storm_hr=2.0, total_hr=4.0, device=None):
    """Fit per-class Manning/infiltration multipliers to SAR by gradient descent.

    Reports held-out CSI before (textbook) and after (calibrated); saves calibrated_params.json
    + calibration_report.json. Pick dates that actually had antecedent rain (else no flood signal).
    """
    from .twin import build_domain
    work = work or CFG.work
    tau = CFG.min_depth_m if tau is None else tau
    dates_test = dates_test or []
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dom = build_domain(work, device=dev)
    phys = LearnablePhysics(dom, device=dev)
    valid = valid_mask(work, dom)

    def prep(dates):
        out = []
        for d in dates:
            rain = _rain_for_date(d, window_days)
            sar = align_sar(work, d, dom)
            if rain < 1.0:
                log.warning("date %s has ~no antecedent rain (%.1f mm) — weak calibration signal", d, rain)
            out.append((d, sar, rain))
        return out

    train, test = prep(dates_train), prep(dates_test)

    def eval_set(dset):
        phys.apply()
        rows = {}
        with torch.no_grad():
            for (d, sar, rain) in dset:
                hmax = dom.simulate(dom.z0, rain_mm=rain, storm_hr=storm_hr, total_hr=total_hr)
                rows[d] = csi_hard(hmax, sar, tau=tau, valid=valid)
        return rows

    phys.reset()
    baseline_train = eval_set(train)
    baseline_test = eval_set(test)
    log.info("baseline (textbook) mean CSI: train=%.3f test=%.3f",
             _mean_csi(baseline_train), _mean_csi(baseline_test))

    opt = torch.optim.Adam(phys.params(), lr=lr)
    history = []
    for it in range(iters):
        opt.zero_grad()
        phys.apply()
        loss = torch.zeros((), device=dev)
        for (d, sar, rain) in train:
            hmax = dom.simulate(dom.z0, rain_mm=rain, storm_hr=storm_hr, total_hr=total_hr, grad=True)
            loss = loss + soft_dice_loss(soft_wet(hmax, tau, beta), sar, valid)
        loss = loss / max(len(train), 1) + lam * phys.prior_penalty()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if it % 5 == 0 or it == iters - 1:
            log.info("iter %3d soft-Dice loss %.4f", it, float(loss.detach()))

    calib_train = eval_set(train)
    calib_test = eval_set(test)
    log.info("calibrated mean CSI: train=%.3f test=%.3f",
             _mean_csi(calib_train), _mean_csi(calib_test))

    mults = phys.multipliers()
    save_json(f"{work}/calibrated_params.json",
              {"multipliers_by_worldcover_class": mults, "tau_m": tau,
               "storm_hr": storm_hr, "total_hr": total_hr, "window_days": window_days})
    report = {
        "dates_train": list(dates_train), "dates_test": list(dates_test),
        "iters": iters, "lr": lr, "lam": lam, "beta": beta, "tau_m": tau,
        "baseline": {"train": baseline_train, "test": baseline_test,
                     "mean_csi_train": _mean_csi(baseline_train), "mean_csi_test": _mean_csi(baseline_test)},
        "calibrated": {"train": calib_train, "test": calib_test,
                       "mean_csi_train": _mean_csi(calib_train), "mean_csi_test": _mean_csi(calib_test)},
        "multipliers": mults, "loss_history": history,
    }
    save_json(f"{work}/calibration_report.json", report)
    log.info("CSI held-out: %.3f (textbook) -> %.3f (calibrated)",
             _mean_csi(baseline_test), _mean_csi(calib_test))
    return report


def _mean_csi(rows):
    return round(float(np.mean([r["csi"] for r in rows.values()])), 4) if rows else float("nan")


def run(dates=None, n_test=1, work=None, project_id=None, **kw):
    """Convenience entry: ensure SAR masks exist for `dates`, split train/test, calibrate.

    `dates` = list of Sentinel-1 overpass dates (with antecedent rain). Downloads any missing
    observed_water_<date>.tif first (needs GEE). Last n_test dates are held out.
    """
    from .validate import observed_water
    from ..ee_auth import init_ee
    import os
    work = work or CFG.work
    if not dates:
        raise ValueError("pass dates=[...] (use validate.list_passes to find Sentinel-1 overpasses)")
    init_ee(project_id)
    for d in dates:
        if not os.path.exists(f"{work}/observed_water_{d}.tif"):
            log.info("downloading SAR water mask for %s", d)
            observed_water(d, work)
    n_test = min(max(n_test, 0), len(dates) - 1)
    train, test = dates[:len(dates) - n_test], dates[len(dates) - n_test:]
    log.info("calibrate on %s | hold out %s", train, test)
    return calibrate(train, test, work=work, **kw)
