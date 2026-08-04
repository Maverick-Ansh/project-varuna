"""Per-cell inferred drainage-sink field — V4's answer to the co-location failure.

Lineage. Two independent validations reached the same verdict: the twin predicts flooded AREA
but cannot rank WHICH street floods worse (SAR CSI ~0.03-0.05 with FAR ~0.95 in both Patna and
Mumbai; wet-site correlation -0.23 against crowdsourced depth reports, DEPTH_VALIDATION.md).
The existing SAR calibration (calibrate.LearnablePhysics) learns ~16 per-WorldCover-class
scalars — it can change how much water ponds, but not WHERE it goes within a class. And the
biggest known physics gap (G2) is that no storm sewer is modelled at all: the twin over-floods
precisely where working drains exist.

This module makes the missing drainage a *learnable per-cell field*. After each verbatim Bates
2010 step, water is removed at a per-cell rate (bounded by what is present) — an effective
storm-sewer/outfall capacity in mm/h. The field is softplus-positive, restricted to built land,
and regularised (L1 toward "no drains" + total variation for spatial coherence), then fitted by
gradient descent through the differentiable simulator so simulated wet extent matches Sentinel-1
water masks on real storm dates. ~65k parameters per 256x256 tile instead of 16: freedom to move
water, priors to keep it physical, held-out storm dates + the quarantined depth reports to keep
it honest.

What the field is NOT: drained water leaves the surface system entirely (sewer -> outfall), so
it is deliberately not added to `infil` — it must not fill the soil column or count as aquifer
recharge, and `rollout`'s rain == surface + infiltrated mass identity does not hold for a
DrainDomain (the drained volume is the difference).

Pure-torch core (DrainDomain, SinkField, fit) is unit-tested offline, including a recovery test
that plants a known drain and checks the inversion finds it. The raster/GEE glue lives in
calibrate (align_sar, valid_mask) and validate (observed_water); the Colab notebook drives them.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from .calibrate import csi_hard, soft_dice_loss, soft_wet
from .twin import Domain

log = logging.getLogger("varuna.build.sinkfield")

DEFAULT_HP = dict(max_mm_h=50.0, init_mm_h=0.5, lr=0.15, iters=60, batch=3,
                  l1=0.02, tv=0.05, lmag=3.0, lvol=1.0, tau=0.15, beta=0.05,
                  storm_hr=2.0, total_hr=4.0, seed=20260804)


def _inv_softplus(y: float) -> float:
    """theta such that softplus(theta) == y, for y > 0."""
    return float(np.log(np.expm1(y)))


class DrainDomain(Domain):
    """Domain + per-cell drainage sink.

    After each verbatim Bates step, water is removed at rate `self.drain` (m/s), bounded by
    what is present in the cell. `self.drain` is set by the caller before each simulate —
    differentiably during fitting (like LearnablePhysics.apply), detached for evaluation.
    With drain == 0 this is bit-exact the plain Domain.
    """

    def __init__(self, base: Domain):
        super().__init__(base.z0.cpu().numpy(), base.mann.cpu().numpy(),
                         base.infil.cpu().numpy(), base.built.cpu().numpy(),
                         dx=base.dx, device=base.device,
                         capacity=None if base.capacity is None else base.capacity.cpu().numpy())
        self.row0, self.col0 = base.row0, base.col0
        self.wc = getattr(base, "wc", None)
        self.drain = torch.zeros_like(self.z0)
        self._drained = None                        # set by drained_volume_m3 to accumulate

    def step(self, h, qx, qy, z, rain_ms, dt, return_infil=False, infil_cum=None):
        out = super().step(h, qx, qy, z, rain_ms, dt, return_infil=return_infil,
                           infil_cum=infil_cum)
        h = out[0]
        d = torch.minimum(self.drain * dt, h)       # sewer removal; never below dry
        h = h - d
        if self._drained is not None:
            self._drained = self._drained + d.sum()
        return (h, *out[1:])

    def drained_volume_m3(self, rain_mm, storm_hr=2.0, total_hr=4.0, dt=10.0):
        """Run a storm and return (hmax, m^3 the drain field removed) — the city's outflow:
        the volume the drainage system actually poured out of the surface during this storm,
        as inferred from the fitted field."""
        with torch.no_grad():
            self._drained = torch.zeros((), device=self.device)
            try:
                hmax = self.simulate(self.z0, rain_mm=rain_mm, storm_hr=storm_hr,
                                     total_hr=total_hr, dt=dt)
                vol = float(self._drained) * self.dx * self.dx
            finally:
                self._drained = None
        return hmax, vol


class SinkField:
    """Unconstrained theta -> per-cell drain rate softplus(theta) * unit, masked to built land.

    init_mm_h ~ 0 keeps step 0 at the textbook baseline (softplus' gradient never vanishes, so
    the field can still wake up anywhere the loss asks it to). max_mm_h sets the rate scale;
    50 mm/h is the order of a modern urban storm-sewer design capacity.
    """

    def __init__(self, dom: Domain, max_mm_h=50.0, init_mm_h=0.05, device=None):
        dev = device or dom.device
        self.unit = max_mm_h / 1000.0 / 3600.0          # softplus output * unit -> m/s
        self.max_mm_h = float(max_mm_h)
        self.theta = torch.full((dom.N, dom.N), _inv_softplus(init_mm_h / max_mm_h),
                                device=dev, requires_grad=True)
        self.mask = dom.built.detach().clone().to(dev)

    def drain_ms(self):
        """Differentiable per-cell drain rate in m/s (zero off built land), hard-capped at
        max_mm_h — softplus is unbounded above and one bad Adam step must not produce a
        physically absurd rate. The penalty uses the UNCAPPED value so the downward gradient
        stays alive even for capped cells."""
        return torch.clamp(F.softplus(self.theta), max=1.0) * self.unit * self.mask

    def drain_mm_h(self):
        return (self.drain_ms() * 3.6e6).detach()

    def penalty(self, l1=0.02, tv=0.05):
        """L1 (prefer no drains) + total variation (drains form coherent structures, not salt)."""
        d = F.softplus(self.theta) * self.mask          # in units of the max rate
        tvv = (d[1:, :] - d[:-1, :]).abs().mean() + (d[:, 1:] - d[:, :-1]).abs().mean()
        return l1 * d.mean() + tv * tvv


def wet_duration_s(dom: Domain, rain_mm, storm_hr=2.0, total_hr=4.0, dt=10.0, hmin=1e-3):
    """Seconds each cell holds water during a storm on the PLAIN domain (no drains) — the
    exposure a drain at that cell would have. One no-grad rollout; used to weight the
    minimum-outflow prior so it is linear in the drain field (checkpoint-safe: accumulating
    drained volume inside the checkpointed step would double-count on recomputation)."""
    with torch.no_grad():
        h = torch.zeros_like(dom.z0)
        qx = torch.zeros(dom.z0.shape[0], dom.z0.shape[1] - 1, device=dom.device)
        qy = torch.zeros(dom.z0.shape[0] - 1, dom.z0.shape[1], device=dom.device)
        cum = torch.zeros_like(dom.z0) if dom.capacity is not None else None
        dur = torch.zeros_like(dom.z0)
        nsteps = int(total_hr * 3600 / dt)
        rain_steps = int(storm_hr * 3600 / dt)
        rain_rate = rain_mm / 1000.0 / (storm_hr * 3600.0)
        for k in range(nsteps):
            r = rain_rate if k < rain_steps else 0.0
            if cum is not None:
                h, qx, qy, fin = dom.step(h, qx, qy, dom.z0, r, dt, return_infil=True,
                                          infil_cum=cum)
                cum = cum + fin
            else:
                h, qx, qy = dom.step(h, qx, qy, dom.z0, r, dt)
            dur = dur + (h > hmin).float() * dt
    return dur


def fit(base: Domain, sar_by_date: dict, valid, rain_by_date: dict, train_dates,
        hp: dict | None = None, progress=None):
    """Fit a sink field by gradient descent through the simulator against SAR wet masks.

    Stochastic over storms: each iteration draws `batch` train dates (seeded — reruns are
    identical). Loss = mean soft-Dice(soft wet extent, SAR) + magnitude hinges + field priors.

    The hinges exist because soft-Dice alone cannot train this field: in deeply flooded cells
    soft_wet saturates (sigmoid((h-tau)/beta) ~= 1 for h >> tau) and its gradient vanishes, so
    only cells within ~beta of the threshold teach anything and the field learns glacially.
    The hinge terms have constant slope regardless of depth: SAR-dry cells push depth down
    toward tau (drains wake up exactly where the twin over-floods), SAR-wet cells push back
    (hits are not sacrificed). Returns (DrainDomain with the fitted field attached detached,
    SinkField, history list).
    """
    hp = {**DEFAULT_HP, **(hp or {})}
    dom = DrainDomain(base)
    sf = SinkField(dom, max_mm_h=hp["max_mm_h"], init_mm_h=hp["init_mm_h"])
    opt = torch.optim.Adam([sf.theta], lr=hp["lr"])
    rng = np.random.default_rng(hp["seed"])
    train_dates = list(train_dates)
    # minimum-outflow prior: drained-volume proxy per date = <drain_rate, wet_duration> / rain
    wetdur = {d: wet_duration_s(base, rain_by_date[d], hp["storm_hr"], hp["total_hr"])
              for d in train_dates} if hp.get("lvol") else {}
    rain_m = {d: rain_by_date[d] / 1000.0 * base.N * base.N for d in train_dates}
    hist = []
    t0 = time.time()
    for it in range(hp["iters"]):
        batch = rng.choice(train_dates, size=min(hp["batch"], len(train_dates)), replace=False)
        opt.zero_grad()
        dom.drain = sf.drain_ms()
        loss = torch.zeros((), device=dom.device)
        for d in batch:
            hmax = dom.simulate(dom.z0, rain_mm=rain_by_date[d], storm_hr=hp["storm_hr"],
                                total_hr=hp["total_hr"], grad=True)
            sar = sar_by_date[d]
            l = soft_dice_loss(soft_wet(hmax, hp["tau"], hp["beta"]), sar, valid)
            # every term below is a VOLUME FRACTION of the storm's rain, so the weights are
            # commensurate: lmag > lvol means removing a m^3 of false water pays more than
            # draining a m^3 costs — drains grow exactly where water is wrong and nowhere else
            if hp.get("lmag"):
                v = valid if valid is not None else torch.ones_like(hmax)
                fp = (torch.relu(hmax - hp["tau"]) * (1.0 - sar) * v).sum() / rain_m[d]
                fn = (torch.relu(hp["tau"] + hp["beta"] - hmax) * sar * v).sum() / rain_m[d]
                l = l + hp["lmag"] * (fp + fn)
            if hp.get("lvol"):
                # minimum-outflow prior: the SMALLEST drained volume that explains the extent.
                # The fitted outflow is therefore a LOWER BOUND on the real one.
                l = l + hp["lvol"] * (dom.drain * wetdur[d]).sum() / rain_m[d]
            loss = loss + l
        loss = loss / len(batch) + sf.penalty(hp["l1"], hp["tv"])
        if not torch.isfinite(loss):
            log.warning("iter %d: non-finite loss — step skipped", it)
            opt.zero_grad()
            hist.append(dict(it=it, loss=float("nan"), skipped=True))
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([sf.theta], 10.0)
        opt.step()
        dmm = sf.drain_mm_h()
        hist.append(dict(it=it, loss=float(loss), mean_mm_h=float(dmm.mean()),
                         max_mm_h=float(dmm.max()),
                         frac_gt1=float((dmm > 1.0).float().mean())))
        if progress and (it % 5 == 0 or it == hp["iters"] - 1):
            progress(it, hp["iters"], hist[-1], (time.time() - t0) / 60.0)
    dom.drain = sf.drain_ms().detach()
    return dom, sf, hist


def eval_dates(dom: Domain, sar_by_date: dict, valid, rain_by_date: dict, dates,
               test_dates=(), tau=0.15, storm_hr=2.0, total_hr=4.0):
    """Hard CSI/POD/FAR per date, always next to the wetness it was bought with (pred_wet /
    sar_wet) — CSI without a wetness figure flatters, see DEPTH_VALIDATION.md."""
    out = {}
    with torch.no_grad():
        for d in dates:
            hmax = dom.simulate(dom.z0, rain_mm=rain_by_date[d], storm_hr=storm_hr,
                                total_hr=total_hr)
            s = csi_hard(hmax, sar_by_date[d], tau=tau, valid=valid)
            s["pred_wet"] = int(((hmax > tau).float() * valid).sum())
            s["sar_wet"] = int((sar_by_date[d] * valid).sum())
            s["split"] = "test" if d in set(test_dates) else "train"
            out[d] = s
    return out


def save_sinkfield(path: str, sf: SinkField, hp: dict):
    torch.save(dict(theta=sf.theta.detach().cpu(), hp=dict(hp),
                    drain_mm_h=sf.drain_mm_h().cpu()), path)
    log.info("saved sink field -> %s", path)


def load_drain_domain(base: Domain, path: str) -> DrainDomain:
    """Rebuild a DrainDomain with a saved fitted field attached (evaluation use)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    dom = DrainDomain(base)
    sf = SinkField(dom, max_mm_h=blob["hp"]["max_mm_h"], init_mm_h=blob["hp"]["init_mm_h"])
    with torch.no_grad():
        sf.theta.copy_(blob["theta"].to(sf.theta.device))
    dom.drain = sf.drain_ms().detach()
    return dom
