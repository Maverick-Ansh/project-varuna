"""Offline tests for the sink-field core (no rasterio / GEE needed).

Same recipe as test_calibrate: a synthetic pure-torch Domain, tiny grid, short storm. The
scientific core gets a recovery test — plant a known drain, simulate "SAR" from that truth,
and check the inversion puts capacity where the truth had it.
"""
import numpy as np
import pytest
import torch

from varuna.build.twin import Domain
from varuna.build import sinkfield as S

SIM = dict(storm_hr=0.1, total_hr=0.3, dt=10.0)   # ~108 steps; quick on CPU


def _synth_domain(N=40, bowl=2.0):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    z = (0.02 * xx + 0.01 * yy).astype("float32")
    z -= bowl * np.exp(-(((xx - N * 0.6) ** 2 + (yy - N * 0.5) ** 2) / (2 * (N / 8.0) ** 2)))
    mann = np.full((N, N), 0.04, "float32")
    infil = np.full((N, N), 1e-6, "float32")
    built = (xx > N * 0.25).astype("float32")      # bowl sits inside built land
    return Domain(z.astype("float32"), mann, infil, built, dx=60.0, device="cpu")


def test_zero_drain_is_bit_exact():
    base = _synth_domain()
    dd = S.DrainDomain(base)
    h0 = base.simulate(base.z0, rain_mm=60.0, **SIM)
    h1 = dd.simulate(dd.z0, rain_mm=60.0, **SIM)
    assert float((h1 - h0).abs().max()) == 0.0


def test_heavy_drain_removes_the_flood():
    base = _synth_domain()
    dd = S.DrainDomain(base)
    h0 = dd.simulate(dd.z0, rain_mm=60.0, **SIM)
    # SIM's 0.1 h burst is a 600 mm/h intensity — the drain must exceed that to zero it
    dd.drain = torch.full_like(dd.z0, 2000.0 / 3.6e6)
    h1 = dd.simulate(dd.z0, rain_mm=60.0, **SIM)
    assert float(h0.max()) > 0.05
    assert float(h1.max()) < 0.01
    assert float(h1.min()) >= 0.0                              # never drains below dry


def test_field_is_masked_to_built_land():
    base = _synth_domain()
    sf = S.SinkField(base, max_mm_h=50.0, init_mm_h=5.0)
    d = sf.drain_ms().detach().numpy()
    assert (d[base.built.numpy() == 0] == 0).all()
    assert (d[base.built.numpy() == 1] > 0).all()


def test_init_field_leaves_baseline_intact():
    base = _synth_domain()
    dd = S.DrainDomain(base)
    h0 = dd.simulate(dd.z0, rain_mm=60.0, **SIM)
    sf = S.SinkField(dd)                                       # default init 0.5 mm/h
    dd.drain = sf.drain_ms().detach()
    h1 = dd.simulate(dd.z0, rain_mm=60.0, **SIM)
    # 0.5 mm/h over a 0.3 h toy sim can move at most ~1.5e-4 m of water
    assert float((h1 - h0).abs().max()) < 5e-4                 # step 0 ~= textbook twin


def test_penalty_grows_with_the_field():
    base = _synth_domain()
    sf = S.SinkField(base)
    p0 = float(sf.penalty())
    with torch.no_grad():
        sf.theta += 4.0
    assert float(sf.penalty()) > p0 > 0.0


def test_gradient_reaches_theta():
    base = _synth_domain()
    dd = S.DrainDomain(base)
    sf = S.SinkField(dd, init_mm_h=1.0)
    dd.drain = sf.drain_ms()
    hmax = dd.simulate(dd.z0, rain_mm=60.0, grad=True, **SIM)
    loss = S.soft_dice_loss(S.soft_wet(hmax), torch.zeros_like(hmax))
    loss.backward()
    assert sf.theta.grad is not None
    assert torch.isfinite(sf.theta.grad).all()
    assert float(sf.theta.grad.abs().max()) > 0.0


def test_save_and_load_roundtrip(tmp_path):
    base = _synth_domain()
    sf = S.SinkField(base, max_mm_h=50.0, init_mm_h=2.0)
    with torch.no_grad():
        sf.theta += torch.randn_like(sf.theta) * 0.1
    path = str(tmp_path / "sinkfield.pt")
    S.save_sinkfield(path, sf, S.DEFAULT_HP)
    dom = S.load_drain_domain(base, path)
    assert torch.allclose(dom.drain, sf.drain_ms().detach(), atol=1e-7)


def test_fit_retries_an_unstable_storm_at_a_smaller_step(monkeypatch):
    """A storm that returns NaN at the default step must be re-run at a smaller one, not
    silently skipped — skipping cost the creek tiles 25 of 40 training iterations."""
    base = _synth_domain()
    calls = []
    real = S.DrainDomain.simulate

    def flaky(self, z, rain_mm, **kw):
        dt = kw.get("dt", 10.0)
        calls.append(dt)
        out = real(self, z, rain_mm, **kw)
        if dt >= 10.0:                       # unstable at the default step only
            return out * float("nan")
        return out

    monkeypatch.setattr(S.DrainDomain, "simulate", flaky)
    rains = {"a": 30.0}
    sar = {"a": torch.zeros(base.N, base.N)}
    hp = dict(S.DEFAULT_HP, iters=1, batch=1, lvol=0.0,
              storm_hr=SIM["storm_hr"], total_hr=SIM["total_hr"])
    _dom, _sf, hist = S.fit(base, sar, None, rains, ["a"], hp=hp)
    assert 10.0 in calls and 5.0 in calls          # retried at half the step
    assert not hist[0].get("skipped")              # and the iteration actually trained


def test_wet_duration_and_drained_volume_helpers():
    base = _synth_domain()
    dur = S.wet_duration_s(base, 30.0, storm_hr=SIM["storm_hr"], total_hr=SIM["total_hr"])
    assert float(dur.max()) > 0.0
    assert float(dur.min()) >= 0.0
    dd = S.DrainDomain(base)
    _, v0 = dd.drained_volume_m3(30.0, storm_hr=SIM["storm_hr"], total_hr=SIM["total_hr"])
    dd.drain = torch.full_like(dd.z0, 20.0 / 3.6e6)
    _, v1 = dd.drained_volume_m3(30.0, storm_hr=SIM["storm_hr"], total_hr=SIM["total_hr"])
    assert v0 == 0.0
    assert v1 > 0.0


@pytest.mark.slow
def test_inversion_recovers_a_planted_drain():
    """Truth: an 8x8, 150 mm/h drain patch in a shallow bowl, strong enough to change the wet
    EXTENT (extent is all the fit sees). Success = capacity concentrates in the patch and the
    fitted outflow is a sane lower bound on the truth's — not exact placement, which extent
    alone cannot identify."""
    SIM2 = dict(storm_hr=0.5, total_hr=1.0, dt=10.0)   # 60 mm/h intensity — realistic burst
    base = _synth_domain(bowl=0.6)                     # shallow bowl: extent responds to drains
    N = base.N
    truth = S.DrainDomain(base)
    patch = torch.zeros_like(truth.z0)
    r0, c0 = int(N * 0.5) - 4, int(N * 0.6) - 4
    patch[r0:r0 + 8, c0:c0 + 8] = 1.0
    truth.drain = patch * (150.0 / 3.6e6)
    rains = {"a": 20.0, "b": 35.0}
    tau = 0.05
    sar, tvol = {}, {}
    with torch.no_grad():
        for d, r in rains.items():
            hm, v = truth.drained_volume_m3(r, storm_hr=SIM2["storm_hr"],
                                            total_hr=SIM2["total_hr"])
            sar[d] = (hm > tau).float()
            tvol[d] = v
            base_wet = int((base.simulate(base.z0, rain_mm=r, **SIM2) > tau).sum())
            assert base_wet > int(sar[d].sum())        # the planted drain IS visible in extent
    hp = dict(S.DEFAULT_HP, iters=40, batch=2, tau=tau, lr=0.25, max_mm_h=300.0,
              storm_hr=SIM2["storm_hr"], total_hr=SIM2["total_hr"])
    dom, sf, hist = S.fit(base, sar, None, rains, list(rains), hp=hp)
    assert hist[-1]["loss"] < hist[0]["loss"]
    dmm = sf.drain_mm_h().numpy()
    inside = dmm[patch.numpy() > 0].mean()
    outside = dmm[(patch.numpy() == 0) & (base.built.numpy() == 1)].mean()
    assert inside > 10.0 * max(outside, 1e-6), (inside, outside)
    for d, r in rains.items():
        _, fv = dom.drained_volume_m3(r, storm_hr=SIM2["storm_hr"], total_hr=SIM2["total_hr"])
        assert 0.2 * tvol[d] < fv < 1.5 * tvol[d], (d, fv, tvol[d])
