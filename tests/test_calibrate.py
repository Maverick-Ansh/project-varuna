"""Offline tests for the SAR-calibration core (no rasterio / GEE needed).

Builds a synthetic torch-only Domain directly from numpy (the simulator is pure torch, like
simulator_test.py) and exercises LearnablePhysics + the differentiable SAR loss + the Adam loop.
The raster glue (align_sar / valid_mask) is covered separately on Colab where rasterio + the bundle
exist. Fast: tiny grid, short storm.
"""
import numpy as np
import torch

from varuna.build.twin import Domain
from varuna.build import calibrate as C

SIM = dict(storm_hr=0.1, total_hr=0.3, dt=10.0)   # ~108 steps; quick on CPU


def _synth_domain(N=40):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    z = (0.02 * xx + 0.01 * yy).astype("float32")                          # gentle tilt
    z -= 2.0 * np.exp(-(((xx - N * 0.5) ** 2 + (yy - N * 0.5) ** 2) / (2 * (N / 8.0) ** 2)))  # bowl
    mann = np.full((N, N), 0.04, "float32")
    infil = np.full((N, N), 1e-6, "float32")                               # m/s, small
    built = (xx > N / 2).astype("float32")
    dom = Domain(z.astype("float32"), mann, infil, built, dx=60.0, device="cpu")
    dom.wc = np.where(xx > N / 2, 50, 10).astype(np.int32)                 # built vs grassland
    return dom


def test_soft_dice_and_csi_edges():
    a = torch.rand(20, 20)
    mask = (a > 0.5).float()
    assert float(C.soft_dice_loss(mask, mask)) < 1e-3                      # identical -> ~0
    full, empty = torch.ones(8, 8), torch.zeros(8, 8)
    perfect = C.csi_hard(full, full)                                       # depth>tau everywhere, all wet
    disjoint = C.csi_hard(full, empty)
    assert perfect["csi"] == 1.0 and perfect["pod"] == 1.0
    assert disjoint["csi"] == 0.0
    for k in ("csi", "pod", "far"):
        assert 0.0 <= perfect[k] <= 1.0 and 0.0 <= disjoint[k] <= 1.0


def test_crop_pooling_shape():
    dom = _synth_domain(32)
    dom.row0, dom.col0 = 0, 0
    arr = np.random.rand(dom.N * 2 + 5, dom.N * 2 + 5).astype("float32")   # oversized full-AOI grid
    pooled = C._crop_to_domain(arr, dom, agg="max")
    assert tuple(pooled.shape) == (dom.N, dom.N)


def test_learnable_identity_and_gradient_flow():
    dom = _synth_domain()
    phys = C.LearnablePhysics(dom)
    phys.apply()
    assert torch.allclose(dom.mann, phys.mann_base)                        # init multipliers = 1
    assert torch.allclose(dom.infil, phys.infil_base)
    hmax = dom.simulate(dom.z0, rain_mm=80.0, grad=True, **SIM)
    loss = C.soft_dice_loss(C.soft_wet(hmax), (dom.z0 < dom.z0.mean()).float())
    loss.backward()
    assert phys.log_n.grad is not None and phys.log_f.grad is not None
    assert torch.isfinite(phys.log_n.grad).all() and torch.isfinite(phys.log_f.grad).all()


def test_calibration_reduces_loss_and_moves_params():
    dom = _synth_domain()
    phys = C.LearnablePhysics(dom)

    # target = water extent under a DIFFERENT (known) physics, so calibration has signal to chase
    with torch.no_grad():
        phys.log_f.copy_(torch.tensor([1.2, -0.8]))                        # perturb infiltration per class
        phys.apply()
        target = (dom.simulate(dom.z0, rain_mm=120.0, **SIM) > 0.15).float()
    phys.log_f.detach().zero_()                                            # reset to textbook start
    phys.log_n.detach().zero_()

    opt = torch.optim.Adam(phys.params(), lr=0.1)
    losses = []
    for _ in range(12):
        opt.zero_grad()
        phys.apply()
        hmax = dom.simulate(dom.z0, rain_mm=120.0, grad=True, **SIM)
        loss = C.soft_dice_loss(C.soft_wet(hmax), target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]                                          # learning happened
    moved = (phys.log_n.abs().sum() + phys.log_f.abs().sum()).item()
    assert moved > 1e-3                                                    # params left the textbook prior
    assert all(np.isfinite(losses))
