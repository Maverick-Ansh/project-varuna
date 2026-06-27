"""Differentiable simulator tests (port of simulator_test.py to the Domain class).

Checks stability, preferential pooling, mass balance, and gradient flow to dig parameters.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _bowl_domain():
    from varuna.build.twin import Domain
    N, DX = 64, 60.0
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    z = 50.0 + 0.002 * DX * xx
    z = z - 1.5 * np.exp(-(((yy - 32) ** 2 + (xx - 20) ** 2) / 30.0))
    mann = np.full((N, N), 0.03)
    infil = np.full((N, N), 2.0 / 1000 / 3600)
    built = np.ones((N, N))
    dom = Domain(z, mann, infil, built, dx=DX, device="cpu")
    bowl = ((yy - 32) ** 2 + (xx - 20) ** 2) < 36
    return dom, bowl, N, DX


def test_stability_pooling_mass():
    dom, bowl, N, DX = _bowl_domain()
    hmax, h_end = dom.simulate(dom.z0, rain_mm=60.0, storm_hr=1.0, total_hr=2.0, return_final=True)
    assert torch.isfinite(hmax).all()
    # pooling + mass balance are checked on the FINAL state (hmax over-counts transients)
    bowl_t = torch.tensor(bowl)
    bowl_depth = float(h_end[bowl_t].mean())
    mid = float(h_end[10:54, 32:48].mean())
    rained = 60.0 / 1000 * N * N * DX * DX
    on_grid = float(h_end.sum() * DX * DX)
    assert bowl_depth > 5 * max(mid, 1e-4)
    assert on_grid <= rained * 1.001


def test_gradient_reaches_dig_params():
    dom, bowl, N, DX = _bowl_domain()
    K = 2
    masks = torch.zeros(K, N, N)
    masks[0, 30:35, 18:23] = 1.0   # in the bowl
    masks[1, 10:15, 50:55] = 1.0   # on the plane
    eval_mask = torch.ones(N, N) * (1.0 - masks.sum(0).clamp(max=1.0))
    theta = torch.tensor([-1.0, -1.0], requires_grad=True)
    D = (3.0 * torch.sigmoid(theta)).view(K, 1, 1).mul(masks).sum(0)
    hmax = dom.simulate(dom.z0 - D, rain_mm=60.0, grad=True, total_hr=1.0)
    loss = (torch.relu(hmax - 0.15) * eval_mask).sum() * DX * DX
    loss.backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    assert theta.grad.abs().sum() > 0
    assert theta.grad[0] < 0   # digging the bowl reduces flooding


def test_unet_forward_shape():
    from varuna.build.twin import UNet
    emu = UNet()
    out = emu(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 1, 128, 128)
    assert (out >= 0).all()   # softplus output
