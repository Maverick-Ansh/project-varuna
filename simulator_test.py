"""Smoke test for the differentiable flood simulator in notebook 05.
Synthetic 64x64 terrain: a tilted plane with a bowl. Checks:
1. no NaN/inf after a full storm
2. water preferentially pools in the bowl
3. mass balance: water on grid <= rain added (infiltration removes some)
4. gradients flow from flooded-volume loss back to dig-depth parameters
"""
import torch, numpy as np

device = "cpu"
N, DX, G, HMIN = 64, 60.0, 9.81, 1e-3

yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
z_np = 50.0 + 0.002 * DX * xx                                  # gentle eastward tilt
bowl = ((yy - 32) ** 2 + (xx - 20) ** 2) < 36
z_np = z_np - 1.5 * np.exp(-(((yy - 32) ** 2 + (xx - 20) ** 2) / 30.0))  # smooth 1.5 m bowl

z0 = torch.tensor(z_np, dtype=torch.float32)
mann = torch.full((N, N), 0.03)
infil = torch.full((N, N), 2.0 / 1000 / 3600)                  # 2 mm/hr
built = torch.ones(N, N)

def step(h, qx, qy, z, rain_ms, dt):
    eta = z + h
    hf = torch.clamp(torch.maximum(eta[:, :-1], eta[:, 1:]) - torch.maximum(z[:, :-1], z[:, 1:]), min=0.0)
    Sx = (eta[:, 1:] - eta[:, :-1]) / DX
    nx = 0.5 * (mann[:, :-1] + mann[:, 1:])
    hf73 = torch.clamp(hf, min=HMIN) ** (7.0 / 3.0)
    qx = (qx - G * hf * dt * Sx) / (1.0 + G * dt * nx ** 2 * torch.abs(qx) / hf73)
    qcap = 0.25 * hf * DX / dt
    qx = torch.where(hf > HMIN, torch.clamp(qx, -qcap, qcap), torch.zeros_like(qx))
    hfy = torch.clamp(torch.maximum(eta[:-1, :], eta[1:, :]) - torch.maximum(z[:-1, :], z[1:, :]), min=0.0)
    Sy = (eta[1:, :] - eta[:-1, :]) / DX
    ny = 0.5 * (mann[:-1, :] + mann[1:, :])
    hfy73 = torch.clamp(hfy, min=HMIN) ** (7.0 / 3.0)
    qy = (qy - G * hfy * dt * Sy) / (1.0 + G * dt * ny ** 2 * torch.abs(qy) / hfy73)
    qcapy = 0.25 * hfy * DX / dt
    qy = torch.where(hfy > HMIN, torch.clamp(qy, -qcapy, qcapy), torch.zeros_like(qy))
    Qx = torch.nn.functional.pad(qx, (1, 1, 0, 0))
    Qy = torch.nn.functional.pad(qy, (0, 0, 1, 1))
    dh = dt / DX * (Qx[:, :-1] - Qx[:, 1:] + Qy[:-1, :] - Qy[1:, :])
    h = torch.clamp(h + dh + rain_ms * dt - infil * dt, min=0.0)
    return h, qx, qy

def simulate(z, rain_mm, storm_hr=1.0, total_hr=2.0, dt=10.0, chunk=60, grad=False):
    h = torch.zeros_like(z)
    qx = torch.zeros(z.shape[0], z.shape[1] - 1)
    qy = torch.zeros(z.shape[0] - 1, z.shape[1])
    hmax = torch.zeros_like(z)
    nsteps = int(total_hr * 3600 / dt); rain_steps = int(storm_hr * 3600 / dt)
    rain_rate = rain_mm / 1000.0 / (storm_hr * 3600.0)
    def run_chunk(h, qx, qy, hmax, k0, k1):
        for k in range(int(k0), int(k1)):
            r = rain_rate if k < rain_steps else 0.0
            h, qx, qy = step(h, qx, qy, z, r, dt)
            hmax = torch.maximum(hmax, h)
        return h, qx, qy, hmax
    k = 0
    while k < nsteps:
        k1 = min(k + chunk, nsteps)
        if grad:
            h, qx, qy, hmax = torch.utils.checkpoint.checkpoint(
                run_chunk, h, qx, qy, hmax, torch.tensor(k), torch.tensor(k1), use_reentrant=False)
        else:
            h, qx, qy, hmax = run_chunk(h, qx, qy, hmax, k, k1)
        k = k1
    return hmax, h

# --- test 1 & 2 & 3: stability, pooling, mass ---
hmax, h_end = simulate(z0, rain_mm=60.0)
assert torch.isfinite(hmax).all(), "NaN/inf in simulation!"
bowl_t = torch.tensor(bowl)
bowl_depth = float(h_end[bowl_t].mean())
mid_plane = float(h_end[10:54, 32:48].mean())          # interior slope, away from bowl and east wall
west_wall = float(h_end[:, :3].mean())                 # terrain rises eastward -> runoff collects west
east_wall = float(h_end[:, -3:].mean())                # uphill edge should be nearly dry
rain_total_m3 = 60.0 / 1000 * N * N * DX * DX
water_m3 = float(h_end.sum() * DX * DX)
print(f"stable: yes | bowl {bowl_depth:.3f} m | mid-plane {mid_plane:.4f} m | west wall {west_wall:.3f} m | east wall {east_wall:.3f} m")
print(f"mass: {water_m3:,.0f} m3 on grid of {rain_total_m3:,.0f} m3 rained (rest infiltrated/edge-pooled)")
assert bowl_depth > 5 * max(mid_plane, 1e-4), "water did not preferentially pool in the bowl"
assert west_wall > mid_plane > east_wall, "downslope flow not happening"
assert water_m3 <= rain_total_m3 * 1.001, "mass created from nothing!"

# --- test 4: gradient flow through dig parameters ---
K = 2
masks = torch.zeros(K, N, N)
masks[0, 30:35, 18:23] = 1.0   # in the bowl
masks[1, 10:15, 50:55] = 1.0   # on the plane
site_any = masks.sum(0).clamp(max=1.0)
eval_mask = built * (1.0 - site_any)
theta = torch.tensor([-1.0, -1.0], requires_grad=True)
D = (3.0 * torch.sigmoid(theta)).view(K, 1, 1).mul(masks).sum(0)
hmax_g, _ = simulate(z0 - D, rain_mm=60.0, grad=True, total_hr=1.0)
loss = (torch.relu(hmax_g - 0.15) * eval_mask).sum() * DX * DX
loss.backward()
print(f"flooded-volume loss {loss.item():,.0f} | dLoss/dtheta = {theta.grad.tolist()}")
assert theta.grad is not None and torch.isfinite(theta.grad).all() and theta.grad.abs().sum() > 0, \
    "no gradient reached the dig parameters"
assert theta.grad[0] < 0, "digging the bowl site should REDUCE flooding (negative gradient)"
print("ALL SIMULATOR TESTS PASSED")
