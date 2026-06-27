"""Offline tests for the v2 canal routing core (numpy + scipy; no rasterio/GEE/skimage)."""
import numpy as np
import torch

from varuna.build.twin import Domain
from varuna.serve import canals as K


def _domain_with_pond(N=60):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    z = (0.05 * (N - yy) + 0.05 * (N - xx)).astype("float32")               # tilt toward (N-1,N-1)
    z -= 3.0 * np.exp(-(((xx - N * 0.4) ** 2 + (yy - N * 0.4) ** 2) / (2 * (N / 12.0) ** 2)))  # pond
    mann = np.full((N, N), 0.04, "float32")
    infil = np.full((N, N), 1e-7, "float32")
    built = np.ones((N, N), "float32")
    return Domain(z.astype("float32"), mann, infil, built, dx=60.0, device="cpu")


def _lowest_edge(z):
    N = z.shape[0]
    edge = ([(0, j) for j in range(N)] + [(N - 1, j) for j in range(N)] +
            [(i, 0) for i in range(N)] + [(i, N - 1) for i in range(N)])
    edge.sort(key=lambda rc: z[rc])
    return [edge[0]]


def test_descending_bed_is_monotone():
    z = np.linspace(10, 0, 20)[None, :].repeat(3, 0).astype("float64")
    path = [(1, j) for j in range(20)]
    bed = K.descending_bed(z, path, channel_depth=1.0, dx=60.0)
    assert np.all(np.diff(bed) <= 1e-9)                                     # strictly non-increasing
    assert np.all(bed <= np.array([z[r, c] for (r, c) in path]) + 1e-9)     # carve only, never raise


def test_dijkstra_reaches_outfall():
    dom = _domain_with_pond()
    z = dom.z0.cpu().numpy()
    path, tgt = K.dijkstra(z, (24, 24), _lowest_edge(z))
    assert path is not None and path[0] == (24, 24) and tuple(tgt) in set(_lowest_edge(z))
    assert len(path) >= 2


def test_route_canals_reduces_flooding():
    dom = _domain_with_pond()
    with torch.no_grad():
        h0 = dom.simulate(dom.z0, rain_mm=120.0)
    flood = (torch.relu(h0 - 0.15) * dom.built).cpu().numpy()
    z = dom.z0.cpu().numpy()
    canals = K.route_canals(z, flood, _lowest_edge(z), n_canals=2, channel_depth=2.0, dx=60.0)
    assert canals, "no canal routed"
    for c in canals:
        assert np.all(np.diff(c["bed"]) <= 1e-9)                            # each bed descends

    z_carved = dom.z0.clone()
    cm = torch.zeros_like(dom.z0)
    for c in canals:
        for k, (r, cc) in enumerate(c["path"]):
            z_carved[r, cc] = min(float(z_carved[r, cc]), float(c["bed"][k]))
            cm[r, cc] = 1.0
    with torch.no_grad():
        h1 = dom.simulate(z_carved, rain_mm=120.0)
    streets = dom.built * (1 - cm)
    v0 = float((torch.relu(h0 - 0.15) * streets).sum())
    v1 = float((torch.relu(h1 - 0.15) * streets).sum())
    assert v1 < v0                                                          # canals drained the pond
