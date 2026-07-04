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


def test_dijkstra_avoids_buildings():
    # flat plain: straight line would cross a building wall at col 10; router must detour
    z = np.zeros((21, 21), dtype="float64")
    mult = np.ones((21, 21))
    mult[3:18, 10] = 30.0                       # building wall with gaps at rows <3 and >17
    path, _ = K.dijkstra(z, (10, 2), [(10, 18)], cost_mult=mult)
    assert path is not None
    crossed = [(r, c) for (r, c) in path if 3 <= r < 18 and c == 10]
    assert crossed == []                        # went around, not through


def test_dijkstra_prefers_road_corridor():
    # flat plain: a long cheap road along row 1 should pull the path off the direct line
    # (the corridor must be long enough that its saving beats the detour cost)
    z = np.zeros((10, 40), dtype="float64")
    mult = np.ones((10, 40))
    mult[1, :] = 0.3
    path, _ = K.dijkstra(z, (8, 0), [(8, 39)], cost_mult=mult)
    assert any(r == 1 for (r, c) in path)       # detours via the road row


def test_basin_weighting_reorders():
    flood = np.zeros((20, 20))
    flood[2:5, 2:5] = 1.0                       # basin A: volume 9
    flood[12:16, 12:16] = 0.7                   # basin B: volume 11.2 (bigger unweighted)
    unweighted = K.basin_sources(flood, min_cells=4)
    w = np.ones_like(flood)
    w[2:5, 2:5] = 5.0                           # A is chronically waterlogged
    weighted = K.basin_sources(flood, min_cells=4, weight=w)
    assert unweighted[0][0] >= 12               # B first without weighting
    assert weighted[0][0] < 12                  # A first with weighting


def test_load_urban_mult_roundtrip(tmp_path):
    b = np.zeros((8, 8), bool); b[4, 4] = True
    r = np.zeros((8, 8), bool); r[0, :] = True
    np.savez(tmp_path / "urban_grid.npz", buildings=b, roads=r)
    mult = K.load_urban_mult(str(tmp_path))
    assert mult[4, 4] == 30.0 and mult[0, 3] == 0.5 and mult[5, 5] == 1.0
    assert K.load_urban_mult(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------- street-spiderweb network routing

SCALE = 0.0005  # deg per grid cell for synthetic street graphs (lat=r*SCALE, lon=c*SCALE)


def _street_graph(ways_rc):
    idx, nodes, ways = {}, [], []
    for chain in ways_rc:
        way = []
        for rc in chain:
            if rc not in idx:
                idx[rc] = len(nodes)
                nodes.append([rc[0] * SCALE, rc[1] * SCALE])
            way.append(idx[rc])
        ways.append(way)
    return {"nodes": nodes, "ways": ways}


def _cell_of(shape):
    H, W = shape

    def cell_of(lat, lon):
        r, c = int(round(lat / SCALE)), int(round(lon / SCALE))
        return (r, c) if (0 <= r < H and 0 <= c < W) else None

    return cell_of


def test_inlet_cells_spread_and_volume_share():
    flood = np.zeros((30, 30))
    yy, xx = np.meshgrid(np.arange(30), np.arange(30), indexing="ij")
    flood[8:24, 8:24] = np.exp(-(((xx - 15.0) ** 2 + (yy - 15.0) ** 2) / 40.0))[8:24, 8:24]
    inlets = K.inlet_cells(flood, n_inlets=6, suppress_radius=3)
    assert len(inlets) >= 3
    for i, (r1, c1, _, _) in enumerate(inlets):
        for (r2, c2, _, _) in inlets[i + 1:]:
            assert max(abs(r1 - r2), abs(c1 - c2)) > 3       # suppression disc spreads them
    assert abs(sum(v for (_, _, v, _) in inlets) - flood.sum()) < 1e-6   # volume fully apportioned


def test_route_network_spiderweb_on_synthetic_streets():
    dom = _domain_with_pond()                                 # pond ~(24,24), tilt toward (59,59)
    z = dom.z0.cpu().numpy()
    with torch.no_grad():
        h0 = dom.simulate(dom.z0, rain_mm=120.0)
    flood = (torch.relu(h0 - 0.15) * dom.built).cpu().numpy()
    graph = _street_graph([
        [(r, 24) for r in range(60)],                         # avenue through the pond, downhill
        [(24, c) for c in range(60)],                         # cross street through the pond
    ])
    routed = K.route_network(z, flood, graph, _cell_of(z.shape), lowland_cells=[],
                             boundary_cells=[(59, 24)], pit_cells=[], n_inlets=8, dx=60.0)
    assert routed is not None
    canals, network = routed
    assert len(canals) >= 2 and len(network["edges"]) >= 2
    assert all(0.0 <= e["weight"] <= 1.0 for e in network["edges"])
    assert network["outfall_points"][0]["kind"] == "boundary"

    # carve exactly as plan_canals does; shared trunks must stay at the deepest bed (water flows)
    z_carved = dom.z0.clone()
    cm = torch.zeros_like(dom.z0)
    for c in canals:
        assert np.all(np.diff(c["bed"]) <= 1e-9)              # every inlet's bed descends
        for k, (r, cc) in enumerate(c["path"]):
            z_carved[r, cc] = min(float(z_carved[r, cc]), float(c["bed"][k]))
            cm[r, cc] = 1.0
    for c in canals:
        for k, (r, cc) in enumerate(c["path"]):
            assert float(z_carved[r, cc]) <= float(c["bed"][k]) + 1e-5   # float32 storage eps

    with torch.no_grad():
        h1 = dom.simulate(z_carved, rain_mm=120.0)
    streets = dom.built * (1 - cm)
    v0 = float((torch.relu(h0 - 0.15) * streets).sum())
    v1 = float((torch.relu(h1 - 0.15) * streets).sum())
    assert v1 < v0                                            # the web actually drains the pond


def test_route_network_falls_back_without_reachable_outfall():
    dom = _domain_with_pond()
    z = dom.z0.cpu().numpy()
    with torch.no_grad():
        h0 = dom.simulate(dom.z0, rain_mm=120.0)
    flood = (torch.relu(h0 - 0.15) * dom.built).cpu().numpy()
    graph = _street_graph([[(r, 5) for r in range(12)]])      # short street far from any outfall
    routed = K.route_network(z, flood, graph, _cell_of(z.shape), lowland_cells=[],
                             boundary_cells=[(59, 59)], pit_cells=[], n_inlets=8, dx=60.0)
    assert routed is None                                     # plan_canals then uses grid routing


def test_lowland_cells_avoid_built_and_spread(tmp_path):
    N = 60
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    z = (0.05 * (N - yy) + 0.05 * (N - xx)).astype("float32")   # lowest corner at (59,59)
    built = np.ones((N, N), "float32")
    built[40:, 40:] = 0.0                                       # open land at the low corner
    mann = np.full((N, N), 0.04, "float32")
    infil = np.full((N, N), 1e-7, "float32")
    dom = Domain(z, mann, infil, built, dx=60.0, device="cpu")
    cells = K._lowland_cells(dom, str(tmp_path), n=3, suppress_radius=5)
    assert cells and cells[0] == (59, 59)                       # lowest safe ground first
    for (r, c) in cells:
        assert r >= 40 and c >= 40                              # never on built land
    for i, (r1, c1) in enumerate(cells):
        for (r2, c2) in cells[i + 1:]:
            assert max(abs(r1 - r2), abs(c1 - c2)) > 5          # dispersal targets spread out


def test_route_network_prefers_lowland_over_handicapped_pit():
    # a pit sits mid-street; the safe lowland is farther but pits carry a start handicap
    z = np.zeros((40, 40), dtype="float64")
    flood = np.zeros((40, 40))
    flood[2:7, 18:23] = 0.5                                     # basin at the top of the street
    graph = _street_graph([[(r, 20) for r in range(40)]])
    routed = K.route_network(z, flood, graph, _cell_of(z.shape),
                             lowland_cells=[(39, 20)],          # 2.1 km away
                             boundary_cells=[], pit_cells=[(20, 20)],   # 0.9 km away
                             n_inlets=4, dx=60.0)
    assert routed is not None
    canals, network = routed
    kinds = {o["kind"] for o in network["outfall_points"]}
    assert kinds == {"lowland"}                                 # 2000 m handicap outweighs 1.2 km
