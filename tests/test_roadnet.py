"""Offline tests for the OSM street-graph storm-drain router (numpy only; no network/rasterio).

Synthetic graphs use lat = r*SCALE, lon = c*SCALE so grid cells and street coordinates line up
through a plain linear cell_of — the real driver swaps in the dem.tif inverse transform.
"""
import numpy as np

from varuna.serve import roadnet as R

SCALE = 0.0005          # deg per grid cell (~55 m — close to the real 60 m cells)


def _cell_of(shape):
    H, W = shape

    def cell_of(lat, lon):
        r, c = int(round(lat / SCALE)), int(round(lon / SCALE))
        return (r, c) if (0 <= r < H and 0 <= c < W) else None

    return cell_of


def _graph(ways_rc):
    """Build a persisted-format graph from ways given as [(r, c), ...] chains."""
    idx, nodes, ways = {}, [], []
    for chain in ways_rc:
        way = []
        for rc in chain:
            if rc not in idx:
                idx[rc] = len(nodes)
                nodes.append([rc[0] * SCALE, rc[1] * SCALE])
            way.append(idx[rc])
        ways.append(way)
    return {"nodes": nodes, "ways": ways}, idx


def test_graph_from_overpass_dedupes_nodes():
    osm = {"elements": [
        {"type": "way", "nodes": [1, 2],
         "geometry": [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 0.001}]},
        {"type": "way", "nodes": [2, 3],                      # node 2 shared with way 1
         "geometry": [{"lat": 0.0, "lon": 0.001}, {"lat": 0.001, "lon": 0.001}]},
        {"type": "node", "id": 9},                            # non-way elements ignored
        {"type": "way", "nodes": [7], "geometry": [{"lat": 5, "lon": 5}]},  # degenerate
    ]}
    g = R.graph_from_overpass(osm)
    assert len(g["nodes"]) == 3                               # 1, 2, 3 — node 2 deduped
    assert g["ways"] == [[0, 1], [1, 2]]


def test_road_graph_roundtrip(tmp_path):
    g = {"bbox": [0, 0, 1, 1], "classes": ["residential"],
         "nodes": [[0.0, 0.0], [0.0, 0.001]], "ways": [[0, 1]]}
    R.save_road_graph(str(tmp_path), g)
    assert R.load_road_graph(str(tmp_path)) == g
    assert R.load_road_graph(str(tmp_path / "nope")) is None


def test_prepare_prunes_and_keeps_largest_component():
    z = np.zeros((10, 10))
    graph, _ = _graph([
        [(1, 1), (1, 2), (1, 3), (1, 4)],                     # big component (4 nodes)
        [(8, 1), (8, 2)],                                     # small fragment (2 nodes)
        [(5, 5), (5, 50)],                                    # second node outside -> edge dropped
    ])
    net = R.prepare(graph, z, _cell_of(z.shape))
    assert int(net.alive.sum()) == 4                          # only the big chain survives
    dead = [i for i in range(net.n) if not net.alive[i]]
    assert all(net.adj[i] == [] for i in dead)


def test_reverse_dijkstra_prefers_downhill_outfall():
    # inlet `a` sits between a NEAR but UPHILL outfall and a FAR but DOWNHILL one
    z = np.zeros((1, 9))
    z[0, 0] = 5.0                                             # uphill outfall's cell
    z[0, 2] = 1.0                                             # inlet a
    z[0, 4] = 0.5
    z[0, 8] = 0.0                                             # downhill outfall's cell
    graph, idx = _graph([[(0, 0), (0, 2), (0, 4), (0, 8)]])
    net = R.prepare(graph, z, _cell_of(z.shape))
    dist, succ = R.reverse_dijkstra(net, [idx[(0, 0)], idx[(0, 8)]])
    a = idx[(0, 2)]
    path = R.inlet_path(succ, a)
    assert path[-1] == idx[(0, 8)]                            # 2 climbing cells would cost 2400*4 m
    assert np.isfinite(dist[a])


def test_snap_respects_radius():
    z = np.zeros((12, 12))
    graph, idx = _graph([[(5, 5), (5, 6)]])
    net = R.prepare(graph, z, _cell_of(z.shape))
    assert R.snap(net, (7, 7), max_cells=3) in (idx[(5, 5)], idx[(5, 6)])
    assert R.snap(net, (11, 11), max_cells=3) is None


def test_nodes_to_cells_bridge_and_4_connectivity():
    z = np.zeros((12, 12))
    graph, idx = _graph([[(0, 0), (0, 5), (5, 5)]])           # long edges: densification must fill
    net = R.prepare(graph, z, _cell_of(z.shape))
    cells = R.nodes_to_cells(net, [idx[(0, 0)], idx[(0, 5)], idx[(5, 5)]],
                             inlet_cell=(3, 0), z_np=z)
    assert cells[0] == (3, 0)                                 # starts at the flood inlet
    assert cells[-1] == (5, 5)
    for a, b in zip(cells, cells[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1       # 4-connected, no jumps, no repeats


def test_build_network_accumulates_shared_trunk():
    # two branches (from (0,0) and (0,4)) merge at (2,2) then share a trunk to the outfall (6,2)
    z = np.zeros((8, 8))
    graph, idx = _graph([
        [(0, 0), (1, 1), (2, 2)],
        [(0, 4), (1, 3), (2, 2)],
        [(2, 2), (4, 2), (6, 2)],
    ])
    net = R.prepare(graph, z, _cell_of(z.shape))
    dist, succ = R.reverse_dijkstra(net, [idx[(6, 2)]])
    inlets = [{"node": idx[(0, 0)], "cell": (0, 0), "drained_m3": 100.0, "basin": 0},
              {"node": idx[(0, 4)], "cell": (0, 4), "drained_m3": 50.0, "basin": 0}]
    netw = R.build_network(net, inlets, succ, dist)
    drains = sorted(round(s["drained_m3"]) for s in netw["segments"])
    assert drains == [50, 100, 150]                           # branch, branch, trunk = sum
    trunk = max(netw["segments"], key=lambda s: s["drained_m3"])
    assert trunk["nodes"][0] == idx[(2, 2)] and trunk["nodes"][-1] == idx[(6, 2)]
