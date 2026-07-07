"""FloodGNN offline tests — synthetic graphs only, no bundle / Earth Engine / rasterio."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from varuna.gnn.evaluate import auc                                      # noqa: E402
from varuna.gnn.graph import EDGE_FEATURES, NODE_FEATURES, StreetGraph, graph_tensors  # noqa: E402
from varuna.gnn.model import FloodGNN                                    # noqa: E402
from varuna.gnn.planner import _dijkstra, _nearest_node                  # noqa: E402
from varuna.serve import roadnet                                         # noqa: E402


# ------------------------------------------------------------------ helpers

def _grid_road_graph(k=6, lat0=25.60, lon0=85.10, step=6e-4):
    """A k x k Manhattan street grid as the persisted road-graph artifact."""
    nodes = [[lat0 + i * step, lon0 + j * step] for i in range(k) for j in range(k)]
    ways = []
    for i in range(k):
        ways.append([i * k + j for j in range(k)])           # rows
        ways.append([j * k + i for j in range(k)])           # columns
    return {"bbox": None, "classes": [], "nodes": nodes, "ways": ways}


def _prepare_net(k=6, tilt=0.5):
    """RoadNet over a synthetic DEM: cell = (i, j) straight from the node index."""
    g = _grid_road_graph(k)
    z = 50.0 + tilt * np.arange(k * 2, dtype="float64")[None, :].repeat(k * 2, 0)

    lat0, lon0, step = 25.60, 85.10, 6e-4

    def cell_of(lat, lon):
        i = int(round((lat - lat0) / step))
        j = int(round((lon - lon0) / step))
        return (i, j) if 0 <= i < k * 2 and 0 <= j < k * 2 else None

    return roadnet.prepare(g, z, cell_of), z


def _toy_streetgraph():
    """Two routes 0->3: direct top corridor (short) and bottom detour (long).

        0 --100m-- 1 --100m-- 3
        |                     |
        250m                  250m
        |                     |
        2 --------250m------- 4   (indices: 2-4 the long way)
    """
    lat = np.array([25.600, 25.6005, 25.598, 25.6010, 25.598])
    lon = np.array([85.100, 85.1010, 85.100, 85.1020, 85.102])
    und = [(0, 1, 100.0), (1, 3, 100.0), (0, 2, 250.0), (2, 4, 250.0), (4, 3, 250.0)]
    src, dst, eattr = [], [], []
    for u, v, L in und:
        for a, b in ((u, v), (v, u)):
            src.append(a)
            dst.append(b)
            eattr.append((L / 1000.0, 0.0, 0.0))
    x = torch.zeros(5, len(NODE_FEATURES))
    sg = StreetGraph(x, torch.tensor([src, dst]), torch.tensor(eattr, dtype=torch.float32),
                     lat, lon, np.zeros((5, 2), dtype="int64"), np.zeros(5),
                     np.arange(5, dtype="int64"))
    return sg, und


# ------------------------------------------------------------------ graph

def test_graph_tensors_from_roadnet():
    net, z = _prepare_net(k=6)
    grids = {"z": z, "built": np.ones_like(z), "dx": 60.0}
    sg = graph_tensors(net, grids)
    assert sg.n == 36 and sg.x.shape == (36, len(NODE_FEATURES))
    assert sg.edge_attr.shape == (sg.e, len(EDGE_FEATURES))
    assert bool(torch.isfinite(sg.x).all()) and bool(torch.isfinite(sg.edge_attr).all())
    # a Manhattan grid interior node has 4 street neighbours -> 4 directed out-edges
    deg = torch.zeros(sg.n).index_add_(0, sg.edge_index[0], torch.ones(sg.e))
    assert float(deg.max()) == 4.0 and float(deg.min()) >= 2.0
    # tilted DEM -> east-pointing edges climb, west-pointing fall (signed dz feature)
    assert float(sg.edge_attr[:, 1].max()) > 0 and float(sg.edge_attr[:, 1].min()) < 0
    adj = sg.adjacency()
    assert all(length > 0 for nbrs in adj for _v, length in nbrs)


def test_graph_compacts_dead_nodes():
    g = _grid_road_graph(k=4)
    g["nodes"].append([0.0, 0.0])                            # far outside -> cell_of None
    net, z = _prepare_net(k=4)
    sg = graph_tensors(net, {"z": z, "dx": 60.0})
    assert sg.n == 16
    assert (sg.compact_of >= -1).all() and sg.compact_of.max() == sg.n - 1


# ------------------------------------------------------------------ model

def test_model_heads_rain_and_gradients():
    net, z = _prepare_net()
    sg = graph_tensors(net, {"z": z, "dx": 60.0})
    model = FloodGNN(hidden=16, layers=2)
    out = model(sg.x, sg.edge_index, sg.edge_attr, 100.0)
    assert out["node_depth"].shape == (sg.n,) and out["edge_depth"].shape == (sg.e,)
    assert float(out["node_depth"].min()) >= 0.0             # softplus heads
    # FiLM: rainfall must change the prediction
    out2 = model(sg.x, sg.edge_index, sg.edge_attr, 220.0)
    assert float((out["node_depth"] - out2["node_depth"]).abs().mean()) > 1e-7
    # gradients reach the message MLPs
    out["edge_depth"].sum().backward()
    g = model.mp[0].msg[0].weight.grad
    assert g is not None and float(g.abs().sum()) > 0


def test_model_layers0_is_valid_ablation():
    net, z = _prepare_net()
    sg = graph_tensors(net, {"z": z, "dx": 60.0})
    out = FloodGNN(hidden=16, layers=0)(sg.x, sg.edge_index, sg.edge_attr, 100.0)
    assert out["node_depth"].shape == (sg.n,)


def test_training_reduces_loss():
    torch.manual_seed(0)
    net, z = _prepare_net(k=6)
    sg = graph_tensors(net, {"z": z, "dx": 60.0})
    # learnable rule: depth rises with rainfall where the standardized elevation is low
    def target(rain):
        return torch.relu(-sg.x[:, 0]) * rain / 200.0
    model = FloodGNN(hidden=16, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for step in range(60):
        rain = float(np.random.default_rng(step).uniform(40, 200))
        pred = model(sg.x, sg.edge_index, sg.edge_attr, rain)["node_depth"]
        loss = torch.nn.functional.huber_loss(pred, target(rain))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])


# ------------------------------------------------------------------ planner

def test_router_avoids_flooded_corridor():
    sg, und = _toy_streetgraph()
    ei = sg.edge_index.numpy()
    lengths = sg.edge_attr[:, 0].numpy() * 1000.0
    depth = np.zeros(sg.e)
    depth[(ei[0] == 1) | (ei[1] == 1)] = 0.8                 # top corridor under water

    def adjacency(with_risk):
        mult = 1.0 + (8.0 * np.minimum(depth, 1.5) + 25.0 * (depth > 0.3)) * with_risk
        adj = [[] for _ in range(sg.n)]
        for k in range(sg.e):
            adj[int(ei[0, k])].append((int(ei[1, k]), float(lengths[k] * mult[k]), k))
        return adj

    p_short, e_short = _dijkstra(sg.n, adjacency(False), 0, 3)
    p_safe, e_safe = _dijkstra(sg.n, adjacency(True), 0, 3)
    assert p_short == [0, 1, 3]                              # 200 m, straight through water
    assert p_safe == [0, 2, 4, 3]                            # 750 m detour, but dry
    assert depth[e_safe].max() == 0.0 and depth[e_short].max() > 0.3


def test_router_wades_when_no_dry_alternative():
    sg, _ = _toy_streetgraph()
    ei = sg.edge_index.numpy()
    lengths = sg.edge_attr[:, 0].numpy() * 1000.0
    depth = np.full(sg.e, 0.9)                               # everything under water
    mult = 1.0 + 8.0 * np.minimum(depth, 1.5) + 25.0
    adj = [[] for _ in range(sg.n)]
    for k in range(sg.e):
        adj[int(ei[0, k])].append((int(ei[1, k]), float(lengths[k] * mult[k]), k))
    p, _e = _dijkstra(sg.n, adj, 0, 3)
    assert p == [0, 1, 3]                                    # still routes; never refuses


def test_nearest_node_snapping():
    sg, _ = _toy_streetgraph()
    assert _nearest_node(sg, 25.6001, 85.1001) == 0
    assert _nearest_node(sg, 10.0, 70.0) is None             # far away -> no snap


# ------------------------------------------------------------------ metrics

def test_auc_properties():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert np.isnan(auc([0.5, 0.6], [1, 1]))                 # one-class -> undefined
