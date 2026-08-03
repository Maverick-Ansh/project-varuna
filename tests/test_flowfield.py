"""Flow-field reconstruction: does the water point the way the ground says it should?

Synthetic terrain only — no bundle, no rasterio, no emulator. The bundle driver
(`flow_for_area`) is exercised by the API smoke tests.
"""
import math

import numpy as np

from varuna.serve.flowfield import (arrow_points, bearing_deg, road_flow, surface_velocity,
                                    velocity_layer)

DX = 60.0
MANN = 0.03


def _plane(n=32, d_drow=0.0, d_dcol=0.0, base=50.0):
    """Bed elevation as a tilted plane: z = base + d_drow*row + d_dcol*col (metres per cell)."""
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return base + d_drow * rr + d_dcol * cc


def _uniform(n=32, depth=0.5):
    return np.full((n, n), depth)


def _interior(a, pad=2):
    return a[pad:-pad, pad:-pad]


def test_bearing_conventions():
    assert bearing_deg(0.0, 1.0) == 0.0        # due north
    assert bearing_deg(1.0, 0.0) == 90.0       # due east
    assert bearing_deg(0.0, -1.0) == 180.0     # due south
    assert bearing_deg(-1.0, 0.0) == 270.0     # due west


def test_flows_east_when_ground_falls_east():
    """Bed dropping toward +col must push water toward +col, i.e. bearing ~90 (east)."""
    z = _plane(d_dcol=-0.5)                    # elevation decreases eastward
    h = _uniform()
    u, v = surface_velocity(h, z, np.full_like(z, MANN), DX)
    assert (_interior(u) > 0).all()            # eastward everywhere
    assert np.allclose(_interior(v), 0.0, atol=1e-9)
    assert abs(bearing_deg(float(u[16, 16]), float(v[16, 16])) - 90.0) < 1e-6


def test_flows_south_when_ground_falls_south():
    """Rows increase southward, so a bed dropping with row must give a NEGATIVE north component."""
    z = _plane(d_drow=-0.5)
    h = _uniform()
    u, v = surface_velocity(h, z, np.full_like(z, MANN), DX)
    assert (_interior(v) < 0).all()
    assert np.allclose(_interior(u), 0.0, atol=1e-9)
    assert abs(bearing_deg(float(u[16, 16]), float(v[16, 16])) - 180.0) < 1e-6


def test_manning_speed_matches_closed_form():
    """On a uniform plane the speed must equal (1/n) h^(2/3) sqrt(S) exactly."""
    slope_per_cell = 0.6
    z = _plane(d_dcol=-slope_per_cell)
    h = _uniform(depth=0.4)
    u, v = surface_velocity(h, z, np.full_like(z, MANN), DX)
    s = slope_per_cell / DX                                  # metres per metre
    expect = (1.0 / MANN) * 0.4 ** (2.0 / 3.0) * math.sqrt(s)
    got = math.hypot(float(u[16, 16]), float(v[16, 16]))
    assert abs(got - expect) < 1e-9


def test_dry_cells_do_not_flow():
    z = _plane(d_dcol=-0.5)
    h = np.zeros((32, 32))
    h[10:20, 10:20] = 0.5
    u, v = surface_velocity(h, z, np.full_like(z, MANN), DX)
    speed = np.hypot(u, v)
    assert speed[0, 0] == 0.0
    assert (speed[10:20, 10:20] > 0).all()


def test_water_surface_beats_bed_inside_a_pond():
    """A filled depression has a FLAT surface, so the water must crawl, not race to the bottom.

    This is the whole reason the gradient is taken on z+h and not on z: a bed-only gradient would
    report fast convergent flow inside a pond that is actually still.
    """
    n = 32
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    bowl = -3.0 * np.exp(-(((rr - 16) ** 2 + (cc - 16) ** 2) / 40.0))
    z = 50.0 + bowl
    h = np.clip(-bowl, 0.0, None)                            # fills the bowl exactly level
    mann = np.full_like(z, MANN)

    centre = (slice(12, 21), slice(12, 21))

    u_surf, v_surf = surface_velocity(h, z, mann, DX)
    still = np.hypot(u_surf, v_surf)[centre].max()
    assert still < 1e-6, f"pond interior should be still, got {still} m/s"

    # counter-check: give those same cells a uniform sheet of water instead of a levelling one
    # and the bed gradient reappears — so the stillness above came from the h term, not from
    # the cells being dry or the slope being absent.
    u_sheet, v_sheet = surface_velocity(np.full_like(h, 0.3), z, mann, DX)
    assert np.hypot(u_sheet, v_sheet)[centre].max() > 0.05


def test_speed_is_capped():
    z = _plane(d_dcol=-500.0)                                # absurd cliff (DEM spike)
    u, v = surface_velocity(_uniform(depth=3.0), z, np.full_like(z, MANN), DX, max_speed=5.0)
    assert np.hypot(u, v).max() <= 5.0 + 1e-9


def _latlon(r, c):
    return [25.6 - 0.0005 * r, 85.1 + 0.0005 * c]


def test_arrow_points_ranks_and_caps():
    """On a uniformly wet plane every arrow points downhill and the list is strongest-first."""
    z = _plane(d_dcol=-0.5)
    h = _uniform(depth=0.5)
    u, v = surface_velocity(h, z, np.full_like(z, MANN), DX)

    arrows = arrow_points(u, v, h, _latlon, step=4)
    assert arrows, "expected arrows over the wet plane"
    speeds = [a["speed_ms"] for a in arrows]
    assert speeds == sorted(speeds, reverse=True)             # strongest first
    assert all(abs(a["bearing"] - 90.0) < 1e-6 for a in arrows)
    assert len(arrow_points(u, v, h, _latlon, step=1, max_arrows=5)) == 5


def test_arrow_points_skips_dry_ground():
    """Only the wet block gets arrows — and its RIM legitimately spills sideways, so the
    bearing check belongs on the interior, not the edge."""
    z = _plane(d_dcol=-0.5)
    h = np.zeros((32, 32))
    h[8:24, 8:24] = 0.5
    u, v = surface_velocity(h, z, np.full_like(z, MANN), DX)

    arrows = arrow_points(u, v, h, _latlon, step=1)
    assert arrows
    assert all(a["depth_m"] >= 0.05 for a in arrows)
    lats = [a["latlon"][0] for a in arrows]
    lons = [a["latlon"][1] for a in arrows]
    assert min(lats) >= _latlon(23, 0)[0] - 1e-9              # nothing north/south of the block
    assert max(lats) <= _latlon(8, 0)[0] + 1e-9
    assert min(lons) >= _latlon(0, 8)[1] - 1e-9
    assert max(lons) <= _latlon(0, 23)[1] + 1e-9
    # interior cells (away from the sheet's rim) still run due east
    interior = [a for a in arrows
                if _latlon(21, 0)[0] <= a["latlon"][0] <= _latlon(10, 0)[0]
                and _latlon(0, 10)[1] <= a["latlon"][1] <= _latlon(0, 21)[1]]
    assert interior
    assert all(abs(a["bearing"] - 90.0) < 1e-6 for a in interior)


def test_velocity_layer_header_matches_grid():
    u = np.zeros((8, 10))
    v = np.ones((8, 10))
    bounds = [[25.0, 85.0], [25.5, 85.5]]
    layer = velocity_layer(u, v, bounds)
    assert len(layer) == 2
    hu, hv = layer[0]["header"], layer[1]["header"]
    assert hu["parameterNumber"] == 2 and hv["parameterNumber"] == 3
    assert hu["nx"] == 10 and hu["ny"] == 8
    assert hu["la1"] == 25.5 and hu["lo1"] == 85.0           # starts at the NORTH-WEST corner
    assert len(layer[0]["data"]) == 80 and len(layer[1]["data"]) == 80


class _Net:
    """Minimal RoadNet stand-in: a chain of nodes with unit adjacency."""

    def __init__(self, lat, lon, cells, length=50.0):
        self.lat = np.asarray(lat, dtype="float64")
        self.lon = np.asarray(lon, dtype="float64")
        self.cell = cells
        self.n = len(lat)
        self.alive = np.ones(self.n, dtype=bool)
        self.adj = [[] for _ in range(self.n)]
        for i in range(self.n - 1):
            self.adj[i].append((i + 1, length))
            self.adj[i + 1].append((i, length))


def test_road_flow_points_the_arrow_downstream_whichever_way_the_street_is_listed():
    """The arrow follows the WATER, not the order OSM happened to store the vertices in."""
    h = np.full((8, 8), 0.4)
    u = np.full((8, 8), 1.0)                                 # 1 m/s due east
    v = np.zeros((8, 8))

    west_to_east = _Net([25.60, 25.60], [85.10, 85.11], [(4, 3), (4, 5)])
    got = road_flow(west_to_east, u, v, h)
    assert len(got) == 1
    assert abs(got[0]["bearing"] - 90.0) < 1e-6              # arrow points east
    assert abs(got[0]["speed_ms"] - 1.0) < 1e-6
    assert abs(got[0]["flux_m2s"] - 0.4) < 1e-6

    east_to_west = _Net([25.60, 25.60], [85.11, 85.10], [(4, 5), (4, 3)])
    flipped = road_flow(east_to_west, u, v, h)
    assert abs(flipped[0]["bearing"] - 90.0) < 1e-6          # same physical direction


def test_road_flow_drops_cross_street_component():
    """Water crossing a road is not flow ALONG it — a north-south street under eastward flow is idle."""
    h = np.full((8, 8), 0.4)
    u = np.full((8, 8), 1.0)
    v = np.zeros((8, 8))
    ns_net = _Net([25.60, 25.61], [85.10, 85.10], [(3, 4), (5, 4)])
    assert road_flow(ns_net, u, v, h) == []


def test_road_flow_skips_dry_streets():
    h = np.zeros((8, 8))
    u = np.full((8, 8), 1.0)
    v = np.zeros((8, 8))
    net = _Net([25.60, 25.60], [85.10, 85.11], [(4, 3), (4, 5)])
    assert road_flow(net, u, v, h) == []


def test_road_flow_excludes_masked_cells():
    """A street arrow sitting ON permanent water is not a flooded street — drop it."""
    h = np.full((8, 8), 0.4)
    u = np.full((8, 8), 1.0)
    v = np.zeros((8, 8))
    net = _Net([25.60, 25.60], [85.10, 85.11], [(4, 3), (4, 5)])
    assert len(road_flow(net, u, v, h)) == 1
    water = np.zeros((8, 8), dtype=bool)
    water[4, 4] = True                                       # the segment's midpoint cell
    assert road_flow(net, u, v, h, exclude=water) == []


def test_road_flow_flags_but_keeps_cells_next_to_water():
    """Lakeside roads are SERVED and MARKED, never silently deleted: dropping them would hide
    real flooding, reporting them bare would claim the lake's depth is the street's."""
    h = np.full((8, 8), 0.4)
    u = np.full((8, 8), 1.0)
    v = np.zeros((8, 8))
    net = _Net([25.60, 25.60], [85.10, 85.11], [(4, 3), (4, 5)])
    near = np.zeros((8, 8), dtype=bool)
    near[4, 4] = True
    got = road_flow(net, u, v, h, flag=near)
    assert len(got) == 1
    assert got[0]["near_water"] is True
    assert "near_water" not in road_flow(net, u, v, h)[0]     # absent, not False, when clean


def test_road_flow_is_one_arrow_per_cell():
    """Many OSM vertices inside one 60 m cell must collapse to a single arrow — the velocity
    field has no sub-cell information, so drawing more would be invented precision."""
    h = np.full((8, 8), 0.4)
    u = np.full((8, 8), 1.0)
    v = np.zeros((8, 8))
    # six vertices marching east, all landing in the SAME domain cell (4, 4)
    lats = [25.60] * 6
    lons = [85.100 + 0.0001 * i for i in range(6)]
    net = _Net(lats, lons, [(4, 4)] * 6)
    got = road_flow(net, u, v, h)
    assert len(got) == 1, f"expected one arrow for one cell, got {len(got)}"


def test_road_flow_keeps_the_strongest_street_in_a_shared_cell():
    """When two streets share a cell, the one carrying more water wins the arrow."""
    h = np.full((8, 8), 0.4)
    h[4, 4] = 0.4
    u = np.zeros((8, 8))
    v = np.zeros((8, 8))
    u[4, 4] = 1.0                                            # flow is due east in that cell
    # node 0-1: an east-west street (fully aligned); node 2-3: a diagonal (partly aligned)
    net = _Net([25.60, 25.60, 25.60, 25.6007], [85.100, 85.101, 85.102, 85.103],
               [(4, 4)] * 4)
    got = road_flow(net, u, v, h)
    assert len(got) == 1
    assert abs(got[0]["bearing"] - 90.0) < 1e-6              # the aligned street won
