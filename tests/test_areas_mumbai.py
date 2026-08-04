"""Mumbai tile registry invariants + per-area grid-size threading."""
import math

import pytest

from varuna.areas import list_areas, CITIES

TILES = [a for a in list_areas() if a.city == "mumbai"]
HOTSPOTS = {"hindmata": (19.008, 72.844), "milan_subway": (19.078, 72.836),
            "kurla": (19.070, 72.880), "powai": (19.120, 72.905), "malad": (19.190, 72.840),
            "mulund": (19.172, 72.956), "chembur_trombay": (19.010, 72.970)}


def _crop_box(a):
    la, lo = a.center
    hl = a.n_grid * 60 / 2 / 111320.0
    ho = a.n_grid * 60 / 2 / (111320.0 * math.cos(math.radians(la)))
    return (la - hl, lo - ho, la + hl, lo + ho)


def test_registry_shape():
    assert len(TILES) == 6
    assert all(a.n_grid == 256 and a.dx in (None, 60.0) for a in TILES)
    assert set(CITIES["mumbai"]["tiles"]) == {a.id for a in TILES}


def test_grid_is_complete_with_no_gaps():
    """Every cell of the declared 2 x 3 grid holds exactly one tile.

    A gap here is invisible until the city mosaic quietly fills it with sea (which is what the
    first four tiles did to Mulund and the harbour), so the registry has to prove completeness.
    """
    grid = CITIES["mumbai"]["grid"]
    lat_e, lon_e = grid["lat_edges"], grid["lon_edges"]
    cells = [(0.5 * (lat_e[r] + lat_e[r + 1]), 0.5 * (lon_e[c] + lon_e[c + 1]))
             for r in range(len(lat_e) - 1) for c in range(len(lon_e) - 1)]
    assert len(cells) == len(TILES)
    for la, lo in cells:
        owners = [a.id for a in TILES if abs(a.center[0] - la) < 1e-3
                  and abs(a.center[1] - lo) < 1e-3]
        assert len(owners) == 1, f"grid cell ({la:.4f}, {lo:.4f}) has owners {owners}"


def test_centers_inside_aoi_and_crop_inside_aoi():
    for a in TILES:
        lon0, lat0, lon1, lat1 = a.aoi
        la, lo = a.center
        assert lat0 < la < lat1 and lon0 < lo < lon1
        b = _crop_box(a)
        assert lat0 <= b[0] and b[2] <= lat1, f"{a.id} lat crop exceeds AOI"
        assert lon0 <= b[1] and b[3] <= lon1, f"{a.id} lon crop exceeds AOI"


def test_hotspots_covered_exactly_once():
    for name, (la, lo) in HOTSPOTS.items():
        hits = [a.id for a in TILES
                if _crop_box(a)[0] <= la <= _crop_box(a)[2]
                and _crop_box(a)[1] <= lo <= _crop_box(a)[3]]
        assert len(hits) == 1, f"{name}: {hits}"


def test_tiles_do_not_overlap():
    """Overlap must stay below half a cell, measured in METRES.

    A degree tolerance is latitude-dependent and lies: the grid's longitude edges are spaced at
    a fixed 0.1460 deg while a 256 x 60 m tile spans 0.1460 deg only at ~19.10 N, so the northern
    row overlaps its neighbour by ~17 m. That is a quarter of a 60 m cell — below the resolution
    at which "which tile owns this ground" is even defined — but it is not zero, so city-wide
    sums over tiles carry that much double-count.
    """
    boxes = [_crop_box(a) for a in TILES]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            lat_m = (min(a[2], b[2]) - max(a[0], b[0])) * 111320.0
            lat_mid = 0.5 * (TILES[i].center[0] + TILES[j].center[0])
            lon_m = ((min(a[3], b[3]) - max(a[1], b[1]))
                     * 111320.0 * math.cos(math.radians(lat_mid)))
            half_cell = 0.5 * 60.0
            assert lat_m <= half_cell or lon_m <= half_cell, (
                TILES[i].id, TILES[j].id, round(lat_m, 1), round(lon_m, 1))


def test_build_domain_explicit_n_grid(synth_bundle):
    pytest.importorskip("torch")
    from varuna.build.twin import build_domain
    dom = build_domain(synth_bundle, center=(25.6255, 85.0845), n_grid=64)
    assert dom.N == 64 and dom.z0.shape == (64, 64)
    dom2 = build_domain(synth_bundle, center=(25.6255, 85.0845))
    assert dom2.N == 128                              # CFG default preserved


def test_emulator_honors_bundle_n_grid(synth_bundle):
    """Regression: load_emulator passed center-only to build_domain, which skipped twin_meta
    and silently cropped 256^2 bundles down to CFG.n_grid=128 (the Mumbai gnn_data crash)."""
    torch = pytest.importorskip("torch")
    from varuna.build.twin import UNet
    from varuna.serve import emulator
    meta = dict(sites=[(5, 5)], zmean=50.0, zstd=1.0, row0=0, col0=0, dx=60.0, n_grid=64,
                val_rmse_m=0.0, center=[25.6255, 85.0845])
    torch.save(meta, f"{synth_bundle}/twin_meta.pt")
    torch.save(UNet().state_dict(), f"{synth_bundle}/emulator.pt")
    emulator._CACHE.clear()
    hmax, _dig, summary = emulator.whatif_grid(80.0, work=synth_bundle)
    assert hmax.shape == (64, 64)
    emulator._CACHE.clear()
