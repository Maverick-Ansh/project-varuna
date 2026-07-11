"""Mumbai tile registry invariants + per-area grid-size threading."""
import math

import pytest

from varuna.areas import list_areas, CITIES

TILES = [a for a in list_areas() if a.city == "mumbai"]
HOTSPOTS = {"hindmata": (19.008, 72.844), "milan_subway": (19.078, 72.836),
            "kurla": (19.070, 72.880), "powai": (19.120, 72.905), "malad": (19.190, 72.840)}


def _crop_box(a):
    la, lo = a.center
    hl = a.n_grid * 60 / 2 / 111320.0
    ho = a.n_grid * 60 / 2 / (111320.0 * math.cos(math.radians(la)))
    return (la - hl, lo - ho, la + hl, lo + ho)


def test_registry_shape():
    assert len(TILES) == 4
    assert all(a.n_grid == 256 and a.dx in (None, 60.0) for a in TILES)
    assert set(CITIES["mumbai"]["tiles"]) == {a.id for a in TILES}


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
    boxes = [_crop_box(a) for a in TILES]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            lat_olap = min(a[2], b[2]) - max(a[0], b[0])
            lon_olap = min(a[3], b[3]) - max(a[1], b[1])
            assert lat_olap <= 1e-4 or lon_olap <= 1e-4, (TILES[i].id, TILES[j].id)


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
