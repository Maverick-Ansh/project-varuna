"""Exposure overlay tests — offline: fake OSM geometries + a synthetic flood grid.

Exercises the depth sampler (lat/lon -> domain cell), the building/road classification, and
the bundle cache, without osmnx or a trained emulator.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rasterio")

from varuna.serve import exposure  # noqa: E402

# conftest synth bundle: 256x256 30 m grid, top-left (85.05, 25.66), PIX=0.00027
PIX, LON0, LAT0 = 0.00027, 85.05, 25.66


def _ll(row30, col30):
    return LAT0 - row30 * PIX, LON0 + col30 * PIX


class _Pt:
    def __init__(self, lon, lat):
        self.x, self.y = lon, lat


class _Poly:
    geom_type = "Polygon"

    def __init__(self, lon, lat):
        self.centroid = _Pt(lon, lat)


class _Line:
    geom_type = "LineString"

    def __init__(self, coords):          # [(lon, lat), ...]
        self.coords = coords
        self.centroid = _Pt(*coords[0])


@pytest.fixture
def fake_flood(monkeypatch, synth_bundle):
    """128x128 domain over the whole synth DEM; wet cell at (5,5), dry elsewhere."""
    hmax = np.zeros((128, 128), dtype="float32")
    hmax[5, 5] = 1.0
    dom = SimpleNamespace(N=128, row0=0, col0=0)
    monkeypatch.setattr(exposure, "_flood_grid", lambda rain, work, device: (hmax, dom))
    return synth_bundle


def test_assess_exposure_classifies(monkeypatch, fake_flood):
    wet_lat, wet_lon = _ll(10, 10)       # 30m cell (10,10) -> domain cell (5,5) = wet
    dry_lat, dry_lon = _ll(40, 40)       # -> domain cell (20,20) = dry

    def fake_osm(north, south, east, west, tags):
        if "building" in tags:
            return SimpleNamespace(geometry=[
                _Poly(wet_lon, wet_lat), _Poly(dry_lon, dry_lat),
                _Poly(0.0, 0.0),                       # outside the domain -> ignored
            ])
        return SimpleNamespace(geometry=[
            _Line([(wet_lon, wet_lat), (dry_lon, dry_lat)]),   # crosses the wet cell
            _Line([(dry_lon, dry_lat), _ll(41, 41)[::-1]]),    # stays dry
        ])

    monkeypatch.setattr(exposure, "_osm_features", fake_osm)
    out = exposure.assess_exposure(rain_mm=100, work=fake_flood)

    assert out["rain_mm"] == 100
    assert out["buildings"]["total_in_domain"] == 2
    assert out["buildings"]["at_risk"] == 1
    assert out["buildings"]["points"][0]["depth_m"] == pytest.approx(1.0)
    assert out["roads"]["total"] == 2
    assert out["roads"]["flooded"] == 1
    assert len(out["roads"]["dry_lines"]) == 1


def test_cache_roundtrip(monkeypatch, fake_flood):
    monkeypatch.setattr(exposure, "_osm_features",
                        lambda *a, **k: SimpleNamespace(geometry=[]))
    saved = exposure.save_exposure(rain_mm=60, work=fake_flood)
    cached = exposure.load_cached(fake_flood)
    assert cached is not None
    assert cached["rain_mm"] == saved["rain_mm"] == 60
    assert cached["buildings"]["total_in_domain"] == 0


def test_load_cached_missing(tmp_path):
    assert exposure.load_cached(str(tmp_path)) is None
