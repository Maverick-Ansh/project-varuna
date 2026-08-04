"""Offline tests for the tile-join core (no rasterio needed).

The raster mosaic (city_grid/mosaic/build_city_domain) needs real bundles and is exercised on
Colab; what is tested here is the placement arithmetic — the part that silently corrupts
everything when it is wrong — and the seam audit's shape/behaviour on synthetic domains.
"""
import numpy as np
import pytest
import torch

from varuna.build import city as C
from varuna.build.twin import Domain

SIM = dict(storm_hr=0.1, total_hr=0.3, dt=10.0)


def _info(grid=(40, 40), offsets=None):
    return {"grid": grid, "offsets60": offsets or {"a": (4, 6), "b": (4, 22)}}


def _tile(N=16, tilt=0.02, device="cpu"):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    z = (tilt * xx + 0.01 * yy).astype("float32")
    dom = Domain(z, np.full((N, N), 0.04, "float32"), np.full((N, N), 1e-6, "float32"),
                 np.ones((N, N), "float32"), dx=60.0, device=device)
    dom.wc = np.full((N, N), 50, dtype=np.int32)
    return dom


def test_place_then_crop_roundtrips():
    info = _info()
    city = np.zeros(info["grid"], dtype="float32")
    tile = np.arange(16 * 16, dtype="float32").reshape(16, 16)
    C.place(city, tile, "a", info)
    assert np.array_equal(C.crop(city, "a", info, 16), tile)


def test_place_honours_the_tile_crop_offset():
    """row0/col0 are in 30 m units and halve into the 60 m grid — the bug that misaligns all."""
    info = _info()
    city = np.zeros(info["grid"], dtype="float32")
    tile = np.ones((8, 8), dtype="float32")
    C.place(city, tile, "a", info, tile_row0=4, tile_col0=6)
    rows, cols = np.nonzero(city)
    assert rows.min() == 4 + 2 and cols.min() == 6 + 3     # offset60 + row0//2, col0//2


def test_place_does_not_disturb_its_neighbour():
    info = _info()
    city = np.zeros(info["grid"], dtype="float32")
    C.place(city, np.full((16, 16), 1.0, dtype="float32"), "a", info)
    C.place(city, np.full((16, 16), 2.0, dtype="float32"), "b", info)
    assert C.crop(city, "a", info, 16).mean() == 1.0
    assert C.crop(city, "b", info, 16).mean() == 2.0


def test_crop_matches_place_for_torch_tensors():
    info = _info()
    city = torch.zeros(info["grid"])
    tile = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16)
    C.place(city, tile, "b", info, tile_row0=2, tile_col0=2)
    assert torch.equal(C.crop(city, "b", info, 16, 2, 2), tile)


def test_city_drain_field_places_each_tile(tmp_path):
    info = _info()
    paths = {}
    for name, val in (("a", 5.0), ("b", 9.0)):
        p = tmp_path / f"{name}.pt"
        torch.save({"drain_mm_h": torch.full((16, 16), val)}, p)
        paths[name] = str(p)
    field = C.city_drain_field(paths, info)
    assert np.isclose(C.crop(field, "a", info, 16).mean() * 3.6e6, 5.0)
    assert np.isclose(C.crop(field, "b", info, 16).mean() * 3.6e6, 9.0)
    assert field.max() * 3.6e6 == pytest.approx(9.0)


@pytest.mark.slow
def test_seam_audit_reports_edge_and_interior():
    """A city domain that CONTAINS a tile's terrain: the tile's closed edges must disagree with
    the open city run more than its interior does."""
    N, pad = 24, 8
    city_z = np.zeros((N + 2 * pad, N + 2 * pad), dtype="float32")
    yy, xx = np.meshgrid(np.arange(N + 2 * pad), np.arange(N + 2 * pad), indexing="ij")
    city_z += (0.05 * xx).astype("float32")                 # steady tilt: water leaves westward
    city = Domain(city_z, np.full(city_z.shape, 0.04, "float32"),
                  np.full(city_z.shape, 1e-6, "float32"),
                  np.ones(city_z.shape, "float32"), dx=60.0, device="cpu")
    city.wc = np.full(city_z.shape, 50, dtype=np.int32)
    tile = Domain(city_z[pad:pad + N, pad:pad + N].copy(),
                  np.full((N, N), 0.04, "float32"), np.full((N, N), 1e-6, "float32"),
                  np.ones((N, N), "float32"), dx=60.0, device="cpu")
    tile.wc = np.full((N, N), 50, dtype=np.int32)
    info = {"grid": city_z.shape, "offsets60": {"t": (pad, pad)}}

    audit = C.seam_audit(city, info, {"t": tile}, rain_mm=40.0, edge_cells=4,
                         storm_hr=SIM["storm_hr"], total_hr=SIM["total_hr"])["t"]
    assert audit["mean_abs_dz_m"] == 0.0                    # same terrain, exactly placed
    assert audit["edge_band_mean_dh_cm"] > audit["interior_mean_dh_cm"]
    assert audit["p99_dh_cm"] >= 0.0
