"""Shared pytest fixtures: a synthetic artifact bundle so tests need no Earth Engine."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
from rasterio.transform import Affine  # noqa: E402

N = 256
PIX = 0.00027  # ~30 m in degrees
LON0, LAT0 = 85.05, 25.66  # top-left
TRANSFORM = Affine(PIX, 0, LON0, 0, -PIX, LAT0)


def _write(path, arr, dtype):
    meta = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                crs="EPSG:4326", transform=TRANSFORM, dtype=dtype)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr.astype(dtype), 1)


@pytest.fixture
def synth_bundle(tmp_path, monkeypatch):
    """Create a minimal 256x256 bundle in tmp_path and point CFG.work at it."""
    import pandas as pd
    from varuna.config import CFG

    work = str(tmp_path)

    # bed: gentle tilt + a bowl (a sink) near (row 64, col 80)
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    dem = 50.0 + 0.002 * 30.0 * xx - 1.5 * np.exp(-(((yy - 64) ** 2 + (xx - 80) ** 2) / 200.0))
    _write(f"{work}/dem.tif", dem, "float32")

    # ponding depth: deep at the bowl
    depth = np.clip(1.5 * np.exp(-(((yy - 64) ** 2 + (xx - 80) ** 2) / 200.0)), 0, None)
    _write(f"{work}/depth.tif", depth, "float32")
    _write(f"{work}/acc.tif", np.abs(np.random.default_rng(0).normal(10, 5, (N, N))), "float32")

    # land cover: mostly cropland (10), a built-up block (50), a water block (80)
    wc = np.full((N, N), 10, dtype="uint8")
    wc[100:160, 100:160] = 50
    wc[0:30, 0:30] = 80
    _write(f"{work}/worldcover.tif", wc, "uint8")
    _write(f"{work}/jrc_occurrence.tif", (wc == 80).astype("uint8") * 100, "uint8")
    _write(f"{work}/clay.tif", np.full((N, N), 300, dtype="float32"), "float32")  # 30% clay -> group C

    # two catchments
    labels = np.zeros((N, N), dtype="int32")
    labels[40:90, 60:100] = 1
    labels[120:160, 120:160] = 2
    _write(f"{work}/catchment_labels.tif", labels, "int32")

    # sinks.csv (row/col + lat/lon consistent with TRANSFORM)
    def ll(r, c):
        lon, lat = TRANSFORM * (c, r)
        return lat, lon
    rows = []
    for sid, (r, c, vol) in {1: (64, 80, 5000.0), 2: (140, 140, 20000.0)}.items():
        lat, lon = ll(r, c)
        rows.append(dict(sink_id=sid, lat=lat, lon=lon, row=r, col=c,
                         area_m2=900 * 50, max_depth_m=1.5, volume_m3=vol, catchment_m2=900 * 2000))
    pd.DataFrame(rows).to_csv(f"{work}/sinks.csv", index=False)

    monkeypatch.setattr(CFG, "work", work)
    return work
