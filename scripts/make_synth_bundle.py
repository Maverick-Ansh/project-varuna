"""Write a synthetic artifact bundle into CFG.work so the serve/agent layers run WITHOUT
Earth Engine. For demos and the no-GEE validation run; replace with a real build for real numbers.

    python scripts/make_synth_bundle.py            # -> CFG.work
    python scripts/make_synth_bundle.py /tmp/work  # -> explicit dir
"""
from __future__ import annotations

import sys

import numpy as np

N = 256
PIX = 0.00027  # ~30 m in degrees
LON0, LAT0 = 85.05, 25.66  # top-left of the synthetic AOI


def make(work):
    import os
    import pandas as pd
    import rasterio
    from rasterio.transform import Affine

    os.makedirs(work, exist_ok=True)
    T = Affine(PIX, 0, LON0, 0, -PIX, LAT0)

    def write(name, arr, dtype):
        meta = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                    crs="EPSG:4326", transform=T, dtype=dtype)
        with rasterio.open(f"{work}/{name}", "w", **meta) as dst:
            dst.write(arr.astype(dtype), 1)

    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    dem = 50.0 + 0.002 * 30.0 * xx - 1.5 * np.exp(-(((yy - 64) ** 2 + (xx - 80) ** 2) / 200.0))
    write("dem.tif", dem, "float32")
    write("depth.tif", np.clip(1.5 * np.exp(-(((yy - 64) ** 2 + (xx - 80) ** 2) / 200.0)), 0, None), "float32")
    write("acc.tif", np.abs(np.random.default_rng(0).normal(10, 5, (N, N))), "float32")

    wc = np.full((N, N), 10, dtype="uint8")
    wc[100:160, 100:160] = 50   # built-up
    wc[0:30, 0:30] = 80         # water
    write("worldcover.tif", wc, "uint8")
    write("jrc_occurrence.tif", (wc == 80).astype("uint8") * 100, "uint8")
    write("clay.tif", np.full((N, N), 300, dtype="float32"), "float32")

    labels = np.zeros((N, N), dtype="int32")
    labels[40:90, 60:100] = 1
    labels[120:160, 120:160] = 2
    write("catchment_labels.tif", labels, "int32")

    rows = []
    for sid, (r, c, vol) in {1: (64, 80, 5000.0), 2: (140, 140, 20000.0)}.items():
        lon, lat = T * (c, r)
        rows.append(dict(sink_id=sid, lat=lat, lon=lon, row=r, col=c,
                         area_m2=900 * 50, max_depth_m=1.5, volume_m3=vol, catchment_m2=900 * 2000))
    pd.DataFrame(rows).to_csv(f"{work}/sinks.csv", index=False)
    print("wrote synthetic bundle ->", work)


if __name__ == "__main__":
    from varuna.config import CFG
    make(sys.argv[1] if len(sys.argv) > 1 else CFG.work)
