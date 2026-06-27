"""Notebook 01 -> functions: DEM -> hydrological conditioning -> ranked sinks -> catchments.

Outputs (into CFG.work): dem.tif, depth.tif, acc.tif, worldcover.tif, jrc_occurrence.tif,
catchment_labels.tif, sinks.csv, sinks.geojson, depth_overlay.png, sink_map.html.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from ..config import CFG
from ..ee_auth import download_ee_image, init_ee, region

log = logging.getLogger("varuna.build.sinks")


def download_layers(work=None, aoi=None, scale=None):
    """DEM (FABDEM -> Copernicus fallback), ESA WorldCover, JRC permanent water."""
    import ee
    work = work or CFG.work
    reg = region(aoi)
    scale = scale or CFG.scale
    try:
        dem_img = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").filterBounds(reg).mosaic()
        download_ee_image(dem_img, f"{work}/dem.tif", reg, scale)
        log.info("DEM source: FABDEM")
    except Exception as e:  # noqa: BLE001
        log.warning("FABDEM failed (%s) — falling back to Copernicus GLO-30", e)
        dem_img = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").filterBounds(reg).mosaic()
        download_ee_image(dem_img, f"{work}/dem.tif", reg, scale)
    download_ee_image(ee.ImageCollection("ESA/WorldCover/v200").first(), f"{work}/worldcover.tif", reg, scale)
    download_ee_image(ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0),
                      f"{work}/jrc_occurrence.tif", reg, scale)


def condition_dem(work=None):
    """Fill pits/depressions, resolve flats, D8 flow direction + accumulation.

    Returns (grid, depth, fdir, acc, transform). `depth` = filled - raw (ponding potential).
    """
    work = work or CFG.work
    from pysheds.grid import Grid
    import rasterio

    grid = Grid.from_raster(f"{work}/dem.tif")
    dem = grid.read_raster(f"{work}/dem.tif")
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    depth = np.asarray(flooded, dtype=np.float64) - np.asarray(dem, dtype=np.float64)
    inflated = grid.resolve_flats(flooded)
    fdir = grid.flowdir(inflated)
    acc = grid.accumulation(fdir)
    with rasterio.open(f"{work}/dem.tif") as src:
        transform = src.transform
    log.info("max ponding depth %.2f m | cells >%.2fm: %d",
             float(np.nanmax(depth)), CFG.min_depth_m, int((depth > CFG.min_depth_m).sum()))
    return grid, depth, fdir, acc, transform


def find_sinks(depth, transform, work=None, min_depth=None, min_cells=None, top_n=None):
    """Label ponding regions (not permanent water), rank by storable volume."""
    import rasterio
    from rasterio.transform import xy as px2coord
    from scipy import ndimage
    import pandas as pd

    work = work or CFG.work
    min_depth = CFG.min_depth_m if min_depth is None else min_depth
    min_cells = CFG.min_cells if min_cells is None else min_cells
    top_n = CFG.top_n if top_n is None else top_n
    cell_area = CFG.scale * CFG.scale

    with rasterio.open(f"{work}/jrc_occurrence.tif") as src:
        jrc = src.read(1)
    r = min(depth.shape[0], jrc.shape[0])
    c = min(depth.shape[1], jrc.shape[1])
    depth_a, jrc_a = depth[:r, :c], jrc[:r, :c]

    mask = (depth_a > min_depth) & (jrc_a < 50)
    labels, n = ndimage.label(mask)
    log.info("raw sink candidates: %d", n)

    rows = []
    for sid in range(1, n + 1):
        cells = labels == sid
        ncells = int(cells.sum())
        if ncells < min_cells:
            continue
        vol = float(depth_a[cells].sum() * cell_area)
        dmax = float(depth_a[cells].max())
        rr, cc = np.unravel_index(np.argmax(np.where(cells, depth_a, -1)), depth_a.shape)
        lon, lat = px2coord(transform, rr, cc)
        rows.append(dict(sink_id=sid, lat=lat, lon=lon, row=int(rr), col=int(cc),
                         area_m2=ncells * cell_area, max_depth_m=round(dmax, 2),
                         volume_m3=round(vol, 0)))
    sinks = pd.DataFrame(rows).sort_values("volume_m3", ascending=False).reset_index(drop=True)
    return sinks.head(top_n).copy(), (r, c)


def delineate_catchments(top, grid, fdir, depth, transform, shape, work=None):
    """D8 catchment per sink (larger sinks claim cells first); write rasters + sinks.csv/geojson."""
    import rasterio
    from ..io import write_raster

    work = work or CFG.work
    r, c = shape
    cell_area = CFG.scale * CFG.scale
    depth_a = depth[:r, :c]
    catch_labels = np.zeros((r, c), dtype=np.int32)
    catch_areas = {}
    for _, s in top.iterrows():
        try:
            cat = grid.catchment(x=s.lon, y=s.lat, fdir=fdir, xytype="coordinate")
            cat = np.asarray(cat, dtype=bool)[:r, :c]
        except Exception as e:  # noqa: BLE001
            log.warning("catchment failed for sink %s: %s", s.sink_id, e)
            continue
        free = cat & (catch_labels == 0)
        catch_labels[free] = int(s.sink_id)
        catch_areas[int(s.sink_id)] = float(cat.sum() * cell_area)

    top = top.copy()
    top["catchment_m2"] = top.sink_id.map(catch_areas).fillna(0)
    top.to_csv(f"{work}/sinks.csv", index=False)

    write_raster(f"{work}/catchment_labels.tif", catch_labels, transform, dtype="int32")
    write_raster(f"{work}/depth.tif", depth_a.astype("float32"), transform, dtype="float32")

    # sinks.geojson for GIS import
    feats = [dict(type="Feature",
                  geometry=dict(type="Point", coordinates=[float(s.lon), float(s.lat)]),
                  properties=dict(sink_id=int(s.sink_id), volume_m3=float(s.volume_m3),
                                  max_depth_m=float(s.max_depth_m),
                                  catchment_m2=float(s.catchment_m2)))
             for _, s in top.iterrows()]
    with open(f"{work}/sinks.geojson", "w") as f:
        json.dump(dict(type="FeatureCollection", features=feats), f)
    return top


def run(work=None, aoi=None, project_id=None, skip_download=False):
    """Full nb01 pipeline -> artifact bundle. Returns the ranked sinks DataFrame."""
    work = work or CFG.work
    init_ee(project_id)
    if not skip_download:
        download_layers(work, aoi)
    grid, depth, fdir, acc, transform = condition_dem(work)
    # persist accumulation for nb02
    from ..io import write_raster
    write_raster(f"{work}/acc.tif", np.asarray(acc, dtype="float32"), transform, dtype="float32")
    top, shape = find_sinks(depth, transform, work)
    top = delineate_catchments(top, grid, fdir, depth, transform, shape, work)
    log.info("nb01 done: %d sinks -> %s/sinks.csv", len(top), work)
    return top
