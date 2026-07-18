"""Multi-area build: produce one artifact bundle per registered Area.

Two paths, dispatched by whether the Area is a sub-crop:

* FULL area (e.g. Bengaluru) — runs the Earth Engine pipeline: sinks -> recharge -> twin. Needs
  EE auth + pysheds, and a GPU for the twin. Run this on Colab.
* SUB-CROP (Patna sub-regions) — reuses a source area's already-downloaded rasters and tables and
  only retrains the twin at a new crop centre. No Earth Engine, no pysheds, CPU-friendly. This is
  how we prove the whole any-area path on the committed Patna DEM.

CLI:  python -m varuna build --area <id> [--n-samples N] [--epochs E]
"""
from __future__ import annotations

import logging
import os
import shutil

from ..areas import get_area, list_areas

log = logging.getLogger("varuna.build.areas")

# Rasters / tables a sub-crop reuses verbatim from its source bundle (everything AOI-wide;
# the twin crop + per-area emulator are regenerated at the sub-crop's own centre).
_SHARED = ["dem.tif", "worldcover.tif", "jrc_occurrence.tif", "clay.tif", "sand.tif",
           "acc.tif", "depth.tif", "catchment_labels.tif", "rsi.tif",
           "sinks.csv", "sinks.geojson", "recharge_sites.csv", "gw_levels.csv"]


def _copy_shared(src, dst):
    os.makedirs(dst, exist_ok=True)
    copied = []
    for f in _SHARED:
        s = os.path.join(src, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, f))
            copied.append(f)
    log.info("copied %d/%d shared artifacts %s -> %s", len(copied), len(_SHARED), src, dst)
    return copied


def _drop_dataset(work):
    """The 57 MB twin_dataset.pt is a build byproduct, not part of the served bundle."""
    p = os.path.join(work, "twin_dataset.pt")
    if os.path.exists(p):
        os.remove(p)


def build_subcrop(area, n_samples=None, epochs=40, device=None):
    """Reuse the source bundle's rasters, retrain a per-area emulator at the sub-crop centre."""
    from .twin import train_twin
    src = area.source_dir()
    if not src or not os.path.isdir(src):
        raise FileNotFoundError(
            f"sub-crop '{area.id}' needs its source bundle; expected rasters at {src}. "
            f"Build the source area '{area.source_work}' first.")
    work = area.work_dir()
    copied = _copy_shared(src, work)
    if "dem.tif" not in copied or "worldcover.tif" not in copied:
        raise FileNotFoundError(f"source {src} lacks dem.tif/worldcover.tif — cannot build sub-crop")
    meta = train_twin(work=work, center=area.center, n_samples=n_samples, epochs=epochs,
                      device=device, n_grid=area.n_grid, dx=area.dx)
    _drop_dataset(work)
    log.info("sub-crop '%s' built at centre %s -> %s", area.id, area.center, work)
    return meta


# Per-stage completion markers for the resume guard: the LAST artifact each stage writes.
# A stage is only skipped when every marker exists — a half-finished stage re-runs.
_STAGE_DONE = {
    "sinks": ["dem.tif", "depth.tif", "catchment_labels.tif", "sinks.csv"],
    "recharge": ["sand.tif", "clay.tif", "rsi.tif", "recharge_sites.csv"],
    "twin": ["twin_meta.pt", "emulator.pt"],
}


def _stage_done(work, stage):
    return all(os.path.exists(os.path.join(work, f)) for f in _STAGE_DONE[stage])


def build_full(area, project_id=None, n_samples=None, epochs=40, device=None, steps=None,
               force=False):
    """Full Earth Engine build for a brand-new area (sinks -> recharge -> twin).

    Idempotent: a stage whose completion markers already exist is skipped (Colab/Kaggle
    sessions die mid-multi-tile-build; a rerun must resume, not redownload). `force=True`
    reruns every requested stage regardless."""
    from . import sinks, recharge, twin
    work = area.work_dir()
    os.makedirs(work, exist_ok=True)
    steps = steps or ["sinks", "recharge", "twin"]
    aoi = list(area.aoi)
    for stage in [s for s in ("sinks", "recharge", "twin") if s in steps]:
        if not force and _stage_done(work, stage):
            log.info("'%s' stage %s already complete — skipping (use force=True to rerun)",
                     area.id, stage)
            continue
        if stage == "sinks":
            sinks.run(work=work, aoi=aoi, project_id=project_id)
        elif stage == "recharge":
            recharge.run(work=work, aoi=aoi, project_id=project_id)
        else:
            twin.train_twin(work=work, center=area.center, n_samples=n_samples, epochs=epochs,
                            device=device, n_grid=area.n_grid, dx=area.dx)
            _drop_dataset(work)
    log.info("full build for '%s' complete -> %s", area.id, work)
    return work


def build_area(area_id, project_id=None, n_samples=None, epochs=40, device=None, steps=None,
               force=False):
    """Build one area's bundle, dispatching to the sub-crop or full-EE path."""
    area = get_area(area_id)
    if area.source_work:
        return build_subcrop(area, n_samples=n_samples, epochs=epochs, device=device)
    return build_full(area, project_id=project_id, n_samples=n_samples, epochs=epochs,
                      device=device, steps=steps, force=force)


def build_all(**kw):
    """Build every registered area that isn't built yet (skips ones with a complete bundle)."""
    from ..areas import is_built
    out = {}
    for a in list_areas():
        if is_built(a.id):
            log.info("area '%s' already built — skipping", a.id)
            continue
        out[a.id] = build_area(a.id, **kw)
    return out
