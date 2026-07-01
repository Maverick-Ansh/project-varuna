"""Area registry — the set of locations Varuna can build a bundle for and serve.

Varuna began single-area: Patna is hard-wired in `config.py::CFG`. To answer "point at any
area", the serve / API layers select an `Area` by id; each Area names its own AOI, twin-crop
centre and artifact bundle directory. The build stage writes one bundle per area under
`artifacts/<id>/`; the serve stage reads whichever bundle a request selects.

`build_domain` reads the crop centre from each bundle's `twin_meta.pt`, so once a bundle is
built the serve layer is area-correct from the bundle alone — the registry just maps id -> dir.

Sub-crops (`source_work` set) reuse another area's already-downloaded rasters (dem / worldcover /
jrc / clay) and differ only by the twin crop centre — so they cost **zero** Earth Engine
downloads and let us prove the multi-area path on the committed Patna DEM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .config import CFG


def artifacts_root() -> str:
    """Directory that holds the per-area bundles (defaults to <repo>/artifacts)."""
    env = os.environ.get("VARUNA_ARTIFACTS")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


@dataclass(frozen=True)
class Area:
    id: str
    name: str
    aoi: tuple                          # lon_min, lat_min, lon_max, lat_max (EE download extent)
    center: tuple                       # lat, lon — twin crop centre
    work: Optional[str] = None          # bundle dir; defaults to artifacts/<id>
    source_work: Optional[str] = None   # area id (or path) whose rasters a sub-crop reuses
    note: str = ""

    def work_dir(self) -> str:
        return self.work or os.path.join(artifacts_root(), self.id)

    def source_dir(self) -> Optional[str]:
        """For a sub-crop: the bundle dir holding the shared rasters (else None)."""
        if not self.source_work:
            return None
        other = _REGISTRY.get(self.source_work)
        return other.work_dir() if other else self.source_work


# --- the registry ---------------------------------------------------------------------------
# Patna sub-crops reuse the committed artifacts/patna rasters (no download); Bengaluru needs a
# real Earth Engine build on Colab. Centres are clamped into the DEM by build_domain, so they
# only need to sit inside the AOI.
_AREAS = [
    Area("patna", "Patna (greater)", tuple(CFG.aoi), tuple(CFG.center),
         note="original build; committed bundle"),
    Area("patna_east", "Patna — east", tuple(CFG.aoi), (25.610, 85.235), source_work="patna",
         note="sub-crop of the Patna DEM (no new download)"),
    Area("patna_west", "Patna — west", tuple(CFG.aoi), (25.585, 85.065), source_work="patna",
         note="sub-crop of the Patna DEM (no new download)"),
    Area("bengaluru", "Bengaluru", (77.50, 12.87, 77.78, 13.01), (12.940, 77.640),
         note="urban stormwater flooding; requires an Earth Engine build"),
]
_REGISTRY = {a.id: a for a in _AREAS}


def list_areas() -> list[Area]:
    return list(_AREAS)


def get_area(area_id: str) -> Area:
    a = _REGISTRY.get(area_id)
    if a is None:
        raise KeyError(f"unknown area '{area_id}'; known: {sorted(_REGISTRY)}")
    return a


def area_work(area_id: str) -> str:
    return get_area(area_id).work_dir()


def default_area_id() -> str:
    return _AREAS[0].id


def is_built(area_id: str) -> bool:
    """True if the area's bundle has the required artifacts (serveable)."""
    from .io import REQUIRED_ARTIFACTS
    work = area_work(area_id)
    return all(os.path.exists(os.path.join(work, f)) for f in REQUIRED_ARTIFACTS)
