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
    n_grid: Optional[int] = None        # twin grid size for a fresh build (None -> CFG.n_grid)
    dx: Optional[float] = None          # cell size m for a fresh build (None -> CFG.dx)
    city: Optional[str] = None          # city group id for multi-tile aggregate views

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
    # --- Mumbai (BMC) as four 256^2 @ 60 m tiles (15.36 km squares) tessellating edge-to-edge:
    # row lat edges 18.8963/19.0348/19.1733/19.3118, col lon edges 72.7570/72.9030/73.0490.
    # Exact tessellation (zero overlap) so city-wide sums are correct by construction. Tiles are
    # independent closed-boundary models — flow across tile seams is not simulated; tidal /
    # storm-surge effects are not modeled. AOIs pad the crop window for the EE download.
    Area("mumbai_south", "Mumbai — Island City", (72.745, 18.884, 72.915, 19.047),
         (18.9655, 72.8300), n_grid=256, city="mumbai",
         note="Colaba-Mahim: Hindmata/Parel, Kings Circle, Dadar TT, Byculla, Worli"),
    Area("mumbai_west", "Mumbai — Western suburbs & Kurla", (72.745, 19.023, 72.915, 19.185),
         (19.1040, 72.8300), n_grid=256, city="mumbai",
         note="Milan & Andheri subways, Khar, Sion, BKC, Kurla/Mithi lower reach, Juhu, airport"),
    Area("mumbai_east", "Mumbai — East (Powai-Bhandup)", (72.891, 19.023, 73.061, 19.185),
         (19.1040, 72.9760), n_grid=256, city="mumbai",
         note="Powai/Mithi headwaters, Ghatkopar, Vikhroli, Bhandup, Mulund, Govandi"),
    Area("mumbai_north", "Mumbai — North (Malad-Dahisar)", (72.745, 19.161, 72.915, 19.324),
         (19.2425, 72.8300), n_grid=256, city="mumbai",
         note="Malad subway, Kandivali, Borivali, Dahisar; Poisar & Dahisar rivers"),
]
_REGISTRY = {a.id: a for a in _AREAS}

# City groups for the aggregate dashboard view. Tiles tessellate exactly, so summing per-tile
# volumes/areas never double-counts.
CITIES = {"mumbai": dict(name="Mumbai (BMC)",
                         tiles=["mumbai_south", "mumbai_west", "mumbai_east", "mumbai_north"],
                         note=("Four independent 15.4 km tiles; cross-seam flow and tides are "
                               "not simulated."))}


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
