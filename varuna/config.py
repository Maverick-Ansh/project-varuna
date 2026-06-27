"""Central configuration for FloodTwin Patna.

Single source of truth for AOI, paths, model parameters and alert thresholds.
Resolution order:  config.yaml (if present)  ->  environment variables  ->  built-in defaults.

Pipeline modules must read everything from `CFG`; do not re-hardcode the Patna bbox, the
0.5/1.0 alert thresholds, or `/content/drive` paths anywhere else.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml  # optional; defaults work without it
except Exception:  # pragma: no cover
    yaml = None


def _default_work() -> str:
    """Persist to Drive on Colab when mounted, else a local ./floodtwin folder."""
    drive = "/content/drive/MyDrive"
    if os.path.isdir(drive):
        return f"{drive}/floodtwin"
    return os.environ.get("VARUNA_WORK", os.path.abspath("./floodtwin"))


@dataclass
class Config:
    # --- geography / Earth Engine ---
    project_id: str = "your-cloud-project-id"        # set via VARUNA_PROJECT_ID or config.yaml
    aoi: tuple = (85.02, 25.54, 85.30, 25.68)        # lon_min, lat_min, lon_max, lat_max (greater Patna)
    center: tuple = (25.605, 85.140)                 # lat, lon of the nb05 crop centre
    scale: int = 30                                  # m/pixel for EE downloads
    work: str = field(default_factory=_default_work)

    # --- sink detection (nb01) ---
    min_depth_m: float = 0.15
    min_cells: int = 10
    top_n: int = 25

    # --- recharge suitability weights (nb02): gw_depth, ksat, pervious, availability ---
    rsi_weights: tuple = (0.35, 0.30, 0.20, 0.15)

    # --- alert thresholds (nb04). PLACEHOLDERS until calibrated against nb03 (see README). ---
    amber_ratio: float = 0.5
    red_ratio: float = 1.0
    cell_area_m2: float = 900.0                      # nominal 30 m pixel

    # --- differentiable twin (nb05) ---
    n_grid: int = 128
    dx: float = 60.0
    design_rain_mm: float = 100.0
    budget_m3: float = 150_000.0
    n_sites: int = 8
    site_radius: int = 2
    max_dig_m: float = 3.0
    n_samples: int = 220
    seed: int = 0

    # --- local LLM ---
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_4bit: bool = False                           # True on 16 GB GPUs (T4) sharing with the sim

    def __post_init__(self):
        self.project_id = os.environ.get("VARUNA_PROJECT_ID", self.project_id)
        self.work = os.environ.get("VARUNA_WORK", self.work)
        self.llm_model = os.environ.get("VARUNA_LLM_MODEL", self.llm_model)
        if os.environ.get("VARUNA_LLM_4BIT"):
            self.llm_4bit = os.environ["VARUNA_LLM_4BIT"].lower() in ("1", "true", "yes")
        os.makedirs(self.work, exist_ok=True)

    # convenience -----------------------------------------------------------
    def path(self, name: str) -> str:
        return os.path.join(self.work, name)

    @property
    def region_list(self):
        """AOI as a plain list for ee.Geometry.Rectangle."""
        return list(self.aoi)

    @property
    def aoi_center(self):
        """(lat, lon) centre of the AOI."""
        return ((self.aoi[1] + self.aoi[3]) / 2, (self.aoi[0] + self.aoi[2]) / 2)


def load_config(yaml_path: Optional[str] = None) -> Config:
    data: dict = {}
    candidates = [yaml_path] if yaml_path else []
    candidates += [
        os.environ.get("VARUNA_CONFIG"),
        "config.yaml",
        os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    ]
    if yaml is not None:
        for p in candidates:
            if p and os.path.exists(p):
                with open(p) as f:
                    data = yaml.safe_load(f) or {}
                break
    known = {f.name for f in dataclasses.fields(Config)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items() if k in known}
    return Config(**kw)


CFG = load_config()
