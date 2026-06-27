"""Build the city model (run rarely, GPU for the twin).

Refactor of notebooks 01 (sinks), 02 (recharge), 03 (SAR validation), 05 (differentiable twin).
Each module exposes a `run(...)` entry point that writes the artifact bundle into CFG.work.
"""
from . import sinks, recharge, validate, twin  # noqa: F401
