"""FloodTwin Patna — production package (varuna).

Two layers:
  * varuna.build  — "build the city model" (GPU, run rarely): refactor of notebooks 01/02/03/05.
  * varuna.serve  — "run it daily"        (CPU, headless):    refactor of notebook 04 + emulator.
  * varuna.agent  — local-LLM tool-calling front end ("talk to the flood model").

Import the singleton config from `varuna.config.CFG`; never hardcode paths/thresholds.
"""
from .config import CFG, Config, load_config  # noqa: F401

__version__ = "0.1.0"
