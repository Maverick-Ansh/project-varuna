"""Serve the model (run daily, CPU, headless).

Refactor of notebook 04 (weather -> ward alerts) plus emulator what-ifs and the gradient
dig-plan optimiser. These are the functions the LLM agent calls as tools.
"""
from .weather import forecast_rain_mm, aoi_max_rain  # noqa: F401
from .alerts import run_alerts  # noqa: F401
from .emulator import whatif, load_emulator  # noqa: F401
from .optimize import optimize_design  # noqa: F401
