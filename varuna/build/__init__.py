"""Build the city model (run rarely, GPU for the twin).

Refactor of notebooks 01 (sinks), 02 (recharge), 03 (SAR validation), 05 (differentiable twin).
Each module exposes a `run(...)` entry point that writes the artifact bundle into CFG.work.

Submodules load lazily (PEP 562): eagerly importing `twin` here pulled torch into every
`from varuna.build.X import ...`, which would break the serve layer's "light endpoints work
without torch" property now that serve imports the recharge provenance helpers.
"""
from importlib import import_module

_SUBMODULES = ("sinks", "recharge", "validate", "twin", "landcover", "state_screen",
               "areas_build", "baselines", "calibrate", "nightlights")


def __getattr__(name):
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_SUBMODULES))
