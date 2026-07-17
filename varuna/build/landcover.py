"""Canonical ESA WorldCover class semantics — defined once, imported everywhere.

Three modules used to disagree about herbaceous wetland (class 90): recharge counted it
pervious, while the twin gives it F_TABLE = 0.0 (already saturated, infiltrates nothing) and
waterbalance groups it with water. A saturated wetland is not recharge-suitable; any module
that needs a pervious/impervious/no-recharge split must import these sets, not restate them.
"""
from __future__ import annotations

CLASS_NAMES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare",
               70: "snow_ice", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss_lichen"}

# Ground that can accept infiltration for recharge purposes.
PERVIOUS = frozenset({10, 20, 30, 40, 60})
# Sealed ground.
IMPERVIOUS = frozenset({50})
# Permanent water + wetlands: already saturated — recharge structures make no sense here.
# (Mangrove 95 is coastal/saline; recharging a drinking aquifer through it is not a thing.)
NO_RECHARGE = frozenset({80, 90, 95})
