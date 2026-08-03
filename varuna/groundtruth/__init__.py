"""Real observed water depths, and scoring the twin against them.

Every validation Varuna has done so far measured EXTENT: Sentinel-1 says wet or dry, the twin
says wet or dry, and CSI compares the two masks. That is why the honest headline is 0.033 — a
co-location score. It has never been asked the question a resident actually asks, which is
"how deep is the water on my street, and were you right?"

This package answers that. `sources` pulls real point observations of standing water depth;
`score` forces the model with the rain that actually fell and compares its prediction, at that
place, on that day, against what somebody standing in it reported.

Crowdsourced depth is messy and the QC in `sources` is not optional — see the module docstring.
Nothing here trains anything: these observations are held for scoring only, which keeps them a
clean test set and mirrors the existing `learn.news_labels` quarantine.
"""
from __future__ import annotations

from .observations import Observation, ObservationSet  # noqa: F401
